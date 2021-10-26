import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.spectral_norm import spectral_norm

from typing import Callable, Any


LayerFunc = Callable[[nn.Module], None]


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def init_default(m: nn.Module, func: LayerFunc = nn.init.kaiming_normal_) -> nn.Module:
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
    return m


def relu(inplace: bool = False, leaky: float = None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)


class SelfAttention(nn.Module):
    "Self attention layer for nd."

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(Tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class SequentialEx(nn.Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i): return self.layers[i]

    def append(self, l): return self.layers.append(l)

    def extend(self, l): return self.layers.extend(l)

    def insert(self, i, l): return self.layers.insert(i, l)


class PixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(self, ni: int, nf: int = None, scale: int = 2, blur: bool = False, norm_type="Norm.Weight",
                 leaky: float = None):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_layer(ni, nf * (scale ** 2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


class SigmoidRange(nn.Module):
    "Sigmoid module with range `(low,x_max)`"

    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x): return sigmoid_range(x, self.low, self.high)


def res_block(nf, dense: bool = False, norm_type: str = "NormBatch", bottle: bool = False,
                            **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type == "NormBatch"): norm2 = "NormBatchZero"
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
                        conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
                        MergeLayer(dense))


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


def batchnorm_2d(nf: int, norm_type: str = "NormBatch"):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type == "NormBatchZero" else 1.)
    return bn


class MergeLayer(nn.Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."

    def __init__(self, dense: bool = False):
        super().__init__()
        self.dense = dense

    def forward(self, x): return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


def conv_layer(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None, bias: bool = None,
               is_1d: bool = False,
               norm_type: str = "NormBatch",
               use_activ: bool = True, leaky: float = None,
               transpose: bool = False, init: Callable = nn.init.kaiming_normal_, self_attention: bool = False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in ("NormBatch", "NormBatchZero")
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if norm_type == "NormWeight":
        conv = weight_norm(conv)
    elif norm_type == "NormSpectral":
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)
