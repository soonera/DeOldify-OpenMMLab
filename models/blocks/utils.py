# modified from fastai

from typing import Callable
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.spectral_norm import spectral_norm

from models.blocks import (init_default, relu, SelfAttention, icnr, ifnone)


def custom_conv_layer(
        ni: int,
        nf: int,
        ks: int = 3,
        stride: int = 1,
        padding: int = None,
        bias: bool = None,
        is_1d: bool = False,
        norm_type: str = "NormBatch",
        use_activ: bool = True,
        leaky: float = None,
        transpose: bool = False,
        init: Callable = nn.init.kaiming_normal_,
        self_attention: bool = False,
        extra_bn: bool = False,
):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in ("NormBatch", "NormBatchZero") or extra_bn == True

    if bias is None:
        bias = not bn

    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d

    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )

    if norm_type == "NormWeight":
        conv = weight_norm(conv)
    elif norm_type == "NormSpectral":
        conv = spectral_norm(conv)

    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))

    return nn.Sequential(*layers)


class CustomPixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(
            self,
            ni: int,
            nf: int = None,
            scale: int = 2,
            blur: bool = False,
            leaky: float = None,
            **kwargs
    ):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(
            ni, nf * (scale ** 2), ks=1, use_activ=False, **kwargs
        )
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