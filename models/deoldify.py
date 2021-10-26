from typing import Tuple, Callable, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.spectral_norm import spectral_norm

# from ..registry import MODELS
from mmedit.models.registry import MODELS

from models.blocks import (init_default, relu, SelfAttention, PixelShuffle_ICNR, SigmoidRange, res_block, icnr, batchnorm_2d,
                           MergeLayer, ifnone)
# from ..builder import build_backbone
from mmedit.models.builder import build_backbone


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


class UnetBlockWide(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
            self,
            up_in_c: int,
            x_in_c: int,
            n_out: int,
            final_div: bool = True,
            blur: bool = False,
            leaky: float = None,
            self_attention: bool = False,
            **kwargs
    ):
        super().__init__()
        up_out = x_out = n_out // 2
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_out, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(
            ni, x_out, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor, s: Tensor) -> Tensor:
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)


@MODELS.register_module()
class DeOldify(nn.Module):
    "Create a U-Net from a given architecture."

    def __init__(
            self,
            encoder,
            n_classes: int,
            blur: bool = False,
            self_attention: bool = False,
            y_range: Optional[Tuple[float, float]] = None,  # SigmoidRange
            last_cross: bool = True,
            bottle: bool = False,
            norm_type: str = "NormSpectral",
            nf_factor: int = 1,
            **kwargs
    ):
        encoder = build_backbone(encoder)

        nf = 512 * nf_factor
        extra_bn = norm_type == "NormSpectral"

        ni = 2048
        kwargs_0 = {}  # 自己加的
        middle_conv = nn.Sequential(
            custom_conv_layer(
                ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs_0
            ),
            custom_conv_layer(
                ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs_0
            ),
        ).eval()

        layers_enc = [encoder]
        layers_mid = [batchnorm_2d(ni), nn.ReLU(), middle_conv]
        layers_dec = []
        layers_post = []

        sfs_idxs = [6, 5, 4, 2]
        x_in_c_list = [1024, 512, 256, 64]
        up_in_c_list = [2048, 512, 512, 512]

        for i in range(len(sfs_idxs)):
            not_final = i != len(sfs_idxs) - 1
            up_in_c = up_in_c_list[i]
            x_in_c = x_in_c_list[i]
            sa = self_attention and (i == len(sfs_idxs) - 3)

            n_out = nf if not_final else nf // 2

            unet_block = UnetBlockWide(
                up_in_c,
                x_in_c,
                n_out,
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                norm_type=norm_type,
                extra_bn=extra_bn,
                **kwargs_0
            ).eval()
            layers_dec.append(unet_block)

        ni = 256
        layers_post.append(PixelShuffle_ICNR(ni, norm_type="NormWeight", **kwargs_0))
        if last_cross:
            layers_post.append(MergeLayer(dense=True))
            ni += n_classes
            layers_post.append(res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs_0))
        layers_post += [
            custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)
        ]
        if y_range is not None:
            layers_post.append(SigmoidRange(*y_range))

        super().__init__()
        self.layers_enc = nn.ModuleList(layers_enc)
        self.layers_mid = nn.ModuleList(layers_mid)
        self.layers_dec = nn.ModuleList(layers_dec)
        self.layers_post = nn.ModuleList(layers_post)

    def forward(self, x):
        res = x

        [x1, x2, x3, x4, res] = self.layers_enc[0](res)

        for layer in self.layers_mid:
            res = layer(res)

        for layer, s in zip(self.layers_dec, [x4, x3, x2, x1]):
            res = layer(res, s)

        for idx, layer in enumerate(self.layers_post):
            if idx == 0:
                res = layer(res)
            elif idx == 1:
                res = torch.cat([res, x], dim=1)
            elif idx > 1:
                res = layer(res)

        return res
