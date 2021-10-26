from mmedit.models.registry import COMPONENTS

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.blocks import (relu, PixelShuffle_ICNR, SigmoidRange, res_block, batchnorm_2d,
                           MergeLayer, custom_conv_layer, CustomPixelShuffle_ICNR)



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


@COMPONENTS.register_module()
class UnetWideDecoder(nn.Module):
    def __init__(
            self,
            self_attention: bool = True,
            x_in_c_list: list = [],
            ni: int = 2048,
            nf_factor: int = 2,
            blur: bool = True,
            norm_type: str = "NormSpectral",
    ):
        super().__init__()
        kwargs_0 = {}  # 自己加的
        extra_bn = norm_type == "NormSpectral"

        layers_dec = []
        x_in_c_list.reverse()
        up_in_c = ni
        for i, x_in_c in enumerate(x_in_c_list):
            not_final = i != len(x_in_c_list) - 1
            sa = self_attention and (i == len(x_in_c_list) - 3)

            # 这个512是原代码自带的
            nf = 512 * nf_factor
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
            up_in_c = n_out // 2
            layers_dec.append(unet_block)
        self.layers_dec = nn.ModuleList(layers_dec)

    def forward(self, x, short_cut_list):
        res = x
        for layer, s in zip(self.layers_dec, short_cut_list):
            res = layer(res, s)
        return res

