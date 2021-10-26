from typing import Tuple, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmedit.models.registry import MODELS

from models.blocks import (relu, PixelShuffle_ICNR, SigmoidRange, res_block, batchnorm_2d,
                           MergeLayer, custom_conv_layer, CustomPixelShuffle_ICNR)

from mmedit.models.builder import build_backbone, build_component


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


# artistic
class UnetBlockDeep(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        final_div: bool = True,
        blur: bool = False,
        leaky: float = None,
        self_attention: bool = False,
        nf_factor: float = 1.5,
        **kwargs
    ):
        super().__init__()

        up_out = up_in_c // 2
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_out, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        nf = int((ni if final_div else ni // 2) * nf_factor)
        self.conv1 = custom_conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = custom_conv_layer(
            nf, nf, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor, s: Tensor) -> Tensor:
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


@MODELS.register_module()
class DynamicUnetWide(nn.Module):
    "Create a U-Net from a given architecture."

    def __init__(
            self,
            encoder,
            mid_layers,
            decoder,
            post_layers,
            **kwargs
    ):
        super().__init__()
        self.layers_enc = build_backbone(encoder)
        self.layers_mid = build_component(mid_layers)
        self.layers_dec = build_component(decoder)
        self.layers_post = build_component(post_layers)

    def forward(self, x):
        res = x

        res, short_cut_out = self.layers_enc(res)

        res = self.layers_mid(res)

        short_cut_out.reverse()
        res = self.layers_dec(res, short_cut_out)

        res = self.layers_post(res, x)

        return res


# artistic
@MODELS.register_module()
class DynamicUnetDeep(nn.Module):
    "Create a U-Net from a given architecture."

    def __init__(
            self,
            encoder,
            mid_layers,
            n_classes: int = 3,
            blur: bool = True,
            self_attention: bool = True,
            y_range: Optional[Tuple[float, float]] = (-3.0, 3.0),  # SigmoidRange
            last_cross: bool = True,
            bottle: bool = False,
            norm_type: str = "NormSpectral",
            nf_factor: int = 1.5,
            **kwargs
    ):
        super().__init__()
        self.layers_enc = build_backbone(encoder)
        self.layers_mid = build_component(mid_layers)

        # encoder = build_backbone(encoder)
        # mid_layers = build_component(mid_layers)

        extra_bn = norm_type == "NormSpectral"

        # ni = 512
        kwargs_0 = {}  # 自己加的
        # middle_conv = nn.Sequential(
        #     custom_conv_layer(
        #         ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs_0
        #     ),
        #     custom_conv_layer(
        #         ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs_0
        #     ),
        # ).eval()

        # layers_enc = [encoder]

        # layers_mid = [batchnorm_2d(ni), nn.ReLU(), middle_conv]
        # layers_mid = [mid_layers]
        layers_dec = []
        layers_post = []

        sfs_idxs = [6, 5, 4, 2]
        x_in_c_list = [256, 128, 64, 64]
        up_in_c_list = [512, 768, 768, 672]

        for i in range(len(sfs_idxs)):
            not_final = i != len(sfs_idxs) - 1
            up_in_c = up_in_c_list[i]
            x_in_c = x_in_c_list[i]
            sa = self_attention and (i == len(sfs_idxs) - 3)

            # n_out = nf if not_final else nf // 2

            unet_block = UnetBlockDeep(
                up_in_c,
                x_in_c,
                # n_out,
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                norm_type=norm_type,
                extra_bn=extra_bn,
                nf_factor=nf_factor,
                **kwargs_0
            ).eval()
            layers_dec.append(unet_block)

        # ni = 256
        ni = 300
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

        # self.layers_enc = nn.ModuleList(layers_enc)
        # self.layers_mid = nn.ModuleList(layers_mid)
        self.layers_dec = nn.ModuleList(layers_dec)
        self.layers_post = nn.ModuleList(layers_post)

    def forward(self, x):
        res = x

        # [x1, x2, x3, x4, res] = self.layers_enc[0](res)
        res, short_cut_out = self.layers_enc(res)

        # for layer in self.layers_mid:
        #     res = layer(res)
        res = self.layers_mid(res)

        short_cut_out.reverse()
        for layer, s in zip(self.layers_dec, short_cut_out):
            res = layer(res, s)

        res = self.layers_post[0](res)
        res = torch.cat([res, x], dim=1)
        for idx, layer in enumerate(self.layers_post[2:]):
            res = layer(res)

        return res
