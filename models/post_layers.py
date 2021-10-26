from mmedit.models.registry import COMPONENTS
from typing import Tuple, Optional

import torch
from torch import nn

from models.blocks import (PixelShuffle_ICNR, SigmoidRange, res_block,
                           MergeLayer, custom_conv_layer)


@COMPONENTS.register_module()
class PostLayer(nn.Module):
    def __init__(
            self,
            ni: int = 256,
            last_cross: bool = True,
            n_classes: int = 3,
            bottle: bool = False,
            norm_type: str = "NormSpectral",
            y_range: Optional[Tuple[float, float]] = (-3.0, 3.0),  # SigmoidRange
    ):
        super().__init__()
        kwargs_0 = {}
        layers_post = []
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
        self.layers_post = nn.ModuleList(layers_post)

    def forward(self, x, x_short):
        res = x
        res = self.layers_post[0](res)
        res = torch.cat([res, x_short], dim=1)
        for idx, layer in enumerate(self.layers_post[2:]):
            res = layer(res)

        return res

