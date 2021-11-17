from torch import nn
from mmcv.runner import load_checkpoint
from mmedit.models.builder import build_component
from mmedit.models.common import (generation_init_weights)

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

@BACKBONES.register_module()
class DeOldifyGenerator(nn.Module):
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
        self.layers_enc = build_component(encoder)
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


    def init_weights(self, pretrained=None, strict=True):
        """Initialize weights for the model.
        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

