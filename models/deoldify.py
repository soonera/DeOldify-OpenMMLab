from torch import nn
from mmedit.models.registry import MODELS
from mmedit.models.builder import build_backbone, build_component


@MODELS.register_module()
class DeOldify(nn.Module):
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


