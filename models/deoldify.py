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

    def forward_test(self, x):
        res = x

        res, short_cut_out = self.layers_enc(res)

        res = self.layers_mid(res)

        short_cut_out.reverse()
        res = self.layers_dec(res, short_cut_out)

        res = self.layers_post(res, x)

        return res

    def forward(self,
                merged,
                trimap,
                meta,
                alpha=None,
                test_mode=False,
                **kwargs):
        """Defines the computation performed at every call.
        Args:
            merged (Tensor): Image to predict alpha matte.
            trimap (Tensor): Trimap of the input image.
            meta (list[dict]): Meta data about the current data batch.
                Defaults to None.
            alpha (Tensor, optional): Ground-truth alpha matte.
                Defaults to None.
            test_mode (bool, optional): Whether in test mode. If ``True``, it
                will call ``forward_test`` of the model. Otherwise, it will
                call ``forward_train`` of the model. Defaults to False.
        Returns:
            dict: Return the output of ``self.forward_test`` if ``test_mode`` \
                are set to ``True``. Otherwise return the output of \
                ``self.forward_train``.
        """
        if test_mode:
            return self.forward_test(merged, trimap, meta, **kwargs)

        return self.forward_train(merged, trimap, meta, alpha, **kwargs)
    
    def forward_train(self):
        raise NotImplementedError
