# modified from deoldify

from models.blocks import conv_layer, Flatten
from torch import nn

from mmcv.runner import load_checkpoint

from mmedit.models.common import generation_init_weights
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


_conv_args = dict(leaky=0.2, norm_type="NormSpectral")


def _conv(ni: int, nf: int, ks: int = 3, stride: int = 1, **kwargs):
    return conv_layer(ni, nf, ks=ks, stride=stride, **_conv_args, **kwargs)


def custom_gan_critic(
    n_channels: int = 3, nf: int = 256, n_blocks: int = 3, p: int = 0.15
):
    "Critic to train a `GAN`."
    layers = [_conv(n_channels, nf, ks=4, stride=2), nn.Dropout2d(p / 2)]
    for i in range(n_blocks):
        layers += [
            _conv(nf, nf, ks=3, stride=1),
            nn.Dropout2d(p),
            _conv(nf, nf * 2, ks=4, stride=2, self_attention=(i == 0)),
        ]
        nf *= 2
    layers += [
        _conv(nf, nf, ks=3, stride=1),
        _conv(nf, 1, ks=4, bias=False, padding=0, use_activ=False),
        Flatten(),
    ]
    return nn.Sequential(*layers)


@COMPONENTS.register_module()
class DeOldifyDiscriminator(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 nf: int = 256,
                 n_blocks: int = 3,
                 p: int = 0.15
                 ):
        super().__init__()

        self.model = custom_gan_critic(n_channels, nf, n_blocks, p)

    def forward(self, x):
        return self.model(x)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')