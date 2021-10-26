from mmedit.models.registry import COMPONENTS
from torch import nn
from models.blocks import custom_conv_layer, batchnorm_2d


@COMPONENTS.register_module()
class MidConvLayer(nn.Module):
    def __init__(self, norm_type: str = "NormSpectral", ni: int = 2048, **kwargs):

        super().__init__()
        extra_bn = norm_type == "NormSpectral"

        kwargs_0 = {}  # 自己加的
        middle_conv = nn.Sequential(
            custom_conv_layer(
                ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs_0
            ),
            custom_conv_layer(
                ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs_0
            ),
        ).eval()

        self.bn = batchnorm_2d(ni)
        self.relu = nn.ReLU()
        self.mid_cov = middle_conv

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.mid_cov(x)

        return x
