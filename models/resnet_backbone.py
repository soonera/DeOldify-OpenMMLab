import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from mmedit.models.registry import BACKBONES


@BACKBONES.register_module()
class ColorizationResNet(ResNet):
    def __init__(self, num_layers, pretrained=None):

        if num_layers == 101:
            super().__init__(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=1, zero_init_residual=True)
        elif num_layers == 34:
            super().__init__(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=1, zero_init_residual=True)
        else:
            raise NotImplementedError

        del self.avgpool
        del self.fc

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        outs.append(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outs.append(x)

        x = self.layer2(x)
        outs.append(x)

        x = self.layer3(x)
        outs.append(x)

        x = self.layer4(x)
        outs.append(x)

        return outs

    def get_channels(self):
        x = torch.rand([1, 3, 64, 64])
        x = torch.Tensor(x)
        model_channels = []

        x = self.conv1(x)
        model_channels.append(x.shape[1])
        x = self.bn1(x)
        model_channels.append(x.shape[1])

        x = self.relu(x)
        model_channels.append(x.shape[1])
        x = self.maxpool(x)
        model_channels.append(x.shape[1])
        x = self.layer1(x)
        model_channels.append(x.shape[1])

        x = self.layer2(x)
        model_channels.append(x.shape[1])

        x = self.layer3(x)
        model_channels.append(x.shape[1])

        x = self.layer4(x)
        model_channels.append(x.shape[1])

        return model_channels
