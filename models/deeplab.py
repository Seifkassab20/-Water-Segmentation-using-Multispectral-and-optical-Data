import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLab(nn.Module):
    def __init__(self, n_channels=6, n_classes=2, pretrained=True):
        super().__init__()

        self.model = deeplabv3_resnet50(pretrained=pretrained)
        old_conv = self.model.backbone.conv1

        new_conv = nn.Conv2d(
            n_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        if pretrained:
            new_conv.weight.data[:, :3] = old_conv.weight.data
            new_conv.weight.data[:, 3:] = old_conv.weight.data.mean(
                dim=1, keepdim=True
            )
        else:
            nn.init.kaiming_normal_(new_conv.weight)

        self.model.backbone.conv1 = new_conv
        self.model.classifier[4] = nn.Conv2d(
            256,
            n_classes,
            kernel_size=1
        )

    def forward(self, x):
        return self.model(x)["out"]