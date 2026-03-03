import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

num_classes = 2 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = deeplabv3_resnet50(pretrained=True)

old_conv = model.backbone.conv1

new_conv = nn.Conv2d(
    6,
    old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False
)

new_conv.weight.data[:, :3] = old_conv.weight.data

# Initialize remaining 3 channels with mean of pretrained weights
new_conv.weight.data[:, 3:] = old_conv.weight.data.mean(dim=1, keepdim=True)

model.backbone.conv1 = new_conv

model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

model = model.to(device)