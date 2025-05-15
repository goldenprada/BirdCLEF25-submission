import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class BirdResNet(nn.Module):
    def __init__(self, num_classes=207):
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))