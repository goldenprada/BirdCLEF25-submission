import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BirdEfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet_b0", num_classes=207):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(self.head(x))