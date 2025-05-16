import timm
import torch.nn as nn
import torch.nn.functional as F
import torch

class BirdViTSmallLunitDINO(nn.Module):
    def __init__(self, model_name="hf-hub:1aurent/vit_small_patch8_224.lunit_dino", num_classes=206):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1, num_classes=1)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Resize лог-мел до 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.backbone(x)
        print(x.shape)
        return torch.sigmoid(self.head(x))
