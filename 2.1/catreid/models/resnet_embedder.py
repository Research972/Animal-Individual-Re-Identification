import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Embedder(nn.Module):
    """
    ResNet50 -> embedding -> L2 normalize
    """
    def __init__(self, emb_dim=256, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [B,2048,1,1]
        self.fc = nn.Linear(2048, emb_dim)

        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)  # [B,2048]
        emb = self.fc(feat)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
