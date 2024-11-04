import torch
import torch.nn as nn
from torchvision import models


class ExtraLayers(nn.Module):
    def __init__(self):
        super(ExtraLayers, self).__init__()
        self.key_generation = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.key_generation(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

model.classifier = ExtraLayers().to(device)
model.to(device)
