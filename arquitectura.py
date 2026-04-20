import torch
import torch.nn as nn
from torchvision import models

class ModeloArtistas(nn.Module):
    def __init__(self, num_classes=4):
        super(ModeloArtistas, self).__init__()

        self.model = models.efficientnet_b0(pretrained=False)

        # Cambiar la capa final
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)