import torch
import torch.nn as nn
import timm

class ModeloArtistas(nn.Module):
    def __init__(self, num_classes=4):
        super(ModeloArtistas, self).__init__()

        self.model = timm.create_model('efficientnet_b0', pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
