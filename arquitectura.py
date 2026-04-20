import timm
import torch.nn as nn

def crear_modelo(num_classes=4):
    model = timm.create_model('efficientnet_b0', pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
