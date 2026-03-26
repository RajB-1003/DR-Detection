import torch.nn as nn
from torchvision import models

def get_model():
    model = models.resnet50(pretrained=False)
    
    # Replace BOTH fc and last_linear
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.last_linear = model.fc  # 🔥 KEY FIX

    return model