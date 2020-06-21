import torch.nn as nn
from torchvision import models


def load_pretrained_model():
    model_tl = models.resnet18(pretrained=True)
    num_ftrs = model_tl.fc.in_features

    model_tl.fc = nn.Linear(num_ftrs, 2)

    return model_tl
