import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .model import get_deeplab_resnet, get_deeplab_vit


def get_model(name: str, num_classes: int):
    if 'resnet' in name:
        model = get_deeplab_resnet(num_classes=(num_classes + 1))  # add 1 for bg
    elif 'vit' in name:
        model = get_deeplab_vit(backbone_name=name, num_classes=(num_classes + 1))  # add 1 for bg
    else:
        raise NotImplementedError()
    return model

