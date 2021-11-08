import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .resnet import get_deeplab_resnet

def get_model(num_classes: int):
    return get_deeplab_resnet(num_classes=(num_classes + 1))  # add 1 for bg
