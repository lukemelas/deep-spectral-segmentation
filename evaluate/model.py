import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class SimpleModel(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained, num_classes=num_classes)

    def forward(self, input, target=None):
        output = self.model(input)
        if target is None:
            return output
        else:
            return self.loss(output, target)

    @classmethod
    def loss(cls, output, target):
        return F.cross_entropy(output, target)


if __name__ == "__main__":
    model = SimpleModel()
    o = torch.optim.Adam(model.parameters())
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    y.sum().backward()
    o.step()
    o.zero_grad()
    print('single iteration complete')
