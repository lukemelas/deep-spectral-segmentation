import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, input, target=None):
        output = self.model(input)
        if target is not None: 
            loss = F.cross_entropy(output, target)
            return loss
        else:
            return output


if __name__ == "__main__":
    model = Model()
    o = torch.optim.Adam(model.parameters())
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    y.sum().backward()
    o.step()
    o.zero_grad()
    print('single iteration complete')