import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import (ASPP, DeepLabHead, DeepLabV3)


def get_deeplab_resnet(num_classes: int, name: str = 'deeplabv3plus', output_stride: int = 8):

    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    elif output_stride == 16:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]
    else:
        raise NotImplementedError()

    backbone = torch.hub.load(
        'facebookresearch/dino:main', 
        'dino_resnet50', 
        replace_stride_with_dilation=replace_stride_with_dilation
    )
    
    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        DeepLab = DeepLabV3Plus
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        DeepLab = DeepLabV3
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLab(backbone, classifier)
    return model


def get_deeplab_vit(num_classes: int, backbone_name: str = 'vits16', name: str = 'deeplabv3plus'):

    # Backbone
    backbone = torch.hub.load('facebookresearch/dino:main', f'dino_{backbone_name}')

    # Classifier
    aspp_dilate = [12, 24, 36]
    inplanes = low_level_planes = backbone.embed_dim
    if name == 'deeplabv3plus':
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        DeepLab = DeepLabV3Plus
    elif name == 'deeplabv3':
        DeepLab = DeepLabV3 
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    # Wrap
    backbone = VisionTransformerWrapper(backbone)
    model = DeepLab(backbone, classifier)
    return model


class VisionTransformerWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # Forward
        output = self.backbone.get_intermediate_layers(x, n=5)
        # Reshaping
        assert (len(output) == 5), f'{output.shape=}'
        H_patch = x.shape[-2] // self.backbone.patch_embed.patch_size
        W_patch = x.shape[-1] // self.backbone.patch_embed.patch_size
        out_ll = output[0][:, 1:, :].transpose(-2, -1).unflatten(-1, (H_patch, W_patch))
        out = output[-1][:, 1:, :].transpose(-2, -1).unflatten(-1, (H_patch, W_patch))
        return {'low_level': out_ll, 'out': out}


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(
            output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
