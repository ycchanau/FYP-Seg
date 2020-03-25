"""Base Model for Semantic Segmentation"""
from base import BaseModel
import torch.nn as nn
from .resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s

__all__ = ['SegBaseModel']


class SegBaseModel(BaseModel):
    r"""Base Model for Semantic Segmentation
    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, num_classes, backbone='resnet50', pretrained=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = True
        self.num_classes = num_classes
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained, dilated=dilated, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained, dilated=dilated, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()
