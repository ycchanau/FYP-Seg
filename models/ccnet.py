"""Criss-Cross Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nn import CrissCrossAttention
from .segbase import SegBaseModel

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class CCNet(SegBaseModel):
    r"""CCNet

    Parameters
    ----------
    num_classes : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Zilong Huang, et al. "CCNet: Criss-Cross Attention for Semantic Segmentation."
        arXiv preprint arXiv:1811.11721 (2018).
    """

    def __init__(self, num_classes, backbone='resnet50', aux=False, pretrained=True, **kwargs):
        super(CCNet, self).__init__(num_classes, backbone=backbone, pretrained=pretrained, **kwargs)
        self.head = _CCHead(num_classes, **kwargs)
        self.aux=aux
        if aux:
            self.auxlayer = _FCNHead(1024, num_classes, **kwargs)


    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)


        if self.aux:
            outputs = []
            outputs.append(x)
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
            return outputs
        return x


class _CCHead(nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_CCHead, self).__init__()
        self.rcca = _RCCAModule(2048, 512, norm_layer, **kwargs)
        self.out = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        x = self.rcca(x)
        x = self.out(x)
        return x


class _RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1))

    def forward(self, x, recurrence=1):
        out = self.conva(x)
        for i in range(recurrence):
            out = self.cca(out)
        out = self.convb(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)

        return out


