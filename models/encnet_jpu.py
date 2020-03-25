import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss

from torch.autograd import Variable
from base import BaseModel
from .resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, **kwargs):
        super(JPU, self).__init__()


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode='bilinear', align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat

class BaseNet(BaseModel):
    def __init__(self, num_classes, backbone, pretrained=True, aux=True, jpu=True, dilated=False, norm_layer=nn.BatchNorm2d,
                 **kwargs):
        super(BaseNet, self).__init__()
        self.num_classes = num_classes
        self.aux = aux

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

        self.backbone = backbone
        self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer) if jpu else None

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

class Encoding(nn.Module):
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert (X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN -> BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW -> Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(self.scale_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = self.aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x' + str(self.D) + '=>' + str(self.K) + 'x' \
               + str(self.D) + ')'

    @staticmethod
    def scale_l2(X, C, S):
        S = S.view(1, 1, C.size(0), 1)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        SL = S * (X - C)
        SL = SL.pow(2).sum(3)
        return SL

    @staticmethod
    def aggregate(A, X, C):
        A = A.unsqueeze(3)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        E = A * (X - C)
        E = E.sum(1)
        return E

class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

class EncNet_JPU(BaseNet):
    def __init__(self, num_classes, backbone='resnet50', aux=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(EncNet_JPU, self).__init__(num_classes, backbone, aux,
                                     norm_layer=norm_layer, **kwargs)
        self.head = EncHead([512, 1024, 2048], self.num_classes, norm_layer=norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, num_classes, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = F.interpolate(x[0], imsize, mode='bilinear', align_corners=True)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.interpolate(auxout, imsize, mode='bilinear', align_corners=True)
            x.append(auxout)
        return tuple(x)


class EncModule(nn.Module):
    def __init__(self, in_channels, num_classes, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes),
            nn.BatchNorm1d(ncodes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, jpu=True, lateral=False,
                 norm_layer=None, **kwargs):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels[-1], 512, 1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True)) if jpu else \
                     nn.Sequential(nn.Conv2d(in_channels[-1], 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[0], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(in_channels[1], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)