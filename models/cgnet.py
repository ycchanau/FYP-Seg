"""Context Guided Network for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import _ConvBNPReLU, _BNPReLU
from base import BaseModel


class CGNet(BaseModel):
    r"""CGNet

    Parameters
    ----------
    num_classes : int
        Number of categories for the training dataset.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Tianyi Wu, et al. "CGNet: A Light-weight Context Guided Network for Semantic Segmentation."
        arXiv preprint arXiv:1811.08201 (2018).
    """

    def __init__(self, num_classes, backbone='', pretrained=True, M=3, N=21, **kwargs):
        super(CGNet, self).__init__()
        # stage 1
        self.stage1_0 = _ConvBNPReLU(3, 32, 3, 2, 1, **kwargs)
        self.stage1_1 = _ConvBNPReLU(32, 32, 3, 1, 1, **kwargs)
        self.stage1_2 = _ConvBNPReLU(32, 32, 3, 1, 1, **kwargs)

        self.sample1 = _InputInjection(1)
        self.sample2 = _InputInjection(2)
        self.bn_prelu1 = _BNPReLU(32 + 3, **kwargs)

        # stage 2
        self.stage2_0 = ContextGuidedBlock(32 + 3, 64, dilation=2, reduction=8, down=True, residual=False, **kwargs)
        self.stage2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.stage2.append(ContextGuidedBlock(64, 64, dilation=2, reduction=8, **kwargs))
        self.bn_prelu2 = _BNPReLU(128 + 3, **kwargs)

        # stage 3
        self.stage3_0 = ContextGuidedBlock(128 + 3, 128, dilation=4, reduction=16, down=True, residual=False, **kwargs)
        self.stage3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.stage3.append(ContextGuidedBlock(128, 128, dilation=4, reduction=16, **kwargs))
        self.bn_prelu3 = _BNPReLU(256, **kwargs)

        self.head = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(256, num_classes, 1))


    def forward(self, x):
        size = x.size()[2:]
        # stage1
        out0 = self.stage1_0(x)
        out0 = self.stage1_1(out0)
        out0 = self.stage1_2(out0)

        inp1 = self.sample1(x)
        inp2 = self.sample2(x)

        # stage 2
        out0_cat = self.bn_prelu1(torch.cat([out0, inp1], dim=1))
        out1_0 = self.stage2_0(out0_cat)
        for i, layer in enumerate(self.stage2):
            if i == 0:
                out1 = layer(out1_0)
            else:
                out1 = layer(out1)
        out1_cat = self.bn_prelu2(torch.cat([out1, out1_0, inp2], dim=1))

        # stage 3
        out2_0 = self.stage3_0(out1_cat)
        for i, layer in enumerate(self.stage3):
            if i == 0:
                out2 = layer(out2_0)
            else:
                out2 = layer(out2)
        out2_cat = self.bn_prelu3(torch.cat([out2_0, out2], dim=1))


        out = self.head(out2_cat)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)

        return out

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

class _ChannelWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, **kwargs):
        super(_ChannelWiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=in_channels, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class _FGlo(nn.Module):
    def __init__(self, in_channels, reduction=16, **kwargs):
        super(_FGlo, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid())

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.gap(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out


class _InputInjection(nn.Module):
    def __init__(self, ratio):
        super(_InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, 2, 1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


class _ConcatInjection(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConcatInjection, self).__init__()
        self.bn = norm_layer(in_channels)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.bn(out)
        out = self.prelu(out)
        return out


class ContextGuidedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2, reduction=16, down=False,
                 residual=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ContextGuidedBlock, self).__init__()
        inter_channels = out_channels // 2 if not down else out_channels
        if down:
            self.conv = _ConvBNPReLU(in_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer, **kwargs)
            self.reduce = nn.Conv2d(inter_channels * 2, out_channels, 1, bias=False)
        else:
            self.conv = _ConvBNPReLU(in_channels, inter_channels, 1, 1, 0, norm_layer=norm_layer, **kwargs)
        self.f_loc = _ChannelWiseConv(inter_channels, inter_channels, **kwargs)
        self.f_sur = _ChannelWiseConv(inter_channels, inter_channels, dilation, **kwargs)
        self.bn = norm_layer(inter_channels * 2)
        self.prelu = nn.PReLU(inter_channels * 2)
        self.f_glo = _FGlo(out_channels, reduction, **kwargs)
        self.down = down
        self.residual = residual

    def forward(self, x):
        out = self.conv(x)
        loc = self.f_loc(out)
        sur = self.f_sur(out)

        joi_feat = torch.cat([loc, sur], dim=1)
        joi_feat = self.prelu(self.bn(joi_feat))
        if self.down:
            joi_feat = self.reduce(joi_feat)

        out = self.f_glo(joi_feat)
        if self.residual:
            out = out + x

        return out



