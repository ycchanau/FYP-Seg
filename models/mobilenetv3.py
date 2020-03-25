from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class _Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(_Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.


class _Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(_Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return self.relu6(x + 3.) / 6.


class _ConvBNHswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNHswish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = _Hswish(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            _Hsigmoid(True)
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out.expand_as(x)


class Identity(nn.Module):
    def __init__(self, in_channels):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, dilation=1, se=True, nl='RE',
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        if nl == 'HS':
            act = _Hswish
        else:
            act = nn.ReLU
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, exp_size, 1, bias=False),
            norm_layer(exp_size),
            act(True),
            # dw
            nn.Conv2d(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation,
                      dilation, groups=exp_size, bias=False),
            norm_layer(exp_size),
            SELayer(exp_size),
            act(True),
            # pw-linear
            nn.Conv2d(exp_size, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, num_classes, mode='large', width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileNetV3, self).__init__()
        if mode == 'large':
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1], ]
            layer2_setting = [
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1], ]
            layer3_setting = [
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 112, True, 'HS', 1], ]
            layer4_setting = [
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1], ]
        elif mode == 'small':
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, True, 'RE', 2], ]
            layer2_setting = [
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1], ]
            layer3_setting = [
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1], ]
            layer4_setting = [
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1], ]
        else:
            raise ValueError('Unknown mode.')

        # building first layer
        self.in_channels = int(16 * width_mult) if width_mult > 1.0 else 16
        self.conv1 = _ConvBNHswish(3, self.in_channels, 3, 2, 1, norm_layer=norm_layer)

        # building bottleneck blocks
        self.layer1 = self._make_layer(Bottleneck, layer1_setting,
                                       width_mult, norm_layer=norm_layer)
        self.layer2 = self._make_layer(Bottleneck, layer2_setting,
                                       width_mult, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Bottleneck, layer3_setting,
                                       width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(Bottleneck, layer4_setting,
                                           width_mult, dilation=2, norm_layer=norm_layer)
        else:
            self.layer4 = self._make_layer(Bottleneck, layer4_setting,
                                           width_mult, norm_layer=norm_layer)

        # building last several layers
        classifier = list()
        if mode == 'large':
            last_bneck_channels = int(960 * width_mult) if width_mult > 1.0 else 960
            self.layer5 = _ConvBNHswish(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(_Hswish(True))
            classifier.append(nn.Conv2d(1280, num_classes, 1))
        elif mode == 'small':
            last_bneck_channels = int(576 * width_mult) if width_mult > 1.0 else 576
            self.layer5 = _ConvBNHswish(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            classifier.append(SEModule(last_bneck_channels))
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(_Hswish(True))
            classifier.append(nn.Conv2d(1280, num_classes, 1))
        else:
            raise ValueError('Unknown mode.')
        self.classifier = nn.Sequential(*classifier)

        self._init_weights()

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for k, exp_size, c, se, nl, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if (dilation == 1) else 1
            exp_channels = int(exp_size * width_mult)
            layers.append(block(self.in_channels, out_channels, exp_channels, k, stride, dilation, se, nl, norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def get_mobilenet_v3(mode, width_mult, **kwargs):
    model = MobileNetV3(mode=mode, width_mult=width_mult, **kwargs)
    return model


def mobilenet_v3_large_1_0(**kwargs):
    return get_mobilenet_v3(mode='large', width_mult=1.0, **kwargs)


def mobilenet_v3_small_1_0(**kwargs):
    return get_mobilenet_v3(mode='small', width_mult=1.0 **kwargs)

class MobileNetV3_Large(BaseModel):
    def __init__(self, num_classes, **kwargs):
        super(MobileNetV3_Large, self).__init__()
        self.backbone = mobilenet_v3_large_1_0(dilated=True, num_classes=num_classes, **kwargs)
        self.head = _Head(num_classes, 960, **kwargs)
        self._init_weights()
        
    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.backbone.conv1(x)
        x = self.backbone.layer1(x)
        c1 = self.backbone.layer2(x)
        c2 = self.backbone.layer3(c1)
        c3 = self.backbone.layer4(c2)
        c4 = self.backbone.layer5(c3)

        return c1, c2, c3, c4

    def forward(self, x):
        size = x.size()[2:]
        _, c2, _, c4 = self.base_forward(x)

        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class MobileNetV3_Small(BaseModel):
    def __init__(self, num_classes, **kwargs):
        super(MobileNetV3_Small, self).__init__()
        self.backbone = mobilenet_v3_small_1_0(dilated=True, num_classes=num_classes, **kwargs)
        self.head = _Head(num_classes, 576, **kwargs)
        self._init_weights()

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.backbone.conv1(x)
        x = self.backbone.layer1(x)
        c1 = self.backbone.layer2(x)
        c2 = self.backbone.layer3(c1)
        c3 = self.backbone.layer4(c2)
        c4 = self.backbone.layer5(c3)

        return c1, c2, c3, c4

    def forward(self, x):
        size = x.size()[2:]
        _, c2, _, c4 = self.base_forward(x)

        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class _Head(nn.Module):
    def __init__(self, num_classes, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()

        self.lr_aspp = _LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)


class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        return x