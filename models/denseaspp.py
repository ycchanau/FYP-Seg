import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict
from base import BaseModel

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


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


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation=1, norm_layer=nn.BatchNorm2d):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, 1, 1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, dilation, dilation, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size,
                 growth_rate, drop_rate, dilation=1, norm_layer=nn.BatchNorm2d):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, dilation, norm_layer)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d):
        super(_Transition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, 1, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, 2))


# Net
class DenseNet(nn.Module):

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=1000, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, 7, 2, 3, bias=False)),
            ('norm0', norm_layer(num_init_features)),
            ('relu0', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(3, 2, 1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, norm_layer=norm_layer)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2, norm_layer=norm_layer)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.num_features = num_features

        # Final batch norm
        self.features.add_module('norm5', norm_layer(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class DilatedDenseNet(DenseNet):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=1000, dilate_scale=8, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DilatedDenseNet, self).__init__(growth_rate, block_config, num_init_features,
                                              bn_size, drop_rate, num_classes, norm_layer)
        assert (dilate_scale == 8 or dilate_scale == 16), "dilate_scale can only set as 8 or 16"
        from functools import partial
        if dilate_scale == 8:
            self.features.denseblock3.apply(partial(self._conv_dilate, dilate=2))
            self.features.denseblock4.apply(partial(self._conv_dilate, dilate=4))
            del self.features.transition2.pool
            del self.features.transition3.pool
        elif dilate_scale == 16:
            self.features.denseblock4.apply(partial(self._conv_dilate, dilate=2))
            del self.features.transition3.pool

    def _conv_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.padding = (dilate, dilate)
                m.dilation = (dilate, dilate)


# Specification
densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                 161: (96, 48, [6, 12, 36, 24]),
                 169: (64, 32, [6, 12, 32, 32]),
                 201: (64, 32, [6, 12, 48, 32])}


# Constructor
def get_densenet(num_layers, pretrained=False, **kwargs):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    """
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet%d' % num_layers])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def get_dilated_densenet(num_layers, dilate_scale, pretrained=False, **kwargs):
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    model = DilatedDenseNet(growth_rate, block_config, num_init_features, dilate_scale=dilate_scale)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet%d' % num_layers])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet121(**kwargs):
    return get_densenet(121, **kwargs)


def densenet161(**kwargs):
    return get_densenet(161, **kwargs)


def densenet169(**kwargs):
    return get_densenet(169, **kwargs)


def densenet201(**kwargs):
    return get_densenet(201, **kwargs)


def dilated_densenet121(dilate_scale, **kwargs):
    return get_dilated_densenet(121, dilate_scale, **kwargs)


def dilated_densenet161(dilate_scale, **kwargs):
    return get_dilated_densenet(161, dilate_scale, **kwargs)


def dilated_densenet169(dilate_scale, **kwargs):
    return get_dilated_densenet(169, dilate_scale, **kwargs)


def dilated_densenet201(dilate_scale, **kwargs):
    return get_dilated_densenet(201, dilate_scale, **kwargs)

class DenseASPP(BaseModel):
    def __init__(self, num_classes, backbone='densenet121', aux=False,
                 pretrained=True, dilate_scale=8, **kwargs):
        super(DenseASPP, self).__init__()
        self.num_classes = num_classes
        self.aux = aux
        self.dilate_scale = dilate_scale
        if backbone == 'densenet121':
            self.pretrained = dilated_densenet121(dilate_scale, pretrained=pretrained, **kwargs)
        elif backbone == 'densenet161':
            self.pretrained = dilated_densenet161(dilate_scale, pretrained=pretrained, **kwargs)
        elif backbone == 'densenet169':
            self.pretrained = dilated_densenet169(dilate_scale, pretrained=pretrained, **kwargs)
        elif backbone == 'densenet201':
            self.pretrained = dilated_densenet201(dilate_scale, pretrained=pretrained, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        in_channels = self.pretrained.num_features

        self.head = _DenseASPPHead(in_channels, num_classes)

        if aux:
            self.auxlayer = _FCNHead(in_channels, num_classes, **kwargs)


    def forward(self, x):
        size = x.size()[2:]
        features = self.pretrained.features(x)
        if self.dilate_scale > 8:
            features = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.head(features)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)


        if self.aux:
            outputs = []
            outputs.append(x)
            auxout = self.auxlayer(features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
            return outputs
        else:
            return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()


class _DenseASPPHead(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(in_channels, 256, 64, norm_layer, norm_kwargs)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels + 5 * 64, num_classes, 1)
        )

    def forward(self, x):
        x = self.dense_aspp_block(x)
        return self.block(x)


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        return x