""" Deep Feature Aggregation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .xception import Enc, FCAttention, get_xception_a
from .basic import _ConvBNReLU
from base import BaseModel

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

class DFANet(BaseModel):
    def __init__(self, num_classes, backbone='', pretrained=False, **kwargs):
        super(DFANet, self).__init__()
        self.pretrained = get_xception_a(pretrained, **kwargs)

        self.enc2_2 = Enc(240, 48, 4, **kwargs)
        self.enc3_2 = Enc(144, 96, 6, **kwargs)
        self.enc4_2 = Enc(288, 192, 4, **kwargs)
        self.fca_2 = FCAttention(192, **kwargs)

        self.enc2_3 = Enc(240, 48, 4, **kwargs)
        self.enc3_3 = Enc(144, 96, 6, **kwargs)
        self.enc3_4 = Enc(288, 192, 4, **kwargs)
        self.fca_3 = FCAttention(192, **kwargs)

        self.enc2_1_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.enc2_2_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.enc2_3_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.conv_fusion = _ConvBNReLU(32, 32, 1, **kwargs)

        self.fca_1_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.fca_2_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.fca_3_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.conv_out = nn.Conv2d(32, num_classes, 1)

        initialize_weights(self)

    def forward(self, x):
        # backbone
        stage1_conv1 = self.pretrained.conv1(x)
        stage1_enc2 = self.pretrained.enc2(stage1_conv1)
        stage1_enc3 = self.pretrained.enc3(stage1_enc2)
        stage1_enc4 = self.pretrained.enc4(stage1_enc3)
        stage1_fca = self.pretrained.fca(stage1_enc4)
        stage1_out = F.interpolate(stage1_fca, scale_factor=4, mode='bilinear', align_corners=True)

        # stage2
        stage2_enc2 = self.enc2_2(torch.cat([stage1_enc2, stage1_out], dim=1))
        stage2_enc3 = self.enc3_2(torch.cat([stage1_enc3, stage2_enc2], dim=1))
        stage2_enc4 = self.enc4_2(torch.cat([stage1_enc4, stage2_enc3], dim=1))
        stage2_fca = self.fca_2(stage2_enc4)
        stage2_out = F.interpolate(stage2_fca, scale_factor=4, mode='bilinear', align_corners=True)

        # stage3
        stage3_enc2 = self.enc2_3(torch.cat([stage2_enc2, stage2_out], dim=1))
        stage3_enc3 = self.enc3_3(torch.cat([stage2_enc3, stage3_enc2], dim=1))
        stage3_enc4 = self.enc3_4(torch.cat([stage2_enc4, stage3_enc3], dim=1))
        stage3_fca = self.fca_3(stage3_enc4)

        stage1_enc2_decoder = self.enc2_1_reduce(stage1_enc2)
        stage2_enc2_docoder = F.interpolate(self.enc2_2_reduce(stage2_enc2), scale_factor=2,
                                            mode='bilinear', align_corners=True)
        stage3_enc2_decoder = F.interpolate(self.enc2_3_reduce(stage3_enc2), scale_factor=4,
                                            mode='bilinear', align_corners=True)
        fusion = stage1_enc2_decoder + stage2_enc2_docoder + stage3_enc2_decoder
        fusion = self.conv_fusion(fusion)

        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_fca), scale_factor=4,
                                           mode='bilinear', align_corners=True)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_fca), scale_factor=8,
                                           mode='bilinear', align_corners=True)
        stage3_fca_decoder = F.interpolate(self.fca_3_reduce(stage3_fca), scale_factor=16,
                                           mode='bilinear', align_corners=True)
        fusion = fusion + stage1_fca_decoder + stage2_fca_decoder + stage3_fca_decoder

        out = self.conv_out(fusion)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

