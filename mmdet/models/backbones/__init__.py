# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet_stn import ResNet_STN, ResNetV1d_STN
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .p2t import p2t_tiny, p2t_small, p2t_base, p2t_large

from .transnext_cuda import transnext_tiny, transnext_small, transnext_base
from .convnext_official import (ConvNeXtOfficial, convnext_base_official,
                                convnext_small_official,
                                convnext_tiny_official)
from .convnextv2_official import (ConvNeXtV2Official,
                                  convnextv2_base_official,
                                  convnextv2_large_official)
from .mambavision_official import (MambaVisionOfficial,
                                   mamba_vision_b_official,
                                   mamba_vision_l2_official,
                                   mamba_vision_s_official,
                                   mamba_vision_t_official)

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNet_STN', 'ResNetV1d_STN', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'p2t_tiny', 'p2t_small', 'p2t_base', 'p2t_large',
    'transnext_tiny', 'transnext_small', 'transnext_base',
    'ConvNeXtOfficial', 'convnext_tiny_official',
    'convnext_small_official', 'convnext_base_official',
    'ConvNeXtV2Official', 'convnextv2_base_official',
    'convnextv2_large_official',
    'MambaVisionOfficial', 'mamba_vision_t_official',
    'mamba_vision_s_official', 'mamba_vision_b_official',
    'mamba_vision_l2_official'
]
