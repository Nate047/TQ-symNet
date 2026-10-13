# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from the official ConvNeXt MMDetection 2.x implementation.
# The upstream implementation is licensed under the MIT License.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from mmdet.utils import get_root_logger
from ..builder import BACKBONES


class ConvNeXtBlock(nn.Module):
    """Official ConvNeXt block using the channels-last MLP implementation."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = ConvNeXtLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim))
                      if layer_scale_init_value > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXtLayerNorm(nn.Module):
    """LayerNorm supporting channels-last and channels-first tensors."""

    def __init__(self, normalized_shape, eps=1e-6,
                 data_format='channels_last'):
        super().__init__()
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError('data_format must be channels_last or channels_first')
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape, )
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


@BACKBONES.register_module()
class ConvNeXtOfficial(nn.Module):
    """ConvNeXt feature backbone adapted from the official repository.

    It returns four feature maps at strides 4/8/16/32, matching the contract
    used by TransNeXt and ``TBFusionCrossScaleNeck`` in this project.
    """

    def __init__(self,
                 in_chans=3,
                 depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768),
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6,
                 out_indices=(0, 1, 2, 3),
                 pretrained=None):
        super().__init__()
        self.pretrained = pretrained
        self.out_indices = tuple(out_indices)
        self.dims = tuple(dims)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            ConvNeXtLayerNorm(dims[0], eps=1e-6,
                              data_format='channels_first')))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                ConvNeXtLayerNorm(dims[i], eps=1e-6,
                                  data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cursor = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dims[i], dp_rates[cursor + j],
                              layer_scale_init_value)
                for j in range(depths[i])
            ]))
            cursor += depths[i]

        norm_layer = partial(ConvNeXtLayerNorm, eps=1e-6,
                             data_format='channels_first')
        for i in self.out_indices:
            self.add_module('norm{}'.format(i), norm_layer(dims[i]))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def init_weights(self, pretrained=None):
        checkpoint = pretrained if pretrained is not None else self.pretrained
        if checkpoint is None:
            return
        if not isinstance(checkpoint, str):
            raise TypeError('pretrained must be a checkpoint path or None')
        state = torch.load(checkpoint, map_location='cpu')
        state = state.get('model', state.get('state_dict', state))
        clean_state = {}
        for key, value in state.items():
            for prefix in ('module.', 'backbone.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            clean_state[key] = value
        incompatible = self.load_state_dict(clean_state, strict=False)
        get_root_logger().info(
            'Loaded ConvNeXt checkpoint %s (missing=%d unexpected=%d)',
            checkpoint, len(incompatible.missing_keys),
            len(incompatible.unexpected_keys))

    def forward(self, x):
        outputs = []
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
            if i in self.out_indices:
                outputs.append(getattr(self, 'norm{}'.format(i))(x))
        return tuple(outputs)


@BACKBONES.register_module()
class convnext_tiny_official(ConvNeXtOfficial):
    def __init__(self, **kwargs):
        kwargs.setdefault('depths', (3, 3, 9, 3))
        kwargs.setdefault('dims', (96, 192, 384, 768))
        super().__init__(**kwargs)


@BACKBONES.register_module()
class convnext_small_official(ConvNeXtOfficial):
    def __init__(self, **kwargs):
        kwargs.setdefault('depths', (3, 3, 27, 3))
        kwargs.setdefault('dims', (96, 192, 384, 768))
        super().__init__(**kwargs)


@BACKBONES.register_module()
class convnext_base_official(ConvNeXtOfficial):
    def __init__(self, **kwargs):
        kwargs.setdefault('depths', (3, 3, 27, 3))
        kwargs.setdefault('dims', (128, 256, 512, 1024))
        super().__init__(**kwargs)
