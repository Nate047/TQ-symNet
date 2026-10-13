# Copyright (c) Meta Platforms, Inc. and affiliates.
"""ConvNeXt V2 detector backbone with strict official-checkpoint loading."""

import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.utils import get_root_logger
from ..builder import BACKBONES


class DropPath(nn.Module):
    """Per-sample stochastic depth without requiring timm at runtime."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm(nn.Module):
    """Official ConvNeXt channels-first/channels-last LayerNorm."""

    def __init__(self, normalized_shape, eps=1e-6,
                 data_format='channels_last'):
        super().__init__()
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError('unsupported LayerNorm data format')
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape, )
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Global Response Normalization from the official ConvNeXt V2 model."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv1(self.norm(x))
        x = self.grn(self.act(x))
        x = self.pwconv2(x).permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


@BACKBONES.register_module()
class ConvNeXtV2Official(nn.Module):
    """Four-level ConvNeXt V2 feature backbone for MMDetection 2.x."""

    def __init__(self,
                 in_chans=3,
                 depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768),
                 drop_path_rate=0.0,
                 out_indices=(0, 1, 2, 3),
                 pretrained=None):
        super().__init__()
        self.pretrained = pretrained
        self.out_indices = tuple(out_indices)
        self.dims = tuple(dims)

        self.downsample_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first'))])
        for index in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(
                    dims[index], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(
                    dims[index], dims[index + 1], kernel_size=2, stride=2)))

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        for index in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtV2Block(dims[index], rates[cursor + offset])
                for offset in range(depths[index])
            ]))
            cursor += depths[index]

        for index in self.out_indices:
            self.add_module(
                f'norm{index}',
                LayerNorm(
                    dims[index], eps=1e-6, data_format='channels_first'))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def init_weights(self, pretrained=None):
        checkpoint = pretrained if pretrained is not None else self.pretrained
        if checkpoint is None:
            return
        if not isinstance(checkpoint, str):
            raise TypeError('pretrained must be a checkpoint path or None')
        checkpoint_hasher = hashlib.sha256()
        with open(checkpoint, 'rb') as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024),
                              b''):
                checkpoint_hasher.update(chunk)
        checkpoint_sha256 = checkpoint_hasher.hexdigest()
        state = torch.load(checkpoint, map_location='cpu')
        state = state.get('model', state.get('state_dict', state))
        clean_state = {}
        for raw_key, value in state.items():
            key = raw_key
            for prefix in ('module.', 'backbone.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            # Classification-only layers have no detector counterpart.
            if key.startswith('head.') or key.startswith('norm.'):
                continue
            clean_state[key] = value

        incompatible = self.load_state_dict(clean_state, strict=False)
        allowed_missing = {
            f'norm{index}.{suffix}'
            for index in self.out_indices for suffix in ('weight', 'bias')
        }
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        illegal_missing = missing - allowed_missing
        missing_grn = sorted(key for key in missing if '.grn.' in key)
        if missing_grn or illegal_missing or unexpected:
            raise RuntimeError(
                'ConvNeXt V2 checkpoint is structurally incompatible: '
                f'missing_grn={missing_grn}, '
                f'illegal_missing={sorted(illegal_missing)}, '
                f'unexpected={sorted(unexpected)}')

        checkpoint_grn = {key for key in clean_state if '.grn.' in key}
        model_grn = {key for key in self.state_dict() if '.grn.' in key}
        if checkpoint_grn != model_grn:
            raise RuntimeError(
                'ConvNeXt V2 GRN key coverage mismatch: '
                f'checkpoint={len(checkpoint_grn)}, model={len(model_grn)}')
        get_root_logger().info(
            '[CONVNEXTV2-LOAD] checkpoint=%s sha256=%s loaded=%d grn=%d '
            'new_output_norms=%s', checkpoint, checkpoint_sha256,
            len(clean_state), len(checkpoint_grn), sorted(allowed_missing))

    def forward(self, x):
        outputs = []
        for index in range(4):
            x = self.stages[index](self.downsample_layers[index](x))
            if index in self.out_indices:
                outputs.append(getattr(self, f'norm{index}')(x))
        return tuple(outputs)


@BACKBONES.register_module()
class convnextv2_base_official(ConvNeXtV2Official):

    def __init__(self, **kwargs):
        kwargs.setdefault('depths', (3, 3, 27, 3))
        kwargs.setdefault('dims', (128, 256, 512, 1024))
        super().__init__(**kwargs)


@BACKBONES.register_module()
class convnextv2_large_official(ConvNeXtV2Official):

    def __init__(self, **kwargs):
        kwargs.setdefault('depths', (3, 3, 27, 3))
        kwargs.setdefault('dims', (192, 384, 768, 1536))
        super().__init__(**kwargs)
