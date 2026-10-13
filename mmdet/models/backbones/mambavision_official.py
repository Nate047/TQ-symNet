"""MMDetection 2.x adapter for NVIDIA's official MambaVision source.

The unmodified upstream implementation and its NVIDIA Source Code License-NC
are vendored under ``third_party/mambavision_official``.  Imports are delayed
until construction so installations without selective-scan can still train
all other registered backbones.
"""

import torch
import torch.nn as nn

from mmdet.utils import get_root_logger
from ..builder import BACKBONES


class ChannelFirstLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


@BACKBONES.register_module()
class MambaVisionOfficial(nn.Module):
    """Four-level detector backbone built from the official MambaVision code.

    This adapter supports the project's MMDetection 2.16 registry and returns
    feature maps at strides 4/8/16/32.  MambaVision's CUDA selective scan is
    not silently replaced by an approximation: the official ``mamba_ssm`` op
    must be installed for scientifically valid comparisons.
    """

    def __init__(self,
                 dim=80,
                 in_dim=32,
                 depths=(1, 3, 8, 4),
                 window_size=(8, 8, 14, 7),
                 mlp_ratio=4.0,
                 num_heads=(2, 4, 8, 16),
                 drop_path_rate=0.2,
                 in_chans=3,
                 out_indices=(0, 1, 2, 3),
                 layer_scale=None,
                 layer_scale_conv=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        try:
            from third_party.mambavision_official.models.mamba_vision import (
                MambaVision, window_partition, window_reverse)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                'MambaVisionOfficial requires the official mamba_ssm CUDA '
                'selective-scan package and einops. The current project stack '
                '(PyTorch 1.10/CUDA 11.3) may require a separate compatible '
                'environment; do not install a mismatched wheel. Original '
                'error: {}'.format(exc)) from exc

        self._window_partition = window_partition
        self._window_reverse = window_reverse
        self.pretrained = pretrained
        self.out_indices = tuple(out_indices)
        self.dims = tuple(int(dim * 2**i) for i in range(4))

        core = MambaVision(
            dim=dim,
            in_dim=in_dim,
            depths=tuple(depths),
            window_size=tuple(window_size),
            mlp_ratio=mlp_ratio,
            num_heads=tuple(num_heads),
            drop_path_rate=drop_path_rate,
            in_chans=in_chans,
            num_classes=0,
            layer_scale=layer_scale,
            layer_scale_conv=layer_scale_conv,
            **kwargs)
        self.patch_embed = core.patch_embed
        self.levels = core.levels
        for i in self.out_indices:
            self.add_module('outnorm{}'.format(i),
                            ChannelFirstLayerNorm(self.dims[i]))

    def init_weights(self, pretrained=None):
        checkpoint_path = pretrained if pretrained is not None else self.pretrained
        if checkpoint_path is None:
            return
        if not isinstance(checkpoint_path, str):
            raise TypeError('pretrained must be a checkpoint path or None')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get(
            'model', checkpoint.get('state_dict', checkpoint))
        clean_state = {}
        for key, value in state_dict.items():
            for prefix in ('module.', 'backbone.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            clean_state[key] = value
        incompatible = self.load_state_dict(clean_state, strict=False)
        get_root_logger().info(
            'Loaded MambaVision checkpoint %s (missing=%d unexpected=%d)',
            checkpoint_path, len(incompatible.missing_keys),
            len(incompatible.unexpected_keys))

    def _forward_level(self, level, x):
        """Run one official stage while retaining its pre-downsample output."""
        _, _, height, width = x.shape
        if level.transformer_block:
            pad_r = (level.window_size - width % level.window_size) % level.window_size
            pad_b = (level.window_size - height % level.window_size) % level.window_size
            if pad_r or pad_b:
                x = nn.functional.pad(x, (0, pad_r, 0, pad_b))
            padded_h, padded_w = x.shape[-2:]
            x = self._window_partition(x, level.window_size)
        for block in level.blocks:
            x = block(x)
        if level.transformer_block:
            x = self._window_reverse(
                x, level.window_size, padded_h, padded_w)
            if pad_r or pad_b:
                x = x[:, :, :height, :width].contiguous()
        output = x
        if level.downsample is not None:
            x = level.downsample(x)
        return x, output

    def forward(self, x):
        x = self.patch_embed(x)
        outputs = []
        for i, level in enumerate(self.levels):
            x, output = self._forward_level(level, x)
            if i in self.out_indices:
                output = getattr(self, 'outnorm{}'.format(i))(output)
                outputs.append(output.contiguous())
        return tuple(outputs)


@BACKBONES.register_module()
class mamba_vision_t_official(MambaVisionOfficial):
    def __init__(self, **kwargs):
        kwargs.setdefault('dim', 80)
        kwargs.setdefault('in_dim', 32)
        kwargs.setdefault('depths', (1, 3, 8, 4))
        kwargs.setdefault('drop_path_rate', 0.2)
        super().__init__(**kwargs)


@BACKBONES.register_module()
class mamba_vision_s_official(MambaVisionOfficial):
    def __init__(self, **kwargs):
        kwargs.setdefault('dim', 96)
        kwargs.setdefault('in_dim', 64)
        kwargs.setdefault('depths', (3, 3, 7, 5))
        kwargs.setdefault('drop_path_rate', 0.2)
        super().__init__(**kwargs)


@BACKBONES.register_module()
class mamba_vision_b_official(MambaVisionOfficial):
    def __init__(self, **kwargs):
        kwargs.setdefault('dim', 128)
        kwargs.setdefault('in_dim', 64)
        kwargs.setdefault('depths', (3, 3, 10, 5))
        kwargs.setdefault('drop_path_rate', 0.3)
        kwargs.setdefault('layer_scale', 1e-5)
        super().__init__(**kwargs)


@BACKBONES.register_module()
class mamba_vision_l2_official(MambaVisionOfficial):
    """MambaVision-L2 matching the official 224 and 512 checkpoints."""

    def __init__(self, use_512_windows=False, **kwargs):
        kwargs.setdefault('dim', 196)
        kwargs.setdefault('in_dim', 64)
        kwargs.setdefault('depths', (3, 3, 12, 5))
        kwargs.setdefault('num_heads', (4, 8, 16, 32))
        kwargs.setdefault(
            'window_size', (8, 8, 32, 16) if use_512_windows
            else (8, 8, 14, 7))
        kwargs.setdefault('drop_path_rate', 0.3)
        kwargs.setdefault('layer_scale', 1e-5)
        super().__init__(**kwargs)
