# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule


def _deterministic_bilinear_grid_sample(input_tensor, grid):
    """Match bilinear zero-padded grid_sample with deterministic grid grads.

    SymFormer samples a fixed sine/cosine positional tensor, so only the
    learned affine grid needs gradients. This avoids PyTorch 1.10 CUDA
    grid-sampler input-backward atomics while preserving affine gradients.
    """
    if input_tensor.requires_grad:
        raise RuntimeError(
            'deterministic Sym sampling expects fixed positional input')
    batch, channels, in_height, in_width = input_tensor.shape
    if grid.shape[0] != batch or grid.shape[-1] != 2:
        raise ValueError('grid must have shape BxHoutxWoutx2')
    out_height, out_width = grid.shape[1:3]
    x = ((grid[..., 0] + 1.0) * in_width - 1.0) * 0.5
    y = ((grid[..., 1] + 1.0) * in_height - 1.0) * 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    x1, y1 = x0 + 1.0, y0 + 1.0
    flat_input = input_tensor.reshape(batch, channels, -1)

    def gather(x_index, y_index):
        valid = ((x_index >= 0) & (x_index < in_width)
                 & (y_index >= 0) & (y_index < in_height))
        x_safe = x_index.clamp(0, in_width - 1).long()
        y_safe = y_index.clamp(0, in_height - 1).long()
        linear = (y_safe * in_width + x_safe).reshape(batch, 1, -1)
        sampled = torch.gather(
            flat_input, 2, linear.expand(-1, channels, -1)).reshape(
                batch, channels, out_height, out_width)
        return sampled * valid[:, None].to(dtype=input_tensor.dtype)

    value00 = gather(x0, y0)
    value01 = gather(x0, y1)
    value10 = gather(x1, y0)
    value11 = gather(x1, y1)
    return (value00 * ((x1 - x) * (y1 - y))[:, None]
            + value01 * ((x1 - x) * (y - y0))[:, None]
            + value10 * ((x - x0) * (y1 - y))[:, None]
            + value11 * ((x - x0) * (y - y0))[:, None])


def _deterministic_adaptive_max_pool2d(input_tensor, output_size):
    """Exact deterministic pooling for integer-ratio feature sizes.

    PyTorch 1.10 has no deterministic CUDA backward for adaptive max pooling.
    SymFormer's 512-pixel inputs produce power-of-two feature maps, for which
    pooling to 8x8 is exactly regular non-overlapping max pooling, identity, or
    integer nearest-neighbour replication. Unsupported shapes fail explicitly
    instead of silently changing the positional branch.
    """
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size
    in_h, in_w = input_tensor.shape[-2:]

    if (in_h, in_w) == (out_h, out_w):
        return input_tensor
    if in_h >= out_h and in_w >= out_w:
        if in_h % out_h != 0 or in_w % out_w != 0:
            raise RuntimeError(
                'Deterministic adaptive max pooling requires integer input/'
                f'output ratios, got {(in_h, in_w)} -> {(out_h, out_w)}')
        kernel = (in_h // out_h, in_w // out_w)
        return F.max_pool2d(input_tensor, kernel_size=kernel, stride=kernel)
    if in_h <= out_h and in_w <= out_w:
        if out_h % in_h != 0 or out_w % in_w != 0:
            raise RuntimeError(
                'Deterministic adaptive max pooling requires integer output/'
                f'input ratios, got {(in_h, in_w)} -> {(out_h, out_w)}')
        repeat_h, repeat_w = out_h // in_h, out_w // in_w
        batch_shape = input_tensor.shape[:-2]
        return input_tensor.reshape(
            *batch_shape, in_h, 1, in_w, 1).expand(
                *batch_shape, in_h, repeat_h, in_w, repeat_w).reshape(
                    *batch_shape, out_h, out_w)
    raise RuntimeError(
        'Deterministic adaptive max pooling does not support mixed resize '
        f'directions: {(in_h, in_w)} -> {(out_h, out_w)}')


@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)  # learned
        self.col_embed = nn.Embedding(col_num_embed, num_feats)  # learned
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class SymPositionalEncoding(BaseModule):
    """Position encoding with symmetry.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 left=True,
                 init_cfg=None):
        super(SymPositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset
        self.left = left

    def forward(self, mask):
        """Forward function for `SymPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        _bs, _fea, _h, _w = pos.shape
        pos_flip = torch.flip(pos, [-1])

        if self.left:
            pos_flip = pos_flip[:,:,:,:(_w//2)]
            pos = pos[:,:,:,(_w//2):]
            pos = torch.cat((pos_flip, pos), dim=-1)
        else:
            pos_flip = pos_flip[:,:,:,(_w//2):]
            pos = pos[:,:,:,:(_w//2)]
            pos = torch.cat((pos, pos_flip), dim=-1)
        assert (_bs, _fea, _h, _w) == pos.shape, "Not match shape"
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class GuidePositionalEncoding(BaseModule):
    """Position encoding with guide.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 left=True,
                 learn=True,
                 use_upper_prior=False,
                 upper_prior_boundary=1.0 / 3.0,
                 upper_prior_temperature=0.08,
                 upper_prior_init=0.02,
                 upper_prior_max=0.15,
                 upper_prior_mode='image',
                 init_cfg=None):
        super(GuidePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset
        self.left = left
        self.learn = learn
        self.use_upper_prior = use_upper_prior
        self.upper_prior_boundary = float(upper_prior_boundary)
        self.upper_prior_temperature = float(upper_prior_temperature)
        self.upper_prior_max = float(upper_prior_max)
        self.upper_prior_mode = str(upper_prior_mode)
        if self.upper_prior_mode not in ('image', 'lung'):
            raise ValueError(
                'upper_prior_mode must be either "image" or "lung"')
        if not 0.0 < self.upper_prior_boundary < 1.0:
            raise ValueError('upper_prior_boundary must be in (0, 1)')
        if self.upper_prior_temperature <= 0.0:
            raise ValueError('upper_prior_temperature must be positive')
        if self.upper_prior_max <= 0.0:
            raise ValueError('upper_prior_max must be positive')
        if not 0.0 <= upper_prior_init < self.upper_prior_max:
            raise ValueError(
                'upper_prior_init must be in [0, upper_prior_max)')

        if self.use_upper_prior:
            # Only the direction is learned. It is RMS-normalized in forward,
            # so the bounded scalar below controls the actual residual size.
            prior_direction = torch.randn(
                1, num_feats * 2, 1, 1, dtype=torch.float32)
            prior_direction = prior_direction / prior_direction.square(
            ).mean().sqrt().clamp_min(self.eps)
            self.upper_prior_direction = nn.Parameter(prior_direction)
            init_ratio = float(upper_prior_init) / self.upper_prior_max
            self.upper_prior_scale_raw = nn.Parameter(torch.tensor(
                math.atanh(init_ratio), dtype=torch.float32))
        else:
            self.register_parameter('upper_prior_direction', None)
            self.register_parameter('upper_prior_scale_raw', None)

        # Detached tensors for the detector head's interval-based diagnostics.
        self._dbg_upper_prior = {}
        # 编码空间变换
        self.extraction = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size=3),  # (bs, 8, 6, 6)
            nn.MaxPool2d(2, stride=2),  # (bs, 8, 3, 3)
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=2),  # (bs, 10, 2, 2)
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.learn_pos = nn.Sequential(
            nn.Linear(10 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 10)
        )

    def _soft_upper_profile(self, support):
        """Build a vertical profile within the occupied rows of ``support``."""
        support = support.to(dtype=torch.float32)
        support_rows = support.amax(dim=2)
        row_rank = support_rows.cumsum(dim=1) - 0.5
        support_height = support_rows.sum(
            dim=1, keepdim=True).clamp_min(1.0)
        y_normalized = (row_rank / support_height).clamp(0.0, 1.0)
        profile = torch.sigmoid(
            (self.upper_prior_boundary - y_normalized)
            / self.upper_prior_temperature)
        profile = profile * support_rows
        profile = profile[:, None, :, None]
        y_map = y_normalized[:, None, :, None].expand(
            -1, 1, -1, support.shape[2])
        support = support[:, None, :, :]
        return profile * support, y_map, support

    def _build_image_upper_prior(self, mask):
        """Build the legacy padding-aware image-coordinate upper profile."""
        valid = (mask == 0).to(dtype=torch.float32)
        return self._soft_upper_profile(valid)

    def _build_lung_upper_prior(self, mask, lung_mask):
        """Build a soft upper-third profile inside each lung half.

        ``lung_mask`` is an anatomy mask already aligned to the padded detector
        canvas. It may include the heart: the heart is predominantly below the
        upper-third boundary, so its residual contribution is softly attenuated
        rather than used to crop the detector input.
        """
        if lung_mask is None:
            raise ValueError(
                'upper_prior_mode="lung" requires a lung_mask tensor')
        if lung_mask.ndim == 4:
            if lung_mask.shape[1] != 1:
                raise ValueError(
                    'lung_mask must have one channel when it is 4-D')
            lung_mask = lung_mask[:, 0]
        if lung_mask.ndim != 3:
            raise ValueError(
                f'lung_mask must be BxHxW, got {tuple(lung_mask.shape)}')
        if lung_mask.shape[0] != mask.shape[0]:
            raise ValueError(
                'lung_mask batch size must match positional mask batch size')
        if tuple(lung_mask.shape[-2:]) != tuple(mask.shape[-2:]):
            lung_mask = F.interpolate(
                lung_mask[:, None].float(),
                size=mask.shape[-2:],
                mode='nearest')[:, 0]

        valid = (mask == 0).to(dtype=torch.float32)
        lung = (lung_mask > 0.5).to(dtype=torch.float32) * valid
        split = (lung.shape[2] + 1) // 2
        left_prior, left_y, left_support = self._soft_upper_profile(
            lung[:, :, :split])
        right_prior, right_y, right_support = self._soft_upper_profile(
            lung[:, :, split:])
        prior = torch.cat((left_prior, right_prior), dim=3)
        y_map = torch.cat((left_y, right_y), dim=3)
        support = torch.cat((left_support, right_support), dim=3)
        return prior, y_map, support

    def _build_upper_prior(self, mask, lung_mask=None):
        """Build the configured image- or lung-coordinate upper prior."""
        if self.upper_prior_mode == 'lung':
            return self._build_lung_upper_prior(mask, lung_mask)
        return self._build_image_upper_prior(mask)

    def effective_upper_prior_scale(self):
        """Return the bounded learnable strength of the upper-lung prior."""
        if self.upper_prior_scale_raw is None:
            return None
        return self.upper_prior_max * torch.tanh(
            self.upper_prior_scale_raw)

    def guide_affine(self, _pos, _feat):
        _bs, _C, _H, _W = _feat.size()
        if (torch.are_deterministic_algorithms_enabled()
                and _feat.is_cuda):
            new_feat = _deterministic_adaptive_max_pool2d(_feat, 8)
        else:
            # Preserve the original SymFormer operation outside strict CUDA.
            new_feat = F.adaptive_max_pool2d(_feat, 8)

        xs = self.extraction(new_feat)
        xs = xs.view(-1, 10 * 2 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, _pos.size(), align_corners=False)
        if (torch.are_deterministic_algorithms_enabled()
                and _pos.is_cuda):
            new_pos = _deterministic_bilinear_grid_sample(_pos, grid)
        else:
            new_pos = F.grid_sample(
                _pos, grid, mode='bilinear', padding_mode='zeros',
                align_corners=False)
        new_pos = torch.flip(new_pos, [-1])  # 翻转可以更好地学习关系
        return new_pos

    def forward(self, mask, feat, lung_mask=None):
        """Forward function for `GuidePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        _bs, _fea, _h, _w = pos.shape

        if self.left:
            pos_right = pos[:,:,:,((_w + 1)//2):]
            pos_flip = self.guide_affine(pos_right, feat)
            pos_right = pos[:,:,:,(_w//2):]
            pos = torch.cat((pos_flip, pos_right), dim=-1)
        else:
            pos_left = pos[:,:,:,:(_w//2)]
            pos_flip = self.guide_affine(pos_left, feat)
            pos_left = pos[:,:,:,:((_w + 1)//2)]
            pos = torch.cat((pos_left, pos_flip), dim=-1)

        if self.use_upper_prior:
            prior, prior_y, prior_support = self._build_upper_prior(
                mask, lung_mask=lung_mask)
            prior = prior.to(dtype=pos.dtype)
            direction = self.upper_prior_direction.to(dtype=pos.dtype)
            direction = direction / direction.square().mean(
            ).sqrt().clamp_min(self.eps)
            prior_scale = self.effective_upper_prior_scale().to(
                dtype=pos.dtype)
            prior_delta = prior_scale * direction * prior
            base_pos = pos
            pos = base_pos + prior_delta

            # Keep diagnostics as detached tensors to avoid a device sync on
            # every FPN level. The head converts them to numbers only at the
            # configured debug interval.
            prior_y = prior_y.to(device=prior.device)
            prior_support = prior_support.to(device=prior.device)
            upper_zone = (
                (prior_y < self.upper_prior_boundary)
                * prior_support)
            middle_zone = (
                (prior_y >= self.upper_prior_boundary)
                * (prior_y < 2.0 * self.upper_prior_boundary)
                * prior_support)
            lower_zone = (
                (prior_y >= 2.0 * self.upper_prior_boundary)
                * prior_support)

            def zone_mean(zone):
                return (
                    (prior.float() * zone).sum()
                    / zone.sum().clamp_min(1.0))

            split = (_w + 1) // 2
            self._dbg_upper_prior = {
                'scale': prior_scale.detach(),
                'delta_rel': (
                    prior_delta.detach().float().norm()
                    / base_pos.detach().float().norm().clamp_min(self.eps)),
                'upper_mean': zone_mean(upper_zone).detach(),
                'middle_mean': zone_mean(middle_zone).detach(),
                'lower_mean': zone_mean(lower_zone).detach(),
                'support_fraction': prior_support.detach().float().mean(),
                'upper_support_fraction': (
                    upper_zone.sum()
                    / prior_support.sum().clamp_min(1.0)).detach(),
                'left_support_fraction': prior_support[
                    :, :, :, :split].detach().float().mean(),
                'right_support_fraction': prior_support[
                    :, :, :, split:].detach().float().mean(),
            }
        else:
            self._dbg_upper_prior = {}

        assert (_bs, _fea, _h, _w) == pos.shape, "Not match shape"
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps}, '
        repr_str += f'use_upper_prior={self.use_upper_prior}, '
        repr_str += f'upper_prior_mode={self.upper_prior_mode}, '
        repr_str += f'upper_prior_boundary={self.upper_prior_boundary}, '
        repr_str += f'upper_prior_max={self.upper_prior_max})'
        return repr_str
