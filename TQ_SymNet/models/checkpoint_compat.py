"""TQ-SymNet: extracted measured implementations; see PROVENANCE.json."""
import math
import logging
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
class _SpatialBiasEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate

        self.activation = nn.GELU()

    def forward(self, *args, **kwargs):
        raise RuntimeError('Inactive checkpoint compatibility slot')


class _SpatialDistanceBias(nn.Module):
    def __init__(self, temperature: float = 10.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, *args, **kwargs):
        raise RuntimeError('Inactive checkpoint compatibility slot')


class _DynamicRegionTokenPool(nn.Module):
    def __init__(self, in_channels: int = 256, token_dim: int = 256):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, token_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim * 2, token_dim),
        )
        # Kaiming init — NOT zero-init. These projection layers must carry
        # gradient from contrastive loss back to backbone features.
        for m in self.token_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args, **kwargs):
        raise RuntimeError('Inactive checkpoint compatibility slot')


class _LogGaborPhaseCongruency(nn.Module):
    def __init__(self,
                 num_scales: int = 3,
                 num_orientations: int = 6,
                 min_wavelength: float = 3.0,
                 scale_multiplier: float = 2.1,
                 sigma_on_frequency: float = 0.55,
                 noise_k: float = 2.0,
                 eps: float = 1e-6):
        super().__init__()
        self.num_scales = int(num_scales)
        self.num_orientations = int(num_orientations)
        self.min_wavelength = float(min_wavelength)
        self.scale_multiplier = float(scale_multiplier)
        self.sigma_on_frequency = float(sigma_on_frequency)
        self.noise_k = float(noise_k)
        self.eps = float(eps)
        self._filter_cache = {}

    def forward(self, *args, **kwargs):
        raise RuntimeError('Inactive checkpoint compatibility slot')


class _RegionTransformerBranchV3_3(nn.Module):
    def __init__(self, feat_channels: int = 256,
                 d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1,
                 drop_path_rate: float = 0.1,
                 token_dropout: float = 0.15,
                 spatial_temp: float = 1.5,
                 max_regions: int = 200,
                 ema_momentum: float = 0.0,
                 positive_purity_threshold: float = 0.5,
                 negative_purity_threshold: float = 0.05,
                 minimum_positive_purity: float = math.pi / 4,
                 ensure_gt_positive: bool = True,
                 contrast_temperature: float = 0.07,
                 purity_weight_floor: float = 0.0,
                 hard_negative_ratio: float = 3.0,
                 hard_negative_expansion: float = 1.5,
                 min_hard_negatives: int = 16,
                 far_negative_weight: float = 0.25,
                 boundary_rank_margin: float = 0.10,
                 boundary_rank_weight: float = 0.0):
        super().__init__()
        if not 0 <= negative_purity_threshold < positive_purity_threshold <= 1:
            raise ValueError(
                'Expected 0 <= negative_purity_threshold < '
                'positive_purity_threshold <= 1')
        self.d_model = d_model
        self.token_dropout = token_dropout
        self.max_regions = max_regions
        self.feat_channels = feat_channels
        self.ema_momentum = ema_momentum
        self.positive_purity_threshold = positive_purity_threshold
        self.negative_purity_threshold = negative_purity_threshold
        self.minimum_positive_purity = float(minimum_positive_purity)
        self.ensure_gt_positive = bool(ensure_gt_positive)
        if not 0.0 < self.minimum_positive_purity <= 1.0:
            raise ValueError(
                'minimum_positive_purity must be in (0, 1]')
        if contrast_temperature <= 0:
            raise ValueError('contrast_temperature must be positive')
        if not 0 <= purity_weight_floor <= 1:
            raise ValueError('purity_weight_floor must be in [0, 1]')
        if ema_momentum != 0:
            raise ValueError(
                'ema_momentum must remain 0: NAM region IDs are image-local, '
                'so an EMA indexed only by region ID mixes unrelated CXRs')
        if hard_negative_ratio <= 0:
            raise ValueError('hard_negative_ratio must be positive')
        if hard_negative_expansion < 1:
            raise ValueError('hard_negative_expansion must be >= 1')
        if min_hard_negatives < 0:
            raise ValueError('min_hard_negatives must be non-negative')
        if not 0 < far_negative_weight <= 1:
            raise ValueError('far_negative_weight must be in (0, 1]')
        if boundary_rank_margin < 0 or boundary_rank_weight < 0:
            raise ValueError(
                'boundary rank margin/weight must be non-negative')
        self.contrast_temperature = float(contrast_temperature)
        self.purity_weight_floor = float(purity_weight_floor)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.hard_negative_expansion = float(hard_negative_expansion)
        self.min_hard_negatives = int(min_hard_negatives)
        self.far_negative_weight = float(far_negative_weight)
        self.boundary_rank_margin = float(boundary_rank_margin)
        self.boundary_rank_weight = float(boundary_rank_weight)

        # --- Dynamic token pooler: raw C2 features pooled via .npz label_map ---
        self.token_pool = _DynamicRegionTokenPool(
            in_channels=feat_channels, token_dim=d_model)

        # Region order is arbitrary across images. Encode real geometry rather
        # than assigning a learned positional identity to token index r.
        self.geometry_position = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model))
        for module in self.geometry_position.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # --- Spatial distance bias ---
        self.spatial_bias_fn = _SpatialDistanceBias(temperature=spatial_temp)

        # --- Transformer encoder ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.encoder_layers = nn.ModuleList([
            _SpatialBiasEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, dropout=dropout,
                drop_path_rate=dpr[i])
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self._last_instance_sim = -1.0  # for debug monitoring
        self._last_boxalign_records = None

        # --- Projection head for contrastive loss (SimCLR-style) ---
        # Kaiming init — NOT zero-init. Absorbs contrastive-specific low-rank
        # pressure while keeping node_out high-rank for detection.
        self.contrast_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),                        # inplace=False: safe for autograd
            nn.Linear(d_model * 2, 128),
        )
        for m in self.contrast_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args, **kwargs):
        raise RuntimeError('Inactive checkpoint compatibility slot')

def _reset_boxalign_debug_state(owner, num_levels: int) -> None:
    """Initialize diagnostics even when the optional BoxAlign path is off."""
    num_levels = max(0, int(num_levels))
    owner._dbg_boxalign_loss_raw = 0.0
    owner._dbg_boxalign_total_gt = 0
    owner._dbg_boxalign_valid_gt = 0
    owner._dbg_boxalign_anchor_count = 0
    owner._dbg_boxalign_level_counts = [0] * num_levels
    owner._dbg_boxalign_mean_iou = 0.0
    owner._dbg_boxalign_positive_similarity = 0.0
    owner._dbg_boxalign_negative_similarity = 0.0
    owner._dbg_boxalign_single_region_gt = 0
    owner._dbg_boxalign_weighted = 0.0
    owner._dbg_last_align_reg_features = None

class UnsupportedBranch(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise ValueError('This historical branch is not part of TQ-SymNet')

_PerPixelLBPBranch = _BoundaryResponsiveGate = UnsupportedBranch
_RegressionDetailInjector = UnsupportedBranch
