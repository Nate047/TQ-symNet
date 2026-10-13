"""
NAMLab_att v3.3: Dynamic Region Token + Cross-Scale Attention + Edge Supervision

v3 -> v3.3 core changes:
  1. Dynamic Region Token (P0 fix): tokens are pooled from FPN P2 features
     using label_map as pooling index, NOT from handcrafted .npz features.
     Contrastive loss gradients now flow: contr_loss -> tokens -> P2 -> FPN
     -> backbone. This is the CRITICAL fix — v3's contrastive loss could
     only optimize input_proj and Transformer params, never the backbone.

  2. Cross-Scale Feature Enhancement (P2): deep P5 features query shallow P2
     features via channel-wise cross-attention. P5's semantic guidance is
     injected into P2's high-resolution spatial details. Applied only at
     P2 level (where spatial detail matters most for small lesions).
     Gates with multiplicative fusion (avoids concat dimension-imbalance).

  3. Edge Supervision (P1): lightweight edge head on P2 features, supervised
     by L* lightness spatial gradient (|L - blur(L)|, computed from same
     lbp_lab tensor already used by BRDG gate). L1 loss encourages backbone
     shallow layers to preserve tissue-boundary information. Training-only;
     discarded at inference.

  4. Retained from v3: Per-pixel LBP+Lab branch, BRDG multiplicative gate,
     Spatial-bias Transformer encoder layers, Token dropout, Stochastic depth.

  5. Removed from v3: Handcrafted region token features (node_wld, node_lab,
     node_stat, node_glcm from .npz). .npz is now only used for label_map
     (pooling index), edge_index, and edge_dc (DC aggregation as optional
     node feature).

Gradient paths (all verified):
  Det Loss       -> head -> FPN -> backbone -> input
  Contrast Loss  -> region_tokens -> P2 -> FPN -> backbone  [NEW]
  Edge Loss      -> edge_head -> P2 -> FPN -> backbone      [NEW]
  LBP Gate       -> gate -> feat_nam -> LBP branch (separate path)

Components:
    _DynamicRegionTokenPool    - Pool region tokens from FPN features [NEW]
    _CrossScaleFeatureEnhance  - Deep queries shallow via channel attention [NEW]
    _EdgeSupervisionHead       - Lightweight edge detection on shallow features [NEW]
    _RegionTransformerBranchV3_3 - Updated: pools from FPN, not .npz [MODIFIED]
    [Reused from v3]: _SpatialBiasEncoderLayer, _SpatialDistanceBias,
                      _PerPixelLBPBranch, _BoundaryResponsiveGate,
                      compute_lbp_lab, _drop_path
    RetinaNAMGuideAttHeadV3_3  - Updated dual-branch Head [MODIFIED]
    RetinaNAMAttV3_3           - Detector [MODIFIED]
    NAMDebugHook               - Reused from v3

Data format (.npz, reused from V2):
    label_map  : (H, W) int32       pixel -> region ID (used for pooling index)
    edge_index : (2, E) int32       adjacency edges
    edge_dc    : (E,) float32       color distance (used for DC agg)
    (node_wld, node_lab, node_stat, node_glcm, node_area are IGNORED in v3.3)

Config example:
    bbox_head=dict(
        type='RetinaNAMGuideAttHeadV3_3',
        num_classes=2, num_query=500, ...
        d_model=256, nhead=8, num_layers=4, dropout=0.1,
        cross_scale=True,                    # enable cross-scale attention
        edge_supervision=True,               # enable edge auxiliary loss
        loss_cls=..., loss_bbox=...)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Reused from v3: per-pixel LBP + Lab computation
# ============================================================================
def compute_lbp_gray(img: torch.Tensor,
                    lbp_radius: int = 1,
                    lbp_points: int = 8) -> torch.Tensor:
    """Compute per-pixel LBP codes and Lab colour from a batch of images.

    LBP is a per-pixel texture descriptor.  This function computes it at
    full image resolution without superpixel aggregation, so downstream
    modules can process texture at the pixel level (no piecewise-constant
    artefacts from region averaging).

    Args:
        img: (B, 3, H, W) image tensor in [0, 255] (before normalisation)
        lbp_radius: LBP sampling radius (default 1 -> 3x3 neighbourhood)
        lbp_points: number of LBP neighbours (default 8)
    Returns:
        (B, 177, H, W) concatenated [LBP_r1(59)|LBP_r2(59)|LBP_r3(59)] — grayscale LBP only
    """
    B, C, H, W = img.shape
    device = img.device

    mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    img_raw = img * std + mean
    gray = 0.299 * img_raw[:, 0:1] + 0.587 * img_raw[:, 1:2] + 0.114 * img_raw[:, 2:3]
    gray = gray[:, 0]

    gray_pad = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]

    lbp_feats = []
    for scale, r in enumerate([1, 2, 3]):
        if r == 1:
            g_pad = gray_pad
        else:
            g_pad = F.pad(gray, (r, r, r, r), mode='reflect')

        code = torch.zeros(B, H, W, device=device)
        for i, (dy, dx) in enumerate(offsets):
            ny, nx = r + dy * r, r + dx * r
            code += (g_pad[:, ny:ny + H, nx:nx + W] >= gray).float() * (2 ** i)
        code_1h = _lbp_to_uniform_onehot(code.long(), device)
        lbp_feats.append(code_1h)

    lbp_cat = torch.cat(lbp_feats, dim=-1)

    out = lbp_cat.permute(0, 3, 1, 2)
    return out


def _lbp_to_uniform_onehot(code: torch.Tensor,
                           device: torch.device) -> torch.Tensor:
    """Map LBP 8-bit codes -> uniform-pattern one-hot (59 bins)."""
    B, H, W = code.shape
    uniform_map = torch.full((256,), 0, dtype=torch.long, device=device)
    uniform_codes = [
        0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56,
        60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143,
        159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240,
        241, 243, 247, 248, 249, 251, 252, 253, 254, 255
    ]
    for i, uc in enumerate(uniform_codes):
        uniform_map[uc] = i
    code_mapped = uniform_map[code.clamp(0, 255)]
    onehot = F.one_hot(code_mapped, num_classes=59).float()
    return onehot


def _lab_f(t: torch.Tensor) -> torch.Tensor:
    """Lab helper: f(t) = t^(1/3) for t > 0.008856 else 7.787*t + 16/116."""
    delta = 6.0 / 29.0
    return torch.where(t > delta ** 3, t ** (1.0 / 3.0),
                       t / (3 * delta ** 2) + 4.0 / 29.0)


# ============================================================================
# _drop_path: stochastic depth (functional form)
# ============================================================================
def _drop_path(x: torch.Tensor, drop_prob: float,
               training: bool) -> torch.Tensor:
    """Stochastic Depth per sample (function, not nn.Module)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = mask.floor_()
    return x * (mask / keep_prob)


# ============================================================================
# _SpatialBiasEncoderLayer: Transformer encoder layer with spatial distance bias
# ============================================================================
class _SpatialBiasEncoderLayer(nn.Module):
    """Transformer encoder layer with additive spatial distance bias.

    Standard Pre-LN Transformer with an extra spatial_bias term added to
    attention logits before softmax.  Nearer regions get higher attention.

    Uses nn.MultiheadAttention with attn_mask as additive float bias.
    """

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

    def forward(self, x: torch.Tensor,
                spatial_bias: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=spatial_bias, need_weights=False)
        x = x + _drop_path(self.dropout1(attn_out),
                           self.drop_path_rate, self.training)
        ff_out = self.linear2(self.dropout(self.activation(
            self.linear1(self.norm2(x)))))
        x = x + _drop_path(self.dropout2(ff_out),
                           self.drop_path_rate, self.training)
        return x


# ============================================================================
# _SpatialDistanceBias: computes spatial attention bias from region centroids
# ============================================================================
class _SpatialDistanceBias(nn.Module):
    """Compute additive attention bias from region centroid distances.

    bias[i,j] = -||c_i - c_j||_2 / temperature

    Negative bias for distant pairs -> lower attention.
    temperature is learnable (softplus to keep > 0).
    """

    def __init__(self, temperature: float = 10.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, centroids: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(centroids, centroids, p=2)
        temp = F.softplus(self.temperature) + 0.01
        bias = -dist / temp
        return bias


# ============================================================================
# _PerPixelLBPBranch: per-pixel grayscale LBP -> FFN -> FPN-compatible features
# ============================================================================
class _PerPixelLBPBranch(nn.Module):
    """Per-pixel LBP + Lab colour -> pointwise MLP -> smooth conv -> feature map.

    Processes texture at pixel resolution without superpixel aggregation,
    avoiding the piecewise-constant artifacts caused by label_map indexing
    of region-level token embeddings.
    """

    def __init__(self, in_channels: int = 177, feat_channels: int = 256):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lbp_lab: torch.Tensor,
                target_size: tuple) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(lbp_lab, target_size)
        x = self.pointwise(x)
        x = self.smooth(x)
        return x


# ============================================================================
# _BoundaryResponsiveGate: NAM modulates SymFormer at texture boundaries
# ============================================================================
class _BoundaryResponsiveGate(nn.Module):
    """Boundary-responsive gating aligned with BRDG (Ahmed et al., CVPR 2026).

    BRDG's Boundary Detector Agent is an independent module whose gating
    signal controls refinement multiplicatively.

    Design (v3.1 BRDG-aligned):
      1. Edge detector: L* spatial gradient -> scalar edge response (B,1,H,W)
      2. Content gate: gate_net(f_nam) -> tanh -> per-channel modulation
      3. Spatial gate: sigmoid(edge_map * temperature) -> per-pixel activation
      4. Multiplicative fusion: gate = tanh(gate_net) * sigmoid(edge * T) * scale
      5. Residual fusion: fused = f_sym + nam_proj(f_nam) * gate

    Why multiplicative: avoids 256:1 dimension-imbalance gradient suppression
    of channel-concat approach (GPONet 2024, GDCN 2023).
    """

    def __init__(self, channels: int = 256, edge_temp_init: float = 1.0):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.edge_temp_raw = nn.Parameter(torch.tensor(edge_temp_init))
        # mod_scale=0.3 constrains max modulation to 30% even with Xavier init.
        # tanh*sigmoid*0.3 structure is inherently bounded — zero-init is
        # unnecessary and causes "dead branch" where gate≈0 for many iters.
        self.mod_scale = nn.Parameter(torch.tensor(0.3))
        # Xavier init — gate_net[-1] output goes through tanh (gain≈1),
        # not ReLU (gain=√2).  Xavier is the theoretically correct choice
        # for tanh; kaiming w/ relu would overscale weights by √2×.
        nn.init.xavier_normal_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)
        self.nam_proj = nn.Conv2d(channels, channels, 1)
        nn.init.kaiming_normal_(self.nam_proj.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.zeros_(self.nam_proj.bias)

    def forward(self, f_sym: torch.Tensor,
                f_nam: torch.Tensor,
                lbp_lab: torch.Tensor) -> tuple:
        h, w = f_sym.shape[2:]
        if lbp_lab.shape[2] != h or lbp_lab.shape[3] != w:
            lbp_down = F.adaptive_avg_pool2d(lbp_lab, (h, w))
        else:
            lbp_down = lbp_lab
        # Edge from grayscale via LBP scale1 response gradient
        # LBP one-hot has spatial structure: compute |oh - blur(oh)| across bins
        lbp_s1 = lbp_down[:, :59]  # (B, 59, H, W) — scale1 one-hot
        lbp_s1_blur = F.avg_pool2d(lbp_s1, 3, 1, 1)
        l_channel = (lbp_s1 - lbp_s1_blur).abs().mean(dim=1, keepdim=True)

        # Edge detection from LBP texture gradient
        l_blur = F.avg_pool2d(l_channel, kernel_size=3, stride=1, padding=1)
        edge_raw = torch.abs(l_channel - l_blur)
        edge_map = self.edge_conv(edge_raw)

        content_gate = torch.tanh(self.gate_net(f_nam))
        edge_temp = F.softplus(self.edge_temp_raw) + 0.01
        spatial_gate = torch.sigmoid(edge_map * edge_temp)
        gate = content_gate * spatial_gate * self.mod_scale

        nam_mod = self.nam_proj(f_nam) * gate
        fused = f_sym + nam_mod

        gate_abs = gate.detach().abs()
        gate_stats = {
            'gate_mean': gate_abs.mean().item(),
            'gate_active': (gate_abs > 0.1).float().mean().item(),
            'edge_mean': edge_map.detach().abs().mean().item(),
            'spatial_gate_mean': spatial_gate.detach().mean().item(),
            'edge_temp': edge_temp.item(),
        }
        return fused, gate_stats


# ============================================================================
# NEW: _DynamicRegionTokenPool
# Pools region tokens from FPN features using label_map as pooling index.
# This is the CRITICAL fix for v3's broken gradient path.
# ============================================================================
class _DynamicRegionTokenPool(nn.Module):
    """Pool region tokens from FPN features using label_map as pooling index.

    Replaces v3's handcrafted-token approach (which used .npz node_wld/node_lab
    constants that have no gradient connection to the backbone).

    Gradient path:
        contr_loss -> contrast_head -> Transformer -> tokens
            -> token_proj -> pooled_features -> P2 -> FPN -> backbone ✓

    Args:
        in_channels: FPN feature channels (256)
        token_dim: output token dimension (= d_model, default 256)
    """

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

    def forward(self, fpn_feat: torch.Tensor,
                label_map: torch.Tensor) -> tuple:
        """Pool region tokens from FPN features.

        Args:
            fpn_feat: (C, H_f, W_f) single sample's P2 feature
            label_map: (H, W) region ID per pixel (original resolution)
        Returns:
            (tokens, centroids, label_map_down) or (None, None, None) if N<=1
            tokens: (N_valid, token_dim)
            centroids: (N_valid, 2)
            label_map_down: (H_f, W_f) region IDs at feature resolution
        """
        C, H_f, W_f = fpn_feat.shape
        H_lm, W_lm = label_map.shape

        # Downsample label_map to match feature resolution
        if H_lm != H_f or W_lm != W_f:
            label_map_ds = F.interpolate(
                label_map.unsqueeze(0).unsqueeze(0).float(),
                size=(H_f, W_f), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        else:
            label_map_ds = label_map.long()

        N = int(label_map_ds.max().item()) + 1
        if N <= 1:
            return None, None, None

        device = fpn_feat.device

        # --- Mean pooling per region via one-hot matrix multiply ---
        # fpn_flat: (C, H_f*W_f), labels_flat: (H_f*W_f,)
        # labels_onehot: (H_f*W_f, N), pooled = fpn_flat @ labels_onehot: (C, N)
        fpn_flat = fpn_feat.reshape(C, -1)          # (C, H_f*W_f)
        labels_flat = label_map_ds.reshape(-1).clamp(0, N - 1)  # (H_f*W_f,)
        labels_onehot = F.one_hot(labels_flat, num_classes=N).float()  # (HW, N)
        pooled = (fpn_flat @ labels_onehot).t()      # (N, C)
        counts = labels_onehot.sum(dim=0)             # (N,)

        valid_mask = counts > 0
        pooled[valid_mask] = pooled[valid_mask] / counts[valid_mask].unsqueeze(-1)

        # --- Filter to valid regions only ---
        valid_ids = valid_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_valid,)
        pooled_valid = pooled[valid_ids]  # (N_valid, C)

        # --- Compute centroids BEFORE remapping (uses original region IDs) ---
        centroids = self._compute_centroids(label_map_ds, valid_ids, N, device)
        centroids = centroids / torch.tensor(
            [W_f, H_f], dtype=torch.float32, device=device)

        # --- Remap label_map from original region IDs → compact 0..N_valid-1 ---
        # label_map_ds contains original IDs (e.g. {0,1,3,5}). After filtering
        # to valid_ids, tokens are indexed 0..N_valid-1. label_map must be
        # remapped so downstream indexing (token dropout remap, contrastive_loss
        # TB masking) uses the same compact space.
        id_to_compact = torch.full((N,), -1, dtype=torch.long, device=device)
        id_to_compact[valid_ids] = torch.arange(
            len(valid_ids), device=device)
        label_map_compact = id_to_compact[label_map_ds.clamp(0, N - 1)]

        # --- Project to token_dim ---
        tokens = self.token_proj(pooled_valid)  # (N_valid, token_dim)

        return tokens, centroids, label_map_compact

    @staticmethod
    def _compute_centroids(label_map: torch.Tensor, valid_ids: torch.Tensor,
                           N: int, device: torch.device) -> torch.Tensor:
        """Compute (x, y) centroids for each valid region."""
        H, W = label_map.shape
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij')

        flat_labels = label_map.reshape(-1).clamp(0, N - 1)
        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)

        cnt = torch.zeros(N, device=device)
        cx = torch.zeros(N, device=device)
        cy = torch.zeros(N, device=device)

        ones = torch.ones_like(x_flat)
        cnt.scatter_add_(0, flat_labels, ones)
        cx.scatter_add_(0, flat_labels, x_flat)
        cy.scatter_add_(0, flat_labels, y_flat)

        valid = cnt > 0
        cx[valid] = cx[valid] / cnt[valid]
        cy[valid] = cy[valid] / cnt[valid]

        centroids = torch.stack([cx[valid_ids], cy[valid_ids]], dim=-1)
        return centroids


# ============================================================================
# NEW: _CrossScaleFeatureEnhance
# Deep features (P5) query shallow features (P2) via channel-wise cross-attn.
# ============================================================================
class _SingleCrossScaleAttn(nn.Module):
    """单层跨尺度注意力：深层 QUERY 浅层 → 增强浅层特征。

    深层语义特征（低分辨率）上采样后作为 QUERY，
    浅层细节特征（高分辨率）提供 KEY 和 VALUE。
    通道压缩到 C//4 降低计算量，乘法门控避免维度不平衡。

    Args:
        channels: feature dimension (256)
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.query_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
        )
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, deep_feat: torch.Tensor,
                shallow_feat: torch.Tensor) -> torch.Tensor:
        h_s, w_s = shallow_feat.shape[2:]
        deep_up = F.interpolate(deep_feat, size=(h_s, w_s),
                                mode='bilinear', align_corners=False)
        Q = self.query_conv(deep_up)       # (B, C//4, h_s, w_s)
        K = self.key_conv(shallow_feat)    # (B, C//4, h_s, w_s)
        V = self.value_conv(shallow_feat)  # (B, C, h_s, w_s)
        attn = torch.sigmoid(
            (Q * K).sum(dim=1, keepdim=True) / (Q.shape[1] ** 0.5))
        enhanced = V * attn
        enhanced = self.out_proj(enhanced)
        gate = self.gate(torch.cat([shallow_feat, enhanced], dim=1))
        return shallow_feat + enhanced * gate



# ============================================================================
# _SpatialCrossAttnNxN: TB-FusionNet spatial cross-attention (NxN matrix)
# ============================================================================
class _SpatialCrossAttnNxN(nn.Module):
    """NxN spatial cross-attention: Q(high) @ K^T(low) -> NxN -> reweight V(low).

    Only used for P5->P4 where N=256~1024, memory ~256KB~4MB (manageable).

    Args:
        channels: feature dimension (256)
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.Wq = nn.Linear(channels, channels // 8)  # compressed Q
        self.Wk = nn.Linear(channels, channels // 8)  # compressed K
        self.Wv = nn.Linear(channels, channels)       # V stays full
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, deep_feat, shallow_feat):
        B, C, h_s, w_s = shallow_feat.shape
        # Upsample deep features
        deep_up = F.interpolate(deep_feat, size=(h_s, w_s),
                                mode='bilinear', align_corners=False)
        N = h_s * w_s
        # Reshape: (B, C, H, W) -> (B, N, C)
        Q = deep_up.reshape(B, C, N).transpose(1, 2)     # (B, N, C)
        K = shallow_feat.reshape(B, C, N).transpose(1, 2) # (B, N, C)
        V = shallow_feat.reshape(B, C, N).transpose(1, 2) # (B, N, C)
        # Linear projections
        Qp = self.Wq(Q)  # (B, N, C//8)
        Kp = self.Wk(K)  # (B, N, C//8)
        Vp = self.Wv(V)  # (B, N, C)
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        scale = (C // 8) ** 0.5
        attn = torch.softmax(Qp @ Kp.transpose(-2, -1) / scale, dim=-1)  # (B, N, N)
        enhanced = attn @ Vp  # (B, N, C)
        # Reshape back
        enhanced = enhanced.transpose(1, 2).reshape(B, C, h_s, w_s)
        enhanced = self.out_proj(enhanced)
        return enhanced


# ============================================================================
# _ChannelCrossAttnFC: TB-FusionNet channel cross-attention (FC-SENet)
# ============================================================================
class _ChannelCrossAttnFC(nn.Module):
    """FC-based channel cross-attention: avgpool -> FC-ReLU-FC-Sigmoid -> reweight.

    High-level channels guide enhancement/suppression of low-level channels.
    Global context via avgpool, nonlinear mapping via FC bottleneck.
    """

    def __init__(self, channels: int = 256, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels * 2, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        # zero-init fc2 → sigmoid(0)=0.5 → residual form: feat * (1 + (w-0.5)*2)
        # initial output = feat * 1.0 (identity at t=0)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, deep_feat, shallow_feat):
        g_deep = deep_feat.mean(dim=[2, 3])
        g_shallow = shallow_feat.mean(dim=[2, 3])
        g_cat = torch.cat([g_deep, g_shallow], dim=-1)
        w = self.fc2(F.relu(self.fc1(g_cat)))
        w = self.sigmoid(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        # Residual form: w∈(0,1), initial w=0.5 → output = 0 (true residual)
        return shallow_feat * ((w - 0.5) * 2.0)

# ============================================================================
# _TBFusionCrossScale: 层级式 P5→P4→P3→P2（替代原 P5→P2）
# ============================================================================
class _TBFusionCrossScale(nn.Module):
    """TB-FusionNet style: SCA(NxN for P5-P4, channel-dot for others) + CCA(FC-SENet) + ADD.

    P5->P4: Full NxN spatial cross-attention (N=16x16=256, ~256KB memory)
    P4->P3, P3->P2: Channel-wise dot-product SCA (O(HW), preserves spatial structure)
    All levels: FC-based channel cross-attention (SENet-style)
    Fusion: Fi+1 + SCA + CCA (ADD preserves feature hierarchy, independent weights)
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.p5_p4_sca = _SpatialCrossAttnNxN(channels)
        self.p5_p4_cca = _ChannelCrossAttnFC(channels)
        self.p4_p3_sca = _SingleCrossScaleAttn(channels)
        self.p4_p3_cca = _ChannelCrossAttnFC(channels)
        self.p3_p2_sca = _SingleCrossScaleAttn(channels)
        self.p3_p2_cca = _ChannelCrossAttnFC(channels)

    def forward(self, feats):
        P2, P3, P4, P5 = feats[0], feats[1], feats[2], feats[3]
        P4e = P4 + self.p5_p4_sca(deep_feat=P5, shallow_feat=P4) \
                 + self.p5_p4_cca(deep_feat=P5, shallow_feat=P4)
        P3e = P3 + self.p4_p3_sca(deep_feat=P4e, shallow_feat=P3) \
                 + self.p4_p3_cca(deep_feat=P4e, shallow_feat=P3)
        P2e = P2 + self.p3_p2_sca(deep_feat=P3e, shallow_feat=P2) \
                 + self.p3_p2_cca(deep_feat=P3e, shallow_feat=P2)
        return [P2e, P3e, P4e, P5]


# ============================================================================
# NEW: _EdgeSupervisionHead
# Lightweight edge detection on shallow features for auxiliary supervision.
# ============================================================================
class _EdgeSupervisionHead(nn.Module):
    """Lightweight edge detection head for auxiliary supervision.

    Predicts a scalar edge response from P2 features, supervised by the
    L* lightness spatial gradient (|L - blur(L)|). This encourages the
    backbone's shallow layers to preserve tissue-boundary information,
    which is critical for TB lesion delineation.

    Training-only; weights are discarded at inference.

    Gradient path:
        edge_loss -> edge_head -> P2 -> FPN -> backbone ✓

    Args:
        in_channels: input feature channels (256)
        hidden_channels: hidden conv channels (64)
    """

    def __init__(self, in_channels: int = 256, hidden_channels: int = 64):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        # No zero-init — edge head must produce non-zero output from start
        # to avoid zero-gradient edge loss.

    def forward(self, shallow_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shallow_feat: (B, C, H, W) P2 features
        Returns:
            (B, 1, H, W) edge prediction
        """
        return self.convs(shallow_feat)

    def compute_edge_loss(self, edge_pred: torch.Tensor,
                          lbp_lab: torch.Tensor) -> torch.Tensor:
        """Compute L1 edge loss against L* spatial gradient.

        Args:
            edge_pred: (B, 1, H, W) predicted edge map
            lbp_lab: (B, 180, H_orig, W_orig) per-pixel grayscale LBP
        Returns:
            scalar L1 loss
        """
        h, w = edge_pred.shape[2:]
        # Downsample lbp_lab to match edge_pred resolution
        if lbp_lab.shape[2] != h or lbp_lab.shape[3] != w:
            lbp_down = F.adaptive_avg_pool2d(lbp_lab, (h, w))
        else:
            lbp_down = lbp_lab

        # L* lightness channel (index 177)
        # Edge from grayscale via LBP scale1 response gradient
        # LBP one-hot has spatial structure: compute |oh - blur(oh)| across bins
        lbp_s1 = lbp_down[:, :59]  # (B, 59, H, W) — scale1 one-hot
        lbp_s1_blur = F.avg_pool2d(lbp_s1, 3, 1, 1)
        l_channel = (lbp_s1 - lbp_s1_blur).abs().mean(dim=1, keepdim=True)
        # Edge GT: |L* - blur(L*)| — same computation as BRDG edge detector
        l_blur = F.avg_pool2d(l_channel, kernel_size=3, stride=1, padding=1)
        edge_gt = torch.abs(l_channel - l_blur)

        return F.l1_loss(edge_pred, edge_gt)


# ============================================================================
# _RegionTransformerBranchV3_3
# Core change: tokens pooled from FPN features, NOT from .npz handcrafted data.
# ============================================================================
class _RegionTransformerBranchV3_3(nn.Module):
    """Region Transformer NAM branch v3.3.

    v3 -> v3.3 core change:
      Tokens are pooled from FPN P2 features using label_map as pooling index,
      NOT built from handcrafted .npz features (node_wld, node_lab, etc.).
      This fixes the broken gradient path — contrastive loss gradients now
      flow through FPN to backbone.

    Pipeline:
      1. Pool region tokens from FPN P2 features via label_map mask
      2. Project to d_model + add learnable position embedding
      3. Compute spatial distance bias from region centroids
      4. TransformerEncoder x L layers (global self-attention + spatial bias)
      5. Contrastive loss (InfoNCE) pushes TB tokens together, away from nonTB
    """

    def __init__(self, feat_channels: int = 256,
                 d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1,
                 drop_path_rate: float = 0.1,
                 token_dropout: float = 0.15,
                 spatial_temp: float = 1.5,
                 max_regions: int = 200,
                 ema_momentum: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.token_dropout = token_dropout
        self.max_regions = max_regions
        self.feat_channels = feat_channels
        self.ema_momentum = ema_momentum

        # --- Dynamic token pooler: FPN P2 features pooled via .npz label_map ---
        self.token_pool = _DynamicRegionTokenPool(
            in_channels=72, token_dim=d_model)  # C2: 72ch, stride=4, 128x128

        # --- Learnable position embedding ---
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_regions, d_model) * 0.02)

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

        # --- EMA token buffer (updated in forward, NOT a learnable parameter) ---
        # Shape: (max_regions, d_model), registered as persistent buffer.
        # When ema_momentum > 0, region tokens are smoothed across iterations:
        #   ema_token = momentum * ema_token + (1-momentum) * current_token.detach()
        # This stabilizes contrastive learning when backbone features drift
        # during early training. The buffer stores tokens keyed by region ID
        # (requires consistent region ordering across epochs — guaranteed by
        # NAMLab static segmentation).
        self.register_buffer('_ema_tokens',
                             torch.zeros(max_regions, d_model),
                             persistent=True)
        self.register_buffer('_ema_initialized',
                             torch.zeros(1, dtype=torch.bool),
                             persistent=True)
        self._last_tb_sim = -1.0  # for debug monitoring

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

    def _process_single_sample(self, fpn_feat: torch.Tensor,
                                graph_data: dict) -> tuple:
        """Process one sample's region tokens through Transformer.
        Uses .npz handcrafted features — contrastive loss does NOT affect backbone.

        Args:
            graph_data: dict from LoadNAMGraph (all .npz fields)
        Returns:
            (node_out, label_map) or (None, None)
        """
        device = fpn_feat.device
        label_map = graph_data['label_map'].to(device)

        # === Pool tokens from FPN features via .npz label_map ===
        tokens, centroids, label_map_ds = self.token_pool(fpn_feat, label_map)

        N = tokens.size(0)
        if N <= 1:
            return None, None

        # === EMA smoothing on raw pooled tokens (before Transformer) ===
        # Stabilizes contrastive learning when backbone features drift during
        # early training. EMA buffer is keyed by region ID; NAMLab static
        # segmentation guarantees consistent region ordering across epochs.
        dropout_will_apply = (self.training and self.token_dropout > 0 and N > 8)
        if self.ema_momentum > 0 and self.training and not dropout_will_apply:
            with torch.no_grad():
                if self._ema_initialized.item():
                    self._ema_tokens[:N] = (
                        self.ema_momentum * self._ema_tokens[:N]
                        + (1 - self.ema_momentum) * tokens.detach())
                else:
                    self._ema_tokens[:N] = tokens.detach()
                    self._ema_initialized.fill_(True)
            # Blend: α * current + (1-α) * EMA. α=0.5 balances stability
            # (from EMA history) with plasticity (gradient to backbone).
            alpha = 0.5
            tokens = alpha * tokens + (1 - alpha) * self._ema_tokens[:N]

        # === Position embedding ===
        assert N <= self.max_regions, \
            f"Too many regions ({N}) for pos_embed ({self.max_regions}). Increase max_regions."
        x = tokens.unsqueeze(0) + self.pos_embed[:, :N, :]  # (1, N, d_model)

        # === Spatial distance bias ===
        spatial_bias = self.spatial_bias_fn(centroids)

        # === Token dropout (SPSE-ViT 2025) ===
        if self.training and self.token_dropout > 0 and N > 8:
            keep_prob = 1.0 - self.token_dropout
            keep_mask = torch.rand(N, device=device) < keep_prob
            if keep_mask.sum() < 4:
                keep_mask[:4] = True

            kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(-1)

            remap = torch.full((N,), -1, dtype=torch.long, device=device)
            for new_pos, old_idx in enumerate(kept_idx.tolist()):
                remap[old_idx] = new_pos

            dropped_idx = (~keep_mask).nonzero(as_tuple=False).squeeze(-1)
            if dropped_idx.numel() > 0:
                d2k = torch.cdist(centroids[dropped_idx],
                                  centroids[kept_idx])
                nearest = d2k.argmin(dim=1)
                for d_pos, old_idx in enumerate(dropped_idx.tolist()):
                    remap[old_idx] = nearest[d_pos].item()

            label_map_remapped = remap[label_map_ds.long()]
            x = x[:, kept_idx, :]
            centroids = centroids[kept_idx]
            spatial_bias = self.spatial_bias_fn(centroids)
        else:
            label_map_remapped = label_map_ds

        # === Transformer encoder ===
        for layer in self.encoder_layers:
            x = layer(x, spatial_bias)
        x = self.final_norm(x).squeeze(0)  # (N_tokens, d_model)

        return x, label_map_remapped

    def contrastive_loss(self, node_out, label_map, gt_bboxes,
                         temperature=0.07):
        """InfoNCE loss: push TB-region tokens together, away from nonTB.

        Args:
            node_out: (N, d_model) token embeddings after Transformer
            label_map: (H, W) region index per pixel
            gt_bboxes: list of (x,y,w,h) bboxes for this image
            temperature: softmax temperature (default 0.07)
        Returns:
            scalar loss, or 0 if <2 TB regions
        """
        if gt_bboxes is None or len(gt_bboxes) == 0:
            return node_out.new_zeros(())
        N, D = node_out.shape
        H, W = label_map.shape
        device = node_out.device

        # Mark TB regions
        tb_region_set = set()
        for bbox in gt_bboxes:
            x, y, bw, bh = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = min(int(x + bw), W), min(int(y + bh), H)
            if x2 > x1 and y2 > y1:
                patch = label_map[y1:y2, x1:x2]
                tb_region_set.update(int(v) for v in patch.unique())
        if len(tb_region_set) < 2:
            return node_out.new_zeros(())

        tb_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for r in tb_region_set:
            if 0 <= r < N:  # -1 from invalid regions must be excluded
                tb_mask[r] = True
        if tb_mask.sum() < 2:
            return node_out.new_zeros(())

        z = self.contrast_head(node_out)
        tokens = F.normalize(z, dim=-1)
        tb_tokens = tokens[tb_mask]
        non_tokens = tokens[~tb_mask]

        pos_sim = (tb_tokens @ tb_tokens.T) / temperature
        neg_sim = (tb_tokens @ non_tokens.T) / temperature
        pos_sim = pos_sim - torch.eye(pos_sim.size(0), device=device) * 1e9

        numerator = pos_sim.exp().sum(dim=1)
        denominator = numerator + neg_sim.exp().sum(dim=1)
        loss = -torch.log(numerator / denominator.clamp(min=1e-8)).mean()
        # Store TB token similarity for debug monitoring
        with torch.no_grad():
            tb_sim_matrix = tb_tokens @ tb_tokens.T
            mask = ~torch.eye(tb_sim_matrix.size(0), dtype=torch.bool, device=device)
            if mask.sum() > 0:
                self._last_tb_sim = tb_sim_matrix[mask].mean().item()
        return loss

    def forward(self, fpn_p2_list: list,
                nam_graphs: list,
                gt_bboxes: list = None) -> torch.Tensor:
        """Batch forward over samples, computing contrastive loss.
        Uses .npz handcrafted features — NO backbone gradient path.

        Args:
            nam_graphs: list[dict] per-sample graph data (length = B)
            gt_bboxes: list[list] per-sample bbox annotations
        Returns:
            scalar contrastive loss averaged over batch
        """
        B = len(nam_graphs) if nam_graphs is not None else 0
        device = self.final_norm.weight.device
        node_outs = []
        label_maps = []
        for b in range(B):
            if nam_graphs[b] is None or fpn_p2_list[b] is None:
                node_outs.append(None)
                label_maps.append(None)
            else:
                g = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in nam_graphs[b].items()}
                fpn_b = fpn_p2_list[b].to(device)
                node_out, label_map = self._process_single_sample(fpn_b, g)
                node_outs.append(node_out)
                label_maps.append(label_map)

        contr_loss = torch.zeros((), device=device)
        if self.training and gt_bboxes is not None:
            n_valid = 0
            for b in range(B):
                if node_outs[b] is not None and label_maps[b] is not None:
                    bboxes = gt_bboxes[b] if b < len(gt_bboxes) else None
                    cl = self.contrastive_loss(
                        node_outs[b], label_maps[b], bboxes)
                    if cl > 0:
                        contr_loss = contr_loss + cl
                        n_valid += 1
            if n_valid > 0:
                contr_loss = contr_loss / n_valid

        return contr_loss


# ============================================================================
# RetinaNAMGuideAttHeadV3_3: Dual-branch Head (SymFormer + Region Transformer v3.3)
# ============================================================================
if __name__ == 'mmdet.models.utils.NAMLab_att_v3_3':

    import sys as _sys

    # ---- RetinaGuideAttHead ----
    _retina_guide = _sys.modules.get(
        'mmdet.models.dense_heads.retina_guide_att_head')
    if _retina_guide is not None:
        RetinaGuideAttHead = _retina_guide.RetinaGuideAttHead
    else:
        try:
            from ..dense_heads.retina_guide_att_head import RetinaGuideAttHead
        except ImportError:
            from mmdet.models.dense_heads.retina_guide_att_head import \
                RetinaGuideAttHead

    # ---- HEADS ----
    _heads_mod = _sys.modules.get('mmdet.models.builder')
    if _heads_mod is not None and hasattr(_heads_mod, 'HEADS'):
        HEADS = _heads_mod.HEADS
    else:
        try:
            from ..builder import HEADS
        except ImportError:
            from mmdet.models.builder import HEADS

    @HEADS.register_module(force=True)
    class RetinaNAMGuideAttHeadV3_3(RetinaGuideAttHead):
        """Dual-branch RetinaNet Head v3.3:
        SymFormer + Dynamic Region Transformer + Cross-Scale Attention + Edge Supervision.

        Key improvements over v3:
          1. Region tokens pooled from FPN P2 (gradient -> backbone)
          2. Cross-scale attention: P5 semantics enhances P2 spatial details
          3. Edge supervision: encourages backbone to preserve boundaries

        Config:
            bbox_head=dict(
                type='RetinaNAMGuideAttHeadV3_3',
                d_model=256, nhead=8, num_layers=4, dropout=0.1,
                cross_scale=True, edge_supervision=True,
                num_classes=2, num_query=500, ...
            )
        """

        def __init__(self,
                     num_classes,
                     in_channels,
                     d_model: int = 256,
                     nhead: int = 8,
                     num_layers: int = 4,
                     dropout: float = 0.1,
                     drop_path_rate: float = 0.1,
                     token_dropout: float = 0.15,
                     spatial_temp: float = 1.5,
                     max_regions: int = 200,
                     ema_momentum: float = 0.0,
                     debug_interval: int = 50,
                     cross_scale: bool = True,
                     edge_supervision: bool = True,
                     edge_loss_weight: float = 0.1,
                     pc_branch: bool = False,
                     feat_channels: int = 256,
                     embed_dims: int = 256,
                     dims_radio: float = 2.0,
                     num_query: int = 100,
                     num_reg_fcs: int = 2,
                     transformer=None,
                     positional_encoding=None,
                     stacked_convs: int = 4,
                     conv_cfg=None,
                     norm_cfg=None,
                     anchor_generator=None,
                     bbox_coder=None,
                     reg_decoded_bbox: bool = False,
                     loss_cls=None,
                     loss_bbox=None,
                     train_cfg=None,
                     test_cfg=None,
                     init_cfg=None,
                     **kwargs):
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.v3_dropout = dropout
            self.v3_drop_path_rate = drop_path_rate
            self.v3_token_dropout = token_dropout
            self.v3_spatial_temp = spatial_temp
            self.v3_max_regions = max_regions
            self.v3_ema_momentum = ema_momentum
            self.debug_interval = debug_interval
            self._debug_step = 1
            self.use_cross_scale = cross_scale
            self.use_edge_sup = edge_supervision
            self.edge_loss_weight = edge_loss_weight
            self.use_pc_branch = pc_branch
            super().__init__(
                num_classes=num_classes,
                in_channels=in_channels,
                feat_channels=feat_channels,
                embed_dims=embed_dims,
                dims_radio=dims_radio,
                num_query=num_query,
                num_reg_fcs=num_reg_fcs,
                transformer=transformer,
                positional_encoding=positional_encoding,
                stacked_convs=stacked_convs,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                anchor_generator=anchor_generator,
                bbox_coder=bbox_coder,
                reg_decoded_bbox=reg_decoded_bbox,
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                init_cfg=init_cfg,
                **kwargs)

        def _init_layers(self):
            super()._init_layers()

            # --- Region Transformer branch (dynamic token pooling from FPN) ---
            self.nam_branch = _RegionTransformerBranchV3_3(
                feat_channels=self.feat_channels,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dropout=self.v3_dropout,
                drop_path_rate=self.v3_drop_path_rate,
                token_dropout=self.v3_token_dropout,
                spatial_temp=self.v3_spatial_temp,
                max_regions=self.v3_max_regions,
                ema_momentum=self.v3_ema_momentum)

            # --- Per-pixel LBP+Lab branch (retained from v3) ---
            self.lbp_branch = _PerPixelLBPBranch(
                in_channels=1062, feat_channels=self.feat_channels)

            # --- Boundary-responsive gate (Xavier init, NOT zero-init) ---
            self.br_gate = _BoundaryResponsiveGate(
                channels=self.feat_channels)

            # --- Lateral 1×1 convs: unify backbone channels → 256 (replaces FPN) ---
            # C2(72)→256, C3(144)→256, C4(288)→256, C5(576)→256
            if self.use_cross_scale:
                self.lateral_convs = nn.ModuleList([
                    nn.Conv2d(72, self.feat_channels, 1),
                    nn.Conv2d(144, self.feat_channels, 1),
                    nn.Conv2d(288, self.feat_channels, 1),
                    nn.Conv2d(576, self.feat_channels, 1),
                ])

            # --- TB-FusionNet Cross-Scale (SCA+CCA+ADD, P5→P4→P3→P2) ---
            if self.use_cross_scale:
                self.cross_scale_enhance = _TBFusionCrossScale(
                    channels=self.feat_channels)

            # --- Edge supervision head (NEW) ---
            if self.use_edge_sup:
                self.edge_head = _EdgeSupervisionHead(
                    in_channels=self.feat_channels, hidden_channels=64)

            # --- Gabor filters for PC+LBP computation (always needed for BRDG gate) ---
            self.pc_gabor = nn.ModuleList([
                nn.Conv2d(1, 1, 7, padding=3, bias=False)
                for _ in range(6)
            ])
            self._init_pc_gabor_weights()

            # --- PC Branch: inject PC+LBP into C2 (before lateral conv) ---
            if self.use_pc_branch:
                # Project PC+LBP(1062ch) → C2(72ch) + gate
                self.pc_proj = nn.Sequential(
                    nn.Conv2d(1062, 256, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 72, 1),   # zero-init
                )
                nn.init.zeros_(self.pc_proj[-1].weight)
                nn.init.zeros_(self.pc_proj[-1].bias)
                self.pc_gate = nn.Sequential(
                    nn.Conv2d(144, 72, 1),  # cat(C2_72, PC_72)=144
                    nn.Sigmoid(),
                )
                nn.init.zeros_(self.pc_gate[0].weight)
                nn.init.zeros_(self.pc_gate[0].bias)

            # Temperature hook: amplify gradient for spatial bias learning.
            self.nam_branch.spatial_bias_fn.temperature.register_hook(
                lambda g: g * 20.)

        def _init_pc_gabor_weights(self):
            """6-direction Gabor, same as _PhaseAwareEdgeDetector."""
            k = 7; half = 3
            y, x = torch.meshgrid(
                torch.arange(-half, half+1, dtype=torch.float32),
                torch.arange(-half, half+1, dtype=torch.float32), indexing='ij')
            for i, conv in enumerate(self.pc_gabor):
                theta = i * math.pi / 6
                xr = x * math.cos(theta) + y * math.sin(theta)
                yr = -x * math.sin(theta) + y * math.cos(theta)
                sigma_x, sigma_y = k/4.0, k/8.0
                freq = 1.0 / (k/3.0)
                kernel = torch.exp(-0.5*((xr/sigma_x)**2+(yr/sigma_y)**2))
                kernel = kernel * torch.cos(2*math.pi*freq*xr)
                kernel = kernel - kernel.mean()
                kernel = kernel / (kernel.std() + 1e-8)
                with torch.no_grad():
                    conv.weight.copy_(kernel.view(1, 1, k, k))

        def compute_pc_lbp(self, img):
            """BPPC pipeline: gray→Gabor×6→per-direction 3-scale LBP→1062ch.

            Returns (B, 1062, 512, 512) = 6 directions × 3 scales × 59 uniform bins.
            Replaces compute_lbp_gray. PC provides illumination invariance;
            LBP on PC response maps extracts texture from structure.
            """
            B, C, H, W = img.shape
            device = img.device
            mean = torch.tensor([123.675,116.28,103.53], device=device).view(1,3,1,1)
            std  = torch.tensor([58.395,57.12,57.375], device=device).view(1,3,1,1)
            img_raw = img * std + mean
            gray = 0.299*img_raw[:,0:1] + 0.587*img_raw[:,1:2] + 0.114*img_raw[:,2:3]

            # 6-direction Gabor response
            pc_dirs = []
            for conv in self.pc_gabor:
                r = conv(gray)                    # (B,1,H,W)
                grad_r = torch.abs(r - F.avg_pool2d(r, 3, 1, 1))
                energy = torch.sqrt(r**2 + grad_r**2 + 1e-8)
                pc_dirs.append(energy.squeeze(1))  # (B,H,W)

            # Per-direction 3-scale LBP → 6×3×59 = 1062ch
            all_lbp = []
            offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
            for pc in pc_dirs:  # pc: (B,H,W)
                for scale_r in [1, 2, 3]:
                    pad_w = scale_r
                    pc_pad = F.pad(pc, (pad_w,pad_w,pad_w,pad_w), mode='reflect')
                    code = torch.zeros(B, H, W, device=device)
                    for i, (dy, dx) in enumerate(offsets):
                        ny, nx = pad_w + dy*scale_r, pad_w + dx*scale_r
                        code += (pc_pad[:, ny:ny+H, nx:nx+W] >= pc).float() * (2**i)
                    code_1h = _lbp_to_uniform_onehot(code.long(), device)
                    all_lbp.append(code_1h)  # (B,H,W,59)

            out = torch.cat(all_lbp, dim=-1)  # (B,H,W,1062)
            return out.permute(0, 3, 1, 2)   # (B,1062,H,W)

        def build_lateral_feats(self, backbone_feats, lbp_lab=None):
            """Lateral 1×1 (+PC+LBP→C2) → cross-scale → [F2,F3,F4,F5] (256ch, 4 levels)."""
            C2, C3, C4, C5 = backbone_feats

            # --- PC+LBP injection into C2 (before lateral, backbone untouched) ---
            if hasattr(self, 'use_pc_branch') and self.use_pc_branch and lbp_lab is not None:
                pc_lbp = F.adaptive_avg_pool2d(lbp_lab, C2.shape[2:])  # (B,1062,128,128)
                pc_feat = self.pc_proj(pc_lbp)                          # (B,72,128,128)
                gate = self.pc_gate(torch.cat([C2, pc_feat], dim=1))
                C2 = C2 + pc_feat * gate                                # zero-init→identity

            F2 = self.lateral_convs[0](C2)  # (B,256,128,128)
            F3 = self.lateral_convs[1](C3)  # (B,256,64,64)
            F4 = self.lateral_convs[2](C4)  # (B,256,32,32)
            F5 = self.lateral_convs[3](C5)  # (B,256,16,16)
            if self.use_cross_scale:
                F2, F3, F4, F5 = self.cross_scale_enhance([F2, F3, F4, F5])
            return [F2, F3, F4, F5]

        def _ensure_shared_feat_list(self, feats, bs):
            """Return per-sample P2 features as list[Tensor] (bs, C, H, W)."""
            if isinstance(feats, (list, tuple)):
                p2 = feats[0]  # (B, C, H, W)
            else:
                p2 = feats
            return [p2[b] for b in range(bs)]

        def forward_single(self, x, lbp_lab=None):
            """Single FPN level forward: SymFormer + LBP modulation.

            Args:
                x: (B, C, H, W) FPN features at one level
                lbp_lab: (B, 180, H_orig, W_orig) per-pixel grayscale LBP
            Returns:
                cls_score, bbox_pred
            """
            bs, _, h, w = x.shape
            masks = x.new_zeros((bs, h, w)).to(torch.bool)

            # Branch 1: SymFormer (unchanged from v3)
            pos_embed = self.positional_encoding(masks, x)
            feat1 = self.transformer(
                tuple([x]), [masks], self.query_embedding.weight, [pos_embed])

            # Branch 2: LBP modulation with BRDG gate
            if lbp_lab is not None:
                feat_nam = self.lbp_branch(lbp_lab, (h, w))
                fused, _ = self.br_gate(feat1, feat_nam, lbp_lab)
            else:
                fused = feat1

            # Gradient routing: cls_feat = clean feat1, reg_feat = NAM-modulated
            cls_feat = feat1
            reg_feat = fused
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
            return cls_score, bbox_pred, feat1, fused

        def _debug_log(self, x, feat1, fused, lbp_lab, gate_stats):
            """Log NAM effectiveness diagnostics."""
            if self.debug_interval <= 0 or self._debug_step % self.debug_interval != 0:
                return
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank != 0 or lbp_lab is None:
                return

            h, w = x.shape[2:]
            nam_delta = (fused - feat1).detach()
            delta_per_pixel = nam_delta.norm(dim=1).mean().item()
            rel_change = nam_delta.norm(dim=1) / (feat1.detach().norm(dim=1) + 1e-8)
            sig_change_frac = (rel_change > 0.01).float().mean().item()

            gate_mean = gate_stats.get('gate_mean', 0)
            gate_active = gate_stats.get('gate_active', 0)
            edge_mean = gate_stats.get('edge_mean', 0)

            print(f'[NAMv3.3] iter={self._debug_step} h={h} '
                  f'g_m={gate_mean:.3f} g_act={gate_active:.3f} '
                  f'e_m={edge_mean:.3f} '
                  f'|feat1|={feat1.detach().norm(dim=1).mean().item():.2f} '
                  f'|fused|={fused.detach().norm(dim=1).mean().item():.2f} '
                  f'ΔNAM={delta_per_pixel:.3f} '
                  f'sig={sig_change_frac:.3f}',
                  flush=True)

        def forward(self, feats, nam_graphs=None, gt_bboxes=None,
                    lbp_lab=None, c2_feat=None):
            """Multi-level forward with late fusion + cross-scale enhancement.

            Processing order (v3.3):
              0. LBP Late Fusion: inject LBP texture into P2 (residual gate)
              1. Cross-scale attention: P5 semantics queries P2 details
              2. Dynamic region token pooling: P2 → tokens → contrastive loss
              3. Edge supervision: auxiliary L1 loss on P2
              4. Per-level SymFormer + BRDG gate modulation

            Args:
                feats: list of FPN features [P2, P3, P4, P5, P6, P7]
                nam_graphs: list[dict] per-sample graph data
                gt_bboxes: list[list] per-sample bbox annotations
                lbp_lab: (B, 180, H_orig, W_orig) per-pixel grayscale LBP
            Returns:
                cls_scores, bbox_preds, contr_loss, edge_loss
            """
            bs = feats[0].shape[0]
            feats = list(feats)  # make mutable

            # === Step 0: Cross-Scale already applied in build_lateral_feats ===
            # (when cross_scale=True, detector calls build_lateral_feats which
            #  runs lateral 1×1 + cross_scale_enhance before forward())

            # === Step 2: Dynamic region token + contrastive loss ===
            contr_loss = torch.zeros((), device=feats[0].device)
            if nam_graphs is not None and any(g is not None for g in nam_graphs):
                # Token pooling from C3 (144ch, 64x64) via NAM label_map regions
                if c2_feat is not None:
                    c3_per_sample = [c2_feat[b] for b in range(bs)]
                else:
                    c3_per_sample = self._ensure_shared_feat_list(feats, bs)
                contr_loss = self.nam_branch(
                    c3_per_sample, nam_graphs, gt_bboxes)

            # === Step 3: Edge supervision (auxiliary, training only) ===
            edge_loss = torch.zeros((), device=feats[0].device)
            if self.use_edge_sup and self.training and lbp_lab is not None:
                edge_pred = self.edge_head(feats[0])
                edge_loss = self.edge_head.compute_edge_loss(
                    edge_pred, lbp_lab)

            # === Step 4: Per-level SymFormer + BRDG gate modulation ===
            cls_scores, bbox_preds = [], []
            feat1_p2, fused_p2 = None, None  # saved for debug
            for i, feat in enumerate(feats):
                cls_score, bbox_pred, f1, f_fused = self.forward_single(
                    feat, lbp_lab)
                cls_scores.append(cls_score)
                bbox_preds.append(bbox_pred)
                if i == 0:  # P2 level
                    feat1_p2, fused_p2 = f1, f_fused

            # === Step 5: Enhanced Debug Logging [NEW] ===
            if self.debug_interval > 0 and self._debug_step % self.debug_interval == 0:
                self._enhanced_debug_log(
                    feats, feat1_p2, fused_p2, lbp_lab, contr_loss, edge_loss)

            self._debug_step += 1
            return tuple(cls_scores), tuple(bbox_preds), contr_loss, edge_loss

        def _enhanced_debug_log(self, feats, feat1, fused, lbp_lab,
                                 contr_loss, edge_loss):
            """一次性定位所有模块有效性的增强诊断。"""
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank != 0:
                return

            # ---- A: NAM Modulation ----
            if feat1 is not None and fused is not None:
                delta = (fused - feat1).detach()
                delta_norm = delta.norm(dim=1).mean().item()
                delta_map = delta.norm(dim=1)  # (B, H, W)
                all_mod = delta_map.mean().item()
                B, H, W = delta_map.shape
                flat_d = delta_map.reshape(B, -1)
                top_k = max(1, int(H * W * 0.2))
                edge_mod = flat_d.topk(top_k, dim=1)[0].mean().item()
                conc = edge_mod / (all_mod + 1e-8)
                cos_sim = F.cosine_similarity(
                    feat1.flatten(1), fused.flatten(1), dim=1).mean().item()
            else:
                delta_norm = conc = cos_sim = 0
                delta_map = None

            # ---- B: FPN Feature Stats ----
            p2_norm = feats[0].detach().norm(dim=1).mean().item() if len(feats) > 0 else 0
            p2_dead = (feats[0].detach().std(dim=1) < 0.01).float().mean().item() if len(feats) > 0 else 0
            # Inter-scale similarity
            if len(feats) >= 4:
                p_feats = [F.adaptive_avg_pool2d(f.detach(), (16, 16)).flatten(1)
                           for f in feats[:4]]
                sims = []
                for i in range(3):
                    s = F.cosine_similarity(p_feats[i][:1], p_feats[i+1][:1], dim=1)
                    sims.append(f'{s.item():.2f}')
                sim_str = '/'.join(sims)
            else:
                sim_str = 'N/A'

            # ---- C: LBP Info ----
            if lbp_lab is not None:
                lbp_hist = lbp_lab[:, :177].detach().mean(dim=[2, 3])
                lbp_ent = -(lbp_hist * (lbp_hist + 1e-8).log()).sum(dim=1).mean().item()
                lbp_svar = lbp_lab.detach().std(dim=[2, 3]).mean().item()
            else:
                lbp_ent = lbp_svar = 0

            # ---- D: Contrastive Token Quality ----
            ch_w = 0
            tb_sim = -1  # -1 = not computed
            if hasattr(self, 'nam_branch'):
                try:
                    ch_w = sum(p.data.norm().item()
                              for p in self.nam_branch.contrast_head.parameters())
                except Exception:
                    pass
                # Compute TB token similarity from last batch (if contrastive loss was active)
                try:
                    if hasattr(self.nam_branch, '_last_tb_sim'):
                        tb_sim = self.nam_branch._last_tb_sim
                except Exception:
                    pass

            # ---- E: Loss Diagnostics ----
            cl = contr_loss.item()
            el = edge_loss.item()

            print(f'[DBG] iter={self._debug_step} '
                  f'Δ={delta_norm:.2f} conc={conc:.2f} cos={cos_sim:.3f} '
                  f'P2_n={p2_norm:.1f} P2_d={p2_dead:.2f} P_sim=[{sim_str}] '
                  f'LBP_e={lbp_ent:.2f} LBP_s={lbp_svar:.2f} '
                  f'Lc={cl:.4f} Le={el:.4f} ch_w={ch_w:.1f} tb_sim={tb_sim:.2f}',
                  flush=True)

            # Deep diagnose every 500 iters
            if self._debug_step % 500 == 0:
                self._deep_diagnose(feats, feat1, fused, delta_map)

        def _deep_diagnose(self, feats, feat1, fused, delta_map):
            """深度诊断：特征相似度矩阵 + 调制空间分布。"""
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank != 0:
                return

            # ---- FPN inter-level cosine similarity matrix ----
            if len(feats) >= 4:
                p_feats = [F.adaptive_avg_pool2d(f.detach(), (16, 16)).flatten(1)
                           for f in feats[:4]]
                sim_mat = torch.zeros(4, 4)
                for i in range(4):
                    for j in range(4):
                        sim_mat[i, j] = F.cosine_similarity(
                            p_feats[i][:1], p_feats[j][:1], dim=1).item()
                sim_rows = []
                for i in range(4):
                    sim_rows.append('  '.join(f'{sim_mat[i,j]:.2f}' for j in range(4)))
                print(f'[DEEP] iter={self._debug_step} FPN-cos:\n' +
                      '\n'.join(f'  P{i+2}: {r}' for i, r in enumerate(sim_rows)),
                      flush=True)

            # ---- Modulation concentration ----
            if delta_map is not None:
                B, H, W = delta_map.shape
                flat_d = delta_map.reshape(B, -1)
                top_k = max(1, int(H * W * 0.2))
                edge_mod = flat_d.topk(top_k, dim=1)[0].mean().item()
                all_mod = flat_d.mean().item()
                conc = edge_mod / (all_mod + 1e-8)
                print(f'[DEEP] iter={self._debug_step} mod_conc={conc:.2f} '
                      f'(>1.5=edges, ≈1.0=uniform)', flush=True)

        def simple_test(self, feats, img_metas, rescale=False, nam_graphs=None,
                        lbp_lab=None, c2_feat=None):
            """Override parent to pass nam_graphs and lbp_lab to forward."""
            outs = self.forward(feats, nam_graphs=nam_graphs, lbp_lab=lbp_lab, c2_feat=c2_feat)
            results_list = self.get_bboxes(
                outs[0], outs[1], img_metas, rescale=rescale)
            return results_list


    # ============================================================================
    # RetinaNAMAttV3_3: Detector
    # ============================================================================
    # ---- SingleStageDetector ----
    _single_stage = _sys.modules.get('mmdet.models.detectors.single_stage')
    if _single_stage is not None:
        SingleStageDetector = _single_stage.SingleStageDetector
    else:
        try:
            from ..detectors.single_stage import SingleStageDetector
        except ImportError:
            from mmdet.models.detectors.single_stage import SingleStageDetector

    # ---- DETECTORS ----
    _det_mod = _sys.modules.get('mmdet.models.builder')
    if _det_mod is not None and hasattr(_det_mod, 'DETECTORS'):
        DETECTORS = _det_mod.DETECTORS
    else:
        try:
            from ..builder import DETECTORS
        except ImportError:
            from mmdet.models.builder import DETECTORS


    @DETECTORS.register_module(force=True)
    class RetinaNAMAttV3_3(SingleStageDetector):
        """Single-stage detector with Dynamic Region Transformer NAM branch.

        v3.3 improvements over v3:
          - Dynamic region token pooling from FPN P2 (gradient to backbone)
          - Cross-scale attention: P5 semantics queries P2 spatial details
          - Edge supervision: auxiliary L1 loss encourages boundary preservation

        Config:
            model=dict(
                type='RetinaNAMAttV3_3',
                backbone=dict(...),
                neck=dict(...),
                bbox_head=dict(type='RetinaNAMGuideAttHeadV3_3', ...),
            )
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._nam_epoch = 0

        def forward_train(self,
                          img,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          **kwargs):
            backbone_feats = self.backbone(img)
            c2_feat = backbone_feats[0]  # C2: (B, 72, 128, 128)

            # --- Build nam_graphs from img_metas ---
            nam_graphs = []
            for meta in img_metas:
                if isinstance(meta, dict) and 'nam_graph' in meta:
                    g = meta['nam_graph']
                    if meta.get('flip', False) and 'label_map' in g:
                        g = {k: v for k, v in g.items()}
                        lm = g['label_map']
                        if hasattr(lm, 'flip'):
                            g['label_map'] = lm.flip(dims=[1])
                        else:
                            g['label_map'] = lm[:, ::-1].copy()
                    nam_graphs.append(g)
                else:
                    nam_graphs.append(None)
            if all(g is None for g in nam_graphs):
                nam_graphs = None

            # --- PC+LBP (before lateral feats → used by C2 injection + BRDG gate) ---
            lbp_lab = self.bbox_head.compute_pc_lbp(img) if nam_graphs is not None else None

            if self.bbox_head.use_cross_scale:
                x = self.bbox_head.build_lateral_feats(backbone_feats, lbp_lab)
            elif self.with_neck:
                x = self.neck(backbone_feats)
            else:
                x = backbone_feats

            # --- Head forward ---
            cls_scores, bbox_preds, contr_loss, edge_loss = self.bbox_head(
                feats=x, nam_graphs=nam_graphs, gt_bboxes=gt_bboxes,
                lbp_lab=lbp_lab, c2_feat=c2_feat)

            # --- Detection losses ---
            losses = self.bbox_head.loss(
                cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)

            # --- Contrastive loss (fixed weight — gradients now reach backbone) ---
            # v3 used decaying weight (0.3→0.01) because gradients couldn't reach
            # the backbone anyway.  v3.3 fixes the gradient path, so a stable
            # weight prevents the loss from vanishing below the noise floor.
            # Adaptive contrastive weight: scales inversely with detection loss
            # Early training (high det_loss): weight ~0.05-0.1
            # Late training (low det_loss): weight ~0.005-0.02
            det_total = 0.0
            for k, v in losses.items():
                if k.startswith('loss_cls') or k.startswith('loss_bbox'):
                    if isinstance(v, (list, tuple)):
                        det_total += sum(x.item() for x in v)
                    else:
                        det_total += v.item()
            # High det_loss (early training) → higher contrast weight (backbone needs guidance)
            # Low det_loss (late training) → lower weight (backbone stable, marginal value)
            adapt_w = 0.02 + 0.08 * min(1.0, det_total / 5.0)
            losses['loss_contrast'] = adapt_w * contr_loss

            # --- Edge loss (constant weight, training-only auxiliary) ---
            if self.bbox_head.use_edge_sup:
                losses['loss_edge'] = (self.bbox_head.edge_loss_weight
                                       * edge_loss)

            return losses

        def simple_test(self, img, img_metas, rescale=False, **kwargs):
            backbone_feats = self.backbone(img)
            c2_feat = backbone_feats[0]  # C2: (B, 72, 128, 128)
            nam_graphs = []
            for meta in img_metas:
                if isinstance(meta, dict) and 'nam_graph' in meta:
                    nam_graphs.append(meta['nam_graph'])
                else:
                    nam_graphs.append(None)
            if all(g is None for g in nam_graphs):
                nam_graphs = None

            lbp_lab = self.bbox_head.compute_pc_lbp(img) if nam_graphs is not None else None

            if self.bbox_head.use_cross_scale:
                x = self.bbox_head.build_lateral_feats(backbone_feats, lbp_lab)
            elif self.with_neck:
                x = self.neck(backbone_feats)
            else:
                x = backbone_feats

            from mmdet.core import bbox2result
            results_list = self.bbox_head.simple_test(
                x, img_metas, rescale=rescale, nam_graphs=nam_graphs,
                lbp_lab=lbp_lab, c2_feat=c2_feat)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results


    # ============================================================================
    # NAMDebugHook (reused from v3)
    # ============================================================================
    from mmcv.runner import HOOKS, Hook


    @HOOKS.register_module(force=True)
    class NAMDebugHookV3_3(Hook):
        """Log NAM branch stats every N epochs.

        Add to config:
            custom_hooks = [
                dict(type='NAMDebugHookV3_3', interval=2),
            ]
        """

        def __init__(self, interval: int = 2):
            self.interval = interval

        def after_epoch(self, runner):
            try:
                model = runner.model.module if hasattr(runner.model, 'module') else runner.model
                if hasattr(model, '_nam_epoch'):
                    model._nam_epoch = runner.epoch + 1
            except Exception:
                pass

            if (runner.epoch + 1) % self.interval != 0:
                return
            try:
                model = runner.model.module if hasattr(runner.model, 'module') else runner.model
                branch = model.bbox_head.nam_branch
                br_gate = model.bbox_head.br_gate
            except AttributeError:
                return
            temp_raw = branch.spatial_bias_fn.temperature.item()
            temp_eff = F.softplus(torch.tensor(temp_raw)).item() + 0.01
            runner.logger.info(
                f'[NAMv3.3] Epoch {runner.epoch + 1:3d} | '
                f'temp={temp_eff:.3f}(raw={temp_raw:.2f})')
