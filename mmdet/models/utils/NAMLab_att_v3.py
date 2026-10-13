"""
NAMLab_att P0U0 stable recovery

Default deployment path:
  TransNeXt C2-C5 -> TBFusionCrossScaleNeck
  -> Sym head (original left/right prior only)
  -> Retina classification and GIoU regression.

Region contrast, C5/pi/4 semantic contrast, BoxAlign, anchor contrast, edge
supervision, PC injection, upper-third position priors, lung-mask/ROI
preprocessing and head-level PC are inactive in the recovery config. Their
implementations remain only for reading old experiment configs; the detector
bypasses all NAM-graph work when the auxiliary weights are zero.

v3 -> v3.3 core changes:
  1. Dynamic Region Token (P0 fix): tokens are pooled from backbone C2 features
     using label_map as pooling index, NOT from handcrafted .npz features.
     Contrastive loss gradients now flow: contr_loss -> tokens -> raw C2
     -> backbone. This is the CRITICAL fix — v3's contrastive loss could
     only optimize input_proj and Transformer params, never the backbone.

  2. Cross-Scale Neck (P2): detail-first P2->P3->P4->P5 fusion uses
     pixel-unshuffle downsampling, local cross-position attention, and true
     CxC channel cross-attention. A separately bounded P5->P2 semantic return
     path retains multi-scale detection context without overwriting raw P2.

  3. PC+LBP adapters remain available for controlled ablations, but the
     completed P/U 2x2 experiment found a negative PC main effect. Both the C2
     adapter and the unhelpful regression-head duplicate are disabled.

  4. The measured default keeps the original left/right symmetry only. The
     image/lung upper-third variants remain controlled alternatives because
     their small observed difference was within run noise.

  5. The main path uses no CDF ROI crop, histogram standardization or CLAHE.
     Raw 512x512 CXR geometry and TransNeXt/ImageNet normalization are kept.

  6. NAM-BoxAlign contrast uses fractional area pooling, continuous bbox
      overlap targets, per-GT prototypes and one-hop region topology negatives.
      Detached C2/NAM prototypes are aligned with matched P2-P4 regression
      features, directly connecting region learning to bbox localization.

  7. The measured recovery stack keeps BoxAlign but disables the harmful C2
     region loss by default. Optional region contrast uses a strict pi/4
     region-purity threshold, ignores low-purity overlaps, and falls back to
     the best region so a small lesion never loses all positives.

  8. Region tokens can be sourced from C2, raw semantic C5, or fused P5 for
     controlled task-alignment ablations. C2 remains the BoxAlign baseline.

  9. Removed from v3: Handcrafted region token features (node_wld, node_lab,
     node_stat, node_glcm from .npz). .npz is now only used for label_map
     (pooling and topology); stored handcrafted node attributes are ignored.

  10. v3.18 controlled additions are disabled by default: FreqFusion-style
      ALPF/AHPF/local-similarity offset replaces only P5->P4 and P4->P3
      bilinear in its child config; a separate C5-gated C2 detail route can
      feed only the P4/P5 bbox regression towers. P3->P2 stays bilinear and
      the classification tower never sees the detail bypass.

Gradient paths:
  Det Loss       -> head -> CrossScale neck -> backbone
  Region Contrast-> disabled after negative ablation
  Region BoxAlign -> disabled after unstable/non-reproducible ablation
  BBox Contrast  -> disabled after negative ablation
  Raw-CXR PC     -> disabled after the P/U 2x2 ablation
  Upper Prior    -> disabled; original left/right Sym prior remains

Components:
    _DynamicRegionTokenPool    - Pool region tokens from raw C2 features [NEW]
    TBFusionCrossScaleNeck     - Four-level direct FPN replacement [NEW]
    _PCShallowAdapter          - Single bounded PC adapter at the C2 seam
    _RegionTransformerBranchV3_3 - Updated: pools from C2, not .npz [MODIFIED]
    [Reused from v3]: _SpatialBiasEncoderLayer, _SpatialDistanceBias,
                      _PerPixelLBPBranch, _BoundaryResponsiveGate,
                      compute_lbp_lab, _drop_path
    RetinaNAMGuideAttHeadV3  - Sym head; head-PC remains optional for ablation
    RetinaNAMAttV3           - Detector [MODIFIED]
    NAMDebugHook               - Reused from v3

Data format (.npz, reused from V2):
    label_map  : (H, W) int32       pixel -> region ID (used for pooling index)
    edge_index : (2, E) int32       retained in data, not consumed in v3.12
    edge_dc    : (E,) float32       retained in data, not consumed in v3.12
    (node_wld, node_lab, node_stat, node_glcm, node_area are IGNORED in v3.3)

Config example:
    model=dict(
        type='RetinaNAMAttV3',
        pc_branch=False,
        neck=dict(type='TBFusionCrossScaleNeck', ...),
        bbox_head=dict(
            type='RetinaNAMGuideAttHeadV3',
            head_pc_modulation=False,
            bbox_contrastive=False, ...))
"""
import hashlib
import json
import math
import os
import logging
import numpy as np
from mmdet.utils.deterministic_ops import deterministic_bilinear_interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

try:
    # FreqFusion uses CARAFE for efficient content-aware reassembly. Keep a
    # pure-Torch fallback for CPU tests and servers without compiled ops.
    from mmcv.ops import carafe as _mmcv_carafe
except (ImportError, OSError):
    _mmcv_carafe = None


NAMLAB_BUILD_ID = (
    'v3.18-freqfusion-reg-detail-ablation-20260730')


def _distribution_refinement_expectation(
        logits: torch.Tensor,
        num_anchors: int,
        reg_max: int) -> tuple:
    """Decode symmetric discrete residuals to Retina delta layout.

    D-FINE predicts and iteratively refines coordinate distributions in a
    DETR decoder.  This project keeps its Retina anchor interface, so the
    compatible unit is a zero-centred distribution over a *residual* to the
    original ``dx, dy, dw, dh`` prediction.  Zero logits have exactly zero
    expectation and therefore preserve the baseline at initialization.

    Returns:
        residual: ``(B, A*4, H, W)`` expected residual in ``[-1, 1]``.
        normalized_entropy: scalar in ``[0, 1]`` for debug diagnostics.
    """
    if reg_max < 2:
        raise ValueError('reg_max must be at least 2')
    if num_anchors < 1:
        raise ValueError('num_anchors must be positive')
    if logits.ndim != 4:
        raise ValueError(
            f'distribution logits must be BCHW, got {tuple(logits.shape)}')
    batch, channels, height, width = logits.shape
    bins = reg_max + 1
    expected_channels = num_anchors * 4 * bins
    if channels != expected_channels:
        raise ValueError(
            f'expected {expected_channels} distribution channels, '
            f'got {channels}')

    shaped = logits.permute(0, 2, 3, 1).reshape(
        batch, height, width, num_anchors, 4, bins)
    probabilities = F.softmax(shaped.float(), dim=-1)
    project = torch.linspace(
        -1.0, 1.0, bins,
        device=probabilities.device,
        dtype=probabilities.dtype)
    residual = (probabilities * project).sum(dim=-1)
    residual = residual.permute(0, 3, 4, 1, 2).reshape(
        batch, num_anchors * 4, height, width)
    entropy = -(
        probabilities.clamp_min(1e-8)
        * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    normalized_entropy = entropy.mean() / math.log(float(bins))
    return residual.to(dtype=logits.dtype), normalized_entropy


def _symmetric_distribution_focal_loss(
        logits: torch.Tensor,
        normalized_targets: torch.Tensor,
        coordinate_weights: torch.Tensor,
        reg_max: int,
        avg_factor: float) -> torch.Tensor:
    """Linear-interpolation DFL on a symmetric ``[-1, 1]`` residual grid."""
    bins = int(reg_max) + 1
    if logits.ndim != 3 or logits.shape[-2:] != (4, bins):
        raise ValueError(
            'logits must have shape (N, 4, reg_max + 1), got '
            f'{tuple(logits.shape)}')
    if normalized_targets.shape != logits.shape[:2]:
        raise ValueError(
            'normalized_targets must have shape (N, 4), got '
            f'{tuple(normalized_targets.shape)}')
    if coordinate_weights.shape != normalized_targets.shape:
        raise ValueError(
            'coordinate_weights must match normalized_targets')

    # Keep the right interpolation bin valid. Saturation is measured before
    # this clamp by the caller and exposed in [DFINE-prev].
    target_index = (
        (normalized_targets.clamp(-1.0, 1.0) + 1.0)
        * (0.5 * float(reg_max)))
    target_index = target_index.clamp(
        min=0.0, max=float(reg_max) - 1e-6)
    left = target_index.floor().long()
    right = (left + 1).clamp(max=reg_max)
    right_weight = target_index - left.float()
    left_weight = 1.0 - right_weight

    flat_logits = logits.reshape(-1, bins)
    left_loss = F.cross_entropy(
        flat_logits, left.reshape(-1), reduction='none')
    right_loss = F.cross_entropy(
        flat_logits, right.reshape(-1), reduction='none')
    loss = (
        left_loss * left_weight.reshape(-1)
        + right_loss * right_weight.reshape(-1))
    loss = loss * coordinate_weights.reshape(-1)
    return loss.sum() / max(float(avg_factor), 1.0)


def _emit_training_debug(message: str,
                         level: int = logging.INFO) -> None:
    """Write debug text through MMDetection's console and file handlers.

    MMCV configures the named ``mmdet`` logger with both a stream handler and
    the run's ``.log`` file handler. Logging one record therefore keeps the
    terminal and training log synchronized. Each handler is flushed
    immediately; lightweight standalone tests fall back to plain stdout.
    """
    logger = logging.getLogger('mmdet')
    if logger.handlers and logger.isEnabledFor(level):
        logger.log(level, message)
        for handler in logger.handlers:
            try:
                handler.flush()
            except (AttributeError, OSError, ValueError):
                pass
        return
    print(message, flush=True)


def _pairwise_bbox_iou(boxes1: torch.Tensor,
                       boxes2: torch.Tensor,
                       eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes without an MMDetection dependency."""
    if boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(
            f'boxes1 must have shape Nx4, got {tuple(boxes1.shape)}')
    if boxes2.ndim != 2 or boxes2.shape[-1] != 4:
        raise ValueError(
            f'boxes2 must have shape Mx4, got {tuple(boxes2.shape)}')
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(
        boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0))
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0))
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(eps)


def _contrast_schedule_factor(epoch: int,
                              start_epoch: int = 2,
                              ramp_epochs: int = 6,
                              hold_end_epoch: int = 16,
                              decay_end_epoch: int = 22) -> float:
    """Detector-first auxiliary-loss schedule for a 1-indexed epoch.

    The auxiliary objective is disabled through ``start_epoch``, ramps to one,
    stays active through ``hold_end_epoch``, then decays to zero so the final
    epochs fit classification and GIoU without a competing representation
    objective.
    """
    epoch = int(epoch)
    start_epoch = int(start_epoch)
    ramp_epochs = int(ramp_epochs)
    hold_end_epoch = int(hold_end_epoch)
    decay_end_epoch = int(decay_end_epoch)
    if start_epoch < 0 or ramp_epochs < 0:
        raise ValueError('start_epoch and ramp_epochs must be non-negative')
    ramp_end = start_epoch + ramp_epochs
    if hold_end_epoch < ramp_end:
        raise ValueError('hold_end_epoch must be >= start_epoch + ramp_epochs')
    if decay_end_epoch <= hold_end_epoch:
        raise ValueError('decay_end_epoch must be > hold_end_epoch')
    if epoch <= start_epoch:
        return 0.0
    if ramp_epochs > 0 and epoch < ramp_end:
        return float(epoch - start_epoch) / float(ramp_epochs)
    if epoch <= hold_end_epoch:
        return 1.0
    if epoch < decay_end_epoch:
        return float(decay_end_epoch - epoch) / float(
            decay_end_epoch - hold_end_epoch)
    return 0.0


def _select_high_purity_region_targets(
        per_box_purity: torch.Tensor,
        valid_token_mask: torch.Tensor,
        minimum_positive_purity: float = math.pi / 4,
        ensure_gt_coverage: bool = True) -> dict:
    """Select clean per-GT region targets without false-negative boundaries.

    ``per_box_purity[g, r]`` is ``area(GT_g ∩ region_r) / area(region_r)``.
    A region is a positive for GT ``g`` only when that fraction reaches the
    configured threshold. Regions that touch any GT but fail the threshold are
    ignored, not used as background negatives. If a small lesion has no region
    above threshold, its highest-purity touching region is retained as a
    coverage fallback.
    """
    if per_box_purity.ndim != 2:
        raise ValueError(
            'per_box_purity must have shape (num_gt, num_regions)')
    if valid_token_mask.ndim != 1:
        raise ValueError('valid_token_mask must be one-dimensional')
    if per_box_purity.shape[1] != valid_token_mask.numel():
        raise ValueError(
            'per_box_purity region dimension must match valid_token_mask')
    threshold = float(minimum_positive_purity)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            'minimum_positive_purity must be in (0, 1]')

    valid = valid_token_mask.bool()
    touching = (per_box_purity > 0) & valid.unsqueeze(0)
    per_gt_positive = (
        (per_box_purity >= threshold) & touching)
    fallback_gt_mask = torch.zeros(
        per_box_purity.shape[0],
        dtype=torch.bool,
        device=per_box_purity.device)

    if ensure_gt_coverage and per_box_purity.shape[0] > 0:
        needs_fallback = ~per_gt_positive.any(dim=1) & touching.any(dim=1)
        fallback_gt_indices = needs_fallback.nonzero(
            as_tuple=False).squeeze(-1)
        if fallback_gt_indices.numel() > 0:
            fallback_region_indices = per_box_purity[
                fallback_gt_indices].argmax(dim=1)
            per_gt_positive[
                fallback_gt_indices, fallback_region_indices] = True
            fallback_gt_mask[fallback_gt_indices] = True

    positive_mask = per_gt_positive.any(dim=0) & valid
    touching_mask = touching.any(dim=0) & valid
    ambiguous_overlap_mask = touching_mask & ~positive_mask
    negative_mask = valid & ~touching_mask
    return {
        'per_gt_positive_mask': per_gt_positive,
        'positive_mask': positive_mask,
        'ambiguous_overlap_mask': ambiguous_overlap_mask,
        'negative_mask': negative_mask,
        'fallback_gt_mask': fallback_gt_mask,
        'touching_mask': touching_mask,
    }


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


def _weighted_instance_contrastive_loss(
        positive_tokens: torch.Tensor,
        instance_ids: torch.Tensor,
        positive_weights: torch.Tensor,
        negative_tokens: torch.Tensor,
        temperature: float) -> tuple:
    """IoU/purity-weighted supervised contrast for object instances.

    Only tokens assigned to the same GT instance form a positive pair.
    Positive tokens from other lesions and the explicit background tokens are
    negatives. This avoids collapsing all Active/Obsolete TB appearances into
    two class prototypes when the optimization target is bbox localization.

    Returns:
        loss, mean_positive_similarity, valid_anchor_count
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if positive_tokens.ndim != 2 or negative_tokens.ndim != 2:
        raise ValueError('positive_tokens and negative_tokens must be 2-D')
    if positive_tokens.shape[1] != negative_tokens.shape[1]:
        raise ValueError('positive and negative embedding dims must match')
    num_positive = positive_tokens.shape[0]
    if (num_positive == 0 or negative_tokens.shape[0] == 0
            or instance_ids.numel() != num_positive
            or positive_weights.numel() != num_positive):
        return positive_tokens.sum() * 0.0, 0.0, 0

    instance_ids = instance_ids.long().flatten()
    positive_weights = positive_weights.flatten().clamp_min(1e-4)
    candidates = torch.cat((positive_tokens, negative_tokens), dim=0)
    similarity = positive_tokens @ candidates.T / float(temperature)
    positive_pair_mask = (
        instance_ids[:, None] == instance_ids[None, :])
    positive_pair_mask.fill_diagonal_(False)
    self_indices = torch.arange(
        num_positive, device=positive_tokens.device)
    similarity[self_indices, self_indices] = float('-inf')
    valid_anchor = positive_pair_mask.any(dim=1)
    valid_count = int(valid_anchor.sum().item())
    if valid_count == 0:
        return positive_tokens.sum() * 0.0, 0.0, 0

    log_denominator = torch.logsumexp(
        similarity, dim=1, keepdim=True)
    positive_log_prob = (
        similarity[:, :num_positive] - log_denominator)
    safe_log_prob = torch.where(
        positive_pair_mask,
        positive_log_prob,
        torch.zeros_like(positive_log_prob))
    pair_weights = (
        positive_pair_mask.float() * positive_weights.unsqueeze(0))
    loss_per_anchor = -(
        (safe_log_prob * pair_weights).sum(dim=1)
        / pair_weights.sum(dim=1).clamp_min(1e-8))
    anchor_weights = positive_weights[valid_anchor]
    loss = (
        loss_per_anchor[valid_anchor] * anchor_weights
    ).sum() / anchor_weights.sum().clamp_min(1e-8)

    with torch.no_grad():
        raw_similarity = positive_tokens @ positive_tokens.T
        mean_positive_similarity = float(
            raw_similarity[positive_pair_mask].mean().item())
    return loss, mean_positive_similarity, valid_count


def _resize_binary_mask(mask, image_shape: tuple) -> torch.Tensor:
    """Return a source mask aligned to ``image_shape`` as a bool tensor."""
    image_h, image_w = (int(image_shape[0]), int(image_shape[1]))
    mask = torch.as_tensor(mask)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError(
                f'lung mask must be single-channel, got {tuple(mask.shape)}')
    if mask.ndim != 2:
        raise ValueError(
            f'lung mask must be 2-D, got {tuple(mask.shape)}')
    if tuple(mask.shape) != (image_h, image_w):
        mask = F.interpolate(
            mask[None, None].float(),
            size=(image_h, image_w),
            mode='nearest')[0, 0]
    return mask > 0


def _cdf_axis_interval(mass: torch.Tensor,
                       low_percentile: float,
                       high_percentile: float,
                       min_extent: int = 2) -> tuple:
    """Find an axis interval from the CDF of non-negative projection mass."""
    mass = torch.as_tensor(mass, dtype=torch.float64).flatten()
    if mass.numel() == 0:
        raise ValueError('CDF projection mass must not be empty')
    if not (0.0 <= low_percentile < high_percentile <= 1.0):
        raise ValueError(
            'CDF percentiles must satisfy 0 <= low < high <= 1')
    mass = torch.where(
        torch.isfinite(mass), mass, torch.zeros_like(mass)).clamp_min(0)
    total = mass.sum()
    if float(total.item()) <= 1e-12:
        return 0, int(mass.numel())

    cdf = mass.cumsum(0)
    start = int(torch.searchsorted(
        cdf, total * float(low_percentile), right=False).item())
    end = int(torch.searchsorted(
        cdf, total * float(high_percentile), right=False).item()) + 1
    start = max(0, min(start, int(mass.numel()) - 1))
    end = max(start + 1, min(end, int(mass.numel())))

    target_extent = min(max(int(min_extent), 1), int(mass.numel()))
    if end - start < target_extent:
        centre = 0.5 * (start + end)
        start = int(math.floor(centre - target_extent / 2))
        start = max(0, min(start, int(mass.numel()) - target_extent))
        end = start + target_extent
    return start, end


def _to_grayscale_cxr(image) -> torch.Tensor:
    """Convert an MMDetection HxWxBGR image to one grayscale intensity map."""
    image = torch.as_tensor(image)
    if image.ndim == 2:
        return image.float()
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f'CXR image must be HxW or HxWxC, got {tuple(image.shape)}')
    channels = image.float()
    if channels.shape[-1] == 1:
        return channels[..., 0]
    return (
        0.114 * channels[..., 0]
        + 0.587 * channels[..., 1]
        + 0.299 * channels[..., 2])


def _compute_cdf_roi_geometry(image,
                              x_percentiles=(0.05, 0.95),
                              y_percentiles=(0.15, 0.95),
                              min_extent: int = 2) -> dict:
    """Compute the paper's CDF-guided CXR crop.

    Projection masses are sums of grayscale intensity along each image axis.
    The retained horizontal CDF interval is 5-95% and the vertical interval
    is 15-95% by default, matching Section 3.2.3 of Singthongchai and
    Wangkhamhan (Journal of Imaging, 2026).
    """
    gray = _to_grayscale_cxr(image)

    image_h, image_w = (int(gray.shape[0]), int(gray.shape[1]))
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f'invalid image shape {(image_h, image_w)}')
    gray = torch.where(
        torch.isfinite(gray), gray, torch.zeros_like(gray)).clamp_min(0)
    x1, x2 = _cdf_axis_interval(
        gray.sum(dim=0), float(x_percentiles[0]),
        float(x_percentiles[1]), min_extent=min_extent)
    y1, y2 = _cdf_axis_interval(
        gray.sum(dim=1), float(y_percentiles[0]),
        float(y_percentiles[1]), min_extent=min_extent)
    crop_h, crop_w = y2 - y1, x2 - x1
    return {
        'crop_bbox': (x1, y1, x2, y2),
        'pad': (0, 0, 0, 0),
        'output_shape': (crop_h, crop_w),
        'retained_area_fraction': float(
            crop_h * crop_w / max(image_h * image_w, 1)),
        'linear_scale_gain': float(
            max(image_h, image_w) / max(crop_h, crop_w, 1)),
    }


def _crop_2d(value, crop_bbox: tuple) -> torch.Tensor:
    """Apply the CDF ROI crop to a categorical or binary 2-D tensor."""
    tensor = torch.as_tensor(value)
    if tensor.ndim != 2:
        raise ValueError(
            f'ROI auxiliary map must be 2-D, got {tuple(tensor.shape)}')
    x1, y1, x2, y2 = (int(v) for v in crop_bbox)
    return tensor[y1:y2, x1:x2]


def _boxes_to_roi(boxes: torch.Tensor,
                  crop_bbox: tuple) -> torch.Tensor:
    """Translate and clip xyxy boxes into the cropped ROI coordinate frame."""
    boxes = torch.as_tensor(boxes)
    if boxes.numel() == 0:
        return boxes.clone()
    if boxes.ndim != 2 or boxes.shape[-1] < 4:
        raise ValueError(
            f'boxes must have shape Nx4(+), got {tuple(boxes.shape)}')
    x1, y1, x2, y2 = (float(v) for v in crop_bbox)
    cropped = boxes.clone()
    cropped[:, 0] = (cropped[:, 0] - x1).clamp(0, x2 - x1)
    cropped[:, 2] = (cropped[:, 2] - x1).clamp(0, x2 - x1)
    cropped[:, 1] = (cropped[:, 1] - y1).clamp(0, y2 - y1)
    cropped[:, 3] = (cropped[:, 3] - y1).clamp(0, y2 - y1)
    return cropped


def _standardize_cxr_intensity(image,
                               target_mean: float = 121.8,
                               target_std: float = 57.1,
                               eps: float = 1e-6) -> tuple:
    """Apply the paper's global histogram mean/std standardization."""
    if target_std <= 0:
        raise ValueError('target_std must be positive')
    tensor = torch.as_tensor(image).float()
    if tensor.numel() == 0:
        raise ValueError('cannot standardize an empty CXR image')
    original_mean = tensor.mean()
    original_std = tensor.std(unbiased=False)
    safe_std = original_std.clamp_min(float(eps))
    normalized = (
        (tensor - original_mean) / safe_std
        * float(target_std) + float(target_mean))
    normalized = normalized.clamp(0.0, 255.0)
    stats = {
        'original_mean': float(original_mean.item()),
        'original_std': float(original_std.item()),
        'normalized_mean': float(normalized.mean().item()),
        'normalized_std': float(
            normalized.std(unbiased=False).item()),
        'degenerate_std': bool(float(original_std.item()) < float(eps)),
    }
    return normalized, stats


def _bbox_crop_coverage(boxes: torch.Tensor,
                        crop_bbox: tuple) -> torch.Tensor:
    """Fraction of every xyxy box retained by a proposed ROI crop."""
    boxes = torch.as_tensor(boxes)
    if boxes.numel() == 0:
        return boxes.new_ones((0,), dtype=torch.float32)
    crop = boxes.new_tensor(crop_bbox).reshape(1, 4)
    top_left = torch.maximum(boxes[:, :2], crop[:, :2])
    bottom_right = torch.minimum(boxes[:, 2:4], crop[:, 2:4])
    intersection_wh = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    box_wh = (boxes[:, 2:4] - boxes[:, :2]).clamp_min(0)
    area = box_wh[:, 0] * box_wh[:, 1]
    return intersection / area.clamp_min(1e-7)


def _restore_roi_bboxes(det_bboxes: torch.Tensor,
                        img_meta: dict) -> torch.Tensor:
    """Map rescaled detector boxes from ROI canvas to original CXR pixels."""
    if (det_bboxes.numel() == 0
            or not bool(img_meta.get('roi_applied', False))):
        return det_bboxes
    crop_bbox = img_meta.get('roi_crop_bbox')
    roi_pad = img_meta.get('roi_pad')
    original_shape = img_meta.get(
        'roi_original_shape', img_meta.get('ori_shape'))
    if crop_bbox is None or roi_pad is None or original_shape is None:
        raise KeyError(
            'ROI metadata is incomplete; expected roi_crop_bbox, roi_pad '
            'and roi_original_shape')

    restored = det_bboxes.clone()
    x1, y1, _, _ = (float(v) for v in crop_bbox)
    pad_left, pad_top, _, _ = (float(v) for v in roi_pad)
    restored[:, 0] = restored[:, 0] - pad_left + x1
    restored[:, 2] = restored[:, 2] - pad_left + x1
    restored[:, 1] = restored[:, 1] - pad_top + y1
    restored[:, 3] = restored[:, 3] - pad_top + y1
    original_h, original_w = (
        int(original_shape[0]), int(original_shape[1]))
    restored[:, 0::2] = restored[:, 0::2].clamp(0, original_w)
    restored[:, 1::2] = restored[:, 1::2].clamp(0, original_h)
    return restored


def _align_nam_label_map(label_map: torch.Tensor,
                         resized_shape: tuple,
                         pad_shape: tuple,
                         flip: bool = False,
                         flip_direction: str = 'horizontal',
                         invalid_label: int = -1) -> torch.Tensor:
    """Apply the detector image geometry to an original-resolution NAM map.

    MMDetection keeps the resized content in the top-left corner and pads the
    bottom/right edges.  NAM region IDs are categorical, so nearest-neighbour
    interpolation is required and padded pixels must stay invalid instead of
    being silently assigned to region 0.
    """
    label_map = torch.as_tensor(label_map).long()
    if label_map.ndim != 2:
        raise ValueError(
            f'NAM label_map must be 2-D, got shape {tuple(label_map.shape)}')

    resized_h, resized_w = (int(resized_shape[0]), int(resized_shape[1]))
    pad_h, pad_w = (int(pad_shape[0]), int(pad_shape[1]))
    if resized_h <= 0 or resized_w <= 0:
        raise ValueError(
            f'Invalid resized NAM shape {(resized_h, resized_w)}')
    if pad_h < resized_h or pad_w < resized_w:
        raise ValueError(
            f'NAM pad shape {(pad_h, pad_w)} is smaller than resized '
            f'shape {(resized_h, resized_w)}')

    if tuple(label_map.shape) != (resized_h, resized_w):
        label_map = F.interpolate(
            label_map[None, None].float(),
            size=(resized_h, resized_w),
            mode='nearest').squeeze(0).squeeze(0).long()

    if flip:
        direction = flip_direction or 'horizontal'
        if direction == 'horizontal':
            label_map = label_map.flip(1)
        elif direction == 'vertical':
            label_map = label_map.flip(0)
        elif direction == 'diagonal':
            label_map = label_map.flip((0, 1))
        else:
            raise ValueError(f'Unsupported NAM flip direction: {direction}')

    if (pad_h, pad_w) != (resized_h, resized_w):
        label_map = F.pad(
            label_map,
            (0, pad_w - resized_w, 0, pad_h - resized_h),
            mode='constant',
            value=invalid_label)
    return label_map


def _resolve_auxiliary_canvas(img_meta: dict,
                              batch_input_shape=None) -> tuple:
    """Resolve the real tensor canvas after per-image and batch padding."""
    recorded_pad = tuple(
        int(v) for v in img_meta.get(
            'pad_shape', img_meta['img_shape'])[:2])
    if batch_input_shape is None:
        return recorded_pad
    batch_shape = tuple(int(v) for v in batch_input_shape[:2])
    if (batch_shape[0] < recorded_pad[0]
            or batch_shape[1] < recorded_pad[1]):
        raise ValueError(
            f'batch canvas {batch_shape} is smaller than per-image '
            f'pad_shape {recorded_pad}')
    return batch_shape


def _align_nam_graph_to_image_meta(graph_data: dict,
                                   img_meta: dict,
                                   batch_input_shape=None) -> dict:
    """Return a NAM graph aligned to the actual collated image canvas."""
    if graph_data is None or 'label_map' not in graph_data:
        return graph_data

    img_shape = tuple(img_meta['img_shape'][:2])
    recorded_pad_shape = tuple(
        img_meta.get('pad_shape', img_meta['img_shape'])[:2])
    pad_shape = _resolve_auxiliary_canvas(
        img_meta, batch_input_shape=batch_input_shape)
    aligned = {key: value for key, value in graph_data.items()}
    aligned['label_map'] = _align_nam_label_map(
        graph_data['label_map'],
        resized_shape=img_shape,
        pad_shape=pad_shape,
        flip=bool(img_meta.get('flip', False)),
        flip_direction=img_meta.get('flip_direction', 'horizontal'))
    aligned['label_map_aligned'] = True
    aligned['original_label_shape'] = tuple(
        torch.as_tensor(graph_data['label_map']).shape)
    aligned['valid_shape'] = img_shape
    aligned['pad_shape'] = pad_shape
    aligned['recorded_pad_shape'] = recorded_pad_shape
    aligned['invalid_pixel_fraction'] = float(
        (aligned['label_map'] < 0).float().mean().item())
    return aligned


def _align_lung_mask_to_image_meta(lung_mask: torch.Tensor,
                                   img_meta: dict,
                                   batch_input_shape=None) -> torch.Tensor:
    """Apply detector Resize/Flip/Pad and collate geometry to an anatomy mask.

    The loader intentionally keeps the mask at source resolution in
    ``img_metas``. MMDetection 2.x records each sample's pipeline pad_shape,
    but DataContainer can add a second right/bottom pad while stacking samples
    of different ROI aspect ratios. ``batch_input_shape`` supplies that final
    tensor canvas so every auxiliary map follows the image exactly.
    """
    lung_mask = torch.as_tensor(lung_mask)
    if lung_mask.ndim == 3:
        if lung_mask.shape[-1] == 1:
            lung_mask = lung_mask[..., 0]
        elif lung_mask.shape[0] == 1:
            lung_mask = lung_mask[0]
        else:
            raise ValueError(
                f'lung_mask must be single-channel, got '
                f'{tuple(lung_mask.shape)}')
    if lung_mask.ndim != 2:
        raise ValueError(
            f'lung_mask must be 2-D, got shape {tuple(lung_mask.shape)}')

    resized_h, resized_w = tuple(int(v) for v in img_meta['img_shape'][:2])
    pad_h, pad_w = _resolve_auxiliary_canvas(
        img_meta, batch_input_shape=batch_input_shape)
    if resized_h <= 0 or resized_w <= 0:
        raise ValueError(
            f'Invalid resized lung-mask shape {(resized_h, resized_w)}')
    if pad_h < resized_h or pad_w < resized_w:
        raise ValueError(
            f'lung-mask pad shape {(pad_h, pad_w)} is smaller than resized '
            f'shape {(resized_h, resized_w)}')

    lung_mask = (lung_mask > 0).to(dtype=torch.float32)
    if tuple(lung_mask.shape) != (resized_h, resized_w):
        lung_mask = F.interpolate(
            lung_mask[None, None],
            size=(resized_h, resized_w),
            mode='nearest')[0, 0]

    if bool(img_meta.get('flip', False)):
        direction = img_meta.get('flip_direction', 'horizontal')
        if direction == 'horizontal':
            lung_mask = lung_mask.flip(1)
        elif direction == 'vertical':
            lung_mask = lung_mask.flip(0)
        elif direction == 'diagonal':
            lung_mask = lung_mask.flip((0, 1))
        else:
            raise ValueError(
                f'Unsupported lung-mask flip direction: {direction}')

    if (pad_h, pad_w) != (resized_h, resized_w):
        lung_mask = F.pad(
            lung_mask,
            (0, pad_w - resized_w, 0, pad_h - resized_h),
            mode='constant',
            value=0.0)
    return lung_mask


def _align_bone_suppressed_to_image_meta(
        bone_image: torch.Tensor,
        img_meta: dict,
        batch_input_shape=None) -> torch.Tensor:
    """Align a source-resolution bone-suppressed CXR to the detector canvas.

    The auxiliary image follows exactly the main image's Resize/Flip/Pad
    geometry, but remains an unnormalised single-channel image in [0, 255].
    It is consumed only by the fixed PC/LBP front-end and never replaces the
    raw CXR supplied to TransNeXt.
    """
    bone_image = torch.as_tensor(bone_image)
    if bone_image.ndim == 3:
        if bone_image.shape[-1] == 1:
            bone_image = bone_image[..., 0]
        elif bone_image.shape[0] == 1:
            bone_image = bone_image[0]
        elif bone_image.shape[-1] in (3, 4):
            # Loader output is normally grayscale. Keep this conversion for
            # externally generated RGB bone-suppression images.
            bgr = bone_image[..., :3].to(dtype=torch.float32)
            bone_image = (
                0.114 * bgr[..., 0]
                + 0.587 * bgr[..., 1]
                + 0.299 * bgr[..., 2])
        else:
            raise ValueError(
                f'bone-suppressed image must be single-channel or HWC RGB, '
                f'got {tuple(bone_image.shape)}')
    if bone_image.ndim != 2:
        raise ValueError(
            f'bone-suppressed image must be 2-D, got '
            f'{tuple(bone_image.shape)}')

    resized_h, resized_w = tuple(int(v) for v in img_meta['img_shape'][:2])
    pad_h, pad_w = _resolve_auxiliary_canvas(
        img_meta, batch_input_shape=batch_input_shape)
    if resized_h <= 0 or resized_w <= 0:
        raise ValueError(
            f'Invalid resized bone-suppressed shape '
            f'{(resized_h, resized_w)}')
    if pad_h < resized_h or pad_w < resized_w:
        raise ValueError(
            f'bone-suppressed pad shape {(pad_h, pad_w)} is smaller than '
            f'resized shape {(resized_h, resized_w)}')

    bone_image = bone_image.to(dtype=torch.float32)
    if tuple(bone_image.shape) != (resized_h, resized_w):
        bone_image = F.interpolate(
            bone_image[None, None],
            size=(resized_h, resized_w),
            mode='bilinear',
            align_corners=False)[0, 0]

    if bool(img_meta.get('flip', False)):
        direction = img_meta.get('flip_direction', 'horizontal')
        if direction == 'horizontal':
            bone_image = bone_image.flip(1)
        elif direction == 'vertical':
            bone_image = bone_image.flip(0)
        elif direction == 'diagonal':
            bone_image = bone_image.flip((0, 1))
        else:
            raise ValueError(
                f'Unsupported bone-suppressed flip direction: {direction}')

    if (pad_h, pad_w) != (resized_h, resized_w):
        bone_image = F.pad(
            bone_image,
            (0, pad_w - resized_w, 0, pad_h - resized_h),
            mode='constant',
            value=0.0)
    return bone_image.clamp_(0.0, 255.0)


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
    """Map LBP 8-bit codes -> uniform-pattern one-hot (59 bins).

    Kept for backward compatibility with compute_lbp_gray.
    New code should prefer _lbp_to_uniform_indices + nn.Embedding.
    """
    indices = _lbp_to_uniform_indices(code, device)
    return F.one_hot(indices, num_classes=59).float()


_UNIFORM_CODES = [
    0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56,
    60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143,
    159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240,
    241, 243, 247, 248, 249, 251, 252, 253, 254, 255
]


def _make_uniform_map(device: torch.device) -> torch.Tensor:
    """Build 256→59 LBP uniform-pattern lookup table (cached per device)."""
    # 58 rotation-specific uniform patterns occupy bins 0..57. All remaining
    # non-uniform patterns belong to the dedicated 59th bin (index 58).
    # The old default of 0 incorrectly merged every non-uniform pattern with
    # the all-zero uniform code.
    uniform_map = torch.full((256,), 58, dtype=torch.long, device=device)
    for i, uc in enumerate(_UNIFORM_CODES):
        uniform_map[uc] = i
    return uniform_map


def _lbp_to_uniform_indices(code: torch.Tensor,
                            device: torch.device) -> torch.Tensor:
    """Map LBP 8-bit codes -> uniform-pattern indices [0-58].

    Returns integer indices suitable for nn.Embedding lookup,
    avoiding the 98.3%-sparse one-hot bottleneck.

    Args:
        code: (B, H, W) integer LBP codes in [0, 255]
        device: target device
    Returns:
        (B, H, W) long tensor with values in [0, 58]
    """
    uniform_map = _make_uniform_map(device)
    return uniform_map[code.clamp(0, 255)]


def _lab_f(t: torch.Tensor) -> torch.Tensor:
    """Lab helper: f(t) = t^(1/3) for t > 0.008856 else 7.787*t + 16/116."""
    delta = 6.0 / 29.0
    return torch.where(t > delta ** 3, t ** (1.0 / 3.0),
                       t / (3 * delta ** 2) + 4.0 / 29.0)


# ============================================================================
# _LayerNorm2d: LayerNorm for (B, C, H, W) tensors
# ============================================================================
class _LayerNorm2d(nn.Module):
    """LayerNorm over channels for 2D feature maps.

    nn.LayerNorm(C) on (B,C,H,W) input is only supported in PyTorch >= 1.10.
    This wrapper permutes to (B,H,W,C), applies LN, and permutes back,
    providing identical behaviour across all PyTorch versions.
    Train/eval behaviour is identical — no running statistics.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → LN over C → (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


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
# _LogGaborPhaseCongruency: source-faithful PC maps for spatial BPPC
# ============================================================================
class _LogGaborPhaseCongruency(nn.Module):
    """Compute orientation-specific PC maps with log-Gabor wavelets.

    The BPPC source first forms phase-congruency maps from quadrature
    log-Gabor responses at multiple scales, then applies LBP to each of six
    orientations. Filters here are deterministic; the downstream spatial LBP
    embedding and detector projections remain learnable.
    """

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

    def _build_filter_bank(self, height: int, width: int,
                           device: torch.device) -> torch.Tensor:
        key = (height, width, device.type, device.index)
        cached = self._filter_cache.get(key)
        if cached is not None:
            return cached

        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
        fx = torch.fft.fftfreq(width, device=device, dtype=torch.float32)
        fy, fx = torch.meshgrid(fy, fx, indexing='ij')
        radius = torch.sqrt(fx.square() + fy.square())
        theta = torch.atan2(-fy, fx)

        radius_safe = radius.clone()
        radius_safe[0, 0] = 1.0  # avoid log(0); DC is removed below
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
        log_sigma = math.log(self.sigma_on_frequency)

        filters = []
        for orient_idx in range(self.num_orientations):
            angle = orient_idx * math.pi / self.num_orientations
            ds = sin_theta * math.cos(angle) - cos_theta * math.sin(angle)
            dc = cos_theta * math.cos(angle) + sin_theta * math.sin(angle)
            delta_theta = torch.abs(torch.atan2(ds, dc))
            delta_theta = torch.clamp(
                delta_theta * self.num_orientations / 2.0,
                max=math.pi)
            spread = (torch.cos(delta_theta) + 1.0) * 0.5

            orient_filters = []
            wavelength = self.min_wavelength
            for _ in range(self.num_scales):
                center_frequency = 1.0 / wavelength
                log_ratio = torch.log(radius_safe / center_frequency)
                radial = torch.exp(
                    -log_ratio.square() / (2.0 * log_sigma ** 2))
                radial[0, 0] = 0.0
                orient_filters.append(radial * spread)
                wavelength *= self.scale_multiplier
            filters.append(torch.stack(orient_filters, dim=0))

        filter_bank = torch.stack(filters, dim=0).detach()
        self._filter_cache[key] = filter_bank
        return filter_bank

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        """Return PC maps shaped ``(B, 6, H, W)`` in [0, 1]."""
        if gray.ndim != 4 or gray.shape[1] != 1:
            raise ValueError('phase congruency expects gray shaped (B,1,H,W)')

        input_dtype = gray.dtype
        work = gray.float().squeeze(1)
        height, width = work.shape[-2:]
        filters = self._build_filter_bank(height, width, work.device)
        image_fft = torch.fft.fft2(work)

        pc_maps = []
        rayleigh_mean = math.sqrt(math.pi / 2.0)
        rayleigh_std = math.sqrt((4.0 - math.pi) / 2.0)
        scale_sum = (
            (1.0 - (1.0 / self.scale_multiplier) ** self.num_scales)
            / (1.0 - 1.0 / self.scale_multiplier))

        for orient_idx in range(self.num_orientations):
            sum_even = torch.zeros_like(work)
            sum_odd = torch.zeros_like(work)
            amplitude_sum = torch.zeros_like(work)
            first_amplitude = None

            for scale_idx in range(self.num_scales):
                response = torch.fft.ifft2(
                    image_fft * filters[orient_idx, scale_idx])
                even, odd = response.real, response.imag
                amplitude = torch.sqrt(
                    even.square() + odd.square() + self.eps)
                if first_amplitude is None:
                    first_amplitude = amplitude
                sum_even = sum_even + even
                sum_odd = sum_odd + odd
                amplitude_sum = amplitude_sum + amplitude

            local_energy = torch.sqrt(
                sum_even.square() + sum_odd.square() + self.eps)

            # Rayleigh noise estimate from the smallest-scale response,
            # expanded across the geometric scale bank as in PC thresholding.
            tau = (
                first_amplitude.detach().mean(dim=(-2, -1), keepdim=True)
                / rayleigh_mean)
            total_tau = tau * scale_sum
            noise_threshold = total_tau * (
                rayleigh_mean + self.noise_k * rayleigh_std)

            pc = F.relu(local_energy - noise_threshold)
            pc = pc / (amplitude_sum + self.eps)
            pc_maps.append(pc.clamp_(0.0, 1.0))

        return torch.stack(pc_maps, dim=1).to(dtype=input_dtype)


# ============================================================================
# _PerPixelLBPBranch: per-pixel grayscale LBP -> FFN -> FPN-compatible features
# ============================================================================
class _PerPixelLBPBranch(nn.Module):
    """Per-pixel LBP + Lab colour -> pointwise MLP -> smooth conv -> feature map.

    Processes texture at pixel resolution without superpixel aggregation,
    avoiding the piecewise-constant artifacts caused by label_map indexing
    of region-level token embeddings.
    """

    def __init__(self, in_channels: int = 576, feat_channels: int = 256):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 1),
            _LayerNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 1),
            _LayerNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            _LayerNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            _LayerNorm2d(feat_channels),
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

    def __init__(self,
                 channels: int = 256,
                 edge_temp_init: float = 1.0,
                 mod_scale_init: float = 0.3,
                 mod_scale_max: float = None,
                 boundary_groups: int = 18):
        super().__init__()
        if boundary_groups <= 0:
            raise ValueError('boundary_groups must be positive')
        self.boundary_groups = int(boundary_groups)
        if mod_scale_max is not None:
            if not 0 < mod_scale_init < mod_scale_max:
                raise ValueError(
                    'Expected 0 < mod_scale_init < mod_scale_max')
            self.mod_scale_max = float(mod_scale_max)
            mod_scale_raw = math.atanh(
                float(mod_scale_init) / self.mod_scale_max)
        else:
            self.mod_scale_max = None
            mod_scale_raw = float(mod_scale_init)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            _LayerNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            _LayerNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.edge_temp_raw = nn.Parameter(torch.tensor(edge_temp_init))
        # Six PC orientations x three LBP radii. Uniform initialization lets
        # every group participate before the regression loss learns a
        # task-specific orientation/scale weighting.
        self.boundary_group_logits = nn.Parameter(
            torch.zeros(self.boundary_groups))
        # The historical head gate keeps its direct learned scale for checkpoint
        # compatibility. The regression-head path requests a bounded tanh scale
        # so handcrafted PC features cannot overwhelm learned detector features.
        self.mod_scale = nn.Parameter(torch.tensor(mod_scale_raw))
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
        if lbp_down.shape[1] % self.boundary_groups != 0:
            raise ValueError(
                f'PC/LBP channels ({lbp_down.shape[1]}) must be divisible by '
                f'boundary_groups ({self.boundary_groups})')
        group_channels = lbp_down.shape[1] // self.boundary_groups
        grouped = lbp_down.reshape(
            lbp_down.shape[0], self.boundary_groups, group_channels, h, w)
        grouped_flat = grouped.reshape(
            lbp_down.shape[0],
            self.boundary_groups * group_channels,
            h,
            w)
        grouped_blur = F.avg_pool2d(grouped_flat, 3, 1, 1).reshape_as(
            grouped)
        group_boundary = (
            grouped - grouped_blur).abs().mean(dim=2)
        group_weights = torch.softmax(
            self.boundary_group_logits, dim=0).view(
                1, self.boundary_groups, 1, 1)
        l_channel = (
            group_boundary * group_weights).sum(
                dim=1, keepdim=True)

        # Edge detection from LBP texture gradient
        l_blur = F.avg_pool2d(l_channel, kernel_size=3, stride=1, padding=1)
        edge_raw = torch.abs(l_channel - l_blur)
        edge_map = self.edge_conv(edge_raw)

        content_gate = torch.tanh(self.gate_net(f_nam))
        edge_temp = F.softplus(self.edge_temp_raw) + 0.01
        spatial_gate = torch.sigmoid(edge_map * edge_temp)
        if self.mod_scale_max is None:
            mod_scale = self.mod_scale
        else:
            mod_scale = self.mod_scale_max * torch.tanh(self.mod_scale)
        gate = content_gate * spatial_gate * mod_scale

        nam_mod = self.nam_proj(f_nam) * gate
        fused = f_sym + nam_mod

        gate_abs = gate.detach().abs()
        gate_stats = {
            'gate_mean': gate_abs.mean().item(),
            'gate_active': (gate_abs > 0.1).float().mean().item(),
            'edge_mean': edge_map.detach().abs().mean().item(),
            'spatial_gate_mean': spatial_gate.detach().mean().item(),
            'edge_temp': edge_temp.item(),
            'mod_scale': mod_scale.detach().abs().item(),
            'boundary_weight_entropy': (
                -(group_weights.flatten()
                  * group_weights.flatten().clamp_min(1e-12).log()).sum()
                / math.log(self.boundary_groups)).detach().item()
            if self.boundary_groups > 1 else 0.0,
            'boundary_weight_peak': (
                group_weights.detach().max().item()),
        }
        return fused, gate_stats


# ============================================================================
# _PCShallowAdapter: the single PC injection used by the 0.361-mAP run
# ============================================================================
class _PCShallowAdapter(nn.Module):
    """Inject spatial BPPC once at the exported C2 seam.

    The adapter is deliberately placed after the backbone and before the neck:
    TransNeXt's internal C2 computation stays intact, while the PC residual can
    reach every CrossScale level.  A bounded BRDG gate limits the handcrafted
    prior and avoids the four duplicate regression-head injections that were
    non-selective in the v3.9 run.
    """

    def __init__(self,
                 pc_channels: int = 576,
                 c2_channels: int = 72,
                 residual_init: float = 0.1,
                 residual_max: float = 0.3):
        super().__init__()
        if not 0 < residual_init < residual_max:
            raise ValueError(
                'residual_init must satisfy 0 < residual_init < residual_max')

        hidden_channels = max(c2_channels * 2, 128)
        self.pc_proj = nn.Sequential(
            nn.Conv2d(pc_channels, hidden_channels, 1),
            _LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, c2_channels, 1),
        )
        self.br_gate = _BoundaryResponsiveGate(
            channels=c2_channels,
            mod_scale_init=residual_init,
            mod_scale_max=residual_max,
            boundary_groups=18)

        self._dbg_pc_feat_norm = 0.0
        self._dbg_pc_gate_mean = 0.0
        self._dbg_c2_pre_pc_norm = 0.0
        self._dbg_residual_scale = residual_init
        self._dbg_contribution_ratio = 0.0

    def forward(self,
                c2_feat: torch.Tensor,
                pc_prior: torch.Tensor = None) -> torch.Tensor:
        if pc_prior is None:
            return c2_feat

        pc_down = F.adaptive_avg_pool2d(
            pc_prior, c2_feat.shape[-2:])
        pc_delta = self.pc_proj(pc_down)
        fused, gate_stats = self.br_gate(
            c2_feat, pc_delta, pc_prior)
        pc_residual = fused - c2_feat

        self._dbg_pc_feat_norm = pc_delta.detach().float().norm().item()
        self._dbg_pc_gate_mean = gate_stats['gate_mean']
        self._dbg_c2_pre_pc_norm = (
            c2_feat.detach().float().norm().item())
        self._dbg_residual_scale = gate_stats['mod_scale']
        self._dbg_contribution_ratio = (
            pc_residual.detach().float().norm()
            / c2_feat.detach().float().norm().clamp_min(1e-6)
        ).item()
        return fused


# ============================================================================
# NEW: _DynamicRegionTokenPool
# Pools region tokens from raw C2 features using label_map as pooling index.
# This is the CRITICAL fix for v3's broken gradient path.
# ============================================================================
class _DynamicRegionTokenPool(nn.Module):
    """Pool region tokens from raw C2 features using label_map as pooling index.

    Replaces v3's handcrafted-token approach (which used .npz node_wld/node_lab
    constants that have no gradient connection to the backbone).

    Gradient path:
        contr_loss -> contrast_head -> Transformer -> tokens
            -> token_proj -> pooled_features -> raw C2 -> backbone ✓

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
        """Pool region tokens with fractional area membership.

        Args:
            fpn_feat: (C, H_f, W_f) single sample's P2 feature
            label_map: (H, W) region ID per pixel (original resolution)
        Returns:
            (tokens, centroids, label_map_down, membership), or four ``None``
            values when fewer than two valid regions exist.
            tokens: (N_valid, token_dim)
            centroids: (N_valid, 2)
            label_map_down: (H_f, W_f) compact hard region IDs
            membership: (N_valid, H_f, W_f) fractional cell ownership

        ``mode='nearest'`` can erase a one-pixel-wide region when a 512 map is
        reduced to C2. Area interpolation of binary masks retains its mass and
        lets bbox overlap be computed continuously at the feature resolution.
        """
        C, H_f, W_f = fpn_feat.shape
        H_lm, W_lm = label_map.shape
        device = fpn_feat.device
        label_map = label_map.to(device=device, dtype=torch.long)
        valid_source = label_map >= 0
        if not valid_source.any():
            return None, None, None, None
        valid_ids = torch.unique(label_map[valid_source], sorted=True)
        if valid_ids.numel() <= 1:
            return None, None, None, None

        # Build masks in chunks to bound peak memory at 512x512. Area mode
        # gives each C2 cell the exact fractional ownership of every region.
        membership_chunks = []
        chunk_size = 32
        for start in range(0, valid_ids.numel(), chunk_size):
            ids = valid_ids[start:start + chunk_size]
            masks = (
                label_map.unsqueeze(0)
                == ids[:, None, None]).to(dtype=fpn_feat.dtype)
            if (H_lm, W_lm) != (H_f, W_f):
                masks = F.interpolate(
                    masks.unsqueeze(1),
                    size=(H_f, W_f),
                    mode='area').squeeze(1)
            membership_chunks.append(masks)
        membership = torch.cat(membership_chunks, dim=0)
        region_mass = membership.sum(dim=(1, 2))
        keep = region_mass > 1e-8
        membership = membership[keep]
        region_mass = region_mass[keep]
        if membership.shape[0] <= 1:
            return None, None, None, None

        pooled = torch.einsum(
            'chw,nhw->nc', fpn_feat, membership)
        pooled = pooled / region_mass.unsqueeze(1).clamp_min(1e-8)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H_f, device=device, dtype=fpn_feat.dtype),
            torch.arange(W_f, device=device, dtype=fpn_feat.dtype),
            indexing='ij')
        centroid_x = (
            membership * x_grid.unsqueeze(0)).sum(dim=(1, 2))
        centroid_y = (
            membership * y_grid.unsqueeze(0)).sum(dim=(1, 2))
        centroids = torch.stack(
            (centroid_x / region_mass, centroid_y / region_mass), dim=1)
        centroids = centroids / centroids.new_tensor(
            [max(W_f - 1, 1), max(H_f - 1, 1)])

        # A hard map is retained only for topology. Ensure a thin region that
        # has non-zero area mass owns at least one cell even when argmax ties.
        total_membership = membership.sum(dim=0)
        label_map_compact = membership.argmax(dim=0).long()
        label_map_compact[total_membership <= 1e-8] = -1
        flat_map = label_map_compact.reshape(-1)
        flat_membership = membership.reshape(membership.shape[0], -1)
        for region_index in range(membership.shape[0]):
            if not (flat_map == region_index).any():
                candidate_order = torch.argsort(
                    flat_membership[region_index], descending=True)
                for candidate in candidate_order.tolist():
                    if flat_membership[region_index, candidate] <= 0:
                        break
                    flat_map[candidate] = region_index
                    break
        label_map_compact = flat_map.reshape(H_f, W_f)

        tokens = self.token_proj(pooled)
        return tokens, centroids, label_map_compact, membership

    @staticmethod
    def _compute_centroids(label_map: torch.Tensor, valid_ids: torch.Tensor,
                           N: int, device: torch.device) -> torch.Tensor:
        """Compute (x, y) centroids for each valid region."""
        H, W = label_map.shape
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij')

        flat_labels_raw = label_map.reshape(-1)
        valid_pixels = flat_labels_raw >= 0
        flat_labels = flat_labels_raw[valid_pixels].clamp(0, N - 1)
        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)
        x_flat = x_flat[valid_pixels]
        y_flat = y_flat[valid_pixels]

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
# CrossScale v3.4: source-faithful detail flow + bounded semantic return
# ============================================================================
class _DetailPreservingDownsample(nn.Module):
    """Move a shallow map to the next level without discarding 2x2 phases.

    Pixel-unshuffle first stores every spatial phase in channels.  A pointwise
    projection and depthwise refinement then adapt it to the target level.
    This follows PtbNet's small-object preservation principle more closely
    than a stride-2 convolution.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            _LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels, channels, 3, padding=1,
                groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.xavier_uniform_(self.project[0].weight)
        nn.init.kaiming_normal_(
            self.project[3].weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.project[-1].weight, gain=0.5)
        nn.init.zeros_(self.project[-1].bias)

    @staticmethod
    def _local_variation(x: torch.Tensor) -> torch.Tensor:
        """Mean first-order spatial variation, normalized by feature RMS."""
        x = x.detach().float()
        terms = []
        if x.shape[-2] > 1:
            terms.append((x[..., 1:, :] - x[..., :-1, :]).abs().mean())
        if x.shape[-1] > 1:
            terms.append((x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        if not terms:
            return x.new_zeros(())
        variation = torch.stack(terms).mean()
        return variation / x.square().mean().sqrt().clamp_min(1e-8)

    def forward(self, x: torch.Tensor, target_size) -> torch.Tensor:
        source = x
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        x = F.pixel_unshuffle(x, downscale_factor=2)
        if x.shape[-2:] != tuple(target_size):
            x = F.interpolate(
                x, size=target_size, mode='bilinear',
                align_corners=False)
        output = self.project(x)
        if getattr(self, '_collect_debug', True):
            with torch.no_grad():
                source_rms = source.detach().float().square().mean().sqrt()
                output_rms = output.detach().float().square().mean().sqrt()
                self._dbg_rms_ratio = (
                    output_rms / source_rms.clamp_min(1e-8))
                source_var = self._local_variation(source)
                output_var = self._local_variation(output)
                self._dbg_variation_ratio = (
                    output_var / source_var.clamp_min(1e-8))
        return output


class _WindowedSpatialCrossAttn(nn.Module):
    """Local cross-position attention with bounded memory at P2/P3.

    ``query_feat`` selects spatial evidence from ``value_feat``.  Unlike the
    old same-location cosine gate, every location can attend to other
    positions inside its window.
    """

    def __init__(self, channels: int = 256, window_size: int = 8):
        super().__init__()
        if window_size < 1:
            raise ValueError('window_size must be positive')
        hidden = max(channels // 8, 8)
        self.window_size = int(window_size)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.xavier_uniform_(self.out_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _partition(self, x: torch.Tensor):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.reshape(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.reshape(-1, ws * ws, C)

    def _restore(self, x: torch.Tensor, batch_size: int,
                 padded_h: int, padded_w: int):
        ws = self.window_size
        C = x.shape[-1]
        x = x.reshape(
            batch_size, padded_h // ws, padded_w // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(batch_size, C, padded_h, padded_w)

    def forward(self, query_feat: torch.Tensor,
                value_feat: torch.Tensor) -> torch.Tensor:
        if value_feat.shape[-2:] != query_feat.shape[-2:]:
            value_feat = F.interpolate(
                value_feat, size=query_feat.shape[-2:],
                mode='bilinear', align_corners=False)
        B, _, H, W = query_feat.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        q = self.query(query_feat)
        k = self.key(value_feat)
        v = self.value(value_feat)
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            q = F.pad(q, padding)
            k = F.pad(k, padding)
            v = F.pad(v, padding)

        q = F.normalize(self._partition(q), p=2, dim=-1, eps=1e-6)
        k = F.normalize(self._partition(k), p=2, dim=-1, eps=1e-6)
        v = self._partition(v)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        attn = torch.softmax(
            logit_scale * (q @ k.transpose(-2, -1)), dim=-1)
        if getattr(self, '_collect_debug', True):
            with torch.no_grad():
                # Sampling keeps diagnostics cheap at high-resolution P2/P3.
                query_stride = max(1, attn.shape[-2] // 16)
                probs = attn.detach().float()[
                    :min(attn.shape[0], 8), ::query_stride]
                probs = probs.clamp_min(1e-8)
                normalizer = math.log(max(2, probs.shape[-1]))
                self._dbg_entropy = (
                    -(probs * probs.log()).sum(dim=-1) / normalizer
                ).mean()
                self._dbg_peak = probs.max(dim=-1).values.mean()
                self._dbg_logit_scale = logit_scale.detach().float()
        enhanced = attn @ v
        enhanced = self._restore(enhanced, B, H + pad_h, W + pad_w)
        enhanced = enhanced[:, :, :H, :W]
        return self.out_proj(enhanced)


class _ChannelCrossAttnMatrix(nn.Module):
    """TB-FusionNet-style CxC cross-channel relation, not an SE gate."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.query_fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.key_fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.value = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query_feat: torch.Tensor,
                value_feat: torch.Tensor) -> torch.Tensor:
        if value_feat.shape[-2:] != query_feat.shape[-2:]:
            value_feat = F.interpolate(
                value_feat, size=query_feat.shape[-2:],
                mode='bilinear', align_corners=False)
        query = self.query_fc(query_feat.mean(dim=(2, 3)))
        key = self.key_fc(value_feat.mean(dim=(2, 3)))
        channel_attn = torch.softmax(
            query.unsqueeze(-1) * key.unsqueeze(-2), dim=-1)
        if getattr(self, '_collect_debug', True):
            with torch.no_grad():
                probs = channel_attn.detach().float().clamp_min(1e-8)
                channels = probs.shape[-1]
                normalizer = math.log(max(2, channels))
                self._dbg_entropy = (
                    -(probs * probs.log()).sum(dim=-1) / normalizer
                ).mean()
                self._dbg_peak = probs.max(dim=-1).values.mean()
                # Uniform attention has diag_gain=1. Values above one indicate
                # preferential same-channel transport.
                self._dbg_diag_gain = (
                    probs.diagonal(dim1=-2, dim2=-1).mean() * channels)
        value = self.value(value_feat).flatten(2)
        enhanced = torch.bmm(channel_attn, value)
        enhanced = enhanced.reshape_as(value_feat)
        return self.out_proj(enhanced)


class _CrossScaleOutputRefine(nn.Module):
    """Bounded local refinement; the old unscaled residual could rewrite P2."""

    def __init__(self, channels: int = 256,
                 residual_init: float = 1e-3,
                 residual_max: float = 0.02):
        super().__init__()
        if not 0 < residual_init < residual_max:
            raise ValueError(
                'refine residual_init must be between zero and residual_max')
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1,
            groups=channels, bias=False)
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.residual_max = float(residual_max)
        self.residual_raw = nn.Parameter(torch.tensor(
            math.atanh(residual_init / residual_max),
            dtype=torch.float32))
        nn.init.kaiming_normal_(
            self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def effective_scale(self) -> torch.Tensor:
        return self.residual_max * torch.tanh(self.residual_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.pointwise(self.act(self.depthwise(x)))
        scaled_delta = self.effective_scale() * delta
        if getattr(self, '_collect_debug', True):
            with torch.no_grad():
                base = x.detach().float().flatten(1)
                change = scaled_delta.detach().float().flatten(1)
                self._dbg_rel_delta = (
                    change.norm(dim=1)
                    / base.norm(dim=1).clamp_min(1e-8)
                ).mean()
                self._dbg_delta_cos = F.cosine_similarity(
                    base, change, dim=1, eps=1e-8).mean()
        return x + scaled_delta


class _FreqFusion2x(nn.Module):
    """Torch-native FreqFusion block for one 2x top-down hop.

    This is the detection-neck subset of FreqFusion: a content-adaptive
    low-pass filter (ALPF) removes intra-class high-frequency noise from the
    deep feature, a content-adaptive high-pass residual (AHPF) preserves the
    high-resolution boundary signal, and a local-similarity-conditioned
    offset aligns the upsampled deep feature.  It intentionally has no MMCV
    CARAFE dependency so the existing PyTorch-1.10/MMCV-1.x server stack can
    run the same implementation.
    """

    def __init__(self, channels: int = 256,
                 compressed_channels: int = 64,
                 kernel_size: int = 5,
                 highpass_kernel_size: int = 3,
                 offset_max: float = 0.5,
                 highpass_init: float = 0.10,
                 highpass_max: float = 0.50):
        super().__init__()
        if compressed_channels <= 0:
            raise ValueError('compressed_channels must be positive')
        if kernel_size % 2 != 1 or highpass_kernel_size % 2 != 1:
            raise ValueError('FreqFusion kernels must be odd')
        if not 0.0 < highpass_init < highpass_max:
            raise ValueError(
                'highpass_init must satisfy 0 < init < max')
        self.kernel_size = int(kernel_size)
        self.highpass_kernel_size = int(highpass_kernel_size)
        self.offset_max = float(offset_max)
        self.highpass_max = float(highpass_max)

        self.hr_compress = nn.Conv2d(
            channels, compressed_channels, 1, bias=False)
        self.lr_compress = nn.Conv2d(
            channels, compressed_channels, 1, bias=False)
        encoder_channels = compressed_channels * 2
        self.alpf_encoder = nn.Conv2d(
            encoder_channels, self.kernel_size ** 2,
            3, padding=1)
        self.ahpf_encoder = nn.Conv2d(
            encoder_channels, self.highpass_kernel_size ** 2,
            3, padding=1)
        # The extra channel is local cosine similarity. Low similarity permits
        # a larger learned correction; high similarity keeps bilinear geometry.
        self.offset_encoder = nn.Conv2d(
            encoder_channels + 1, 2, 3, padding=1)
        self.highpass_raw = nn.Parameter(torch.tensor(
            math.atanh(highpass_init / highpass_max),
            dtype=torch.float32))

        nn.init.kaiming_normal_(
            self.hr_compress.weight, mode='fan_out',
            nonlinearity='relu')
        nn.init.kaiming_normal_(
            self.lr_compress.weight, mode='fan_out',
            nonlinearity='relu')
        nn.init.normal_(self.alpf_encoder.weight, std=1e-3)
        nn.init.zeros_(self.alpf_encoder.bias)
        nn.init.normal_(self.ahpf_encoder.weight, std=1e-3)
        nn.init.zeros_(self.ahpf_encoder.bias)
        # Zero offset is an exact bilinear alignment at initialization. The
        # offset branch still receives gradients through grid_sample.
        nn.init.zeros_(self.offset_encoder.weight)
        nn.init.zeros_(self.offset_encoder.bias)

        self.register_buffer(
            'alpf_hamming',
            self._hamming_2d(self.kernel_size),
            persistent=False)
        self.register_buffer(
            'ahpf_hamming',
            self._hamming_2d(self.highpass_kernel_size),
            persistent=False)
        self._collect_debug = True
        self._dbg_alpf_entropy = 0.0
        self._dbg_ahpf_entropy = 0.0
        self._dbg_offset_abs = 0.0
        self._dbg_lr_vs_bilinear = 0.0
        self._dbg_hr_detail_rel = 0.0
        self._dbg_local_similarity = 0.0

    @staticmethod
    def _hamming_2d(kernel_size: int) -> torch.Tensor:
        window = torch.hamming_window(
            kernel_size, periodic=False, dtype=torch.float32)
        window = torch.outer(window, window).reshape(
            1, kernel_size * kernel_size, 1, 1)
        return window / window.mean().clamp_min(1e-8)

    @staticmethod
    def _normalise_kernel(
            logits: torch.Tensor,
            hamming: torch.Tensor) -> torch.Tensor:
        kernel = torch.softmax(logits, dim=1)
        kernel = kernel * hamming.to(
            device=kernel.device, dtype=kernel.dtype)
        return kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _dynamic_filter(
            feature: torch.Tensor,
            kernel: torch.Tensor,
            kernel_size: int) -> torch.Tensor:
        if _mmcv_carafe is not None and feature.is_cuda:
            return _mmcv_carafe(
                feature, kernel,
                kernel_size=kernel_size,
                group_size=1,
                scale_factor=1)
        batch, channels, height, width = feature.shape
        patches = F.unfold(
            feature, kernel_size=kernel_size,
            padding=kernel_size // 2)
        patches = patches.view(
            batch, channels, kernel_size * kernel_size,
            height, width)
        return (patches * kernel.unsqueeze(1)).sum(dim=2)

    @staticmethod
    def _offset_grid(offset: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = offset.shape
        dtype, device = offset.dtype, offset.device
        y = (
            (torch.arange(height, device=device, dtype=dtype) + 0.5)
            * (2.0 / max(height, 1)) - 1.0)
        x = (
            (torch.arange(width, device=device, dtype=dtype) + 0.5)
            * (2.0 / max(width, 1)) - 1.0)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(
            batch, -1, -1, -1)
        dx = offset[:, 0] * (2.0 / max(width, 1))
        dy = offset[:, 1] * (2.0 / max(height, 1))
        return base + torch.stack((dx, dy), dim=-1)

    def effective_highpass_scale(self) -> torch.Tensor:
        return self.highpass_max * torch.tanh(self.highpass_raw)

    def forward(self, hr_feat: torch.Tensor,
                lr_feat: torch.Tensor) -> tuple:
        target_size = hr_feat.shape[-2:]
        bilinear = F.interpolate(
            lr_feat, size=target_size,
            mode='bilinear', align_corners=False)
        hr_comp = self.hr_compress(hr_feat)
        lr_comp = self.lr_compress(bilinear)
        hr_unit = F.normalize(hr_comp, dim=1, eps=1e-6)
        lr_unit = F.normalize(lr_comp, dim=1, eps=1e-6)
        local_similarity = (hr_unit * lr_unit).sum(
            dim=1, keepdim=True).clamp(-1.0, 1.0)
        content = torch.cat((hr_comp, lr_comp), dim=1)

        # Local similarity guides the maximum alignment correction. This
        # prevents arbitrary geometric warping where the two scales agree.
        mismatch = (1.0 - local_similarity) * 0.5
        offset = (
            self.offset_max
            * torch.tanh(self.offset_encoder(torch.cat(
                (content, local_similarity), dim=1)))
            * mismatch)
        aligned_lr = F.grid_sample(
            bilinear, self._offset_grid(offset),
            mode='bilinear', padding_mode='border',
            align_corners=False)

        alpf_kernel = self._normalise_kernel(
            self.alpf_encoder(content), self.alpf_hamming)
        filtered_lr = self._dynamic_filter(
            aligned_lr, alpf_kernel, self.kernel_size)
        ahpf_kernel = self._normalise_kernel(
            self.ahpf_encoder(content), self.ahpf_hamming)
        low_hr = self._dynamic_filter(
            hr_feat, ahpf_kernel, self.highpass_kernel_size)
        high_residual = hr_feat - low_hr
        enhanced_hr = (
            hr_feat
            + self.effective_highpass_scale() * high_residual)

        if self._collect_debug:
            with torch.no_grad():
                def normalized_entropy(kernel):
                    entropy = -(
                        kernel.clamp_min(1e-8)
                        * kernel.clamp_min(1e-8).log()
                    ).sum(dim=1)
                    return (
                        entropy / math.log(kernel.shape[1])
                    ).mean().item()

                lr_norm = bilinear.detach().float().flatten(1).norm(
                    dim=1).clamp_min(1e-8)
                hr_norm = hr_feat.detach().float().flatten(1).norm(
                    dim=1).clamp_min(1e-8)
                self._dbg_alpf_entropy = normalized_entropy(alpf_kernel)
                self._dbg_ahpf_entropy = normalized_entropy(ahpf_kernel)
                self._dbg_offset_abs = float(
                    offset.detach().float().abs().mean().item())
                self._dbg_lr_vs_bilinear = float((
                    (filtered_lr.detach().float() - bilinear.detach().float())
                    .flatten(1).norm(dim=1) / lr_norm).mean().item())
                self._dbg_hr_detail_rel = float((
                    (enhanced_hr.detach().float() - hr_feat.detach().float())
                    .flatten(1).norm(dim=1) / hr_norm).mean().item())
                self._dbg_local_similarity = float(
                    local_similarity.detach().float().mean().item())
        return enhanced_hr, filtered_lr


class _C5GatedC2RegressionDetail(nn.Module):
    """Independent C2->P4/P5 detail routes gated, not sourced, by C5.

    The C5 feature supplies only multiplicative spatial/channel confidence.
    Therefore the branch cannot hallucinate a detail tensor when C2 is zero,
    and no deep semantic tensor is additively duplicated into the head.
    """

    def __init__(self, channels: int = 256,
                 compressed_channels: int = 32):
        super().__init__()
        if compressed_channels <= 0:
            raise ValueError('compressed_channels must be positive')
        self.compressed_channels = int(compressed_channels)
        self.detail_compress = nn.Conv2d(
            channels, compressed_channels, 1, bias=False)
        self.route_factors = (4, 8)
        self.route_projections = nn.ModuleList([
            nn.Conv2d(
                compressed_channels * factor * factor,
                channels, 1, bias=False)
            for factor in self.route_factors
        ])
        self.channel_gates = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, bias=True)
            for _ in self.route_factors
        ])
        self.spatial_gates = nn.ModuleList([
            nn.Conv2d(channels, 1, 1, bias=True)
            for _ in self.route_factors
        ])
        nn.init.kaiming_normal_(
            self.detail_compress.weight, mode='fan_out',
            nonlinearity='relu')
        for projection in self.route_projections:
            nn.init.kaiming_normal_(
                projection.weight, mode='fan_out',
                nonlinearity='relu')
        # Tiny non-zero weights keep the initial gate close to one while
        # preserving a measurable bbox-gradient route into C5.
        for gate in list(self.channel_gates) + list(self.spatial_gates):
            nn.init.normal_(gate.weight, std=1e-3)
            nn.init.zeros_(gate.bias)
        self._collect_debug = True
        self._dbg_gate_mean = [0.0, 0.0]
        self._dbg_detail_rms = [0.0, 0.0]
        self._dbg_highpass_ratio = 0.0

    def forward(self, c2: torch.Tensor, c5: torch.Tensor,
                target_sizes) -> tuple:
        if len(target_sizes) != 2:
            raise ValueError('target_sizes must contain P4 and P5 sizes')
        smooth = F.avg_pool2d(
            c2, kernel_size=3, stride=1, padding=1)
        highpass = c2 - smooth
        compressed = self.detail_compress(highpass)
        c2_norm = c2.detach().float().flatten(1).norm(
            dim=1).clamp_min(1e-8)
        if self._collect_debug:
            self._dbg_highpass_ratio = float((
                highpass.detach().float().flatten(1).norm(dim=1)
                / c2_norm).mean().item())

        outputs, gate_means, detail_rms = [], [], []
        for index, (factor, target_size) in enumerate(zip(
                self.route_factors, target_sizes)):
            target_size = tuple(int(value) for value in target_size)
            route_input = F.interpolate(
                compressed,
                size=(target_size[0] * factor,
                      target_size[1] * factor),
                mode='bilinear', align_corners=False)
            route_input = F.pixel_unshuffle(route_input, factor)
            detail = self.route_projections[index](route_input)
            channel_gate = 2.0 * torch.sigmoid(
                self.channel_gates[index](
                    F.adaptive_avg_pool2d(c5, 1)))
            spatial_gate = 2.0 * torch.sigmoid(F.interpolate(
                self.spatial_gates[index](c5),
                size=target_size, mode='bilinear',
                align_corners=False))
            gate = channel_gate * spatial_gate
            output = detail * gate
            outputs.append(output)
            if self._collect_debug:
                gate_means.append(float(
                    gate.detach().float().mean().item()))
                detail_rms.append(float(
                    output.detach().float().square().mean().sqrt().item()))
        if self._collect_debug:
            self._dbg_gate_mean = gate_means
            self._dbg_detail_rms = detail_rms
        return tuple(outputs)

    def debug_stats(self) -> dict:
        return {
            'gate_mean': list(self._dbg_gate_mean),
            'detail_rms': list(self._dbg_detail_rms),
            'highpass_ratio': float(self._dbg_highpass_ratio),
        }


class _RegressionDetailInjector(nn.Module):
    """Bounded, regression-only injection at selected pyramid levels."""

    def __init__(self, channels: int = 256, num_levels: int = 4,
                 levels=(2, 3), residual_init: float = 1e-2,
                 residual_max: float = 0.1):
        super().__init__()
        self.num_levels = int(num_levels)
        self.levels = tuple(int(level) for level in levels)
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError('levels must be a non-empty unique sequence')
        if min(self.levels) < 0 or max(self.levels) >= self.num_levels:
            raise ValueError('regression detail level is out of range')
        if not 0.0 < residual_init < residual_max:
            raise ValueError(
                'detail residual_init must satisfy 0 < init < max')
        self.residual_max = float(residual_max)
        raw_init = math.atanh(residual_init / residual_max)
        self.residual_raw = nn.Parameter(torch.full(
            (len(self.levels),), raw_init, dtype=torch.float32))
        self.adapters = nn.ModuleDict()
        for level in self.levels:
            adapter = nn.Sequential(
                nn.Conv2d(
                    channels, channels, 3, padding=1,
                    groups=channels, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1, bias=False))
            nn.init.kaiming_normal_(
                adapter[0].weight, mode='fan_out',
                nonlinearity='relu')
            nn.init.xavier_uniform_(adapter[2].weight)
            self.adapters[str(level)] = adapter
        self._dbg_rel_delta = [0.0] * self.num_levels
        self._dbg_detail_rms = [0.0] * self.num_levels

    def effective_scales(self) -> torch.Tensor:
        return self.residual_max * torch.tanh(self.residual_raw)

    def forward(self, reg_feat: torch.Tensor,
                detail_feat: torch.Tensor,
                level_index: int) -> torch.Tensor:
        if level_index not in self.levels or detail_feat is None:
            return reg_feat
        parameter_index = self.levels.index(level_index)
        if detail_feat.shape[-2:] != reg_feat.shape[-2:]:
            detail_feat = F.interpolate(
                detail_feat, size=reg_feat.shape[-2:],
                mode='bilinear', align_corners=False)
        delta = (
            self.effective_scales()[parameter_index]
            * self.adapters[str(level_index)](detail_feat))
        with torch.no_grad():
            base_norm = reg_feat.detach().float().flatten(1).norm(
                dim=1).clamp_min(1e-8)
            self._dbg_rel_delta[level_index] = float((
                delta.detach().float().flatten(1).norm(dim=1)
                / base_norm).mean().item())
            self._dbg_detail_rms[level_index] = float(
                detail_feat.detach().float().square().mean().sqrt().item())
        return reg_feat + delta

    def debug_stats(self) -> dict:
        scales = [0.0] * self.num_levels
        for index, level in enumerate(self.levels):
            scales[level] = float(
                self.effective_scales()[index].detach().float().item())
        return {
            'scale': scales,
            'rel_delta': list(self._dbg_rel_delta),
            'detail_rms': list(self._dbg_detail_rms),
        }


class _TBFusionCrossScale(nn.Module):
    """Direct FPN replacement with detail-first bidirectional fusion.

    Main path (source-faithful): P2→P3→P4→P5.  Pixel-unshuffle preserves
    small-lesion phases before windowed spatial and CxC channel cross-attention.

    Optional semantic return: P5→P4→P3→P2.  Its maximum contribution is
    separately bounded and decreases toward P2, while raw P2 bypasses output
    refinement.  Rows returned by :meth:`effective_scales` are the three
    bottom-up hops followed by the three top-down hops.
    """

    def __init__(self, channels: int = 256,
                 residual_init: float = 1e-3,
                 residual_max: float = 0.1,
                 semantic_residual_init: float = 5e-4,
                 semantic_residual_max: float = 0.02,
                 refine_residual_init: float = 1e-3,
                 refine_residual_max: float = 0.02,
                 window_size: int = 8,
                 use_top_down: bool = True,
                 learnable_fusion_weights: bool = False,
                 detail_fusion_init: float = 0.75,
                 semantic_fusion_init: float = 0.25,
                 freqfusion_top_down: bool = False,
                 freqfusion_compressed_channels: int = 64):
        super().__init__()
        self._validate_scale(
            'detail', residual_init, residual_max)
        self._validate_scale(
            'semantic', semantic_residual_init, semantic_residual_max)
        self.use_top_down = bool(use_top_down)
        self.learnable_fusion_weights = bool(
            learnable_fusion_weights)
        self.freqfusion_top_down = bool(freqfusion_top_down)
        self.detail_residual_max = float(residual_max)
        self.semantic_residual_max = float(semantic_residual_max)
        self._validate_fusion_fraction(
            'detail_fusion_init', detail_fusion_init)
        self._validate_fusion_fraction(
            'semantic_fusion_init', semantic_fusion_init)

        self.bottom_up_downsamples = nn.ModuleList([
            _DetailPreservingDownsample(channels) for _ in range(3)
        ])
        self.bottom_up_sca = nn.ModuleList([
            _WindowedSpatialCrossAttn(channels, window_size)
            for _ in range(3)
        ])
        self.bottom_up_cca = nn.ModuleList([
            _ChannelCrossAttnMatrix(channels) for _ in range(3)
        ])
        self.top_down_sca = nn.ModuleList([
            _WindowedSpatialCrossAttn(channels, window_size)
            for _ in range(3)
        ])
        self.top_down_cca = nn.ModuleList([
            _ChannelCrossAttnMatrix(channels) for _ in range(3)
        ])
        self.freqfusion_blocks = nn.ModuleList([
            _FreqFusion2x(
                channels=channels,
                compressed_channels=freqfusion_compressed_channels)
            for _ in range(2)
        ] if self.freqfusion_top_down else [])

        detail_raw_init = math.atanh(residual_init / residual_max)
        semantic_raw_init = math.atanh(
            semantic_residual_init / semantic_residual_max)
        self.detail_residual_raw = nn.Parameter(torch.full(
            (3, 2), detail_raw_init, dtype=torch.float32))
        self.semantic_residual_raw = nn.Parameter(torch.full(
            (3, 2), semantic_raw_init, dtype=torch.float32))
        self.detail_weight_raw = nn.Parameter(
            torch.zeros((3, 2), dtype=torch.float32))
        self.semantic_weight_raw = nn.Parameter(
            torch.zeros((3, 2), dtype=torch.float32))
        self.register_buffer(
            'semantic_level_factors',
            torch.tensor([[1.0], [0.75], [0.5]], dtype=torch.float32))
        detail_logits = torch.log(torch.tensor(
            [1.0 - detail_fusion_init, detail_fusion_init],
            dtype=torch.float32)).repeat(3, 1)
        semantic_logits = torch.log(torch.tensor(
            [1.0 - semantic_fusion_init, semantic_fusion_init],
            dtype=torch.float32)).repeat(3, 1)
        if self.learnable_fusion_weights:
            self.detail_fusion_logits = nn.Parameter(detail_logits)
            self.semantic_fusion_logits = nn.Parameter(semantic_logits)
        else:
            # Buffers preserve checkpoint/config compatibility without adding
            # unused optimizer parameters to the P0U0 baseline.
            self.register_buffer(
                'detail_fusion_logits', detail_logits)
            self.register_buffer(
                'semantic_fusion_logits', semantic_logits)

        # P2 deliberately has no post-fusion convolution.  This keeps its raw
        # high-resolution route intact for small-object anchors.
        self.output_refines = nn.ModuleList([
            _CrossScaleOutputRefine(
                channels, refine_residual_init, refine_residual_max)
            for _ in range(3)
        ])
        self._dbg_hops = []
        self._dbg_output_grad_rms = [None] * 4
        self._dbg_top_down_modes = ['bilinear'] * 3
        # The detector enables these only around periodic debug iterations;
        # direct unit use defaults to full collection.
        self.collect_forward_debug = True
        self.collect_gradient_debug = True

    @staticmethod
    def _validate_scale(name: str, initial: float, maximum: float):
        if not 0 < initial < maximum:
            raise ValueError(
                f'{name} residual_init must satisfy '
                '0 < residual_init < residual_max')

    @staticmethod
    def _validate_fusion_fraction(name: str, value: float):
        if not 0.0 < float(value) < 1.0:
            raise ValueError(f'{name} must be in (0, 1)')

    @staticmethod
    def _branch_scales(raw: torch.Tensor, weights: torch.Tensor,
                       maximum: float) -> torch.Tensor:
        branch_weights = 2.0 * torch.softmax(weights, dim=-1)
        residual_scales = maximum * torch.tanh(raw)
        return branch_weights * residual_scales

    def effective_scales(self) -> torch.Tensor:
        detail = self._branch_scales(
            self.detail_residual_raw,
            self.detail_weight_raw,
            self.detail_residual_max)
        semantic = self._branch_scales(
            self.semantic_residual_raw,
            self.semantic_weight_raw,
            self.semantic_residual_max)
        semantic = semantic * self.semantic_level_factors
        if not self.use_top_down:
            semantic = torch.zeros_like(semantic)
        return torch.cat([detail, semantic], dim=0)

    def effective_fusion_weights(self) -> torch.Tensor:
        """Return ``[base, cross-scale proposal]`` weights for six hops."""
        if not self.learnable_fusion_weights:
            # Disabled means byte-for-byte feature algebra from v3.16:
            # accept the complete residual proposal returned by _fuse.
            disabled = self.detail_fusion_logits.new_tensor(
                [0.0, 1.0]).repeat(3, 1)
            detail = disabled
            semantic = disabled.clone()
        else:
            detail = F.softmax(self.detail_fusion_logits, dim=-1)
            semantic = F.softmax(self.semantic_fusion_logits, dim=-1)
        if not self.use_top_down:
            semantic = semantic.new_tensor([1.0, 0.0]).repeat(3, 1)
        return torch.cat([detail, semantic], dim=0)

    @staticmethod
    def _weighted_feature_fusion(
            base: torch.Tensor,
            proposal: torch.Tensor,
            weights: torch.Tensor) -> torch.Tensor:
        return weights[0] * base + weights[1] * proposal

    def _fuse(self, target: torch.Tensor, source: torch.Tensor,
              sca: nn.Module, cca: nn.Module,
              scales: torch.Tensor) -> torch.Tensor:
        sca_raw = sca(query_feat=target, value_feat=source)
        cca_raw = cca(query_feat=target, value_feat=source)
        sca_delta = scales[0] * sca_raw
        cca_delta = scales[1] * cca_raw
        total_delta = sca_delta + cca_delta
        if self.collect_forward_debug:
            with torch.no_grad():
                target_flat = target.detach().float().flatten(1)
                sca_flat = sca_delta.detach().float().flatten(1)
                cca_flat = cca_delta.detach().float().flatten(1)
                total_flat = total_delta.detach().float().flatten(1)
                target_norm = target_flat.norm(dim=1).clamp_min(1e-8)

                def relative_norm(value):
                    return (value.norm(dim=1) / target_norm).mean()

                self._dbg_hops.append({
                    'rel_sca': relative_norm(sca_flat),
                    'rel_cca': relative_norm(cca_flat),
                    'rel_total': relative_norm(total_flat),
                    'delta_cos': F.cosine_similarity(
                        target_flat, total_flat, dim=1, eps=1e-8).mean(),
                    'branch_cos': F.cosine_similarity(
                        sca_flat, cca_flat, dim=1, eps=1e-8).mean(),
                })
        return target + total_delta

    @staticmethod
    def _debug_value(module: nn.Module, name: str,
                     default: float = 0.0) -> float:
        value = getattr(module, name, default)
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        value = float(value)
        return value if math.isfinite(value) else float(default)

    def debug_stats(self) -> dict:
        """Return forward-path evidence used by the periodic debug logger.

        The residual scales show intent; ``hop_rel_*`` shows the actual
        feature change after attention, which is the stronger effectiveness
        test. Attention entropy is normalized to [0, 1], where values near
        one indicate an almost uniform transport matrix.
        """
        scales = self.effective_scales().detach().float().cpu()
        hops = list(self._dbg_hops)
        while len(hops) < 6:
            hops.append({})

        spatial_modules = list(self.bottom_up_sca) + list(self.top_down_sca)
        channel_modules = list(self.bottom_up_cca) + list(self.top_down_cca)
        return {
            'scale_sca': scales[:, 0].tolist(),
            'scale_cca': scales[:, 1].tolist(),
            'fusion_base_weight':
                self.effective_fusion_weights()[:, 0]
                .detach().float().cpu().tolist(),
            'fusion_proposal_weight':
                self.effective_fusion_weights()[:, 1]
                .detach().float().cpu().tolist(),
            'hop_rel_sca': [
                self._debug_value_from_mapping(hop, 'rel_sca')
                for hop in hops],
            'hop_rel_cca': [
                self._debug_value_from_mapping(hop, 'rel_cca')
                for hop in hops],
            'hop_rel_total': [
                self._debug_value_from_mapping(hop, 'rel_total')
                for hop in hops],
            'hop_delta_cos': [
                self._debug_value_from_mapping(hop, 'delta_cos')
                for hop in hops],
            'hop_branch_cos': [
                self._debug_value_from_mapping(hop, 'branch_cos')
                for hop in hops],
            'sca_entropy': [
                self._debug_value(module, '_dbg_entropy')
                for module in spatial_modules],
            'sca_peak': [
                self._debug_value(module, '_dbg_peak')
                for module in spatial_modules],
            'cca_entropy': [
                self._debug_value(module, '_dbg_entropy')
                for module in channel_modules],
            'cca_peak': [
                self._debug_value(module, '_dbg_peak')
                for module in channel_modules],
            'cca_diag_gain': [
                self._debug_value(module, '_dbg_diag_gain')
                for module in channel_modules],
            'downsample_rms_ratio': [
                self._debug_value(module, '_dbg_rms_ratio')
                for module in self.bottom_up_downsamples],
            'downsample_variation_ratio': [
                self._debug_value(module, '_dbg_variation_ratio')
                for module in self.bottom_up_downsamples],
            'refine_rel_delta': [
                self._debug_value(module, '_dbg_rel_delta')
                for module in self.output_refines],
            'refine_delta_cos': [
                self._debug_value(module, '_dbg_delta_cos')
                for module in self.output_refines],
            'freq_alpf_entropy': [
                self._debug_value(module, '_dbg_alpf_entropy')
                for module in self.freqfusion_blocks],
            'freq_ahpf_entropy': [
                self._debug_value(module, '_dbg_ahpf_entropy')
                for module in self.freqfusion_blocks],
            'freq_offset_abs': [
                self._debug_value(module, '_dbg_offset_abs')
                for module in self.freqfusion_blocks],
            'freq_lr_vs_bilinear': [
                self._debug_value(module, '_dbg_lr_vs_bilinear')
                for module in self.freqfusion_blocks],
            'freq_hr_detail_rel': [
                self._debug_value(module, '_dbg_hr_detail_rel')
                for module in self.freqfusion_blocks],
            'freq_local_similarity': [
                self._debug_value(module, '_dbg_local_similarity')
                for module in self.freqfusion_blocks],
        }

    @staticmethod
    def _debug_value_from_mapping(mapping: dict, name: str,
                                  default: float = 0.0) -> float:
        value = mapping.get(name, default)
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        value = float(value)
        return value if math.isfinite(value) else float(default)

    @staticmethod
    def _gradient_group(parameters) -> tuple:
        sum_sq = 0.0
        seen_numel = 0
        total_numel = 0
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            total_numel += parameter.numel()
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            sum_sq += grad.square().sum().item()
            seen_numel += parameter.numel()
        rms = math.sqrt(sum_sq / max(1, seen_numel))
        coverage = seen_numel / max(1, total_numel)
        return rms, coverage

    def gradient_stats(self) -> dict:
        """Report whether detection gradients reach every CrossScale branch."""
        parameter_groups = {
            'bottom_up_downsample':
                self.bottom_up_downsamples.parameters(),
            'bottom_up_sca': self.bottom_up_sca.parameters(),
            'bottom_up_cca': self.bottom_up_cca.parameters(),
            'top_down_sca': self.top_down_sca.parameters(),
            'top_down_cca': self.top_down_cca.parameters(),
            'output_refine': self.output_refines.parameters(),
            'detail_scale': [
                self.detail_residual_raw, self.detail_weight_raw],
            'semantic_scale': [
                self.semantic_residual_raw, self.semantic_weight_raw],
        }
        if self.learnable_fusion_weights:
            parameter_groups['fusion_weight'] = [
                self.detail_fusion_logits,
                self.semantic_fusion_logits]
        if self.freqfusion_top_down:
            parameter_groups['freqfusion'] = (
                self.freqfusion_blocks.parameters())
        grad_rms = {}
        grad_coverage = {}
        for name, parameters in parameter_groups.items():
            rms, coverage = self._gradient_group(parameters)
            grad_rms[name] = rms
            grad_coverage[name] = coverage

        output_grad_rms = []
        for value in self._dbg_output_grad_rms:
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().item()
            value = 0.0 if value is None else float(value)
            output_grad_rms.append(
                value if math.isfinite(value) else 0.0)
        return {
            'output_grad_rms': output_grad_rms,
            'group_grad_rms': grad_rms,
            'group_grad_coverage': grad_coverage,
        }

    def _capture_output_gradient(self, index: int, grad: torch.Tensor):
        self._dbg_output_grad_rms[index] = (
            grad.detach().float().square().mean().sqrt())
        return grad

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError('CrossScale requires P2/P3/P4/P5 inputs')
        P2, P3, P4, P5 = feats
        scales = self.effective_scales()
        fusion_weights = self.effective_fusion_weights()
        self._dbg_hops = []
        self._dbg_output_grad_rms = [None] * 4
        debug_modules = (
            list(self.bottom_up_downsamples)
            + list(self.bottom_up_sca)
            + list(self.bottom_up_cca)
            + list(self.top_down_sca)
            + list(self.top_down_cca)
            + list(self.freqfusion_blocks)
            + list(self.output_refines))
        for module in debug_modules:
            module._collect_debug = self.collect_forward_debug

        # Detail-first path: source-faithful low→high fusion.
        P2d = self.bottom_up_downsamples[0](P2, P3.shape[-2:])
        P3_proposal = self._fuse(
            P3, P2d, self.bottom_up_sca[0],
            self.bottom_up_cca[0], scales[0])
        P3b = self._weighted_feature_fusion(
            P3, P3_proposal, fusion_weights[0])
        P3d = self.bottom_up_downsamples[1](P3b, P4.shape[-2:])
        P4_proposal = self._fuse(
            P4, P3d, self.bottom_up_sca[1],
            self.bottom_up_cca[1], scales[1])
        P4b = self._weighted_feature_fusion(
            P4, P4_proposal, fusion_weights[1])
        P4d = self.bottom_up_downsamples[2](P4b, P5.shape[-2:])
        P5_proposal = self._fuse(
            P5, P4d, self.bottom_up_sca[2],
            self.bottom_up_cca[2], scales[2])
        P5b = self._weighted_feature_fusion(
            P5, P5_proposal, fusion_weights[2])

        if self.use_top_down:
            # Lightweight semantic return.  Here the deep feature is VALUE,
            # so semantic content, rather than another shallow reweighting,
            # is actually injected into the target level.
            if self.freqfusion_top_down:
                P4_target, P5u = self.freqfusion_blocks[0](
                    P4b, P5b)
                self._dbg_top_down_modes = [
                    'freqfusion', 'freqfusion', 'bilinear']
            else:
                P4_target = P4b
                P5u = deterministic_bilinear_interpolate(
                    P5b, size=P4b.shape[-2:], align_corners=False)
                self._dbg_top_down_modes = ['bilinear'] * 3
            P4_proposal = self._fuse(
                P4_target, P5u, self.top_down_sca[0],
                self.top_down_cca[0], scales[3])
            P4o = self._weighted_feature_fusion(
                P4_target, P4_proposal, fusion_weights[3])
            if self.freqfusion_top_down:
                P3_target, P4u = self.freqfusion_blocks[1](
                    P3b, P4o)
            else:
                P3_target = P3b
                P4u = deterministic_bilinear_interpolate(
                    P4o, size=P3b.shape[-2:], align_corners=False)
            P3_proposal = self._fuse(
                P3_target, P4u, self.top_down_sca[1],
                self.top_down_cca[1], scales[4])
            P3o = self._weighted_feature_fusion(
                P3_target, P3_proposal, fusion_weights[4])
            P3u = deterministic_bilinear_interpolate(
                P3o, size=P2.shape[-2:], align_corners=False)
            P2_proposal = self._fuse(
                P2, P3u, self.top_down_sca[2],
                self.top_down_cca[2], scales[5])
            P2o = self._weighted_feature_fusion(
                P2, P2_proposal, fusion_weights[5])
        else:
            P2o, P3o, P4o = P2, P3b, P4b
            self._dbg_top_down_modes = ['disabled'] * 3

        outputs = [
            P2o,
            self.output_refines[0](P3o),
            self.output_refines[1](P4o),
            self.output_refines[2](P5b),
        ]
        if (self.collect_gradient_debug and self.training
                and torch.is_grad_enabled()):
            for index, output in enumerate(outputs):
                if output.requires_grad:
                    output.register_hook(
                        lambda grad, i=index:
                        self._capture_output_gradient(i, grad))
        return outputs

# ============================================================================
# _RegionTransformerBranchV3_3
# Core change: tokens pooled from raw C2, NOT from .npz handcrafted data.
# ============================================================================
class _RegionTransformerBranchV3_3(nn.Module):
    """Region Transformer NAM branch v3.3.

    v3 -> v3.3 core change:
      Tokens are pooled from raw backbone C2 using label_map as pooling index,
      NOT built from handcrafted .npz features (node_wld, node_lab, etc.).
      This fixes the broken gradient path — contrastive loss gradients now
      flow directly to the backbone.

    Pipeline:
      1. Pool region tokens from raw C2 features via label_map mask
      2. Project to d_model + add learnable position embedding
      3. Compute spatial distance bias from region centroids
      4. TransformerEncoder x L layers (global self-attention + spatial bias)
      5. Instance contrast pulls regions assigned to the same GT bbox together
         and separates them from other lesions and bbox-exterior regions
    """

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

    def _process_single_sample(self, fpn_feat: torch.Tensor,
                                graph_data: dict) -> tuple:
        """Process one sample's region tokens through Transformer.
        The graph supplies only the region map; token values come from C2, so
        the contrastive loss has a live gradient path to the backbone.

        Args:
            graph_data: dict from LoadNAMGraph (all .npz fields)
        Returns:
            (node_out, label_map, fractional_membership), or three ``None``.
        """
        device = fpn_feat.device
        label_map = graph_data['label_map'].to(device)

        # === Pool tokens from FPN features via .npz label_map ===
        tokens, centroids, label_map_ds, membership = self.token_pool(
            fpn_feat, label_map)

        if (tokens is None or centroids is None or label_map_ds is None
                or membership is None):
            return None, None, None
        N = tokens.size(0)
        if N <= 1:
            return None, None, None

        if N > self.max_regions:
            raise ValueError(
                f'Too many regions ({N}); increase max_regions='
                f'{self.max_regions}')
        region_area = membership.sum(dim=(1, 2))
        area_fraction = region_area / float(
            membership.shape[-2] * membership.shape[-1])
        geometry = torch.stack((
            centroids[:, 0],
            centroids[:, 1],
            area_fraction.sqrt(),
            torch.log1p(region_area)
            / math.log1p(
                membership.shape[-2] * membership.shape[-1])),
            dim=1)
        x = (
            tokens + self.geometry_position(geometry)
        ).unsqueeze(0)

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

            label_map_remapped = torch.full_like(label_map_ds, -1)
            valid_map_pixels = label_map_ds >= 0
            label_map_remapped[valid_map_pixels] = remap[
                label_map_ds[valid_map_pixels].long()]
            remap_matrix = F.one_hot(
                remap, num_classes=kept_idx.numel()).to(
                    dtype=membership.dtype).T
            membership = (
                remap_matrix @ membership.reshape(N, -1)
            ).reshape(
                kept_idx.numel(),
                membership.shape[-2],
                membership.shape[-1])
            x = x[:, kept_idx, :]
            centroids = centroids[kept_idx]
            spatial_bias = self.spatial_bias_fn(centroids)
        else:
            label_map_remapped = label_map_ds

        # === Transformer encoder ===
        for layer in self.encoder_layers:
            x = layer(x, spatial_bias)
        x = self.final_norm(x).squeeze(0)  # (N_tokens, d_model)

        return x, label_map_remapped, membership

    @staticmethod
    def _region_adjacency(label_map: torch.Tensor,
                          num_regions: int) -> torch.Tensor:
        """Return 4-neighbour region topology from an aligned compact map."""
        adjacency = torch.zeros(
            (num_regions, num_regions),
            dtype=torch.bool,
            device=label_map.device)
        edge_pairs = []
        if label_map.shape[1] > 1:
            edge_pairs.append(torch.stack((
                label_map[:, :-1].reshape(-1),
                label_map[:, 1:].reshape(-1)), dim=1))
        if label_map.shape[0] > 1:
            edge_pairs.append(torch.stack((
                label_map[:-1, :].reshape(-1),
                label_map[1:, :].reshape(-1)), dim=1))
        if not edge_pairs:
            return adjacency
        pairs = torch.cat(edge_pairs, dim=0)
        valid = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 1] >= 0)
            & (pairs[:, 0] != pairs[:, 1]))
        pairs = pairs[valid]
        if pairs.numel() > 0:
            adjacency[pairs[:, 0], pairs[:, 1]] = True
            adjacency[pairs[:, 1], pairs[:, 0]] = True
        return adjacency

    def _build_contrastive_targets(self, node_out, label_map, gt_bboxes,
                                   gt_labels=None, img_shape=None,
                                   region_membership=None):
        """Build high-purity per-GT targets and graph-local background rings.

        Region purity is intersection area divided by region area. Touching
        regions below ``minimum_positive_purity`` are ignored rather than
        turned into false negatives. The optional best-region fallback keeps
        contrast supervision for small lesions, but is excluded from BoxAlign
        so a low-purity prototype cannot directly steer bbox regression.
        """
        N = node_out.shape[0]
        H_f, W_f = label_map.shape
        device = node_out.device
        stats = {
            'gt_count': 0,
            'tb_count': 0,
            'total_tokens': N,
            'bbox_hit_ratio': 0.0,
            'tb_token_fraction': 0.0,
            'region_purity_mean': 0.0,
            'region_purity50': 0.0,
            'best_bbox_region_purity': 0.0,
            'ambiguous_count': 0,
            'low_purity_positive_count': 0,
            'boundary_positive_count': 0,
            'rejected_overlap_count': 0,
            'fallback_positive_gt_count': 0,
            'selection_purity_threshold': self.minimum_positive_purity,
            'single_region_gt_count': 0,
            'regions_per_gt_mean': 0.0,
            'gt_two_region_fraction': 0.0,
            'graph_ring_candidate_count': 0,
            'confirmed_negative_count': 0,
            'invalid_pixel_fraction': float(
                (label_map < 0).float().mean().item()),
        }
        if gt_bboxes is None or len(gt_bboxes) == 0:
            return None, stats

        if img_shape is not None:
            H_img, W_img = img_shape
            scale_y, scale_x = H_f / H_img, W_f / W_img
        else:
            scale_y, scale_x = 1.0, 1.0

        if region_membership is not None:
            region_membership = torch.as_tensor(
                region_membership, device=device, dtype=node_out.dtype)
            if tuple(region_membership.shape) != (N, H_f, W_f):
                raise ValueError(
                    'region_membership must have shape '
                    f'{(N, H_f, W_f)}, got '
                    f'{tuple(region_membership.shape)}')
            total_region_pixels = region_membership.sum(dim=(1, 2))
        else:
            valid_regions = label_map >= 0
            if valid_regions.any():
                total_region_pixels = torch.bincount(
                    label_map[valid_regions].long(), minlength=N).float()
            else:
                total_region_pixels = torch.zeros(N, device=device)

        labels = (
            torch.as_tensor(gt_labels, device=device).long().flatten()
            if gt_labels is not None else None)
        box_region_counts = []
        box_classes = []
        bbox_hit_count = 0

        for box_index, bbox in enumerate(gt_bboxes):
            # MMDetection boxes use xyxy in augmented image coordinates.
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            x1 = max(0, min(math.floor(x1 * scale_x), W_f))
            y1 = max(0, min(math.floor(y1 * scale_y), H_f))
            x2 = max(0, min(math.ceil(x2 * scale_x), W_f))
            y2 = max(0, min(math.ceil(y2 * scale_y), H_f))
            if x2 <= x1 or y2 <= y1:
                continue

            if region_membership is not None:
                patch_counts = region_membership[
                    :, y1:y2, x1:x2].sum(dim=(1, 2))
            else:
                patch = label_map[y1:y2, x1:x2]
                patch_valid = patch >= 0
                patch_counts = (
                    torch.bincount(
                        patch[patch_valid].long(), minlength=N).float()
                    if patch_valid.any() else torch.zeros(N, device=device))
            if patch_counts.sum() > 0:
                bbox_hit_count += 1
            box_region_counts.append(patch_counts)
            if labels is not None and box_index < labels.numel():
                box_classes.append(labels[box_index])
            else:
                box_classes.append(torch.zeros(
                    (), dtype=torch.long, device=device))

        if not box_region_counts:
            return None, stats

        box_region_counts = torch.stack(box_region_counts, dim=0)
        stats['gt_count'] = int(box_region_counts.shape[0])
        box_classes = torch.stack(box_classes).long()
        per_box_purity = (
            box_region_counts
            / total_region_pixels.clamp_min(1.0).unsqueeze(0))
        region_purity = per_box_purity.max(dim=0).values
        valid_token_mask = total_region_pixels > 0
        selection = _select_high_purity_region_targets(
            per_box_purity,
            valid_token_mask,
            minimum_positive_purity=self.minimum_positive_purity,
            ensure_gt_coverage=self.ensure_gt_positive)
        per_gt_positive = selection['per_gt_positive_mask']
        positive_mask = selection['positive_mask']
        ambiguous_overlap_mask = selection['ambiguous_overlap_mask']
        negative_mask = selection['negative_mask']
        fallback_gt_mask = selection['fallback_gt_mask']
        adjacency = self._region_adjacency(label_map, N)
        per_gt_ring_mask = []
        for gt_positive in per_gt_positive:
            one_hop = adjacency[gt_positive].any(dim=0)
            # Global negative prevents another lesion instance becoming a
            # localization negative for this GT.
            per_gt_ring_mask.append(one_hop & negative_mask)
        per_gt_ring_mask = torch.stack(per_gt_ring_mask, dim=0)
        ring_mask = per_gt_ring_mask.any(dim=0)

        selected_purity_by_gt = per_box_purity.masked_fill(
            ~per_gt_positive, -1.0)
        selected_purity, best_box_index = selected_purity_by_gt.max(dim=0)
        class_ids = torch.full(
            (N,), -1, dtype=torch.long, device=device)
        class_ids[positive_mask] = box_classes[
            best_box_index[positive_mask]]
        instance_ids = torch.full(
            (N,), -1, dtype=torch.long, device=device)
        instance_ids[positive_mask] = best_box_index[positive_mask]
        positive_weights = torch.zeros(N, device=device)
        positive_weights[positive_mask] = (
            self.purity_weight_floor
            + (1.0 - self.purity_weight_floor)
            * selected_purity[positive_mask])

        positive_count = int(positive_mask.sum().item())
        stats['tb_count'] = positive_count
        stats['bbox_hit_ratio'] = (
            bbox_hit_count / max(1, len(gt_bboxes)))
        stats['tb_token_fraction'] = positive_count / max(1, N)
        stats['ambiguous_count'] = int(
            (per_gt_positive.sum(dim=0) > 1).sum().item())
        stats['low_purity_positive_count'] = int(
            (
                positive_mask
                & (selected_purity <= self.positive_purity_threshold)
            ).sum().item())
        stats['boundary_positive_count'] = (
            stats['low_purity_positive_count'])
        stats['rejected_overlap_count'] = int(
            ambiguous_overlap_mask.sum().item())
        stats['fallback_positive_gt_count'] = int(
            fallback_gt_mask.sum().item())
        stats['confirmed_negative_count'] = int(
            negative_mask.sum().item())
        regions_per_gt = per_gt_positive.sum(dim=1)
        stats['single_region_gt_count'] = int(
            (regions_per_gt == 1).sum().item())
        stats['regions_per_gt_mean'] = float(
            regions_per_gt.float().mean().item())
        stats['gt_two_region_fraction'] = float(
            (regions_per_gt >= 2).float().mean().item())
        stats['graph_ring_candidate_count'] = int(
            per_gt_ring_mask.sum().item())
        if positive_count > 0:
            selected_positive_purity = selected_purity[positive_mask]
            stats['region_purity_mean'] = (
                selected_positive_purity.mean().item())
            stats['region_purity50'] = (
                (selected_positive_purity >= 0.5).float().mean().item())
        box_has_region = box_region_counts.sum(dim=1) > 0
        if box_has_region.any():
            stats['best_bbox_region_purity'] = (
                per_box_purity[box_has_region].amax(dim=1).mean().item())

        return {
            'positive_mask': positive_mask,
            'negative_mask': negative_mask,
            'ambiguous_overlap_mask': ambiguous_overlap_mask,
            'ring_mask': ring_mask,
            'per_gt_ring_mask': per_gt_ring_mask,
            'per_gt_positive_mask': per_gt_positive,
            'fallback_gt_mask': fallback_gt_mask,
            'box_region_counts': box_region_counts,
            'per_box_purity': per_box_purity,
            'class_ids': class_ids,
            'instance_ids': instance_ids,
            'purity': region_purity,
            'positive_weights': positive_weights,
        }, stats

    def _select_hard_negatives(self, projected_tokens, targets,
                               gt_index, prototype):
        """Select one-hop NAM neighbours first, then feature-hard background."""
        negative_indices = targets['negative_mask'].nonzero(
            as_tuple=False).squeeze(-1)
        positive_count = int(
            targets['per_gt_positive_mask'][gt_index].sum().item())
        if positive_count == 0 or negative_indices.numel() == 0:
            return (
                projected_tokens[:0],
                projected_tokens.new_zeros((0,)),
                0.0,
                0)

        negative_similarity = (
            projected_tokens[negative_indices]
            @ prototype.detach())
        ring_candidates = targets['per_gt_ring_mask'][
            gt_index, negative_indices]

        negative_budget = max(
            self.min_hard_negatives,
            int(math.ceil(
                self.hard_negative_ratio * positive_count)))
        negative_budget = min(negative_budget, negative_indices.numel())

        # Cosine lies in [-1, 1], so +2 strictly prioritizes the graph-local
        # boundary. Remaining slots are feature-hard background fallbacks.
        selection_score = (
            negative_similarity + 2.0 * ring_candidates.float())
        selected_local = torch.topk(
            selection_score, k=negative_budget, largest=True).indices
        selected_indices = negative_indices[selected_local]
        selected_similarity = negative_similarity[selected_local]
        selected_ring = ring_candidates[selected_local]
        selected_weights = torch.where(
            selected_ring,
            torch.ones_like(selected_similarity),
            torch.full_like(
                selected_similarity, self.far_negative_weight))
        selected_ring_count = int(
            selected_ring.sum().item())
        return (
            projected_tokens[selected_indices],
            selected_weights,
            float(selected_similarity.detach().mean().item()),
            selected_ring_count)

    def _prototype_region_loss(self, projected_tokens, targets):
        """Per-GT soft region-to-prototype contrast.

        A GT prototype is weighted by intersection mass, while each touching
        region is weighted by its inside fraction. Other GT instances are not
        placed in the denominator. This preserves same-class TB semantics and
        makes a one-region GT trainable through graph-background separation.
        """
        losses = []
        loss_weights = []
        records = []
        positive_similarities = []
        negative_similarities = []
        selected_count = 0
        selected_ring_count = 0
        valid_gt_count = 0
        rank_loss_sum = 0.0

        for gt_index in range(targets['box_region_counts'].shape[0]):
            positive_indices = targets['per_gt_positive_mask'][
                gt_index].nonzero(
                as_tuple=False).squeeze(-1)
            if positive_indices.numel() == 0:
                continue
            prototype_weights = targets['per_box_purity'][
                gt_index, positive_indices].clamp_min(1e-4)
            prototype = F.normalize(
                (
                    projected_tokens[positive_indices]
                    * prototype_weights.unsqueeze(1)
                ).sum(dim=0),
                dim=0)
            negatives, negative_weights, hard_similarity, ring_count = (
                self._select_hard_negatives(
                    projected_tokens, targets, gt_index, prototype))
            records.append({
                'gt_index': gt_index,
                'prototype': prototype,
                'is_fallback': bool(
                    targets['fallback_gt_mask'][gt_index].item()),
                'negative_tokens': negatives,
                'negative_weights': negative_weights,
                'positive_region_count': int(positive_indices.numel()),
                'boundary_region_count': int(
                    (
                        targets['purity'][positive_indices]
                        < self.positive_purity_threshold
                    ).sum().item()),
            })
            if negatives.numel() == 0:
                continue

            positive_tokens = projected_tokens[positive_indices]
            detached_prototype = prototype.detach()
            positive_logits = (
                positive_tokens @ detached_prototype
            ).unsqueeze(1) / self.contrast_temperature
            negative_logits = (
                positive_tokens @ negatives.T
            ) / self.contrast_temperature
            negative_logits = (
                negative_logits
                + negative_weights.clamp_min(1e-6).log().unsqueeze(0))
            logits = torch.cat((positive_logits, negative_logits), dim=1)
            target = torch.zeros(
                positive_indices.numel(),
                dtype=torch.long,
                device=projected_tokens.device)
            per_region_loss = F.cross_entropy(
                logits, target, reduction='none')
            prototype_positive_similarity = (
                positive_tokens @ detached_prototype)
            prototype_negative_similarity = (
                negatives @ detached_prototype)
            rank_loss = F.relu(
                self.boundary_rank_margin
                + prototype_negative_similarity.max()
                - prototype_positive_similarity)
            per_region_loss = (
                per_region_loss
                + self.boundary_rank_weight * rank_loss)
            region_weights = targets['positive_weights'][
                positive_indices].clamp_min(1e-4)
            losses.append((per_region_loss * region_weights).sum())
            loss_weights.append(region_weights.sum())

            positive_similarities.append(
                (positive_tokens.detach()
                 @ detached_prototype).mean())
            negative_similarities.append(
                negatives.detach().mean(dim=0) @ detached_prototype)
            selected_count += int(negatives.shape[0])
            selected_ring_count += int(ring_count)
            valid_gt_count += 1
            rank_loss_sum += float(
                (rank_loss.detach() * region_weights).sum().item()
                / region_weights.sum().clamp_min(1e-8).item())

        if losses:
            loss = torch.stack(losses).sum() / torch.stack(
                loss_weights).sum().clamp_min(1e-8)
        else:
            loss = projected_tokens.sum() * 0.0
        stats = {
            'valid_gt_count': valid_gt_count,
            'selected_negative_count': selected_count,
            'selected_ring_count': selected_ring_count,
            'positive_similarity': float(
                torch.stack(positive_similarities).mean().item())
            if positive_similarities else 0.0,
            'negative_similarity': float(
                torch.stack(negative_similarities).mean().item())
            if negative_similarities else 0.0,
            'boundary_rank_loss': (
                rank_loss_sum / valid_gt_count
                if valid_gt_count > 0 else 0.0),
        }
        return loss, records, stats

    def contrastive_loss(self, node_out, label_map, gt_bboxes,
                         gt_labels=None, img_shape=None):
        """Single-image diagnostic wrapper around the instance contrast loss."""
        targets, stats = self._build_contrastive_targets(
            node_out, label_map, gt_bboxes,
            gt_labels=gt_labels, img_shape=img_shape)
        self._dbg_tb_count = stats['tb_count']
        self._dbg_total_tokens = stats['total_tokens']
        self._dbg_bbox_hit_ratio = stats['bbox_hit_ratio']
        self._dbg_tb_token_fraction = stats['tb_token_fraction']
        self._dbg_region_purity_mean = stats['region_purity_mean']
        self._dbg_region_purity50 = stats['region_purity50']
        self._dbg_best_bbox_region_purity = (
            stats['best_bbox_region_purity'])
        self._dbg_ambiguous_count = stats['ambiguous_count']
        self._dbg_rejected_overlap_count = (
            stats['rejected_overlap_count'])
        self._dbg_fallback_positive_gt_count = (
            stats['fallback_positive_gt_count'])
        self._dbg_selection_purity_threshold = (
            stats['selection_purity_threshold'])
        self._dbg_confirmed_negative_count = (
            stats['confirmed_negative_count'])
        self._dbg_invalid_pixel_fraction = (
            stats['invalid_pixel_fraction'])
        if targets is None:
            return node_out.sum() * 0.0

        projected = F.normalize(
            self.contrast_head(node_out), dim=1)
        loss, _, proto_stats = self._prototype_region_loss(
            projected, targets)
        self._dbg_selected_negative_count = (
            proto_stats['selected_negative_count'])
        self._dbg_selected_ring_count = (
            proto_stats['selected_ring_count'])
        self._dbg_hard_negative_similarity = (
            proto_stats['negative_similarity'])
        self._last_instance_sim = proto_stats['positive_similarity']
        positive_indices = targets['positive_mask'].nonzero(
            as_tuple=False).squeeze(-1)
        if positive_indices.numel() > 0:
            self._dbg_positive_weight_mean = float(
                targets['positive_weights'][
                    positive_indices].mean().item())
        return loss

    def forward(self, fpn_p2_list: list,
                nam_graphs: list,
                gt_bboxes: list = None,
                gt_labels: list = None,
                img_shapes: list = None) -> torch.Tensor:
        """Compute per-GT soft region contrast and cache BoxAlign records."""
        B = len(nam_graphs) if nam_graphs is not None else 0
        device = self.final_norm.weight.device
        self._dbg_tb_count = 0
        self._dbg_total_gt_count = 0
        self._dbg_total_tokens = 0
        self._dbg_bbox_hit_ratio = 0.0
        self._dbg_tb_token_fraction = 0.0
        self._dbg_region_purity_mean = 0.0
        self._dbg_region_purity50 = 0.0
        self._dbg_best_bbox_region_purity = 0.0
        self._dbg_ambiguous_count = 0
        self._dbg_confirmed_negative_count = 0
        self._dbg_invalid_pixel_fraction = 0.0
        self._dbg_contrast_valid_samples = 0
        self._dbg_contrast_seen_samples = 0
        self._dbg_selected_negative_count = 0
        self._dbg_selected_ring_count = 0
        self._dbg_positive_weight_mean = 0.0
        self._dbg_hard_negative_similarity = 0.0
        self._dbg_class_counts = [0, 0]
        self._dbg_valid_instance_anchors = 0
        self._dbg_valid_instance_count = 0
        self._dbg_boundary_positive_count = 0
        self._dbg_rejected_overlap_count = 0
        self._dbg_fallback_positive_gt_count = 0
        self._dbg_selection_purity_threshold = (
            self.minimum_positive_purity)
        self._dbg_single_region_gt_count = 0
        self._dbg_regions_per_gt_mean = 0.0
        self._dbg_gt_two_region_fraction = 0.0
        self._dbg_graph_ring_candidate_count = 0
        self._dbg_region_positive_similarity = 0.0
        self._dbg_region_negative_similarity = 0.0
        self._dbg_boundary_rank_loss = 0.0
        self._last_boxalign_records = [None] * B

        node_outs, label_maps, memberships = [], [], []
        for b in range(B):
            if nam_graphs[b] is None or fpn_p2_list[b] is None:
                node_outs.append(None)
                label_maps.append(None)
                memberships.append(None)
            else:
                graph = {
                    key: value.to(device)
                    if isinstance(value, torch.Tensor) else value
                    for key, value in nam_graphs[b].items()}
                node_out, label_map, membership = self._process_single_sample(
                    fpn_p2_list[b].to(device), graph)
                node_outs.append(node_out)
                label_maps.append(label_map)
                memberships.append(membership)

        contr_loss = torch.zeros((), device=device)
        if self.training and gt_bboxes is not None:
            sample_losses = []
            debug_records = []
            selected_negative_count = 0
            selected_ring_count = 0
            positive_similarity_sum = 0.0
            negative_similarity_sum = 0.0
            valid_gt_count = 0
            boundary_rank_sum = 0.0
            all_positive_weights = []
            all_positive_classes = []

            for b in range(B):
                if node_outs[b] is None or label_maps[b] is None:
                    continue
                bboxes = gt_bboxes[b] if b < len(gt_bboxes) else None
                labels = (
                    gt_labels[b] if gt_labels is not None
                    and b < len(gt_labels) else None)
                img_shape = (
                    img_shapes[b] if img_shapes is not None
                    and b < len(img_shapes) else None)
                targets, stats = self._build_contrastive_targets(
                    node_outs[b], label_maps[b], bboxes,
                    gt_labels=labels, img_shape=img_shape,
                    region_membership=memberships[b])
                debug_records.append(stats)
                if targets is None:
                    continue

                projected = F.normalize(
                    self.contrast_head(node_outs[b]), dim=-1)
                positive_mask = targets['positive_mask']
                all_positive_weights.append(
                    targets['positive_weights'][positive_mask])
                all_positive_classes.append(
                    targets['class_ids'][positive_mask])
                sample_loss, boxalign_records, proto_stats = (
                    self._prototype_region_loss(projected, targets))
                self._last_boxalign_records[b] = boxalign_records
                if proto_stats['valid_gt_count'] > 0:
                    sample_losses.append(sample_loss)
                    self._dbg_contrast_valid_samples += 1
                selected_negative_count += (
                    proto_stats['selected_negative_count'])
                selected_ring_count += (
                    proto_stats['selected_ring_count'])
                positive_similarity_sum += (
                    proto_stats['positive_similarity']
                    * proto_stats['valid_gt_count'])
                negative_similarity_sum += (
                    proto_stats['negative_similarity']
                    * proto_stats['valid_gt_count'])
                valid_gt_count += proto_stats['valid_gt_count']
                boundary_rank_sum += (
                    proto_stats['boundary_rank_loss']
                    * proto_stats['valid_gt_count'])

            if sample_losses:
                contr_loss = torch.stack(sample_losses).mean()

            if debug_records:
                count = len(debug_records)
                self._dbg_tb_count = sum(
                    record['tb_count'] for record in debug_records)
                self._dbg_total_gt_count = sum(
                    record['gt_count'] for record in debug_records)
                self._dbg_total_tokens = sum(
                    record['total_tokens'] for record in debug_records)
                self._dbg_ambiguous_count = sum(
                    record['ambiguous_count'] for record in debug_records)
                self._dbg_rejected_overlap_count = sum(
                    record['rejected_overlap_count']
                    for record in debug_records)
                self._dbg_fallback_positive_gt_count = sum(
                    record['fallback_positive_gt_count']
                    for record in debug_records)
                self._dbg_selection_purity_threshold = (
                    self.minimum_positive_purity)
                self._dbg_confirmed_negative_count = sum(
                    record['confirmed_negative_count']
                    for record in debug_records)
                self._dbg_boundary_positive_count = sum(
                    record['boundary_positive_count']
                    for record in debug_records)
                self._dbg_single_region_gt_count = sum(
                    record['single_region_gt_count']
                    for record in debug_records)
                self._dbg_graph_ring_candidate_count = sum(
                    record['graph_ring_candidate_count']
                    for record in debug_records)
                for key in (
                        'bbox_hit_ratio', 'tb_token_fraction',
                        'region_purity_mean', 'region_purity50',
                        'best_bbox_region_purity',
                        'invalid_pixel_fraction',
                        'regions_per_gt_mean',
                        'gt_two_region_fraction'):
                    setattr(
                        self, f'_dbg_{key}',
                        sum(record[key] for record in debug_records) / count)
                self._dbg_contrast_seen_samples = count

            self._dbg_selected_negative_count = selected_negative_count
            self._dbg_selected_ring_count = selected_ring_count
            self._dbg_valid_instance_count = int(valid_gt_count)
            self._dbg_valid_instance_anchors = int(self._dbg_tb_count)
            if valid_gt_count > 0:
                self._dbg_region_positive_similarity = (
                    positive_similarity_sum / valid_gt_count)
                self._dbg_region_negative_similarity = (
                    negative_similarity_sum / valid_gt_count)
                self._dbg_boundary_rank_loss = (
                    boundary_rank_sum / valid_gt_count)
                self._last_instance_sim = (
                    self._dbg_region_positive_similarity)
                self._dbg_hard_negative_similarity = (
                    self._dbg_region_negative_similarity)
            if all_positive_weights:
                self._dbg_positive_weight_mean = float(
                    torch.cat(
                        all_positive_weights).detach().mean().item())
                all_classes = torch.cat(all_positive_classes)
                self._dbg_class_counts = [
                    int((all_classes == class_id).sum().item())
                    for class_id in range(2)]

        return contr_loss


# ============================================================================
# RetinaNAMGuideAttHeadV3: Dual-branch Head (SymFormer + Region Transformer v3.3)
# ============================================================================
if __name__ in (
        'NAMLab_att_v3_dropin',
        'NAMLab_att_v320_dropin',
        'NAMLab_att_v321_dropin',
        'mmdet.models.utils.NAMLab_att_v3',
        'mmdet.models.utils.NAMLab_att_v3_dropin',
        'mmdet.models.utils.NAMLab_att_v320_dropin',
        'mmdet.models.utils.NAMLab_att_v321_dropin'):

    import sys as _sys
    import mmcv as _mmcv
    import numpy as _np

    try:
        from ...datasets.builder import PIPELINES
    except ImportError:
        from mmdet.datasets.builder import PIPELINES

    @PIPELINES.register_module(force=True)
    class LoadLungMask:
        """Load an offline lung/heart mask into image metadata.

        The source-resolution mask is not registered as a semantic target and
        is therefore not used to replace the detector image. Geometry is
        applied later from ``img_metas`` by
        :func:`_align_lung_mask_to_image_meta`.
        """

        def __init__(self,
                     mask_dir,
                     key='lung_mask',
                     extensions=('.png', '.jpg', '.jpeg', '.bmp',
                                 '.tif', '.tiff'),
                     recursive=True,
                     strict=True,
                     invert=False):
            self.mask_dir = os.path.abspath(os.path.expanduser(mask_dir))
            self.key = key
            self.extensions = tuple(
                ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                for ext in extensions)
            self.recursive = bool(recursive)
            self.strict = bool(strict)
            self.invert = bool(invert)
            self._stem_index = None

        def _build_index(self):
            if self._stem_index is not None:
                return self._stem_index
            index = {}
            if os.path.isdir(self.mask_dir):
                walker = (
                    os.walk(self.mask_dir)
                    if self.recursive
                    else [(self.mask_dir, [], os.listdir(self.mask_dir))])
                for directory, _, names in walker:
                    for name in names:
                        stem, extension = os.path.splitext(name)
                        if extension.lower() not in self.extensions:
                            continue
                        index.setdefault(stem, []).append(
                            os.path.join(directory, name))
            self._stem_index = index
            return index

        def _resolve_mask_path(self, img_path):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            parent = os.path.basename(os.path.dirname(img_path))
            direct_candidates = []
            for extension in self.extensions:
                direct_candidates.extend([
                    os.path.join(self.mask_dir, f'{stem}{extension}'),
                    os.path.join(
                        self.mask_dir, parent, f'{stem}{extension}'),
                ])
            for candidate in direct_candidates:
                if os.path.isfile(candidate):
                    return candidate

            matches = self._build_index().get(stem, [])
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                parent_matches = [
                    path for path in matches
                    if os.path.basename(os.path.dirname(path)) == parent]
                if len(parent_matches) == 1:
                    return parent_matches[0]
                raise RuntimeError(
                    f'Ambiguous lung masks for image {img_path}: {matches}')
            return None

        def __call__(self, results):
            img_path = results.get('filename', '')
            if not img_path:
                raise KeyError(
                    'LoadLungMask requires LoadImageFromFile to run first')
            mask_path = self._resolve_mask_path(img_path)
            if mask_path is None:
                if self.strict:
                    raise FileNotFoundError(
                        f'No lung mask found for {img_path} below '
                        f'{self.mask_dir}')
                results[self.key] = None
                results[f'{self.key}_path'] = ''
                return results

            mask = _mmcv.imread(mask_path, flag='grayscale')
            if mask is None:
                raise OSError(f'Failed to read lung mask: {mask_path}')
            if self.invert:
                mask = 255 - mask
            mask = (mask > 0).astype(_np.uint8)
            foreground_fraction = float(mask.mean())
            if self.strict and not 0.01 < foreground_fraction < 0.95:
                raise ValueError(
                    f'Implausible lung-mask foreground fraction '
                    f'{foreground_fraction:.4f} in {mask_path}; check '
                    f'whether invert={self.invert} is correct')

            image_shape = tuple(results['img'].shape[:2])
            mask_shape = tuple(mask.shape[:2])
            image_ratio = image_shape[1] / max(image_shape[0], 1)
            mask_ratio = mask_shape[1] / max(mask_shape[0], 1)
            if (self.strict
                    and abs(image_ratio - mask_ratio)
                    / max(image_ratio, 1e-6) > 0.02):
                raise ValueError(
                    f'Lung mask {mask_path} has aspect ratio '
                    f'{mask_ratio:.4f}, but image {img_path} has '
                    f'{image_ratio:.4f}')

            results[self.key] = mask
            results[f'{self.key}_path'] = mask_path
            return results

        def __repr__(self):
            return (
                f'{self.__class__.__name__}('
                f'mask_dir={self.mask_dir!r}, key={self.key!r}, '
                f'recursive={self.recursive}, strict={self.strict}, '
                f'invert={self.invert})')

    @PIPELINES.register_module(force=True)
    class LoadBoneSuppressedCXR:
        """Load a paired bone-suppressed CXR for the PC branch only.

        The image is kept at source resolution in ``img_metas``. Detector-side
        alignment later replays the main image's Resize/Flip/Pad geometry. A
        strict missing-file policy is intentional: silently falling back to
        the raw CXR would mix two PC input domains in the same experiment.
        """

        def __init__(self,
                     image_dir,
                     key='bone_suppressed_img',
                     suffix='',
                     extensions=('.png', '.jpg', '.jpeg', '.bmp',
                                 '.tif', '.tiff'),
                     recursive=True,
                     strict=True,
                     invert=False):
            self.image_dir = os.path.abspath(os.path.expanduser(image_dir))
            self.key = key
            self.suffix = str(suffix)
            self.extensions = tuple(
                ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                for ext in extensions)
            self.recursive = bool(recursive)
            self.strict = bool(strict)
            self.invert = bool(invert)
            self._stem_index = None

        def _build_index(self):
            if self._stem_index is not None:
                return self._stem_index
            index = {}
            if os.path.isdir(self.image_dir):
                walker = (
                    os.walk(self.image_dir)
                    if self.recursive
                    else [(self.image_dir, [], os.listdir(self.image_dir))])
                for directory, _, names in walker:
                    for name in names:
                        stem, extension = os.path.splitext(name)
                        if extension.lower() not in self.extensions:
                            continue
                        index.setdefault(stem, []).append(
                            os.path.join(directory, name))
            self._stem_index = index
            return index

        def _resolve_image_path(self, img_path):
            raw_stem = os.path.splitext(os.path.basename(img_path))[0]
            target_stem = f'{raw_stem}{self.suffix}'
            parent = os.path.basename(os.path.dirname(img_path))
            direct_candidates = []
            for extension in self.extensions:
                direct_candidates.extend([
                    os.path.join(
                        self.image_dir, f'{target_stem}{extension}'),
                    os.path.join(
                        self.image_dir, parent,
                        f'{target_stem}{extension}'),
                ])
            for candidate in direct_candidates:
                if os.path.isfile(candidate):
                    return candidate

            matches = self._build_index().get(target_stem, [])
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                parent_matches = [
                    path for path in matches
                    if os.path.basename(os.path.dirname(path)) == parent]
                if len(parent_matches) == 1:
                    return parent_matches[0]
                raise RuntimeError(
                    f'Ambiguous bone-suppressed CXRs for image '
                    f'{img_path}: {matches}')
            return None

        def __call__(self, results):
            img_path = results.get('filename', '')
            if not img_path:
                raise KeyError(
                    'LoadBoneSuppressedCXR requires LoadImageFromFile first')
            bone_path = self._resolve_image_path(img_path)
            if bone_path is None:
                if self.strict:
                    raise FileNotFoundError(
                        f'No bone-suppressed CXR found for {img_path} below '
                        f'{self.image_dir}; expected filename stem '
                        f'{os.path.splitext(os.path.basename(img_path))[0]}'
                        f'{self.suffix}')
                results[self.key] = None
                results[f'{self.key}_path'] = ''
                return results

            bone_image = _mmcv.imread(bone_path, flag='grayscale')
            if bone_image is None:
                raise OSError(
                    f'Failed to read bone-suppressed CXR: {bone_path}')
            if self.invert:
                bone_image = 255 - bone_image

            image_shape = tuple(results['img'].shape[:2])
            bone_shape = tuple(bone_image.shape[:2])
            image_ratio = image_shape[1] / max(image_shape[0], 1)
            bone_ratio = bone_shape[1] / max(bone_shape[0], 1)
            if (self.strict
                    and abs(image_ratio - bone_ratio)
                    / max(image_ratio, 1e-6) > 0.02):
                raise ValueError(
                    f'Bone-suppressed CXR {bone_path} has aspect ratio '
                    f'{bone_ratio:.4f}, but image {img_path} has '
                    f'{image_ratio:.4f}')
            intensity_std = float(bone_image.astype(_np.float32).std())
            if self.strict and intensity_std < 1.0:
                raise ValueError(
                    f'Bone-suppressed CXR {bone_path} is nearly constant '
                    f'(std={intensity_std:.4f})')

            results[self.key] = bone_image
            results[f'{self.key}_path'] = bone_path
            return results

        def __repr__(self):
            return (
                f'{self.__class__.__name__}('
                f'image_dir={self.image_dir!r}, key={self.key!r}, '
                f'suffix={self.suffix!r}, recursive={self.recursive}, '
                f'strict={self.strict}, invert={self.invert})')

    @PIPELINES.register_module(force=True)
    class AdaptiveCXRPreprocess:
        """CDF ROI cropping plus histogram standardization for CXR detection.

        The numerical method follows Section 3.2.3 of:
        Singthongchai and Wangkhamhan, "Adaptive Normalization Enhances the
        Generalization of Deep Learning Model in Chest X-Ray Classification",
        Journal of Imaging 12(1):14, 2026, doi:10.3390/jimaging12010014.

        Detection-specific additions are deliberately geometric only:
        bounding boxes, the lung mask, and the NAM label map receive the exact
        same crop. The lung mask and GT boxes can veto a crop that would remove
        supervised anatomy, but they never define or tighten the CDF crop.
        """

        def __init__(self,
                     x_percentiles=(0.05, 0.95),
                     y_percentiles=(0.15, 0.95),
                     target_mean=121.8,
                     target_std=57.1,
                     lung_mask_key='lung_mask',
                     nam_graph_key='nam_graph',
                     min_lung_coverage=0.98,
                     min_bbox_coverage=0.98,
                     force_grayscale=True,
                     safety_fallback=True):
            self.x_percentiles = tuple(float(v) for v in x_percentiles)
            self.y_percentiles = tuple(float(v) for v in y_percentiles)
            self.target_mean = float(target_mean)
            self.target_std = float(target_std)
            self.lung_mask_key = lung_mask_key
            self.nam_graph_key = nam_graph_key
            self.min_lung_coverage = float(min_lung_coverage)
            self.min_bbox_coverage = float(min_bbox_coverage)
            self.force_grayscale = bool(force_grayscale)
            self.safety_fallback = bool(safety_fallback)
            if len(self.x_percentiles) != 2 or len(
                    self.y_percentiles) != 2:
                raise ValueError('x/y_percentiles must each contain two values')
            if not (0.0 <= self.min_lung_coverage <= 1.0):
                raise ValueError(
                    'min_lung_coverage must lie in [0, 1]')
            if not (0.0 <= self.min_bbox_coverage <= 1.0):
                raise ValueError(
                    'min_bbox_coverage must lie in [0, 1]')

        @staticmethod
        def _minimum_bbox_coverage(results, crop_bbox):
            values = []
            for key in results.get('bbox_fields', []):
                boxes = results.get(key)
                if boxes is None or len(boxes) == 0:
                    continue
                coverage = _bbox_crop_coverage(
                    torch.as_tensor(boxes).float(), crop_bbox)
                if coverage.numel() > 0:
                    values.append(float(coverage.min().item()))
            return min(values) if values else 1.0

        def __call__(self, results):
            if 'img' not in results:
                raise KeyError(
                    'AdaptiveCXRPreprocess requires LoadImageFromFile first')
            image = results['img']
            original_shape = tuple(int(v) for v in image.shape)
            image_h, image_w = original_shape[:2]
            proposed = _compute_cdf_roi_geometry(
                image,
                x_percentiles=self.x_percentiles,
                y_percentiles=self.y_percentiles)
            proposed_bbox = proposed['crop_bbox']

            aligned_lung_mask = None
            lung_coverage = 1.0
            lung_mask = results.get(self.lung_mask_key)
            if lung_mask is not None:
                aligned_lung_mask = _resize_binary_mask(
                    lung_mask, (image_h, image_w))
                x1, y1, x2, y2 = proposed_bbox
                lung_total = aligned_lung_mask.sum().clamp_min(1)
                lung_inside = aligned_lung_mask[y1:y2, x1:x2].sum()
                lung_coverage = float(
                    (lung_inside.float() / lung_total.float()).item())

            bbox_coverage = self._minimum_bbox_coverage(
                results, proposed_bbox)
            fallback_reasons = []
            if (aligned_lung_mask is not None
                    and lung_coverage < self.min_lung_coverage):
                fallback_reasons.append('lung')
            if bbox_coverage < self.min_bbox_coverage:
                fallback_reasons.append('bbox')
            use_fallback = bool(fallback_reasons and self.safety_fallback)
            crop_bbox = (
                (0, 0, image_w, image_h)
                if use_fallback else proposed_bbox)
            x1, y1, x2, y2 = crop_bbox

            cropped_image = image[y1:y2, x1:x2]
            if self.force_grayscale:
                gray = _to_grayscale_cxr(cropped_image)
                if cropped_image.ndim == 3:
                    cropped_image = gray[..., None].repeat(1, 1, 3)
                else:
                    cropped_image = gray
            standardized, intensity_stats = _standardize_cxr_intensity(
                cropped_image,
                target_mean=self.target_mean,
                target_std=self.target_std)
            results['img'] = _np.ascontiguousarray(
                standardized.cpu().numpy().astype(_np.float32))
            results['img_shape'] = results['img'].shape
            results['pad_shape'] = results['img'].shape

            # Keep source annotation dtypes while translating/clipping boxes.
            for key in results.get('bbox_fields', []):
                boxes = results.get(key)
                if boxes is None:
                    continue
                source_dtype = boxes.dtype
                boxes_tensor = torch.as_tensor(boxes).float()
                roi_boxes = _boxes_to_roi(boxes_tensor, crop_bbox)
                valid = (
                    (roi_boxes[:, 2] > roi_boxes[:, 0])
                    & (roi_boxes[:, 3] > roi_boxes[:, 1]))
                results[key] = (
                    roi_boxes[valid].cpu().numpy().astype(
                        source_dtype, copy=False))
                if key == 'gt_bboxes' and 'gt_labels' in results:
                    results['gt_labels'] = results['gt_labels'][
                        valid.cpu().numpy()]

            if results.get('mask_fields'):
                raise NotImplementedError(
                    'AdaptiveCXRPreprocess currently supports bbox detection '
                    'only; instance masks require a synchronized crop.')

            if aligned_lung_mask is not None:
                results[self.lung_mask_key] = (
                    _crop_2d(aligned_lung_mask, crop_bbox)
                    .to(dtype=torch.uint8).cpu().numpy())

            graph_data = results.get(self.nam_graph_key)
            if graph_data is not None and 'label_map' in graph_data:
                label_map = torch.as_tensor(
                    graph_data['label_map']).long()
                if tuple(label_map.shape) != (image_h, image_w):
                    label_map = F.interpolate(
                        label_map[None, None].float(),
                        size=(image_h, image_w),
                        mode='nearest')[0, 0].long()
                cropped_graph = {
                    key: value for key, value in graph_data.items()}
                cropped_graph['label_map'] = _crop_2d(
                    label_map, crop_bbox).contiguous()
                cropped_graph['roi_cropped'] = True
                cropped_graph['pre_roi_label_shape'] = tuple(
                    label_map.shape)
                results[self.nam_graph_key] = cropped_graph

            crop_h, crop_w = y2 - y1, x2 - x1
            roi_applied = crop_bbox != (0, 0, image_w, image_h)
            results.update({
                'roi_applied': roi_applied,
                'roi_crop_bbox': tuple(int(v) for v in crop_bbox),
                'roi_cdf_bbox': tuple(int(v) for v in proposed_bbox),
                'roi_pad': (0, 0, 0, 0),
                'roi_original_shape': original_shape,
                'roi_retained_area': float(
                    crop_h * crop_w / max(image_h * image_w, 1)),
                'roi_linear_scale_gain': float(
                    max(image_h, image_w) / max(crop_h, crop_w, 1)),
                'roi_lung_coverage': lung_coverage,
                'roi_bbox_min_coverage': bbox_coverage,
                'roi_safety_fallback': use_fallback,
                'roi_fallback_reason': '+'.join(fallback_reasons),
                'roi_intensity_mean_before': (
                    intensity_stats['original_mean']),
                'roi_intensity_std_before': (
                    intensity_stats['original_std']),
                'roi_intensity_mean_after': (
                    intensity_stats['normalized_mean']),
                'roi_intensity_std_after': (
                    intensity_stats['normalized_std']),
                'roi_intensity_degenerate': (
                    intensity_stats['degenerate_std']),
                'roi_force_grayscale': self.force_grayscale,
                'adaptive_norm_source': (
                    'Singthongchai2026_JImaging_12_14'),
            })
            return results

        def __repr__(self):
            return (
                f'{self.__class__.__name__}('
                f'x_percentiles={self.x_percentiles}, '
                f'y_percentiles={self.y_percentiles}, '
                f'target_mean={self.target_mean}, '
                f'target_std={self.target_std}, '
                f'min_lung_coverage={self.min_lung_coverage}, '
                f'min_bbox_coverage={self.min_bbox_coverage}, '
                f'force_grayscale={self.force_grayscale}, '
                f'safety_fallback={self.safety_fallback})')

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

    # ---- HEADS / NECKS ----
    _heads_mod = _sys.modules.get('mmdet.models.builder')
    if (_heads_mod is not None
            and hasattr(_heads_mod, 'HEADS')
            and hasattr(_heads_mod, 'NECKS')):
        HEADS = _heads_mod.HEADS
        NECKS = _heads_mod.NECKS
    else:
        try:
            from ..builder import HEADS, NECKS
        except ImportError:
            from mmdet.models.builder import HEADS, NECKS

    @NECKS.register_module(force=True)
    class TBFusionCrossScaleNeck(nn.Module):
        """Four-level CrossScale neck that directly replaces FPN.

        Interface:
            inputs: C2/C3/C4/C5 backbone tensors
            returns: P2/P3/P4/P5 tensors with ``out_channels`` channels

        PC enhancement is deliberately outside this module and is applied only
        in the bbox regression tower, so this neck can be ablated independently
        against a four-level FPN.
        """

        def __init__(self,
                     in_channels=(72, 144, 288, 576),
                     out_channels: int = 256,
                     residual_init: float = 1e-3,
                     residual_max: float = 0.1,
                     semantic_residual_init: float = 5e-4,
                     semantic_residual_max: float = 0.02,
                     refine_residual_init: float = 1e-3,
                     refine_residual_max: float = 0.02,
                     window_size: int = 8,
                     use_top_down: bool = True,
                     learnable_fusion_weights: bool = False,
                     detail_fusion_init: float = 0.75,
                     semantic_fusion_init: float = 0.25,
                     freqfusion_top_down: bool = False,
                     freqfusion_compressed_channels: int = 64,
                     reg_detail_bypass: bool = False,
                     reg_detail_compressed_channels: int = 32):
            super().__init__()
            if len(in_channels) != 4:
                raise ValueError(
                    'TBFusionCrossScaleNeck requires four input levels')
            self.in_channels = list(in_channels)
            self.out_channels = out_channels
            self.use_reg_detail_bypass = bool(reg_detail_bypass)
            self.lateral_convs = nn.ModuleList([
                nn.Conv2d(in_ch, out_channels, 1)
                for in_ch in self.in_channels
            ])
            self.cross_scale = _TBFusionCrossScale(
                channels=out_channels,
                residual_init=residual_init,
                residual_max=residual_max,
                semantic_residual_init=semantic_residual_init,
                semantic_residual_max=semantic_residual_max,
                refine_residual_init=refine_residual_init,
                refine_residual_max=refine_residual_max,
                window_size=window_size,
                use_top_down=use_top_down,
                learnable_fusion_weights=learnable_fusion_weights,
                detail_fusion_init=detail_fusion_init,
                semantic_fusion_init=semantic_fusion_init,
                freqfusion_top_down=freqfusion_top_down,
                freqfusion_compressed_channels=(
                    freqfusion_compressed_channels))
            self.reg_detail_generator = (
                _C5GatedC2RegressionDetail(
                    channels=out_channels,
                    compressed_channels=(
                        reg_detail_compressed_channels))
                if self.use_reg_detail_bypass else None)
            self.regression_detail_feats = (None, None, None, None)

            for conv in self.lateral_convs:
                nn.init.xavier_uniform_(conv.weight)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)

        def forward(self, inputs):
            if len(inputs) != len(self.in_channels):
                raise ValueError(
                    f'expected {len(self.in_channels)} backbone levels, '
                    f'got {len(inputs)}')
            laterals = [
                conv(feat)
                for conv, feat in zip(self.lateral_convs, inputs)
            ]
            if self.use_reg_detail_bypass:
                detail_p4, detail_p5 = self.reg_detail_generator(
                    laterals[0], laterals[-1],
                    target_sizes=(
                        laterals[2].shape[-2:],
                        laterals[3].shape[-2:]))
                self.regression_detail_feats = (
                    None, None, detail_p4, detail_p5)
            else:
                self.regression_detail_feats = (
                    None, None, None, None)
            return tuple(self.cross_scale(laterals))

    @HEADS.register_module(force=True)
    class RetinaNAMGuideAttHeadV3(RetinaGuideAttHead):
        """Dual-branch RetinaNet Head v3.3:
        SymFormer + Dynamic Region Transformer + Cross-Scale Attention.

        Key improvements over v3:
          1. Region tokens pooled from raw C2 (gradient -> backbone)
          2. CrossScale is supplied by the configured neck
          3. PC+LBP reaches only the bbox regression tower

        Config:
            bbox_head=dict(
                type='RetinaNAMGuideAttHeadV3',
                d_model=256, nhead=8, num_layers=4, dropout=0.1,
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
                     positive_purity_threshold: float = 0.5,
                     negative_purity_threshold: float = 0.05,
                     minimum_positive_purity: float = math.pi / 4,
                     ensure_gt_positive: bool = True,
                     contrast_temperature: float = 0.07,
                     purity_weight_floor: float = 0.25,
                     hard_negative_ratio: float = 3.0,
                     hard_negative_expansion: float = 1.5,
                     min_hard_negatives: int = 16,
                     far_negative_weight: float = 0.25,
                     boundary_rank_margin: float = 0.10,
                     boundary_rank_weight: float = 0.0,
                     region_in_channels: int = 72,
                     boxalign: bool = False,
                     boxalign_dim: int = 128,
                     boxalign_temperature: float = 0.10,
                     boxalign_min_iou: float = 0.30,
                     boxalign_levels=(0, 1, 2),
                     boxalign_max_anchors_per_level: int = 2,
                     boxalign_loss_ratio: float = 0.5,
                     debug_interval: int = 50,
                     deep_debug_interval: int = 500,
                     cross_scale: bool = False,
                     pc_branch: bool = False,
                     head_pc_modulation: bool = False,
                     head_pc_residual_init: float = 1e-2,
                     head_pc_residual_max: float = 0.1,
                     bbox_contrastive: bool = False,
                     bbox_contrast_loss_weight: float = 0.01,
                     bbox_contrast_dim: int = 128,
                     bbox_contrast_temperature: float = 0.10,
                     bbox_contrast_pos_iou: float = 0.5,
                     bbox_contrast_neg_iou: float = 0.4,
                     bbox_contrast_max_pos_per_gt: int = 16,
                     bbox_contrast_hard_negative_ratio: float = 3.0,
                     bbox_contrast_min_negatives: int = 32,
                     reg_detail_bypass: bool = False,
                     reg_detail_levels=(2, 3),
                     reg_detail_residual_init: float = 1e-2,
                     reg_detail_residual_max: float = 0.1,
                     d_fine_style: bool = False,
                     d_fine_reg_max: int = 16,
                     d_fine_refine_init: float = 0.25,
                     d_fine_refine_max: float = 1.0,
                     d_fine_loss_weight: float = 0.25,
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
            self.v3_positive_purity_threshold = positive_purity_threshold
            self.v3_negative_purity_threshold = negative_purity_threshold
            self.v3_minimum_positive_purity = minimum_positive_purity
            self.v3_ensure_gt_positive = ensure_gt_positive
            self.v3_contrast_temperature = contrast_temperature
            self.v3_purity_weight_floor = purity_weight_floor
            self.v3_hard_negative_ratio = hard_negative_ratio
            self.v3_hard_negative_expansion = hard_negative_expansion
            self.v3_min_hard_negatives = min_hard_negatives
            self.v3_far_negative_weight = far_negative_weight
            self.v3_boundary_rank_margin = boundary_rank_margin
            self.v3_boundary_rank_weight = boundary_rank_weight
            self.region_in_channels = int(region_in_channels)
            if self.region_in_channels <= 0:
                raise ValueError('region_in_channels must be positive')
            self.use_boxalign = bool(boxalign)
            self.boxalign_dim = int(boxalign_dim)
            self.boxalign_temperature = float(boxalign_temperature)
            self.boxalign_min_iou = float(boxalign_min_iou)
            self.boxalign_levels = tuple(int(level) for level in boxalign_levels)
            self.boxalign_max_anchors_per_level = int(
                boxalign_max_anchors_per_level)
            self.boxalign_loss_ratio = float(boxalign_loss_ratio)
            if self.boxalign_dim <= 0:
                raise ValueError('boxalign_dim must be positive')
            if self.boxalign_dim != 128:
                raise ValueError(
                    'boxalign_dim must match the 128-D NAM contrast head')
            if self.boxalign_temperature <= 0:
                raise ValueError('boxalign_temperature must be positive')
            if not 0 <= self.boxalign_min_iou <= 1:
                raise ValueError('boxalign_min_iou must be in [0, 1]')
            if not self.boxalign_levels or min(self.boxalign_levels) < 0:
                raise ValueError('boxalign_levels must contain valid levels')
            if self.boxalign_max_anchors_per_level < 1:
                raise ValueError(
                    'boxalign_max_anchors_per_level must be positive')
            if self.boxalign_loss_ratio < 0:
                raise ValueError('boxalign_loss_ratio must be non-negative')
            self.debug_interval = int(debug_interval)
            self.deep_debug_interval = int(deep_debug_interval)
            if self.debug_interval < 0 or self.deep_debug_interval < 0:
                raise ValueError(
                    'debug intervals must be non-negative')
            self._debug_step = 1
            legacy_edge_supervision = bool(
                kwargs.pop('edge_supervision', False))
            kwargs.pop('edge_loss_weight', None)
            if legacy_edge_supervision:
                raise ValueError(
                    'edge_supervision was removed in NAMLab v3.9 because '
                    'its LBP-derived target duplicated PC texture cues')
            self.use_head_pc_modulation = bool(head_pc_modulation)
            self.head_pc_residual_init = float(head_pc_residual_init)
            self.head_pc_residual_max = float(head_pc_residual_max)
            if not (0.0 < self.head_pc_residual_init
                    < self.head_pc_residual_max):
                raise ValueError(
                    'head_pc_residual_init must satisfy 0 < init < max')
            self.use_bbox_contrastive = bool(bbox_contrastive)
            if self.use_bbox_contrastive and self.use_boxalign:
                raise ValueError(
                    'boxalign replaces the harmful anchor-anchor '
                    'bbox_contrastive objective; do not enable both')
            self.bbox_contrast_loss_weight = float(
                bbox_contrast_loss_weight)
            self.bbox_contrast_dim = int(bbox_contrast_dim)
            self.bbox_contrast_temperature = float(
                bbox_contrast_temperature)
            self.bbox_contrast_pos_iou = float(bbox_contrast_pos_iou)
            self.bbox_contrast_neg_iou = float(bbox_contrast_neg_iou)
            self.bbox_contrast_max_pos_per_gt = int(
                bbox_contrast_max_pos_per_gt)
            self.bbox_contrast_hard_negative_ratio = float(
                bbox_contrast_hard_negative_ratio)
            self.bbox_contrast_min_negatives = int(
                bbox_contrast_min_negatives)
            if self.bbox_contrast_loss_weight < 0:
                raise ValueError(
                    'bbox_contrast_loss_weight must be non-negative')
            if self.bbox_contrast_dim <= 0:
                raise ValueError('bbox_contrast_dim must be positive')
            if self.bbox_contrast_temperature <= 0:
                raise ValueError(
                    'bbox_contrast_temperature must be positive')
            if not (0.0 <= self.bbox_contrast_neg_iou
                    < self.bbox_contrast_pos_iou <= 1.0):
                raise ValueError(
                    'Expected 0 <= bbox_contrast_neg_iou < '
                    'bbox_contrast_pos_iou <= 1')
            if self.bbox_contrast_max_pos_per_gt < 2:
                raise ValueError(
                    'bbox_contrast_max_pos_per_gt must be at least 2')
            if self.bbox_contrast_hard_negative_ratio <= 0:
                raise ValueError(
                    'bbox_contrast_hard_negative_ratio must be positive')
            if self.bbox_contrast_min_negatives < 1:
                raise ValueError(
                    'bbox_contrast_min_negatives must be positive')
            self.use_reg_detail_bypass = bool(reg_detail_bypass)
            self.reg_detail_levels = tuple(
                int(level) for level in reg_detail_levels)
            self.reg_detail_residual_init = float(
                reg_detail_residual_init)
            self.reg_detail_residual_max = float(
                reg_detail_residual_max)
            if not 0.0 < self.reg_detail_residual_init \
                    < self.reg_detail_residual_max:
                raise ValueError(
                    'reg_detail_residual_init must satisfy 0 < init < max')
            # D-FINE-style anchor adapter. This intentionally implements only
            # fine-grained distribution refinement (FDR): GO-LSD requires
            # iterative DETR decoder layers and must not be approximated by
            # distilling across semantically different FPN levels.
            self.use_d_fine_style = bool(d_fine_style)
            self.d_fine_reg_max = int(d_fine_reg_max)
            self.d_fine_refine_init = float(d_fine_refine_init)
            self.d_fine_refine_max = float(d_fine_refine_max)
            self.d_fine_loss_weight = float(d_fine_loss_weight)
            if self.d_fine_reg_max < 2:
                raise ValueError('d_fine_reg_max must be at least 2')
            if not (0.0 < self.d_fine_refine_init
                    < self.d_fine_refine_max):
                raise ValueError(
                    'd_fine_refine_init must satisfy 0 < init < max')
            if self.d_fine_loss_weight < 0:
                raise ValueError(
                    'd_fine_loss_weight must be non-negative')
            self._last_d_fine_logits = None
            self._last_d_fine_base_preds = None
            self._dbg_d_fine_entropy = [0.0] * 4
            self._dbg_d_fine_loss_raw = 0.0
            self._dbg_d_fine_saturation = 0.0
            self._dbg_d_fine_target_abs = 0.0
            self._dbg_d_fine_expected_abs = [0.0] * 4
            self._last_reg_features = None
            # Deprecated compatibility arguments.  CrossScale now lives at
            # the neck seam. Detector-level pc_branch no longer injects C2.
            self._legacy_cross_scale = cross_scale
            self._legacy_pc_branch = pc_branch
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

            # --- Region Transformer branch (dynamic token pooling from C2) ---
            self.nam_branch = _RegionTransformerBranchV3_3(
                feat_channels=self.region_in_channels,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dropout=self.v3_dropout,
                drop_path_rate=self.v3_drop_path_rate,
                token_dropout=self.v3_token_dropout,
                spatial_temp=self.v3_spatial_temp,
                max_regions=self.v3_max_regions,
                ema_momentum=self.v3_ema_momentum,
                positive_purity_threshold=(
                    self.v3_positive_purity_threshold),
                negative_purity_threshold=(
                    self.v3_negative_purity_threshold),
                minimum_positive_purity=(
                    self.v3_minimum_positive_purity),
                ensure_gt_positive=self.v3_ensure_gt_positive,
                contrast_temperature=self.v3_contrast_temperature,
                purity_weight_floor=self.v3_purity_weight_floor,
                hard_negative_ratio=self.v3_hard_negative_ratio,
                hard_negative_expansion=(
                    self.v3_hard_negative_expansion),
                min_hard_negatives=self.v3_min_hard_negatives,
                far_negative_weight=self.v3_far_negative_weight,
                boundary_rank_margin=self.v3_boundary_rank_margin,
                boundary_rank_weight=self.v3_boundary_rank_weight)

            # PC is fused only into the bbox regression tower after SymFormer.
            # A bounded scale protects the pretrained detector at startup.
            if self.use_head_pc_modulation:
                self.lbp_branch = _PerPixelLBPBranch(
                    in_channels=576, feat_channels=self.feat_channels)
                self.br_gate = _BoundaryResponsiveGate(
                    channels=self.feat_channels,
                    mod_scale_init=self.head_pc_residual_init,
                    mod_scale_max=self.head_pc_residual_max,
                    boundary_groups=18)
            else:
                self.lbp_branch = None
                self.br_gate = None

            self.reg_detail_injector = (
                _RegressionDetailInjector(
                    channels=self.feat_channels,
                    num_levels=self.anchor_generator.num_levels,
                    levels=self.reg_detail_levels,
                    residual_init=self.reg_detail_residual_init,
                    residual_max=self.reg_detail_residual_max)
                if self.use_reg_detail_bypass else None)

            if self.use_d_fine_style:
                bins = self.d_fine_reg_max + 1
                # RetinaGuideAttHead invokes _init_layers before constructing
                # SymFormer. Preserve the global RNG so merely enabling this
                # zero-start branch does not silently change transformer
                # initialization and confound the ablation.
                with torch.random.fork_rng(devices=[]):
                    self.d_fine_reg = nn.Conv2d(
                        self.feat_channels,
                        self.num_anchors * 4 * bins,
                        kernel_size=3,
                        padding=1)
                # A uniform symmetric distribution has zero expectation, so
                # the first forward is exactly the Retina/GIoU baseline.
                nn.init.zeros_(self.d_fine_reg.weight)
                nn.init.zeros_(self.d_fine_reg.bias)
                raw_init = math.atanh(
                    self.d_fine_refine_init
                    / self.d_fine_refine_max)
                self.d_fine_refine_raw = nn.Parameter(torch.full(
                    (self.anchor_generator.num_levels,),
                    raw_init,
                    dtype=torch.float32))
            else:
                self.d_fine_reg = None
                self.register_parameter(
                    'd_fine_refine_raw', None)

            # Detection-aligned auxiliary branch. It is evaluated only during
            # training: high-IoU anchors matched to the same GT instance are
            # positives; bbox-neighbouring background anchors are negatives.
            # No convolution or sampling is executed during inference.
            if self.use_bbox_contrastive:
                self.bbox_contrast_proj = nn.Conv2d(
                    self.feat_channels,
                    self.bbox_contrast_dim,
                    kernel_size=1)
                self.bbox_anchor_embed = nn.Embedding(
                    self.num_anchors,
                    self.bbox_contrast_dim)
                nn.init.kaiming_normal_(
                    self.bbox_contrast_proj.weight,
                    mode='fan_out',
                    nonlinearity='linear')
                nn.init.zeros_(self.bbox_contrast_proj.bias)
                nn.init.normal_(
                    self.bbox_anchor_embed.weight, std=0.02)
            else:
                self.bbox_contrast_proj = None
                self.bbox_anchor_embed = None

            # C2/NAM instance prototypes are one view; matched P2-P4
            # regression-tower cells are the detection view. Region prototypes
            # are detached in the alignment loss so this branch updates
            # CrossScale/Sym/Retina without destabilising the C2 prototype.
            if self.use_boxalign:
                self.boxalign_proj = nn.Conv2d(
                    self.feat_channels, self.boxalign_dim, kernel_size=1)
                self.boxalign_anchor_embed = nn.Embedding(
                    self.num_anchors, self.boxalign_dim)
                nn.init.kaiming_normal_(
                    self.boxalign_proj.weight,
                    mode='fan_out',
                    nonlinearity='linear')
                nn.init.zeros_(self.boxalign_proj.bias)
                nn.init.normal_(
                    self.boxalign_anchor_embed.weight, std=0.02)
            else:
                self.boxalign_proj = None
                self.boxalign_anchor_embed = None
            _reset_boxalign_debug_state(
                self, self.anchor_generator.num_levels)

            # --- Learned LBP embedding: 59 uniform patterns → 32-dim dense vector ---
            # Replaces sparse one-hot (98.3% zeros) with dense learnable embedding.
            # 6 directions × 3 scales × 32 dims = 576 output channels (was 1062).
            self.lbp_embed = nn.Embedding(59, 32)

            # --- Source-faithful PC front-end for spatial BPPC ---
            # Six orientations match the paper (0,30,...,150 degrees). Three
            # log-Gabor scales supply odd/even quadrature responses, amplitude
            # normalization and noise-thresholded local phase congruency.
            self.phase_congruency = _LogGaborPhaseCongruency(
                num_scales=3,
                num_orientations=6,
                min_wavelength=3.0,
                scale_multiplier=2.1,
                sigma_on_frequency=0.55,
                noise_k=2.0)

            # Temperature hook: amplify gradient for spatial bias learning.
            self.nam_branch.spatial_bias_fn.temperature.register_hook(
                lambda g: g * 20.)

        def init_weights(self):
            super().init_weights()
            if self.use_d_fine_style:
                # Protect the zero-residual start even if a parent init_cfg
                # initializes all Conv2d layers.
                nn.init.zeros_(self.d_fine_reg.weight)
                nn.init.zeros_(self.d_fine_reg.bias)

        def effective_d_fine_scales(self) -> torch.Tensor:
            if not self.use_d_fine_style:
                return self.retina_reg.weight.new_zeros(
                    self.anchor_generator.num_levels)
            return (
                self.d_fine_refine_max
                * torch.tanh(self.d_fine_refine_raw))

        def compute_pc_lbp(self, img, pc_gray=None):
            """Spatial BPPC: gray→PC×6→per-direction 3-radius LBP→576ch.

            6 directions × 3 scales × 32-dim learned embedding = 576 channels.
            Learned Embedding(59, 32) replaces fixed one-hot — eliminates 98.3%
            sparsity while retaining the location information required by bbox
            detection. The source paper instead globally histograms these maps
            into 6×177=1062 values, which is suitable for image classification
            but would discard lesion coordinates.

            ``pc_gray`` is the preferred v3.11 input: an aligned raw
            bone-suppressed CXR in [0, 255]. ``img`` remains the
            ImageNet-normalised raw CXR supplied to TransNeXt. The legacy
            inverse-normalisation path is retained only for controlled
            ablation with ``pc_gray=None``.

            Returns (B, 576, H, W).
            PC provides illumination/contrast invariance; LBP on the six
            orientation PC maps extracts local texture from that structure.
            """
            B, C, H, W = img.shape
            device = img.device
            if pc_gray is None:
                mean = torch.tensor(
                    [123.675, 116.28, 103.53],
                    device=device, dtype=img.dtype).view(1, 3, 1, 1)
                std = torch.tensor(
                    [58.395, 57.12, 57.375],
                    device=device, dtype=img.dtype).view(1, 3, 1, 1)
                img_raw = img * std + mean
                gray = (
                    0.299 * img_raw[:, 0:1]
                    + 0.587 * img_raw[:, 1:2]
                    + 0.114 * img_raw[:, 2:3])
            else:
                gray = torch.as_tensor(
                    pc_gray, device=device, dtype=img.dtype)
                if gray.ndim == 3:
                    gray = gray.unsqueeze(1)
                if gray.ndim != 4 or gray.shape[1] != 1:
                    raise ValueError(
                        f'pc_gray must have shape (B,1,H,W) or (B,H,W), '
                        f'got {tuple(gray.shape)}')
                if gray.shape[0] != B or tuple(gray.shape[-2:]) != (H, W):
                    raise ValueError(
                        f'pc_gray shape {tuple(gray.shape)} does not match '
                        f'detector image shape {(B, C, H, W)}')
                if not torch.isfinite(gray).all():
                    raise ValueError('pc_gray contains NaN or Inf')
                gray = gray.clamp(0.0, 255.0)

            # (B,6,H,W): true multi-scale quadrature PC per orientation.
            pc_maps = self.phase_congruency(gray)

            # Per-direction 3-scale LBP → 6×3×32 = 576ch (learned embedding)
            all_lbp = []
            offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
            for orient_idx in range(pc_maps.shape[1]):
                pc = pc_maps[:, orient_idx]  # (B,H,W)
                for scale_r in [1, 2, 3]:
                    pad_w = scale_r
                    pc_pad = F.pad(pc, (pad_w,pad_w,pad_w,pad_w), mode='reflect')
                    code = torch.zeros(B, H, W, device=device)
                    for i, (dy, dx) in enumerate(offsets):
                        ny, nx = pad_w + dy*scale_r, pad_w + dx*scale_r
                        code += (pc_pad[:, ny:ny+H, nx:nx+W] >= pc).float() * (2**i)
                    # Learned embedding replaces sparse one-hot
                    indices = _lbp_to_uniform_indices(code.long(), device)  # (B,H,W)
                    embedded = self.lbp_embed(indices)  # (B,H,W,32)
                    all_lbp.append(embedded)

            out = torch.cat(all_lbp, dim=-1)  # (B,H,W, 6*3*32=576)
            return out.permute(0, 3, 1, 2)   # (B,576,H,W)

        def _ensure_shared_feat_list(self, feats, bs):
            """Return per-sample P2 features as list[Tensor] (bs, C, H, W)."""
            if isinstance(feats, (list, tuple)):
                p2 = feats[0]  # (B, C, H, W)
            else:
                p2 = feats
            return [p2[b] for b in range(bs)]

        def forward_single(self, x, lbp_lab=None, lung_masks=None,
                           level_index: int = 0,
                           reg_detail_feat=None):
            """Single CrossScale level with regression-only PC injection.

            Args:
                x: (B, C, H, W) pyramid features at one level
                lbp_lab: (B, 576, H_orig, W_orig) PC+LBP embedding features
                lung_masks: (B, H_pad, W_pad) aligned anatomy masks
            Returns:
                cls_score, bbox_pred
            """
            bs, _, h, w = x.shape
            masks = x.new_zeros((bs, h, w)).to(torch.bool)

            # Branch 1: SymFormer (unchanged from v3)
            pos_embed = self.positional_encoding(
                masks, x, lung_mask=lung_masks)
            feat1 = self.transformer(
                tuple([x]), [masks], self.query_embedding.weight, [pos_embed])

            # PC bypasses CrossScale and modulates only localization features.
            if self.use_head_pc_modulation and lbp_lab is not None:
                feat_nam = self.lbp_branch(lbp_lab, (h, w))
                fused, gate_stats = self.br_gate(
                    feat1, feat_nam, lbp_lab)
                pc_residual = fused - feat1
                gate_stats['pc_feat_norm'] = (
                    feat_nam.detach().float().norm(
                        dim=1).mean().item())
                gate_stats['pc_contribution_ratio'] = (
                    pc_residual.detach().float().norm()
                    / feat1.detach().float().norm().clamp_min(1e-8)
                    ).item()
            else:
                fused = feat1
                gate_stats = {
                    'gate_mean': 0.0,
                    'gate_active': 0.0,
                    'edge_mean': 0.0,
                    'spatial_gate_mean': 0.0,
                    'edge_temp': 0.0,
                    'mod_scale': 0.0,
                    'boundary_weight_entropy': 0.0,
                    'boundary_weight_peak': 0.0,
                    'pc_feat_norm': 0.0,
                    'pc_contribution_ratio': 0.0,
                }

            # Gradient routing: cls_feat = clean feat1, reg_feat = NAM-modulated
            cls_feat = feat1
            reg_feat = fused
            if self.use_reg_detail_bypass:
                reg_feat = self.reg_detail_injector(
                    reg_feat, reg_detail_feat,
                    level_index=level_index)
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.retina_cls(cls_feat)
            bbox_base = self.retina_reg(reg_feat)
            distribution_logits = None
            if self.use_d_fine_style:
                if not (0 <= level_index
                        < self.anchor_generator.num_levels):
                    raise IndexError(
                        f'invalid pyramid level {level_index}')
                distribution_logits = self.d_fine_reg(reg_feat)
                normalized_residual, entropy = (
                    _distribution_refinement_expectation(
                        distribution_logits,
                        self.num_anchors,
                        self.d_fine_reg_max))
                refine_scale = self.effective_d_fine_scales()[
                    level_index].to(dtype=bbox_base.dtype)
                bbox_pred = (
                    bbox_base
                    + refine_scale * normalized_residual)
                self._dbg_d_fine_entropy[level_index] = float(
                    entropy.detach().float().item())
                self._dbg_d_fine_expected_abs[level_index] = float(
                    normalized_residual.detach().float()
                    .abs().mean().item())
            else:
                bbox_pred = bbox_base
            return (
                cls_score, bbox_pred, feat1, fused, reg_feat, gate_stats,
                bbox_base, distribution_logits)

        def _gather_boxalign_features(
                self,
                projected_levels,
                batch_index,
                global_anchor_indices,
                num_level_anchors):
            """Gather regression-cell embeddings for sparse anchor indices."""
            if global_anchor_indices.numel() == 0:
                return projected_levels[0].new_zeros(
                    (0, self.boxalign_dim))
            chunks, positions = [], []
            level_start = 0
            for level_index, level_count in enumerate(num_level_anchors):
                level_end = level_start + int(level_count)
                in_level = (
                    (global_anchor_indices >= level_start)
                    & (global_anchor_indices < level_end))
                if in_level.any():
                    output_positions = in_level.nonzero(
                        as_tuple=False).squeeze(-1)
                    local_indices = (
                        global_anchor_indices[output_positions] - level_start)
                    cell_indices = local_indices // self.num_anchors
                    anchor_types = local_indices % self.num_anchors
                    cells = projected_levels[
                        level_index][batch_index].permute(
                            1, 2, 0).reshape(-1, self.boxalign_dim)
                    chunks.append(
                        cells[cell_indices]
                        + self.boxalign_anchor_embed(anchor_types))
                    positions.append(output_positions)
                level_start = level_end
            if not chunks:
                return projected_levels[0].new_zeros(
                    (0, self.boxalign_dim))
            features = torch.cat(chunks, dim=0)
            restore = torch.cat(positions, dim=0).argsort()
            return F.normalize(features[restore], dim=-1)

        def region_anchor_align_loss(self, gt_bboxes):
            """Align per-GT NAM prototypes with matched P2-P4 reg features."""
            zero = self.retina_reg.weight.sum() * 0.0
            _reset_boxalign_debug_state(
                self, self.anchor_generator.num_levels)

            reg_features = self._last_reg_features
            self._last_reg_features = None
            records_by_image = self.nam_branch._last_boxalign_records
            self.nam_branch._last_boxalign_records = None
            if (not self.training or not self.use_boxalign
                    or reg_features is None or records_by_image is None
                    or gt_bboxes is None):
                return zero
            if max(self.boxalign_levels) >= len(reg_features):
                raise ValueError(
                    f'boxalign_levels={self.boxalign_levels} exceeds '
                    f'{len(reg_features)} feature levels')

            projected_levels = [
                self.boxalign_proj(feature)
                for feature in reg_features]
            featmap_sizes = [
                tuple(feature.shape[-2:])
                for feature in reg_features]
            mlvl_anchors = self.anchor_generator.grid_anchors(
                featmap_sizes, device=reg_features[0].device)
            num_level_anchors = [
                anchors.shape[0] for anchors in mlvl_anchors]
            all_anchors = torch.cat(mlvl_anchors, dim=0)
            level_offsets = []
            offset = 0
            for count in num_level_anchors:
                level_offsets.append(offset)
                offset += int(count)

            weighted_losses = []
            total_weights = []
            iou_values = []
            positive_similarities = []
            negative_similarities = []
            level_counts = [0] * len(num_level_anchors)
            valid_gt = 0
            total_gt = 0
            anchor_count = 0
            single_region_gt = 0

            for batch_index, boxes in enumerate(gt_bboxes):
                boxes = boxes.to(
                    device=all_anchors.device,
                    dtype=all_anchors.dtype)
                total_gt += int(boxes.shape[0])
                if boxes.numel() == 0:
                    continue
                records = (
                    records_by_image[batch_index]
                    if batch_index < len(records_by_image) else None)
                if not records:
                    continue
                overlaps = _pairwise_bbox_iou(all_anchors, boxes)
                max_iou, best_gt = overlaps.max(dim=1)

                for record in records:
                    gt_index = int(record['gt_index'])
                    if gt_index >= boxes.shape[0]:
                        continue
                    # A fallback keeps the semantic contrast term trainable for
                    # tiny lesions, but may lie below the requested purity
                    # threshold. Never let that noisy prototype steer bbox
                    # regression through BoxAlign.
                    if record.get('is_fallback', False):
                        continue
                    prototype = record['prototype'].detach()
                    negatives = record['negative_tokens'].detach()
                    negative_weights = record[
                        'negative_weights'].detach()
                    if negatives.numel() == 0:
                        continue
                    selected_indices = []
                    selected_ious = []
                    selected_levels = []
                    for level_index in self.boxalign_levels:
                        start = level_offsets[level_index]
                        end = start + int(num_level_anchors[level_index])
                        level_global = torch.arange(
                            start, end, device=all_anchors.device)
                        assigned = (
                            (best_gt[start:end] == gt_index)
                            & (overlaps[start:end, gt_index]
                               >= self.boxalign_min_iou))
                        candidate = level_global[assigned]
                        if candidate.numel() == 0:
                            continue
                        candidate_iou = overlaps[candidate, gt_index]
                        keep = min(
                            self.boxalign_max_anchors_per_level,
                            candidate.numel())
                        chosen_local = torch.topk(
                            candidate_iou, k=keep, largest=True).indices
                        selected_indices.append(candidate[chosen_local])
                        selected_ious.append(candidate_iou[chosen_local])
                        selected_levels.extend([level_index] * keep)

                    # MaxIoUAssigner guarantees each GT a best match even when
                    # no anchor crosses the positive threshold. Mirror that
                    # behaviour to avoid losing small/single-region lesions.
                    if not selected_indices:
                        enabled_indices = []
                        for level_index in self.boxalign_levels:
                            start = level_offsets[level_index]
                            end = start + int(num_level_anchors[level_index])
                            enabled_indices.append(torch.arange(
                                start, end, device=all_anchors.device))
                        enabled_indices = torch.cat(enabled_indices)
                        best_local = torch.argmax(
                            overlaps[enabled_indices, gt_index])
                        chosen = enabled_indices[best_local].reshape(1)
                        selected_indices = [chosen]
                        selected_ious = [
                            overlaps[chosen, gt_index]]
                        chosen_value = int(chosen.item())
                        chosen_level = next(
                            level_index
                            for level_index in self.boxalign_levels
                            if level_offsets[level_index]
                            <= chosen_value
                            < (
                                level_offsets[level_index]
                                + num_level_anchors[level_index]))
                        selected_levels = [chosen_level]

                    selected_indices = torch.cat(selected_indices)
                    selected_ious = torch.cat(selected_ious)
                    anchors = self._gather_boxalign_features(
                        projected_levels,
                        batch_index,
                        selected_indices,
                        num_level_anchors)
                    positive_logits = (
                        anchors @ prototype
                    ).unsqueeze(1) / self.boxalign_temperature
                    negative_logits = (
                        anchors @ negatives.T
                    ) / self.boxalign_temperature
                    negative_logits = (
                        negative_logits
                        + negative_weights.clamp_min(
                            1e-6).log().unsqueeze(0))
                    logits = torch.cat(
                        (positive_logits, negative_logits), dim=1)
                    target = torch.zeros(
                        anchors.shape[0],
                        dtype=torch.long,
                        device=anchors.device)
                    per_anchor_loss = F.cross_entropy(
                        logits, target, reduction='none')
                    anchor_weights = (
                        0.25 + 0.75 * selected_ious.clamp(0.0, 1.0))
                    weighted_losses.append(
                        (per_anchor_loss * anchor_weights).sum())
                    total_weights.append(anchor_weights.sum())

                    valid_gt += 1
                    anchor_count += int(anchors.shape[0])
                    if record['positive_region_count'] == 1:
                        single_region_gt += 1
                    for level_index in selected_levels:
                        level_counts[level_index] += 1
                    iou_values.append(selected_ious.detach())
                    positive_similarities.append(
                        (anchors.detach() @ prototype).mean())
                    negative_similarities.append(
                        (anchors.detach() @ negatives.T).mean())

            if weighted_losses:
                loss = torch.stack(weighted_losses).sum() / torch.stack(
                    total_weights).sum().clamp_min(1e-8)
            else:
                loss = zero
            self._dbg_boxalign_loss_raw = float(loss.detach().item())
            self._dbg_boxalign_total_gt = int(total_gt)
            self._dbg_boxalign_valid_gt = int(valid_gt)
            self._dbg_boxalign_anchor_count = int(anchor_count)
            self._dbg_boxalign_level_counts = level_counts
            self._dbg_boxalign_single_region_gt = int(single_region_gt)
            if iou_values:
                self._dbg_boxalign_mean_iou = float(
                    torch.cat(iou_values).mean().item())
            if positive_similarities:
                self._dbg_boxalign_positive_similarity = float(
                    torch.stack(positive_similarities).mean().item())
                self._dbg_boxalign_negative_similarity = float(
                    torch.stack(negative_similarities).mean().item())
            # Retain live tensors only on the iteration immediately preceding
            # the next deep-debug print; otherwise they would keep an old
            # autograd graph alive across iterations.
            retain_for_grad_debug = (
                self.deep_debug_interval > 0
                and (self._debug_step + 1)
                % self.deep_debug_interval == 0)
            self._dbg_last_align_reg_features = (
                reg_features if retain_for_grad_debug else None)
            return loss

        def _gather_bbox_contrast_features(
                self,
                projected_levels,
                batch_index,
                global_anchor_indices,
                num_level_anchors):
            """Gather sparse anchor embeddings without expanding HxWxA maps."""
            if global_anchor_indices.numel() == 0:
                return projected_levels[0].new_zeros(
                    (0, self.bbox_contrast_dim))

            feature_chunks = []
            output_positions = []
            level_start = 0
            for level_index, level_count in enumerate(num_level_anchors):
                level_end = level_start + int(level_count)
                in_level = (
                    (global_anchor_indices >= level_start)
                    & (global_anchor_indices < level_end))
                if in_level.any():
                    positions = in_level.nonzero(
                        as_tuple=False).squeeze(-1)
                    local_anchor_indices = (
                        global_anchor_indices[positions] - level_start)
                    cell_indices = (
                        local_anchor_indices // self.num_anchors)
                    anchor_types = (
                        local_anchor_indices % self.num_anchors)
                    level_features = projected_levels[
                        level_index][batch_index].permute(
                            1, 2, 0).reshape(
                                -1, self.bbox_contrast_dim)
                    gathered = (
                        level_features[cell_indices]
                        + self.bbox_anchor_embed(anchor_types))
                    feature_chunks.append(gathered)
                    output_positions.append(positions)
                level_start = level_end

            if not feature_chunks:
                return projected_levels[0].new_zeros(
                    (0, self.bbox_contrast_dim))
            features = torch.cat(feature_chunks, dim=0)
            positions = torch.cat(output_positions, dim=0)
            restore_order = positions.argsort()
            return F.normalize(features[restore_order], dim=-1)

        def bbox_instance_contrast_loss(self, gt_bboxes, img_metas=None):
            """IoU-aware instance contrast on bbox regression-tower features.

            The normal Retina/GIoU target remains unchanged. This auxiliary
            objective selects high-IoU anchors assigned to each GT lesion,
            treats anchors of the same GT as positives, ignores the ambiguous
            IoU band, and mines bbox-neighbouring background anchors as
            negatives. It uses only the TB images already in stage one.
            """
            zero = self.retina_reg.weight.sum() * 0.0
            self._dbg_bbox_contrast_loss_raw = 0.0
            self._dbg_bbox_contrast_positive_count = 0
            self._dbg_bbox_contrast_negative_count = 0
            self._dbg_bbox_contrast_valid_instances = 0
            self._dbg_bbox_contrast_valid_anchors = 0
            self._dbg_bbox_contrast_mean_iou = 0.0
            self._dbg_bbox_contrast_positive_similarity = 0.0
            self._dbg_bbox_contrast_total_gt = 0
            self._dbg_bbox_contrast_skipped_gt = 0
            self._dbg_bbox_contrast_gt_coverage = 0.0
            self._dbg_bbox_contrast_ambiguous_count = 0
            self._dbg_bbox_contrast_positive_levels = [
                0] * self.anchor_generator.num_levels
            self._dbg_bbox_contrast_negative_levels = [
                0] * self.anchor_generator.num_levels
            self._dbg_bbox_contrast_mean_negative_iou = 0.0
            self._dbg_bbox_contrast_positive_weight_mean = 0.0
            self._dbg_bbox_contrast_weighted = 0.0
            self._dbg_bbox_contrast_to_giou_ratio = 0.0

            reg_features = self._last_reg_features
            self._last_reg_features = None
            if (not self.training or not self.use_bbox_contrastive
                    or reg_features is None or gt_bboxes is None):
                return zero

            projected_levels = [
                self.bbox_contrast_proj(feature)
                for feature in reg_features]
            featmap_sizes = [
                tuple(feature.shape[-2:])
                for feature in reg_features]
            mlvl_anchors = self.anchor_generator.grid_anchors(
                featmap_sizes, device=reg_features[0].device)
            num_level_anchors = [
                anchors.shape[0] for anchors in mlvl_anchors]
            all_anchors = torch.cat(mlvl_anchors, dim=0)

            def anchor_level_counts(indices):
                counts = []
                level_start = 0
                for level_count in num_level_anchors:
                    level_end = level_start + int(level_count)
                    counts.append(int(
                        ((indices >= level_start)
                         & (indices < level_end)).sum().item()))
                    level_start = level_end
                return counts

            valid_flags = None
            if img_metas is not None:
                _, valid_flag_list = self.get_anchors(
                    featmap_sizes,
                    img_metas,
                    device=reg_features[0].device)
                valid_flags = [
                    torch.cat(flags, dim=0)
                    for flags in valid_flag_list]

            batch_positive_tokens = []
            batch_instance_ids = []
            batch_positive_weights = []
            batch_negative_tokens = []
            positive_iou_values = []
            negative_iou_values = []
            positive_level_counts = [0] * len(num_level_anchors)
            negative_level_counts = [0] * len(num_level_anchors)
            valid_instance_count = 0
            total_gt_count = 0
            ambiguous_anchor_count = 0
            instance_offset = 0

            for batch_index, boxes in enumerate(gt_bboxes):
                boxes = boxes.to(
                    device=all_anchors.device,
                    dtype=all_anchors.dtype)
                if boxes.numel() == 0:
                    instance_offset += 1
                    continue
                total_gt_count += int(boxes.shape[0])

                overlaps = _pairwise_bbox_iou(all_anchors, boxes)
                max_iou, best_gt = overlaps.max(dim=1)
                valid_anchor_mask = (
                    valid_flags[batch_index]
                    if valid_flags is not None
                    else torch.ones_like(max_iou, dtype=torch.bool))
                positive_indices = []
                positive_instances = []
                positive_ious = []
                sample_valid_instance_count = 0
                ambiguous_anchor_count += int(
                    (
                        valid_anchor_mask
                        & (max_iou >= self.bbox_contrast_neg_iou)
                        & (max_iou < self.bbox_contrast_pos_iou)
                    ).sum().item())

                for gt_index in range(boxes.shape[0]):
                    gt_candidates = (
                        valid_anchor_mask
                        & (best_gt == gt_index)
                        & (max_iou >= self.bbox_contrast_pos_iou))
                    candidate_indices = gt_candidates.nonzero(
                        as_tuple=False).squeeze(-1)
                    if candidate_indices.numel() < 2:
                        continue
                    candidate_ious = overlaps[
                        candidate_indices, gt_index]
                    keep_count = min(
                        self.bbox_contrast_max_pos_per_gt,
                        candidate_indices.numel())
                    keep_local = torch.topk(
                        candidate_ious,
                        k=keep_count,
                        largest=True).indices
                    selected_indices = candidate_indices[keep_local]
                    selected_ious = candidate_ious[keep_local]
                    positive_indices.append(selected_indices)
                    positive_ious.append(selected_ious)
                    positive_instances.append(torch.full(
                        (keep_count,),
                        instance_offset + gt_index,
                        dtype=torch.long,
                        device=all_anchors.device))
                    sample_valid_instance_count += 1

                instance_offset += max(int(boxes.shape[0]), 1)
                if not positive_indices:
                    continue

                positive_indices = torch.cat(positive_indices)
                positive_instances = torch.cat(positive_instances)
                positive_ious = torch.cat(positive_ious)
                positive_tokens = self._gather_bbox_contrast_features(
                    projected_levels,
                    batch_index,
                    positive_indices,
                    num_level_anchors)
                positive_weights = (
                    0.25
                    + 0.75 * (
                        (positive_ious - self.bbox_contrast_pos_iou)
                        / max(
                            1.0 - self.bbox_contrast_pos_iou,
                            1e-6)).clamp(0.0, 1.0))

                negative_mask = (
                    valid_anchor_mask
                    & (max_iou < self.bbox_contrast_neg_iou))
                negative_indices = negative_mask.nonzero(
                    as_tuple=False).squeeze(-1)
                if negative_indices.numel() == 0:
                    continue
                negative_budget = max(
                    self.bbox_contrast_min_negatives,
                    int(math.ceil(
                        self.bbox_contrast_hard_negative_ratio
                        * positive_indices.numel())))
                negative_budget = min(
                    negative_budget, negative_indices.numel())

                # First retain a small spatially hard candidate pool by IoU,
                # then mine the candidates most similar to lesion anchors.
                candidate_budget = min(
                    negative_indices.numel(),
                    max(
                        negative_budget * 4,
                        self.bbox_contrast_min_negatives * 4))
                candidate_local = torch.topk(
                    max_iou[negative_indices],
                    k=candidate_budget,
                    largest=True).indices
                candidate_indices = negative_indices[candidate_local]
                candidate_tokens = self._gather_bbox_contrast_features(
                    projected_levels,
                    batch_index,
                    candidate_indices,
                    num_level_anchors)
                with torch.no_grad():
                    prototype = F.normalize(
                        (
                            positive_tokens.detach()
                            * positive_weights[:, None]
                        ).sum(dim=0),
                        dim=0)
                    candidate_similarity = (
                        candidate_tokens.detach() @ prototype)
                    iou_hardness = (
                        max_iou[candidate_indices]
                        / max(self.bbox_contrast_neg_iou, 1e-6))
                    selection_score = (
                        iou_hardness + 0.25 * candidate_similarity)
                    selected_negative_local = torch.topk(
                        selection_score,
                        k=negative_budget,
                        largest=True).indices
                negative_tokens = candidate_tokens[
                    selected_negative_local]
                selected_negative_indices = candidate_indices[
                    selected_negative_local]
                sample_positive_level_counts = anchor_level_counts(
                    positive_indices)
                sample_negative_level_counts = anchor_level_counts(
                    selected_negative_indices)
                positive_level_counts = [
                    total + current
                    for total, current in zip(
                        positive_level_counts,
                        sample_positive_level_counts)]
                negative_level_counts = [
                    total + current
                    for total, current in zip(
                        negative_level_counts,
                        sample_negative_level_counts)]
                valid_instance_count += sample_valid_instance_count

                batch_positive_tokens.append(positive_tokens)
                batch_instance_ids.append(positive_instances)
                batch_positive_weights.append(positive_weights)
                batch_negative_tokens.append(negative_tokens)
                positive_iou_values.append(positive_ious)
                negative_iou_values.append(
                    max_iou[selected_negative_indices])

            self._dbg_bbox_contrast_total_gt = int(total_gt_count)
            self._dbg_bbox_contrast_valid_instances = int(
                valid_instance_count)
            self._dbg_bbox_contrast_skipped_gt = int(
                max(total_gt_count - valid_instance_count, 0))
            self._dbg_bbox_contrast_gt_coverage = float(
                valid_instance_count / max(total_gt_count, 1))
            self._dbg_bbox_contrast_ambiguous_count = int(
                ambiguous_anchor_count)
            self._dbg_bbox_contrast_positive_levels = (
                positive_level_counts)
            self._dbg_bbox_contrast_negative_levels = (
                negative_level_counts)

            if not batch_positive_tokens or not batch_negative_tokens:
                return zero

            positive_tokens = torch.cat(batch_positive_tokens, dim=0)
            instance_ids = torch.cat(batch_instance_ids, dim=0)
            positive_weights = torch.cat(
                batch_positive_weights, dim=0)
            negative_tokens = torch.cat(batch_negative_tokens, dim=0)
            loss, positive_similarity, valid_anchor_count = (
                _weighted_instance_contrastive_loss(
                    positive_tokens,
                    instance_ids,
                    positive_weights,
                    negative_tokens,
                    self.bbox_contrast_temperature))

            self._dbg_bbox_contrast_loss_raw = float(
                loss.detach().item())
            self._dbg_bbox_contrast_positive_count = int(
                positive_tokens.shape[0])
            self._dbg_bbox_contrast_negative_count = int(
                negative_tokens.shape[0])
            self._dbg_bbox_contrast_valid_anchors = int(
                valid_anchor_count)
            self._dbg_bbox_contrast_mean_iou = float(
                torch.cat(positive_iou_values).mean().detach().item())
            self._dbg_bbox_contrast_positive_similarity = (
                positive_similarity)
            if negative_iou_values:
                self._dbg_bbox_contrast_mean_negative_iou = float(
                    torch.cat(negative_iou_values).mean().detach().item())
            self._dbg_bbox_contrast_positive_weight_mean = float(
                positive_weights.mean().detach().item())
            return loss

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

            _emit_training_debug(
                f'[NAMv3] iter={self._debug_step} h={h} '
                f'g_m={gate_mean:.3f} g_act={gate_active:.3f} '
                f'e_m={edge_mean:.3f} '
                f'|feat1|={feat1.detach().norm(dim=1).mean().item():.2f} '
                f'|fused|={fused.detach().norm(dim=1).mean().item():.2f} '
                f'ΔNAM={delta_per_pixel:.3f} '
                f'sig={sig_change_frac:.3f}')

        def forward(self, feats, nam_graphs=None, gt_bboxes=None,
                    gt_labels=None,
                    lbp_lab=None, c2_feat=None, region_feat=None,
                    img_shapes=None,
                    lung_masks=None, reg_detail_feats=None):
            """Multi-level head forward after CrossScale neck fusion.

            Processing order (v3.12):
              1. High-purity region pooling on the selected C2/C5/P5 feature
                 → optional per-GT topology contrast
              2. Per-level SymFormer regression features
              3. C2/NAM GT prototype → P2-P4 regression-anchor BoxAlign

            Args:
                feats: CrossScale neck features [P2, P3, P4, P5]
                nam_graphs: list[dict] per-sample graph data
                gt_bboxes: list[list] per-sample bbox annotations
                lbp_lab: (B, 576, H_orig, W_orig) PC+LBP embedding features
                lung_masks: (B, H_pad, W_pad) aligned lung/heart masks
                img_shapes: list[(H,W)] per-sample image shapes for bbox scaling
            Returns:
                cls_scores, bbox_preds, region_contrast_loss, boxalign_loss
            """
            bs = feats[0].shape[0]
            feats = list(feats)  # make mutable
            if reg_detail_feats is None:
                reg_detail_feats = [None] * len(feats)
            if len(reg_detail_feats) != len(feats):
                raise ValueError(
                    'reg_detail_feats must match pyramid level count')

            # === Step 1: Dynamic region token + contrastive loss ===
            contr_loss = torch.zeros((), device=feats[0].device)
            if nam_graphs is not None and any(g is not None for g in nam_graphs):
                selected_region_feat = (
                    region_feat if region_feat is not None else c2_feat)
                if selected_region_feat is not None:
                    if selected_region_feat.shape[1] != self.region_in_channels:
                        raise ValueError(
                            f'NAM region feature has '
                            f'{selected_region_feat.shape[1]} channels but '
                            f'region_in_channels={self.region_in_channels}. '
                            'Match region_in_channels to the configured '
                            'region_feature_source.')
                    region_per_sample = [
                        selected_region_feat[b] for b in range(bs)]
                else:
                    region_per_sample = self._ensure_shared_feat_list(
                        feats, bs)
                contr_loss = self.nam_branch(
                    region_per_sample, nam_graphs, gt_bboxes,
                    gt_labels=gt_labels,
                    img_shapes=img_shapes)

            # === Step 2: Per-level SymFormer (+ regression-only PC gate) ===
            cls_scores, bbox_preds, reg_features = [], [], []
            d_fine_logits, d_fine_base_preds = [], []
            feat1_p2, fused_p2 = None, None  # saved for debug
            level_features, level_gate_stats = [], []
            upper_prior_level_stats = []
            debug_this_step = (
                self.training
                and self.debug_interval > 0
                and self._debug_step % self.debug_interval == 0)
            for i, feat in enumerate(feats):
                (cls_score, bbox_pred, f1, f_fused, reg_feat, gate_stats,
                 bbox_base, distribution_logits) = \
                    self.forward_single(
                        feat, lbp_lab, lung_masks=lung_masks,
                        level_index=i,
                        reg_detail_feat=reg_detail_feats[i])
                cls_scores.append(cls_score)
                bbox_preds.append(bbox_pred)
                reg_features.append(reg_feat)
                level_features.append((f1, f_fused))
                level_gate_stats.append(gate_stats)
                if self.use_d_fine_style:
                    d_fine_logits.append(distribution_logits)
                    d_fine_base_preds.append(bbox_base)
                if debug_this_step:
                    prior_stats = getattr(
                        self.positional_encoding, '_dbg_upper_prior', {})
                    upper_prior_level_stats.append({
                        name: (
                            value.detach().float().item()
                            if torch.is_tensor(value) else float(value))
                        for name, value in prior_stats.items()
                    })
                if i == 0:  # P2 level
                    feat1_p2, fused_p2 = f1, f_fused

            # Keep live references only for training-time auxiliary losses.
            self._last_reg_features = (
                tuple(reg_features)
                if self.training
                and (self.use_bbox_contrastive or self.use_boxalign)
                else None)
            self._last_d_fine_logits = (
                tuple(d_fine_logits)
                if self.training and self.use_d_fine_style
                else None)
            self._last_d_fine_base_preds = (
                tuple(d_fine_base_preds)
                if self.training and self.use_d_fine_style
                else None)
            boxalign_loss = (
                self.region_anchor_align_loss(gt_bboxes)
                if self.training and self.use_boxalign
                else contr_loss * 0.0)
            self._dbg_region_loss_raw = float(
                contr_loss.detach().float().item())
            self._dbg_region_contrast_weighted = float(
                abs(
                    getattr(self, '_dbg_contrast_loss_weight', 0.0)
                    * self._dbg_region_loss_raw))
            self._dbg_boxalign_weighted = float(
                abs(
                    getattr(self, '_dbg_boxalign_loss_weight', 0.0)
                    * getattr(self, '_dbg_boxalign_loss_raw', 0.0)))

            # === Step 3: Enhanced Debug Logging ===
            if debug_this_step:
                with torch.no_grad():
                    rel_changes, cosines = [], []
                    for f1, f_fused in level_features:
                        base = f1.detach().float().flatten(1)
                        delta = (
                            f_fused.detach().float() - f1.detach().float()
                        ).flatten(1)
                        rel_changes.append(
                            (delta.norm(dim=1)
                             / base.norm(dim=1).clamp_min(1e-8)
                             ).mean().item())
                        cosines.append(F.cosine_similarity(
                            base, f_fused.detach().float().flatten(1),
                            dim=1, eps=1e-8).mean().item())
                    self._dbg_nam_level_stats = {
                        'rel_change': rel_changes,
                        'cosine': cosines,
                        'gate_mean': [
                            stats.get('gate_mean', 0.0)
                            for stats in level_gate_stats],
                        'gate_active': [
                            stats.get('gate_active', 0.0)
                            for stats in level_gate_stats],
                        'spatial_gate_mean': [
                            stats.get('spatial_gate_mean', 0.0)
                            for stats in level_gate_stats],
                        'boundary_weight_entropy': [
                            stats.get('boundary_weight_entropy', 0.0)
                            for stats in level_gate_stats],
                        'boundary_weight_peak': [
                            stats.get('boundary_weight_peak', 0.0)
                            for stats in level_gate_stats],
                        'pc_contribution_ratio': [
                            stats.get('pc_contribution_ratio', 0.0)
                            for stats in level_gate_stats],
                    }
                    # When PC is consumed at C2, ``_build_pyramid`` already
                    # copied the adapter diagnostics here.  Do not overwrite
                    # those real values with four zero head-PC records merely
                    # because head_pc_modulation=False.
                    if self.use_head_pc_modulation:
                        self._dbg_pc_feat_norm = sum(
                            stats.get('pc_feat_norm', 0.0)
                            for stats in level_gate_stats) / max(
                                len(level_gate_stats), 1)
                        self._dbg_pc_gate_mean = sum(
                            stats.get('gate_mean', 0.0)
                            for stats in level_gate_stats) / max(
                                len(level_gate_stats), 1)
                        self._dbg_pc_residual_scale = sum(
                            stats.get('mod_scale', 0.0)
                            for stats in level_gate_stats) / max(
                                len(level_gate_stats), 1)
                        self._dbg_pc_contribution_ratio = sum(
                            stats.get('pc_contribution_ratio', 0.0)
                            for stats in level_gate_stats) / max(
                                len(level_gate_stats), 1)
                    self._dbg_sym_upper_prior_stats = {
                        name: [
                            stats.get(name, 0.0)
                            for stats in upper_prior_level_stats]
                        for name in (
                            'scale', 'delta_rel', 'upper_mean',
                            'middle_mean', 'lower_mean',
                            'support_fraction',
                            'upper_support_fraction',
                            'left_support_fraction',
                            'right_support_fraction')
                    }
                self._enhanced_debug_log(
                    feats, feat1_p2, fused_p2, lbp_lab,
                    contr_loss)

            # Keep this counter in training-iteration coordinates. Validation
            # forwards previously advanced it by 200 each evaluation, repeated
            # stale train stats in the log and could make the 500-step
            # loss-gradient diagnostic fall inside evaluation (and disappear).
            if self.training:
                self._debug_step += 1
            return (
                tuple(cls_scores),
                tuple(bbox_preds),
                contr_loss,
                boxalign_loss)

        def _fine_grained_distribution_loss(
                self,
                cls_scores,
                gt_bboxes,
                gt_labels,
                img_metas,
                gt_bboxes_ignore=None):
            """DFL-supervise the residual distribution on positive anchors."""
            logits_per_level = self._last_d_fine_logits
            base_per_level = self._last_d_fine_base_preds
            self._last_d_fine_logits = None
            self._last_d_fine_base_preds = None
            if logits_per_level is None or base_per_level is None:
                raise RuntimeError(
                    'D-FINE-style loss requires a preceding training forward')

            featmap_sizes = [
                featmap.size()[-2:] for featmap in cls_scores]
            device = cls_scores[0].device
            anchor_list, valid_flag_list = self.get_anchors(
                featmap_sizes, img_metas, device=device)
            label_channels = (
                self.cls_out_channels if self.use_sigmoid_cls else 1)
            targets = self.get_targets(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                label_channels=label_channels)
            if targets is None:
                return [
                    logits.sum() * 0.0
                    for logits in logits_per_level]
            (_, _, bbox_targets_list, bbox_weights_list,
             num_total_pos, _) = targets

            losses = []
            saturation_count = 0
            coordinate_count = 0
            target_abs_sum = 0.0
            scales = self.effective_d_fine_scales()
            for level_index, (logits, base_pred) in enumerate(zip(
                    logits_per_level, base_per_level)):
                batch, _, height, width = logits.shape
                bins = self.d_fine_reg_max + 1
                shaped_logits = logits.permute(
                    0, 2, 3, 1).reshape(
                        batch, height, width,
                        self.num_anchors, 4, bins).reshape(
                            -1, 4, bins)
                shaped_base = base_pred.permute(
                    0, 2, 3, 1).reshape(-1, 4)
                shaped_targets = bbox_targets_list[
                    level_index].reshape(-1, 4)
                shaped_weights = bbox_weights_list[
                    level_index].reshape(-1, 4)
                level_anchors = torch.stack([
                    per_image[level_index]
                    for per_image in anchor_list], dim=0).reshape(-1, 4)
                positive = shaped_weights.sum(dim=1) > 0
                if not positive.any():
                    losses.append(shaped_logits.sum() * 0.0)
                    continue

                if self.reg_decoded_bbox:
                    target_delta = self.bbox_coder.encode(
                        level_anchors[positive],
                        shaped_targets[positive])
                else:
                    target_delta = shaped_targets[positive]
                scale = scales[level_index].detach().float().clamp_min(
                    1e-6)
                normalized_target = (
                    target_delta.float()
                    - shaped_base[positive].detach().float()
                ) / scale
                positive_weights = shaped_weights[positive].float()
                saturation_count += int(
                    (normalized_target.abs() >= 1.0).sum().item())
                coordinate_count += int(normalized_target.numel())
                target_abs_sum += float(
                    normalized_target.abs().sum().item())
                raw_loss = _symmetric_distribution_focal_loss(
                    shaped_logits[positive].float(),
                    normalized_target,
                    positive_weights,
                    self.d_fine_reg_max,
                    avg_factor=max(num_total_pos * 4, 1))
                losses.append(self.d_fine_loss_weight * raw_loss)

            self._dbg_d_fine_loss_raw = float(
                sum(
                    loss.detach().float().item()
                    for loss in losses)
                / max(self.d_fine_loss_weight, 1e-12))
            self._dbg_d_fine_saturation = (
                saturation_count / max(coordinate_count, 1))
            self._dbg_d_fine_target_abs = (
                target_abs_sum / max(coordinate_count, 1))
            return losses

        def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
                 img_metas, gt_bboxes_ignore=None):
            losses = super().loss(
                cls_scores, bbox_preds, gt_bboxes, gt_labels,
                img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
            if losses is None or not self.use_d_fine_style:
                return losses
            losses['loss_dfr'] = self._fine_grained_distribution_loss(
                cls_scores,
                gt_bboxes,
                gt_labels,
                img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)
            return losses

        def _enhanced_debug_log(self, feats, feat1, fused, lbp_lab,
                                 contr_loss):
            """Per-module effectiveness diagnostics (every debug_interval iters).

            Sections:
              A: NAM Modulation (BRDG gate) — is the NAM branch changing features?
              B: FPN Feature Health — are features alive and diverse?
              C: LBP Embedding Quality — is the learned embedding being used?
              D: Cross-Scale — how much does each hop contribute?
              E: Contrastive Loss — are TB regions being found?
              F: PC Branch — is PC injection active and contributing?
              G: Gradient health
              H: Sym upper-lung positional prior — strength and soft profile
              I: CDF ROI and histogram-standardization health
            """
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank != 0:
                return

            B = feats[0].shape[0]

            # ==== A: NAM Modulation (BRDG gate) ====
            if feat1 is not None and fused is not None:
                delta = (fused - feat1).detach()
                delta_norm = delta.norm(dim=1).mean().item()
                delta_map = delta.norm(dim=1)
                all_mod = delta_map.mean().item()
                _, H_m, W_m = delta_map.shape
                flat_d = delta_map.reshape(B, -1)
                top_k = max(1, int(H_m * W_m * 0.2))
                edge_mod = flat_d.topk(top_k, dim=1)[0].mean().item()
                conc = edge_mod / (all_mod + 1e-8)
                cos_sim = F.cosine_similarity(
                    feat1.flatten(1), fused.flatten(1), dim=1).mean().item()
            else:
                delta_norm = conc = cos_sim = 0
                delta_map = None

            # ==== B: FPN Feature Health ====
            p_stats = []
            for i, f in enumerate(feats[:4]):
                f_d = f.detach()
                nrm = f_d.norm(dim=1).mean().item()
                dead = (f_d.std(dim=1) < 0.01).float().mean().item()
                p_stats.append((nrm, dead))

            # Inter-scale similarity matrix
            if len(feats) >= 4:
                p_feats = [F.adaptive_avg_pool2d(f.detach(), (16, 16)).flatten(1)
                           for f in feats[:4]]
                sim_vals = []
                for i in range(3):
                    s = F.cosine_similarity(
                        p_feats[i][:1], p_feats[i+1][:1], dim=1).item()
                    sim_vals.append(s)
            else:
                sim_vals = [0, 0, 0]

            # ==== C: LBP Embedding Quality ====
            if lbp_lab is not None and hasattr(self, 'lbp_embed'):
                lbp_d = lbp_lab.detach()
                lbp_svar = lbp_d.std(dim=[2, 3]).mean().item()  # spatial diversity
                lbp_mean = lbp_d.abs().mean().item()
                # Embedding weight gradient (is it learning?)
                emb_grad_norm = 0.0
                if self.lbp_embed.weight.grad is not None:
                    emb_grad_norm = self.lbp_embed.weight.grad.norm().item()
                # Embedding usage: cosine similarity between embedding vectors
                with torch.no_grad():
                    w = self.lbp_embed.weight  # (59, 32)
                    w_n = F.normalize(w, dim=1)
                    # Mean pairwise cosine similarity (low = diverse usage)
                    sim_mat = w_n @ w_n.T
                    mask = ~torch.eye(59, dtype=torch.bool, device=w.device)
                    emb_div = 1.0 - sim_mat[mask].mean().item()  # 0=all same, 1=orthogonal
            else:
                lbp_svar = lbp_mean = emb_grad_norm = emb_div = 0

            # ==== D: CrossScale bounded residual scales ====
            cs_scales = getattr(
                self, '_dbg_cs_scales', [0, 0, 0, 0, 0, 0])
            if len(cs_scales) < 6:
                cs_scales = list(cs_scales) + [0] * (6 - len(cs_scales))
            cs_stats = getattr(self, '_dbg_cs_stats', {})

            def stat_values(name, length, default=0.0):
                values = list(cs_stats.get(name, []))
                values += [default] * max(0, length - len(values))
                return values[:length]

            cs_scale_sca = stat_values('scale_sca', 6)
            cs_scale_cca = stat_values('scale_cca', 6)
            cs_fusion_base = stat_values(
                'fusion_base_weight', 6)
            cs_fusion_proposal = stat_values(
                'fusion_proposal_weight', 6, default=1.0)
            cs_rel_sca = stat_values('hop_rel_sca', 6)
            cs_rel_cca = stat_values('hop_rel_cca', 6)
            cs_rel_total = stat_values('hop_rel_total', 6)
            # _fuse records proposal-base before the optional normalized
            # fusion gate.  Report the effective output change as well so a
            # small Wcross cannot be mistaken for an active CrossScale hop.
            cs_rel_effective = [
                delta * weight
                for delta, weight in zip(
                    cs_rel_total, cs_fusion_proposal)]
            cs_delta_cos = stat_values('hop_delta_cos', 6)
            cs_branch_cos = stat_values('hop_branch_cos', 6)
            cs_sca_entropy = stat_values('sca_entropy', 6)
            cs_sca_peak = stat_values('sca_peak', 6)
            cs_cca_entropy = stat_values('cca_entropy', 6)
            cs_cca_peak = stat_values('cca_peak', 6)
            cs_cca_diag = stat_values('cca_diag_gain', 6)
            cs_down_rms = stat_values('downsample_rms_ratio', 3)
            cs_down_var = stat_values(
                'downsample_variation_ratio', 3)
            cs_refine_delta = stat_values('refine_rel_delta', 3)
            freq_alpf_entropy = stat_values(
                'freq_alpf_entropy', 2)
            freq_ahpf_entropy = stat_values(
                'freq_ahpf_entropy', 2)
            freq_offset_abs = stat_values(
                'freq_offset_abs', 2)
            freq_lr_delta = stat_values(
                'freq_lr_vs_bilinear', 2)
            freq_hr_delta = stat_values(
                'freq_hr_detail_rel', 2)
            freq_similarity = stat_values(
                'freq_local_similarity', 2)
            reg_detail_neck = getattr(
                self, '_dbg_reg_detail_neck_stats', {})
            reg_detail_head = (
                self.reg_detail_injector.debug_stats()
                if self.use_reg_detail_bypass
                else {})

            # ==== E: Contrastive Loss Breakdown ====
            cl_raw = contr_loss.item()
            contrast_loss_weight = getattr(
                self, '_dbg_contrast_loss_weight', 0.0)
            contrast_schedule_factor = getattr(
                self, '_dbg_contrast_schedule_factor', 0.0)
            contrast_epoch = getattr(self, '_dbg_contrast_epoch', 0)
            region_loss_raw = getattr(
                self, '_dbg_region_loss_raw', 0.0)
            region_loss_weighted = getattr(
                self, '_dbg_region_contrast_weighted', 0.0)
            # TB token stats are set on nam_branch while targets are built.
            if hasattr(self, 'nam_branch'):
                tb_token_count = getattr(self.nam_branch, '_dbg_tb_count', -1)
                total_token_count = getattr(self.nam_branch, '_dbg_total_tokens', -1)
                bbox_hit_ratio = getattr(self.nam_branch, '_dbg_bbox_hit_ratio', -1)
                tb_token_fraction = getattr(
                    self.nam_branch, '_dbg_tb_token_fraction', -1)
                region_purity = getattr(
                    self.nam_branch, '_dbg_region_purity_mean', -1)
                region_purity50 = getattr(
                    self.nam_branch, '_dbg_region_purity50', -1)
                best_bbox_purity = getattr(
                    self.nam_branch,
                    '_dbg_best_bbox_region_purity', -1)
                ambiguous_count = getattr(
                    self.nam_branch, '_dbg_ambiguous_count', -1)
                rejected_overlap_count = getattr(
                    self.nam_branch, '_dbg_rejected_overlap_count', -1)
                fallback_positive_gt_count = getattr(
                    self.nam_branch,
                    '_dbg_fallback_positive_gt_count', -1)
                region_total_gt = getattr(
                    self.nam_branch, '_dbg_total_gt_count', 0)
                selection_purity_threshold = getattr(
                    self.nam_branch,
                    '_dbg_selection_purity_threshold', -1.0)
                confirmed_negative_count = getattr(
                    self.nam_branch,
                    '_dbg_confirmed_negative_count', -1)
                invalid_pixel_fraction = getattr(
                    self.nam_branch,
                    '_dbg_invalid_pixel_fraction', -1)
                contrast_valid = getattr(
                    self.nam_branch,
                    '_dbg_contrast_valid_samples', 0)
                contrast_seen = getattr(
                    self.nam_branch,
                    '_dbg_contrast_seen_samples', 0)
                selected_negative_count = getattr(
                    self.nam_branch,
                    '_dbg_selected_negative_count', 0)
                selected_ring_count = getattr(
                    self.nam_branch,
                    '_dbg_selected_ring_count', 0)
                positive_weight_mean = getattr(
                    self.nam_branch,
                    '_dbg_positive_weight_mean', 0.0)
                hard_negative_similarity = getattr(
                    self.nam_branch,
                    '_dbg_hard_negative_similarity', 0.0)
                contrast_class_counts = getattr(
                    self.nam_branch,
                    '_dbg_class_counts', [0, 0])
                region_valid_instances = getattr(
                    self.nam_branch,
                    '_dbg_valid_instance_count', 0)
                region_valid_instance_anchors = getattr(
                    self.nam_branch,
                    '_dbg_valid_instance_anchors', 0)
                boundary_positive_count = getattr(
                    self.nam_branch,
                    '_dbg_boundary_positive_count', 0)
                single_region_gt_count = getattr(
                    self.nam_branch,
                    '_dbg_single_region_gt_count', 0)
                regions_per_gt_mean = getattr(
                    self.nam_branch,
                    '_dbg_regions_per_gt_mean', 0.0)
                gt_two_region_fraction = getattr(
                    self.nam_branch,
                    '_dbg_gt_two_region_fraction', 0.0)
                graph_ring_candidate_count = getattr(
                    self.nam_branch,
                    '_dbg_graph_ring_candidate_count', 0)
                region_positive_similarity = getattr(
                    self.nam_branch,
                    '_dbg_region_positive_similarity', 0.0)
                region_negative_similarity = getattr(
                    self.nam_branch,
                    '_dbg_region_negative_similarity', 0.0)
                boundary_rank_loss = getattr(
                    self.nam_branch,
                    '_dbg_boundary_rank_loss', 0.0)
            else:
                tb_token_count = total_token_count = bbox_hit_ratio = -1
                tb_token_fraction = region_purity = -1
                region_purity50 = best_bbox_purity = -1
                ambiguous_count = confirmed_negative_count = -1
                rejected_overlap_count = -1
                fallback_positive_gt_count = -1
                region_total_gt = 0
                selection_purity_threshold = -1.0
                invalid_pixel_fraction = -1
                contrast_valid = contrast_seen = 0
                selected_negative_count = selected_ring_count = 0
                positive_weight_mean = hard_negative_similarity = 0.0
                contrast_class_counts = [0, 0]
                region_valid_instances = 0
                region_valid_instance_anchors = 0
                boundary_positive_count = single_region_gt_count = 0
                regions_per_gt_mean = gt_two_region_fraction = 0.0
                graph_ring_candidate_count = 0
                region_positive_similarity = 0.0
                region_negative_similarity = 0.0
                boundary_rank_loss = 0.0
            instance_sim = -1.0
            if (hasattr(self, 'nam_branch')
                    and hasattr(
                        self.nam_branch, '_last_instance_sim')):
                instance_sim = self.nam_branch._last_instance_sim

            # ==== F: Regression-head PC Adapter Effectiveness ====
            pc_feat_norm = getattr(self, '_dbg_pc_feat_norm', 0)
            pc_gate_mean = getattr(self, '_dbg_pc_gate_mean', 0)
            pc_residual_scale = getattr(
                self, '_dbg_pc_residual_scale', 0)
            pc_contribution_ratio = getattr(
                self, '_dbg_pc_contribution_ratio', 0)
            pc_input_stats = getattr(
                self, '_dbg_pc_input_stats', {})

            # ==== F2: IoU-aware bbox regression-tower contrast ====
            bbox_contrast_raw = getattr(
                self, '_dbg_bbox_contrast_loss_raw', 0.0)
            bbox_contrast_weight = getattr(
                self, '_dbg_bbox_contrast_loss_weight', 0.0)
            bbox_contrast_positive_count = getattr(
                self, '_dbg_bbox_contrast_positive_count', 0)
            bbox_contrast_negative_count = getattr(
                self, '_dbg_bbox_contrast_negative_count', 0)
            bbox_contrast_valid_instances = getattr(
                self, '_dbg_bbox_contrast_valid_instances', 0)
            bbox_contrast_valid_anchors = getattr(
                self, '_dbg_bbox_contrast_valid_anchors', 0)
            bbox_contrast_mean_iou = getattr(
                self, '_dbg_bbox_contrast_mean_iou', 0.0)
            bbox_contrast_positive_similarity = getattr(
                self, '_dbg_bbox_contrast_positive_similarity', 0.0)
            bbox_contrast_total_gt = getattr(
                self, '_dbg_bbox_contrast_total_gt', 0)
            bbox_contrast_skipped_gt = getattr(
                self, '_dbg_bbox_contrast_skipped_gt', 0)
            bbox_contrast_gt_coverage = getattr(
                self, '_dbg_bbox_contrast_gt_coverage', 0.0)
            bbox_contrast_ambiguous_count = getattr(
                self, '_dbg_bbox_contrast_ambiguous_count', 0)
            bbox_contrast_positive_levels = list(getattr(
                self, '_dbg_bbox_contrast_positive_levels',
                [0] * len(feats)))
            bbox_contrast_negative_levels = list(getattr(
                self, '_dbg_bbox_contrast_negative_levels',
                [0] * len(feats)))
            bbox_contrast_negative_iou = getattr(
                self, '_dbg_bbox_contrast_mean_negative_iou', 0.0)
            bbox_contrast_positive_weight = getattr(
                self, '_dbg_bbox_contrast_positive_weight_mean', 0.0)
            bbox_contrast_weighted = getattr(
                self, '_dbg_bbox_contrast_weighted', 0.0)
            bbox_contrast_to_giou = getattr(
                self, '_dbg_bbox_contrast_to_giou_ratio', 0.0)
            boxalign_raw = getattr(
                self, '_dbg_boxalign_loss_raw', 0.0)
            boxalign_weight = getattr(
                self, '_dbg_boxalign_loss_weight', 0.0)
            boxalign_weighted = getattr(
                self, '_dbg_boxalign_weighted', 0.0)
            boxalign_total_gt = getattr(
                self, '_dbg_boxalign_total_gt', 0)
            boxalign_valid_gt = getattr(
                self, '_dbg_boxalign_valid_gt', 0)
            boxalign_anchor_count = getattr(
                self, '_dbg_boxalign_anchor_count', 0)
            boxalign_levels = list(getattr(
                self, '_dbg_boxalign_level_counts',
                [0] * len(feats)))
            boxalign_iou = getattr(
                self, '_dbg_boxalign_mean_iou', 0.0)
            boxalign_positive_similarity = getattr(
                self, '_dbg_boxalign_positive_similarity', 0.0)
            boxalign_negative_similarity = getattr(
                self, '_dbg_boxalign_negative_similarity', 0.0)
            boxalign_single_region_gt = getattr(
                self, '_dbg_boxalign_single_region_gt', 0)

            # ==== F3: D-FINE-style residual distribution refinement ====
            d_fine_scales = (
                self.effective_d_fine_scales()
                .detach().float().cpu().tolist())
            d_fine_entropy = list(getattr(
                self, '_dbg_d_fine_entropy', [0.0] * len(feats)))
            d_fine_expected_abs = list(getattr(
                self, '_dbg_d_fine_expected_abs',
                [0.0] * len(feats)))
            d_fine_loss_raw = getattr(
                self, '_dbg_d_fine_loss_raw', 0.0)
            d_fine_saturation = getattr(
                self, '_dbg_d_fine_saturation', 0.0)
            d_fine_target_abs = getattr(
                self, '_dbg_d_fine_target_abs', 0.0)

            # ==== G: Gradient Health ====
            cs_grad_stats = getattr(self, '_dbg_cs_grad_stats', {})
            cs_output_grad = list(
                cs_grad_stats.get('output_grad_rms', [0] * 4))
            cs_group_grad = cs_grad_stats.get('group_grad_rms', {})
            cs_group_coverage = cs_grad_stats.get(
                'group_grad_coverage', {})
            module_grad_stats = getattr(
                self, '_dbg_module_grad_stats', {})
            module_group_grad = module_grad_stats.get(
                'group_grad_rms', {})
            module_group_coverage = module_grad_stats.get(
                'group_grad_coverage', {})
            grad_source_step = module_grad_stats.get('source_step', -1)
            loss_grad_stats = getattr(
                self, '_dbg_loss_grad_stats', {})
            # Reading gradients in the forward pass is unreliable because
            # OptimizerHook normally clears them first. Use NAMDebugHook's
            # previous-iteration snapshot instead.
            emb_grad_norm = module_group_grad.get(
                'lbp_embedding', emb_grad_norm)
            # Contrast head weight norm
            ch_w = 0.0
            if hasattr(self, 'nam_branch'):
                try:
                    ch_w = sum(p.data.norm().item()
                              for p in self.nam_branch.contrast_head.parameters())
                except Exception:
                    pass

            # ---- Format output ----
            p2_n, p2_d = p_stats[0] if p_stats else (0, 0)
            p3_n, p3_d = p_stats[1] if len(p_stats) > 1 else (0, 0)
            p4_n, p4_d = p_stats[2] if len(p_stats) > 2 else (0, 0)
            p5_n, p5_d = p_stats[3] if len(p_stats) > 3 else (0, 0)

            sim_str = f'{sim_vals[0]:.2f}/{sim_vals[1]:.2f}/{sim_vals[2]:.2f}' if sim_vals else 'N/A'
            mode = 'train' if self.training else 'eval'

            _emit_training_debug(
                f'[DBG] mode={mode} iter={self._debug_step} '
                # A: NAM
                f'Δ={delta_norm:.2f} conc={conc:.2f} cos={cos_sim:.3f} | '
                # B: FPN norms
                f'Fn=[{p2_n:.0f}/{p3_n:.0f}/{p4_n:.0f}/{p5_n:.0f}] '
                f'Fd=[{p2_d:.4f}/{p3_d:.4f}/{p4_d:.4f}/{p5_d:.4f}] '
                f'Fsim=[{sim_str}] | '
                # C: LBP Emb
                f'E_s={lbp_svar:.3f} E_m={lbp_mean:.3f} '
                f'E_g={emb_grad_norm:.4f} E_div={emb_div:.2f} | '
                # D: CrossScale detail-up / semantic-down residual scales
                f'CSbu=[{cs_scales[0]:.4f}/{cs_scales[1]:.4f}/'
                f'{cs_scales[2]:.4f}] '
                f'CStd=[{cs_scales[3]:.4f}/{cs_scales[4]:.4f}/'
                f'{cs_scales[5]:.4f}] | '
                # E+F: Contrastive + PC
                f'Lc={cl_raw:.4f} tb#={tb_token_count}/{total_token_count} '
                f'bb_hit={bbox_hit_ratio:.2f} '
                f'inst_sim={instance_sim:.2f} | '
                f'PC_n={pc_feat_norm:.3f} PC_g={pc_gate_mean:.3f} '
                f'PC_α={pc_residual_scale:.4f} '
                f'PC_r={pc_contribution_ratio:.5f} | '
                f'ch_w={ch_w:.1f}')
            if pc_input_stats:
                source_paths = pc_input_stats.get('paths', [])
                path_sample = (
                    os.path.basename(source_paths[0])
                    if source_paths and source_paths[0] else 'none')
                _emit_training_debug(
                    f'[PC-INPUT] iter={self._debug_step} '
                    f'source={pc_input_stats.get("source", "unknown")} '
                    f'μ={pc_input_stats.get("mean", 0.0):.2f} '
                    f'σ={pc_input_stats.get("std", 0.0):.2f} '
                    f'range={pc_input_stats.get("min", 0.0):.1f}-'
                    f'{pc_input_stats.get("max", 0.0):.1f} '
                    f'raw_mae={pc_input_stats.get("raw_mae", 0.0):.2f} '
                    f'raw_corr={pc_input_stats.get("raw_corr", 0.0):.3f} '
                    f'file={path_sample}')

            def slash(values, pattern):
                return '/'.join(format(value, pattern) for value in values)

            _emit_training_debug(
                f'[CS-FWD] iter={self._debug_step} '
                f'αS=[{slash(cs_scale_sca, ".2e")}] '
                f'αC=[{slash(cs_scale_cca, ".2e")}] | '
                f'ΔS=[{slash(cs_rel_sca, ".2e")}] '
                f'ΔC=[{slash(cs_rel_cca, ".2e")}] '
                f'Δtot=[{slash(cs_rel_total, ".2e")}] '
                f'Δeff=[{slash(cs_rel_effective, ".2e")}] | '
                f'cosΔ=[{slash(cs_delta_cos, ".2f")}] '
                f'cosSC=[{slash(cs_branch_cos, ".2f")}] | '
                f'Wbase=[{slash(cs_fusion_base, ".3f")}] '
                f'Wcross=[{slash(cs_fusion_proposal, ".3f")}]')
            _emit_training_debug(
                f'[CS-ATTN] iter={self._debug_step} '
                f'Hs=[{slash(cs_sca_entropy, ".3f")}] '
                f'Ps=[{slash(cs_sca_peak, ".3f")}] | '
                f'Hc=[{slash(cs_cca_entropy, ".3f")}] '
                f'Pc=[{slash(cs_cca_peak, ".3f")}] '
                f'Dg=[{slash(cs_cca_diag, ".2f")}] | '
                f'DSrms=[{slash(cs_down_rms, ".2f")}] '
                f'DSvar=[{slash(cs_down_var, ".2f")}] '
                f'RefΔ=[{slash(cs_refine_delta, ".2e")}]')
            _emit_training_debug(
                f'[FREQ] iter={self._debug_step} '
                f'ALPF-H=[{slash(freq_alpf_entropy, ".3f")}] '
                f'AHPF-H=[{slash(freq_ahpf_entropy, ".3f")}] '
                f'offset=[{slash(freq_offset_abs, ".3e")}] '
                f'LRΔ=[{slash(freq_lr_delta, ".3e")}] '
                f'HRΔ=[{slash(freq_hr_delta, ".3e")}] '
                f'sim=[{slash(freq_similarity, ".3f")}]')
            _emit_training_debug(
                f'[REG-DETAIL] iter={self._debug_step} '
                f'HP={reg_detail_neck.get("highpass_ratio", 0.0):.3f} '
                f'gate=[{slash(reg_detail_neck.get("gate_mean", [0, 0]), ".3f")}] '
                f'Drms=[{slash(reg_detail_neck.get("detail_rms", [0, 0]), ".3e")}] '
                f'α=[{slash(reg_detail_head.get("scale", [0] * 4), ".3f")}] '
                f'Δrel=[{slash(reg_detail_head.get("rel_delta", [0] * 4), ".3e")}] '
                f'Hrms=[{slash(reg_detail_head.get("detail_rms", [0] * 4), ".3e")}]')

            nam_levels = getattr(self, '_dbg_nam_level_stats', {})
            _emit_training_debug(
                f'[NAM-LVL] iter={self._debug_step} '
                f'Δrel=[{slash(nam_levels.get("rel_change", [0] * 4), ".2e")}] '
                f'cos=[{slash(nam_levels.get("cosine", [0] * 4), ".4f")}] '
                f'gate=[{slash(nam_levels.get("gate_mean", [0] * 4), ".3f")}] '
                f'act=[{slash(nam_levels.get("gate_active", [0] * 4), ".3f")}] '
                f'PCr=[{slash(nam_levels.get("pc_contribution_ratio", [0] * 4), ".2e")}] '
                f'BH=[{slash(nam_levels.get("boundary_weight_entropy", [0] * 4), ".3f")}] '
                f'BP=[{slash(nam_levels.get("boundary_weight_peak", [0] * 4), ".3f")}]')
            sym_prior = getattr(self, '_dbg_sym_upper_prior_stats', {})
            _emit_training_debug(
                f'[SYM-POS] iter={self._debug_step} '
                f'α=[{slash(sym_prior.get("scale", [0] * 4), ".3f")}] '
                f'Δrel=[{slash(sym_prior.get("delta_rel", [0] * 4), ".2e")}] '
                f'U=[{slash(sym_prior.get("upper_mean", [0] * 4), ".3f")}] '
                f'M=[{slash(sym_prior.get("middle_mean", [0] * 4), ".3f")}] '
                f'L=[{slash(sym_prior.get("lower_mean", [0] * 4), ".3f")}] | '
                f'mask=[{slash(sym_prior.get("support_fraction", [0] * 4), ".3f")}] '
                f'up%=[{slash(sym_prior.get("upper_support_fraction", [0] * 4), ".3f")}] '
                f'LR=[{slash(sym_prior.get("left_support_fraction", [0] * 4), ".3f")};'
                f'{slash(sym_prior.get("right_support_fraction", [0] * 4), ".3f")}]')
            roi_stats = getattr(self, '_dbg_roi_stats', {})
            recorded_pad_text = '/'.join(
                f'{int(shape[0])}x{int(shape[1])}'
                for shape in roi_stats.get(
                    'recorded_pad_shapes', [])) or 'none'
            batch_canvas = roi_stats.get('batch_canvas', (0, 0))
            batch_canvas_text = (
                f'{int(batch_canvas[0])}x{int(batch_canvas[1])}'
                if len(batch_canvas) == 2 else 'none')
            if roi_stats.get('enabled', False):
                fallback_reasons = ','.join(
                    roi_stats.get('fallback_reasons', [])) or 'none'
                _emit_training_debug(
                    f'[CXR-ROI] iter={self._debug_step} '
                    f'applied={roi_stats.get("applied_fraction", 0.0):.2f} '
                    f'fallback={roi_stats.get("fallback_fraction", 0.0):.2f} '
                    f'area={roi_stats.get("retained_area", 1.0):.3f} '
                    f'gain={roi_stats.get("linear_scale_gain", 1.0):.3f} '
                    f'lung_cov={roi_stats.get("lung_coverage", 1.0):.4f} '
                    f'box_cov_min={roi_stats.get("bbox_min_coverage", 1.0):.4f} | '
                    f'μ={roi_stats.get("mean_before", 0.0):.1f}'
                    f'->{roi_stats.get("mean_after", 0.0):.1f} '
                    f'σ={roi_stats.get("std_before", 0.0):.1f}'
                    f'->{roi_stats.get("std_after", 0.0):.1f} '
                    f'deg={roi_stats.get("degenerate_fraction", 0.0):.2f} '
                    f'reason={fallback_reasons} | '
                    f'pad=[{recorded_pad_text}]->{batch_canvas_text} '
                    f'extra={roi_stats.get("collate_extra_pad_fraction", 0.0):.3f}')
            else:
                _emit_training_debug(
                    f'[INPUT] iter={self._debug_step} '
                    f'mode={roi_stats.get("mode", "raw_imagenet")} '
                    f'pad=[{recorded_pad_text}]->{batch_canvas_text} '
                    f'extra={roi_stats.get("collate_extra_pad_fraction", 0.0):.3f}')
            _emit_training_debug(
                f'[REGION-CONTR] iter={self._debug_step} '
                f'valid={contrast_valid}/{contrast_seen} '
                f'ep={contrast_epoch} sched={contrast_schedule_factor:.3f} '
                f'λ={contrast_loss_weight:.4f} '
                f'Lraw/w={region_loss_raw:.4f}/'
                f'{region_loss_weighted:.4f} '
                f'tb_frac={tb_token_fraction:.3f} '
                f'pur={region_purity:.3f} '
                f'pur50={region_purity50:.3f} '
                f'pos_thr={selection_purity_threshold:.3f} '
                f'best_box_pur={best_bbox_purity:.3f} '
                f'boundary={boundary_positive_count} '
                f'amb={ambiguous_count} rej={rejected_overlap_count} '
                f'fallback={fallback_positive_gt_count}/'
                f'{region_total_gt}'
                f'({fallback_positive_gt_count / max(region_total_gt, 1):.3f}) '
                f'neg={confirmed_negative_count} | '
                f'R/GT={regions_per_gt_mean:.2f} '
                f'GT>=2={gt_two_region_fraction:.3f} '
                f'GTsingle={single_region_gt_count} | '
                f'sel={selected_negative_count} '
                f'ring={selected_ring_count}/{graph_ring_candidate_count} '
                f'w+={positive_weight_mean:.3f} '
                f'sim+/-={region_positive_similarity:.3f}/'
                f'{region_negative_similarity:.3f} '
                f'Lrank={boundary_rank_loss:.4f} '
                f'inst={region_valid_instances} '
                f'inst_anchor={region_valid_instance_anchors} '
                f'cls={contrast_class_counts[0]}/{contrast_class_counts[1]} '
                f'invalid_px={invalid_pixel_fraction:.3f} '
                f'(bb_hit={bbox_hit_ratio:.3f})')
            _emit_training_debug(
                f'[BOXALIGN] iter={self._debug_step} '
                f'λ={boxalign_weight:.4f} '
                f'Lraw/w={boxalign_raw:.4f}/{boxalign_weighted:.4f} '
                f'GT={boxalign_valid_gt}/{boxalign_total_gt} '
                f'single={boxalign_single_region_gt} '
                f'anchors={boxalign_anchor_count} '
                f'P=[{slash(boxalign_levels, "d")}] '
                f'IoU={boxalign_iou:.3f} '
                f'sim+/-={boxalign_positive_similarity:.3f}/'
                f'{boxalign_negative_similarity:.3f}')
            _emit_training_debug(
                f'[BBOX-CONTR-prev] src_iter={max(0, self._debug_step - 1)} '
                f'λ={bbox_contrast_weight:.3f} '
                f'Lraw/w={bbox_contrast_raw:.4f}/'
                f'{bbox_contrast_weighted:.4f} '
                f'L/G={bbox_contrast_to_giou:.3f} | '
                f'GT={bbox_contrast_valid_instances}/'
                f'{bbox_contrast_total_gt} '
                f'cov={bbox_contrast_gt_coverage:.3f} '
                f'skip={bbox_contrast_skipped_gt} '
                f'valid_anchor={bbox_contrast_valid_anchors} '
                f'pos/neg={bbox_contrast_positive_count}/'
                f'{bbox_contrast_negative_count} '
                f'amb={bbox_contrast_ambiguous_count} | '
                f'P=[{slash(bbox_contrast_positive_levels, "d")}] '
                f'N=[{slash(bbox_contrast_negative_levels, "d")}] | '
                f'IoU+/-={bbox_contrast_mean_iou:.3f}/'
                f'{bbox_contrast_negative_iou:.3f} '
                f'w+={bbox_contrast_positive_weight:.3f} '
                f'sim+={bbox_contrast_positive_similarity:.3f}')
            _emit_training_debug(
                f'[DFINE] iter={self._debug_step} '
                f'Lsrc={max(0, self._debug_step - 1)} '
                f'on={int(self.use_d_fine_style)} '
                f'K={self.d_fine_reg_max + 1} '
                f'λ={self.d_fine_loss_weight:.3f} '
                f'Lraw_prev={d_fine_loss_raw:.4f} '
                f'sat_prev={d_fine_saturation:.3f} '
                f'|t_res|_prev={d_fine_target_abs:.3f} | '
                f'Fcur:'
                f'α=[{slash(d_fine_scales, ".3f")}] '
                f'H=[{slash(d_fine_entropy, ".3f")}] '
                f'|E|=[{slash(d_fine_expected_abs, ".3e")}]')

            cs_grad_order = [
                'bottom_up_downsample', 'bottom_up_sca',
                'bottom_up_cca', 'top_down_sca', 'top_down_cca',
                'output_refine', 'detail_scale', 'semantic_scale',
                'fusion_weight', 'freqfusion']
            module_grad_order = [
                'pc_adapter', 'lbp_branch', 'br_gate',
                'region_transformer', 'contrast_head',
                'boxalign', 'boxalign_anchor_embed',
                'bbox_contrast', 'bbox_anchor_embed',
                'symformer', 'lbp_embedding',
                'd_fine_reg', 'd_fine_scale',
                'reg_detail_generator', 'reg_detail_injector']
            cs_grad_values = [
                cs_group_grad.get(name, 0.0)
                for name in cs_grad_order]
            module_grad_values = [
                module_group_grad.get(name, 0.0)
                for name in module_grad_order]
            cs_coverage = [
                cs_group_coverage.get(name, 0.0)
                for name in cs_grad_order]
            module_coverage = [
                module_group_coverage.get(name, 0.0)
                for name in module_grad_order]
            _emit_training_debug(
                f'[GRAD-prev] src_iter={grad_source_step} '
                f'P=[{slash(cs_output_grad, ".2e")}] | '
                f'CS(DS/S/C/tS/tC/R/aD/aS/W/Freq)='
                f'[{slash(cs_grad_values, ".2e")}] '
                f'cov=[{slash(cs_coverage, ".2f")}] | '
                f'MOD(PC/LBP/BR/RT/CH/BA/BAEmb/BBoxC/AEmb/'
                f'Sym/Emb/DFR/DFRα/RDgen/RDinj)='
                f'[{slash(module_grad_values, ".2e")}] '
                f'cov=[{slash(module_coverage, ".2f")}]')
            if loss_grad_stats:
                region_loss_grad = loss_grad_stats.get(
                    'region', loss_grad_stats.get('c2', {}))
                region_source = loss_grad_stats.get(
                    'region_source', 'c2')
                reg_loss_grad = loss_grad_stats.get('reg', {})
                _emit_training_debug(
                    f'[AUX-GRAD-prev] '
                    f'src_iter={loss_grad_stats.get("source_step", -1)} '
                    f'Rsrc={region_source}(aux/det/cos)='
                    f'{region_loss_grad.get("aux_rms", 0.0):.2e}/'
                    f'{region_loss_grad.get("det_rms", 0.0):.2e}/'
                    f'{region_loss_grad.get("cosine", 0.0):.3f} '
                    f'REG(aux/det/cos)='
                    f'{reg_loss_grad.get("aux_rms", 0.0):.2e}/'
                    f'{reg_loss_grad.get("det_rms", 0.0):.2e}/'
                    f'{reg_loss_grad.get("cosine", 0.0):.3f}')
            det_stats = getattr(self, '_dbg_det_loss_stats', {})
            gt_sizes = list(det_stats.get(
                'gt_size_counts', [0, 0, 0]))
            class_counts = det_stats.get('gt_class_counts', {})
            class_text = '/'.join(
                f'{key}:{class_counts[key]}'
                for key in sorted(class_counts)) or 'none'
            _emit_training_debug(
                f'[DET-prev] src_iter={det_stats.get("source_step", -1)} '
                f'GT(s/m/l)=[{slash(gt_sizes, "d")}] '
                f'GTcls=[{class_text}] | '
                f'Lcls=[{slash(det_stats.get("loss_cls", [0] * 4), ".3e")}] '
                f'Lbox=[{slash(det_stats.get("loss_bbox", [0] * 4), ".3e")}] | '
                f'Lregcon=[{slash(det_stats.get("loss_region_contrast", [0]), ".3e")}] '
                f'Lalign=[{slash(det_stats.get("loss_boxalign", [0]), ".3e")}] | '
                f'Lbcon=[{slash(det_stats.get("loss_bbox_contrast", [0]), ".3e")}] '
                f'Ldfr=[{slash(det_stats.get("loss_dfr", [0]), ".3e")}] '
                f'Lb/G={det_stats.get("bbox_contrast_to_giou_ratio", 0.0):.3f} | '
                f'|δ|=[{slash(det_stats.get("bbox_delta_abs", [0] * 4), ".2e")}] '
                f'maxδ=[{slash(det_stats.get("bbox_delta_max", [0] * 4), ".2e")}] '
                f'finite=[{slash(det_stats.get("bbox_finite", [0] * 4), ".3f")}]')
            # Store for deep diagnose
            self._dbg_delta_map = delta_map

            # Expensive feature-rank diagnosis has its own interval.
            if (self.deep_debug_interval > 0
                    and self._debug_step
                    % self.deep_debug_interval == 0):
                self._deep_diagnose(feats, feat1, fused, delta_map)

        def _deep_diagnose(self, feats, feat1, fused, delta_map):
            """Occasional FPN-rank, modulation and embedding analysis."""
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
                _emit_training_debug(
                    f'[DEEP] iter={self._debug_step} FPN-cos:\n'
                    + '\n'.join(
                        f'  P{i+2}: {r}'
                        for i, r in enumerate(sim_rows)))

            # ---- FPN effective rank (95% energy) ----
            if len(feats) >= 4:
                raw_ranks, centered_ranks, stable_ranks = [], [], []

                def rank_metrics(matrix):
                    _, singular, _ = torch.linalg.svd(
                        matrix, full_matrices=False)
                    energy = singular.square()
                    total = energy.sum()
                    if total <= 1e-12:
                        return 0, 0.0
                    ratio = torch.cumsum(energy, 0) / total
                    rank95 = int(
                        (ratio < 0.95).sum().item() + 1)
                    stable_rank = float(
                        (total / energy.max().clamp_min(1e-12)).item())
                    return rank95, stable_rank

                for i, f in enumerate(feats[:4]):
                    f_flat = f.detach().float().reshape(
                        f.shape[0], f.shape[1], -1)[0]
                    raw_rank, stable_rank = rank_metrics(f_flat)
                    centered_rank, _ = rank_metrics(
                        f_flat - f_flat.mean(dim=1, keepdim=True))
                    raw_ranks.append(raw_rank)
                    centered_ranks.append(centered_rank)
                    stable_ranks.append(stable_rank)
                _emit_training_debug(
                    f'[DEEP] iter={self._debug_step} FPN-rank '
                    f'raw95=[{"/".join(map(str, raw_ranks))}] '
                    f'center95=[{"/".join(map(str, centered_ranks))}] '
                    f'stable=[{"/".join(f"{v:.1f}" for v in stable_ranks)}]')

            # ---- Modulation concentration ----
            if delta_map is not None:
                B, H, W = delta_map.shape
                flat_d = delta_map.reshape(B, -1)
                top_k = max(1, int(H * W * 0.2))
                edge_mod = flat_d.topk(top_k, dim=1)[0].mean().item()
                all_mod = flat_d.mean().item()
                conc = edge_mod / (all_mod + 1e-8)
                _emit_training_debug(
                    f'[DEEP] iter={self._debug_step} mod_conc={conc:.2f} '
                    f'(>1.5=edges, ≈1.0=uniform)')

            # ---- LBP Embedding nearest-neighbor analysis ----
            if hasattr(self, 'lbp_embed'):
                with torch.no_grad():
                    w = F.normalize(self.lbp_embed.weight, dim=1)  # (59, 32)
                    sim = w @ w.T  # (59, 59)
                    # Per-pattern nearest neighbor similarity
                    sim.fill_diagonal_(-1)
                    nn_sim = sim.max(dim=1)[0]  # (59,)
                    # Patterns where NN is very similar (potential collapse)
                    collapsed = (nn_sim > 0.95).sum().item()
                    mean_nn = nn_sim.mean().item()
                _emit_training_debug(
                    f'[DEEP] iter={self._debug_step} LBP-emb: '
                    f'mean_NN_sim={mean_nn:.3f} '
                    f'collapsed(>0.95)={collapsed}/59 '
                    f'({"COLLAPSING" if collapsed > 5 else "healthy"})')

            # ---- Cross-Scale attention effectiveness ----
            if len(feats) >= 4:
                norms = []
                for i, f in enumerate(feats[:4]):
                    norms.append(f'{f.detach().norm(dim=1).mean().item():.0f}')
                _emit_training_debug(
                    f'[DEEP] iter={self._debug_step} CrossScale-norms: '
                    f'P2e={norms[0]} P3e={norms[1]} '
                    f'P4e={norms[2]} P5e={norms[3]}')

        def simple_test(self, feats, img_metas, rescale=False, nam_graphs=None,
                        lbp_lab=None, c2_feat=None, region_feat=None,
                        lung_masks=None, reg_detail_feats=None):
            """Override parent to pass auxiliary branches to ``forward``."""
            outs = self.forward(
                feats,
                nam_graphs=nam_graphs,
                lbp_lab=lbp_lab,
                c2_feat=c2_feat,
                region_feat=region_feat,
                lung_masks=lung_masks,
                reg_detail_feats=reg_detail_feats)
            results_list = self.get_bboxes(
                outs[0], outs[1], img_metas, rescale=rescale)
            return results_list


    # ============================================================================
    # RetinaNAMAttV3: Detector
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
    class RetinaNAMAttV3(SingleStageDetector):
        """Single-stage detector with Dynamic Region Transformer NAM branch.

        v3.12 NAM-BoxAlign stack:
          - PC+LBP is injected once at the exported C2 seam
          - CrossScale neck directly replaces FPN
          - Dynamic region token pooling from enhanced C2
          - Lung-mask-normalized upper-third positional prior
          - Raw ImageNet-normalized CXR input (adaptive ROI preprocessing off)
          - Soft NAM memberships, per-GT topology contrast and P2-P4 BoxAlign

        Config:
            model=dict(
                type='RetinaNAMAttV3',
                backbone=dict(...),
                neck=dict(...),
                bbox_head=dict(type='RetinaNAMGuideAttHeadV3', ...),
            )
        """

        def __init__(self,
                     *args,
                     pc_branch: bool = False,
                     pc_residual_init: float = 1e-2,
                     pc_residual_max: float = 0.1,
                     pc_image_source: str = 'raw',
                     region_feature_source: str = 'c2',
                     input_preprocess: str = 'raw_imagenet',
                     contrast_loss_weight: float = 0.0,
                     region_contrast_loss_weight=None,
                     boxalign_loss_weight=None,
                     contrast_warmup_epochs: int = 2,
                     contrast_start_epoch: int = 2,
                     contrast_ramp_epochs: int = 6,
                     contrast_hold_end_epoch: int = 16,
                     contrast_decay_end_epoch: int = 22,
                     experiment_tag: str = 'unspecified',
                     **kwargs):
            super().__init__(*args, **kwargs)
            self._nam_epoch = 0
            if contrast_loss_weight < 0:
                raise ValueError('contrast_loss_weight must be non-negative')
            if (region_contrast_loss_weight is not None
                    and region_contrast_loss_weight < 0):
                raise ValueError(
                    'region_contrast_loss_weight must be non-negative')
            if (boxalign_loss_weight is not None
                    and boxalign_loss_weight < 0):
                raise ValueError(
                    'boxalign_loss_weight must be non-negative')
            if contrast_warmup_epochs < 0:
                raise ValueError(
                    'contrast_warmup_epochs must be non-negative')
            self.contrast_loss_weight = float(contrast_loss_weight)
            # Keep ``contrast_loss_weight`` as a backwards-compatible default,
            # but expose independent coefficients for controlled ablations.
            # BoxAlign still uses the same epoch schedule; only its magnitude
            # is decoupled from the region objective.
            self.region_contrast_loss_weight = float(
                contrast_loss_weight
                if region_contrast_loss_weight is None
                else region_contrast_loss_weight)
            self.boxalign_loss_weight = float(
                contrast_loss_weight * self.bbox_head.boxalign_loss_ratio
                if boxalign_loss_weight is None
                else boxalign_loss_weight)
            self.use_region_auxiliaries = bool(
                self.region_contrast_loss_weight > 0.0
                or self.boxalign_loss_weight > 0.0)
            self.experiment_tag = str(experiment_tag)
            # ``contrast_warmup_epochs`` remains accepted for old configs but
            # v3.12 uses a detector-only start, ramp, hold and late decay.
            self.contrast_warmup_epochs = int(contrast_warmup_epochs)
            self.contrast_start_epoch = int(contrast_start_epoch)
            self.contrast_ramp_epochs = int(contrast_ramp_epochs)
            self.contrast_hold_end_epoch = int(contrast_hold_end_epoch)
            self.contrast_decay_end_epoch = int(contrast_decay_end_epoch)
            _contrast_schedule_factor(
                epoch=1,
                start_epoch=self.contrast_start_epoch,
                ramp_epochs=self.contrast_ramp_epochs,
                hold_end_epoch=self.contrast_hold_end_epoch,
                decay_end_epoch=self.contrast_decay_end_epoch)
            self.input_preprocess = str(input_preprocess)
            self.use_pc_branch = bool(pc_branch)
            self.pc_image_source = str(pc_image_source).lower()
            if self.pc_image_source not in ('raw', 'bone_suppressed'):
                raise ValueError(
                    'pc_image_source must be "raw" or "bone_suppressed", '
                    f'got {pc_image_source!r}')
            self.region_feature_source = str(
                region_feature_source).lower()
            if self.region_feature_source not in ('c2', 'c5', 'p5'):
                raise ValueError(
                    'region_feature_source must be one of "c2", "c5", '
                    f'or "p5", got {region_feature_source!r}')
            neck_in_channels = getattr(
                self.neck, 'in_channels', [72, 144, 288, 576])
            neck_reg_detail = bool(getattr(
                self.neck, 'use_reg_detail_bypass', False))
            if neck_reg_detail != self.bbox_head.use_reg_detail_bypass:
                raise ValueError(
                    'neck.reg_detail_bypass and '
                    'bbox_head.reg_detail_bypass must match')
            expected_region_channels = {
                'c2': int(neck_in_channels[0]),
                'c5': int(neck_in_channels[-1]),
                'p5': int(getattr(self.neck, 'out_channels', 256)),
            }[self.region_feature_source]
            if self.bbox_head.region_in_channels != expected_region_channels:
                raise ValueError(
                    f'region_feature_source={self.region_feature_source!r} '
                    f'provides {expected_region_channels} channels, but '
                    'bbox_head.region_in_channels='
                    f'{self.bbox_head.region_in_channels}')
            if (self.use_pc_branch
                    and self.bbox_head.use_head_pc_modulation):
                raise ValueError(
                    'Enable only one PC consumer: pc_branch or '
                    'bbox_head.head_pc_modulation, not both')
            if self.use_pc_branch:
                if not self.with_neck:
                    raise ValueError(
                        'pc_branch requires a neck receiving enhanced C2')
                self.pc_shallow_adapter = _PCShallowAdapter(
                    pc_channels=576,
                    c2_channels=int(neck_in_channels[0]),
                    residual_init=float(pc_residual_init),
                    residual_max=float(pc_residual_max))
            else:
                self.pc_shallow_adapter = None

        def _build_pyramid(self, backbone_feats, lbp_lab=None):
            """Apply the optional C2 PC adapter, then build CrossScale maps."""
            neck_inputs = list(backbone_feats)
            c2_feat = neck_inputs[0]
            if self.use_pc_branch:
                if lbp_lab is None:
                    raise RuntimeError(
                        'pc_branch=True requires a computed PC/LBP prior')
                c2_feat = self.pc_shallow_adapter(c2_feat, lbp_lab)
                neck_inputs[0] = c2_feat
                adapter = self.pc_shallow_adapter
                self.bbox_head._dbg_pc_feat_norm = (
                    adapter._dbg_pc_feat_norm)
                self.bbox_head._dbg_pc_gate_mean = (
                    adapter._dbg_pc_gate_mean)
                self.bbox_head._dbg_pc_residual_scale = (
                    adapter._dbg_residual_scale)
                self.bbox_head._dbg_pc_contribution_ratio = (
                    adapter._dbg_contribution_ratio)

            cross_scale = getattr(
                getattr(self, 'neck', None), 'cross_scale', None)
            debug_interval = int(getattr(
                self.bbox_head, 'debug_interval', 0))
            debug_step = int(getattr(
                self.bbox_head, '_debug_step', 0))
            forward_debug_due = (
                debug_interval > 0
                and debug_step % debug_interval == 0)
            gradient_debug_due = (
                self.training and debug_interval > 0
                and (debug_step + 1) % debug_interval == 0)
            if cross_scale is not None:
                cross_scale.collect_forward_debug = forward_debug_due
                cross_scale.collect_gradient_debug = gradient_debug_due
            detail_generator = getattr(
                getattr(self, 'neck', None),
                'reg_detail_generator', None)
            if detail_generator is not None:
                detail_generator._collect_debug = forward_debug_due

            if self.with_neck:
                pyramid_feats = self.neck(tuple(neck_inputs))
            else:
                pyramid_feats = tuple(neck_inputs)

            if cross_scale is not None and forward_debug_due:
                with torch.no_grad():
                    stats = cross_scale.debug_stats()
                    self.bbox_head._dbg_cs_stats = stats
                    self.bbox_head._dbg_cs_scales = [
                        (abs(sca) + abs(cca)) * 0.5
                        for sca, cca in zip(
                            stats['scale_sca'],
                            stats['scale_cca'])]
            if detail_generator is not None and forward_debug_due:
                self.bbox_head._dbg_reg_detail_neck_stats = (
                    detail_generator.debug_stats())
            return pyramid_feats, c2_feat

        def _collect_bone_suppressed_images(
                self, img_metas, device, batch_input_shape=None):
            """Align and batch the strict paired PC-only CXR input."""
            aligned_images = []
            source_paths = []
            for index, meta in enumerate(img_metas):
                bone_image = (
                    meta.get('bone_suppressed_img', None)
                    if isinstance(meta, dict) else None)
                if bone_image is None:
                    filename = (
                        meta.get('filename', '<unknown>')
                        if isinstance(meta, dict) else '<unknown>')
                    raise KeyError(
                        f'Missing bone_suppressed_img for batch item '
                        f'{index}: {filename}. Add LoadBoneSuppressedCXR to '
                        f'both train and test pipelines and collect it in '
                        f'img_metas.')
                aligned_images.append(
                    _align_bone_suppressed_to_image_meta(
                        bone_image, meta,
                        batch_input_shape=batch_input_shape))
                source_paths.append(
                    meta.get('bone_suppressed_img_path', ''))

            stacked = torch.stack(aligned_images).unsqueeze(1).to(
                device=device, dtype=torch.float32)
            debug_interval = int(getattr(
                self.bbox_head, 'debug_interval', 0))
            debug_step = int(getattr(
                self.bbox_head, '_debug_step', 0))
            if (debug_interval > 0
                    and debug_step % debug_interval == 0):
                self.bbox_head._dbg_pc_input_stats = {
                    'source': 'bone_suppressed',
                    'mean': float(stacked.detach().mean().item()),
                    'std': float(
                        stacked.detach().std(unbiased=False).item()),
                    'min': float(stacked.detach().amin().item()),
                    'max': float(stacked.detach().amax().item()),
                    'raw_mae': 0.0,
                    'raw_corr': 0.0,
                    'paths': source_paths,
                }
            return stacked

        def _record_pc_input_comparison(self, img, pc_gray):
            """Record whether the paired image is distinct yet correlated."""
            debug_interval = int(getattr(
                self.bbox_head, 'debug_interval', 0))
            debug_step = int(getattr(
                self.bbox_head, '_debug_step', 0))
            if (debug_interval <= 0
                    or debug_step % debug_interval != 0):
                return
            with torch.no_grad():
                mean = img.new_tensor(
                    [123.675, 116.28, 103.53]).view(1, 3, 1, 1)
                std = img.new_tensor(
                    [58.395, 57.12, 57.375]).view(1, 3, 1, 1)
                raw = img * std + mean
                raw_gray = (
                    0.299 * raw[:, 0:1]
                    + 0.587 * raw[:, 1:2]
                    + 0.114 * raw[:, 2:3])
                if raw_gray.shape != pc_gray.shape:
                    raise ValueError(
                        f'PC input shape {tuple(pc_gray.shape)} does not '
                        f'match raw gray shape {tuple(raw_gray.shape)}')
                raw_flat = raw_gray.float().flatten(1)
                pc_flat = pc_gray.float().flatten(1)
                raw_centered = raw_flat - raw_flat.mean(
                    dim=1, keepdim=True)
                pc_centered = pc_flat - pc_flat.mean(
                    dim=1, keepdim=True)
                corr = F.cosine_similarity(
                    raw_centered, pc_centered, dim=1, eps=1e-8)
                stats = getattr(
                    self.bbox_head, '_dbg_pc_input_stats', {})
                stats['raw_mae'] = float(
                    (raw_gray.float() - pc_gray.float())
                    .abs().mean().item())
                stats['raw_corr'] = float(corr.mean().item())
                self.bbox_head._dbg_pc_input_stats = stats

        def _collect_lung_masks(self, img_metas, device,
                                batch_input_shape=None):
            """Align and batch anatomy masks required by lung-coordinate Sym."""
            positional_encoding = getattr(
                self.bbox_head, 'positional_encoding', None)
            needs_lung_mask = bool(
                getattr(positional_encoding, 'use_upper_prior', False)
                and getattr(
                    positional_encoding,
                    'upper_prior_mode',
                    'image') == 'lung')
            if not needs_lung_mask:
                return None

            aligned_masks = []
            for index, meta in enumerate(img_metas):
                if not isinstance(meta, dict) or meta.get(
                        'lung_mask', None) is None:
                    filename = (
                        meta.get('filename', '<unknown>')
                        if isinstance(meta, dict) else '<unknown>')
                    raise KeyError(
                        f'Missing lung_mask for batch item {index}: '
                        f'{filename}. Add LoadLungMask to the pipeline and '
                        f'collect lung_mask in img_metas.')
                aligned_masks.append(
                    _align_lung_mask_to_image_meta(
                        meta['lung_mask'], meta,
                        batch_input_shape=batch_input_shape))
            return torch.stack(aligned_masks).to(
                device=device, dtype=torch.float32)

        def _collect_roi_debug(self, img_metas):
            """Expose batch-level preprocessing health to periodic debug logs."""
            if not img_metas:
                return

            def mean_value(key, default=0.0):
                values = [
                    float(meta.get(key, default))
                    for meta in img_metas if isinstance(meta, dict)]
                return sum(values) / max(len(values), 1)

            valid_metas = [
                meta for meta in img_metas if isinstance(meta, dict)]
            batch_canvas = tuple(
                valid_metas[0].get('batch_input_shape', (0, 0))[:2]
                if valid_metas else (0, 0))
            recorded_pad_shapes = [
                tuple(int(v) for v in meta.get(
                    'pad_shape', meta['img_shape'])[:2])
                for meta in valid_metas]
            batch_area = (
                int(batch_canvas[0]) * int(batch_canvas[1])
                if len(batch_canvas) == 2 else 0)
            self.bbox_head._dbg_roi_stats = {
                'enabled': any(
                    'roi_applied' in meta for meta in valid_metas),
                'mode': self.input_preprocess,
                'applied_fraction': mean_value('roi_applied'),
                'fallback_fraction': mean_value('roi_safety_fallback'),
                'retained_area': mean_value('roi_retained_area', 1.0),
                'linear_scale_gain': mean_value(
                    'roi_linear_scale_gain', 1.0),
                'lung_coverage': mean_value('roi_lung_coverage', 1.0),
                'bbox_min_coverage': min(
                    [float(meta.get(
                        'roi_bbox_min_coverage', 1.0))
                     for meta in valid_metas] or [1.0]),
                'mean_before': mean_value('roi_intensity_mean_before'),
                'std_before': mean_value('roi_intensity_std_before'),
                'mean_after': mean_value('roi_intensity_mean_after'),
                'std_after': mean_value('roi_intensity_std_after'),
                'degenerate_fraction': mean_value(
                    'roi_intensity_degenerate'),
                'fallback_reasons': [
                    meta.get('roi_fallback_reason', '')
                    for meta in valid_metas
                    if meta.get('roi_fallback_reason', '')],
                'batch_canvas': batch_canvas,
                'recorded_pad_shapes': recorded_pad_shapes,
                'collate_extra_pad_fraction': (
                    sum(
                        1.0 - (
                            shape[0] * shape[1]
                            / max(batch_area, 1))
                        for shape in recorded_pad_shapes)
                    / max(len(recorded_pad_shapes), 1)),
            }

        def forward_train(self,
                          img,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          **kwargs):
            batch_input_shape = tuple(int(v) for v in img.shape[-2:])
            for meta in img_metas:
                if isinstance(meta, dict):
                    meta['batch_input_shape'] = batch_input_shape
            self._collect_roi_debug(img_metas)
            backbone_feats = self.backbone(img)

            # NAM graphs are not even aligned on the historical-core recovery
            # path. Zero auxiliary coefficients therefore remove both the
            # objective and its data/compute path instead of multiplying a
            # still-executed region loss by zero.
            nam_graphs = None
            if self.use_region_auxiliaries:
                nam_graphs = []
                for meta in img_metas:
                    if isinstance(meta, dict) and 'nam_graph' in meta:
                        nam_graphs.append(_align_nam_graph_to_image_meta(
                            meta['nam_graph'], meta,
                            batch_input_shape=batch_input_shape))
                    else:
                        nam_graphs.append(None)
                if all(g is None for g in nam_graphs):
                    nam_graphs = None

            # Compute the PC prior only when exactly one configured consumer
            # needs it. The main rollback path consumes it once at C2.
            need_pc_lbp = (
                self.use_pc_branch
                or self.bbox_head.use_head_pc_modulation)
            pc_gray = None
            if need_pc_lbp and self.pc_image_source == 'bone_suppressed':
                pc_gray = self._collect_bone_suppressed_images(
                    img_metas, device=img.device,
                    batch_input_shape=batch_input_shape)
                self._record_pc_input_comparison(img, pc_gray)
            lbp_lab = (
                self.bbox_head.compute_pc_lbp(img, pc_gray=pc_gray)
                if need_pc_lbp else None)

            x, c2_feat = self._build_pyramid(
                backbone_feats, lbp_lab=lbp_lab)
            region_feat = None
            if self.use_region_auxiliaries:
                region_feat = {
                    'c2': c2_feat,
                    'c5': backbone_feats[-1],
                    'p5': x[-1],
                }[self.region_feature_source]
            lung_masks = self._collect_lung_masks(
                img_metas, device=img.device,
                batch_input_shape=batch_input_shape)

            # --- Per-sample image shapes for bbox→feature coord scaling ---
            # DataContainer may pad beyond each sample's recorded pad_shape.
            # GT and the aligned label_map therefore share the real batch
            # tensor canvas for bbox→feature scaling.
            img_shapes = [batch_input_shape for _ in img_metas]

            current_epoch = int(self._nam_epoch) + 1
            warmup_factor = _contrast_schedule_factor(
                epoch=current_epoch,
                start_epoch=self.contrast_start_epoch,
                ramp_epochs=self.contrast_ramp_epochs,
                hold_end_epoch=self.contrast_hold_end_epoch,
                decay_end_epoch=self.contrast_decay_end_epoch)
            region_contrast_weight = (
                self.region_contrast_loss_weight * warmup_factor)
            boxalign_weight = (
                self.boxalign_loss_weight * warmup_factor)
            self.bbox_head._dbg_contrast_loss_weight = (
                region_contrast_weight)
            self.bbox_head._dbg_boxalign_loss_weight = boxalign_weight
            self.bbox_head._dbg_contrast_schedule_factor = warmup_factor
            self.bbox_head._dbg_contrast_epoch = current_epoch

            # --- Head forward ---
            head_lbp = (
                lbp_lab if self.bbox_head.use_head_pc_modulation else None)
            cls_scores, bbox_preds, contr_loss, boxalign_loss = self.bbox_head(
                feats=x, nam_graphs=nam_graphs, gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                lbp_lab=head_lbp, c2_feat=c2_feat,
                region_feat=region_feat, img_shapes=img_shapes,
                lung_masks=lung_masks,
                reg_detail_feats=getattr(
                    self.neck, 'regression_detail_feats', None))

            # --- Detection losses ---
            losses = self.bbox_head.loss(
                cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)
            bbox_contrast_loss = (
                self.bbox_head.bbox_instance_contrast_loss(
                    gt_bboxes, img_metas=img_metas)
                if self.bbox_head.use_bbox_contrastive
                else bbox_preds[0].sum() * 0.0)
            bbox_contrast_weight = (
                self.bbox_head.bbox_contrast_loss_weight
                * warmup_factor)
            self.bbox_head._dbg_bbox_contrast_loss_weight = (
                bbox_contrast_weight)
            bbox_contrast_weighted = (
                bbox_contrast_weight * bbox_contrast_loss)
            if self.bbox_head.use_bbox_contrastive:
                losses['loss_bbox_contrast'] = (
                    bbox_contrast_weighted)
            region_contrast_weighted = (
                region_contrast_weight * contr_loss)
            boxalign_weighted = boxalign_weight * boxalign_loss
            self.bbox_head._dbg_boxalign_loss_weight = boxalign_weight
            self.bbox_head._dbg_region_contrast_weighted = float(
                abs(region_contrast_weighted.detach().float().item()))
            self.bbox_head._dbg_boxalign_weighted = float(
                abs(boxalign_weighted.detach().float().item()))
            if self.region_contrast_loss_weight > 0.0:
                losses['loss_contrast'] = region_contrast_weighted
            if self.bbox_head.use_boxalign:
                losses['loss_boxalign'] = boxalign_weighted
            with torch.no_grad():
                bbox_loss_items = losses.get('loss_bbox', [])
                if not isinstance(bbox_loss_items, (list, tuple)):
                    bbox_loss_items = [bbox_loss_items]
                giou_total = sum(
                    abs(float(item.detach().float().item()))
                    if isinstance(item, torch.Tensor)
                    else abs(float(item))
                    for item in bbox_loss_items)
                weighted_bbox_contrast_value = abs(float(
                    bbox_contrast_weighted.detach().float().item()))
                self.bbox_head._dbg_bbox_contrast_weighted = (
                    weighted_bbox_contrast_value)
                self.bbox_head._dbg_bbox_contrast_to_giou_ratio = (
                    weighted_bbox_contrast_value
                    / max(giou_total, 1e-12))

            # Expensive loss-specific gradient comparison is sampled only at
            # the deep-debug cadence. It does not modify .grad; retain_graph
            # keeps the normal runner backward path intact.
            deep_debug_interval = int(getattr(
                self.bbox_head, 'deep_debug_interval', 0))
            next_debug_step = int(getattr(
                self.bbox_head, '_debug_step', 0))
            if (self.use_region_auxiliaries
                    and deep_debug_interval > 0
                    and next_debug_step % deep_debug_interval == 0):
                bbox_total_tensor = sum(
                    item for item in bbox_loss_items
                    if isinstance(item, torch.Tensor))

                def gradient_pair_stats(loss_a, loss_b, tensors):
                    tensors = tuple(
                        tensor for tensor in tensors
                        if isinstance(tensor, torch.Tensor)
                        and tensor.requires_grad)
                    if (not tensors
                            or not isinstance(loss_a, torch.Tensor)
                            or not isinstance(loss_b, torch.Tensor)
                            or not loss_a.requires_grad
                            or not loss_b.requires_grad):
                        return {
                            'aux_rms': 0.0,
                            'det_rms': 0.0,
                            'cosine': 0.0,
                        }
                    grad_a = torch.autograd.grad(
                        loss_a, tensors,
                        retain_graph=True,
                        allow_unused=True)
                    grad_b = torch.autograd.grad(
                        loss_b, tensors,
                        retain_graph=True,
                        allow_unused=True)
                    pairs = [
                        (a.detach().float(), b.detach().float())
                        for a, b in zip(grad_a, grad_b)
                        if a is not None and b is not None]
                    if not pairs:
                        return {
                            'aux_rms': 0.0,
                            'det_rms': 0.0,
                            'cosine': 0.0,
                        }
                    dot = sum((a * b).sum() for a, b in pairs)
                    norm_a_sq = sum((a * a).sum() for a, _ in pairs)
                    norm_b_sq = sum((b * b).sum() for _, b in pairs)
                    element_count = sum(a.numel() for a, _ in pairs)
                    return {
                        'aux_rms': float(torch.sqrt(
                            norm_a_sq / max(element_count, 1)).item()),
                        'det_rms': float(torch.sqrt(
                            norm_b_sq / max(element_count, 1)).item()),
                        'cosine': float(
                            dot
                            / torch.sqrt(
                                norm_a_sq * norm_b_sq).clamp_min(1e-12)),
                    }

                region_gradient_stats = gradient_pair_stats(
                    region_contrast_weighted,
                    bbox_total_tensor,
                    (region_feat,))
                align_reg_features = getattr(
                    self.bbox_head,
                    '_dbg_last_align_reg_features',
                    ())
                reg_gradient_stats = gradient_pair_stats(
                    boxalign_weighted,
                    bbox_total_tensor,
                    align_reg_features or ())
                self.bbox_head._dbg_loss_grad_stats = {
                    'source_step': max(0, next_debug_step - 1),
                    'region_source': self.region_feature_source,
                    'region': region_gradient_stats,
                    'reg': reg_gradient_stats,
                }
                self.bbox_head._dbg_last_align_reg_features = None

            # Snapshot the batch immediately before the next periodic print.
            # Per-level zero bbox loss is a practical warning that a pyramid
            # level received no useful positive regression supervision.
            debug_interval = int(getattr(
                self.bbox_head, 'debug_interval', 0))
            debug_step = int(getattr(
                self.bbox_head, '_debug_step', 0))
            if (debug_interval > 0
                    and debug_step % debug_interval == 0):
                def loss_values(value):
                    if isinstance(value, (list, tuple)):
                        return [
                            float(item.detach().float().item())
                            for item in value]
                    if isinstance(value, torch.Tensor):
                        return [float(value.detach().float().item())]
                    return [float(value)]

                gt_size_counts = [0, 0, 0]
                class_counts = {}
                with torch.no_grad():
                    for boxes, labels in zip(gt_bboxes, gt_labels):
                        if boxes.numel() > 0:
                            areas = (
                                (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
                                * (boxes[:, 3] - boxes[:, 1]).clamp_min(0))
                            gt_size_counts[0] += int(
                                (areas < 32 ** 2).sum().item())
                            gt_size_counts[1] += int(
                                ((areas >= 32 ** 2)
                                 & (areas < 96 ** 2)).sum().item())
                            gt_size_counts[2] += int(
                                (areas >= 96 ** 2).sum().item())
                        for label in labels.detach().cpu().tolist():
                            class_counts[int(label)] = (
                                class_counts.get(int(label), 0) + 1)
                    bbox_delta_abs = [
                        pred.detach().float().abs().mean().item()
                        for pred in bbox_preds]
                    bbox_delta_max = [
                        pred.detach().float().abs().max().item()
                        for pred in bbox_preds]
                    bbox_finite = [
                        torch.isfinite(pred.detach()).float().mean().item()
                        for pred in bbox_preds]
                self.bbox_head._dbg_det_loss_stats = {
                    'source_step': max(0, debug_step - 1),
                    'loss_cls': loss_values(
                        losses.get('loss_cls', 0.0)),
                    'loss_bbox': loss_values(
                        losses.get('loss_bbox', 0.0)),
                    'loss_bbox_contrast': loss_values(
                        losses.get('loss_bbox_contrast', 0.0)),
                    'loss_region_contrast': loss_values(
                        losses.get('loss_contrast', 0.0)),
                    'loss_boxalign': loss_values(
                        losses.get('loss_boxalign', 0.0)),
                    'loss_dfr': loss_values(
                        losses.get('loss_dfr', 0.0)),
                    'bbox_contrast_to_giou_ratio': float(
                        self.bbox_head
                        ._dbg_bbox_contrast_to_giou_ratio),
                    'gt_size_counts': gt_size_counts,
                    'gt_class_counts': class_counts,
                    'bbox_delta_abs': bbox_delta_abs,
                    'bbox_delta_max': bbox_delta_max,
                    'bbox_finite': bbox_finite,
                }

            side_refine_train = getattr(self, '_v45_side_refine_train', None)
            if callable(side_refine_train):
                losses.update(side_refine_train(
                    feats=x, cls_scores=cls_scores, bbox_preds=bbox_preds,
                    img_metas=img_metas, gt_bboxes=gt_bboxes,
                    detail_img=kwargs.get('detail_img', None)))

            local_refine_train = getattr(
                self, '_v330_local_refine_train', None)
            if callable(local_refine_train):
                losses.update(local_refine_train(
                    original_img=kwargs.get('original_img', None),
                    img_metas=img_metas,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    cls_scores=cls_scores,
                    bbox_preds=bbox_preds))
            return losses

        def simple_test(self, img, img_metas, rescale=False, **kwargs):
            batch_input_shape = tuple(int(v) for v in img.shape[-2:])
            for meta in img_metas:
                if isinstance(meta, dict):
                    meta['batch_input_shape'] = batch_input_shape
            self._collect_roi_debug(img_metas)
            backbone_feats = self.backbone(img)
            nam_graphs = None
            if self.use_region_auxiliaries:
                nam_graphs = []
                for meta in img_metas:
                    if isinstance(meta, dict) and 'nam_graph' in meta:
                        nam_graphs.append(_align_nam_graph_to_image_meta(
                            meta['nam_graph'], meta,
                            batch_input_shape=batch_input_shape))
                    else:
                        nam_graphs.append(None)
                if all(g is None for g in nam_graphs):
                    nam_graphs = None

            need_pc_lbp = (
                self.use_pc_branch
                or self.bbox_head.use_head_pc_modulation)
            pc_gray = None
            if need_pc_lbp and self.pc_image_source == 'bone_suppressed':
                pc_gray = self._collect_bone_suppressed_images(
                    img_metas, device=img.device,
                    batch_input_shape=batch_input_shape)
                self._record_pc_input_comparison(img, pc_gray)
            lbp_lab = (
                self.bbox_head.compute_pc_lbp(img, pc_gray=pc_gray)
                if need_pc_lbp else None)
            x, c2_feat = self._build_pyramid(
                backbone_feats, lbp_lab=lbp_lab)
            region_feat = None
            if self.use_region_auxiliaries:
                region_feat = {
                    'c2': c2_feat,
                    'c5': backbone_feats[-1],
                    'p5': x[-1],
                }[self.region_feature_source]
            lung_masks = self._collect_lung_masks(
                img_metas, device=img.device,
                batch_input_shape=batch_input_shape)
            head_lbp = (
                lbp_lab if self.bbox_head.use_head_pc_modulation else None)

            from mmdet.core import bbox2result
            results_list = self.bbox_head.simple_test(
                x, img_metas, rescale=rescale, nam_graphs=nam_graphs,
                lbp_lab=head_lbp, c2_feat=c2_feat,
                region_feat=region_feat,
                lung_masks=lung_masks,
                reg_detail_feats=getattr(
                    self.neck, 'regression_detail_feats', None))
            local_refine_test = getattr(
                self, '_v330_local_refine_test', None)
            if callable(local_refine_test):
                results_list = local_refine_test(
                    results_list=results_list,
                    original_img=kwargs.get('original_img', None),
                    img_metas=img_metas,
                    results_are_rescaled=bool(rescale))
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
    class NAMDebugHook(Hook):
        """Log NAM branch stats every N epochs.

        Add to config:
            custom_hooks = [
                dict(type='NAMDebugHook', interval=1),
            ]
        """

        def __init__(self, interval: int = 2):
            self.interval = interval

        def before_run(self, runner):
            """Log an unmistakable build/config fingerprint on the server."""
            try:
                model = (runner.model.module
                         if hasattr(runner.model, 'module')
                         else runner.model)
                cross_scale = getattr(
                    getattr(model, 'neck', None),
                    'cross_scale', None)
                positional_encoding = getattr(
                    getattr(model, 'bbox_head', None),
                    'positional_encoding', None)
                exp_name = ''
                run_seed = None
                if isinstance(getattr(runner, 'meta', None), dict):
                    exp_name = runner.meta.get('exp_name', '')
                    run_seed = runner.meta.get('seed', None)
                active_build_id = getattr(
                    model, 'integrated_build_id', NAMLAB_BUILD_ID)
                runner.logger.info(
                    f'[NAM-BUILD] id={active_build_id} '
                    f'file={__file__} exp={exp_name} seed={run_seed} '
                    f'tag={getattr(model, "experiment_tag", None)} '
                    f'neck={type(getattr(model, "neck", None)).__name__} '
                    f'top_down={getattr(cross_scale, "use_top_down", None)} '
                    f'cs_weighted={getattr(cross_scale, "learnable_fusion_weights", None)} '
                    f'freqfusion={getattr(cross_scale, "freqfusion_top_down", None)} '
                    f'reg_detail_neck={getattr(getattr(model, "neck", None), "use_reg_detail_bypass", None)} '
                    f'reg_detail_head={getattr(model.bbox_head, "use_reg_detail_bypass", None)} '
                    f'reg_detail_levels={getattr(model.bbox_head, "reg_detail_levels", None)} '
                    f'pc={getattr(model, "use_pc_branch", None)} '
                    f'pc_source={getattr(model, "pc_image_source", None)} '
                    f'pc_rmax={getattr(getattr(getattr(model, "pc_shallow_adapter", None), "br_gate", None), "mod_scale_max", None)} '
                    f'region_aux={getattr(model, "use_region_auxiliaries", None)} '
                    f'region_src={getattr(model, "region_feature_source", None)} '
                    f'head_pc={getattr(model.bbox_head, "use_head_pc_modulation", None)} '
                    f'head_pc_rmax={getattr(model.bbox_head, "head_pc_residual_max", None)} '
                    f'bbox_con={getattr(model.bbox_head, "use_bbox_contrastive", None)} '
                    f'bbox_con_w={getattr(model.bbox_head, "bbox_contrast_loss_weight", None)} '
                    f'bbox_iou={getattr(model.bbox_head, "bbox_contrast_pos_iou", None)}/'
                    f'{getattr(model.bbox_head, "bbox_contrast_neg_iou", None)} '
                    f'dfine={getattr(model.bbox_head, "use_d_fine_style", None)} '
                    f'dfine_k={getattr(model.bbox_head, "d_fine_reg_max", None)} '
                    f'dfine_w={getattr(model.bbox_head, "d_fine_loss_weight", None)} '
                    f'lung_neg={getattr(model.bbox_head, "use_lung_negative_weighting", None)} '
                    f'lung_far_w={getattr(model.bbox_head, "lung_negative_far_weight", None)} '
                    f'lung_bias={getattr(model.bbox_head, "use_lung_anatomy_bias", None)} '
                    f'lung_levels={getattr(model.bbox_head, "lung_anatomy_levels", None)} '
                    f'loc_q={getattr(model.bbox_head, "use_localization_quality", None)} '
                    f'loc_q_w={getattr(model.bbox_head, "localization_quality_loss_weight", None)} '
                    f'boxalign={getattr(model.bbox_head, "use_boxalign", None)} '
                    f'boxalign_levels={getattr(model.bbox_head, "boxalign_levels", None)} '
                    f'boxalign_ratio={getattr(model.bbox_head, "boxalign_loss_ratio", None)} '
                    f'boxalign_w={getattr(model, "boxalign_loss_weight", None)} '
                    f'upper={getattr(positional_encoding, "use_upper_prior", None)} '
                    f'upper_mode={getattr(positional_encoding, "upper_prior_mode", None)} '
                    f'preproc={getattr(model, "input_preprocess", None)} '
                    f'tok_drop={getattr(model.bbox_head, "v3_token_dropout", None)} '
                    f'region_c={getattr(model.bbox_head, "region_in_channels", None)} '
                    f'pos_thr={getattr(model.bbox_head, "v3_minimum_positive_purity", None)} '
                    f'gt_fallback={getattr(model.bbox_head, "v3_ensure_gt_positive", None)} '
                    f'pur_floor={getattr(model.bbox_head, "v3_purity_weight_floor", None)} '
                    f'hard_neg={getattr(model.bbox_head, "v3_hard_negative_ratio", None)} '
                    f'far_neg_w={getattr(model.bbox_head, "v3_far_negative_weight", None)} '
                    f'rank={getattr(model.bbox_head, "v3_boundary_rank_margin", None)}/'
                    f'{getattr(model.bbox_head, "v3_boundary_rank_weight", None)} '
                    f'region_w={getattr(model, "region_contrast_loss_weight", None)} '
                    f'contr_sched={getattr(model, "contrast_start_epoch", None)}/'
                    f'{getattr(model, "contrast_ramp_epochs", None)}/'
                    f'{getattr(model, "contrast_hold_end_epoch", None)}/'
                    f'{getattr(model, "contrast_decay_end_epoch", None)} '
                    f'debug_every={getattr(model.bbox_head, "debug_interval", None)} '
                    f'deep_debug_every={getattr(model.bbox_head, "deep_debug_interval", None)}')
            except Exception as error:
                runner.logger.warning(
                    f'[NAM-BUILD] fingerprint failed: {error}')

        @staticmethod
        def _module_gradient_stats(module):
            if module is None:
                return 0.0, 0.0
            return _TBFusionCrossScale._gradient_group(
                module.parameters())

        def after_train_iter(self, runner):
            """Capture previous-iteration gradients for the next debug print."""
            try:
                model = (runner.model.module
                         if hasattr(runner.model, 'module')
                         else runner.model)
                head = model.bbox_head
                debug_interval = int(getattr(
                    head, 'debug_interval', 0))
                debug_step = int(getattr(head, '_debug_step', 0))
            except (AttributeError, TypeError, ValueError):
                return
            if (debug_interval <= 0
                    or debug_step % debug_interval != 0):
                return

            cross_scale = getattr(
                getattr(model, 'neck', None), 'cross_scale', None)
            if cross_scale is not None:
                head._dbg_cs_grad_stats = cross_scale.gradient_stats()

            module_groups = {
                'pc_adapter': getattr(
                    model, 'pc_shallow_adapter', None),
                'lbp_branch': getattr(head, 'lbp_branch', None),
                'br_gate': getattr(head, 'br_gate', None),
                'region_transformer': getattr(
                    head, 'nam_branch', None),
                'contrast_head': getattr(
                    getattr(head, 'nam_branch', None),
                    'contrast_head', None),
                'boxalign': getattr(
                    head, 'boxalign_proj', None),
                'boxalign_anchor_embed': getattr(
                    head, 'boxalign_anchor_embed', None),
                'bbox_contrast': getattr(
                    head, 'bbox_contrast_proj', None),
                'bbox_anchor_embed': getattr(
                    head, 'bbox_anchor_embed', None),
                'symformer': getattr(head, 'transformer', None),
                'd_fine_reg': getattr(head, 'd_fine_reg', None),
                'reg_detail_generator': getattr(
                    getattr(model, 'neck', None),
                    'reg_detail_generator', None),
                'reg_detail_injector': getattr(
                    head, 'reg_detail_injector', None),
            }
            grad_rms, grad_coverage = {}, {}
            for name, module in module_groups.items():
                rms, coverage = self._module_gradient_stats(module)
                grad_rms[name] = rms
                grad_coverage[name] = coverage
            embedding = getattr(head, 'lbp_embed', None)
            if embedding is not None:
                rms, coverage = _TBFusionCrossScale._gradient_group(
                    embedding.parameters())
                grad_rms['lbp_embedding'] = rms
                grad_coverage['lbp_embedding'] = coverage
            d_fine_scale = getattr(
                head, 'd_fine_refine_raw', None)
            if d_fine_scale is not None:
                rms, coverage = _TBFusionCrossScale._gradient_group(
                    [d_fine_scale])
                grad_rms['d_fine_scale'] = rms
                grad_coverage['d_fine_scale'] = coverage
            head._dbg_module_grad_stats = {
                'group_grad_rms': grad_rms,
                'group_grad_coverage': grad_coverage,
                'source_step': max(0, debug_step - 1),
            }

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
                f'[NAMv3] Epoch {runner.epoch + 1:3d} | '
                f'temp={temp_eff:.3f}(raw={temp_raw:.2f})')


# ============================================================================
# v3.20 single-file deployment profiles
# ============================================================================
# Everything below intentionally lives in this file.  Deploy this file as
# ``mmdet/models/utils/NAMLab_att_v3.py``; no sibling best_slim/lung utility
# module is required on the server.
import torch.distributed as dist
from mmdet.core import bbox_overlaps, multiclass_nms


NAMLAB_SLIM_BUILD_ID = 'v3.19-best-g2-strict-slim-integrated-20260804'
NAMLAB_LUNG_BUILD_ID = 'v3.20-g2-lung-soft-quality-integrated-20260804'


def _v320_reject_enabled(kwargs, names, owner):
    enabled = [name for name in names if bool(kwargs.get(name, False))]
    if enabled:
        raise ValueError(
            f'{owner} rejects failed auxiliary modules: {enabled}.')


def _v320_require_zero(kwargs, names, owner):
    nonzero = [
        name for name in names
        if kwargs.get(name) is not None
        and not math.isclose(float(kwargs[name]), 0.0, abs_tol=1e-12)
    ]
    if nonzero:
        raise ValueError(
            f'{owner} requires disabled auxiliary losses: {nonzero}.')


@NECKS.register_module(force=True)
class TBFusionCrossScaleNeckV319(TBFusionCrossScaleNeck):
    """Measured original CrossScale path with harmful switches locked off."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('use_top_down', True)
        kwargs.setdefault('learnable_fusion_weights', False)
        kwargs.setdefault('freqfusion_top_down', False)
        kwargs.setdefault('reg_detail_bypass', False)
        if not bool(kwargs['use_top_down']):
            raise ValueError('V319 requires the measured bidirectional path')
        if bool(kwargs['learnable_fusion_weights']):
            raise ValueError(
                'V319 forbids learnable_fusion_weights after negative G3')
        if (bool(kwargs['freqfusion_top_down'])
                and bool(kwargs['reg_detail_bypass'])):
            raise ValueError(
                'FreqFusion and regression detail must be tested separately')
        super().__init__(*args, **kwargs)
        self.slim_build_id = NAMLAB_SLIM_BUILD_ID


@HEADS.register_module(force=True)
class RetinaNAMGuideAttHeadV319(RetinaNAMGuideAttHeadV3):
    """Sym head locked to the reproducible G2 FDR settings."""

    _REMOVED_SWITCHES = (
        'head_pc_modulation', 'bbox_contrastive', 'boxalign',
        'edge_supervision')

    def __init__(self, *args, **kwargs):
        _v320_reject_enabled(
            kwargs, self._REMOVED_SWITCHES, self.__class__.__name__)
        kwargs.setdefault('head_pc_modulation', False)
        kwargs.setdefault('bbox_contrastive', False)
        kwargs.setdefault('boxalign', False)
        kwargs.setdefault('edge_supervision', False)
        kwargs.setdefault('token_dropout', 0.0)
        kwargs.setdefault('d_fine_style', True)
        kwargs.setdefault('d_fine_reg_max', 16)
        kwargs.setdefault('d_fine_refine_init', 0.25)
        kwargs.setdefault('d_fine_refine_max', 1.0)
        kwargs.setdefault('d_fine_loss_weight', 0.05)
        kwargs.setdefault('reg_detail_bypass', False)
        if not bool(kwargs['d_fine_style']):
            raise ValueError('V319 G2 requires d_fine_style=True')
        if int(kwargs['d_fine_reg_max']) != 16:
            raise ValueError('V319 G2 requires d_fine_reg_max=16')
        if not math.isclose(
                float(kwargs['d_fine_loss_weight']), 0.05,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('V319 G2 requires d_fine_loss_weight=0.05')
        positional = kwargs.get('positional_encoding')
        if isinstance(positional, dict) and bool(
                positional.get('use_upper_prior', False)):
            raise ValueError(
                'V319 retains left/right Sym and rejects upper prior')
        super().__init__(*args, **kwargs)
        self.slim_build_id = NAMLAB_SLIM_BUILD_ID


@DETECTORS.register_module(force=True)
class RetinaNAMAttV319(RetinaNAMAttV3):
    """Detector profile preserving only CrossScale, Sym and G2 FDR."""

    def __init__(self, *args, **kwargs):
        _v320_reject_enabled(
            kwargs, ('pc_branch',), self.__class__.__name__)
        _v320_require_zero(
            kwargs,
            ('contrast_loss_weight', 'region_contrast_loss_weight',
             'boxalign_loss_weight'),
            self.__class__.__name__)
        kwargs.setdefault('pc_branch', False)
        kwargs.setdefault('contrast_loss_weight', 0.0)
        kwargs.setdefault('region_contrast_loss_weight', 0.0)
        kwargs.setdefault('boxalign_loss_weight', 0.0)
        kwargs.setdefault('input_preprocess', 'raw_imagenet')
        kwargs.setdefault('experiment_tag', 'V319_G2_STRICT_INTEGRATED')
        if kwargs['input_preprocess'] != 'raw_imagenet':
            raise ValueError('V319 requires raw CXR + ImageNet normalization')
        super().__init__(*args, **kwargs)
        if not isinstance(self.neck, TBFusionCrossScaleNeckV319):
            raise TypeError('RetinaNAMAttV319 requires V319 CrossScale neck')
        if not isinstance(self.bbox_head, RetinaNAMGuideAttHeadV319):
            raise TypeError('RetinaNAMAttV319 requires V319 head')
        cross_scale = self.neck.cross_scale
        logging.getLogger('mmdet').info(
            '[NAM-SLIM] id=%s profile=G2 cs_weighted=%s freqfusion=%s '
            'reg_detail=%s dfine=%s dfine_w=%.3f',
            NAMLAB_SLIM_BUILD_ID,
            cross_scale.learnable_fusion_weights,
            cross_scale.freqfusion_top_down,
            self.neck.use_reg_detail_bypass,
            self.bbox_head.use_d_fine_style,
            self.bbox_head.d_fine_loss_weight)


def _v320_soft_lung_support(mask, dilation=16, blur=9):
    """Convert a single binary mask to a dilated, soft support map."""
    mask = (mask > 0.5).to(dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f'Expected BxHxW or Bx1xHxW mask, got {mask.shape}')
    dilation, blur = int(dilation), int(blur)
    if dilation < 0 or blur < 0:
        raise ValueError('lung-mask dilation and blur must be non-negative')
    if dilation:
        mask = F.max_pool2d(
            mask, 2 * dilation + 1, stride=1, padding=dilation)
    if blur:
        mask = F.avg_pool2d(
            mask, 2 * blur + 1, stride=1, padding=blur)
    return mask.clamp_(0.0, 1.0)


def _v320_lung_coordinate_channels(
        lung_masks, output_size, dilation=16, blur=9):
    """Build support and continuous within-mask x/y coordinates."""
    support = _v320_soft_lung_support(
        lung_masks, dilation=dilation, blur=blur)
    batch, _, height, width = support.shape
    hard = support[:, 0] > 0.05
    x_grid = support.new_zeros((batch, height, width))
    y_grid = support.new_zeros((batch, height, width))
    for batch_index in range(batch):
        ys, xs = torch.where(hard[batch_index])
        if ys.numel() == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        xs_all = torch.arange(
            width, device=support.device, dtype=support.dtype)
        ys_all = torch.arange(
            height, device=support.device, dtype=support.dtype)
        x_grid[batch_index] = (
            2.0 * (xs_all - x0) / (x1 - x0).clamp_min(1) - 1.0
        )[None].expand(height, width)
        y_grid[batch_index] = (
            2.0 * (ys_all - y0) / (y1 - y0).clamp_min(1) - 1.0
        )[:, None].expand(height, width)
    anatomy = torch.cat([
        support,
        support * x_grid[:, None],
        support * y_grid[:, None],
    ], dim=1)
    return F.interpolate(
        anatomy, size=output_size, mode='bilinear', align_corners=False)


def _v320_rank_zero():
    return (not dist.is_available() or not dist.is_initialized()
            or dist.get_rank() == 0)


@HEADS.register_module(force=True)
class RetinaNAMGuideAttHeadV320(RetinaNAMGuideAttHeadV319):
    """G2 head with isolated binary-lung and FDR-quality switches."""

    def __init__(
            self, *args,
            lung_negative_weighting=False,
            lung_negative_far_weight=0.25,
            lung_mask_dilation=16,
            lung_mask_blur=9,
            lung_anatomy_bias=False,
            lung_anatomy_levels=(2, 3),
            lung_anatomy_residual_init=0.01,
            lung_anatomy_residual_max=0.05,
            localization_quality=False,
            localization_quality_alpha=0.5,
            localization_quality_loss_weight=0.25,
            localization_quality_prior=0.5,
            **kwargs):
        self.use_lung_negative_weighting = bool(lung_negative_weighting)
        self.lung_negative_far_weight = float(lung_negative_far_weight)
        self.lung_mask_dilation = int(lung_mask_dilation)
        self.lung_mask_blur = int(lung_mask_blur)
        self.use_lung_anatomy_bias = bool(lung_anatomy_bias)
        self.lung_anatomy_levels = tuple(int(v) for v in lung_anatomy_levels)
        self.lung_anatomy_residual_init = float(
            lung_anatomy_residual_init)
        self.lung_anatomy_residual_max = float(
            lung_anatomy_residual_max)
        self.use_localization_quality = bool(localization_quality)
        self.localization_quality_alpha = float(localization_quality_alpha)
        self.localization_quality_loss_weight = float(
            localization_quality_loss_weight)
        self.localization_quality_prior = float(localization_quality_prior)
        if not 0.0 <= self.lung_negative_far_weight <= 1.0:
            raise ValueError('lung_negative_far_weight must be in [0, 1]')
        if self.lung_mask_dilation < 0 or self.lung_mask_blur < 0:
            raise ValueError('lung-mask dilation/blur must be non-negative')
        if not 0.0 < self.lung_anatomy_residual_init \
                < self.lung_anatomy_residual_max:
            raise ValueError(
                'lung_anatomy_residual_init must satisfy 0 < init < max')
        if any(level not in (0, 1, 2, 3)
               for level in self.lung_anatomy_levels):
            raise ValueError('lung_anatomy_levels must be within P2..P5')
        if self.localization_quality_alpha <= 0:
            raise ValueError('localization_quality_alpha must be positive')
        if self.localization_quality_loss_weight < 0:
            raise ValueError(
                'localization_quality_loss_weight must be non-negative')
        if not 0.0 < self.localization_quality_prior < 1.0:
            raise ValueError('localization_quality_prior must be in (0, 1)')
        super().__init__(*args, **kwargs)
        if self.use_localization_quality and not self.use_d_fine_style:
            raise ValueError('localization_quality requires G2 FDR')
        self._last_quality_logits = None
        self._quality_logits_current = None
        self._lung_debug_seen = set()
        self._lung_debug = {}
        self._anatomy_debug = {}
        self._quality_debug = {}

    def _init_layers(self):
        super()._init_layers()
        if self.use_lung_anatomy_bias:
            self.lung_anatomy_proj = nn.ModuleList([
                nn.Conv2d(3, self.feat_channels, 3, padding=1)
                for _ in range(self.anchor_generator.num_levels)
            ])
            raw = math.atanh(
                self.lung_anatomy_residual_init
                / self.lung_anatomy_residual_max)
            self.lung_anatomy_scale_raw = nn.Parameter(torch.full(
                (self.anchor_generator.num_levels,), raw,
                dtype=torch.float32))
        else:
            self.lung_anatomy_proj = None
            self.register_parameter('lung_anatomy_scale_raw', None)
        if self.use_localization_quality:
            # Four maximum probabilities + four normalized entropies per
            # anchor, calculated directly from the existing FDR distribution.
            self.localization_quality_pred = nn.Conv2d(
                self.num_anchors * 8, self.num_anchors,
                kernel_size=1, groups=self.num_anchors)
        else:
            self.localization_quality_pred = None
        self._v320_reset_new_layers()

    def _v320_reset_new_layers(self):
        if self.lung_anatomy_proj is not None:
            for projection in self.lung_anatomy_proj:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
        if self.localization_quality_pred is not None:
            nn.init.zeros_(self.localization_quality_pred.weight)
            prior_logit = math.log(
                self.localization_quality_prior
                / (1.0 - self.localization_quality_prior))
            nn.init.constant_(
                self.localization_quality_pred.bias, prior_logit)

    def init_weights(self):
        # The parent init_cfg touches every Conv2d, so restore the exact
        # zero-start contract after it and after the parent's FDR reset.
        super().init_weights()
        self._v320_reset_new_layers()

    def _v320_aligned_lung_mask(self, img_meta, device):
        mask = img_meta.get('lung_mask') if isinstance(img_meta, dict) else None
        if mask is None:
            filename = (
                img_meta.get('filename', '<unknown>')
                if isinstance(img_meta, dict) else '<unknown>')
            raise KeyError(
                f'Missing lung_mask for {filename}; add LoadLungMask before '
                'Resize and collect lung_mask in img_metas')
        return _align_lung_mask_to_image_meta(
            mask, img_meta,
            batch_input_shape=img_meta.get('batch_input_shape')).to(
                device=device, dtype=torch.float32)

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes,
                            gt_bboxes_ignore, gt_labels, img_meta,
                            label_channels=1, unmap_outputs=True):
        result = super()._get_targets_single(
            flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels,
            img_meta, label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        if not self.use_lung_negative_weighting or result[0] is None:
            return result
        if not unmap_outputs:
            raise ValueError(
                'V320 lung weighting requires standard unmap_outputs=True')
        labels, label_weights, bbox_targets, bbox_weights = result[:4]
        mask = self._v320_aligned_lung_mask(
            img_meta, device=flat_anchors.device)
        support = _v320_soft_lung_support(
            mask, self.lung_mask_dilation,
            self.lung_mask_blur)[0, 0]
        center_x = ((flat_anchors[:, 0] + flat_anchors[:, 2]) * 0.5).long()
        center_y = ((flat_anchors[:, 1] + flat_anchors[:, 3]) * 0.5).long()
        center_x.clamp_(0, support.shape[1] - 1)
        center_y.clamp_(0, support.shape[0] - 1)
        anchor_support = support[center_y, center_x]
        negative = (labels == self.num_classes) & (label_weights > 0)
        soft_weight = (
            self.lung_negative_far_weight
            + (1.0 - self.lung_negative_far_weight) * anchor_support)
        label_weights = label_weights.clone()
        label_weights[negative] *= soft_weight[negative]

        # FDR obtains targets again. Count each image only once per loss call.
        debug_key = id(img_meta)
        if debug_key not in self._lung_debug_seen:
            self._lung_debug_seen.add(debug_key)
            positive = (labels >= 0) & (labels < self.num_classes)
            neg_support = anchor_support[negative]
            pos_support = anchor_support[positive]
            coverage = []
            for box in gt_bboxes:
                x1 = int(torch.floor(box[0]).clamp(0, support.shape[1]).item())
                y1 = int(torch.floor(box[1]).clamp(0, support.shape[0]).item())
                x2 = int(torch.ceil(box[2]).clamp(0, support.shape[1]).item())
                y2 = int(torch.ceil(box[3]).clamp(0, support.shape[0]).item())
                if x2 > x1 and y2 > y1:
                    coverage.append(support[y1:y2, x1:x2].mean())
            zero = support.new_zeros(())
            self._lung_debug.setdefault('negative_support', []).append(
                neg_support.mean().detach() if neg_support.numel() else zero)
            self._lung_debug.setdefault('negative_weight', []).append(
                soft_weight[negative].mean().detach()
                if negative.any() else zero)
            self._lung_debug.setdefault('positive_support', []).append(
                pos_support.mean().detach() if pos_support.numel() else zero)
            self._lung_debug.setdefault('gt_coverage', []).append(
                torch.stack(coverage).mean().detach() if coverage else zero)
        return ((labels, label_weights, bbox_targets, bbox_weights)
                + result[4:])

    def _v320_apply_anatomy_bias(self, feats, lung_masks):
        if not self.use_lung_anatomy_bias:
            return list(feats)
        if lung_masks is None:
            raise RuntimeError('lung_anatomy_bias requires aligned lung_masks')
        output = list(feats)
        changes = []
        scales = self.lung_anatomy_residual_max * torch.tanh(
            self.lung_anatomy_scale_raw)
        for level, feat in enumerate(output):
            if level not in self.lung_anatomy_levels:
                changes.append(feat.new_zeros(()))
                continue
            anatomy = _v320_lung_coordinate_channels(
                lung_masks, feat.shape[-2:],
                self.lung_mask_dilation,
                self.lung_mask_blur).to(dtype=feat.dtype)
            delta = scales[level].to(dtype=feat.dtype) \
                * self.lung_anatomy_proj[level](anatomy)
            output[level] = feat + delta
            changes.append(
                delta.detach().float().norm()
                / feat.detach().float().norm().clamp_min(1e-8))
        self._anatomy_debug = {
            'scales': [float(v.detach().float().item()) for v in scales],
            'delta_rel': [float(v.detach().float().item()) for v in changes],
        }
        return output

    def _v320_distribution_quality(self, distribution_logits):
        batch, _, height, width = distribution_logits.shape
        bins = self.d_fine_reg_max + 1
        logits = distribution_logits.reshape(
            batch, self.num_anchors, 4, bins, height, width)
        probabilities = logits.float().softmax(dim=3)
        maximum = probabilities.amax(dim=3)
        entropy = -(probabilities
                    * probabilities.clamp_min(1e-8).log()).sum(
                        dim=3) / math.log(bins)
        statistics = torch.cat([maximum, entropy], dim=2).reshape(
            batch, self.num_anchors * 8, height, width)
        # Isolate score-quality learning from the measured FDR regression.
        return self.localization_quality_pred(statistics.detach())

    def forward_single(self, *args, **kwargs):
        output = super().forward_single(*args, **kwargs)
        if not self.use_localization_quality:
            return output
        (cls_score, bbox_pred, feat1, fused, reg_feat, gate_stats,
         bbox_base, distribution_logits) = output
        quality_logits = self._v320_distribution_quality(
            distribution_logits)
        if self._quality_logits_current is not None:
            self._quality_logits_current.append(quality_logits)
        if not self.training:
            cls_probability = cls_score.float().sigmoid()
            quality = quality_logits.float().sigmoid().repeat_interleave(
                self.cls_out_channels, dim=1)
            quality_factor = (
                quality / self.localization_quality_prior
            ).pow(self.localization_quality_alpha)
            calibrated = (
                cls_probability * quality_factor
            ).clamp(1e-6, 1.0 - 1e-6)
            cls_score = torch.logit(calibrated).to(dtype=cls_score.dtype)
        return (cls_score, bbox_pred, feat1, fused, reg_feat, gate_stats,
                bbox_base, distribution_logits)

    def forward(self, feats, *args, lung_masks=None, **kwargs):
        feats = self._v320_apply_anatomy_bias(feats, lung_masks)
        self._quality_logits_current = (
            [] if self.use_localization_quality else None)
        output = super().forward(
            feats, *args, lung_masks=lung_masks, **kwargs)
        self._last_quality_logits = (
            tuple(self._quality_logits_current)
            if self.training and self.use_localization_quality else None)
        self._quality_logits_current = None
        return output

    def _v320_quality_loss(self, bbox_preds, gt_bboxes, gt_labels,
                           img_metas, gt_bboxes_ignore=None):
        quality_logits = self._last_quality_logits
        self._last_quality_logits = None
        if quality_logits is None:
            raise RuntimeError(
                'quality loss requires a preceding training forward')
        featmap_sizes = [prediction.shape[-2:] for prediction in bbox_preds]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=bbox_preds[0].device)
        targets = self.get_targets(
            anchor_list, valid_flag_list, gt_bboxes, img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=(
                self.cls_out_channels if self.use_sigmoid_cls else 1))
        if targets is None:
            return [value.sum() * 0.0 for value in quality_logits]
        _, _, bbox_targets, bbox_weights, num_total_pos, _ = targets
        losses, predictions, quality_targets = [], [], []
        for level, (quality, bbox_pred) in enumerate(zip(
                quality_logits, bbox_preds)):
            q = quality.permute(0, 2, 3, 1).reshape(-1)
            pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).float()
            target = bbox_targets[level].reshape(-1, 4).float()
            weight = bbox_weights[level].reshape(-1, 4)
            anchors = torch.stack([
                image_anchors[level] for image_anchors in anchor_list
            ], dim=0).reshape(-1, 4).float()
            positive = weight.sum(dim=1) > 0
            if not positive.any():
                losses.append(q.sum() * 0.0)
                continue
            decoded_pred = self.bbox_coder.decode(
                anchors[positive], pred[positive])
            decoded_target = (
                target[positive] if self.reg_decoded_bbox
                else self.bbox_coder.decode(
                    anchors[positive], target[positive]))
            iou_target = bbox_overlaps(
                decoded_pred.detach(), decoded_target,
                is_aligned=True).clamp(0.0, 1.0)
            raw_loss = F.binary_cross_entropy_with_logits(
                q[positive].float(), iou_target, reduction='sum')
            losses.append(
                self.localization_quality_loss_weight
                * raw_loss / max(int(num_total_pos), 1))
            predictions.append(q[positive].detach().float().sigmoid())
            quality_targets.append(iou_target.detach().float())
        if predictions:
            predicted = torch.cat(predictions)
            target = torch.cat(quality_targets)
            pred_centered = predicted - predicted.mean()
            target_centered = target - target.mean()
            correlation = (
                (pred_centered * target_centered).sum()
                / torch.sqrt(
                    pred_centered.square().sum()
                    * target_centered.square().sum()).clamp_min(1e-8))
            self._quality_debug = {
                'pred_mean': float(predicted.mean().item()),
                'target_mean': float(target.mean().item()),
                'mae': float((predicted - target).abs().mean().item()),
                'corr': float(correlation.item()),
                'positive_count': int(target.numel()),
            }
        return losses

    def _v320_emit_debug(self):
        interval = int(getattr(self, 'debug_interval', 0))
        source_step = int(getattr(self, '_debug_step', 0)) - 1
        if (interval <= 0 or source_step % interval != 0
                or not _v320_rank_zero()):
            return
        logger = logging.getLogger('mmdet')
        if self._lung_debug:
            means = {
                key: float(torch.stack(values).mean().item())
                for key, values in self._lung_debug.items() if values
            }
            logger.info(
                '[LUNG-NEG] iter=%d neg_support=%.3f neg_weight=%.3f '
                'pos_support=%.3f gt_coverage=%.3f dilation=%d blur=%d',
                source_step,
                means.get('negative_support', 0.0),
                means.get('negative_weight', 0.0),
                means.get('positive_support', 0.0),
                means.get('gt_coverage', 0.0),
                self.lung_mask_dilation, self.lung_mask_blur)
        if self._anatomy_debug:
            logger.info(
                '[LUNG-BIAS] iter=%d levels=%s scale=%s delta_rel=%s',
                source_step, self.lung_anatomy_levels,
                ['%.3e' % value
                 for value in self._anatomy_debug.get('scales', [])],
                ['%.3e' % value
                 for value in self._anatomy_debug.get('delta_rel', [])])
        if self._quality_debug:
            logger.info(
                '[LOC-Q] iter=%d pred=%.3f target=%.3f mae=%.3f '
                'corr=%.3f positives=%d alpha=%.2f',
                source_step,
                self._quality_debug.get('pred_mean', 0.0),
                self._quality_debug.get('target_mean', 0.0),
                self._quality_debug.get('mae', 0.0),
                self._quality_debug.get('corr', 0.0),
                self._quality_debug.get('positive_count', 0),
                self.localization_quality_alpha)

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
             img_metas, gt_bboxes_ignore=None):
        self._lung_debug_seen = set()
        self._lung_debug = {}
        losses = super().loss(
            cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        if losses is not None and self.use_localization_quality:
            losses['loss_loc_quality'] = self._v320_quality_loss(
                bbox_preds, gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)
        self._v320_emit_debug()
        return losses


@DETECTORS.register_module(force=True)
class RetinaNAMAttV320(RetinaNAMAttV319):
    """Final single-file detector profile for the five-way ablation."""

    def _collect_lung_masks(self, img_metas, device,
                            batch_input_shape=None):
        if not bool(getattr(
                self.bbox_head, 'use_lung_anatomy_bias', False)):
            return super()._collect_lung_masks(
                img_metas, device, batch_input_shape=batch_input_shape)
        aligned = []
        for index, meta in enumerate(img_metas):
            if not isinstance(meta, dict) or meta.get('lung_mask') is None:
                filename = (
                    meta.get('filename', '<unknown>')
                    if isinstance(meta, dict) else '<unknown>')
                raise KeyError(
                    f'Missing lung_mask for batch item {index}: {filename}')
            aligned.append(_align_lung_mask_to_image_meta(
                meta['lung_mask'], meta,
                batch_input_shape=batch_input_shape))
        return torch.stack(aligned).to(
            device=device, dtype=torch.float32)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('experiment_tag', 'V320_G2_CONTROL_INTEGRATED')
        super().__init__(*args, **kwargs)
        self.integrated_build_id = NAMLAB_LUNG_BUILD_ID
        if not isinstance(self.bbox_head, RetinaNAMGuideAttHeadV320):
            raise TypeError('RetinaNAMAttV320 requires V320 head')
        logging.getLogger('mmdet').info(
            '[NAM-LUNG] id=%s file=%s base=%s neg=%s far_w=%.2f '
            'bias=%s levels=%s loc_q=%s q_alpha=%.2f q_w=%.2f',
            NAMLAB_LUNG_BUILD_ID, __file__, NAMLAB_SLIM_BUILD_ID,
            self.bbox_head.use_lung_negative_weighting,
            self.bbox_head.lung_negative_far_weight,
            self.bbox_head.use_lung_anatomy_bias,
            self.bbox_head.lung_anatomy_levels,
            self.bbox_head.use_localization_quality,
            self.bbox_head.localization_quality_alpha,
            self.bbox_head.localization_quality_loss_weight)


# ============================================================================
# v3.21: RNG-fair lung anatomy and late-training stability
# ============================================================================
NAMLAB_V321_BUILD_ID = (
    'v3.21-g2-rngfair-reglung-ema-ready-20260805')


@HEADS.register_module(force=True)
class RetinaNAMGuideAttHeadV321(RetinaNAMGuideAttHeadV320):
    """Strict G2 head with RNG-fair lung-coordinate injection.

    ``lung_anatomy_injection`` controls only the forward route.  The anatomy
    projections are instantiated for every mode inside ``fork_rng`` and are
    zero-initialized, so changing the ablation switch cannot perturb the
    initialization of SymFormer, Retina towers, or FDR.

    Modes:
        off: exact G2 forward path.
        public: controlled reproduction of V320 A2 (before SymFormer).
        regression: inject after the regression tower only; classification
            and the shared SymFormer representation remain unchanged.
    """

    _V321_MODES = ('off', 'public', 'regression')

    def __init__(
            self, *args,
            lung_anatomy_injection='off',
            lung_anatomy_levels=(3,),
            lung_anatomy_residual_init=0.01,
            lung_anatomy_residual_max=0.05,
            **kwargs):
        mode = str(lung_anatomy_injection).lower()
        if mode not in self._V321_MODES:
            raise ValueError(
                f'lung_anatomy_injection must be one of '
                f'{self._V321_MODES}, got {mode!r}')
        levels = tuple(int(value) for value in lung_anatomy_levels)
        if any(level not in (0, 1, 2, 3) for level in levels):
            raise ValueError('lung_anatomy_levels must be within P2..P5')
        if not levels and mode != 'off':
            raise ValueError('enabled lung anatomy requires at least one level')
        initial = float(lung_anatomy_residual_init)
        maximum = float(lung_anatomy_residual_max)
        if not 0.0 < initial < maximum:
            raise ValueError(
                'lung_anatomy_residual_init must satisfy 0 < init < max')

        # These attributes must exist before AnchorHead dynamically calls the
        # most-derived _init_layers() from inside its constructor.
        self.v321_lung_anatomy_mode = mode
        self.v321_lung_anatomy_levels = levels
        self.v321_lung_anatomy_residual_init = initial
        self.v321_lung_anatomy_residual_max = maximum
        self._v321_anatomy_debug = {}
        self.v321_core_sha256 = None

        # V320's conditional public branch caused the RNG confound.  It must
        # stay disabled; V321 implements both public and regression routes
        # using an always-present, RNG-isolated module set below.
        if bool(kwargs.get('lung_anatomy_bias', False)):
            raise ValueError(
                'V321 forbids legacy lung_anatomy_bias=True; use '
                'lung_anatomy_injection="public" for a fair reproduction')
        kwargs['lung_anatomy_bias'] = False
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        # Conv2d constructors consume the CPU RNG even when their weights are
        # immediately zeroed.  Preserve that RNG across both dynamic calls to
        # _init_layers so all V321 ablations share identical G2 parameters.
        with torch.random.fork_rng(devices=[]):
            self.v321_lung_anatomy_proj = nn.ModuleList([
                nn.Conv2d(3, self.feat_channels, 3, padding=1)
                for _ in range(self.anchor_generator.num_levels)
            ])
        raw = math.atanh(
            self.v321_lung_anatomy_residual_init
            / self.v321_lung_anatomy_residual_max)
        self.v321_lung_anatomy_scale_raw = nn.Parameter(torch.full(
            (self.anchor_generator.num_levels,), raw,
            dtype=torch.float32))
        self._v321_reset_anatomy_layers()

    def _v321_reset_anatomy_layers(self):
        for projection in self.v321_lung_anatomy_proj:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def init_weights(self):
        super().init_weights()
        self._v321_reset_anatomy_layers()
        self.v321_core_sha256 = self._v321_core_parameter_sha256()

    def _v321_core_parameter_sha256(self):
        """Hash non-V321 parameters to prove initialization equivalence."""
        digest = hashlib.sha256()
        excluded = (
            'v321_lung_anatomy_proj',
            'v321_lung_anatomy_scale_raw')
        with torch.no_grad():
            for name, parameter in sorted(self.named_parameters()):
                if name.startswith(excluded):
                    continue
                value = parameter.detach().cpu().contiguous()
                digest.update(name.encode('utf-8'))
                digest.update(str(tuple(value.shape)).encode('ascii'))
                digest.update(str(value.dtype).encode('ascii'))
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def effective_v321_lung_anatomy_scales(self):
        return self.v321_lung_anatomy_residual_max * torch.tanh(
            self.v321_lung_anatomy_scale_raw)

    def _v321_anatomy_delta(self, feat, lung_masks, level_index):
        if lung_masks is None:
            raise RuntimeError(
                f'lung anatomy mode {self.v321_lung_anatomy_mode!r} '
                'requires aligned lung_masks')
        anatomy = _v320_lung_coordinate_channels(
            lung_masks, feat.shape[-2:],
            self.lung_mask_dilation,
            self.lung_mask_blur).to(dtype=feat.dtype)
        scale = self.effective_v321_lung_anatomy_scales()[
            level_index].to(dtype=feat.dtype)
        delta = scale * self.v321_lung_anatomy_proj[
            level_index](anatomy)
        relative = (
            delta.detach().float().norm()
            / feat.detach().float().norm().clamp_min(1e-8))
        self._v321_anatomy_debug.setdefault(
            'delta_rel', [0.0] * self.anchor_generator.num_levels)
        self._v321_anatomy_debug['delta_rel'][level_index] = float(
            relative.item())
        return delta

    def _v321_apply_public_anatomy(self, feats, lung_masks):
        output = list(feats)
        for level_index in self.v321_lung_anatomy_levels:
            output[level_index] = (
                output[level_index]
                + self._v321_anatomy_delta(
                    output[level_index], lung_masks, level_index))
        return output

    def forward_single(self, x, lbp_lab=None, lung_masks=None,
                       level_index=0, reg_detail_feat=None):
        output = super().forward_single(
            x, lbp_lab, lung_masks=lung_masks,
            level_index=level_index,
            reg_detail_feat=reg_detail_feat)
        if (self.v321_lung_anatomy_mode != 'regression'
                or level_index not in self.v321_lung_anatomy_levels):
            return output

        (cls_score, _, feat1, fused, reg_feat, gate_stats,
         _, _) = output
        # Inject after the regression tower.  The classification logits and
        # shared SymFormer output above are byte-for-byte independent of the
        # lung prior for a fixed set of core weights.
        reg_feat = reg_feat + self._v321_anatomy_delta(
            reg_feat, lung_masks, level_index)
        bbox_base = self.retina_reg(reg_feat)
        distribution_logits = None
        if self.use_d_fine_style:
            distribution_logits = self.d_fine_reg(reg_feat)
            normalized_residual, entropy = (
                _distribution_refinement_expectation(
                    distribution_logits,
                    self.num_anchors,
                    self.d_fine_reg_max))
            refine_scale = self.effective_d_fine_scales()[
                level_index].to(dtype=bbox_base.dtype)
            bbox_pred = bbox_base + refine_scale * normalized_residual
            self._dbg_d_fine_entropy[level_index] = float(
                entropy.detach().float().item())
            self._dbg_d_fine_expected_abs[level_index] = float(
                normalized_residual.detach().float().abs().mean().item())
        else:
            bbox_pred = bbox_base
        return (
            cls_score, bbox_pred, feat1, fused, reg_feat, gate_stats,
            bbox_base, distribution_logits)

    def forward(self, feats, *args, lung_masks=None, **kwargs):
        scales = self.effective_v321_lung_anatomy_scales()
        self._v321_anatomy_debug = {
            'scales': [
                float(value.detach().float().item()) for value in scales],
            'delta_rel': [0.0] * self.anchor_generator.num_levels,
        }
        if self.v321_lung_anatomy_mode == 'public':
            feats = self._v321_apply_public_anatomy(feats, lung_masks)
        return super().forward(
            feats, *args, lung_masks=lung_masks, **kwargs)

    def _v321_emit_debug(self):
        interval = int(getattr(self, 'debug_interval', 0))
        source_step = int(getattr(self, '_debug_step', 0)) - 1
        if (interval <= 0 or source_step % interval != 0
                or not _v320_rank_zero()):
            return
        logging.getLogger('mmdet').info(
            '[V321-ANAT] iter=%d mode=%s levels=%s scale=%s delta_rel=%s',
            source_step,
            self.v321_lung_anatomy_mode,
            self.v321_lung_anatomy_levels,
            ['%.3e' % value for value in
             self._v321_anatomy_debug.get('scales', [])],
            ['%.3e' % value for value in
             self._v321_anatomy_debug.get('delta_rel', [])])

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
             img_metas, gt_bboxes_ignore=None):
        losses = super().loss(
            cls_scores, bbox_preds, gt_bboxes, gt_labels,
            img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        self._v321_emit_debug()
        return losses


@DETECTORS.register_module(force=True)
class RetinaNAMAttV321(RetinaNAMAttV320):
    """V321 detector collecting masks only for the selected anatomy mode."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('experiment_tag', 'V321_G2_RNGFAIR_STABLE')
        super().__init__(*args, **kwargs)
        if not isinstance(self.bbox_head, RetinaNAMGuideAttHeadV321):
            raise TypeError('RetinaNAMAttV321 requires V321 head')
        self.integrated_build_id = NAMLAB_V321_BUILD_ID

    def _collect_lung_masks(self, img_metas, device,
                            batch_input_shape=None):
        if self.bbox_head.v321_lung_anatomy_mode == 'off':
            return None
        aligned = []
        for index, meta in enumerate(img_metas):
            if not isinstance(meta, dict) or meta.get('lung_mask') is None:
                filename = (
                    meta.get('filename', '<unknown>')
                    if isinstance(meta, dict) else '<unknown>')
                raise KeyError(
                    f'Missing lung_mask for batch item {index}: {filename}')
            aligned.append(_align_lung_mask_to_image_meta(
                meta['lung_mask'], meta,
                batch_input_shape=batch_input_shape))
        return torch.stack(aligned).to(
            device=device, dtype=torch.float32)


@HOOKS.register_module(force=True)
class NAMV321DebugHook(NAMDebugHook):
    """V321 build fingerprint plus anatomy-gradient diagnostics."""

    def before_run(self, runner):
        super().before_run(runner)
        model = (runner.model.module
                 if hasattr(runner.model, 'module') else runner.model)
        head = model.bbox_head
        core_hash = getattr(head, 'v321_core_sha256', None)
        if core_hash is None:
            core_hash = head._v321_core_parameter_sha256()
            head.v321_core_sha256 = core_hash
        ema_hooks = [
            type(hook).__name__ for hook in runner.hooks
            if 'EMA' in type(hook).__name__]
        runner.logger.info(
            '[V321-BUILD] id=%s file=%s mode=%s levels=%s '
            'core_sha256=%s ema_hooks=%s',
            NAMLAB_V321_BUILD_ID,
            __file__,
            head.v321_lung_anatomy_mode,
            head.v321_lung_anatomy_levels,
            core_hash,
            ema_hooks)

    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        model = (runner.model.module
                 if hasattr(runner.model, 'module') else runner.model)
        head = model.bbox_head
        interval = int(getattr(head, 'debug_interval', 0))
        debug_step = int(getattr(head, '_debug_step', 0))
        if (interval <= 0 or debug_step % interval != 0
                or not _v320_rank_zero()):
            return
        rms, coverage = _TBFusionCrossScale._gradient_group(
            head.v321_lung_anatomy_proj.parameters())
        scale_rms, scale_coverage = _TBFusionCrossScale._gradient_group(
            [head.v321_lung_anatomy_scale_raw])
        runner.logger.info(
            '[V321-GRAD] iter=%d anatomy=%.3e/%.2f scale=%.3e/%.2f',
            max(0, debug_step - 1), rms, coverage,
            scale_rms, scale_coverage)


# ============================================================================
# v3.30: original-pixel proposal refinement
# ============================================================================
NAMLAB_V330_BUILD_ID = (
    'v3.30-v321-raw-patch-local-bbox-refine-20260811')


class _V330LocalBlock(nn.Module):
    """Small residual block with local 3x3/5x5 receptive fields."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise3 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels)
        self.depthwise5 = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        update = self.depthwise3(x) + self.depthwise5(x)
        update = self.mix(F.silu(self.norm(update)))
        return x + update


class _V330RawPatchRefiner(nn.Module):
    """Independent low-stride encoder predicting coarse-box residuals.

    Input channel 0 is an untouched, unnormalised original-CXR crop scaled to
    [0, 1]. Channel 1 is the coarse-box support mask inside the context crop.
    The main detector score and box geometry are fused only after local visual
    encoding, keeping classification semantics separate from boundary detail.
    """

    def __init__(self, patch_size=192, base_channels=32,
                 max_log_scale=0.6931471805599453):
        super().__init__()
        self.patch_size = int(patch_size)
        self.max_log_scale = float(max_log_scale)
        c = int(base_channels)
        if self.patch_size < 64:
            raise ValueError('local_refine patch_size must be at least 64')
        if c < 8:
            raise ValueError('local_refine base_channels must be at least 8')
        self.encoder = nn.Sequential(
            nn.Conv2d(2, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            _V330LocalBlock(c),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 2), c * 2),
            nn.SiLU(inplace=True),
            _V330LocalBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.SiLU(inplace=True),
            _V330LocalBlock(c * 4),
            nn.AdaptiveAvgPool2d(1))
        self.geometry = nn.Sequential(
            nn.Linear(7, c * 2), nn.SiLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Linear(c * 6, c * 4), nn.SiLU(inplace=True),
            nn.Linear(c * 4, c * 2), nn.SiLU(inplace=True))
        self.delta_head = nn.Linear(c * 2, 4)
        self.quality_delta_head = nn.Linear(c * 2, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.quality_delta_head.weight)
        nn.init.zeros_(self.quality_delta_head.bias)

    def forward(self, patches, geometry):
        visual = self.encoder(patches).flatten(1)
        geom = self.geometry(geometry)
        fused = self.fusion(torch.cat([visual, geom], dim=1))
        delta = self.delta_head(fused)
        delta_xy = delta[:, :2].clamp(min=-1.0, max=1.0)
        delta_wh = delta[:, 2:].clamp(
            min=-self.max_log_scale, max=self.max_log_scale)
        quality_delta = self.quality_delta_head(fused).squeeze(1)
        return torch.cat([delta_xy, delta_wh], dim=1), quality_delta


@DETECTORS.register_module(force=True)
class RetinaNAMAttV330(RetinaNAMAttV321):
    """V321 plus an original-pixel, proposal-conditioned bbox refiner.

    The main detector is unchanged. Coarse boxes are detached, mapped back to
    the untouched decoded image and expanded into context crops. The local
    branch is trained with jittered GT boxes early and real detector boxes
    progressively later. It never changes classification scores.
    """

    def __init__(
            self, *args,
            local_refine=False,
            local_refine_patch_size=192,
            local_refine_context=1.5,
            local_refine_base_channels=32,
            local_refine_max_samples=8,
            local_refine_topk=20,
            local_refine_min_train_iou=0.30,
            local_refine_delta_weight=1.0,
            local_refine_giou_weight=0.5,
            local_refine_quality_weight=0.25,
            local_refine_gate_threshold=0.01,
            local_refine_pred_start_epoch=5,
            local_refine_pred_full_epoch=11,
            local_refine_preserve_rng=False,
            **kwargs):
        kwargs.setdefault(
            'experiment_tag', 'V330_RAW_PATCH_LOCAL_BBOX_REFINE')
        super().__init__(*args, **kwargs)
        self.use_local_refine = bool(local_refine)
        self.local_refine_context = float(local_refine_context)
        self.local_refine_max_samples = int(local_refine_max_samples)
        self.local_refine_topk = int(local_refine_topk)
        self.local_refine_min_train_iou = float(
            local_refine_min_train_iou)
        self.local_refine_delta_weight = float(local_refine_delta_weight)
        self.local_refine_giou_weight = float(local_refine_giou_weight)
        self.local_refine_quality_weight = float(
            local_refine_quality_weight)
        self.local_refine_gate_threshold = float(
            local_refine_gate_threshold)
        self.local_refine_pred_start_epoch = int(
            local_refine_pred_start_epoch)
        self.local_refine_pred_full_epoch = int(
            local_refine_pred_full_epoch)
        self.local_refine_preserve_rng = bool(local_refine_preserve_rng)
        if self.local_refine_context < 1.0:
            raise ValueError('local_refine_context must be >= 1.0')
        if self.local_refine_max_samples < 1 or self.local_refine_topk < 1:
            raise ValueError('local refinement sample counts must be positive')
        if not 0.0 <= self.local_refine_min_train_iou < 1.0:
            raise ValueError('local_refine_min_train_iou must be in [0, 1)')
        if self.local_refine_pred_full_epoch <= (
                self.local_refine_pred_start_epoch):
            raise ValueError(
                'local_refine_pred_full_epoch must exceed start epoch')
        if self.use_local_refine:
            if self.local_refine_preserve_rng:
                # Model construction happens on CPU in MMDetection.  Forking
                # CPU RNG here prevents an optional auxiliary branch from
                # shifting the first DropPath/dropout sequence of the main
                # detector.  V330 defaults to False for historical fidelity.
                with torch.random.fork_rng(devices=[], enabled=True):
                    self.local_refiner = _V330RawPatchRefiner(
                        patch_size=local_refine_patch_size,
                        base_channels=local_refine_base_channels)
            else:
                self.local_refiner = _V330RawPatchRefiner(
                    patch_size=local_refine_patch_size,
                    base_channels=local_refine_base_channels)
        else:
            self.local_refiner = None
        self._v330_debug = {}
        self.integrated_build_id = NAMLAB_V330_BUILD_ID

    @staticmethod
    def _v330_unwrap_original_images(original_img, batch_size):
        if original_img is None:
            raise KeyError(
                'local_refine=True requires original_img in the data '
                'pipeline. Add PreserveOriginalImage, ImageToTensor and '
                'ToDataContainer(cpu_only=True), then Collect original_img.')
        value = original_img
        while isinstance(value, (list, tuple)) and len(value) == 1:
            nested = value[0]
            if isinstance(nested, (list, tuple)):
                value = nested
            else:
                break
        if isinstance(value, torch.Tensor):
            if value.dim() == 4:
                images = [item for item in value]
            elif value.dim() == 3:
                images = [value]
            else:
                raise ValueError(
                    f'original_img tensor must be CHW/NCHW, got {value.shape}')
        elif isinstance(value, (list, tuple)):
            images = list(value)
        else:
            raise TypeError(
                f'Unsupported original_img container {type(value)!r}')
        if len(images) != batch_size:
            raise ValueError(
                f'original_img batch {len(images)} != img_metas '
                f'batch {batch_size}')
        output = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)
            if image.dim() != 3:
                raise ValueError(
                    f'each original image must be CHW, got {image.shape}')
            output.append(image)
        return output

    @staticmethod
    def _v330_flip_boxes(boxes, width, height, direction):
        output = boxes.clone()
        if direction == 'horizontal':
            output[:, 0] = width - boxes[:, 2]
            output[:, 2] = width - boxes[:, 0]
        elif direction == 'vertical':
            output[:, 1] = height - boxes[:, 3]
            output[:, 3] = height - boxes[:, 1]
        elif direction == 'diagonal':
            output[:, 0] = width - boxes[:, 2]
            output[:, 2] = width - boxes[:, 0]
            output[:, 1] = height - boxes[:, 3]
            output[:, 3] = height - boxes[:, 1]
        else:
            raise ValueError(f'Unsupported flip direction {direction!r}')
        return output

    @classmethod
    def _v330_boxes_to_original(cls, boxes, meta, already_rescaled=False):
        output = boxes.clone()
        ori_h, ori_w = [int(v) for v in meta['ori_shape'][:2]]
        img_h, img_w = [int(v) for v in meta['img_shape'][:2]]
        if bool(meta.get('flip', False)):
            width = ori_w if already_rescaled else img_w
            height = ori_h if already_rescaled else img_h
            output = cls._v330_flip_boxes(
                output, width, height,
                meta.get('flip_direction', 'horizontal'))
        if not already_rescaled:
            scale = output.new_tensor(meta.get('scale_factor', 1.0))
            if scale.numel() == 1:
                output = output / scale
            else:
                output = output / scale.flatten()[:4]
        output[:, 0::2].clamp_(0, ori_w)
        output[:, 1::2].clamp_(0, ori_h)
        return output

    @classmethod
    def _v330_boxes_from_original(cls, boxes, meta):
        output = boxes.clone()
        scale = output.new_tensor(meta.get('scale_factor', 1.0))
        if scale.numel() == 1:
            output = output * scale
        else:
            output = output * scale.flatten()[:4]
        if bool(meta.get('flip', False)):
            img_h, img_w = [int(v) for v in meta['img_shape'][:2]]
            output = cls._v330_flip_boxes(
                output, img_w, img_h,
                meta.get('flip_direction', 'horizontal'))
        return output

    @staticmethod
    def _v330_encode_delta(proposals, targets):
        p_wh = (proposals[:, 2:] - proposals[:, :2]).clamp_min(1.0)
        p_xy = 0.5 * (proposals[:, 2:] + proposals[:, :2])
        t_wh = (targets[:, 2:] - targets[:, :2]).clamp_min(1.0)
        t_xy = 0.5 * (targets[:, 2:] + targets[:, :2])
        return torch.cat([
            (t_xy - p_xy) / p_wh,
            torch.log(t_wh / p_wh)], dim=1)

    @staticmethod
    def _v330_decode_delta(proposals, deltas, image_shapes):
        p_wh = (proposals[:, 2:] - proposals[:, :2]).clamp_min(1.0)
        p_xy = 0.5 * (proposals[:, 2:] + proposals[:, :2])
        out_xy = p_xy + deltas[:, :2] * p_wh
        out_wh = p_wh * torch.exp(deltas[:, 2:])
        output = torch.cat([
            out_xy - 0.5 * out_wh,
            out_xy + 0.5 * out_wh], dim=1)
        maximum = output.new_tensor([
            [width, height, width, height]
            for height, width in image_shapes])
        return torch.minimum(torch.clamp_min(output, 0.0), maximum)

    @staticmethod
    def _v330_gray_original(image):
        # original_img is intentionally cpu_only. Crop/resize on CPU and move
        # only the compact patch batch to GPU; this scales to native CXR sizes.
        image = image.to(dtype=torch.float32)
        if image.shape[0] == 1:
            gray = image
        elif image.shape[0] >= 3:
            # LoadImageFromFile uses BGR before Normalize(to_rgb=True).
            gray = (0.114 * image[0:1]
                    + 0.587 * image[1:2]
                    + 0.299 * image[2:3])
        else:
            raise ValueError(
                f'original image has unsupported channels {image.shape[0]}')
        if gray.max().item() > 1.5:
            gray = gray / 255.0
        return gray.clamp(0.0, 1.0)

    def _v330_make_patch(self, gray, proposal):
        _, height, width = gray.shape
        if isinstance(proposal, torch.Tensor):
            proposal = proposal.detach().cpu().tolist()
        x1, y1, x2, y2 = [float(v) for v in proposal]
        box_w = max(x2 - x1, 2.0)
        box_h = max(y2 - y1, 2.0)
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        crop_w = box_w * self.local_refine_context
        crop_h = box_h * self.local_refine_context
        left = max(0, int(math.floor(center_x - 0.5 * crop_w)))
        top = max(0, int(math.floor(center_y - 0.5 * crop_h)))
        right = min(width, int(math.ceil(center_x + 0.5 * crop_w)))
        bottom = min(height, int(math.ceil(center_y + 0.5 * crop_h)))
        if right <= left or bottom <= top:
            raise ValueError(
                f'degenerate local crop {(left, top, right, bottom)} '
                f'from proposal {(x1, y1, x2, y2)}')
        crop = gray[:, top:bottom, left:right]
        crop = F.interpolate(
            crop.unsqueeze(0),
            size=(self.local_refiner.patch_size,
                  self.local_refiner.patch_size),
            mode='bilinear', align_corners=False).squeeze(0)
        mask = crop.new_zeros(
            (1, self.local_refiner.patch_size,
             self.local_refiner.patch_size))
        scale_x = self.local_refiner.patch_size / max(right - left, 1)
        scale_y = self.local_refiner.patch_size / max(bottom - top, 1)
        mx1 = max(0, min(self.local_refiner.patch_size,
                         int(math.floor((x1 - left) * scale_x))))
        my1 = max(0, min(self.local_refiner.patch_size,
                         int(math.floor((y1 - top) * scale_y))))
        mx2 = max(0, min(self.local_refiner.patch_size,
                         int(math.ceil((x2 - left) * scale_x))))
        my2 = max(0, min(self.local_refiner.patch_size,
                         int(math.ceil((y2 - top) * scale_y))))
        if mx2 > mx1 and my2 > my1:
            mask[:, my1:my2, mx1:mx2] = 1.0
        return torch.cat([crop, mask], dim=0)

    @staticmethod
    def _v330_geometry(proposals, scores, image_shapes):
        rows = []
        for box, score, (height, width) in zip(
                proposals, scores, image_shapes):
            box_w = (box[2] - box[0]).clamp_min(1.0)
            box_h = (box[3] - box[1]).clamp_min(1.0)
            center_x = 0.5 * (box[0] + box[2]) / max(width, 1)
            center_y = 0.5 * (box[1] + box[3]) / max(height, 1)
            norm_w = box_w / max(width, 1)
            norm_h = box_h / max(height, 1)
            aspect = torch.log(box_w / box_h).clamp(-3.0, 3.0) / 3.0
            area = (norm_w * norm_h).sqrt()
            rows.append(torch.stack([
                center_x, center_y, norm_w, norm_h,
                score, aspect, area]))
        return torch.stack(rows, dim=0)

    def _v330_run_refiner(self, originals, proposals, scores,
                          image_indices):
        device = next(self.local_refiner.parameters()).device
        grays = [self._v330_gray_original(image)
                 for image in originals]
        image_shapes = [tuple(gray.shape[-2:]) for gray in grays]
        # One batched device synchronization instead of one .cpu() call per
        # proposal. Crop coordinates are deliberately non-differentiable.
        proposal_values = proposals.detach().cpu().tolist()
        patches = torch.stack([
            self._v330_make_patch(grays[image_index], proposal)
            for proposal, image_index in zip(
                proposal_values, image_indices)],
            dim=0).to(device=device, non_blocking=True)
        proposal_shapes = [image_shapes[index] for index in image_indices]
        geometry = self._v330_geometry(
            proposals, scores, proposal_shapes)
        deltas, quality_delta = self.local_refiner(patches, geometry)
        refined = self._v330_decode_delta(
            proposals, deltas, proposal_shapes)
        return deltas, quality_delta, refined, patches

    def _v330_prediction_fraction(self):
        epoch = int(self._nam_epoch) + 1
        if epoch < self.local_refine_pred_start_epoch:
            return 0.0
        if epoch >= self.local_refine_pred_full_epoch:
            return 0.8
        span = max(1, self.local_refine_pred_full_epoch
                   - self.local_refine_pred_start_epoch)
        return 0.5 + 0.3 * (
            epoch - self.local_refine_pred_start_epoch) / span

    @staticmethod
    def _v330_jitter_gt(gt_boxes):
        if gt_boxes.numel() == 0:
            return gt_boxes
        wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp_min(2.0)
        center = 0.5 * (gt_boxes[:, 2:] + gt_boxes[:, :2])
        center_noise = (
            torch.rand_like(center) * 0.30 - 0.15) * wh
        scale = torch.exp(
            torch.empty_like(wh).uniform_(math.log(0.75), math.log(1.50)))
        jitter_center = center + center_noise
        jitter_wh = wh * scale
        return torch.cat([
            jitter_center - 0.5 * jitter_wh,
            jitter_center + 0.5 * jitter_wh], dim=1)

    def _v330_local_refine_train(
            self, original_img, img_metas, gt_bboxes, gt_labels,
            cls_scores, bbox_preds):
        if not self.use_local_refine:
            return {}
        originals = self._v330_unwrap_original_images(
            original_img, len(img_metas))
        prediction_fraction = self._v330_prediction_fraction()
        predicted_results = None
        if prediction_fraction > 0.0:
            with torch.no_grad():
                predicted_results = self.bbox_head.get_bboxes(
                    cls_scores, bbox_preds, img_metas,
                    cfg=self.bbox_head.test_cfg,
                    rescale=False, with_nms=True)

        all_proposals = []
        all_targets = []
        all_scores = []
        all_image_indices = []
        predicted_count = 0
        jitter_count = 0
        for image_index, (meta, transformed_gt) in enumerate(
                zip(img_metas, gt_bboxes)):
            original_gt = self._v330_boxes_to_original(
                transformed_gt.detach(), meta, already_rescaled=False)
            if original_gt.numel() == 0:
                continue
            ori_h, ori_w = [int(v) for v in meta['ori_shape'][:2]]
            predicted_quota = int(round(
                self.local_refine_max_samples * prediction_fraction))
            image_proposals = []
            image_targets = []
            image_scores = []
            if predicted_results is not None and predicted_quota > 0:
                det_boxes, _ = predicted_results[image_index]
                det_boxes = det_boxes[:self.local_refine_topk]
                if det_boxes.numel() > 0:
                    proposals = self._v330_boxes_to_original(
                        det_boxes[:, :4], meta, already_rescaled=False)
                    ious = bbox_overlaps(proposals, original_gt)
                    best_iou, best_gt = ious.max(dim=1)
                    valid = torch.nonzero(
                        best_iou >= self.local_refine_min_train_iou,
                        as_tuple=False).flatten()
                    if valid.numel() > 0:
                        order = det_boxes[valid, 4].argsort(descending=True)
                        valid = valid[order[:predicted_quota]]
                        image_proposals.append(proposals[valid].detach())
                        image_targets.append(original_gt[best_gt[valid]])
                        image_scores.append(det_boxes[valid, 4].detach())
                        predicted_count += int(valid.numel())
            used = sum(len(value) for value in image_proposals)
            jitter_quota = max(1, self.local_refine_max_samples - used)
            repeats = int(math.ceil(jitter_quota / len(original_gt)))
            jitter = torch.cat([
                self._v330_jitter_gt(original_gt)
                for _ in range(repeats)], dim=0)[:jitter_quota]
            jitter[:, 0::2].clamp_(0, ori_w)
            jitter[:, 1::2].clamp_(0, ori_h)
            target_indices = torch.arange(
                jitter_quota, device=original_gt.device) % len(original_gt)
            image_proposals.append(jitter.detach())
            image_targets.append(original_gt[target_indices])
            image_scores.append(jitter.new_ones(jitter_quota))
            jitter_count += jitter_quota

            proposals = torch.cat(image_proposals, dim=0)
            targets = torch.cat(image_targets, dim=0)
            scores = torch.cat(image_scores, dim=0)
            all_proposals.append(proposals)
            all_targets.append(targets)
            all_scores.append(scores)
            all_image_indices.extend([image_index] * len(proposals))

        if not all_proposals:
            zero = cls_scores[0].sum() * 0.0
            return {
                'loss_local_delta': zero,
                'loss_local_giou': zero,
                'loss_local_quality': zero}
        proposals = torch.cat(all_proposals, dim=0)
        targets = torch.cat(all_targets, dim=0)
        scores = torch.cat(all_scores, dim=0)
        target_delta = self._v330_encode_delta(proposals, targets)
        deltas, quality_delta, refined, patches = self._v330_run_refiner(
            originals, proposals, scores, all_image_indices)
        delta_loss = F.smooth_l1_loss(
            deltas, target_delta, reduction='mean', beta=1.0 / 9.0)
        aligned_giou = bbox_overlaps(
            refined, targets, mode='giou', is_aligned=True)
        giou_loss = (1.0 - aligned_giou).mean()
        with torch.no_grad():
            coarse_iou = bbox_overlaps(
                proposals, targets, is_aligned=True)
            refined_iou = bbox_overlaps(
                refined.detach(), targets, is_aligned=True)
            quality_target = refined_iou - coarse_iou
        quality_loss = F.smooth_l1_loss(
            quality_delta, quality_target, reduction='mean', beta=0.05)
        with torch.no_grad():
            improved = refined_iou > coarse_iou
            self._v330_debug = {
                'epoch': int(self._nam_epoch) + 1,
                'pred_fraction': float(prediction_fraction),
                'samples': int(len(proposals)),
                'predicted': int(predicted_count),
                'jittered': int(jitter_count),
                'coarse_iou': float(coarse_iou.mean().item()),
                'refined_iou': float(refined_iou.mean().item()),
                'improved_fraction': float(improved.float().mean().item()),
                'quality_delta_pred': float(quality_delta.mean().item()),
                'quality_delta_target': float(quality_target.mean().item()),
                'delta_abs': float(deltas.abs().mean().item()),
                'patch_mean': float(patches[:, :1].mean().item()),
                'patch_std': float(patches[:, :1].std().item()),
            }
        return {
            'loss_local_delta': self.local_refine_delta_weight * delta_loss,
            'loss_local_giou': self.local_refine_giou_weight * giou_loss,
            'loss_local_quality': (
                self.local_refine_quality_weight * quality_loss)}

    def _v330_local_refine_test(
            self, results_list, original_img, img_metas,
            results_are_rescaled):
        if not self.use_local_refine:
            return results_list
        originals = self._v330_unwrap_original_images(
            original_img, len(img_metas))
        output = []
        accepted_total = 0
        candidate_total = 0
        quality_values = []
        for image_index, ((det_boxes, det_labels), meta) in enumerate(
                zip(results_list, img_metas)):
            if det_boxes.numel() == 0:
                output.append((det_boxes, det_labels))
                continue
            count = min(self.local_refine_topk, len(det_boxes))
            source_boxes = det_boxes[:count, :4]
            original_boxes = self._v330_boxes_to_original(
                source_boxes, meta,
                already_rescaled=results_are_rescaled)
            scores = det_boxes[:count, 4]
            indices = [image_index] * count
            _, quality_delta, refined, _ = self._v330_run_refiner(
                originals, original_boxes, scores, indices)
            accept = quality_delta > self.local_refine_gate_threshold
            selected = torch.where(
                accept[:, None], refined, original_boxes)
            if not results_are_rescaled:
                selected = self._v330_boxes_from_original(selected, meta)
            updated = det_boxes.clone()
            updated[:count, :4] = selected
            output.append((updated, det_labels))
            candidate_total += count
            accepted_total += int(accept.sum().item())
            quality_values.append(quality_delta.detach())
        if quality_values:
            quality = torch.cat(quality_values)
            self._v330_debug.update({
                'test_candidates': int(candidate_total),
                'test_accepted': int(accepted_total),
                'test_accept_fraction': float(
                    accepted_total / max(candidate_total, 1)),
                'test_quality_mean': float(quality.mean().item()),
            })
        return output


@HOOKS.register_module(force=True)
class NAMV330DebugHook(NAMV321DebugHook):
    """Build fingerprint and local-refinement health diagnostics."""

    def before_run(self, runner):
        super().before_run(runner)
        model = (runner.model.module
                 if hasattr(runner.model, 'module') else runner.model)
        runner.logger.info(
            '[V330-BUILD] id=%s file=%s enabled=%s patch=%s context=%.2f '
            'base_c=%s max_samples=%s topk=%s min_iou=%.2f gate=%.3f '
            'schedule=%s/%s',
            NAMLAB_V330_BUILD_ID, __file__, model.use_local_refine,
            getattr(model.local_refiner, 'patch_size', None),
            model.local_refine_context,
            (model.local_refiner.encoder[0].out_channels
             if model.local_refiner is not None else None),
            model.local_refine_max_samples, model.local_refine_topk,
            model.local_refine_min_train_iou,
            model.local_refine_gate_threshold,
            model.local_refine_pred_start_epoch,
            model.local_refine_pred_full_epoch)

    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        model = (runner.model.module
                 if hasattr(runner.model, 'module') else runner.model)
        if not model.use_local_refine:
            return
        head = model.bbox_head
        interval = int(getattr(head, 'debug_interval', 0))
        debug_step = int(getattr(head, '_debug_step', 0))
        if (interval <= 0 or debug_step % interval != 0
                or not _v320_rank_zero()):
            return
        stats = model._v330_debug
        rms, coverage = _TBFusionCrossScale._gradient_group(
            model.local_refiner.parameters())
        runner.logger.info(
            '[V330-LOCAL] iter=%d epoch=%s samples=%s pred/jit=%s/%s '
            'pred_frac=%.2f IoU=%.3f->%.3f improve=%.2f '
            'q=%.3f/%.3f delta=%.3e patch=%.3f/%.3f grad=%.3e/%.2f',
            max(0, debug_step - 1), stats.get('epoch', -1),
            stats.get('samples', 0), stats.get('predicted', 0),
            stats.get('jittered', 0), stats.get('pred_fraction', 0.0),
            stats.get('coarse_iou', 0.0),
            stats.get('refined_iou', 0.0),
            stats.get('improved_fraction', 0.0),
            stats.get('quality_delta_pred', 0.0),
            stats.get('quality_delta_target', 0.0),
            stats.get('delta_abs', 0.0), stats.get('patch_mean', 0.0),
            stats.get('patch_std', 0.0), rms, coverage)


# ---------------------------------------------------------------------------
# V331: distribution-aligned, side-aware raw-patch bbox refinement.
#
# V330 is intentionally left intact above so old checkpoints/configs remain
# reproducible.  V331 fixes four diagnosed failure modes without changing the
# backbone, CrossScale neck, or the original dense head:
#   1) no global RNG is consumed by proposal jitter;
#   2) spatial side profiles replace global average pooled cx/cy/w/h offsets;
#   3) real unmatched detector proposals supervise localisation-quality gates;
#   4) changed boxes are quality re-ranked and passed through NMS again.
# ---------------------------------------------------------------------------

NAMLAB_V331_BUILD_ID = (
    'v3.31-side-aware-quality-gated-local-refine-20260811')


class _V331SideAwareRefiner(nn.Module):
    """Low-stride raw-patch encoder with four direction-specific heads.

    Horizontal and vertical feature profiles retain the ordering of local
    evidence.  Each side is predicted independently as a normalised edge
    displacement (dx1, dy1, dx2, dy2), avoiding the spatial information loss
    caused by V330's AdaptiveAvgPool2d(1) + centre/size representation.
    """

    def __init__(self, patch_size=192, base_channels=32, profile_bins=16,
                 max_edge_shift=1.0):
        super().__init__()
        self.patch_size = int(patch_size)
        self.profile_bins = int(profile_bins)
        self.max_edge_shift = float(max_edge_shift)
        c = int(base_channels)
        if self.patch_size < 64:
            raise ValueError('local_refine patch_size must be at least 64')
        if c < 8 or self.profile_bins < 4:
            raise ValueError('invalid V331 local-refiner capacity')
        self.encoder = nn.Sequential(
            nn.Conv2d(2, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c), nn.SiLU(inplace=True),
            _V330LocalBlock(c),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 2), c * 2), nn.SiLU(inplace=True),
            _V330LocalBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 4), c * 4), nn.SiLU(inplace=True),
            _V330LocalBlock(c * 4))
        channels = c * 4
        self.geometry = nn.Sequential(
            nn.Linear(7, c * 2), nn.SiLU(inplace=True))
        side_input = channels * self.profile_bins + c * 2

        def make_side_head():
            return nn.Sequential(
                nn.Linear(side_input, c * 4), nn.SiLU(inplace=True),
                nn.Linear(c * 4, c), nn.SiLU(inplace=True),
                nn.Linear(c, 1))

        self.left_head = make_side_head()
        self.right_head = make_side_head()
        self.top_head = make_side_head()
        self.bottom_head = make_side_head()
        # A 4x4 grid preserves coarse spatial layout for quality estimation.
        quality_input = channels * 16 + c * 2 + 4
        self.quality = nn.Sequential(
            nn.Linear(quality_input, c * 4), nn.SiLU(inplace=True),
            nn.Linear(c * 4, c * 2), nn.SiLU(inplace=True))
        self.coarse_quality_head = nn.Linear(c * 2, 1)
        self.refined_quality_head = nn.Linear(c * 2, 1)

        # Identity refinement and reject-all gate at initialisation.
        for head in (self.left_head[-1], self.right_head[-1],
                     self.top_head[-1], self.bottom_head[-1],
                     self.coarse_quality_head, self.refined_quality_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, patches, geometry):
        features = self.encoder(patches)
        geom = self.geometry(geometry)
        # [B,C,W] and [B,C,H], retaining left/right and top/bottom order.
        horizontal = F.adaptive_avg_pool1d(
            features.mean(dim=2), self.profile_bins).flatten(1)
        vertical = F.adaptive_avg_pool1d(
            features.mean(dim=3), self.profile_bins).flatten(1)
        horizontal = torch.cat([horizontal, geom], dim=1)
        vertical = torch.cat([vertical, geom], dim=1)
        edge_delta = torch.cat([
            self.left_head(horizontal), self.top_head(vertical),
            self.right_head(horizontal), self.bottom_head(vertical)], dim=1)
        edge_delta = torch.tanh(edge_delta) * self.max_edge_shift

        quality_feature = F.adaptive_avg_pool2d(
            features, (4, 4)).flatten(1)
        quality_feature = self.quality(torch.cat(
            [quality_feature, geom, edge_delta], dim=1))
        coarse_logit = self.coarse_quality_head(
            quality_feature).squeeze(1)
        refined_logit = self.refined_quality_head(
            quality_feature).squeeze(1)
        return edge_delta, coarse_logit, refined_logit


@DETECTORS.register_module(force=True)
class RetinaNAMAttV331(RetinaNAMAttV330):
    """V330-compatible detector with a corrected local refinement loop."""

    def __init__(
            self, *args, local_refine_profile_bins=16,
            local_refine_max_edge_shift=1.0,
            local_refine_negative_fraction=0.34,
            local_refine_improve_epsilon=0.02,
            local_refine_min_quality=0.30,
            local_refine_quality_alpha=0.0,
            local_refine_inference_mode='gated',
            local_refine_post_nms=True,
            local_refine_post_nms_iou=None,
            local_refine_test_debug_interval=50,
            **kwargs):
        kwargs.setdefault(
            'experiment_tag', 'V331_SIDE_AWARE_LOCAL_BBOX_REFINE')
        kwargs.setdefault('local_refine_preserve_rng', True)
        super().__init__(*args, **kwargs)
        self.local_refine_negative_fraction = float(
            local_refine_negative_fraction)
        self.local_refine_improve_epsilon = float(
            local_refine_improve_epsilon)
        self.local_refine_min_quality = float(local_refine_min_quality)
        self.local_refine_quality_alpha = float(local_refine_quality_alpha)
        self.local_refine_inference_mode = str(
            local_refine_inference_mode).lower()
        self.local_refine_post_nms = bool(local_refine_post_nms)
        self.local_refine_post_nms_iou = local_refine_post_nms_iou
        self.local_refine_test_debug_interval = int(
            local_refine_test_debug_interval)
        if not 0.0 <= self.local_refine_negative_fraction < 1.0:
            raise ValueError('local_refine_negative_fraction must be in [0,1)')
        if self.local_refine_inference_mode not in ('off', 'all', 'gated'):
            raise ValueError(
                'local_refine_inference_mode must be off/all/gated')
        if self.use_local_refine:
            # Replace only V330's local module; all main-detector weights and
            # behaviours remain unchanged.
            old = self.local_refiner
            with torch.random.fork_rng(devices=[], enabled=True):
                self.local_refiner = _V331SideAwareRefiner(
                    patch_size=old.patch_size,
                    base_channels=old.encoder[0].out_channels,
                    profile_bins=local_refine_profile_bins,
                    max_edge_shift=local_refine_max_edge_shift)
            del old
        self._v331_test_accumulator = {
            'images': 0, 'candidates': 0, 'accepted': 0,
            'pre_nms': 0, 'post_nms': 0,
            'coarse_quality_sum': 0.0, 'refined_quality_sum': 0.0}
        self.integrated_build_id = NAMLAB_V331_BUILD_ID

    @staticmethod
    def _v331_encode_edge_delta(proposals, targets):
        wh = (proposals[:, 2:] - proposals[:, :2]).clamp_min(1.0)
        scale = torch.cat([wh, wh], dim=1)
        return (targets - proposals) / scale

    @staticmethod
    def _v331_decode_edge_delta(proposals, deltas, image_shapes):
        wh = (proposals[:, 2:] - proposals[:, :2]).clamp_min(1.0)
        scale = torch.cat([wh, wh], dim=1)
        raw = proposals + deltas * scale
        # Avoid inverted/degenerate boxes while preserving side identities.
        max_width = raw.new_tensor(
            [max(float(width), 1.0) for _, width in image_shapes])
        max_height = raw.new_tensor(
            [max(float(height), 1.0) for height, _ in image_shapes])
        left = torch.minimum(
            torch.clamp_min(raw[:, 0], 0.0), max_width - 1.0)
        top = torch.minimum(
            torch.clamp_min(raw[:, 1], 0.0), max_height - 1.0)
        right = torch.minimum(
            torch.maximum(raw[:, 2], left + 1.0), max_width)
        bottom = torch.minimum(
            torch.maximum(raw[:, 3], top + 1.0), max_height)
        return torch.stack([left, top, right, bottom], dim=1)

    @staticmethod
    def _v331_stateless_uniform(shape, key, device, dtype):
        """Deterministic local randomness without advancing global RNG."""
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(key) & 0x7FFFFFFFFFFFFFFF)
        return torch.rand(
            shape, generator=generator, device='cpu',
            dtype=torch.float32).to(device=device, dtype=dtype)

    def _v331_jitter_gt(self, gt_boxes, key):
        if gt_boxes.numel() == 0:
            return gt_boxes
        wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp_min(2.0)
        center = 0.5 * (gt_boxes[:, 2:] + gt_boxes[:, :2])
        center_u = self._v331_stateless_uniform(
            center.shape, key, center.device, center.dtype)
        scale_u = self._v331_stateless_uniform(
            wh.shape, key + 104729, wh.device, wh.dtype)
        center_noise = (center_u * 0.30 - 0.15) * wh
        log_min, log_max = math.log(0.75), math.log(1.50)
        scale = torch.exp(log_min + scale_u * (log_max - log_min))
        jitter_center = center + center_noise
        jitter_wh = wh * scale
        return torch.cat([
            jitter_center - 0.5 * jitter_wh,
            jitter_center + 0.5 * jitter_wh], dim=1)

    def _v331_run_refiner(self, originals, proposals, scores, image_indices):
        device = next(self.local_refiner.parameters()).device
        grays = [self._v330_gray_original(image) for image in originals]
        image_shapes = [tuple(gray.shape[-2:]) for gray in grays]
        proposal_values = proposals.detach().cpu().tolist()
        patches = torch.stack([
            self._v330_make_patch(grays[image_index], proposal)
            for proposal, image_index in zip(
                proposal_values, image_indices)], dim=0).to(
                    device=device, non_blocking=True)
        proposal_shapes = [image_shapes[index] for index in image_indices]
        geometry = self._v330_geometry(proposals, scores, proposal_shapes)
        edge_delta, coarse_logit, refined_logit = self.local_refiner(
            patches, geometry)
        refined = self._v331_decode_edge_delta(
            proposals, edge_delta, proposal_shapes)
        return (edge_delta, coarse_logit, refined_logit, refined, patches)

    @staticmethod
    def _v331_select_predictions(det_boxes, best_iou, quota,
                                 min_positive_iou, negative_fraction):
        """Select real proposals with both matched and unmatched examples."""
        if quota <= 0 or det_boxes.numel() == 0:
            return det_boxes.new_zeros((0,), dtype=torch.long)
        order = det_boxes[:, 4].argsort(descending=True)
        positive = order[best_iou[order] >= min_positive_iou]
        negative = order[best_iou[order] < min_positive_iou]
        negative_quota = min(
            len(negative), int(round(quota * negative_fraction)))
        positive_quota = min(len(positive), quota - negative_quota)
        selected = [positive[:positive_quota], negative[:negative_quota]]
        used = positive_quota + negative_quota
        if used < quota:
            chosen = torch.cat(selected) if used else order[:0]
            remaining_mask = torch.ones(
                len(det_boxes), dtype=torch.bool, device=det_boxes.device)
            if chosen.numel():
                remaining_mask[chosen] = False
            remaining = order[remaining_mask[order]]
            selected.append(remaining[:quota - used])
        return torch.cat(selected) if selected else order[:0]

    def _v330_local_refine_train(
            self, original_img, img_metas, gt_bboxes, gt_labels,
            cls_scores, bbox_preds):
        if not self.use_local_refine:
            return {}
        originals = self._v330_unwrap_original_images(
            original_img, len(img_metas))
        prediction_fraction = self._v330_prediction_fraction()
        predicted_results = None
        if prediction_fraction > 0.0:
            with torch.no_grad():
                predicted_results = self.bbox_head.get_bboxes(
                    cls_scores, bbox_preds, img_metas,
                    cfg=self.bbox_head.test_cfg,
                    rescale=False, with_nms=True)

        all_proposals, all_targets, all_scores = [], [], []
        all_positive, all_image_indices = [], []
        predicted_count = jitter_count = negative_count = 0
        debug_step = int(getattr(self.bbox_head, '_debug_step', 0))
        for image_index, (meta, transformed_gt) in enumerate(
                zip(img_metas, gt_bboxes)):
            original_gt = self._v330_boxes_to_original(
                transformed_gt.detach(), meta, already_rescaled=False)
            if original_gt.numel() == 0:
                continue
            ori_h, ori_w = [int(v) for v in meta['ori_shape'][:2]]
            predicted_quota = int(round(
                self.local_refine_max_samples * prediction_fraction))
            image_proposals, image_targets = [], []
            image_scores, image_positive = [], []
            if predicted_results is not None and predicted_quota > 0:
                det_boxes, _ = predicted_results[image_index]
                det_boxes = det_boxes[:self.local_refine_topk]
                if det_boxes.numel() > 0:
                    proposals = self._v330_boxes_to_original(
                        det_boxes[:, :4], meta, already_rescaled=False)
                    ious = bbox_overlaps(proposals, original_gt)
                    best_iou, best_gt = ious.max(dim=1)
                    selected = self._v331_select_predictions(
                        det_boxes, best_iou, predicted_quota,
                        self.local_refine_min_train_iou,
                        self.local_refine_negative_fraction)
                    if selected.numel() > 0:
                        positive = (best_iou[selected]
                                    >= self.local_refine_min_train_iou)
                        image_proposals.append(proposals[selected].detach())
                        image_targets.append(original_gt[best_gt[selected]])
                        image_scores.append(det_boxes[selected, 4].detach())
                        image_positive.append(positive)
                        predicted_count += int(selected.numel())
                        negative_count += int((~positive).sum().item())

            used = sum(len(value) for value in image_proposals)
            jitter_quota = max(0, self.local_refine_max_samples - used)
            if jitter_quota:
                repeats = int(math.ceil(jitter_quota / len(original_gt)))
                jitter_batches = []
                for repeat_index in range(repeats):
                    # Keyed by iteration/image/repeat: resumable and isolated
                    # from DropPath, dropout, dataloader and global CUDA RNG.
                    key = (debug_step * 1000003 + image_index * 1009
                           + repeat_index * 9176 + 331)
                    jitter_batches.append(
                        self._v331_jitter_gt(original_gt, key))
                jitter = torch.cat(jitter_batches, dim=0)[:jitter_quota]
                jitter[:, 0::2].clamp_(0, ori_w)
                jitter[:, 1::2].clamp_(0, ori_h)
                target_indices = torch.arange(
                    jitter_quota, device=original_gt.device) % len(original_gt)
                image_proposals.append(jitter.detach())
                image_targets.append(original_gt[target_indices])
                # A neutral score is closer to detector proposals than the
                # old artificial confidence of 1.0 and prevents the geometry
                # MLP from using score as a GT-jitter shortcut.
                image_scores.append(jitter.new_full((jitter_quota,), 0.5))
                image_positive.append(torch.ones(
                    jitter_quota, dtype=torch.bool, device=jitter.device))
                jitter_count += jitter_quota

            if not image_proposals:
                continue
            proposals = torch.cat(image_proposals, dim=0)
            targets = torch.cat(image_targets, dim=0)
            scores = torch.cat(image_scores, dim=0)
            positives = torch.cat(image_positive, dim=0)
            all_proposals.append(proposals)
            all_targets.append(targets)
            all_scores.append(scores)
            all_positive.append(positives)
            all_image_indices.extend([image_index] * len(proposals))

        if not all_proposals:
            zero = cls_scores[0].sum() * 0.0
            return {
                'loss_local_delta': zero, 'loss_local_giou': zero,
                'loss_local_quality': zero, 'loss_local_gate': zero}
        proposals = torch.cat(all_proposals, dim=0)
        targets = torch.cat(all_targets, dim=0)
        scores = torch.cat(all_scores, dim=0)
        positive = torch.cat(all_positive, dim=0)
        target_delta = self._v331_encode_edge_delta(proposals, targets)
        (edge_delta, coarse_logit, refined_logit,
         refined, patches) = self._v331_run_refiner(
             originals, proposals, scores, all_image_indices)
        with torch.no_grad():
            coarse_iou = bbox_overlaps(
                proposals, targets, is_aligned=True)
            refined_iou = bbox_overlaps(
                refined.detach(), targets, is_aligned=True)
            # Unmatched proposals teach the gate to reject refinement; their
            # arbitrary edge output is not treated as a useful quality gain.
            refined_quality_target = torch.where(
                positive, refined_iou, coarse_iou)
            improve_target = (
                positive & (refined_iou - coarse_iou
                            > self.local_refine_improve_epsilon)).float()

        if positive.any():
            delta_loss = F.smooth_l1_loss(
                edge_delta[positive], target_delta[positive],
                reduction='mean', beta=1.0 / 9.0)
            aligned_giou = bbox_overlaps(
                refined[positive], targets[positive],
                mode='giou', is_aligned=True)
            giou_loss = (1.0 - aligned_giou).mean()
        else:
            delta_loss = edge_delta.sum() * 0.0
            giou_loss = edge_delta.sum() * 0.0
        quality_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(coarse_logit, coarse_iou)
            + F.binary_cross_entropy_with_logits(
                refined_logit, refined_quality_target))
        gain_logit = refined_logit - coarse_logit
        gate_loss = F.binary_cross_entropy_with_logits(
            gain_logit, improve_target)

        with torch.no_grad():
            predicted_gain = (refined_logit.sigmoid()
                              - coarse_logit.sigmoid())
            accepted = (
                (predicted_gain > self.local_refine_gate_threshold)
                & (refined_logit.sigmoid()
                   >= self.local_refine_min_quality))
            truly_improved = improve_target.bool()
            gate_tp = (accepted & truly_improved).sum().float()
            gate_precision = gate_tp / accepted.sum().clamp_min(1)
            gate_recall = gate_tp / truly_improved.sum().clamp_min(1)
            self._v330_debug = {
                'epoch': int(self._nam_epoch) + 1,
                'pred_fraction': float(prediction_fraction),
                'samples': int(len(proposals)),
                'predicted': int(predicted_count),
                'jittered': int(jitter_count),
                'negatives': int(negative_count),
                'coarse_iou': float(coarse_iou.mean().item()),
                'refined_iou': float(refined_iou.mean().item()),
                'positive_refined_iou': float(
                    refined_iou[positive].mean().item()
                    if positive.any() else 0.0),
                'improved_fraction': float(
                    (refined_iou[positive] > coarse_iou[positive]).float()
                    .mean().item() if positive.any() else 0.0),
                'quality_coarse_pred': float(
                    coarse_logit.sigmoid().mean().item()),
                'quality_coarse_target': float(coarse_iou.mean().item()),
                'quality_refined_pred': float(
                    refined_logit.sigmoid().mean().item()),
                'quality_refined_target': float(
                    refined_quality_target.mean().item()),
                'gate_accept_fraction': float(accepted.float().mean().item()),
                'gate_precision': float(gate_precision.item()),
                'gate_recall': float(gate_recall.item()),
                'delta_abs': float(edge_delta.abs().mean().item()),
                'patch_mean': float(patches[:, :1].mean().item()),
                'patch_std': float(patches[:, :1].std().item())}
        return {
            'loss_local_delta': self.local_refine_delta_weight * delta_loss,
            'loss_local_giou': self.local_refine_giou_weight * giou_loss,
            'loss_local_quality': (
                self.local_refine_quality_weight * quality_loss),
            'loss_local_gate': (
                self.local_refine_quality_weight * gate_loss)}

    def _v330_local_refine_test(
            self, results_list, original_img, img_metas,
            results_are_rescaled):
        if (not self.use_local_refine
                or self.local_refine_inference_mode == 'off'):
            return results_list
        originals = self._v330_unwrap_original_images(
            original_img, len(img_metas))
        output, accepted_total, candidate_total = [], 0, 0
        pre_nms_total = post_nms_total = 0
        coarse_quality_values, refined_quality_values = [], []
        from mmcv.ops import batched_nms
        for image_index, ((det_boxes, det_labels), meta) in enumerate(
                zip(results_list, img_metas)):
            if det_boxes.numel() == 0:
                output.append((det_boxes, det_labels))
                continue
            count = min(self.local_refine_topk, len(det_boxes))
            source_boxes = det_boxes[:count, :4]
            original_boxes = self._v330_boxes_to_original(
                source_boxes, meta,
                already_rescaled=results_are_rescaled)
            scores = det_boxes[:count, 4]
            indices = [image_index] * count
            (_, coarse_logit, refined_logit,
             refined, _) = self._v331_run_refiner(
                 originals, original_boxes, scores, indices)
            coarse_quality = coarse_logit.sigmoid()
            refined_quality = refined_logit.sigmoid()
            predicted_gain = refined_quality - coarse_quality
            if self.local_refine_inference_mode == 'all':
                accept = torch.ones_like(predicted_gain, dtype=torch.bool)
            else:
                accept = (
                    (predicted_gain > self.local_refine_gate_threshold)
                    & (refined_quality >= self.local_refine_min_quality))
            selected = torch.where(
                accept[:, None], refined, original_boxes)
            if not results_are_rescaled:
                selected = self._v330_boxes_from_original(selected, meta)
            updated = det_boxes.clone()
            updated[:count, :4] = selected
            if self.local_refine_quality_alpha > 0.0:
                selected_quality = torch.where(
                    accept, refined_quality, coarse_quality).clamp_min(0.05)
                updated[:count, 4] = (
                    updated[:count, 4]
                    * selected_quality.pow(self.local_refine_quality_alpha))
            # Refinement invalidates the old suppression relation.  Re-sort
            # and re-run NMS on all retained detections after coordinates and
            # optional localisation-aware scores have changed.
            if self.local_refine_post_nms and len(updated) > 1:
                pre_nms_total += len(updated)
                nms_cfg = dict(self.bbox_head.test_cfg.get(
                    'nms', dict(type='nms', iou_threshold=0.5)))
                if self.local_refine_post_nms_iou is not None:
                    if 'iou_threshold' in nms_cfg:
                        nms_cfg['iou_threshold'] = float(
                            self.local_refine_post_nms_iou)
                    else:
                        nms_cfg['iou_thr'] = float(
                            self.local_refine_post_nms_iou)
                dets, keep = batched_nms(
                    updated[:, :4], updated[:, 4], det_labels, nms_cfg)
                max_per_img = int(self.bbox_head.test_cfg.get(
                    'max_per_img', len(dets)))
                if max_per_img > 0:
                    dets, keep = dets[:max_per_img], keep[:max_per_img]
                post_nms_total += len(dets)
                output.append((dets, det_labels[keep]))
            else:
                order = updated[:, 4].argsort(descending=True)
                pre_nms_total += len(updated)
                post_nms_total += len(updated)
                output.append((updated[order], det_labels[order]))
            candidate_total += count
            accepted_total += int(accept.sum().item())
            coarse_quality_values.append(coarse_quality.detach())
            refined_quality_values.append(refined_quality.detach())
        if coarse_quality_values:
            coarse_quality = torch.cat(coarse_quality_values)
            refined_quality = torch.cat(refined_quality_values)
            self._v330_debug.update({
                'test_candidates': int(candidate_total),
                'test_accepted': int(accepted_total),
                'test_accept_fraction': float(
                    accepted_total / max(candidate_total, 1)),
                'test_coarse_quality': float(coarse_quality.mean().item()),
                'test_refined_quality': float(refined_quality.mean().item())})
            accumulator = self._v331_test_accumulator
            accumulator['images'] += len(output)
            accumulator['candidates'] += candidate_total
            accumulator['accepted'] += accepted_total
            accumulator['pre_nms'] += pre_nms_total
            accumulator['post_nms'] += post_nms_total
            accumulator['coarse_quality_sum'] += float(
                coarse_quality.sum().item())
            accumulator['refined_quality_sum'] += float(
                refined_quality.sum().item())
            interval = self.local_refine_test_debug_interval
            if interval > 0 and accumulator['images'] >= interval:
                candidates = max(accumulator['candidates'], 1)
                _emit_training_debug(
                    '[V331-TEST] mode={} images={} candidates={} '
                    'accept={:.3f} Qc/Qr={:.3f}/{:.3f} NMS={}/{}'.format(
                        self.local_refine_inference_mode,
                        accumulator['images'], accumulator['candidates'],
                        accumulator['accepted'] / candidates,
                        accumulator['coarse_quality_sum'] / candidates,
                        accumulator['refined_quality_sum'] / candidates,
                        accumulator['pre_nms'], accumulator['post_nms']))
                self._v331_test_accumulator = {
                    'images': 0, 'candidates': 0, 'accepted': 0,
                    'pre_nms': 0, 'post_nms': 0,
                    'coarse_quality_sum': 0.0,
                    'refined_quality_sum': 0.0}
        return output


@HOOKS.register_module(force=True)
class NAMV331DebugHook(NAMV321DebugHook):
    """V331 build fingerprint and gate-calibration diagnostics."""

    def before_run(self, runner):
        super().before_run(runner)
        model = (runner.model.module
                 if hasattr(runner.model, 'module') else runner.model)
        runner.logger.info(
            '[V331-BUILD] id=%s file=%s enabled=%s patch=%s context=%.2f '
            'bins=%s max_shift=%.2f max_samples=%s topk=%s min_iou=%.2f '
            'neg_frac=%.2f gate=%.3f min_q=%.2f q_alpha=%.2f mode=%s '
            'post_nms=%s schedule=%s/%s',
            NAMLAB_V331_BUILD_ID, __file__, model.use_local_refine,
            getattr(model.local_refiner, 'patch_size', None),
            model.local_refine_context,
            getattr(model.local_refiner, 'profile_bins', None),
            getattr(model.local_refiner, 'max_edge_shift', 0.0),
            model.local_refine_max_samples, model.local_refine_topk,
            model.local_refine_min_train_iou,
            model.local_refine_negative_fraction,
            model.local_refine_gate_threshold,
            model.local_refine_min_quality,
            model.local_refine_quality_alpha,
            model.local_refine_inference_mode,
            model.local_refine_post_nms,
            model.local_refine_pred_start_epoch,
            model.local_refine_pred_full_epoch)

    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        model = (runner.model.module
                 if hasattr(runner.model, 'module') else runner.model)
        if not model.use_local_refine:
            return
        head = model.bbox_head
        interval = int(getattr(head, 'debug_interval', 0))
        debug_step = int(getattr(head, '_debug_step', 0))
        if (interval <= 0 or debug_step % interval != 0
                or not _v320_rank_zero()):
            return
        stats = model._v330_debug
        rms, coverage = _TBFusionCrossScale._gradient_group(
            model.local_refiner.parameters())
        runner.logger.info(
            '[V331-LOCAL] iter=%d epoch=%s samples=%s pred/jit/neg=%s/%s/%s '
            'pred_frac=%.2f IoU=%.3f->%.3f pos_ref=%.3f improve=%.2f '
            'Qc=%.3f/%.3f Qr=%.3f/%.3f gate=%.2f P/R=%.2f/%.2f '
            'delta=%.3e patch=%.3f/%.3f grad=%.3e/%.2f',
            max(0, debug_step - 1), stats.get('epoch', -1),
            stats.get('samples', 0), stats.get('predicted', 0),
            stats.get('jittered', 0), stats.get('negatives', 0),
            stats.get('pred_fraction', 0.0),
            stats.get('coarse_iou', 0.0), stats.get('refined_iou', 0.0),
            stats.get('positive_refined_iou', 0.0),
            stats.get('improved_fraction', 0.0),
            stats.get('quality_coarse_pred', 0.0),
            stats.get('quality_coarse_target', 0.0),
            stats.get('quality_refined_pred', 0.0),
            stats.get('quality_refined_target', 0.0),
            stats.get('gate_accept_fraction', 0.0),
            stats.get('gate_precision', 0.0),
            stats.get('gate_recall', 0.0),
            stats.get('delta_abs', 0.0), stats.get('patch_mean', 0.0),
            stats.get('patch_std', 0.0), rms, coverage)


# =============================================================================
# V44: CrossScale geometry path + Sym semantic auxiliary path
# =============================================================================
# The V331 local refiner is deliberately not reused by the V44 config.  V44
# refines the *main head design* first: SymFormer remains useful for semantic
# discrimination, while the bbox tower starts from the unmodified CrossScale
# feature and can learn only a bounded Sym residual.  This avoids an external
# crop/gate/second-NMS decision from confounding the head ablation.


@HEADS.register_module(force=True)
class RetinaNAMGuideAttHeadV44(RetinaNAMGuideAttHeadV321):
    """Decouple Sym semantic features from CrossScale geometry features.

    Classification receives the full SymFormer output.  Regression can use
    the historical coupled Sym feature, strict CrossScale, or a bounded soft
    route.  Soft routes start from CrossScale ``C`` and use one of
    ``C + alpha * (S - C)``, ``C + alpha * A(S - C)`` or
    ``C + alpha * A(S)``.  The scale is exactly zero at initialization, so
    the geometry path has a clean, inspectable baseline and may use Sym
    information only when optimisation demonstrates that it helps
    localisation.

    ``C + alpha * A(S)`` is a deliberately distinct ablation, not an
    interpolation: it can re-inject semantic content and magnitude already
    shared with ``C``.  Its actual contribution must therefore be monitored,
    rather than interpreting alpha alone as the amount of new Sym evidence.

    ``nms_oracle_debug`` is a training-only diagnostic.  It never changes
    logits, boxes, losses, NMS, gradients or evaluation predictions.
    """

    def __init__(self, *args, sym_regression_source='crossscale',
                 sym_reg_residual_max=0.10,
                 sym_reg_residual_trainable=True,
                 sym_reg_fusion='direct_delta',
                 nms_oracle_debug=False, nms_debug_interval=250,
                 **kwargs):
        self.sym_regression_source = str(sym_regression_source).lower()
        self.sym_reg_residual_max = float(sym_reg_residual_max)
        self.sym_reg_residual_trainable = bool(sym_reg_residual_trainable)
        self.sym_reg_fusion = str(sym_reg_fusion).lower()
        self.nms_oracle_debug = bool(nms_oracle_debug)
        self.nms_debug_interval = int(nms_debug_interval)
        if self.sym_regression_source not in ('sym', 'crossscale'):
            raise ValueError(
                "sym_regression_source must be 'sym' or 'crossscale'")
        if self.sym_reg_fusion not in (
                'direct_delta', 'adapted_delta', 'adapted_sym'):
            raise ValueError(
                'sym_reg_fusion must be direct_delta, adapted_delta, or '
                'adapted_sym')
        if not 0.0 < self.sym_reg_residual_max <= 1.0:
            raise ValueError('sym_reg_residual_max must be in (0, 1]')
        if self.nms_debug_interval < 0:
            raise ValueError('nms_debug_interval must be non-negative')
        super().__init__(*args, **kwargs)
        # Zero is an identity initialisation for the regression route.  It is
        # intentionally created after all historical layers, so this module
        # consumes no RNG and cannot shift the legacy model initialisation.
        self.sym_reg_residual_raw = nn.Parameter(
            torch.zeros(self.anchor_generator.num_levels, dtype=torch.float32),
            requires_grad=(self.sym_regression_source == 'crossscale'
                           and self.sym_reg_residual_trainable))
        # Adapters are created only for the two task-adapted routes and only
        # after all historical layers.  Conv construction is RNG-isolated;
        # the explicit identity reset makes A_l initially transparent without
        # shifting any legacy core-layer initialization.
        self.sym_reg_adapters = nn.ModuleList()
        if (self.sym_regression_source == 'crossscale'
                and self.sym_reg_fusion in ('adapted_delta', 'adapted_sym')):
            with torch.random.fork_rng(devices=[]):
                self.sym_reg_adapters = nn.ModuleList([
                    nn.Conv2d(self.feat_channels, self.feat_channels, 1)
                    for _ in range(self.anchor_generator.num_levels)
                ])
            self._reset_sym_reg_adapters()
        self._v44_decouple_stats = [None] * self.anchor_generator.num_levels
        self._v44_nms_stats = {}

    def _reset_sym_reg_adapters(self):
        """Set every per-level 1x1 adapter to an exact identity map."""
        for adapter in self.sym_reg_adapters:
            nn.init.zeros_(adapter.weight)
            diagonal = min(adapter.out_channels, adapter.in_channels)
            indices = torch.arange(diagonal, device=adapter.weight.device)
            adapter.weight.data[indices, indices, 0, 0] = 1.0
            if adapter.bias is not None:
                nn.init.zeros_(adapter.bias)

    def init_weights(self):
        super().init_weights()
        # Parent init_cfg may visit every Conv2d.  Reassert identity after it.
        self._reset_sym_reg_adapters()

    def effective_sym_reg_scales(self):
        return self.sym_reg_residual_max * torch.tanh(
            self.sym_reg_residual_raw)

    def sym_guided_mirror_boxes(self, boxes, image_width):
        """Return the correspondence actually encoded by SymPositionalEncoding.

        Sym copies the positional code from one lung half to the other.  The
        ``left`` flag determines the receiving half, so only candidates on
        that guided half are valid mirror hard-negative candidates.
        """
        if not hasattr(self.positional_encoding, 'left'):
            raise RuntimeError(
                'pTB mirror mining requires SymPositionalEncoding.left')
        mirrored = boxes.clone()
        mirrored[:, 0] = float(image_width) - boxes[:, 2]
        mirrored[:, 2] = float(image_width) - boxes[:, 0]
        center_x = (boxes[:, 0] + boxes[:, 2]) * .5
        if bool(self.positional_encoding.left):
            guided_half = center_x < float(image_width) * .5
        else:
            guided_half = center_x >= float(image_width) * .5
        return mirrored, guided_half

    def forward_single(self, x, lbp_lab=None, lung_masks=None,
                       level_index=0, reg_detail_feat=None):
        bs, _, h, w = x.shape
        masks = x.new_zeros((bs, h, w)).to(torch.bool)
        pos_embed = self.positional_encoding(
            masks, x, lung_mask=lung_masks)
        sym_feat = self.transformer(
            tuple([x]), [masks], self.query_embedding.weight, [pos_embed])

        # Classification intentionally uses the semantic/symmetry route.
        cls_feat = sym_feat
        sym_delta = sym_feat - x
        if self.sym_regression_source == 'sym':
            # Coupled control: exact historical no-PC V321/V331 feature path.
            sym_scale = x.new_tensor(1.0)
            reg_feat = sym_feat
            adapter_input = sym_feat
            adapter_output = sym_feat
        else:
            # Decoupled path. This is not a detach: bbox gradients still
            # improve CrossScale/backbone; Sym receives regression gradients
            # only through this bounded residual when it is trainable.
            sym_scale = self.effective_sym_reg_scales()[level_index].to(
                dtype=x.dtype)
            if self.sym_reg_fusion == 'adapted_sym':
                adapter_input = sym_feat
            else:
                adapter_input = sym_delta
            if self.sym_reg_fusion.startswith('adapted_'):
                adapter_output = self.sym_reg_adapters[level_index](
                    adapter_input)
            else:
                adapter_output = adapter_input
            reg_feat = x + sym_scale * adapter_output

        gate_stats = {
            'gate_mean': 0.0, 'gate_active': 0.0, 'edge_mean': 0.0,
            'spatial_gate_mean': 0.0, 'edge_temp': 0.0,
            'mod_scale': 0.0, 'boundary_weight_entropy': 0.0,
            'boundary_weight_peak': 0.0, 'pc_feat_norm': 0.0,
            'pc_contribution_ratio': 0.0,
        }
        # Head PC was invalidated by the P/U ablations.  Do not silently route
        # it into this new geometry study.
        if self.use_head_pc_modulation and lbp_lab is not None:
            raise RuntimeError(
                'V44 geometry ablation requires head_pc_modulation=False')
        if self.use_reg_detail_bypass:
            reg_feat = self.reg_detail_injector(
                reg_feat, reg_detail_feat, level_index=level_index)

        with torch.no_grad():
            native = x.detach().float().flatten(1)
            delta = sym_delta.detach().float().flatten(1)
            adapter_input_flat = adapter_input.detach().float().flatten(1)
            adapter_output_flat = adapter_output.detach().float().flatten(1)
            reg_flat = reg_feat.detach().float().flatten(1)
            # Measure the actual change relative to CrossScale, not merely
            # alpha*A(input).  They are equal for the two soft routes, while
            # coupled control must correctly report |S-C|/|C| rather than
            # the misleading feature magnitude |S|/|C|.
            contribution = reg_flat - native
            self._v44_decouple_stats[level_index] = {
                'scale': float(sym_scale.detach().float().item()),
                'sym_delta_rel': float((
                    delta.norm(dim=1) / native.norm(dim=1).clamp_min(1e-8)
                ).mean().item()),
                'sym_cos': float(F.cosine_similarity(
                    native, sym_feat.detach().float().flatten(1),
                    dim=1, eps=1e-8).mean().item()),
                'adapter_in_rel': float((
                    adapter_input_flat.norm(dim=1)
                    / native.norm(dim=1).clamp_min(1e-8)).mean().item()),
                'adapter_out_rel': float((
                    adapter_output_flat.norm(dim=1)
                    / native.norm(dim=1).clamp_min(1e-8)).mean().item()),
                'actual_contribution': float((
                    contribution.norm(dim=1)
                    / native.norm(dim=1).clamp_min(1e-8)).mean().item()),
                'reg_vs_c_cos': float(F.cosine_similarity(
                    reg_flat, native, dim=1, eps=1e-8).mean().item()),
            }

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_base = self.retina_reg(reg_feat)
        distribution_logits = None
        if self.use_d_fine_style:
            distribution_logits = self.d_fine_reg(reg_feat)
            normalized_residual, entropy = _distribution_refinement_expectation(
                distribution_logits, self.num_anchors, self.d_fine_reg_max)
            refine_scale = self.effective_d_fine_scales()[level_index].to(
                dtype=bbox_base.dtype)
            bbox_pred = bbox_base + refine_scale * normalized_residual
            self._dbg_d_fine_entropy[level_index] = float(
                entropy.detach().float().item())
            self._dbg_d_fine_expected_abs[level_index] = float(
                normalized_residual.detach().float().abs().mean().item())
        else:
            bbox_pred = bbox_base
        return (
            cls_score, bbox_pred, sym_feat, sym_feat, reg_feat, gate_stats,
            bbox_base, distribution_logits)

    @staticmethod
    def _v44_coverage(boxes, gt, threshold):
        if gt.numel() == 0:
            return 0
        if boxes.numel() == 0:
            return 0
        return int((bbox_overlaps(boxes[:, :4], gt).max(dim=0)[0]
                    >= threshold).sum().item())

    @torch.no_grad()
    def _v44_collect_nms_oracle(self, cls_scores, bbox_preds,
                                gt_bboxes, img_metas):
        """Measure candidate, production-NMS and oracle-NMS coverage.

        The oracle uses true best-IoU as a *temporary score* over exactly the
        decoded pre-NMS candidates.  It cannot add candidates or alter boxes;
        its gap from production therefore quantifies ranking/NMS headroom.
        """
        raw_results = self.get_bboxes(
            cls_scores, bbox_preds, img_metas, cfg=self.test_cfg,
            rescale=False, with_nms=False)
        production_results = self.get_bboxes(
            cls_scores, bbox_preds, img_metas, cfg=self.test_cfg,
            rescale=False, with_nms=True)
        from mmcv.ops import batched_nms

        summary = dict(images=0, gt=0, pre=0, production=0, oracle=0,
                       fp50=0, candidates=0, prod_dets=0)
        thresholds = (0.50, 0.75)
        for (raw_boxes, raw_scores), (prod_boxes, _), gt in zip(
                raw_results, production_results, gt_bboxes):
            gt = gt.detach()
            if gt.numel() == 0:
                continue
            fg_scores = raw_scores[:, :self.num_classes]
            scores, labels = fg_scores.max(dim=1)
            keep_score = scores >= float(self.test_cfg.get('score_thr', 0.0))
            boxes = raw_boxes[keep_score]
            scores = scores[keep_score]
            labels = labels[keep_score]
            summary['images'] += 1
            summary['gt'] += int(gt.shape[0])
            summary['candidates'] += int(boxes.shape[0])
            summary['prod_dets'] += int(prod_boxes.shape[0])

            best_iou = (bbox_overlaps(boxes, gt).max(dim=1)[0]
                        if boxes.numel() else boxes.new_zeros((0,)))
            if prod_boxes.numel():
                summary['fp50'] += int((bbox_overlaps(
                    prod_boxes[:, :4], gt).max(dim=1)[0] < 0.50
                ).sum().item())

            oracle_boxes = boxes.new_zeros((0, 4))
            if boxes.numel():
                # Same per-class NMS geometry as production, but perfect
                # ranking.  A tiny score floor avoids a zero-IoU candidate
                # becoming a meaningless NMS winner.
                oracle_scores = best_iou.clamp_min(1e-6)
                dets, keep = batched_nms(
                    boxes, oracle_scores, labels,
                    dict(self.test_cfg.get(
                        'nms', dict(type='nms', iou_threshold=0.5))))
                max_per_img = int(self.test_cfg.get('max_per_img', len(dets)))
                if max_per_img > 0:
                    keep = keep[:max_per_img]
                oracle_boxes = boxes[keep]
            for threshold in thresholds:
                key = str(int(threshold * 100))
                summary.setdefault('pre' + key, 0)
                summary.setdefault('production' + key, 0)
                summary.setdefault('oracle' + key, 0)
                summary['pre' + key] += self._v44_coverage(
                    boxes, gt, threshold)
                summary['production' + key] += self._v44_coverage(
                    prod_boxes, gt, threshold)
                summary['oracle' + key] += self._v44_coverage(
                    oracle_boxes, gt, threshold)
        self._v44_nms_stats = summary

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
             img_metas, gt_bboxes_ignore=None):
        losses = super().loss(
            cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        debug_step = int(getattr(self, '_debug_step', 0))
        if (self.training and self.nms_oracle_debug
                and self.nms_debug_interval > 0
                and debug_step % self.nms_debug_interval == 0):
            self._v44_collect_nms_oracle(
                cls_scores, bbox_preds, gt_bboxes, img_metas)
            stats = self._v44_nms_stats
            gt_count = max(stats.get('gt', 0), 1)
            logging.getLogger('mmdet').info(
                '[V44-NMS] iter=%d img=%d cand=%d prod=%d fp50=%d '
                'R50 pre/prod/oracle=%.3f/%.3f/%.3f '
                'R75 pre/prod/oracle=%.3f/%.3f/%.3f',
                max(0, debug_step - 1), stats.get('images', 0),
                stats.get('candidates', 0), stats.get('prod_dets', 0),
                stats.get('fp50', 0),
                stats.get('pre50', 0) / gt_count,
                stats.get('production50', 0) / gt_count,
                stats.get('oracle50', 0) / gt_count,
                stats.get('pre75', 0) / gt_count,
                stats.get('production75', 0) / gt_count,
                stats.get('oracle75', 0) / gt_count)
        if (self.training and self.debug_interval > 0
                and debug_step % self.debug_interval == 0):
            values = [item or {} for item in self._v44_decouple_stats]
            logging.getLogger('mmdet').info(
                '[V44-DECOUPLE] iter=%d fusion=%s sym_reg_scale=%s '
                'sym_delta_rel=%s sym_cos=%s adapter_in_rel=%s '
                'adapter_out_rel=%s actual_contribution=%s reg_vs_c_cos=%s',
                max(0, debug_step - 1),
                self.sym_reg_fusion,
                ['%.3e' % item.get('scale', 0.0) for item in values],
                ['%.3f' % item.get('sym_delta_rel', 0.0) for item in values],
                ['%.3f' % item.get('sym_cos', 0.0) for item in values],
                ['%.3f' % item.get('adapter_in_rel', 0.0) for item in values],
                ['%.3f' % item.get('adapter_out_rel', 0.0) for item in values],
                ['%.3e' % item.get('actual_contribution', 0.0)
                 for item in values],
                ['%.3f' % item.get('reg_vs_c_cos', 0.0)
                 for item in values])
        return losses


@DETECTORS.register_module(force=True)
class RetinaNAMAttV44(RetinaNAMAttV331):
    """V44 detector: V331 data/geometry contract with V44 decoupled head."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('experiment_tag', 'V44_SYM_AUX_CROSSSCALE_GEOMETRY')
        super().__init__(*args, **kwargs)
        if not isinstance(self.bbox_head, RetinaNAMGuideAttHeadV44):
            raise TypeError('RetinaNAMAttV44 requires RetinaNAMGuideAttHeadV44')
        self.integrated_build_id = 'v3.44-sym-aux-crossscale-geometry-20260901'


@HOOKS.register_module(force=True)
class NAMV44DebugHook(NAMV331DebugHook):
    """Build fingerprint for the Sym/geometry-decoupled V44 study."""

    def before_run(self, runner):
        super().before_run(runner)
        model = (runner.model.module
                 if hasattr(runner.model, 'module') else runner.model)
        head = model.bbox_head
        runner.logger.info(
            '[V44-BUILD] cls=sym reg=%s fusion=%s sym_reg_max=%.3f trainable=%s '
            'nms_oracle=%s nms_interval=%d local_refine=%s',
            head.sym_regression_source, head.sym_reg_fusion,
            head.sym_reg_residual_max,
            head.sym_reg_residual_trainable, head.nms_oracle_debug,
            head.nms_debug_interval, model.use_local_refine)


# =============================================================================
# V45: CrossScale-conditioned short-range side refinement
# =============================================================================


class _V45CrossScaleSideRefiner(nn.Module):
    """Sample P2/P3 around a decoded coarse box and predict four side shifts."""

    def __init__(self, channels=256, hidden=128, mode='direct', bins=9,
                 max_ratio=0.25, feature_sampling='boundary',
                 use_raw_detail=False, raw_channels=32, raw_beta_max=0.10):
        super().__init__()
        self.mode = str(mode)
        self.bins = int(bins)
        self.max_ratio = float(max_ratio)
        self.feature_sampling = str(feature_sampling)
        self.use_raw_detail = bool(use_raw_detail)
        self.raw_beta_max = float(raw_beta_max)
        if self.mode not in ('direct', 'bucket', 'gated_direct'):
            raise ValueError('side refiner mode is invalid')
        if self.feature_sampling not in ('boundary', 'star'):
            raise ValueError("feature_sampling must be 'boundary' or 'star'")
        if self.bins < 3 or self.bins % 2 == 0:
            raise ValueError('side refiner bins must be odd and >= 3')
        self.p2_proj = nn.Conv2d(channels, hidden // 2, 1, bias=False)
        self.p3_proj = nn.Conv2d(channels, hidden // 2, 1, bias=False)
        if self.use_raw_detail:
            # Stride-one local detail only; this is not a second backbone.
            # Isolate construction RNG so enabling a zero-weight raw branch
            # cannot change the pre-existing H2 side-MLP initialization.
            with torch.random.fork_rng(devices=[], enabled=True):
                self.raw_encoder = nn.Sequential(
                    nn.Conv2d(1, raw_channels, 3, padding=1, bias=False),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(raw_channels, raw_channels, 3, padding=1,
                              bias=False), nn.SiLU(inplace=True),
                    nn.Conv2d(raw_channels, hidden, 1, bias=False))
            self.raw_beta = nn.Parameter(torch.zeros(4))
        else:
            self.raw_encoder = None
            self.register_parameter('raw_beta', None)
        self._raw_debug = {}
        in_dim = hidden * 4  # center + edge + inside + outside, P2/P3 fused
        out_dim = 1 if self.mode == 'direct' else (2 if self.mode == 'gated_direct' else self.bins)
        self.side_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
                          nn.Linear(hidden, out_dim)) for _ in range(4)])
        for module in (self.p2_proj, self.p3_proj):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
        for mlp in self.side_mlps:
            nn.init.kaiming_normal_(mlp[0].weight, mode='fan_in',
                                    nonlinearity='relu')
            nn.init.zeros_(mlp[0].bias)
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        if self.raw_encoder is not None:
            for module in self.raw_encoder:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                            nonlinearity='relu')

    def _points(self, boxes, image_h, image_w):
        """center plus edge/inside/outside samples for L/R/T/B."""
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        w = (x2 - x1).clamp_min(2.0)
        h = (y2 - y1).clamp_min(2.0)
        cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
        band_x, band_y = .08 * w, .08 * h
        if self.feature_sampling == 'star':
            # VFNet-style center, side midpoints and four corners.
            points = torch.stack([
                torch.stack([cx, cy], dim=1),
                torch.stack([x1, cy], dim=1), torch.stack([x2, cy], dim=1),
                torch.stack([cx, y1], dim=1), torch.stack([cx, y2], dim=1),
                torch.stack([x1, y1], dim=1), torch.stack([x2, y1], dim=1),
                torch.stack([x1, y2], dim=1), torch.stack([x2, y2], dim=1),
            ], dim=1)
        else:
            # center, then [edge, inside, outside] per L/R/T/B.
            points = torch.stack([
            torch.stack([cx, cy], dim=1),
            torch.stack([x1, cy], dim=1), torch.stack([x1 + band_x, cy], dim=1), torch.stack([x1 - band_x, cy], dim=1),
            torch.stack([x2, cy], dim=1), torch.stack([x2 - band_x, cy], dim=1), torch.stack([x2 + band_x, cy], dim=1),
            torch.stack([cx, y1], dim=1), torch.stack([cx, y1 + band_y], dim=1), torch.stack([cx, y1 - band_y], dim=1),
            torch.stack([cx, y2], dim=1), torch.stack([cx, y2 - band_y], dim=1), torch.stack([cx, y2 + band_y], dim=1),
            ], dim=1)
        points[..., 0].clamp_(0, max(float(image_w) - 1.0, 0.0))
        points[..., 1].clamp_(0, max(float(image_h) - 1.0, 0.0))
        return points

    @staticmethod
    def _sample(feature, points, image_h, image_w):
        # align_corners=False mapping for image-coordinate points.
        grid = points.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / max(float(image_w), 1.0) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(float(image_h), 1.0) - 1.0
        sampled = F.grid_sample(feature, grid.unsqueeze(0), mode='bilinear',
                                padding_mode='border', align_corners=False)
        return sampled[0].permute(1, 2, 0).contiguous()  # N, points, C

    def forward(self, p2, p3, boxes, image_h, image_w, detail_img=None):
        if boxes.numel() == 0:
            out = boxes.new_zeros((0, 4))
            return out, None
        points = self._points(boxes, image_h, image_w)
        p2_values = self._sample(self.p2_proj(p2), points, image_h, image_w)
        p3_values = self._sample(self.p3_proj(p3), points, image_h, image_w)
        values = torch.cat([p2_values, p3_values], dim=-1)
        groups = (((0, 1, 2, 3), (0, 4, 5, 6),
                   (0, 7, 8, 9), (0, 10, 11, 12))
                  if self.feature_sampling == 'boundary' else
                  ((0, 1, 5, 7), (0, 3, 5, 6),
                   (0, 2, 6, 8), (0, 4, 7, 8)))
        raw_values = None
        if self.raw_encoder is not None:
            if detail_img is None:
                raise RuntimeError('side_refine_raw_detail=True requires detail_img')
            if detail_img.ndim != 4:
                raise ValueError('detail_img must be BCHW')
            # ``PreserveAlignedImage`` intentionally keeps uint8 intensities.
            # PyTorch 1.10 cannot reduce a Byte tensor with mean(), so cast
            # before channel reduction rather than after it.
            raw_gray = detail_img.float().mean(dim=1, keepdim=True) / 255.0
            raw_values = self._sample(self.raw_encoder(raw_gray), points,
                                      image_h, image_w).to(values.dtype)
            beta = self.raw_beta_max * torch.tanh(self.raw_beta)
            self._raw_debug = {
                'beta': [float(v) for v in beta.detach().cpu()],
                'feature_abs': float(raw_values.detach().abs().mean()),
                'base_abs': float(values.detach().abs().mean()),
                'relative_contribution': float(
                    beta.abs().mean() * raw_values.detach().abs().mean()
                    / values.detach().abs().mean().clamp_min(1e-12)),
            }
        logits = torch.stack([
            self.side_mlps[i](
                (values[:, list(group)] if raw_values is None else
                 values[:, list(group)] + beta[i] * raw_values[:, list(group)])
                .flatten(1))
            for i, group in enumerate(groups)], dim=1)
        if self.mode == 'direct':
            return self.max_ratio * torch.tanh(logits.squeeze(-1)), None
        if self.mode == 'gated_direct':
            return (self.max_ratio * torch.tanh(logits[..., 0]),
                    logits[..., 1])
        support = torch.linspace(-self.max_ratio, self.max_ratio, self.bins,
                                 device=boxes.device, dtype=boxes.dtype)
        probs = logits.softmax(dim=-1)
        return (probs * support).sum(dim=-1), logits


@DETECTORS.register_module(force=True)
class RetinaNAMAttV45(RetinaNAMAttV44):
    """V44 plus a pre-NMS, feature-space four-side residual refiner."""

    def __init__(self, *args, side_refine=False, side_refine_mode='direct',
                 side_refine_bins=9, side_refine_max_ratio=.25,
                 side_refine_topk=64, side_refine_min_iou=.30,
                 side_refine_start_epoch=4, side_refine_loss_weight=.5,
                 side_refine_debug_interval=100,
                 side_refine_huber_beta=1. / 9.,
                 side_refine_expectation_huber_weight=0.0,
                 side_refine_gate_weight=.15,
                 side_refine_gate_threshold=.5,
                 side_refine_feature_sampling='boundary',
                 side_refine_raw_detail=False,
                 side_refine_raw_channels=32,
                 side_refine_raw_beta_max=.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_side_refine = bool(side_refine)
        self.side_refine_topk = int(side_refine_topk)
        self.side_refine_min_iou = float(side_refine_min_iou)
        self.side_refine_start_epoch = int(side_refine_start_epoch)
        self.side_refine_loss_weight = float(side_refine_loss_weight)
        self.side_refine_debug_interval = int(side_refine_debug_interval)
        self.side_refine_huber_beta = float(side_refine_huber_beta)
        self.side_refine_expectation_huber_weight = float(side_refine_expectation_huber_weight)
        self.side_refine_gate_weight = float(side_refine_gate_weight)
        self.side_refine_gate_threshold = float(side_refine_gate_threshold)
        self.side_refine_raw_detail = bool(side_refine_raw_detail)
        if self.side_refine_topk < 1 or not 0 <= self.side_refine_min_iou < 1:
            raise ValueError('invalid V45 side-refine candidate settings')
        if self.use_side_refine:
            with torch.random.fork_rng(devices=[], enabled=True):
                self.side_refiner = _V45CrossScaleSideRefiner(
                    mode=side_refine_mode, bins=side_refine_bins,
                    max_ratio=side_refine_max_ratio,
                    feature_sampling=side_refine_feature_sampling,
                    use_raw_detail=self.side_refine_raw_detail,
                    raw_channels=side_refine_raw_channels,
                    raw_beta_max=side_refine_raw_beta_max)
        else:
            self.side_refiner = None
        self._v45_debug = {}
        self.integrated_build_id = 'v3.45-crossscale-side-refine-20260901'

    @staticmethod
    def _v45_decode(boxes, delta, image_h, image_w):
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(2.0)
        scale = torch.stack([wh[:, 0], wh[:, 1], wh[:, 0], wh[:, 1]], dim=1)
        refined = boxes + delta * scale
        left = refined[:, 0].clamp(0, image_w - 1)
        top = refined[:, 1].clamp(0, image_h - 1)
        right = torch.maximum(refined[:, 2], left + 1).clamp(max=image_w)
        bottom = torch.maximum(refined[:, 3], top + 1).clamp(max=image_h)
        return torch.stack([left, top, right, bottom], dim=1)

    @staticmethod
    def _v45_target(boxes, targets):
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(2.0)
        scale = torch.stack([wh[:, 0], wh[:, 1], wh[:, 0], wh[:, 1]], dim=1)
        return (targets - boxes) / scale

    def _v45_select(self, boxes, scores):
        count = min(self.side_refine_topk, len(boxes))
        return scores.max(dim=1)[0].topk(count).indices if count else scores.new_zeros((0,), dtype=torch.long)

    @staticmethod
    def _v45_detail_tensor(detail_img):
        """Unwrap the single-test augmentation container for detail_img."""
        while isinstance(detail_img, (list, tuple)):
            if len(detail_img) != 1:
                raise ValueError('raw detail supports one test augmentation')
            detail_img = detail_img[0]
        return detail_img

    def _v45_run(self, feats, image_index, boxes, meta, detail_img=None):
        image_h, image_w = [int(v) for v in meta['img_shape'][:2]]
        delta, logits = self.side_refiner(feats[0][image_index:image_index + 1],
                                          feats[1][image_index:image_index + 1],
                                          boxes.detach(), image_h, image_w,
                                          detail_img=(detail_img[image_index:image_index + 1]
                                                      if detail_img is not None else None))
        return delta, logits, self._v45_decode(boxes, delta, image_h, image_w)

    def _v45_side_refine_train(self, feats, cls_scores, bbox_preds, img_metas,
                               gt_bboxes, detail_img=None):
        if not self.use_side_refine or int(self._nam_epoch) + 1 < self.side_refine_start_epoch:
            return {}
        detail_img = self._v45_detail_tensor(detail_img)
        raw = self.bbox_head.get_bboxes(cls_scores, bbox_preds, img_metas,
                                        cfg=self.bbox_head.test_cfg, rescale=False,
                                        with_nms=False)
        losses, coarse_iou, refined_iou, deltas = [], [], [], []
        gate_probs, gate_targets = [], []
        up50 = down50 = up75 = down75 = 0
        for i, ((boxes, scores), gt, meta) in enumerate(zip(raw, gt_bboxes, img_metas)):
            if not len(boxes) or not len(gt):
                continue
            index = self._v45_select(boxes, scores[:, :self.bbox_head.num_classes])
            boxes, scores = boxes[index], scores[index, :self.bbox_head.num_classes]
            overlap = bbox_overlaps(boxes, gt)
            best_iou, best = overlap.max(dim=1)
            valid = best_iou >= self.side_refine_min_iou
            if not valid.any():
                continue
            boxes, target = boxes[valid], gt[best[valid]]
            delta, logits, refined = self._v45_run(
                feats, i, boxes, meta, detail_img=detail_img)
            target_delta = self._v45_target(boxes, target).clamp(
                -self.side_refiner.max_ratio, self.side_refiner.max_ratio)
            if logits is None or self.side_refiner.mode == 'gated_direct':
                side_loss = F.smooth_l1_loss(
                    delta, target_delta, beta=self.side_refine_huber_beta)
                if self.side_refiner.mode == 'gated_direct':
                    edge_before = (boxes - target).abs() / torch.stack([
                        (boxes[:, 2] - boxes[:, 0]).clamp_min(2.),
                        (boxes[:, 3] - boxes[:, 1]).clamp_min(2.),
                        (boxes[:, 2] - boxes[:, 0]).clamp_min(2.),
                        (boxes[:, 3] - boxes[:, 1]).clamp_min(2.)], dim=1)
                    edge_after = (refined.detach() - target).abs() / torch.stack([
                        (boxes[:, 2] - boxes[:, 0]).clamp_min(2.),
                        (boxes[:, 3] - boxes[:, 1]).clamp_min(2.),
                        (boxes[:, 2] - boxes[:, 0]).clamp_min(2.),
                        (boxes[:, 3] - boxes[:, 1]).clamp_min(2.)], dim=1)
                    gate_target = (edge_after + .005 < edge_before).float()
                    side_loss = side_loss + self.side_refine_gate_weight * F.binary_cross_entropy_with_logits(logits, gate_target)
                    gate_probs.append(logits.detach().sigmoid())
                    gate_targets.append(gate_target.detach())
            else:
                support = torch.linspace(-self.side_refiner.max_ratio, self.side_refiner.max_ratio,
                                         self.side_refiner.bins, device=delta.device, dtype=delta.dtype)
                pos = ((target_delta - support[0]) / (support[1] - support[0])).clamp(0, self.side_refiner.bins - 1)
                lo, hi = pos.floor().long(), pos.ceil().long()
                weight_hi = pos - lo.float()
                ce_lo = F.cross_entropy(logits.reshape(-1, self.side_refiner.bins), lo.reshape(-1), reduction='none')
                ce_hi = F.cross_entropy(logits.reshape(-1, self.side_refiner.bins), hi.reshape(-1), reduction='none')
                side_loss = ((1 - weight_hi.reshape(-1)) * ce_lo + weight_hi.reshape(-1) * ce_hi).mean()
                if self.side_refine_expectation_huber_weight > 0:
                    side_loss = side_loss + self.side_refine_expectation_huber_weight * F.smooth_l1_loss(
                        delta, target_delta, beta=self.side_refine_huber_beta)
            giou = 1 - bbox_overlaps(refined, target, mode='giou', is_aligned=True).mean()
            losses.append(side_loss + giou)
            ciou = bbox_overlaps(boxes, target, is_aligned=True)
            riou = bbox_overlaps(refined.detach(), target, is_aligned=True)
            coarse_iou.append(ciou); refined_iou.append(riou); deltas.append(delta.detach())
            up50 += int(((ciou < .5) & (riou >= .5)).sum()); down50 += int(((ciou >= .5) & (riou < .5)).sum())
            up75 += int(((ciou < .75) & (riou >= .75)).sum()); down75 += int(((ciou >= .75) & (riou < .75)).sum())
        if not losses:
            return {}
        c, r, d = torch.cat(coarse_iou), torch.cat(refined_iou), torch.cat(deltas)
        self._v45_debug = dict(samples=len(c), coarse=float(c.mean()), refined=float(r.mean()),
                                improve=float((r > c).float().mean()), up50=up50, down50=down50,
                                up75=up75, down75=down75, side_abs=[float(v) for v in d.abs().mean(0)])
        if self.side_refiner._raw_debug:
            self._v45_debug['raw'] = self.side_refiner._raw_debug.copy()
        if gate_probs:
            probs, target = torch.cat(gate_probs), torch.cat(gate_targets)
            chosen = probs >= self.side_refine_gate_threshold
            self._v45_debug.update(gate_open=float(chosen.float().mean()),
                                   gate_target=float(target.mean()),
                                   gate_precision=float((target[chosen].mean() if chosen.any() else target.new_zeros(())).item()))
        step = int(getattr(self.bbox_head, '_debug_step', 0))
        if self.side_refine_debug_interval and step % self.side_refine_debug_interval == 0:
            _emit_training_debug('[V45-SIDE] iter={} n={} IoU={:.3f}->{:.3f} imp={:.3f} up/down50={}/{} up/down75={}/{} side={} gate=open:{:.3f}/target:{:.3f}/prec:{:.3f}'.format(
                max(0, step - 1), len(c), self._v45_debug['coarse'], self._v45_debug['refined'],
                self._v45_debug['improve'], up50, down50, up75, down75,
                '/'.join('{:.3f}'.format(v) for v in self._v45_debug['side_abs']),
                self._v45_debug.get('gate_open', 0.), self._v45_debug.get('gate_target', 0.), self._v45_debug.get('gate_precision', 0.)))
            if 'raw' in self._v45_debug:
                raw = self._v45_debug['raw']
                _emit_training_debug(
                    '[V45-RAW] beta={} feature_abs={:.4f} base_abs={:.4f} rel={:.5f}'.format(
                        '/'.join('{:.4f}'.format(v) for v in raw['beta']),
                        raw['feature_abs'], raw['base_abs'],
                        raw['relative_contribution']))
        return {'loss_side_refine': self.side_refine_loss_weight * sum(losses) / len(losses)}

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Refine decoded pre-NMS candidates, then execute the sole final NMS."""
        if not self.use_side_refine:
            return super().simple_test(img, img_metas, rescale=rescale, **kwargs)
        detail_img = self._v45_detail_tensor(kwargs.get('detail_img', None))
        batch_shape = tuple(int(v) for v in img.shape[-2:])
        for meta in img_metas:
            meta['batch_input_shape'] = batch_shape
        self._collect_roi_debug(img_metas)
        backbone_feats = self.backbone(img)
        x, c2_feat = self._build_pyramid(backbone_feats, lbp_lab=None)
        lung_masks = self._collect_lung_masks(img_metas, device=img.device,
                                              batch_input_shape=batch_shape)
        outs = self.bbox_head.forward(
            x, nam_graphs=None, lbp_lab=None, c2_feat=c2_feat,
            region_feat=None, img_shapes=[batch_shape for _ in img_metas],
            lung_masks=lung_masks,
            reg_detail_feats=getattr(self.neck, 'regression_detail_feats', None))
        raw = self.bbox_head.get_bboxes(outs[0], outs[1], img_metas,
                                        cfg=self.bbox_head.test_cfg,
                                        rescale=False, with_nms=False)
        from mmdet.core import bbox2result, multiclass_nms
        results = []
        for i, ((boxes, scores), meta) in enumerate(zip(raw, img_metas)):
            fg_scores = scores[:, :self.bbox_head.num_classes]
            index = self._v45_select(boxes, fg_scores)
            updated = boxes.clone()
            if index.numel():
                delta, aux, refined = self._v45_run(
                    x, i, boxes[index], meta,
                    detail_img=detail_img)
                if self.side_refiner.mode == 'gated_direct':
                    gate = (aux.sigmoid() >= self.side_refine_gate_threshold).to(delta.dtype)
                    h, w = [int(v) for v in meta['img_shape'][:2]]
                    refined = self._v45_decode(boxes[index], delta * gate, h, w)
                updated[index] = refined
            det, labels = multiclass_nms(updated, scores,
                                         self.bbox_head.test_cfg.score_thr,
                                         self.bbox_head.test_cfg.nms,
                                         self.bbox_head.test_cfg.max_per_img)
            if rescale and det.numel():
                det[:, :4] /= det.new_tensor(meta['scale_factor'])
            results.append((det, labels))
        return [bbox2result(det, label, self.bbox_head.num_classes)
                for det, label in results]


@HOOKS.register_module(force=True)
class NAMV45DebugHook(NAMV44DebugHook):
    """Records the exact coarse-to-side-refinement contract at startup."""

    def before_run(self, runner):
        super().before_run(runner)
        model = (runner.model.module if hasattr(runner.model, 'module')
                 else runner.model)
        refiner = model.side_refiner
        runner.logger.info(
            '[V45-BUILD] enabled=%s mode=%s bins=%s rmax=%.3f topk=%d '
            'min_iou=%.2f start_epoch=%d loss_w=%.2f raw_detail=%s final_nms=one',
            model.use_side_refine, getattr(refiner, 'mode', 'off'),
            getattr(refiner, 'bins', 0), getattr(refiner, 'max_ratio', 0.0),
            model.side_refine_topk, model.side_refine_min_iou,
            model.side_refine_start_epoch, model.side_refine_loss_weight,
            model.side_refine_raw_detail)


class _V47CandidateScoreHead(nn.Module):
    """Candidate-aligned TB existence, localisation-quality and box-gate head.

    It deliberately consumes CrossScale P2/P3 rather than the Sym subtype
    logits.  The three outputs have non-overlapping semantics:
    ``p_tb`` answers whether a candidate is a TB lesion, ``q_ref`` estimates
    final decoded-box IoU, and ``g_box`` decides whether the H2 residual is
    worth applying as one geometrically coherent box update.
    """

    def __init__(self, channels=256, hidden=128):
        super().__init__()
        self.p2_proj = nn.Conv2d(channels, hidden // 2, 1, bias=False)
        self.p3_proj = nn.Conv2d(channels, hidden // 2, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 5, hidden), nn.SiLU(inplace=True))
        self.out = nn.Linear(hidden, 3)  # TB-objectness, IoU quality, box gate
        for layer in (self.p2_proj, self.p3_proj):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                    nonlinearity='relu')
        nn.init.kaiming_normal_(self.mlp[0].weight, mode='fan_in',
                                nonlinearity='relu')
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.out.weight)
        # Initial p(TB) is deliberately low, while quality remains neutral.
        # This avoids a new untrained branch flooding NMS at epoch one.
        with torch.no_grad():
            self.out.bias.copy_(self.out.bias.new_tensor([-4.0, 0.0, -1.0]))

    @staticmethod
    def _points(boxes, image_h, image_w):
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
        points = torch.stack([
            torch.stack([cx, cy], dim=1),
            torch.stack([x1, cy], dim=1), torch.stack([x2, cy], dim=1),
            torch.stack([cx, y1], dim=1), torch.stack([cx, y2], dim=1),
        ], dim=1)
        points[..., 0].clamp_(0, max(float(image_w) - 1., 0.))
        points[..., 1].clamp_(0, max(float(image_h) - 1., 0.))
        return points

    def forward(self, p2, p3, boxes, image_h, image_w):
        if not boxes.numel():
            return boxes.new_zeros((0, 3))
        points = self._points(boxes, image_h, image_w)
        p2_value = _V45CrossScaleSideRefiner._sample(
            self.p2_proj(p2), points, image_h, image_w)
        p3_value = _V45CrossScaleSideRefiner._sample(
            self.p3_proj(p3), points, image_h, image_w)
        # Preserve the ordered identity of centre/L/R/T/B.  Mean pooling these
        # five locations made geometrically different boundary errors
        # indistinguishable to the scalar IoU-quality predictor.
        feature = torch.cat([p2_value, p3_value], dim=-1).flatten(1)
        return self.out(self.mlp(feature))


@DETECTORS.register_module(force=True)
class RetinaNAMAttV47(RetinaNAMAttV45):
    """V45 with training/inference candidate-contract gate supervision."""

    def __init__(self, *args, side_refine_unified_candidates=False,
                 side_refine_box_iou_gate=False,
                 side_refine_detach_features=False,
                 side_refine_gate_pos_weight=1.5,
                 side_refine_box_gate=False,
                 side_refine_box_gate_weight=.15,
                 side_refine_box_gate_threshold=.5,
                 side_refine_box_gate_margin=.005,
                 nms_score_mode='cls', nms_quality_beta=.5,
                 nms_objectness_loss_weight=.25,
                 nms_quality_loss_weight=.25,
                 nms_score_pos_iou=.5,
                 nms_score_train_topk=128,
                 nms_score_detach_features=False,
                 nms_residual_scale=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.side_refine_unified_candidates = bool(side_refine_unified_candidates)
        self.side_refine_box_iou_gate = bool(side_refine_box_iou_gate)
        self.side_refine_detach_features = bool(side_refine_detach_features)
        self.side_refine_gate_pos_weight = float(side_refine_gate_pos_weight)
        self.side_refine_box_gate = bool(side_refine_box_gate)
        self.side_refine_box_gate_weight = float(side_refine_box_gate_weight)
        self.side_refine_box_gate_threshold = float(side_refine_box_gate_threshold)
        self.side_refine_box_gate_margin = float(side_refine_box_gate_margin)
        self.nms_score_mode = str(nms_score_mode)
        self.nms_quality_beta = float(nms_quality_beta)
        self.nms_objectness_loss_weight = float(nms_objectness_loss_weight)
        self.nms_quality_loss_weight = float(nms_quality_loss_weight)
        self.nms_score_pos_iou = float(nms_score_pos_iou)
        self.nms_score_train_topk = int(nms_score_train_topk)
        self.nms_score_detach_features = bool(nms_score_detach_features)
        self.nms_residual_scale = float(nms_residual_scale)
        if self.nms_score_mode not in ('cls', 'tb_objectness', 'tb_quality',
                                       'pure_quality'):
            raise ValueError('nms_score_mode must be cls, tb_objectness, '
                             'tb_quality or pure_quality')
        if not 0.0 <= self.nms_quality_beta <= 2.0:
            raise ValueError('nms_quality_beta must be in [0, 2]')
        if not 0.0 < self.side_refine_box_gate_threshold < 1.0:
            raise ValueError('side_refine_box_gate_threshold must be in (0, 1)')
        if self.side_refine_box_gate_margin < 0.0:
            raise ValueError('side_refine_box_gate_margin must be non-negative')
        if not 0.0 < self.nms_score_pos_iou <= 1.0:
            raise ValueError('nms_score_pos_iou must be in (0, 1]')
        if self.nms_score_train_topk < 1:
            raise ValueError('nms_score_train_topk must be positive')
        if not 0.0 <= self.nms_residual_scale <= 2.0:
            raise ValueError('nms_residual_scale must be in [0, 2]')
        if (self.nms_objectness_loss_weight < 0.0
                or self.nms_quality_loss_weight < 0.0):
            raise ValueError('V47 NMS score loss weights must be non-negative')
        self.use_v47_candidate_head = (
            self.side_refine_box_gate or self.nms_score_mode != 'cls')
        if self.use_v47_candidate_head:
            # Isolate RNG so merely enabling this zero-impact-off branch does
            # not perturb H2/refiner initialisation in the legacy control.
            with torch.random.fork_rng(devices=[], enabled=True):
                self.v47_candidate_head = _V47CandidateScoreHead()
        else:
            self.v47_candidate_head = None
        self._v47_nms_debug = {}

    def _v45_run(self, feats, image_index, boxes, meta, detail_img=None):
        image_h, image_w = [int(v) for v in meta['img_shape'][:2]]
        p2 = feats[0][image_index:image_index + 1]
        p3 = feats[1][image_index:image_index + 1]
        if self.nms_score_detach_features:
            p2, p3 = p2.detach(), p3.detach()
        delta, logits = self.side_refiner(
            p2, p3, boxes.detach(), image_h, image_w,
            detail_img=(detail_img[image_index:image_index + 1]
                        if detail_img is not None else None))
        return delta, logits, self._v45_decode(boxes, delta, image_h, image_w)

    def _v47_candidate_logits(self, feats, image_index, boxes, meta):
        """Return [TB-objectness, refined-IoU quality, box-gate] logits."""
        if self.v47_candidate_head is None:
            return None
        p2 = feats[0][image_index:image_index + 1]
        p3 = feats[1][image_index:image_index + 1]
        if self.side_refine_detach_features:
            p2, p3 = p2.detach(), p3.detach()
        image_h, image_w = [int(v) for v in meta['img_shape'][:2]]
        return self.v47_candidate_head(p2, p3, boxes.detach(), image_h, image_w)

    def _v47_nms_score(self, candidate_logits, fallback_scores):
        """Identity-initialised residual calibration of the mature H2 score.

        Replacing the detector score by a new sigmoid caused near-tied scores
        and 100 arbitrary detections per image during early epochs.  Anchoring
        to the existing score preserves the exact baseline at initialisation;
        the new heads can only learn bounded multiplicative re-ranking.
        """
        base = fallback_scores.max(dim=1)[0]
        if self.nms_score_mode == 'cls' or candidate_logits is None:
            return base
        # The objectness output is initialised with bias -4, quality with 0.
        # Subtract those priors before tanh so both residual factors start at 1.
        obj_residual = torch.tanh(candidate_logits[:, 0] + 4.0)
        score = base * torch.exp(self.nms_residual_scale * obj_residual)
        if self.nms_score_mode == 'tb_objectness':
            return score
        quality_residual = torch.tanh(candidate_logits[:, 1])
        score = score * torch.exp(
            self.nms_residual_scale * self.nms_quality_beta
            * quality_residual)
        # Do not clamp at one: COCO/NMS only require an ordering score, and
        # clipping would create ties among the strongest candidates.
        return score

    @staticmethod
    def _v47_single_class_nms(boxes, score, fg_scores, score_thr, nms_cfg,
                              max_per_img):
        """One class-agnostic NMS; subtype is assigned only after keeping box."""
        merged = torch.stack((score, score.new_zeros(score.shape)), dim=1)
        det, _, keep = multiclass_nms(
            boxes, merged, score_thr, nms_cfg, max_per_img, return_inds=True)
        labels = fg_scores[keep].argmax(dim=1) if len(keep) else keep
        return det, labels, keep

    @staticmethod
    def _v47_coverage(boxes, gt, threshold):
        if boxes.numel() == 0 or gt.numel() == 0:
            return 0
        return int((bbox_overlaps(boxes[:, :4], gt).max(dim=0)[0]
                    >= threshold).sum().item())

    @staticmethod
    def _v47_nms_quality(candidate_iou, score, keep, nms):
        """Accumulate candidate-quality counts around the single final NMS.

        Pool = every candidate handed to NMS (IoU to its best GT); kept =
        the detections NMS returned (``keep`` indexes the candidate list,
        None when the class-score path cannot return indices); top-100 =
        good-box hit rate of the first 100 candidates under the actual
        ranking score; oracle top-100 = the same under the true candidate
        IoU, i.e. the ranking upper bound.  All counts are per processed
        image, normalised at emission time by ``nms['img']``.
        """
        with torch.no_grad():
            nms['img'] += 1
            good50 = int((candidate_iou >= .5).sum())
            good75 = int((candidate_iou >= .75).sum())
            nms['pool50'] += good50
            nms['pool75'] += good75
            if keep is not None:
                kept = candidate_iou[keep]
                kept50 = int((kept >= .5).sum())
                nms['kept50'] += kept50
                nms['kept_bad'] += int(len(kept)) - kept50
                nms['supp50'] += good50 - kept50
            topk = min(100, int(len(candidate_iou)))
            if topk:
                top = score.topk(topk).indices
                nms['top50'] += int((candidate_iou[top] >= .5).sum())
                nms['top75'] += int((candidate_iou[top] >= .75).sum())
                oracle_top = candidate_iou.topk(topk).indices
                nms['otop50'] += int((candidate_iou[oracle_top] >= .5).sum())
                nms['otop75'] += int((candidate_iou[oracle_top] >= .75).sum())
        return nms

    def _v47_candidate_score_train(self, feats, cls_scores, bbox_preds,
                                   img_metas, gt_bboxes, detail_img=None):
        """Train only NMS scoring while preserving the exact original H2 gate.

        This path is used by B1-B3 with
        ``side_refine_unified_candidates=False``.  It mirrors V45 inference:
        select top-K by the mature detector score, apply the original learned
        per-side hard gate, then supervise score heads at those final boxes.
        No alternative gate loss or unified-refiner loss enters this ablation.
        """
        if (self.nms_score_mode == 'cls' or self.v47_candidate_head is None
                or int(self._nam_epoch) + 1 < self.side_refine_start_epoch):
            return {}
        detail_img = self._v45_detail_tensor(detail_img)
        raw = self.bbox_head.get_bboxes(
            cls_scores, bbox_preds, img_metas, cfg=self.bbox_head.test_cfg,
            rescale=False, with_nms=False)
        objectness_losses, quality_losses = [], []
        obj_pred, obj_target_values = [], []
        q_pred, q_target_values = [], []
        nms = dict(gt=0, img=0, actual50=0, actual75=0,
                   oracle50=0, oracle75=0,
                   pool50=0, pool75=0, kept50=0, kept_bad=0, supp50=0,
                   top50=0, top75=0, otop50=0, otop75=0)
        for i, ((all_boxes, all_scores), gt, meta) in enumerate(
                zip(raw, gt_bboxes, img_metas)):
            if not len(all_boxes) or not len(gt):
                continue
            fg_scores = all_scores[:, :self.bbox_head.num_classes]
            index = self._v45_select(all_boxes, fg_scores)
            updated = all_boxes.clone()
            if index.numel():
                # Super() already trains the original H2 branch.  This second
                # pass only reconstructs inference coordinates for score
                # labels and must not add another gradient path to H2.
                with torch.no_grad():
                    delta, side_gate, _ = self._v45_run(
                        feats, i, all_boxes[index], meta,
                        detail_img=detail_img)
                    hard = (side_gate.sigmoid()
                            >= self.side_refine_gate_threshold).to(delta.dtype)
                    refined = self._v45_decode(
                        all_boxes[index], delta * hard,
                        int(meta['img_shape'][0]), int(meta['img_shape'][1]))
                updated[index] = refined

            logits = self._v47_candidate_logits(feats, i, updated, meta)
            final_iou_all = bbox_overlaps(
                updated.detach(), gt).max(dim=1)[0]
            train_count = min(self.nms_score_train_topk, len(all_boxes))
            ranked = fg_scores.max(dim=1)[0].topk(train_count).indices
            positive = (final_iou_all >= self.nms_score_pos_iou).nonzero(
                as_tuple=False).squeeze(1)
            score_index = torch.unique(torch.cat((ranked, positive)))
            target = (final_iou_all[score_index]
                      >= self.nms_score_pos_iou).float()
            pos_count = target.sum().clamp_min(1.)
            neg_count = (len(target) - target.sum()).clamp_min(1.)
            pos_weight = (neg_count / pos_count).clamp(max=8.)
            objectness_losses.append(F.binary_cross_entropy_with_logits(
                logits[score_index, 0], target, pos_weight=pos_weight))
            quality_losses.append(F.binary_cross_entropy_with_logits(
                logits[score_index, 1], final_iou_all[score_index]))
            obj_pred.append(logits[score_index, 0].detach().sigmoid())
            obj_target_values.append(target.detach())
            q_pred.append(logits[score_index, 1].detach().sigmoid())
            q_target_values.append(final_iou_all[score_index].detach())

            score = self._v47_nms_score(logits, fg_scores)
            det, _, keep = self._v47_single_class_nms(
                updated, score, fg_scores,
                self.bbox_head.test_cfg.score_thr,
                self.bbox_head.test_cfg.nms,
                self.bbox_head.test_cfg.max_per_img)
            oracle_det, _, _ = self._v47_single_class_nms(
                updated, final_iou_all, fg_scores,
                self.bbox_head.test_cfg.score_thr,
                self.bbox_head.test_cfg.nms,
                self.bbox_head.test_cfg.max_per_img)
            self._v47_nms_quality(final_iou_all, score, keep, nms)
            nms['gt'] += int(gt.shape[0])
            for threshold, key in ((.50, '50'), (.75, '75')):
                nms['actual' + key] += self._v47_coverage(det, gt, threshold)
                nms['oracle' + key] += self._v47_coverage(
                    oracle_det, gt, threshold)

        if not objectness_losses:
            return {}
        obj_p, obj_t = torch.cat(obj_pred), torch.cat(obj_target_values)
        quality_p, quality_t = torch.cat(q_pred), torch.cat(q_target_values)
        self._v47_nms_debug.update(
            obj_pred=float(obj_p.mean()), obj_target=float(obj_t.mean()),
            quality_pred=float(quality_p.mean()),
            quality_target=float(quality_t.mean()),
            pool50=float(nms['pool50']), pool75=float(nms['pool75']),
            kept50=float(nms['kept50']), kept_bad=float(nms['kept_bad']),
            supp50=float(nms['supp50']))
        step = int(getattr(self.bbox_head, '_debug_step', 0))
        if (self.side_refine_debug_interval
                and step % self.side_refine_debug_interval == 0):
            gt_count = max(nms['gt'], 1)
            img_count = max(nms['img'], 1)
            _emit_training_debug(
                '[V47-SCORE] iter={} mode={} TB={:.3f}/{:.3f} '
                'Q={:.3f}/{:.3f} R50 a/o={:.3f}/{:.3f} '
                'R75 a/o={:.3f}/{:.3f} '
                'pool50/75={:.1f}/{:.1f} kept50/FP={:.1f}/{:.1f} '
                'supp50={:.1f} top100 a50/a75/o75={:.1f}/{:.1f}/{:.1f}'.format(
                    max(0, step - 1), self.nms_score_mode,
                    float(obj_p.mean()), float(obj_t.mean()),
                    float(quality_p.mean()), float(quality_t.mean()),
                    nms['actual50'] / gt_count, nms['oracle50'] / gt_count,
                    nms['actual75'] / gt_count, nms['oracle75'] / gt_count,
                    nms['pool50'] / img_count, nms['pool75'] / img_count,
                    nms['kept50'] / img_count, nms['kept_bad'] / img_count,
                    nms['supp50'] / img_count,
                    nms['top50'] / img_count, nms['top75'] / img_count,
                    nms['otop75'] / img_count))
        output = {
            'loss_v47_tb_objectness': self.nms_objectness_loss_weight
            * sum(objectness_losses) / len(objectness_losses)}
        if self.nms_quality_loss_weight > 0:
            output['loss_v47_refined_quality'] = (
                self.nms_quality_loss_weight
                * sum(quality_losses) / len(quality_losses))
        return output

    def _v45_side_refine_train(self, feats, cls_scores, bbox_preds, img_metas,
                               gt_bboxes, detail_img=None):
        if not self.side_refine_unified_candidates:
            output = super()._v45_side_refine_train(
                feats, cls_scores, bbox_preds, img_metas, gt_bboxes,
                detail_img=detail_img)
            output.update(self._v47_candidate_score_train(
                feats, cls_scores, bbox_preds, img_metas, gt_bboxes,
                detail_img=detail_img))
            return output
        if not self.use_side_refine:
            return {}
        detail_img = self._v45_detail_tensor(detail_img)
        raw = self.bbox_head.get_bboxes(
            cls_scores, bbox_preds, img_metas, cfg=self.bbox_head.test_cfg,
            rescale=False, with_nms=False)
        from mmdet.core import multiclass_nms
        # ``losses`` is one complete objective per image.  V47-A1/A2 used to
        # append regression/GIoU and gate objectives as separate list items,
        # then average the list.  That silently halved the H2 geometry weight
        # whenever a gate loss existed.  Keep the original H2 contract: sum
        # all terms for an image, then average images.
        losses, objectness_losses, quality_losses = [], [], []
        ciou_all, riou_all, delta_all = [], [], []
        probs_all, target_all = [], []
        nms = dict(gt=0, img=0, coarse50=0, coarse75=0, refined50=0,
                   refined75=0, actual50=0, actual75=0,
                   oracle50=0, oracle75=0,
                   pool50=0, pool75=0, kept50=0, kept_bad=0, supp50=0,
                   top50=0, top75=0, otop50=0, otop75=0)
        for i, ((all_boxes, all_scores), gt, meta) in enumerate(
                zip(raw, gt_bboxes, img_metas)):
            if not len(all_boxes) or not len(gt):
                continue
            fg_scores = all_scores[:, :self.bbox_head.num_classes]
            index = self._v45_select(all_boxes, fg_scores)
            boxes = all_boxes[index]
            all_overlap = bbox_overlaps(all_boxes, gt)
            overlap = all_overlap[index]
            best_iou, best_gt = overlap.max(dim=1)
            valid = best_iou >= self.side_refine_min_iou
            # A box gate must inspect the coarse proposal before deciding
            # whether to apply H2.  TB/objectness and localisation quality are
            # evaluated later at the final gate-applied coordinates.
            coarse_candidate_logits_all = self._v47_candidate_logits(
                feats, i, all_boxes, meta)
            candidate_logits = (coarse_candidate_logits_all[index]
                                if coarse_candidate_logits_all is not None
                                else None)
            delta, gate_logits, refined = self._v45_run(
                feats, i, boxes, meta, detail_img=detail_img)
            gate_target = torch.zeros_like(delta)
            image_loss = None
            coarse_side_iou = best_iou.new_zeros(best_iou.shape)
            if valid.any():
                v_boxes, target = boxes[valid], gt[best_gt[valid]]
                v_delta, v_refined = delta[valid], refined[valid]
                target_delta = self._v45_target(v_boxes, target).clamp(
                    -self.side_refiner.max_ratio, self.side_refiner.max_ratio)
                side_loss = F.smooth_l1_loss(
                    v_delta, target_delta, beta=self.side_refine_huber_beta)
                giou = 1 - bbox_overlaps(
                    v_refined, target, mode='giou', is_aligned=True).mean()
                image_loss = side_loss + giou
                coarse_side_iou = bbox_overlaps(
                    v_boxes, target, is_aligned=True)
                image_h, image_w = [int(v) for v in meta['img_shape'][:2]]
                for side in range(4):
                    side_delta = torch.zeros_like(v_delta)
                    side_delta[:, side] = v_delta[:, side]
                    one_side = self._v45_decode(
                        v_boxes, side_delta, image_h, image_w)
                    side_iou = bbox_overlaps(
                        one_side, target, is_aligned=True)
                    gate_target[valid, side] = (
                        side_iou > coarse_side_iou + .005).float()
                ciou_all.append(coarse_side_iou.detach())
                riou_all.append(bbox_overlaps(
                    v_refined.detach(), target, is_aligned=True))
                delta_all.append(v_delta.detach())

            # Legacy per-side gate remains available for A0/A1/A2.  New B
            # configurations use one learned box gate instead, so all four
            # residual sides are accepted/rejected coherently.
            if (self.side_refiner.mode == 'gated_direct'
                    and not self.side_refine_box_gate):
                pos_weight = gate_logits.new_full(
                    (4,), self.side_refine_gate_pos_weight)
                gate_loss = F.binary_cross_entropy_with_logits(
                    gate_logits, gate_target, pos_weight=pos_weight)
                image_loss = ((image_loss if image_loss is not None else 0.)
                              + self.side_refine_gate_weight * gate_loss)
                probs_all.append(gate_logits.detach().sigmoid())
                target_all.append(gate_target.detach())

            # Apply the exact hard decision used at inference.  The box gate
            # has a counterfactual target based on full-box IoU improvement;
            # ``side_refine_min_iou`` only filters residual supervision and
            # never decides the gate at inference.
            selected = refined
            box_gate = None
            if self.side_refine_box_gate:
                if candidate_logits is None:
                    raise RuntimeError('box gate requires V47 candidate head')
                full_coarse_iou = best_iou
                full_refined_iou = best_iou.new_zeros(best_iou.shape)
                if valid.any():
                    full_refined_iou[valid] = bbox_overlaps(
                        refined[valid], gt[best_gt[valid]], is_aligned=True)
                box_target = (
                    full_refined_iou > full_coarse_iou
                    + self.side_refine_box_gate_margin).float()
                box_pos_weight = candidate_logits.new_tensor(
                    self.side_refine_gate_pos_weight)
                box_loss = F.binary_cross_entropy_with_logits(
                    candidate_logits[:, 2], box_target,
                    pos_weight=box_pos_weight)
                image_loss = ((image_loss if image_loss is not None else 0.)
                              + self.side_refine_box_gate_weight * box_loss)
                box_gate = (candidate_logits[:, 2].sigmoid()
                            >= self.side_refine_box_gate_threshold).to(delta.dtype)
                selected = self._v45_decode(
                    boxes, delta * box_gate[:, None],
                    int(meta['img_shape'][0]), int(meta['img_shape'][1]))
                probs_all.append(candidate_logits[:, 2].detach().sigmoid())
                target_all.append(box_target.detach())
            elif self.side_refiner.mode == 'gated_direct':
                hard = (gate_logits.sigmoid() >= self.side_refine_gate_threshold).to(delta.dtype)
                h, w = [int(v) for v in meta['img_shape'][:2]]
                selected = self._v45_decode(boxes, delta * hard, h, w)
            updated = all_boxes.clone()
            updated[index] = selected

            final_candidate_logits_all = (
                self._v47_candidate_logits(feats, i, updated, meta)
                if self.nms_score_mode != 'cls'
                else coarse_candidate_logits_all)

            # Objectness and quality are supervised independently of subtype:
            # p(TB)=1 iff the *final gate-applied* candidate matches a lesion;
            # q is the actual IoU of the final box used by NMS.  Training uses
            # top current-score candidates plus every matched positive, while
            # inference scores all candidates so recall is not hard-limited to
            # H2's refinement top-K.
            if (final_candidate_logits_all is not None
                    and self.nms_score_mode != 'cls'):
                final_iou_all = bbox_overlaps(
                    updated.detach(), gt).max(dim=1)[0]
                train_count = min(self.nms_score_train_topk, len(all_boxes))
                ranked = fg_scores.max(dim=1)[0].topk(train_count).indices
                positive = (final_iou_all >= self.nms_score_pos_iou).nonzero(
                    as_tuple=False).squeeze(1)
                score_index = torch.unique(torch.cat((ranked, positive)))
                obj_target = (final_iou_all[score_index]
                              >= self.nms_score_pos_iou).float()
                pos_count = obj_target.sum().clamp_min(1.)
                neg_count = (len(obj_target) - obj_target.sum()).clamp_min(1.)
                pos_weight = (neg_count / pos_count).clamp(max=8.)
                objectness_loss = F.binary_cross_entropy_with_logits(
                    final_candidate_logits_all[score_index, 0], obj_target,
                    pos_weight=pos_weight)
                final_iou = final_iou_all[score_index]
                quality_loss = F.binary_cross_entropy_with_logits(
                    final_candidate_logits_all[score_index, 1], final_iou)
                objectness_losses.append(objectness_loss)
                quality_losses.append(quality_loss)
                self._v47_nms_debug.update(
                    obj_pred=float(final_candidate_logits_all[score_index, 0].detach().sigmoid().mean()),
                    obj_target=float(obj_target.mean()),
                    quality_pred=float(final_candidate_logits_all[score_index, 1].detach().sigmoid().mean()),
                    quality_target=float(final_iou.mean()))

            if image_loss is not None:
                losses.append(image_loss)
            nms['gt'] += int(gt.shape[0])
            for threshold, key in ((.50, '50'), (.75, '75')):
                nms['coarse' + key] += self._v47_coverage(all_boxes, gt, threshold)
                nms['refined' + key] += self._v47_coverage(updated, gt, threshold)
            nms_score = self._v47_nms_score(
                final_candidate_logits_all, fg_scores)
            if self.nms_score_mode == 'cls':
                det, _ = multiclass_nms(
                    updated, all_scores, self.bbox_head.test_cfg.score_thr,
                    self.bbox_head.test_cfg.nms, self.bbox_head.test_cfg.max_per_img)
                keep = None
            else:
                det, _, keep = self._v47_single_class_nms(
                    updated, nms_score, fg_scores,
                    self.bbox_head.test_cfg.score_thr,
                    self.bbox_head.test_cfg.nms,
                    self.bbox_head.test_cfg.max_per_img)
            best_updated = bbox_overlaps(updated, gt).max(dim=1)[0]
            self._v47_nms_quality(best_updated, nms_score, keep, nms)
            if self.nms_score_mode == 'cls':
                oracle_scores = torch.zeros_like(all_scores)
                _, labels = fg_scores.max(dim=1)
                oracle_scores[
                    torch.arange(len(updated), device=updated.device),
                    labels] = best_updated
                oracle_det, _ = multiclass_nms(
                    updated, oracle_scores,
                    self.bbox_head.test_cfg.score_thr,
                    self.bbox_head.test_cfg.nms,
                    self.bbox_head.test_cfg.max_per_img)
            else:
                # B-series actual and oracle now share the exact same
                # class-agnostic NMS; only the ranking score differs.
                oracle_det, _, _ = self._v47_single_class_nms(
                    updated, best_updated, fg_scores,
                    self.bbox_head.test_cfg.score_thr,
                    self.bbox_head.test_cfg.nms,
                    self.bbox_head.test_cfg.max_per_img)
            for threshold, key in ((.50, '50'), (.75, '75')):
                nms['actual' + key] += self._v47_coverage(det, gt, threshold)
                nms['oracle' + key] += self._v47_coverage(oracle_det, gt, threshold)
        if not losses:
            return {}
        loss = sum(losses) / len(losses)
        debug = {'nms': nms}
        if ciou_all:
            c, r, d = torch.cat(ciou_all), torch.cat(riou_all), torch.cat(delta_all)
            debug.update(samples=len(c), coarse=float(c.mean()), refined=float(r.mean()),
                         improve=float((r > c).float().mean()),
                         up75=int(((c < .75) & (r >= .75)).sum()),
                         down75=int(((c >= .75) & (r < .75)).sum()),
                         side_abs=[float(v) for v in d.abs().mean(0)])
        if probs_all:
            probs, target = torch.cat(probs_all), torch.cat(target_all)
            gate_threshold = (self.side_refine_box_gate_threshold
                              if self.side_refine_box_gate
                              else self.side_refine_gate_threshold)
            chosen = probs >= gate_threshold
            debug.update(gate_open=float(chosen.float().mean()),
                         gate_target=float(target.mean()),
                         gate_precision=float((target[chosen].mean() if chosen.any() else target.new_zeros(())).item()),
                         gate_recall=float((chosen[target.bool()].float().mean() if target.bool().any() else target.new_zeros(())).item()))
        self._v45_debug = debug
        step = int(getattr(self.bbox_head, '_debug_step', 0))
        if self.side_refine_debug_interval and step % self.side_refine_debug_interval == 0:
            gt_count = max(nms['gt'], 1)
            img_count = max(nms['img'], 1)
            _emit_training_debug(
                '[V47-REFINE] iter={} n={} IoU={:.3f}->{:.3f} gate={:.3f}/{:.3f}/{:.3f}/{:.3f} '
                'R50 c/r/a/o={:.3f}/{:.3f}/{:.3f}/{:.3f} R75 c/r/a/o={:.3f}/{:.3f}/{:.3f}/{:.3f} '
                'TB={:.3f}/{:.3f} Q={:.3f}/{:.3f} '
                'pool50/75={:.1f}/{:.1f} kept50/FP={:.1f}/{:.1f} '
                'supp50={:.1f} top100 a50/a75/o75={:.1f}/{:.1f}/{:.1f}'.format(
                    max(0, step - 1), debug.get('samples', 0), debug.get('coarse', 0.), debug.get('refined', 0.),
                    debug.get('gate_open', 0.), debug.get('gate_target', 0.), debug.get('gate_precision', 0.), debug.get('gate_recall', 0.),
                    nms['coarse50']/gt_count, nms['refined50']/gt_count, nms['actual50']/gt_count, nms['oracle50']/gt_count,
                    nms['coarse75']/gt_count, nms['refined75']/gt_count, nms['actual75']/gt_count, nms['oracle75']/gt_count,
                    self._v47_nms_debug.get('obj_pred', 0.), self._v47_nms_debug.get('obj_target', 0.),
                    self._v47_nms_debug.get('quality_pred', 0.), self._v47_nms_debug.get('quality_target', 0.),
                    nms['pool50']/img_count, nms['pool75']/img_count,
                    nms['kept50']/img_count, nms['kept_bad']/img_count,
                    nms['supp50']/img_count,
                    nms['top50']/img_count, nms['top75']/img_count,
                    nms['otop75']/img_count))
        output_losses = {
            'loss_side_refine': self.side_refine_loss_weight * loss}
        if objectness_losses:
            output_losses['loss_v47_tb_objectness'] = (
                self.nms_objectness_loss_weight
                * sum(objectness_losses) / len(objectness_losses))
        if quality_losses and self.nms_quality_loss_weight > 0:
            output_losses['loss_v47_refined_quality'] = (
                self.nms_quality_loss_weight
                * sum(quality_losses) / len(quality_losses))
        return output_losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """H2 refinement followed by exactly one final NMS.

        Legacy V45 behaviour is preserved unless a V47 box gate or explicit
        TB-quality NMS score is enabled.  In the latter case NMS is class
        agnostic; subtype logits are read only after a box has survived.
        """
        if (not self.use_side_refine or not self.use_v47_candidate_head):
            return super().simple_test(img, img_metas, rescale=rescale, **kwargs)
        detail_img = self._v45_detail_tensor(kwargs.get('detail_img', None))
        batch_shape = tuple(int(v) for v in img.shape[-2:])
        for meta in img_metas:
            meta['batch_input_shape'] = batch_shape
        self._collect_roi_debug(img_metas)
        backbone_feats = self.backbone(img)
        feats, c2_feat = self._build_pyramid(backbone_feats, lbp_lab=None)
        lung_masks = self._collect_lung_masks(
            img_metas, device=img.device, batch_input_shape=batch_shape)
        outs = self.bbox_head.forward(
            feats, nam_graphs=None, lbp_lab=None, c2_feat=c2_feat,
            region_feat=None, img_shapes=[batch_shape for _ in img_metas],
            lung_masks=lung_masks,
            reg_detail_feats=getattr(self.neck, 'regression_detail_feats', None))
        raw = self.bbox_head.get_bboxes(
            outs[0], outs[1], img_metas, cfg=self.bbox_head.test_cfg,
            rescale=False, with_nms=False)
        results = []
        score_values, gate_values = [], []
        for i, ((boxes, scores), meta) in enumerate(zip(raw, img_metas)):
            fg_scores = scores[:, :self.bbox_head.num_classes]
            # Coarse-aligned logits decide the box gate.  NMS quality is
            # recomputed below at the final gate-applied coordinates.
            coarse_candidate_logits = self._v47_candidate_logits(
                feats, i, boxes, meta)
            index = self._v45_select(boxes, fg_scores)
            updated = boxes.clone()
            if index.numel():
                delta, side_gate, refined = self._v45_run(
                    feats, i, boxes[index], meta, detail_img=detail_img)
                if self.side_refine_box_gate:
                    box_gate = (coarse_candidate_logits[index, 2].sigmoid()
                                >= self.side_refine_box_gate_threshold).to(delta.dtype)
                    refined = self._v45_decode(
                        boxes[index], delta * box_gate[:, None],
                        int(meta['img_shape'][0]), int(meta['img_shape'][1]))
                    gate_values.append(box_gate.detach())
                elif self.side_refiner.mode == 'gated_direct':
                    gate = (side_gate.sigmoid() >= self.side_refine_gate_threshold).to(delta.dtype)
                    refined = self._v45_decode(
                        boxes[index], delta * gate,
                        int(meta['img_shape'][0]), int(meta['img_shape'][1]))
                updated[index] = refined

            final_candidate_logits = (
                self._v47_candidate_logits(feats, i, updated, meta)
                if self.nms_score_mode != 'cls'
                else coarse_candidate_logits)
            nms_score = self._v47_nms_score(
                final_candidate_logits, fg_scores)
            if self.nms_score_mode == 'cls':
                det, labels = multiclass_nms(
                    updated, scores, self.bbox_head.test_cfg.score_thr,
                    self.bbox_head.test_cfg.nms,
                    self.bbox_head.test_cfg.max_per_img)
            else:
                det, labels, _ = self._v47_single_class_nms(
                    updated, nms_score, fg_scores,
                    self.bbox_head.test_cfg.score_thr,
                    self.bbox_head.test_cfg.nms,
                    self.bbox_head.test_cfg.max_per_img)
            if rescale and det.numel():
                det[:, :4] /= det.new_tensor(meta['scale_factor'])
            results.append((det, labels))
            score_values.append(nms_score.detach())
        if score_values:
            score = torch.cat(score_values)
            self._v47_nms_debug.update(
                score_mean=float(score.mean()), score_p90=float(
                    torch.quantile(score, .9)),
                infer_gate_open=float(torch.cat(gate_values).float().mean())
                if gate_values else 0.)
        from mmdet.core import bbox2result
        return [bbox2result(det, label, self.bbox_head.num_classes)
                for det, label in results]


@HOOKS.register_module(force=True)
class NAMV47DebugHook(NAMV45DebugHook):
    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        runner.logger.info(
            '[V47-BUILD] unified_candidates=%s side_gate=%s box_gate=%s detach_refiner=%s '
            'detach_score=%s gate_pos_weight=%.2f nms_score=%s beta=%.2f '
            'res_scale=%.2f obj_w=%.2f q_w=%.2f',
            model.side_refine_unified_candidates,
            (getattr(model.side_refiner, 'mode', None) == 'gated_direct'
             and not model.side_refine_box_gate),
            model.side_refine_box_gate, model.side_refine_detach_features,
            model.nms_score_detach_features, model.side_refine_gate_pos_weight,
            model.nms_score_mode, model.nms_quality_beta,
            model.nms_residual_scale, model.nms_objectness_loss_weight,
            model.nms_quality_loss_weight)


@DETECTORS.register_module(force=True)
class RetinaNAMAttV48LegacyPureQuality(RetinaNAMAttV47):
    """V48 phase-1: pure predicted-IoU NMS score plus optional Soft-NMS.

    E1 isolates Gaussian Soft-NMS on top of the B3 ``tb_quality`` score;
    E2 replaces the score with the candidate head's IoU-quality logit alone
    (no subtype-class anchor, no TB-objectness term); E1+E2 combines both.
    The score-head training objective is deliberately untouched, so every
    configuration warm-starts from the B3 checkpoint.
    """

    def __init__(self, *args,
                 side_refine_softnms=False,
                 side_refine_softnms_sigma=.5,
                 side_refine_softnms_min_score=1e-3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.side_refine_softnms = bool(side_refine_softnms)
        self.side_refine_softnms_sigma = float(side_refine_softnms_sigma)
        self.side_refine_softnms_min_score = float(
            side_refine_softnms_min_score)
        if self.side_refine_softnms_sigma <= 0.0:
            raise ValueError('side_refine_softnms_sigma must be positive')
        if not 0.0 < self.side_refine_softnms_min_score < 1.0:
            raise ValueError(
                'side_refine_softnms_min_score must be in (0, 1)')
        self.integrated_build_id = 'v3.48-pure-quality-nms-20260903'

    def _v47_nms_score(self, logits, fallback_scores):
        """``pure_quality``: the IoU-quality logit is the entire NMS score."""
        if self.nms_score_mode != 'pure_quality':
            return super()._v47_nms_score(logits, fallback_scores)
        if logits is None or len(logits) == 0:
            return fallback_scores.max(dim=1)[0]
        return logits[:, 1].sigmoid()

    def _v47_single_class_nms(self, boxes, score, fg_scores, score_thr,
                              nms_cfg, max_per_img):
        if not self.side_refine_softnms:
            return super()._v47_single_class_nms(
                boxes, score, fg_scores, score_thr, nms_cfg, max_per_img)
        # Gaussian Soft-NMS with mmdet soft_nms semantics: the score_thr
        # pre-filter matches the hard path, ``keep`` indexes the original
        # candidate list, and decayed scores are returned in the dets.
        scores = score.detach().clone()
        valid = torch.nonzero(scores > score_thr).flatten()
        if len(valid) == 0:
            empty = boxes.new_zeros((0, 5))
            return (empty, boxes.new_zeros((0,), dtype=torch.long),
                    boxes.new_zeros((0,), dtype=torch.long))
        order = valid[scores[valid].argsort(descending=True)]
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i.item())
            if order.numel() == 1:
                break
            rest = order[1:]
            overlap = bbox_overlaps(boxes[i:i + 1], boxes[rest])[0]
            decay = torch.exp(-(overlap * overlap)
                              / self.side_refine_softnms_sigma)
            scores[rest] *= decay
            alive = torch.nonzero(
                scores[rest] > self.side_refine_softnms_min_score).flatten()
            order = rest[alive]
            if order.numel():
                order = order[scores[order].argsort(descending=True)]
        keep_t = boxes.new_tensor(keep, dtype=torch.long)
        if max_per_img > 0 and len(keep_t) > max_per_img:
            keep_t = keep_t[:max_per_img]
        labels = fg_scores[keep_t].argmax(dim=1) if len(keep_t) else keep_t
        det = torch.cat([boxes[keep_t], scores[keep_t, None]], dim=1)
        return det, labels, keep_t


@HOOKS.register_module(force=True)
class NAMV48LegacyDebugHook(NAMV47DebugHook):
    """Records the V48 phase-1 score/NMS contract at startup."""

    def before_run(self, runner):
        super().before_run(runner)
        model = (runner.model.module if hasattr(runner.model, 'module')
                 else runner.model)
        runner.logger.info(
            '[V48-BUILD] score_mode=%s softnms=%s sigma=%.2f min_score=%.1e',
            model.nms_score_mode, model.side_refine_softnms,
            model.side_refine_softnms_sigma,
            model.side_refine_softnms_min_score)


# -------------------------------------------------------------------------
# V48 (2026-09-04): H2-preserving quality-aware ranking experiment.
#
# This implementation intentionally does NOT inherit V47's unified candidate
# or box-gate contracts.  With ``v48_iacs=False`` it delegates to V45 exactly,
# which makes A0 a real H2 control.  The experimental branch only changes the
# score used after the original H2 four-side gate has produced its final box.
# -------------------------------------------------------------------------
class _V48IACSHead(nn.Module):
    """Candidate features -> IACS residual plus four boundary log variances."""

    def __init__(self, channels=256, hidden=128):
        super().__init__()
        self.p2_proj = nn.Conv2d(channels, hidden // 2, 1, bias=False)
        self.p3_proj = nn.Conv2d(channels, hidden // 2, 1, bias=False)
        self.mlp = nn.Sequential(nn.Linear(hidden * 5, hidden),
                                 nn.SiLU(inplace=True))
        # residual-IACS logit, logvar(left, top, right, bottom)
        self.out = nn.Linear(hidden, 5)
        for layer in (self.p2_proj, self.p3_proj):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                    nonlinearity='relu')
        nn.init.kaiming_normal_(self.mlp[0].weight, mode='fan_in',
                                nonlinearity='relu')
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            # IACS residual zero -> exact original H2 ranking initially.
            # exp(-4/2) gives a finite but small initial boundary std.
            self.out.bias.copy_(self.out.bias.new_tensor([0., -4., -4., -4., -4.]))

    def forward(self, p2, p3, boxes, image_h, image_w):
        if not boxes.numel():
            return boxes.new_zeros((0, 5))
        points = _V47CandidateScoreHead._points(boxes, image_h, image_w)
        p2_value = _V45CrossScaleSideRefiner._sample(
            self.p2_proj(p2), points, image_h, image_w)
        p3_value = _V45CrossScaleSideRefiner._sample(
            self.p3_proj(p3), points, image_h, image_w)
        return self.out(self.mlp(torch.cat([p2_value, p3_value], dim=-1).flatten(1)))


@DETECTORS.register_module(force=True)
class RetinaNAMAttV48(RetinaNAMAttV45):
    """Original H2 gate + IACS ranking / ranking loss / boundary uncertainty.

    ``v48_iacs=False`` is deliberately a byte-for-byte behavioural H2 path:
    no extra module is instantiated and V45 owns training and inference.
    All experimental variants execute a *single class-agnostic* final NMS;
    subtype logits are used only to label boxes that survived it.
    """

    def __init__(self, *args, v48_iacs=False, v48_rank_loss=False,
                 v48_uncertainty=False, v48_variance_voting=False,
                 v48_softnms=False, v48_iacs_residual_scale=.5,
                 v48_iacs_loss_weight=.25, v48_rank_loss_weight=.10,
                 v48_rank_margin=.05, v48_rank_iou_gap=.10,
                 v48_uncertainty_loss_weight=.05,
                 v48_sigma_calibration=.25, v48_vote_iou=.50,
                 v48_vote_center=.25,
                 v48_vote_topk=5, v48_softnms_sigma=.50,
                 v48_softnms_min_score=1e-3,
                 v48_candidates_per_bin=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.v48_iacs = bool(v48_iacs)
        self.v48_rank_loss = bool(v48_rank_loss)
        self.v48_uncertainty = bool(v48_uncertainty)
        self.v48_variance_voting = bool(v48_variance_voting)
        self.v48_softnms = bool(v48_softnms)
        self.v48_iacs_residual_scale = float(v48_iacs_residual_scale)
        self.v48_iacs_loss_weight = float(v48_iacs_loss_weight)
        self.v48_rank_loss_weight = float(v48_rank_loss_weight)
        self.v48_rank_margin = float(v48_rank_margin)
        self.v48_rank_iou_gap = float(v48_rank_iou_gap)
        self.v48_uncertainty_loss_weight = float(v48_uncertainty_loss_weight)
        self.v48_sigma_calibration = float(v48_sigma_calibration)
        self.v48_vote_iou = float(v48_vote_iou)
        self.v48_vote_center = float(v48_vote_center)
        self.v48_vote_topk = int(v48_vote_topk)
        self.v48_softnms_sigma = float(v48_softnms_sigma)
        self.v48_softnms_min_score = float(v48_softnms_min_score)
        self.v48_candidates_per_bin = int(v48_candidates_per_bin)
        if self.v48_rank_loss and not self.v48_iacs:
            raise ValueError('v48_rank_loss requires v48_iacs=True')
        if self.v48_uncertainty and not self.v48_iacs:
            raise ValueError('v48_uncertainty requires v48_iacs=True')
        if (self.v48_variance_voting or self.v48_softnms) and not self.v48_iacs:
            raise ValueError('V48 post-processing requires v48_iacs=True')
        if self.v48_candidates_per_bin < 1 or self.v48_vote_topk < 1:
            raise ValueError('V48 candidate/vote counts must be positive')
        if not 0.0 < self.v48_vote_center <= 1.0:
            raise ValueError('v48_vote_center must be in (0, 1]')
        if self.v48_softnms_sigma <= 0 or not 0 < self.v48_softnms_min_score < 1:
            raise ValueError('invalid V48 Soft-NMS settings')
        self.v48_head = _V48IACSHead() if self.v48_iacs else None
        self._v48_debug = {}
        self.integrated_build_id = 'v3.48-h2-iacs-rank-uncertainty-20260904'

    def _v48_logits(self, feats, image_index, boxes, meta):
        h, w = [int(v) for v in meta['img_shape'][:2]]
        return self.v48_head(feats[0][image_index:image_index + 1],
                             feats[1][image_index:image_index + 1],
                             boxes.detach(), h, w)

    @staticmethod
    def _v48_base_logit(fg_scores):
        return torch.logit(fg_scores.max(dim=1)[0].clamp(1e-4, 1. - 1e-4))

    def _v48_score(self, logits, fg_scores, boxes):
        """IACS score with an identity-initialised residual and bounded UQ."""
        logit = self._v48_base_logit(fg_scores)
        logit = logit + self.v48_iacs_residual_scale * torch.tanh(logits[:, 0])
        score = logit.sigmoid()
        sigma = logits[:, 1:5].clamp(-6., 3.).mul(.5).exp()
        if not self.v48_uncertainty:
            return logit, score, sigma
        width = (boxes[:, 2] - boxes[:, 0]).clamp_min(2.)
        height = (boxes[:, 3] - boxes[:, 1]).clamp_min(2.)
        px = sigma * torch.stack([width, height, width, height], dim=1)
        expanded = torch.stack([
            boxes[:, 0] - px[:, 0], boxes[:, 1] - px[:, 1],
            boxes[:, 2] + px[:, 2], boxes[:, 3] + px[:, 3]], dim=1)
        # IoU(B, expanded B) and centre-position stability are GT-free at test.
        expand_iou = bbox_overlaps(boxes, expanded, is_aligned=True).clamp(0., 1.)
        centre_std = .5 * torch.sqrt(px[:, 0] ** 2 + px[:, 2] ** 2) / width
        centre_std = centre_std + .5 * torch.sqrt(px[:, 1] ** 2 + px[:, 3] ** 2) / height
        centre = torch.exp(-centre_std.clamp(max=8.))
        stability = (expand_iou * centre).clamp_min(1e-4)
        # logvar is initialised to -4 (relative sigma=e^-2).  Centre the
        # uncertainty factor on that prior so enabling A3/A4 is an exact
        # score identity at iteration zero, including score-thresholding.
        prior_sigma = math.exp(-2.0)
        prior_stability = ((1.0 + 2.0 * prior_sigma) ** -2
                           * math.exp(-math.sqrt(2.0) * prior_sigma))
        calibration = torch.exp(
            self.v48_sigma_calibration
            * torch.tanh(torch.log(stability / prior_stability)))
        return logit, score * calibration, sigma

    def _v48_balanced_indices(self, iou, base_score):
        """Deterministically sample high/mid/low/hard-FP candidate strata."""
        bins = ((iou >= .75), ((iou >= .50) & (iou < .75)),
                ((iou >= .10) & (iou < .50)), (iou < .10))
        selected = []
        taken = torch.zeros_like(iou, dtype=torch.bool)
        for mask in bins:
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            if idx.numel():
                idx = idx[base_score[idx].argsort(descending=True)]
                idx = idx[:self.v48_candidates_per_bin]
                selected.append(idx); taken[idx] = True
        budget = self.v48_candidates_per_bin * 4
        if selected:
            out = torch.cat(selected)
        else:
            out = iou.new_zeros((0,), dtype=torch.long)
        if out.numel() < min(budget, iou.numel()):
            rest = (~taken).nonzero(as_tuple=False).squeeze(1)
            rest = rest[base_score[rest].argsort(descending=True)]
            out = torch.cat((out, rest[:budget - out.numel()]))
        return out

    @staticmethod
    def _v48_qfl(logit, target):
        pred = logit.sigmoid()
        weight = (target - pred).abs().pow(2)
        return (F.binary_cross_entropy_with_logits(logit, target, reduction='none')
                * weight).mean()

    def _v48_pairwise_loss(self, score, iou, gt_index):
        terms = []
        for gid in gt_index.unique():
            idx = (gt_index == gid).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue
            diff = iou[idx, None] - iou[None, idx]
            pair = diff >= self.v48_rank_iou_gap
            if pair.any():
                delta = score[idx, None] - score[None, idx]
                terms.append(F.relu(self.v48_rank_margin - delta[pair]).mean())
        return sum(terms) / len(terms) if terms else score.sum() * 0.

    @staticmethod
    def _v48_proxy_hits(score, iou, matched, gt_count):
        top1 = top3 = 0
        for gid in range(int(gt_count)):
            idx = (matched == gid).nonzero(as_tuple=False).squeeze(1)
            if not idx.numel():
                continue
            rank = idx[score[idx].argsort(descending=True)]
            top1 += int(iou[rank[0]] >= .75)
            top3 += int((iou[rank[:3]] >= .75).any())
        return top1, top3

    @staticmethod
    def _v48_coverage(boxes, gt, threshold):
        """Number of GT boxes covered by at least one candidate."""
        if boxes.numel() == 0 or gt.numel() == 0:
            return 0
        return int((bbox_overlaps(boxes[:, :4], gt).max(dim=0)[0]
                    >= threshold).sum().item())

    @staticmethod
    def _v48_greedy_detection_counts(det, gt, threshold):
        """COCO-like score-ordered one-to-one TP/FP/FN diagnostic."""
        if not len(det):
            return 0, 0, int(len(gt)), 0.0
        if not len(gt):
            return 0, int(len(det)), 0, 0.0
        order = det[:, 4].argsort(descending=True)
        overlap = bbox_overlaps(det[order, :4], gt)
        used = torch.zeros(len(gt), dtype=torch.bool, device=gt.device)
        tp, matched_iou = 0, []
        for row in overlap:
            available = row.masked_fill(used, -1.)
            value, index = available.max(dim=0)
            if float(value) >= threshold:
                used[index] = True
                tp += 1
                matched_iou.append(value)
        fp = int(len(det)) - tp
        fn = int(len(gt)) - tp
        mean_iou = (float(torch.stack(matched_iou).mean())
                    if matched_iou else 0.0)
        return tp, fp, fn, mean_iou

    @staticmethod
    def _v48_spearman(score, target):
        if score.numel() < 3 or float(target.std()) == 0.:
            return score.new_zeros(())
        rx = score.argsort().argsort().float()
        ry = target.argsort().argsort().float()
        return ((rx - rx.mean()) * (ry - ry.mean())).mean() / (
            rx.std(unbiased=False) * ry.std(unbiased=False)).clamp_min(1e-6)

    def _v48_original_h2_boxes(self, feats, image_index, boxes, fg_scores,
                                meta, detail_img):
        """Rebuild precisely the V45 H2 final boxes without a second gradient."""
        index = self._v45_select(boxes, fg_scores)
        updated = boxes.clone()
        if index.numel():
            with torch.no_grad():
                delta, gate, _ = self._v45_run(
                    feats, image_index, boxes[index], meta, detail_img=detail_img)
                hard = (gate.sigmoid() >= self.side_refine_gate_threshold).to(delta.dtype)
                refined = self._v45_decode(
                    boxes[index], delta * hard,
                    int(meta['img_shape'][0]), int(meta['img_shape'][1]))
            updated[index] = refined
        return updated, index

    def _v48_score_train(self, feats, cls_scores, bbox_preds, img_metas,
                          gt_bboxes, detail_img=None):
        if not self.v48_iacs or int(self._nam_epoch) + 1 < self.side_refine_start_epoch:
            return {}
        detail_img = self._v45_detail_tensor(detail_img)
        raw = self.bbox_head.get_bboxes(cls_scores, bbox_preds, img_metas,
                                        cfg=self.bbox_head.test_cfg,
                                        rescale=False, with_nms=False)
        iacs_losses, rank_losses, uq_losses = [], [], []
        bins = [0, 0, 0, 0]; correlations = []; proxy1 = proxy3 = proxy_gt = 0
        nms = dict(
            gt=0, candidates=0, cand50=0, cand75=0, cand_low=0,
            cand_bg=0, cand_iou_sum=0.0, score_pos_sum=0.0,
            score_pos_n=0, score_neg_sum=0.0, score_neg_n=0,
            kept=0, tp50=0, fp50=0, fn50=0, tp75=0, fp75=0,
            fn75=0, tp50_iou_sum=0.0, tp75_iou_sum=0.0,
            actual50=0, actual75=0, oracle50=0, oracle75=0,
            lost_gt75=0, vote_up=0, vote_down=0)
        sigma_values = []
        for i, ((boxes, scores), gt, meta) in enumerate(zip(raw, gt_bboxes, img_metas)):
            if not len(boxes) or not len(gt):
                continue
            fg = scores[:, :self.bbox_head.num_classes]
            updated, h2_index = self._v48_original_h2_boxes(
                feats, i, boxes, fg, meta, detail_img)
            overlap = bbox_overlaps(updated.detach(), gt)
            iou, matched = overlap.max(dim=1)
            sample = self._v48_balanced_indices(iou, fg.max(dim=1)[0])
            if not sample.numel():
                continue
            logits = self._v48_logits(feats, i, updated, meta)
            iacs_logit, score, sigma = self._v48_score(logits, fg, updated)
            target = iou[sample]
            # Candidates below 0.1 IoU are explicit background.  Keeping a
            # small positive target for them contradicts QFL's zero-negative
            # contract and makes high-score false positives harder to reject.
            qfl_target = torch.where(target >= .10, target,
                                     target.new_zeros(target.shape))
            iacs_losses.append(self._v48_qfl(
                iacs_logit[sample], qfl_target))
            sigma_values.append(sigma[sample].detach())
            bins[0] += int((target >= .75).sum()); bins[1] += int(((target >= .5) & (target < .75)).sum())
            bins[2] += int(((target >= .1) & (target < .5)).sum()); bins[3] += int((target < .1).sum())
            correlations.append(self._v48_spearman(score[sample].detach(), target.detach()))
            h1, h3 = self._v48_proxy_hits(score[sample].detach(), target.detach(),
                                           matched[sample], len(gt))
            proxy1 += h1; proxy3 += h3; proxy_gt += int(len(gt))
            if self.v48_rank_loss:
                rank_losses.append(self._v48_pairwise_loss(score[sample], target, matched[sample]))
            if self.v48_uncertainty and h2_index.numel():
                h2_overlap = bbox_overlaps(boxes[h2_index], gt)
                _, h2_match = h2_overlap.max(dim=1)
                valid = h2_overlap.max(dim=1)[0] >= self.side_refine_min_iou
                if valid.any():
                    hidx = h2_index[valid]
                    # Uncertainty describes the boundary error of the actual
                    # gate-applied box handed to NMS, not the unused full H2
                    # residual before its four hard gates.
                    final_error = self._v45_target(
                        updated[hidx], gt[h2_match[valid]]).clamp(
                            -self.side_refiner.max_ratio,
                            self.side_refiner.max_ratio)
                    logvar = logits[hidx, 1:5].clamp(-6., 3.)
                    nll = .5 * (final_error.detach().pow(2)
                                * torch.exp(-logvar) + logvar)
                    uq_losses.append(nll.mean())
            # Keep a same-ranking, no-voting reference so the log reports
            # whether coordinate voting helped rather than merely counting
            # how many clusters were processed.
            if self.v48_variance_voting:
                if self.v48_softnms:
                    base_det, _, _ = self._v48_soft_nms(
                        updated, score.detach(), fg,
                        self.bbox_head.test_cfg.score_thr,
                        self.bbox_head.test_cfg.max_per_img)
                else:
                    base_det, _, _ = self._v48_class_agnostic_hard_nms(
                        updated, score.detach(), fg,
                        self.bbox_head.test_cfg.score_thr,
                        self.bbox_head.test_cfg.nms,
                        self.bbox_head.test_cfg.max_per_img)
            else:
                base_det = None
            det, _, _ = self._v48_final_nms(updated, score.detach(), fg,
                                             sigma.detach())
            if base_det is not None and len(det) == len(base_det) and len(det):
                before = bbox_overlaps(base_det[:, :4], gt).max(dim=1)[0]
                after = bbox_overlaps(det[:, :4], gt).max(dim=1)[0]
                nms['vote_up'] += int((after > before + 1e-4).sum())
                nms['vote_down'] += int((after + 1e-4 < before).sum())
            oracle, _, _ = self._v48_final_nms(updated, iou.detach(), fg,
                                                sigma.detach())
            candidate_count = int(len(updated))
            positive50 = iou >= .50
            positive75 = iou >= .75
            low = (iou >= .10) & (iou < .50)
            background = iou < .10
            nms['candidates'] += candidate_count
            nms['cand50'] += int(positive50.sum())
            nms['cand75'] += int(positive75.sum())
            nms['cand_low'] += int(low.sum())
            nms['cand_bg'] += int(background.sum())
            nms['cand_iou_sum'] += float(iou.sum())
            nms['score_pos_sum'] += float(score[positive50].sum())
            nms['score_pos_n'] += int(positive50.sum())
            nms['score_neg_sum'] += float(score[~positive50].sum())
            nms['score_neg_n'] += int((~positive50).sum())
            nms['kept'] += int(len(det))
            tp50, fp50, fn50, tp50_iou = self._v48_greedy_detection_counts(
                det.detach(), gt, .50)
            tp75, fp75, fn75, tp75_iou = self._v48_greedy_detection_counts(
                det.detach(), gt, .75)
            nms['tp50'] += tp50; nms['fp50'] += fp50; nms['fn50'] += fn50
            nms['tp75'] += tp75; nms['fp75'] += fp75; nms['fn75'] += fn75
            nms['tp50_iou_sum'] += tp50_iou * tp50
            nms['tp75_iou_sum'] += tp75_iou * tp75
            nms['gt'] += int(len(gt))
            for threshold, key in ((.50, '50'), (.75, '75')):
                nms['actual' + key] += self._v48_coverage(det, gt, threshold)
                nms['oracle' + key] += self._v48_coverage(oracle, gt, threshold)
            pre_gt75 = self._v48_coverage(updated, gt, .75)
            post_gt75 = self._v48_coverage(det, gt, .75)
            nms['lost_gt75'] += max(pre_gt75 - post_gt75, 0)
        if not iacs_losses:
            return {}
        mean_sigma = torch.cat(sigma_values).mean(0) if sigma_values else torch.zeros(4)
        self._v48_debug = dict(bins=bins, spearman=float(torch.stack(correlations).mean()) if correlations else 0.,
                                top1=proxy1 / max(proxy_gt, 1), top3=proxy3 / max(proxy_gt, 1),
                                sigma=[float(v) for v in mean_sigma], nms=nms)
        step = int(getattr(self.bbox_head, '_debug_step', 0))
        if self.side_refine_debug_interval and step % self.side_refine_debug_interval == 0:
            denom = max(nms['gt'], 1)
            candidate_denom = max(nms['candidates'], 1)
            positive_denom = max(nms['score_pos_n'], 1)
            negative_denom = max(nms['score_neg_n'], 1)
            _emit_training_debug(
                '[V48-CAND] iter={} total={} geo50/75={}/{} low/bg={}/{} '
                'IoUmean={:.3f} scoreTP50/FP50={:.3f}/{:.3f} bins=H/M/L/B={}/{}/{}/{}'.format(
                    max(0, step - 1), nms['candidates'], nms['cand50'],
                    nms['cand75'], nms['cand_low'], nms['cand_bg'],
                    nms['cand_iou_sum']/candidate_denom,
                    nms['score_pos_sum']/positive_denom,
                    nms['score_neg_sum']/negative_denom, *bins))
            _emit_training_debug(
                '[V48-NMS] iter={} kept={} TP/FP/FN50={}/{}/{} '
                'TP/FP/FN75={}/{}/{} TP-IoU50/75={:.3f}/{:.3f} '
                'lostGT75={} sp={:.3f} TP75@1/@3={:.3f}/{:.3f} '
                'R50 a/o={:.3f}/{:.3f} R75 a/o={:.3f}/{:.3f} '
                'vote+/-={}/{} sigma={}'.format(
                    max(0, step - 1), nms['kept'],
                    nms['tp50'], nms['fp50'], nms['fn50'],
                    nms['tp75'], nms['fp75'], nms['fn75'],
                    nms['tp50_iou_sum']/max(nms['tp50'], 1),
                    nms['tp75_iou_sum']/max(nms['tp75'], 1),
                    nms['lost_gt75'], self._v48_debug['spearman'],
                    self._v48_debug['top1'], self._v48_debug['top3'],
                    nms['actual50']/denom, nms['oracle50']/denom,
                    nms['actual75']/denom, nms['oracle75']/denom,
                    nms['vote_up'], nms['vote_down'],
                    '/'.join('{:.3f}'.format(v) for v in self._v48_debug['sigma'])))
        output = {'loss_v48_iacs': self.v48_iacs_loss_weight * sum(iacs_losses) / len(iacs_losses)}
        if rank_losses:
            output['loss_v48_rank'] = self.v48_rank_loss_weight * sum(rank_losses) / len(rank_losses)
        if uq_losses:
            output['loss_v48_uncertainty'] = self.v48_uncertainty_loss_weight * sum(uq_losses) / len(uq_losses)
        return output

    def _v45_side_refine_train(self, feats, cls_scores, bbox_preds, img_metas,
                               gt_bboxes, detail_img=None):
        # V45 remains the sole owner of H2 geometry/gate gradients.
        output = super()._v45_side_refine_train(
            feats, cls_scores, bbox_preds, img_metas, gt_bboxes,
            detail_img=detail_img)
        output.update(self._v48_score_train(feats, cls_scores, bbox_preds,
                                             img_metas, gt_bboxes,
                                             detail_img=detail_img))
        return output

    @staticmethod
    def _v48_class_agnostic_hard_nms(boxes, score, fg_scores, score_thr, nms_cfg, max_per_img):
        merged = torch.stack((score, score.new_zeros(score.shape)), dim=1)
        det, _, keep = multiclass_nms(boxes, merged, score_thr, nms_cfg,
                                      max_per_img, return_inds=True)
        labels = fg_scores[keep].argmax(dim=1) if len(keep) else keep
        return det, labels, keep

    def _v48_soft_nms(self, boxes, score, fg_scores, score_thr, max_per_img):
        """Proper Gaussian Soft-NMS: re-sort remaining candidates after decay."""
        work = score.clone()
        active = (work > score_thr).nonzero(as_tuple=False).squeeze(1)
        keep = []
        while active.numel() and len(keep) < max_per_img:
            active = active[work[active].argsort(descending=True)]
            winner, rest = active[0], active[1:]
            keep.append(int(winner))
            if not rest.numel():
                break
            overlap = bbox_overlaps(boxes[winner:winner + 1], boxes[rest])[0]
            work[rest] *= torch.exp(-overlap.square() / self.v48_softnms_sigma)
            active = rest[work[rest] > self.v48_softnms_min_score]
        keep = boxes.new_tensor(keep, dtype=torch.long)
        det = torch.cat((boxes[keep], work[keep, None]), dim=1) if keep.numel() else boxes.new_zeros((0, 5))
        labels = fg_scores[keep].argmax(1) if keep.numel() else keep
        return det, labels, keep

    def _v48_apply_voting(self, det, keep, boxes, score, sigma):
        if not self.v48_variance_voting or not keep.numel():
            return det, 0, 0
        voted = det.clone(); improved = harmed = 0
        for pos, idx in enumerate(keep):
            idx_int = int(idx.item())
            overlap = bbox_overlaps(boxes[idx_int:idx_int + 1], boxes)[0]
            width = (boxes[:, 2] - boxes[:, 0]).clamp_min(2.)
            height = (boxes[:, 3] - boxes[:, 1]).clamp_min(2.)
            cx = (boxes[:, 0] + boxes[:, 2]) * .5
            cy = (boxes[:, 1] + boxes[:, 3]) * .5
            centre_distance = torch.sqrt(
                ((cx - cx[idx_int]) / width[idx_int]).pow(2)
                + ((cy - cy[idx_int]) / height[idx_int]).pow(2))
            peers = ((overlap >= self.v48_vote_iou)
                     & (centre_distance <= self.v48_vote_center)).nonzero(
                         as_tuple=False).squeeze(1)
            peers = peers[score[peers].argsort(descending=True)[:self.v48_vote_topk]]
            # one variance per side; score prevents a low-confidence distant box
            # from dominating merely because its variance is underestimated.
            peer_scale = torch.stack(
                [width[peers], height[peers], width[peers], height[peers]],
                dim=1)
            sigma_px = sigma[peers] * peer_scale
            weight = (score[peers, None].clamp_min(1e-4)
                      / sigma_px.square().clamp_min(1e-4))
            fused = (boxes[peers] * weight).sum(0) / weight.sum(0)
            if bool((fused[2] > fused[0]).item()) and bool((fused[3] > fused[1]).item()):
                voted[pos, :4] = fused
        return voted, improved, harmed

    def _v48_final_nms(self, boxes, score, fg_scores, sigma):
        if self.v48_softnms:
            det, labels, keep = self._v48_soft_nms(
                boxes, score, fg_scores, self.bbox_head.test_cfg.score_thr,
                self.bbox_head.test_cfg.max_per_img)
        else:
            det, labels, keep = self._v48_class_agnostic_hard_nms(
                boxes, score, fg_scores, self.bbox_head.test_cfg.score_thr,
                self.bbox_head.test_cfg.nms, self.bbox_head.test_cfg.max_per_img)
        det, _, _ = self._v48_apply_voting(det, keep, boxes, score, sigma)
        return det, labels, keep

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        if not self.v48_iacs:
            return super().simple_test(img, img_metas, rescale=rescale, **kwargs)
        detail_img = self._v45_detail_tensor(kwargs.get('detail_img', None))
        batch_shape = tuple(int(v) for v in img.shape[-2:])
        for meta in img_metas:
            meta['batch_input_shape'] = batch_shape
        self._collect_roi_debug(img_metas)
        backbone_feats = self.backbone(img)
        feats, c2_feat = self._build_pyramid(backbone_feats, lbp_lab=None)
        lung_masks = self._collect_lung_masks(img_metas, device=img.device,
                                              batch_input_shape=batch_shape)
        outs = self.bbox_head.forward(
            feats, nam_graphs=None, lbp_lab=None, c2_feat=c2_feat,
            region_feat=None, img_shapes=[batch_shape for _ in img_metas],
            lung_masks=lung_masks,
            reg_detail_feats=getattr(self.neck, 'regression_detail_feats', None))
        raw = self.bbox_head.get_bboxes(outs[0], outs[1], img_metas,
                                        cfg=self.bbox_head.test_cfg,
                                        rescale=False, with_nms=False)
        results = []
        for i, ((boxes, scores), meta) in enumerate(zip(raw, img_metas)):
            fg = scores[:, :self.bbox_head.num_classes]
            updated, _ = self._v48_original_h2_boxes(feats, i, boxes, fg, meta, detail_img)
            logits = self._v48_logits(feats, i, updated, meta)
            _, score, sigma = self._v48_score(logits, fg, updated)
            det, labels, _ = self._v48_final_nms(updated, score, fg, sigma)
            if rescale and det.numel():
                det[:, :4] /= det.new_tensor(meta['scale_factor'])
            results.append((det, labels))
        from mmdet.core import bbox2result
        return [bbox2result(det, labels, self.bbox_head.num_classes)
                for det, labels in results]


@HOOKS.register_module(force=True)
class NAMV48DebugHook(NAMV45DebugHook):
    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        runner.logger.info(
            '[V48-BUILD] id=%s iacs=%s rank=%s uncertainty=%s voting=%s softnms=%s '
            'gate=original_h2_per_side candidates/bin=%d from0=true',
            model.integrated_build_id, model.v48_iacs, model.v48_rank_loss,
            model.v48_uncertainty, model.v48_variance_voting, model.v48_softnms,
            model.v48_candidates_per_bin)

    def after_train_iter(self, runner):
        # Intentionally suppress inherited legacy gradient diagnostics.  V48
        # emits candidate/NMS evidence directly from the real training path.
        pass

    def after_epoch(self, runner):
        # Preserve epoch scheduling without printing unrelated NAM branches.
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if hasattr(model, '_nam_epoch'):
            model._nam_epoch = runner.epoch + 1
