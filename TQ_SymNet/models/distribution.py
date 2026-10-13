"""TQ-SymNet: extracted measured implementations; see PROVENANCE.json."""
import math
import logging
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
