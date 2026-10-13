"""Horizontal-box adaptation of Pixel-level Quality Assessment (PQA).

The original method predicts one global Gaussian position heatmap and scores
each candidate with the volume IoU between the predicted heatmap and the
analytic Gaussian induced by that candidate.  This file contains only the
PQA math and small neural head, so it is independently CPU-testable.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PQAHeatmapHead(nn.Module):
    """Predict one class-agnostic position heatmap from P2 and P3."""

    def __init__(self, channels=256, hidden=64):
        super().__init__()
        self.p2 = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU())
        self.p3 = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU())
        # A residual gate keeps the initial representation P2-led while still
        # allowing P3 context to enter. It is not part of the original PQA.
        self.p3_gate = nn.Parameter(torch.tensor(-2.0))
        self.output = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.constant_(self.output.bias, -2.19)  # sigmoid ~= 0.10

    def forward(self, p2, p3):
        detail = self.p2(p2)
        context = F.interpolate(self.p3(p3), size=detail.shape[-2:],
                                mode='bilinear', align_corners=False)
        return self.output(detail + self.p3_gate.sigmoid() * context)


def feature_grid(height, width, image_shape, device, dtype):
    """Pixel-centre grid in the detector input coordinate system."""
    image_h, image_w = image_shape
    y = (torch.arange(height, device=device, dtype=dtype) + .5) * image_h / height
    x = (torch.arange(width, device=device, dtype=dtype) + .5) * image_w / width
    yy, xx = torch.meshgrid(y, x)
    return xx, yy


def gaussian_box_maps(boxes, xx, yy, sigma_divisor=4.0):
    """Analytic truncated Gaussian maps for horizontal xyxy boxes."""
    boxes = boxes.reshape(-1, 4)
    if not len(boxes):
        return xx.new_empty((0,) + tuple(xx.shape))
    x1, y1, x2, y2 = [value[:, None, None] for value in boxes.unbind(1)]
    width = (x2 - x1).clamp_min(1e-4)
    height = (y2 - y1).clamp_min(1e-4)
    cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
    sx, sy = width / sigma_divisor, height / sigma_divisor
    exponent = -.5 * (((xx[None] - cx) / sx).square() +
                      ((yy[None] - cy) / sy).square())
    inside = ((xx[None] >= x1) & (xx[None] <= x2) &
              (yy[None] >= y1) & (yy[None] <= y2))
    return exponent.clamp_min(-30.).exp() * inside.to(exponent.dtype)


def pqa_target(gt_boxes, map_shape, image_shape, device=None, dtype=None):
    """PQA Gaussian target: maximum response over all GT boxes."""
    if device is None:
        device = gt_boxes.device
    if dtype is None:
        dtype = gt_boxes.dtype
    xx, yy = feature_grid(map_shape[0], map_shape[1], image_shape,
                          device, dtype)
    maps = gaussian_box_maps(gt_boxes.to(device=device, dtype=dtype), xx, yy)
    if not len(maps):
        return xx.new_zeros((1,) + tuple(map_shape))
    return maps.max(0, keepdim=True)[0]


def pqa_location_loss(logits, target, alpha=.25, beta=2.):
    """Continuous-positive focal loss used for position-map learning."""
    target = target.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    positive = target > 0
    weight = torch.where(
        positive,
        alpha * (target - probability).abs(),
        (1. - alpha) * probability.pow(beta))
    return (weight * bce).sum() / target.sum().clamp_min(1.)


def pqa_scores(logits, boxes, image_shape, chunk=64, eps=1e-6):
    """Volume-IoU consistency score between predicted and box heatmaps."""
    if not len(boxes):
        return logits.new_empty((0,))
    probability = logits.sigmoid()[0, 0]
    xx, yy = feature_grid(probability.shape[0], probability.shape[1],
                          image_shape, probability.device, probability.dtype)
    outputs = []
    for candidate_chunk in boxes.to(probability.dtype).split(chunk):
        analytic = gaussian_box_maps(candidate_chunk, xx, yy)
        inside = analytic > 0
        predicted = probability[None] * inside.to(probability.dtype)
        numerator = torch.minimum(predicted, analytic).flatten(1).sum(1)
        denominator = torch.maximum(predicted, analytic).flatten(1).sum(1)
        outputs.append(numerator / denominator.clamp_min(eps))
    return torch.cat(outputs)


def pairwise_iou(boxes):
    left_top = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    right_bottom = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0.).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.).prod(-1)
    return intersection / (area[:, None] + area[None] - intersection).clamp_min(1e-8)


def pqa_competition_rank(scores, true_iou, matched_gt, boxes,
                         min_gap=.05, competition_iou=.50,
                         max_pairs_per_gt=128):
    """GT-balanced ranking of pairs that can suppress each other.

    This is the A2 project adaptation, not part of the original PQA method.
    """
    if len(scores) < 2:
        return scores.sum() * 0., 0, 0
    overlap = pairwise_iou(boxes)
    logits = torch.logit(scores.clamp(1e-5, 1.-1e-5))
    losses, pair_count, gt_count = [], 0, 0
    valid_gt = matched_gt[true_iou >= .50].unique()
    valid_gt = valid_gt[valid_gt >= 0]
    for gt_index in valid_gt:
        same = matched_gt == gt_index
        # Spend ranking capacity on AP75/AP85-relevant candidates. Low-IoU
        # ordering is already represented by the spatial-map loss.
        higher = same[:, None] & same[None] & (true_iou[:, None] >= .75)
        gap = true_iou[:, None] - true_iou[None]
        valid = higher & (gap > min_gap) & (overlap > competition_iou)
        row, col = valid.nonzero(as_tuple=True)
        if not len(row):
            continue
        violation = F.softplus(.5 * gap[row, col] -
                               (logits[row] - logits[col]))
        violation = violation * (1. + (true_iou[row] >= .85).to(
            violation.dtype))
        if len(violation) > max_pairs_per_gt:
            violation = violation.topk(max_pairs_per_gt).values
        losses.append(violation.mean())
        pair_count += len(violation)
        gt_count += 1
    if not losses:
        return scores.sum() * 0., 0, 0
    return torch.stack(losses).mean(), pair_count, gt_count
