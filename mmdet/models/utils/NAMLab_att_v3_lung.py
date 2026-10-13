"""Rollback-safe lung-mask and localization-quality ablations for NAM v3.19.

This module deliberately subclasses the strict V319/G2 profile.  The measured
baseline remains untouched in ``NAMLab_att_v3_best_slim.py``.  A single binary
lung mask is used only as a soft anatomical support; it never replaces, crops,
or zeros the raw CXR and it never changes positive-anchor supervision.
"""

import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox_overlaps
from mmdet.models.builder import DETECTORS, HEADS

from .NAMLab_att_v3 import _align_lung_mask_to_image_meta
from .NAMLab_att_v3_best_slim import (
    NAMLAB_SLIM_BUILD_ID, RetinaNAMAttV319, RetinaNAMGuideAttHeadV319)


NAMLAB_LUNG_BUILD_ID = 'v3.20-g2-lung-soft-quality-ablation-20260804'


def _soft_lung_support(mask, dilation=16, blur=9):
    """Return a differentiability-free soft support map in [0, 1]."""
    mask = (mask > 0.5).to(dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f'Expected BxHxW or Bx1xHxW mask, got {mask.shape}')
    dilation = int(dilation)
    blur = int(blur)
    if dilation < 0 or blur < 0:
        raise ValueError('lung-mask dilation and blur must be non-negative')
    if dilation:
        kernel = 2 * dilation + 1
        mask = F.max_pool2d(mask, kernel, stride=1, padding=dilation)
    if blur:
        kernel = 2 * blur + 1
        mask = F.avg_pool2d(mask, kernel, stride=1, padding=blur)
    return mask.clamp_(0.0, 1.0)


def _lung_coordinate_channels(lung_masks, output_size, dilation=16, blur=9):
    """Build support, within-mask x and within-mask y channels.

    The mask is a single foreground component, so no left/right instance label
    is invented.  Coordinates are normalized inside the foreground bounding
    extent and multiplied by soft support.  The existing Sym left/right prior
    remains unchanged.
    """
    support = _soft_lung_support(lung_masks, dilation=dilation, blur=blur)
    batch, _, height, width = support.shape
    hard = support[:, 0] > 0.05
    x_grid = support.new_zeros((batch, height, width))
    y_grid = support.new_zeros((batch, height, width))
    for batch_index in range(batch):
        ys, xs = torch.where(hard[batch_index])
        if ys.numel() == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        xs_all = torch.arange(width, device=support.device,
                              dtype=support.dtype)
        ys_all = torch.arange(height, device=support.device,
                              dtype=support.dtype)
        # [-1, 1]: continuous left/right and apex/base coordinates.
        x_grid[batch_index] = (
            2.0 * (xs_all - x0) / (x1 - x0).clamp_min(1) - 1.0
        )[None, :].expand(height, width)
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


def _rank_zero():
    return not dist.is_available() or not dist.is_initialized() \
        or dist.get_rank() == 0


@HEADS.register_module(force=True)
class RetinaNAMGuideAttHeadV320(RetinaNAMGuideAttHeadV319):
    """Strict G2 head plus independently switchable soft-mask/quality seams."""

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
            raise ValueError(
                'localization_quality requires the measured G2 FDR branch')
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
            # Exactly reproduce G2 at step zero while retaining gradients to
            # the projection weights through the small non-zero scale.
            for projection in self.lung_anatomy_proj:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
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
            # Per anchor: four maximum probabilities + four normalized
            # entropies from the FDR distribution (DGQP-style statistics).
            self.localization_quality_pred = nn.Conv2d(
                self.num_anchors * 8,
                self.num_anchors,
                kernel_size=1,
                groups=self.num_anchors)
            nn.init.zeros_(self.localization_quality_pred.weight)
            prior_logit = math.log(
                self.localization_quality_prior
                / (1.0 - self.localization_quality_prior))
            nn.init.constant_(self.localization_quality_pred.bias, prior_logit)
        else:
            self.localization_quality_pred = None

    def init_weights(self):
        """Restore zero-start guarantees after the parent's global init_cfg."""
        super().init_weights()
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

    def _require_lung_mask(self, img_meta):
        mask = img_meta.get('lung_mask') if isinstance(img_meta, dict) else None
        if mask is None:
            filename = (
                img_meta.get('filename', '<unknown>')
                if isinstance(img_meta, dict) else '<unknown>')
            raise KeyError(
                f'Missing lung_mask for {filename}; add LoadLungMask before '
                'Resize and collect lung_mask in img_metas.')
        return _align_lung_mask_to_image_meta(
            mask,
            img_meta,
            batch_input_shape=img_meta.get('batch_input_shape'))

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes,
                            gt_bboxes_ignore, gt_labels, img_meta,
                            label_channels=1, unmap_outputs=True):
        result = super()._get_targets_single(
            flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels,
            img_meta, label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        if not self.use_lung_negative_weighting or result[0] is None:
            return result

        labels, label_weights, bbox_targets, bbox_weights = result[:4]
        mask = self._require_lung_mask(img_meta).to(
            device=flat_anchors.device, dtype=torch.float32)
        support = _soft_lung_support(
            mask,
            dilation=self.lung_mask_dilation,
            blur=self.lung_mask_blur)[0, 0]
        anchors = flat_anchors
        if not unmap_outputs:
            # This branch is kept correct for non-default callers.
            from mmdet.core import anchor_inside_flags
            inside = anchor_inside_flags(
                flat_anchors, valid_flags, img_meta['img_shape'][:2],
                self.train_cfg.allowed_border)
            anchors = flat_anchors[inside]
        centers_x = ((anchors[:, 0] + anchors[:, 2]) * 0.5).long()
        centers_y = ((anchors[:, 1] + anchors[:, 3]) * 0.5).long()
        centers_x.clamp_(0, support.shape[1] - 1)
        centers_y.clamp_(0, support.shape[0] - 1)
        anchor_support = support[centers_y, centers_x]
        negative = (labels == self.num_classes) & (label_weights > 0)
        soft_weight = (
            self.lung_negative_far_weight
            + (1.0 - self.lung_negative_far_weight) * anchor_support)
        label_weights = label_weights.clone()
        label_weights[negative] *= soft_weight[negative]

        # Record once per image even though FDR obtains targets a second time.
        debug_key = id(img_meta)
        if debug_key not in self._lung_debug_seen:
            self._lung_debug_seen.add(debug_key)
            neg_support = anchor_support[negative]
            positive = (labels >= 0) & (labels < self.num_classes)
            pos_support = anchor_support[positive]
            coverages = []
            for box in gt_bboxes:
                x1 = int(torch.floor(box[0]).clamp(0, support.shape[1]).item())
                y1 = int(torch.floor(box[1]).clamp(0, support.shape[0]).item())
                x2 = int(torch.ceil(box[2]).clamp(0, support.shape[1]).item())
                y2 = int(torch.ceil(box[3]).clamp(0, support.shape[0]).item())
                if x2 > x1 and y2 > y1:
                    coverages.append(support[y1:y2, x1:x2].mean())
            self._lung_debug.setdefault('negative_support', []).append(
                neg_support.mean().detach() if neg_support.numel()
                else support.new_zeros(()))
            self._lung_debug.setdefault('negative_weight', []).append(
                soft_weight[negative].mean().detach() if negative.any()
                else support.new_zeros(()))
            self._lung_debug.setdefault('positive_support', []).append(
                pos_support.mean().detach() if pos_support.numel()
                else support.new_zeros(()))
            self._lung_debug.setdefault('gt_coverage', []).append(
                torch.stack(coverages).mean().detach() if coverages
                else support.new_zeros(()))
        return ((labels, label_weights, bbox_targets, bbox_weights)
                + result[4:])

    def _apply_anatomy_bias(self, feats, lung_masks):
        if not self.use_lung_anatomy_bias:
            return list(feats)
        if lung_masks is None:
            raise RuntimeError(
                'lung_anatomy_bias=True requires aligned lung_masks')
        output = list(feats)
        relative_changes = []
        scales = self.lung_anatomy_residual_max * torch.tanh(
            self.lung_anatomy_scale_raw)
        for level, feat in enumerate(output):
            if level not in self.lung_anatomy_levels:
                relative_changes.append(feat.new_zeros(()))
                continue
            anatomy = _lung_coordinate_channels(
                lung_masks, feat.shape[-2:],
                dilation=self.lung_mask_dilation,
                blur=self.lung_mask_blur).to(dtype=feat.dtype)
            delta = scales[level].to(dtype=feat.dtype) \
                * self.lung_anatomy_proj[level](anatomy)
            output[level] = feat + delta
            relative_changes.append(
                delta.detach().float().norm()
                / feat.detach().float().norm().clamp_min(1e-8))
        self._anatomy_debug = {
            'scales': [float(v.detach().float().item()) for v in scales],
            'delta_rel': [
                float(v.detach().float().item()) for v in relative_changes],
        }
        return output

    def _distribution_quality_logits(self, distribution_logits):
        batch, _, height, width = distribution_logits.shape
        bins = self.d_fine_reg_max + 1
        logits = distribution_logits.reshape(
            batch, self.num_anchors, 4, bins, height, width)
        probabilities = logits.float().softmax(dim=3)
        maximum = probabilities.amax(dim=3)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=3) / math.log(bins)
        statistics = torch.cat([maximum, entropy], dim=2).reshape(
            batch, self.num_anchors * 8, height, width)
        # Do not let the quality objective change the measured FDR regression
        # distribution; this ablation tests score/localization alignment only.
        return self.localization_quality_pred(statistics.detach())

    def forward_single(self, *args, **kwargs):
        output = super().forward_single(*args, **kwargs)
        if not self.use_localization_quality:
            return output
        cls_score, bbox_pred, feat1, fused, reg_feat, gate_stats, \
            bbox_base, distribution_logits = output
        quality_logits = self._distribution_quality_logits(
            distribution_logits)
        if self._quality_logits_current is not None:
            self._quality_logits_current.append(quality_logits)
        if not self.training:
            cls_probability = cls_score.float().sigmoid()
            quality = quality_logits.float().sigmoid().repeat_interleave(
                self.cls_out_channels, dim=1)
            # Normalize by the initialization prior: the first inference is
            # exactly the G2 score, while learned above/below-prior quality
            # raises/lowers a box for ranking and NMS.
            quality_factor = (
                quality / self.localization_quality_prior
            ).pow(self.localization_quality_alpha)
            calibrated = (
                cls_probability
                * quality_factor
            ).clamp(1e-6, 1.0 - 1e-6)
            cls_score = torch.logit(calibrated).to(dtype=cls_score.dtype)
        return (cls_score, bbox_pred, feat1, fused, reg_feat, gate_stats,
                bbox_base, distribution_logits)

    def forward(self, feats, *args, lung_masks=None, **kwargs):
        feats = self._apply_anatomy_bias(feats, lung_masks)
        self._quality_logits_current = [] if self.use_localization_quality \
            else None
        output = super().forward(
            feats, *args, lung_masks=lung_masks, **kwargs)
        self._last_quality_logits = (
            tuple(self._quality_logits_current)
            if self.training and self.use_localization_quality else None)
        self._quality_logits_current = None
        return output

    def _localization_quality_loss(self, bbox_preds, gt_bboxes, gt_labels,
                                   img_metas, gt_bboxes_ignore=None):
        quality_logits = self._last_quality_logits
        self._last_quality_logits = None
        if quality_logits is None:
            raise RuntimeError(
                'localization-quality loss requires a preceding train forward')
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
            return [prediction.sum() * 0.0 for prediction in quality_logits]
        _, _, bbox_targets, bbox_weights, num_total_pos, _ = targets
        losses = []
        predicted_quality = []
        target_quality = []
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
            raw = F.binary_cross_entropy_with_logits(
                q[positive].float(), iou_target, reduction='sum')
            losses.append(
                self.localization_quality_loss_weight
                * raw / max(int(num_total_pos), 1))
            predicted_quality.append(q[positive].detach().float().sigmoid())
            target_quality.append(iou_target.detach().float())
        if predicted_quality:
            predicted = torch.cat(predicted_quality)
            target = torch.cat(target_quality)
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

    def _emit_lung_debug(self):
        interval = int(getattr(self, 'debug_interval', 0))
        source_step = int(getattr(self, '_debug_step', 0)) - 1
        if interval <= 0 or source_step % interval != 0 or not _rank_zero():
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
            losses['loss_loc_quality'] = self._localization_quality_loss(
                bbox_preds, gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)
        self._emit_lung_debug()
        return losses


@DETECTORS.register_module(force=True)
class RetinaNAMAttV320(RetinaNAMAttV319):
    """V319 detector that collects a binary mask only when an ablation needs it."""

    def _collect_lung_masks(self, img_metas, device,
                            batch_input_shape=None):
        head_needs_mask = bool(
            getattr(self.bbox_head, 'use_lung_anatomy_bias', False))
        if not head_needs_mask:
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
        return torch.stack(aligned).to(device=device, dtype=torch.float32)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('experiment_tag', 'V320_G2_CONTROL')
        super().__init__(*args, **kwargs)
        if not isinstance(self.bbox_head, RetinaNAMGuideAttHeadV320):
            raise TypeError('RetinaNAMAttV320 requires the V320 head')
        logging.getLogger('mmdet').info(
            '[NAM-LUNG] id=%s base=%s neg=%s far_w=%.2f bias=%s levels=%s '
            'loc_q=%s q_alpha=%.2f q_w=%.2f',
            NAMLAB_LUNG_BUILD_ID, NAMLAB_SLIM_BUILD_ID,
            self.bbox_head.use_lung_negative_weighting,
            self.bbox_head.lung_negative_far_weight,
            self.bbox_head.use_lung_anatomy_bias,
            self.bbox_head.lung_anatomy_levels,
            self.bbox_head.use_localization_quality,
            self.bbox_head.localization_quality_alpha,
            self.bbox_head.localization_quality_loss_weight)


__all__ = [
    'NAMLAB_LUNG_BUILD_ID',
    'RetinaNAMGuideAttHeadV320',
    'RetinaNAMAttV320',
]
