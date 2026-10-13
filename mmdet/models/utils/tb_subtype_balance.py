"""Subtype-balanced Sym heads for the two-class TBX11K detector.

The original RetinaNet head emits independent sigmoid logits for Active and
Obsolete.  This file provides two controlled alternatives while preserving
the SymFormer feature path and the existing bbox-regression path:

* ``EqualizedFocalLossTB`` is an EFL-style replacement for the original
  dense sigmoid focal loss.
* ``RetinaNAMGuideAttHeadSeesawSubtype`` separates TB/background objectness
  from Active/Obsolete classification.  The latter is supervised only on
  matched positive anchors and uses the Seesaw CE rule.

Neither component uses selector/quality-network logits.
"""
import math
import json
import os.path as osp

import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import HOOKS, Hook, force_fp32

from mmdet.core import bbox_overlaps
from mmdet.models.builder import HEADS, LOSSES, DETECTORS
from mmdet.models.losses.seesaw_loss import seesaw_ce_loss

from .NAMLab_att_v3 import RetinaNAMGuideAttHeadV44, RetinaNAMAttV44


def _sum_across_workers(value):
    value = value.clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value)
    return value


def _add_stat(owner, key, value):
    value = value.detach()
    if not hasattr(owner, 'epoch_stats'):
        owner.epoch_stats = {}
    if key not in owner.epoch_stats:
        owner.epoch_stats[key] = value.clone()
    else:
        owner.epoch_stats[key].add_(value)


def _watch_gradient(owner, logits, labels, weights):
    """Record actual backward gradients, split by their target source."""
    if not logits.requires_grad:
        return
    classes = logits.shape[1]
    valid = weights.detach() > 0
    positive = _one_hot(labels.long(), classes, logits.dtype).bool() & valid[:, None]
    other = (~positive) & valid[:, None] & (labels < classes)[:, None]
    background = valid[:, None] & (labels == classes)[:, None]

    def capture(gradient):
        magnitude = gradient.detach().abs()
        parts = torch.stack([(magnitude * mask).sum(0)
                             for mask in (positive, other, background)])
        _add_stat(owner, 'gradient_pos_other_background', parts)
        if isinstance(owner, EqualizedFocalLossTB):
            owner.pending_gradient.add_(parts)
        return gradient
    logits.register_hook(capture)


def _one_hot(labels, num_classes, dtype):
    return F.one_hot(labels.clamp(0, num_classes), num_classes + 1)[
        :, :num_classes].to(dtype=dtype)


@LOSSES.register_module(force=True)
class EqualizedFocalLossTB(nn.Module):
    """EFL-style dense sigmoid focal loss with per-class gradient history.

    Actual loss gradients are collected across every pyramid level during
    backward. The runner hook commits cumulative statistics ONCE per step;
    all levels in the next forward therefore share the same gamma.
    """

    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25,
                 scale=4.0, loss_weight=1.0,
                 reduction='mean', eps=1e-6):
        super().__init__()
        if not use_sigmoid:
            raise ValueError('EqualizedFocalLossTB requires sigmoid logits')
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.scale = float(scale)
        if gamma <= 0 or scale < 0 or not 0 <= alpha <= 1:
            raise ValueError('Invalid EFL gamma/scale/alpha')
        self.loss_weight = float(loss_weight)
        self.reduction = str(reduction)
        self.eps = float(eps)
        self.register_buffer('positive_gradient', torch.zeros(2))
        self.register_buffer('negative_gradient', torch.zeros(2))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('pending_gradient', torch.zeros(3, 2), persistent=False)
        self.epoch_stats = {}
        self.last_stats = {}

    def _ensure_classes(self, count, device):
        if self.positive_gradient.numel() == count:
            return
        raise ValueError('This experiment requires exactly two classes')

    @torch.no_grad()
    def commit_gradients(self):
        parts = _sum_across_workers(self.pending_gradient)
        self.positive_gradient.add_(parts[0])
        self.negative_gradient.add_(parts[1] + parts[2])
        self.pending_gradient.zero_()
        self.updates.add_(1)

    def factors(self):
        ratio = torch.where(self.updates > 0,
            self.positive_gradient / self.negative_gradient.clamp_min(self.eps),
            torch.ones_like(self.positive_gradient)).clamp(0., 1.)
        gamma = self.gamma + self.scale * (1. - ratio)
        return ratio, gamma, gamma / self.gamma

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        classes = pred.shape[-1]
        self._ensure_classes(classes, pred.device)
        target_onehot = _one_hot(target.long(), classes, pred.dtype)
        probability = pred.sigmoid()
        valid = (weight.reshape(-1, 1).to(pred.dtype)
                 if weight is not None else torch.ones_like(pred[:, :1]))
        _, dynamic_gamma, dynamic_weight = self.factors()
        if self.training:
            _watch_gradient(self, pred, target, valid[:, 0])
        pt = (1. - probability) * target_onehot + probability * (1. - target_onehot)
        alpha = self.alpha * target_onehot + (1. - self.alpha) * (1. - target_onehot)
        loss = F.binary_cross_entropy_with_logits(pred, target_onehot,
                                                  reduction='none')
        loss = loss * alpha * pt.pow(dynamic_gamma[None]) * dynamic_weight[None]
        loss = loss * valid
        loss = self.loss_weight * loss
        reduction = reduction_override or self.reduction
        if reduction == 'none':
            return loss
        if reduction == 'sum':
            return loss.sum()
        if reduction != 'mean':
            raise ValueError('Unknown reduction: ' + reduction)
        if avg_factor is None:
            return loss.mean() if loss.numel() else loss.sum()
        return loss.sum() / max(float(avg_factor), self.eps)


@HEADS.register_module(force=True)
class RetinaNAMGuideAttHeadSubtype(RetinaNAMGuideAttHeadV44):
    """Original head with dense normalisation and accumulated diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AnchorHead recognises FocalLoss by STRING, not its sigmoid property.
        # Custom EFL must keep positive-anchor normalisation for bbox and cls.
        self.sampling = False
        if hasattr(self, 'sampler') and type(self.sampler).__name__ != 'PseudoSampler':
            raise ValueError('Subtype experiments require PseudoSampler')
        self.epoch_stats = {}
        self.diagnostic_rows = None

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        if self.training:
            flat = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            target, weight = labels.reshape(-1), label_weights.reshape(-1)
            valid = weight > 0
            with torch.no_grad():
                _add_stat(self, 'anchors_active_obsolete_background',
                    torch.bincount(target[valid], minlength=3))
                pos = valid & (target < self.num_classes)
                confusion = torch.bincount(
                    target[pos] * 2 + flat.detach()[pos].argmax(1), minlength=4).reshape(2, 2)
                _add_stat(self, 'anchor_confusion', confusion)
            # Register on the tensor actually used by the parent loss. View
            # gradients on cls_score avoid creating a second, unused graph.
            if not isinstance(self, RetinaNAMGuideAttHeadSeesawSubtype):
                def capture(gradient):
                    g = gradient.permute(0, 2, 3, 1).reshape(-1, 2).detach().abs()
                    onehot = _one_hot(target, 2, g.dtype).bool()
                    masks = (onehot & valid[:, None],
                        ~onehot & pos[:, None], (valid & ~pos)[:, None])
                    _add_stat(self, 'gradient_pos_other_background',
                              torch.stack([(g * mask).sum(0) for mask in masks]))
                    return gradient
                cls_score.register_hook(capture)
        return super().loss_single(cls_score, bbox_pred, anchors, labels,
            label_weights, bbox_targets, bbox_weights, num_total_samples)

    @torch.no_grad()
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg=None,
                   rescale=False, with_nms=True):
        result = super().get_bboxes(cls_scores, bbox_preds, img_metas,
            cfg=cfg, rescale=rescale, with_nms=with_nms)
        if with_nms and not self.training and self.diagnostic_rows is not None:
            raw = super().get_bboxes(cls_scores, bbox_preds, img_metas,
                cfg=cfg, rescale=rescale, with_nms=False)
            for (boxes, scores), (final, labels), meta in zip(raw, result, img_metas):
                scale = boxes.new_tensor(meta['scale_factor']) if not rescale else 1.
                self.diagnostic_rows.append(dict(
                    filename=meta.get('ori_filename', meta.get('filename', '')),
                    boxes=(boxes / scale).cpu(), scores=scores[:, :2].cpu(),
                    final=(final[:, :4] / scale).cpu(), labels=labels.cpu()))
        return result

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
             gt_bboxes_ignore=None):
        losses = super().loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
            img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        interval = self.nms_debug_interval
        if self.training and interval > 0 and self._debug_step % interval == 0:
            with torch.no_grad():
                raw = super().get_bboxes(cls_scores, bbox_preds, img_metas,
                                         rescale=False, with_nms=False)
                final = super().get_bboxes(cls_scores, bbox_preds, img_metas,
                                           rescale=False, with_nms=True)
                summary = {}
                for (boxes, scores), (det, predicted), gt, labels in zip(
                        raw, final, gt_bboxes, gt_labels):
                    _stage_diagnostics(summary, boxes, scores[:, :2], det[:, :4],
                        predicted, gt, labels, self.test_cfg.score_thr)
                import logging
                logging.getLogger('mmdet').info(
                    '[SUBTYPE-MATCH] split=train sampledBatch=True '
                    'GTbestIoU=True thresholds=.50/.75/.85 stats=%s', summary)
        return losses


def _stage_diagnostics(summary, boxes, scores, final, final_labels,
                       gt, gt_labels, threshold):
    """Per-GT coverage/confusion; counts are not COCO one-to-one TP counts."""
    predicted = scores.argmax(1)
    keep = scores.max(1)[0] > threshold
    for name, candidates, labels in (
            ('raw', boxes, predicted), ('threshold', boxes[keep], predicted[keep]),
            ('final', final, final_labels)):
        state = summary.setdefault(name, dict(candidates=0, gt=[0, 0],
            covered=[[0, 0] for _ in range(3)],
            correct_subtype=[[0, 0] for _ in range(3)],
            confusion=[[[0, 0], [0, 0]] for _ in range(3)]))
        state['candidates'] += len(candidates)
        overlap = bbox_overlaps(candidates, gt)
        for g, target in enumerate(gt_labels.tolist()):
            state['gt'][target] += 1
            if not len(candidates):
                continue
            iou = overlap[:, g]
            best = iou.argmax()
            for t, cutoff in enumerate((.5, .75, .85)):
                if iou[best] >= cutoff:
                    state['covered'][t][target] += 1
                    state['confusion'][t][target][int(labels[best])] += 1
                state['correct_subtype'][t][target] += int(
                    ((iou >= cutoff) & (labels == target)).any())


@DETECTORS.register_module(force=True)
class RetinaNAMSubtypeDetector(RetinaNAMAttV44):
    """Original detector plus read-only validation subtype diagnostics."""

    def v50_begin_oracle(self, epoch):
        self.bbox_head.diagnostic_rows = []

    def v50_finish_oracle(self, dataset, runner):
        rows = self.bbox_head.diagnostic_rows
        self.bbox_head.diagnostic_rows = None
        if rows is None or len(rows) != len(dataset):
            raise RuntimeError('Subtype diagnostics dataset/cache length mismatch')
        summary = {}
        for index, item in enumerate(rows):
            info = dataset.data_infos[index]
            expected = osp.basename(info.get('filename', info.get('file_name', '')))
            if expected and osp.basename(item['filename']) != expected:
                raise RuntimeError('Subtype diagnostics image order mismatch')
            ann = dataset.get_ann_info(index)
            gt = torch.as_tensor(ann['bboxes'], dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(ann['labels'], dtype=torch.long)
            _stage_diagnostics(summary, item['boxes'], item['scores'], item['final'],
                item['labels'], gt, labels, self.bbox_head.test_cfg.score_thr)
        runner.logger.info('[SUBTYPE-MATCH] split=val allImages=True '
            'GTbestIoU=True thresholds=.50/.75/.85 stats=%s', summary)
        import mmcv
        mmcv.dump(summary, osp.join(runner.work_dir,
                  'subtype_stages_epoch_%02d.json' % (runner.epoch + 1)))


@HEADS.register_module(force=True)
class RetinaNAMGuideAttHeadSeesawSubtype(RetinaNAMGuideAttHeadSubtype):
    """Preserve Sym features while factorising TBness and subtype prediction.

    A separate TB logit and the original two-logit subtype predictor share
    the Sym classification tower. Background loss has no path to the subtype
    output layer. This is NOT an identity mapping of legacy sigmoid scores.
    """

    def __init__(self, *args, subtype_loss_weight=1.0,
                 objectness_loss_weight=1.0, objectness_gamma=2.0,
                 objectness_alpha=.25, seesaw_p=.8, seesaw_q=2.0,
                 **kwargs):
        if int(kwargs.get('num_classes', 2)) != 2:
            raise ValueError('Seesaw subtype head is defined for two TB subtypes')
        self.subtype_loss_weight = float(subtype_loss_weight)
        self.objectness_loss_weight = float(objectness_loss_weight)
        self.objectness_gamma = float(objectness_gamma)
        self.objectness_alpha = float(objectness_alpha)
        self.seesaw_p = float(seesaw_p)
        self.seesaw_q = float(seesaw_q)
        self._hier_features_current = None
        self._last_tb_logits = None
        self._last_subtype_logits = None
        self.subtype_stats = {}
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        # Preserve original subtype output weights; introduce an independent
        # objectness output, RNG-isolated from all legacy initialisations.
        with torch.random.fork_rng(devices=[]):
            self.tb_residual = nn.Conv2d(
                self.feat_channels, self.num_anchors, 3, padding=1)
        nn.init.zeros_(self.tb_residual.weight)
        nn.init.constant_(self.tb_residual.bias, math.log(.02 / .98))
        # RetinaGuideAttHead invokes this method twice during construction.
        # Keep one persistent counter and one capture hook across those calls.
        if not hasattr(self, 'subtype_cumulative'):
            self.register_buffer('subtype_cumulative',
                                 torch.zeros(self.num_classes))
        old_handle = getattr(self, '_retina_capture_handle', None)
        if old_handle is not None:
            old_handle.remove()
        self._retina_capture_handle = self.retina_cls.register_forward_pre_hook(
            self._capture_sym_class_feature)

    def init_weights(self):
        super().init_weights()
        nn.init.zeros_(self.tb_residual.weight)
        nn.init.constant_(self.tb_residual.bias, math.log(.02 / .98))

    def _capture_sym_class_feature(self, module, inputs):
        if self._hier_features_current is not None:
            self._hier_features_current.append(inputs[0])

    def _compose_scores(self, legacy_logits, cls_feature):
        batch, _, height, width = legacy_logits.shape
        legacy = legacy_logits.reshape(
            batch, self.num_anchors, self.num_classes, height, width)
        tb_logit = self.tb_residual(cls_feature)
        conditional_logit = legacy
        conditional_probability = conditional_logit.softmax(2)
        final_probability = (tb_logit.sigmoid().unsqueeze(2)
                             * conditional_probability).clamp(1e-6, 1.-1e-6)
        return (torch.logit(final_probability).reshape(
                    batch, self.num_anchors * self.num_classes, height, width),
                tb_logit, conditional_logit)

    def forward(self, feats, *args, **kwargs):
        self._hier_features_current = []
        try:
            output = super().forward(feats, *args, **kwargs)
            features = self._hier_features_current
        finally:
            self._hier_features_current = None
        cls_scores, bbox_preds, contrast_loss, boxalign_loss = output
        if len(features) != len(cls_scores):
            raise RuntimeError('Sym subtype cache must contain one map per FPN level')
        composed, tb_logits, subtype_logits = [], [], []
        for legacy, feature in zip(cls_scores, features):
            score, tb, subtype = self._compose_scores(legacy, feature)
            composed.append(score)
            tb_logits.append(tb)
            subtype_logits.append(subtype)
        self._last_tb_logits = tuple(tb_logits) if self.training else None
        self._last_subtype_logits = tuple(subtype_logits) if self.training else None
        return tuple(composed), bbox_preds, contrast_loss, boxalign_loss

    @staticmethod
    def _focal_binary(logits, target, weight, gamma, alpha, normalizer):
        probability = logits.sigmoid()
        pt = torch.where(target > 0, 1. - probability, probability)
        alpha_t = torch.where(target > 0, logits.new_tensor(alpha),
                              logits.new_tensor(1. - alpha))
        loss = F.binary_cross_entropy_with_logits(logits, target,
                                                  reduction='none')
        return (loss * alpha_t * pt.pow(gamma) * weight).sum() / max(normalizer, 1)

    def _hierarchical_loss(self, bbox_preds, gt_bboxes, gt_labels, img_metas,
                           gt_bboxes_ignore=None):
        tb_levels, subtype_levels = self._last_tb_logits, self._last_subtype_logits
        self._last_tb_logits, self._last_subtype_logits = None, None
        if tb_levels is None or subtype_levels is None:
            raise RuntimeError('Subtype loss requires a preceding training forward')
        sizes = [score.shape[-2:] for score in tb_levels]
        anchors, flags = self.get_anchors(sizes, img_metas, device=bbox_preds[0].device)
        targets = self.get_targets(
            anchors, flags, gt_bboxes, img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels,
            label_channels=self.cls_out_channels if self.use_sigmoid_cls else 1)
        if targets is None:
            zero = bbox_preds[0].sum() * 0.
            return [zero for _ in tb_levels], [zero for _ in tb_levels]
        labels, label_weights, _, _, total_positive, _ = targets
        object_losses, subtype_losses = [], []
        total_pos = 0
        subtype_count = torch.zeros_like(self.subtype_cumulative)
        for target, weight in zip(labels, label_weights):
            mask = (weight > 0) & (target >= 0) & (target < self.num_classes)
            subtype_count += torch.bincount(target[mask], minlength=2).to(subtype_count)
        if self.training:
            self.subtype_cumulative.add_(_sum_across_workers(subtype_count))
        cumulative = self.subtype_cumulative.detach().clone()
        object_pos_probability, object_neg_probability = [], []
        subtype_correct, subtype_total = 0, 0
        for level, (tb, subtype, level_labels, level_weights) in enumerate(
                zip(tb_levels, subtype_levels, labels, label_weights)):
            tb = tb.permute(0, 2, 3, 1).reshape(-1)
            subtype = subtype.float().permute(0, 3, 4, 1, 2).reshape(-1, self.num_classes)
            tb = tb.float()
            level_labels = level_labels.reshape(-1)
            level_weights = level_weights.reshape(-1).to(tb.dtype)
            valid = level_weights > 0
            positive = valid & (level_labels < self.num_classes)
            target = positive.to(tb.dtype)
            object_losses.append(self.objectness_loss_weight * self._focal_binary(
                tb, target, level_weights, self.objectness_gamma,
                self.objectness_alpha, total_positive))
            if positive.any():
                pos_labels = level_labels[positive]
                pos_weight = level_weights[positive]
                subtype_losses.append(self.subtype_loss_weight * seesaw_ce_loss(
                    subtype[positive], pos_labels, pos_weight,
                    cumulative, self.num_classes,
                    self.seesaw_p, self.seesaw_q, 1e-2,
                    reduction='mean', avg_factor=max(int(total_positive), 1)))
                subtype_correct += int((subtype[positive].detach().argmax(1)
                                        == pos_labels).sum())
                subtype_total += int(positive.sum())
            else:
                subtype_losses.append(subtype.sum() * 0.)
            _watch_gradient(self, subtype, level_labels,
                            level_weights * positive.to(level_weights.dtype))
            if tb.requires_grad:
                def record_object(gradient, positive=positive, valid=valid):
                    _add_stat(self, 'object_gradient_pos_background', torch.stack((
                        gradient[positive].abs().sum(),
                        gradient[valid & ~positive].abs().sum())))
                    return gradient
                tb.register_hook(record_object)
            if positive.any():
                object_pos_probability.append(tb[positive].detach().sigmoid())
            negative = valid & ~positive
            if negative.any():
                object_neg_probability.append(tb[negative].detach().sigmoid())
            total_pos += int(positive.sum())
        self.subtype_stats = dict(
            pos=int(total_pos), subtype_count=subtype_count.detach().cpu().tolist(),
            cumulative=self.subtype_cumulative.detach().cpu().tolist(),
            subtype_acc=float(subtype_correct / max(subtype_total, 1)),
            tb_pos=float(torch.cat(object_pos_probability).mean())
            if object_pos_probability else 0.,
            tb_neg=float(torch.cat(object_neg_probability).mean())
            if object_neg_probability else 0.)
        _add_stat(self, 'subtype_correct_total', tb.new_tensor([subtype_correct, subtype_total]))
        return object_losses, subtype_losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
             gt_bboxes_ignore=None):
        # Parent retains bbox, GIoU and optional D-FINE losses.  Its temporary
        # dense class loss is replaced before the runner consumes the graph.
        losses = super().loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        object_losses, subtype_losses = self._hierarchical_loss(
            bbox_preds, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        losses['loss_cls'] = object_losses
        losses['loss_subtype'] = subtype_losses
        return losses


@HOOKS.register_module(force=True)
class SubtypeBalanceDebugHook(Hook):
    """Compact gradient/score diagnostics for EFL and Seesaw subtype runs."""

    def __init__(self, interval=1):
        self.interval = int(interval)

    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = model.bbox_head
        if isinstance(head, RetinaNAMGuideAttHeadSeesawSubtype):
            runner.logger.info('[SUBTYPE-BUILD] mode=seesaw-hierarchical '
                'SymFeature=True objectness=allAnchors subtype=positiveAnchors '
                'p/q=%.2f/%.2f', head.seesaw_p, head.seesaw_q)
        else:
            runner.logger.info('[SUBTYPE-BUILD] mode=%s SymFeature=True '
                'classification=legacyTwoSigmoid sampling=%s',
                'EFL' if isinstance(head.loss_cls, EqualizedFocalLossTB) else 'Focal',
                head.sampling)

    def before_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.bbox_head.epoch_stats = {}
        if isinstance(model.bbox_head.loss_cls, EqualizedFocalLossTB):
            model.bbox_head.loss_cls.epoch_stats = {}

    def after_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        loss = model.bbox_head.loss_cls
        if isinstance(loss, EqualizedFocalLossTB):
            loss.commit_gradients()

    def after_train_epoch(self, runner):
        if (runner.epoch + 1) % self.interval:
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = model.bbox_head
        runner.logger.info('[SUBTYPE-EPOCH] epoch=%d allBatches=True '
            'classOrder=Active,Obsolete gradientRows=positive,otherSubtype,background stats=%s',
            runner.epoch + 1, {k: v.cpu().tolist() for k, v in head.epoch_stats.items()})
        if isinstance(head, RetinaNAMGuideAttHeadSeesawSubtype):
            stat = head.subtype_stats
            runner.logger.info('[SUBTYPE-TRAIN] mode=seesaw lastBatch=True pos=%d '
                'Active/Obsolete=%s cumulative=%s subtypeAcc=%.3f '
                'pTB(pos/neg)=%.3f/%.5f', stat.get('pos', 0),
                stat.get('subtype_count', []), stat.get('cumulative', []),
                stat.get('subtype_acc', 0.), stat.get('tb_pos', 0.),
                stat.get('tb_neg', 0.))
        elif isinstance(head.loss_cls, EqualizedFocalLossTB):
            loss = head.loss_cls
            ratio, gamma, weight = loss.factors()
            runner.logger.info('[SUBTYPE-TRAIN] mode=EFL pos/negGradRatio=%s '
                'gamma=%s weight=%s updates=%s', ratio.cpu().tolist(),
                gamma.cpu().tolist(), weight.cpu().tolist(), int(loss.updates))
