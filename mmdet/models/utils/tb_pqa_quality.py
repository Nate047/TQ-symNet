"""Frozen Sym detector with PQA localization quality and Gaussian Soft-NMS."""
import hashlib
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.ops.nms import batched_nms
from mmcv.runner import HOOKS, Hook
from torch import nn

from mmdet.core import bbox2result, bbox_overlaps
from mmdet.models.builder import DETECTORS

from .NAMLab_att_v3 import RetinaNAMAttV44
from .tb_pqa import (PQAHeatmapHead, pqa_competition_rank,
                     pqa_location_loss, pqa_scores, pqa_target)


def _coverage(boxes, gt):
    if not len(boxes) or not len(gt):
        return [0, 0, 0]
    best = bbox_overlaps(boxes, gt).max(0)[0]
    return [int((best >= threshold).sum()) for threshold in (.50, .75, .85)]


def _bins(iou):
    return [int((iou < .10).sum()),
            int(((iou >= .10) & (iou < .50)).sum()),
            int(((iou >= .50) & (iou < .75)).sum()),
            int(((iou >= .75) & (iou < .85)).sum()),
            int((iou >= .85).sum())]


def _oracle_result(boxes, sym, gt, classes):
    output = [np.empty((0, 5), np.float32) for _ in range(classes)]
    if not len(boxes) or not len(gt):
        return output
    overlap = bbox_overlaps(boxes, gt)
    used_boxes, used_gt, selected = set(), set(), []
    for flat in overlap.flatten().argsort(descending=True).tolist():
        box_index, gt_index = divmod(flat, len(gt))
        if box_index not in used_boxes and gt_index not in used_gt:
            used_boxes.add(box_index)
            used_gt.add(gt_index)
            selected.append((box_index, float(overlap[box_index, gt_index])))
        if len(used_gt) == len(gt):
            break
    labels = sym[:, :-1].argmax(1)[[index for index, _ in selected]]
    detections = torch.stack([
        torch.cat((boxes[index], boxes.new_tensor([score])))
        for index, score in selected])
    return bbox2result(detections.numpy(), labels.numpy(), classes)


def _main_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.startswith('pqa_heatmap.'):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@DETECTORS.register_module()
class RetinaNAMPQA(RetinaNAMAttV44):
    """Independent PQA replacement for the V55 scalar qIoU network."""

    def __init__(self, *args, pqa_hidden=64, pqa_chunk=64,
                 pqa_rank_weight=0., pqa_score_mode='product',
                 pqa_soft_sigma=.70, pqa_soft_iou=.50,
                 pqa_soft_min_score=0., pqa_soft_topk=128, **kwargs):
        kwargs['local_refine'] = False
        super().__init__(*args, **kwargs)
        if pqa_score_mode not in ('sym', 'pqa', 'product'):
            raise ValueError('pqa_score_mode must be sym/pqa/product')
        if pqa_chunk < 1 or pqa_soft_topk < 1 or pqa_soft_sigma <= 0:
            raise ValueError('Invalid PQA chunk/Soft-NMS configuration')
        with torch.random.fork_rng(devices=[]):
            self.pqa_heatmap = PQAHeatmapHead(hidden=int(pqa_hidden))
        self.pqa_chunk = int(pqa_chunk)
        self.pqa_rank_weight = float(pqa_rank_weight)
        self.pqa_score_mode = pqa_score_mode
        self.pqa_soft_sigma = float(pqa_soft_sigma)
        self.pqa_soft_iou = float(pqa_soft_iou)
        self.pqa_soft_min_score = float(pqa_soft_min_score)
        self.pqa_soft_topk = int(pqa_soft_topk)
        # The requested experiments always freeze the complete detector.
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('pqa_heatmap.'))
        self.pqa_epoch = 0
        self.pqa_collect = False
        self.pqa_cache = []
        self.pqa_train_stats = []
        self.best_pqa_key = None
        self.integrated_build_id = 'v56-pqa-frozen-softnms-20260915'

    def train(self, mode=True):
        # Keep backbone, CrossScale, Sym head and all their stateful layers in
        # evaluation mode while allowing only the PQA head to train.
        nn.Module.train(self, False)
        self.training = mode
        self.pqa_heatmap.train(mode)
        return self

    def _frozen_main(self, img, metas):
        shape = tuple(img.shape[-2:])
        with torch.no_grad():
            features, c2 = self._build_pyramid(self.backbone(img), lbp_lab=None)
            outputs = self.bbox_head.forward(
                features, nam_graphs=None, lbp_lab=None, c2_feat=c2,
                region_feat=None, gt_bboxes=None, gt_labels=None,
                img_shapes=[shape] * len(metas),
                lung_masks=self._collect_lung_masks(
                    metas, img.device, shape),
                reg_detail_feats=getattr(
                    self.neck, 'regression_detail_feats', None))
            raw = self.bbox_head.get_bboxes(
                outputs[0], outputs[1], metas,
                cfg=self.bbox_head.test_cfg, rescale=False, with_nms=False)
        return [feature.detach() for feature in features], raw

    @staticmethod
    def _actual_iou(boxes, gt):
        if not len(gt):
            return (boxes.new_zeros(len(boxes)),
                    torch.full((len(boxes),), -1, dtype=torch.long,
                               device=boxes.device))
        return bbox_overlaps(boxes, gt).max(1)

    def _base_score(self, sym, quality):
        subtype = sym[:, :-1].argmax(1)
        tb_score = sym[:, :-1].max(1)[0]
        if self.pqa_score_mode == 'sym':
            score = tb_score
        elif self.pqa_score_mode == 'pqa':
            score = quality
        else:
            # This is Eq. (8) of PQA with max subtype confidence as C(B).
            score = tb_score * quality
        return score, subtype, tb_score

    def _soft_nms(self, boxes, score):
        if not len(boxes):
            return (boxes.new_empty((0, 5)),
                    boxes.new_empty((0,), dtype=torch.long))
        labels = torch.zeros(len(boxes), dtype=torch.long,
                             device=boxes.device)
        cfg = dict(type='soft_nms', method='gaussian',
                   iou_threshold=self.pqa_soft_iou,
                   sigma=self.pqa_soft_sigma,
                   min_score=self.pqa_soft_min_score)
        detections, keep = batched_nms(boxes, score, labels, cfg)
        limit = min(self.pqa_soft_topk, len(detections))
        return detections[:limit], keep[:limit]

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, **kwargs):
        if gt_bboxes_ignore is not None and any(len(v) for v in gt_bboxes_ignore):
            raise ValueError('PQA requires explicit treatment of ignored GT')
        features, raw = self._frozen_main(img, img_metas)
        logits = self.pqa_heatmap(features[0], features[1])
        image_shape = tuple(img.shape[-2:])
        location_losses, rank_losses = [], []
        stats = dict(location=0., images=0, rank=0., pairs=0, rank_gt=0,
                     candidates=0)
        for index, ((boxes, sym), gt) in enumerate(zip(raw, gt_bboxes)):
            target = pqa_target(gt, logits.shape[-2:], image_shape,
                                logits.device, logits.dtype)
            location = pqa_location_loss(logits[index:index + 1],
                                         target[None])
            location_losses.append(location)
            stats['location'] += float(location.detach())
            stats['images'] += 1
            if self.pqa_rank_weight <= 0:
                continue
            keep = (sym[:, :-1] > self.bbox_head.test_cfg.score_thr).any(1)
            candidate_boxes = boxes[keep].detach()
            if not len(candidate_boxes):
                continue
            actual, matched = self._actual_iou(candidate_boxes, gt)
            quality = pqa_scores(logits[index:index + 1], candidate_boxes,
                                 image_shape, self.pqa_chunk)
            rank, pairs, rank_gt = pqa_competition_rank(
                quality, actual.detach(), matched.detach(),
                candidate_boxes, competition_iou=self.pqa_soft_iou)
            rank_losses.append(rank)
            stats['rank'] += float(rank.detach())
            stats['pairs'] += pairs
            stats['rank_gt'] += rank_gt
            stats['candidates'] += len(candidate_boxes)
        zero = self.pqa_heatmap.output.weight.sum() * 0.
        losses = {
            'loss_pqa_location': (torch.stack(location_losses).mean()
                                  if location_losses else zero),
            'loss_pqa_rank': self.pqa_rank_weight * (
                torch.stack(rank_losses).mean() if rank_losses else zero),
        }
        self.pqa_train_stats.append(stats)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        features, raw = self._frozen_main(img, img_metas)
        logits = self.pqa_heatmap(features[0], features[1])
        image_shape = tuple(img.shape[-2:])
        results = []
        for index, ((boxes, sym), meta) in enumerate(zip(raw, img_metas)):
            raw_count = len(boxes)
            threshold = self.bbox_head.test_cfg.score_thr
            candidate_mask = (sym[:, :-1] > threshold).any(1)
            boxes, sym = boxes[candidate_mask], sym[candidate_mask]
            quality = pqa_scores(logits[index:index + 1], boxes,
                                 image_shape, self.pqa_chunk)
            score, subtype, tb_score = self._base_score(sym, quality)
            soft_dets, soft_source = self._soft_nms(boxes, score)
            final_count = min(int(self.bbox_head.test_cfg.max_per_img),
                              len(soft_dets))
            detections = soft_dets[:final_count]
            final_source = soft_source[:final_count]
            labels = subtype[final_source]
            if self.pqa_collect:
                scale = boxes.new_tensor(meta['scale_factor'])
                if meta.get('flip', False):
                    raise ValueError('PQA diagnostics require unflipped validation')
                self.pqa_cache.append(dict(
                    boxes=(boxes / scale).cpu(), sym=sym.cpu(),
                    quality=quality.cpu(), tb_score=tb_score.cpu(),
                    base_score=score.cpu(), soft_source=soft_source.cpu(),
                    soft_score=soft_dets[:, 4].cpu(),
                    final_source=final_source.cpu(),
                    final_score=detections[:, 4].cpu(),
                    labels=labels.cpu(), raw_count=raw_count,
                    filename=meta.get('filename')))
            if rescale and len(detections):
                detections[:, :4] /= detections.new_tensor(meta['scale_factor'])
            results.append(bbox2result(detections.cpu().numpy(),
                                       labels.cpu().numpy(),
                                       self.bbox_head.num_classes))
        return results

    def v50_begin_oracle(self, epoch):
        self.pqa_epoch = int(epoch)
        self.pqa_collect = True
        self.pqa_cache = []
        return True

    def v50_finish_oracle(self, dataset, runner):
        summaries, hard, candidates = {}, [], []
        oracle_rows = dict(sym_threshold=[], softnms_top128=[], final_top100=[])
        order_correct, order_pairs = 0, 0
        regret_sum = dict(sym=0., pqa=0., base=0.)
        regret_gt = 0
        for image_index, item in enumerate(self.pqa_cache):
            gt = torch.as_tensor(dataset.get_ann_info(image_index)['bboxes'],
                                 dtype=torch.float32).reshape(-1, 4)
            boxes = item['boxes']
            actual, assigned = self._actual_iou(boxes, gt)
            stages = {
                'sym_threshold': torch.arange(len(boxes)),
                'sym_top128': item['tb_score'].topk(min(128, len(boxes))).indices,
                'pqa_top64': item['quality'].topk(min(64, len(boxes))).indices,
                'pqa_top128': item['quality'].topk(min(128, len(boxes))).indices,
                'pqa_top256': item['quality'].topk(min(256, len(boxes))).indices,
                'base_score_top128': item['base_score'].topk(
                    min(128, len(boxes))).indices,
                'softnms_top128': item['soft_source'],
                'final_top100': item['final_source'],
            }
            for stage, indices in stages.items():
                summary = summaries.setdefault(
                    stage, dict(count=0, bins=[0] * 5, gt=[0] * 3))
                summary['count'] += len(indices)
                summary['bins'] = [left + right for left, right in zip(
                    summary['bins'], _bins(actual[indices]))]
                summary['gt'] = [left + right for left, right in zip(
                    summary['gt'], _coverage(boxes[indices], gt))]
            for gt_index in assigned[actual >= .50].unique():
                if gt_index < 0:
                    continue
                ids = ((assigned == gt_index) & (actual >= .50)).nonzero(
                    as_tuple=False).squeeze(1)
                gap = actual[ids, None] - actual[ids][None]
                valid = gap > .05
                difference = item['quality'][ids, None] - item['quality'][ids][None]
                order_correct += int(((difference > 0) & valid).sum())
                order_pairs += int(valid.sum())
                best = float(actual[ids].max())
                for name, values in (('sym', item['tb_score']),
                                     ('pqa', item['quality']),
                                     ('base', item['base_score'])):
                    chosen = ids[values[ids].argmax()]
                    regret_sum[name] += best - float(actual[chosen])
                regret_gt += 1
            retained = torch.zeros(len(boxes), dtype=torch.bool)
            retained[item['soft_source']] = True
            for kind, mask in (
                    ('overestimated_low', (actual < .50) & (item['quality'] >= .75)),
                    ('underestimated_good', (actual >= .75) & (item['quality'] < .50))):
                ids = mask.nonzero(as_tuple=False).squeeze(1)
                discrepancy = (item['quality'][ids] - actual[ids]).abs()
                ids = ids[discrepancy.topk(min(20, len(ids))).indices]
                for candidate in ids.tolist():
                    hard.append(dict(
                        image_index=image_index, filename=item['filename'],
                        kind=kind, bbox=boxes[candidate].tolist(),
                        true_iou=float(actual[candidate]),
                        pqa=float(item['quality'][candidate]),
                        sym=float(item['tb_score'][candidate]),
                        base_score=float(item['base_score'][candidate]),
                        softnms_kept=bool(retained[candidate])))
            candidates.append(dict(
                image_index=image_index, filename=item['filename'],
                raw_count=item['raw_count'], boxes=boxes.tolist(),
                true_iou=actual.tolist(), matched_gt=assigned.tolist(),
                pqa=item['quality'].tolist(), sym=item['tb_score'].tolist(),
                base_score=item['base_score'].tolist(),
                soft_source=item['soft_source'].tolist(),
                soft_score=item['soft_score'].tolist()))
            for label, indices in (
                    ('sym_threshold', stages['sym_threshold']),
                    ('softnms_top128', stages['softnms_top128']),
                    ('final_top100', stages['final_top100'])):
                oracle_rows[label].append(_oracle_result(
                    boxes[indices], item['sym'][indices], gt,
                    self.bbox_head.num_classes))
        for stage, values in summaries.items():
            runner.logger.info(
                '[PQA-STAGE] epoch=%d %s count=%d IoUbins=%s GT50/75/85=%s',
                self.pqa_epoch, stage, values['count'], values['bins'], values['gt'])
        runner.logger.info(
            '[PQA-RANK] epoch=%d sameGTorder=%.5f pairs=%d '
            'meanTop1Regret(sym/pqa/base)=%.5f/%.5f/%.5f GT=%d scoreMode=%s', self.pqa_epoch,
            order_correct / max(order_pairs, 1), order_pairs,
            regret_sum['sym'] / max(regret_gt, 1),
            regret_sum['pqa'] / max(regret_gt, 1),
            regret_sum['base'] / max(regret_gt, 1),
            regret_gt, self.pqa_score_mode)
        for label, rows in (('candidates', candidates),
                            ('hard_samples', hard), ('stages', summaries)):
            path = osp.join(runner.work_dir, 'pqa_%s_epoch_%02d.json' %
                            (label, self.pqa_epoch))
            mmcv.dump(rows, path)
            runner.logger.info('[PQA-EXPORT] %s', path)
        if self.pqa_epoch % 4 == 0:
            for stage, rows in oracle_rows.items():
                metrics = dataset.evaluate(
                    rows, metric='bbox', tb_official_agnostic=True,
                    tb_class_agnostic=False)
                runner.logger.info(
                    '[PQA-ORACLE] epoch=%d stage=%s AP/AP50/AP75=%.3f/%.3f/%.3f',
                    self.pqa_epoch, stage,
                    metrics['official_tb_bbox_mAP'],
                    metrics['official_tb_bbox_mAP_50'],
                    metrics['official_tb_bbox_mAP_75'])
        if summaries:
            selected = summaries['softnms_top128']
            key = (selected['gt'][2], selected['gt'][1],
                   order_correct / max(order_pairs, 1),
                   -regret_sum['base'] / max(regret_gt, 1))
            improved = self.best_pqa_key is None or key > self.best_pqa_key
            runner.logger.info(
                '[PQA-SELECT] Soft128_GT85/75=%d/%d order=%.5f regret=%.5f '
                'improved=%s notMAP=True', key[0], key[1], key[2], -key[3], improved)
            if improved:
                self.best_pqa_key = key
                runner.save_checkpoint(
                    runner.work_dir, filename_tmpl='best_pqa_quality.pth',
                    save_optimizer=True, create_symlink=False,
                    meta=dict(pqa_quality_key=key))
        self.pqa_cache = []
        self.pqa_collect = False


@HOOKS.register_module()
class PQADebugHook(Hook):
    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.initial_hash = _main_hash(model)
        trainable = sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad)
        runner.logger.info(
            '[PQA-BUILD] id=%s frozenMain=True trainable=%d P2P3=True '
            'originalPQA=GaussianMap+VolumeIoU adaptation=P2P3+optionalRank '
            'rankWeight=%.3f score=%s softNMS=gaussian sigma=%.2f topk=%d '
            'mainSHA256=%s', model.integrated_build_id, trainable,
            model.pqa_rank_weight, model.pqa_score_mode,
            model.pqa_soft_sigma, model.pqa_soft_topk, self.initial_hash)

    def before_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model._nam_epoch = runner.epoch

    def after_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        totals = dict(location=0., images=0, rank=0., pairs=0,
                      rank_gt=0, candidates=0)
        for item in model.pqa_train_stats:
            for key in totals:
                totals[key] += item[key]
        runner.logger.info(
            '[PQA-TRAIN] epoch=%d location=%.6f rankRaw=%.6f images=%d '
            'rankPairs=%d rankGT=%d candidates=%d', runner.epoch + 1,
            totals['location'] / max(totals['images'], 1),
            totals['rank'] / max(totals['images'], 1),
            totals['images'], totals['pairs'], totals['rank_gt'],
            totals['candidates'])
        current = _main_hash(model)
        runner.logger.info(
            '[PQA-MAIN] frozen=True mainChanged=%s mainSHA256=%s',
            current != self.initial_hash, current)
        if current != self.initial_hash:
            raise RuntimeError('Frozen main detector weights changed during PQA training')
        model.pqa_train_stats = []
