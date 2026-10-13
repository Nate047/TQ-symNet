"""V62: rank a GT's best few boxes above its actual pre-NMS competitors.

The detector, V59 quality network, score fusion and Soft-NMS are unchanged.
Only the optional full-candidate ranking objective changes.
"""
import json
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F
from mmcv.runner import HOOKS, Hook

from mmdet.core import bbox_overlaps
from mmdet.models.builder import DETECTORS
from .tb_detail_selector import RetinaNAMDetailSelector, DetailSelectorDebugHook
from .tb_high_quality_selector import HighQualitySelectorDebugHook


def bestset_rank_loss(positive, negative, gaps, temperature=.2,
                      margin_scale=.5):
    """All rivals contribute; adding easy rivals cannot reduce the penalty.

    Use the actual best positive score, avoiding a bonus for duplicate positive
    boxes. The gradient improves that member; other members retain their BCE.
    """
    if not len(positive) or not len(negative):
        return positive.sum() * 0. + negative.sum() * 0.
    pos = positive.max()
    margin = (gaps * margin_scale).clamp_max(.35)
    violations = (negative + margin - pos) / temperature
    return temperature * F.softplus(torch.logsumexp(violations, 0))


def bestset_group(boxes, actual, matched, gt_index, max_positive=4,
                  positive_gap=.03, min_gap=.02):
    """Return near-best positives and ALL eligible geometric competitors."""
    eligible = (matched == gt_index) & (actual >= .5)
    ids = eligible.nonzero(as_tuple=False).squeeze(1)
    if not len(ids):
        return ids, ids, eligible
    best = actual[ids].max()
    near = ids[actual[ids] >= best - positive_gap]
    positive = near[actual[near].topk(min(max_positive, len(near))).indices]
    overlap = bbox_overlaps(boxes[positive], boxes).max(0)[0]
    # Protect a different GT's correct detection. Low-quality overlapping
    # candidates remain in the competition pool, including hard background FP.
    pool = eligible | ((actual < .5) & (overlap > .5))
    valid = pool & (overlap > .5)
    valid &= actual < best - max(min_gap, positive_gap)
    valid[positive] = False
    return positive, valid.nonzero(as_tuple=False).squeeze(1), pool


def bestset_eval_summary(cache, dataset, max_positive=4, positive_gap=.03,
                        min_gap=.02):
    """Top1 includes overlapping FP; use identical pools on train and val."""
    summary = dict(available85=0, final85=0,
                   sym_top1=0, quality_top1=0, fused_top1=0,
                   groups=0, competitor_groups=0, competitors=0)
    for name in ('sym', 'quality', 'fused'):
        for key in ('fp_wins', 'nearbest_top1', 'set_wins', 'gap_sum'):
            summary[name + '_' + key] = 0.
    for index, item in enumerate(cache):
        boxes = item['boxes']
        gt = torch.as_tensor(dataset.get_ann_info(index)['bboxes'],
                             dtype=boxes.dtype).reshape(-1, 4)
        if not len(boxes) or not len(gt):
            continue
        overlap = bbox_overlaps(boxes, gt)
        actual, matched = overlap.max(1)
        sym = torch.logit(item['sym'][:, :-1].max(1)[0].clamp(1e-6, 1.-1e-6))
        quality = item['logit']
        fused = torch.logit(item['corrected'].clamp(1e-6, 1.-1e-6))
        for g in range(len(gt)):
            positive, rivals, pool = bestset_group(
                boxes, actual, matched, g, max_positive, positive_gap, min_gap)
            if not len(positive):
                continue
            best = actual[positive].max()
            available85 = bool(best >= .85)
            summary['groups'] += 1
            summary['available85'] += int(available85)
            if available85:
                summary['final85'] += int((overlap[item['source'], g] >= .85).any())
            summary['competitors'] += len(rivals)
            summary['competitor_groups'] += int(len(rivals) > 0)
            ids = pool.nonzero(as_tuple=False).squeeze(1)
            for name, score in (('sym', sym), ('quality', quality), ('fused', fused)):
                winner = ids[score[ids].argmax()]
                summary[name + '_top1'] += int(available85 and overlap[winner, g] >= .85)
                summary[name + '_fp_wins'] += int(actual[winner] < .5)
                summary[name + '_nearbest_top1'] += int(
                    matched[winner] == g and actual[winner] >= best - positive_gap)
                if len(rivals):
                    gap = float(score[positive].max() - score[rivals].max())
                    summary[name + '_gap_sum'] += gap
                    summary[name + '_set_wins'] += float(gap > 0) + .5 * float(gap == 0)
    for name in ('sym', 'quality', 'fused'):
        summary[name + '_gt85_rate'] = summary[name + '_top1'] / max(summary['available85'], 1)
        summary[name + '_nearbest_rate'] = summary[name + '_nearbest_top1'] / max(summary['groups'], 1)
        summary[name + '_win_rate'] = summary[name + '_set_wins'] / max(summary['competitor_groups'], 1)
        summary[name + '_gap_mean'] = summary[name + '_gap_sum'] / max(summary['competitor_groups'], 1)
    return summary


@DETECTORS.register_module()
class RetinaNAMBestSetSelector(RetinaNAMDetailSelector):
    def __init__(self, *args, bestset_rank_weight=0., bestset_max_positive=4,
                 bestset_positive_gap=.03, bestset_max_negative=None,
                 bestset_min_gap=.02, bestset_temperature=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.bestset_rank_weight = float(bestset_rank_weight)
        self.bestset_max_positive = int(bestset_max_positive)
        self.bestset_positive_gap = float(bestset_positive_gap)
        if bestset_max_negative is not None:
            raise ValueError('V62 competitors are uncapped; set bestset_max_negative=None')
        self.bestset_min_gap = float(bestset_min_gap)
        self.bestset_temperature = float(bestset_temperature)
        if (self.bestset_rank_weight < 0 or self.bestset_max_positive < 1 or
                self.bestset_min_gap < 0 or
                self.bestset_positive_gap < 0 or self.bestset_temperature <= 0):
            raise ValueError('Invalid best-set ranking configuration')
        self.bestset_train_stats = []
        self.integrated_build_id = 'v62.1-uncapped-bestset-competition-aligned-debug'

    def _full_candidate_training_loss(self, boxes, sym, gt, matched, actual,
                                      memory, meta, scan=None,
                                      corrected_scan=None):
        zero = self.candidate_quality.high_head.weight.sum() * 0.
        if self.bestset_rank_weight == 0 or not len(boxes) or not len(gt):
            return zero
        groups = []
        counts = dict(gt=0, gt85=0, positives=0, negatives=0, threats=0)
        # Selection uses detached full-pool scores and GT only in training.
        with torch.no_grad():
            for g in matched[actual >= .5].unique().tolist():
                positive, negative, _ = bestset_group(boxes, actual, matched, g,
                    self.bestset_max_positive, self.bestset_positive_gap, self.bestset_min_gap)
                if not len(positive) or not len(negative):
                    continue
                best = actual[positive].max()
                groups.append((positive, negative, best))
                counts['gt'] += 1
                counts['gt85'] += int(best >= .85)
                counts['positives'] += len(positive)
                counts['negatives'] += len(negative)
                counts['threats'] += int((corrected_scan[negative] >=
                                         corrected_scan[positive].max()).sum())
        self.bestset_train_stats.append(counts)
        if not groups:
            return zero
        selected = torch.cat([ids for pos, neg, _ in groups for ids in (pos, neg)]).unique()
        outputs, _, _ = self._score_candidates(self._normalized(boxes[selected], meta),
                                                memory, sym[selected], True)
        # searchsorted is valid because unique() returns sorted IDs. Grouping
        # shares each candidate's forward pass and preserves gradients.
        quality = outputs[:, 0]
        losses = []
        for positive, negative, best in groups:
            pos = quality[torch.searchsorted(selected, positive)]
            bad = quality[torch.searchsorted(selected, negative)]
            gaps = best - actual[negative]
            losses.append(bestset_rank_loss(pos, bad, gaps,
                                             self.bestset_temperature))
        return self.bestset_rank_weight * torch.stack(losses).mean()

    def v50_finish_oracle(self, dataset, runner):
        stats = bestset_eval_summary(self.quality_cache, dataset,
            self.bestset_max_positive, self.bestset_positive_gap, self.bestset_min_gap)
        self.bestset_val_summary = (self.quality_epoch, stats)
        runner.logger.info('[BESTSET-VAL] epoch=%d availableGT85=%d '
            'finalGT85=%d sym/quality/fusedTop1=%d/%d/%d',
            self.quality_epoch, stats['available85'], stats['final85'],
            stats['sym_top1'], stats['quality_top1'], stats['fused_top1'])
        runner.logger.info('[BESTSET-RANK-EVAL] split=val epoch=%d FPIncluded=True stats=%s',
                           self.quality_epoch, json.dumps(stats))
        return super().v50_finish_oracle(dataset, runner)


@HOOKS.register_module()
class BestSetSelectorDebugHook(DetailSelectorDebugHook):
    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        runner.logger.info('[BESTSET-BUILD] enabled=%s weight=%.3f '
            'bestPositive<=%d positiveGap=%.3f negativeCap=None '
            'minGap=%.3f temp=%.3f fullPreNMS=True detectorFrozen=True',
            model.bestset_rank_weight > 0, model.bestset_rank_weight,
            model.bestset_max_positive, model.bestset_positive_gap,
            model.bestset_min_gap,
            model.bestset_temperature)

    def after_train_epoch(self, runner):
        HighQualitySelectorDebugHook.after_train_epoch(self, runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        totals = {key: sum(row[key] for row in model.bestset_train_stats)
                  for key in ('gt', 'gt85', 'positives', 'negatives', 'threats')}
        runner.logger.info('[BESTSET-TRAIN] epoch=%d groups=%d gt85=%d '
            'positives=%d hardNegatives=%d currentlyOutranking=%d '
            'qualityOnly=True perGTMean=True', runner.epoch+1,
            totals['gt'], totals['gt85'], totals['positives'],
            totals['negatives'], totals['threats'])
        model.bestset_train_stats = []


@contextmanager
def diagnostic_rng():
    states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.random.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@HOOKS.register_module()
class BestSetTrainValHook(Hook):
    """Extra train-set inference uses val preprocessing and restores RNG."""
    def __init__(self, train_ann_file, interval=2):
        self.train_ann_file, self.interval = train_ann_file, int(interval)
        if self.interval < 1:
            raise ValueError('interval must be positive')

    def before_run(self, runner):
        from mmdet.core.evaluation.eval_hooks import EvalHook
        from mmdet.datasets import build_dataset, build_dataloader
        hooks = [h for h in runner.hooks if isinstance(h, EvalHook)]
        if len(hooks) != 1:
            raise RuntimeError('BestSetTrainValHook requires one single-GPU EvalHook')
        val = hooks[0].dataloader.dataset
        with diagnostic_rng():
            train = build_dataset(dict(type='CocoDataset', ann_file=self.train_ann_file,
                img_prefix=val.img_prefix, classes=val.CLASSES, test_mode=True,
                pipeline=val.pipeline.transforms))
            train_names = {row['filename'] for row in train.data_infos}
            if train_names & {row['filename'] for row in val.data_infos}:
                raise RuntimeError('Training and validation filenames overlap')
            self.loader = build_dataloader(train, samples_per_gpu=1, workers_per_gpu=2,
                num_gpus=1, dist=False, shuffle=False, seed=torch.initial_seed() % (2**32),
                generator=torch.Generator().manual_seed(torch.initial_seed()))
        runner.logger.info('[BESTSET-TRAINVAL-SETUP] train=%d val=%d interval=%d '
                           'samePipeline=True uniqueImages=True', len(train), len(val), self.interval)

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch % self.interval:
            return
        keys = ('official_tb_bbox_mAP', 'official_tb_bbox_mAP_50', 'official_tb_bbox_mAP_75')
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        previous = getattr(model, 'bestset_val_summary', None)
        if previous is None or previous[0] != epoch or any(
                key not in runner.log_buffer.output for key in keys):
            runner.logger.warning('[BESTSET-TRAINVAL] skipped: no current validation results')
            return
        val_ap = [float(runner.log_buffer.output[key]) for key in keys]
        val_stats = previous[1]
        modes = [(module, module.training) for module in runner.model.modules()]
        from mmdet.apis import single_gpu_test
        try:
            with diagnostic_rng():
                model.detail_debug = []
                model.v50_begin_oracle(epoch)
                predictions = single_gpu_test(runner.model, self.loader, show=False)
                metrics = self.loader.dataset.evaluate(predictions, metric='bbox',
                    tb_official_agnostic=True, tb_class_agnostic=False)
                stats = bestset_eval_summary(model.quality_cache, self.loader.dataset,
                    model.bestset_max_positive, model.bestset_positive_gap, model.bestset_min_gap)
        finally:
            model.quality_collect = False
            model.quality_cache = []
            model.detail_debug = []
            model.candidate_quality.attention_stats = []
            for module, mode in modes:
                module.training = mode
        train_ap = [float(metrics[key]) for key in keys]
        runner.logger.info('[BESTSET-TRAINVAL] epoch=%d AP/AP50/AP75 train=%s val=%s '
            'gap=%s RNGRestored=True', epoch, train_ap, val_ap,
            [a-b for a, b in zip(train_ap, val_ap)])
        runner.logger.info('[BESTSET-RANK-EVAL] split=train epoch=%d FPIncluded=True stats=%s',
                           epoch, json.dumps(stats))
        gaps = {key: stats[key] - val_stats[key] for key in stats
                if key.endswith(('_rate', '_mean'))}
        runner.logger.info('[BESTSET-RANK-GAP] epoch=%d trainMinusVal=%s', epoch, json.dumps(gaps))
