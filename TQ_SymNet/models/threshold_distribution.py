"""A0 reader + cumulative threshold supervision. No ROI-query changes."""
import json
from pathlib import Path

import torch
from torch import nn
from mmcv.runner import HOOKS, Hook
from mmdet.core import bbox_overlaps
from mmdet.models.builder import DETECTORS

from ..threshold_distribution_math import (ThresholdDistributionHead,
    distribution_element_loss, threshold_statistics, format_threshold_statistics)
from .attention import CandidateQualityAttention
from .detail import DetailQualityAttention
from .network import TQSymNet


class DistributionQualityAttention(DetailQualityAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.high_head = ThresholdDistributionHead(self.hidden)
        self.threshold_distribution = True

    def forward(self, boxes, memories, sym, training_checkpoint=False,
                return_thresholds=False):
        # CandidateQualityAttention.forward calls self._score dynamically, so
        # the retained edge and native-detail descriptor paths are ALL retained.
        if not len(boxes):
            result = (boxes.new_empty((0,2)), boxes.new_empty((0,7)),
                      boxes.new_empty((0,3)))
            return result + (boxes.new_empty((0,10)),) if return_thresholds else result
        descriptor, gate, entropy = CandidateQualityAttention.forward(
            self, boxes, memories, training_checkpoint)
        quality, thresholds = self.high_head.predict(descriptor)
        # Existing quality-only rank and alpha fusion consume this scalar logit.
        result = (torch.stack((quality, quality), 1), gate, entropy)
        return result + (thresholds,) if return_thresholds else result


@DETECTORS.register_module()
class TQSymNetThresholdDistribution(TQSymNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.quality_arch != 'attention' or not self.quality_detach
                or not self.selector_learned_gate or not self.selector_gate_trainable
                or self.selector_quality_target != 'coco_all_iou'):
            raise ValueError('Distribution experiment requires frozen original A0 and learned alpha')
        previous = self.candidate_quality
        with torch.random.fork_rng(devices=[]):
            replacement = DistributionQualityAttention(hidden=previous.hidden,
                heads=previous.heads, texture=previous.texture, gaussian=previous.gaussian,
                chunk=previous.chunk, checkpoint_candidates=previous.checkpoint_candidates,
                edges=previous.edges, detail=previous.detail,
                lbp_enabled=previous.lbp_enabled, gabor_enabled=previous.gabor_enabled)
        replacement.fusion_gate_logit = nn.Parameter(previous.fusion_gate_logit.detach().clone())
        replacement.tb_lse_temperature_raw = nn.Parameter(
            previous.tb_lse_temperature_raw.detach().clone(), requires_grad=False)
        common = {k:v for k,v in previous.state_dict().items() if not k.startswith('high_head.')}
        result = replacement.load_state_dict(common, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {'high_head.weight', 'high_head.bias'}:
            raise RuntimeError('Distribution retained-reader transfer mismatch: %s' % (result,))
        for parameter in replacement.parameters():
            parameter.requires_grad_(self.selector_quality_trainable)
        replacement.fusion_gate_logit.requires_grad_(self.selector_gate_trainable)
        replacement.tb_lse_temperature_raw.requires_grad_(False)
        self.candidate_quality = replacement
        self.distribution_train_stats = None
        self.distribution_val_cache = []
        self.integrated_build_id = 'TQ-A0-cumulative-11bin-10threshold'

    def _score_candidates(self, boxes, memory, sym, training=False):
        outputs, gate, entropy, threshold_logits = self.candidate_quality(
            boxes, memory, sym, training_checkpoint=training, return_thresholds=True)
        if (not self.candidate_quality.training and self.quality_collect):
            self.distribution_val_cache.append(threshold_logits.detach().cpu())
        return outputs, gate, entropy

    def _training_quality_predictions(self, boxes, memory, sym):
        return self.candidate_quality(boxes, memory, sym,
            training_checkpoint=True, return_thresholds=True)

    def _quality_element_loss(self, logit, target, actual, supervision):
        # Replace, NOT add to, the aggregate BCE. Sampling and alpha still use
        # the same scalar mean target, and rank stays quality-only.
        element = distribution_element_loss(supervision, actual,
            self.selector_quality_target_temperature)
        stats = threshold_statistics(supervision, actual,
            self.selector_quality_target_temperature)
        self.distribution_train_stats = (stats if self.distribution_train_stats is None
                                         else self.distribution_train_stats + stats)
        return element

    def v50_begin_oracle(self, epoch):
        self.distribution_val_cache = []
        return super().v50_begin_oracle(epoch)

    def v50_finish_oracle(self, dataset, runner):
        if self.quality_collect:
            if len(self.distribution_val_cache) != len(self.quality_cache):
                raise RuntimeError('Distribution/box validation caches have different image counts')
            total = None
            for image_index, (item, logits) in enumerate(zip(
                    self.quality_cache, self.distribution_val_cache)):
                boxes = item['boxes']
                if len(logits) != len(boxes):
                    raise RuntimeError('Distribution/box validation candidate alignment changed')
                gt = torch.as_tensor(dataset.get_ann_info(image_index)['bboxes'],
                                     dtype=torch.float32).reshape(-1,4)
                actual = (bbox_overlaps(boxes, gt).max(1)[0]
                          if len(boxes) and len(gt) else torch.zeros(len(boxes)))
                row = threshold_statistics(logits, actual, self.selector_quality_target_temperature)
                total = row if total is None else total + row
            if total is not None:
                record = dict(epoch=self.quality_epoch, thresholds=[.5+.05*k for k in range(10)],
                    domain='all_eligible_pre_softnms_candidates',
                    stats=format_threshold_statistics(total))
                path = Path(runner.work_dir) / ('threshold_val_epoch_%02d.json' % self.quality_epoch)
                path.write_text(json.dumps(record, indent=2), encoding='utf-8')
                runner.logger.info('[THRESHOLD-DISTRIBUTION-VAL] %s', json.dumps(record))
        # Always release, even ordinary epochs/CLI when Oracle is deferred.
        self.distribution_val_cache = []
        return super().v50_finish_oracle(dataset, runner)


@HOOKS.register_module()
class TQThresholdDistributionAuditHook(Hook):
    def before_run(self, runner):
        self.model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.enabled = isinstance(self.model, TQSymNetThresholdDistribution)
        if not self.enabled:
            return
        model = self.model
        row = dict(reader='original_A0_no_ROI_query', bins=11,
            thresholds=[.5+.05*k for k in range(10)],
            temperature=model.selector_quality_target_temperature,
            supervision='mean_per_threshold_BCE', aggregate_bce=False,
            output='single_logit_of_mean_survival', alpha_trainable=True,
            rank='unchanged_quality_only', detector_frozen=True)
        if runner.meta.get('threshold_distribution_contract', row) != row:
            raise RuntimeError('Resume threshold distribution contract changed')
        runner.meta['threshold_distribution_contract'] = row
        (Path(runner.work_dir) / 'threshold_distribution_contract.json').write_text(
            json.dumps(row, indent=2), encoding='utf-8')
        runner.logger.info('[THRESHOLD-DISTRIBUTION-BUILD] %s', json.dumps(row))

    def before_train_epoch(self, runner):
        if self.enabled:
            self.model.distribution_train_stats = None

    def after_train_epoch(self, runner):
        if not self.enabled:
            return
        stats = self.model.distribution_train_stats
        if stats is not None:
            row = dict(epoch=runner.epoch+1, domain='supervised_sampled_candidates',
                       stats=format_threshold_statistics(stats))
            runner.logger.info('[THRESHOLD-DISTRIBUTION-TRAIN] %s', json.dumps(row))
            with (Path(runner.work_dir) / 'threshold_train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
        self.model.distribution_train_stats = None
