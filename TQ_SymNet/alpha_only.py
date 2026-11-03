"""Continue one scalar alpha from a complete selected quality checkpoint."""
import hashlib
import json
from pathlib import Path

import torch
from mmcv.runner import HOOKS
from mmdet.core import bbox_overlaps
from mmdet.models.builder import DETECTORS

from .alpha_experiment import TQOriginalCohortInitHook
from .checkpoints import load_retained_checkpoint
from .data_contract import sha256_file
from .models.network import TQSymNet


GATE = 'candidate_quality.fusion_gate_logit'


def frozen_state_hash(model):
    """Include persistent buffers as well as weights, excluding ONLY alpha."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name == GATE:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@DETECTORS.register_module()
class TQSymNetAlphaOnly(TQSymNet):
    def __init__(self, *args, **kwargs):
        kwargs.update(selector_quality_trainable=False,
                      selector_gate_trainable=True, selector_learned_gate=True)
        super().__init__(*args, **kwargs)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name == GATE)
        if self.selector_tb_fusion != 'max' or self.selector_fusion_loss_weight != 0:
            raise ValueError('Alpha continuation expects MAX and existing calibration loss')
        self.integrated_build_id = 'TQ-SymNet-alpha-only-continuation'
        self.train()

    def train(self, mode=True):
        # Freeze BN/dropout/module state too. Explicit return_loss=True still
        # dispatches to forward_train; autograd for the scalar stays enabled.
        torch.nn.Module.train(self, False)
        return self

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, **kwargs):
        if gt_bboxes_ignore is not None and any(len(g) for g in gt_bboxes_ignore):
            raise ValueError('Ignored GT needs an explicit calibration policy')
        with torch.no_grad():
            features, raw, _ = self._main(img, img_metas)
            originals = self._v330_unwrap_original_images(
                kwargs.get('original_img'), len(raw))
        losses = []
        for index, ((boxes, sym), gt, meta, original) in enumerate(
                zip(raw, gt_bboxes, img_metas, originals)):
            with torch.no_grad():
                keep = (sym[:, :-1] > self.bbox_head.test_cfg.score_thr).any(1)
                boxes, sym = boxes[keep].detach(), sym[keep].detach()
                if not len(boxes):
                    continue
                memory = self._image_memory(features, img, meta, original, index)
                if len(gt):
                    actual, matched = bbox_overlaps(boxes, gt).max(1)
                else:
                    actual = boxes.new_zeros(len(boxes))
                    matched = torch.full((len(boxes),), -1, dtype=torch.long,
                                         device=boxes.device)
                scan = self._score_candidates(self._normalized(boxes, meta),
                                              memory, sym)[0]
                target = self._quality_target(actual)
                corrected = self._final_score(sym, scan[:, 0], scan[:, 1])
                # Same strata, lesion representatives and competitor mining
                # as the quality-training experiment. Only alpha is updated.
                ids = self._selector_sample(actual, matched, target,
                                            scan[:, 0].sigmoid(), corrected, boxes)
                base = torch.logit(sym[ids, :-1].max(1)[0].clamp(1e-6, 1.-1e-6))
            if len(ids):
                losses.append(self._residual_loss(base, scan[ids, 0], target[ids],
                                                  actual[ids], matched[ids]))
        # Evaluation-mode attention logs must not accumulate throughout train.
        self.detail_debug = []
        self.subtype_gate_stats = []
        self.candidate_quality.attention_stats = []
        zero = self.candidate_quality.fusion_gate_logit * 0.
        return dict(loss_alpha_calibration=torch.stack(losses).mean() if losses else zero)


@HOOKS.register_module()
class TQAlphaContinuationInitHook(TQOriginalCohortInitHook):
    def __init__(self, quality_checkpoint, expected_epoch=7, **kwargs):
        super().__init__(**kwargs)
        self.quality_checkpoint = quality_checkpoint
        self.expected_epoch = int(expected_epoch)

    def before_run(self, runner):
        # Validate original cohort and detector identity before loading quality.
        # This is a new optimizer/schedule; --resume-from is intentionally rejected.
        if runner.epoch or runner.iter:
            raise RuntimeError('Use a fresh work-dir, not --resume-from, for alpha-only calibration')
        super().before_run(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        detector_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                           if k.startswith(('backbone.', 'neck.', 'bbox_head.'))}
        checkpoint, audit = load_retained_checkpoint(model, self.quality_checkpoint)
        meta = checkpoint.get('meta', {})
        if meta.get('epoch') != self.expected_epoch:
            raise RuntimeError('Expected source epoch %d, got %r' %
                               (self.expected_epoch, meta.get('epoch')))
        if meta.get('tq_original_cohort') != runner.meta['tq_original_cohort']:
            raise RuntimeError('Quality checkpoint is not from the matched original cohort')
        for name, value in detector_before.items():
            if not torch.equal(value, model.state_dict()[name].detach().cpu()):
                raise RuntimeError('Source quality checkpoint changed detector: ' + name)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if names != [GATE]:
            raise RuntimeError('Only scalar gate may be trainable: %r' % names)
        model.train()
        self.model = model
        self.initial_hash = frozen_state_hash(model)
        record = dict(source=str(self.quality_checkpoint),
                      source_sha256=sha256_file(self.quality_checkpoint),
                      source_epoch=meta['epoch'], frozen_sha256=self.initial_hash,
                      initial_alpha=float((.1+.8*model.candidate_quality.fusion_gate_logit.sigmoid()).detach()),
                      trainable=names, optimizer_reset=True, **audit)
        runner.meta['tq_alpha_continuation'] = record
        runner.meta['tq_initialization']['quality_initialization'] = 'retained_selected_checkpoint'
        runner.meta['tq_initialization']['quality_checkpoint_sha256'] = record['source_sha256']
        runner.meta['tq_initialization']['quality_checkpoint'] = str(self.quality_checkpoint)
        (Path(runner.work_dir)/'original_cohort.json').write_text(json.dumps(
            dict(contract=runner.meta['tq_original_cohort'],
                 initialization=runner.meta['tq_initialization']), indent=2), encoding='utf-8')
        (Path(runner.work_dir)/'alpha_continuation.json').write_text(
            json.dumps(record, indent=2), encoding='utf-8')
        runner.logger.info('[ALPHA-ONLY-BUILD] %s', json.dumps(record))

    def before_train_epoch(self, runner):
        super().before_train_epoch(runner)
        self.model.train()

    def after_train_epoch(self, runner):
        current = frozen_state_hash(self.model)
        if current != self.initial_hash:
            raise RuntimeError('Alpha-only run changed frozen weights or persistent buffers')
        runner.logger.info('[ALPHA-ONLY-FROZEN] epoch=%d changed=False SHA256=%s',
                           runner.epoch+1, current)
