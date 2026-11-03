"""Isolated low-LR A13 continuation; no model or scoring changes."""
import json
import math
from pathlib import Path

from mmcv.runner import HOOKS
from mmcv.runner.hooks.lr_updater import LrUpdaterHook


@HOOKS.register_module()
class TQROIFinetuneLrUpdaterHook(LrUpdaterHook):
    """Resume optimizer, then apply a separate four-epoch cosine phase.

    Original optimizer initial_lr fields remain unchanged: resuming this phase
    twice must not repeatedly multiply the LR by 0.1. The phase uses absolute
    quality_lr; frozen groups keep their relative initial-LR scale.
    """
    def __init__(self, quality_lr=1e-5, start_epoch=8, end_epoch=12,
                 steps_per_epoch=454,
                 source_dir='work_dirs/publication_supplement/a13_s0',
                 by_epoch=True, **kwargs):
        if not by_epoch or kwargs.get('warmup') is not None:
            raise ValueError('A13 continuation uses epoch-wise decay without warmup')
        super().__init__(by_epoch=True, **kwargs)
        self.quality_lr = float(quality_lr)
        self.start_epoch, self.end_epoch = int(start_epoch), int(end_epoch)
        self.steps_per_epoch = int(steps_per_epoch)
        self.source_dir = Path(source_dir).resolve()
        if (not math.isfinite(self.quality_lr) or self.quality_lr <= 0
                or self.start_epoch != 8 or self.end_epoch <= self.start_epoch
                or self.steps_per_epoch != 454):
            raise ValueError('Invalid A13 continuation schedule')

    def before_run(self, runner):
        if isinstance(runner.optimizer, dict):
            raise RuntimeError('Only the original single AdamW optimizer is supported')
        if not self.start_epoch <= runner.epoch < self.end_epoch:
            raise RuntimeError('Resume A13 epoch8, or an unfinished checkpoint of this continuation')
        if runner.max_epochs != self.end_epoch:
            raise RuntimeError('runner.max_epochs must equal continuation end_epoch')
        if runner.iter != runner.epoch*self.steps_per_epoch:
            raise RuntimeError('Expected an epoch-boundary, single-GPU batch2 checkpoint')
        model = runner.model.module if hasattr(runner.model,'module') else runner.model
        q = model.candidate_quality
        if (model.quality_arch != 'pooled' or q.hidden != 96 or q.mlp_hidden != 880
                or model.selector_quality_target != 'direct_iou'
                or model.quality_rank_weight != 0 or model.selector_rank_weight != 0):
            raise RuntimeError('Expected unchanged capacity-matched A13/direct-IoU/no-rank')
        if any(p.requires_grad for n,p in model.named_parameters()
               if not n.startswith('candidate_quality.')):
            raise RuntimeError('The detector must remain frozen')
        if q.fusion_gate_logit.requires_grad or abs(float(.1+.8*q.fusion_gate_logit.detach().sigmoid())-.5)>1e-6:
            raise RuntimeError('Keep the original fixed alpha=0.5')
        meta = runner.meta or {}
        if meta.get('tq_supplement',{}).get('parameters',{}).get('trainable') != 2341713:
            raise RuntimeError('Checkpoint does not carry the expected A13 provenance')
        if meta.get('tq_initialization',{}).get('detector_checkpoint_sha256') != (
                '4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74'):
            raise RuntimeError('A13 checkpoint used a different detector')
        if not any(p.requires_grad and p in runner.optimizer.state
                   for p in model.parameters()):
            raise RuntimeError('Missing resumed AdamW state; do not use load_from')
        contract = dict(start_epoch=self.start_epoch,end_epoch=self.end_epoch,
                        quality_lr=self.quality_lr,steps_per_epoch=self.steps_per_epoch)
        previous = meta.get('tq_roi_continuation')
        if previous is None and runner.epoch != self.start_epoch:
            raise RuntimeError('Later epochs require continuation provenance')
        if previous is not None and previous['schedule'] != contract:
            raise RuntimeError('Do not change the schedule while resuming this continuation')
        output = Path(runner.work_dir).resolve()
        if output == self.source_dir:
            raise RuntimeError('Use a NEW work-dir; preserve the original eight-epoch run')
        if previous is not None and Path(previous['work_dir']).resolve() != output:
            raise RuntimeError('Resume an interrupted continuation in its own work-dir')
        if previous is None and (output/'roi_continuation.json').exists():
            raise RuntimeError('Continuation already exists; resume its checkpoint or use a new work-dir')
        # EvalHook restores this path and may DELETE it when a score improves.
        # Keep the score comparator, but remove references outside the new run.
        messages = meta.setdefault('hook_msgs',{})
        if previous is None and (not math.isfinite(float(messages.get('best_score',float('nan'))))
                                 or abs(float(messages['best_score'])-.396)>1e-7):
            raise RuntimeError('Use the AP-selected A13 epoch8 best checkpoint with best_score=0.396')
        removed = {}
        for key in ('best_ckpt','last_ckpt'):
            old = messages.get(key)
            if old and output not in Path(old).resolve().parents:
                removed[key] = messages.pop(key)
        for hook in runner.hooks:
            old = getattr(hook,'best_ckpt_path',None)
            if old and output not in Path(old).resolve().parents:
                hook.best_ckpt_path = None
        trainable = [g for g in runner.optimizer.param_groups
                     if any(p.requires_grad for p in g['params'])]
        if not trainable or any('initial_lr' not in g for g in runner.optimizer.param_groups):
            raise RuntimeError('Missing original optimizer initial_lr fields')
        original_quality_lrs = {float(g['initial_lr']) for g in trainable}
        if any(abs(v-1e-4)>1e-10 for v in original_quality_lrs):
            raise RuntimeError('Unexpected original A13 quality learning rate')
        super().before_run(runner)
        self.base_lr = [float(g['initial_lr'])*(self.quality_lr/1e-4)
                        for g in runner.optimizer.param_groups]
        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner,self.regular_lr)
        record = dict(schedule=contract,work_dir=str(output),
            source_dir=str(self.source_dir),resumed_at_epoch=runner.epoch,
            inherited_best_score=messages.get('best_score'),
            removed_external_checkpoint_references=removed,
            optimizer_state_restored=True,exact_rng_state_restored=False,
            scope='Additional-budget low-LR fine-tuning; not a 12-epoch-from-scratch matched control')
        meta['tq_roi_continuation'] = record
        runner.meta = meta
        output.mkdir(parents=True,exist_ok=True)
        (output/'roi_continuation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        runner.logger.info('[ROI-CONTINUE] %s',json.dumps(record))

    def get_lr(self, runner, base_lr):
        progress = (runner.epoch-self.start_epoch)/(self.end_epoch-self.start_epoch)
        if not 0. <= progress <= 1.:
            raise RuntimeError('Continuation epoch is outside its declared phase')
        return .5*base_lr*(1.+math.cos(math.pi*progress))

    def before_train_epoch(self, runner):
        if len(runner.data_loader) != self.steps_per_epoch:
            raise RuntimeError('Keep original 599/200 sampling and single-GPU batch2')
        super().before_train_epoch(runner)
        rates = sorted({g['lr'] for g in runner.optimizer.param_groups
                        if any(p.requires_grad for p in g['params'])})
        runner.logger.info('[ROI-CONTINUE-LR] epoch=%d qualityLRs=%s',runner.epoch+1,rates)
