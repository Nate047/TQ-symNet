"""F2 querygate-v2 epoch8 -> epoch12: isolated, explicit low-LR phase.

No architecture/loss/scoring changes. Resume AdamW/alpha; preserve the source
checkpoint even when MMCV EvalHook replaces the best checkpoint in a new dir.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from mmcv.runner import HOOKS
from mmcv.runner.hooks.lr_updater import LrUpdaterHook


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@HOOKS.register_module()
class TQF2FinetuneLrUpdaterHook(LrUpdaterHook):
    def __init__(self, quality_lr=1e-5, start_epoch=8, end_epoch=12,
                 steps_per_epoch=454,
                 source_checkpoint='work_dirs/roi_threshold/f2_s0/best_official_tb_bbox_mAP_epoch_8.pth',
                 by_epoch=True, **kwargs):
        if not by_epoch or kwargs.get('warmup') is not None:
            raise ValueError('F2 continuation uses an epoch-wise phase without warmup')
        super().__init__(by_epoch=True, **kwargs)
        self.quality_lr = float(quality_lr)
        self.start_epoch, self.end_epoch = int(start_epoch), int(end_epoch)
        self.steps_per_epoch = int(steps_per_epoch)
        self.source = Path(source_checkpoint).resolve()
        if (not math.isfinite(self.quality_lr) or self.quality_lr <= 0
                or self.start_epoch != 8 or self.end_epoch != 12 or self.steps_per_epoch != 454):
            raise ValueError('Expected the declared F2 8->12 / 454-step continuation')

    def before_run(self, runner):
        if isinstance(runner.optimizer, dict):
            raise RuntimeError('Expected the original single AdamW optimizer')
        if not self.start_epoch <= runner.epoch < self.end_epoch or runner.max_epochs != self.end_epoch:
            raise RuntimeError('Resume epoch8, or epoch9..11 of this continuation; max_epochs=12')
        if runner.iter != runner.epoch*self.steps_per_epoch:
            raise RuntimeError('Expected a single-GPU batch2 epoch-boundary checkpoint')
        model = runner.model.module if hasattr(runner.model,'module') else runner.model
        q = model.candidate_quality
        if (model.integrated_build_id != 'TQ-ROI-content-role-paired-querygate-v2'
                or model.quality_arch != 'attention' or not model.quality_detach
                or model.selector_quality_target != 'coco_all_iou'
                or not model.selector_learned_gate or not model.selector_gate_trainable
                or model.selector_tb_fusion != 'max' or model.selector_rank_weight != .1
                or model.selector_sigma != .7 or model.selector_fusion_loss_weight != 0
                or model.selector_edge_weight != 0
                or not getattr(q,'paired_boundary',False) or q.roi_pool_size != 5
                or q.hidden != 96 or q.heads != 6 or not q.texture or not q.detail
                or not q.lbp_enabled or not q.gabor_enabled):
            raise RuntimeError('Keep the completed F2 querygate-v2 architecture and recipe unchanged')
        if any(p.requires_grad for n,p in model.named_parameters()
               if not n.startswith('candidate_quality.')):
            raise RuntimeError('The detector must stay frozen')
        if sum(p.numel() for p in q.parameters() if p.requires_grad) != 2465096:
            raise RuntimeError('F2 trainable parameter count changed')
        alpha = float((.1+.8*q.fusion_gate_logit.detach().sigmoid()).cpu())
        if not q.fusion_gate_logit.requires_grad or not math.isfinite(alpha) or not .1 < alpha < .9:
            raise RuntimeError('Resume the learned scalar alpha, do not freeze/reinitialize it')
        meta = runner.meta or {}
        if meta.get('seed') != 0:
            raise RuntimeError('Expected seed0 source and continuation')
        original = meta.get('roi_query_contract',{})
        if (original.get('model') != model.integrated_build_id
                or original.get('trainable') != 2465096 or not original.get('alpha_trainable')):
            raise RuntimeError('Source checkpoint lacks matching F2-v2 provenance')
        if meta.get('tq_initialization',{}).get('detector_checkpoint_sha256') != (
                '4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74'):
            raise RuntimeError('Source F2 used a different main detector')
        required_state = (q.fusion_gate_logit,q.high_head.weight,
                          q.visual_projection[0].weight,
                          q.roi_reader.level_gate[0].weight,q.pair_reader.level_gate[0].weight)
        for parameter in required_state:
            state = runner.optimizer.state.get(parameter,{})
            if (not {'step','exp_avg','exp_avg_sq'} <= set(state)
                    or float(state['step']) < 1
                    or state['exp_avg'].shape != parameter.shape
                    or state['exp_avg_sq'].shape != parameter.shape):
                raise RuntimeError('AdamW state was not restored for quality/Query gates/alpha; use resume_from')
        if not self.source.is_file() or self.source.stat().st_size == 0:
            raise RuntimeError('Missing/empty source checkpoint: use the SERVER original, not zero-byte local copies')
        source_hash = checkpoint_sha256(self.source)
        contract = dict(start_epoch=8,end_epoch=12,quality_lr=self.quality_lr,steps_per_epoch=454)
        previous = meta.get('tq_f2_continuation')
        if previous is None and runner.epoch != self.start_epoch:
            raise RuntimeError('Later resumed epochs require continuation provenance')
        if previous is None and abs(alpha-.5364386439323425)>1e-6:
            raise RuntimeError('Source epoch8 alpha was not restored (expected about .53643864)')
        if previous is not None and (previous['schedule'] != contract
                or previous['source_sha256'] != source_hash
                or Path(previous['source_checkpoint']).resolve() != self.source):
            raise RuntimeError('Continuation schedule/source changed')
        output = Path(runner.work_dir).resolve()
        if output == self.source.parent:
            raise RuntimeError('Use a NEW output directory; preserve the eight-epoch run')
        if previous is not None and Path(previous['work_dir']).resolve() != output:
            raise RuntimeError('Resume an interrupted continuation in its existing output directory')
        if previous is None and (output/'f2_continuation.json').exists():
            raise RuntimeError('Continuation exists: resume its checkpoint or choose a new output directory')
        messages = meta.setdefault('hook_msgs',{})
        if previous is None and (not math.isfinite(float(messages.get('best_score',float('nan'))))
                                 or abs(float(messages['best_score'])-.402)>1e-7):
            raise RuntimeError('Expected the selected F2 epoch8 checkpoint with best AP=.402')
        # An external best_ckpt path can otherwise be removed by EvalHook when
        # a better AP appears. Keep the comparator, remove only the old paths.
        removed = {}
        for key in ('best_ckpt','last_ckpt'):
            old = messages.get(key)
            if old and output not in Path(old).resolve().parents:
                removed[key] = messages.pop(key)
        for hook in runner.hooks:
            old = getattr(hook,'best_ckpt_path',None)
            if old and output not in Path(old).resolve().parents:
                hook.best_ckpt_path = None
        if any('initial_lr' not in g for g in runner.optimizer.param_groups):
            raise RuntimeError('Missing original optimizer initial_lr values')
        active = [g for g in runner.optimizer.param_groups if any(p.requires_grad for p in g['params'])]
        if not active or any(abs(float(g['initial_lr'])-1e-4)>1e-10 for g in active):
            raise RuntimeError('Original quality and alpha LR must be 1e-4')
        super().before_run(runner)
        # Do NOT mutate initial_lr: repeated resume must not scale twice.
        self.base_lr = [float(g['initial_lr'])*(self.quality_lr/1e-4)
                        for g in runner.optimizer.param_groups]
        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner,self.regular_lr)
        record = dict(schedule=contract,work_dir=str(output),source_checkpoint=str(self.source),
            source_sha256=source_hash,resumed_at_epoch=runner.epoch,alpha_at_resume=alpha,
            original_best_ap=.402,current_best_comparator=messages.get('best_score'),
            optimizer_state_restored=True,exact_rng_state_restored=False,
            removed_external_checkpoint_references=removed,
            scope='Extra-budget four-epoch low-LR phase; not a matched 12-epoch-from-scratch comparison')
        meta['tq_f2_continuation'] = record
        runner.meta = meta
        output.mkdir(parents=True,exist_ok=True)
        (output/'f2_continuation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        runner.logger.info('[F2-CONTINUE] %s',json.dumps(record))

    def get_lr(self, runner, base_lr):
        progress = (runner.epoch-self.start_epoch)/(self.end_epoch-self.start_epoch)
        if not 0. <= progress <= 1.:
            raise RuntimeError('Continuation epoch is outside the declared phase')
        return .5*base_lr*(1.+math.cos(math.pi*progress))

    def before_train_epoch(self, runner):
        if len(runner.data_loader) != self.steps_per_epoch:
            raise RuntimeError('Keep the original sampling and single-GPU batch2 (454 steps/epoch)')
        super().before_train_epoch(runner)
        rates = sorted({g['lr'] for g in runner.optimizer.param_groups
                        if any(p.requires_grad for p in g['params'])})
        runner.logger.info('[F2-CONTINUE-LR] epoch=%d qualityAndAlphaLRs=%s',runner.epoch+1,rates)
