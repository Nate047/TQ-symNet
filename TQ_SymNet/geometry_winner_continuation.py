"""Continue the completed G1 epoch8 state through epochs9..12, unchanged model."""
import json
import math
from pathlib import Path

from mmcv.runner import HOOKS
from mmcv.runner.hooks.lr_updater import LrUpdaterHook

from .roi_query_continuation import checkpoint_sha256


@HOOKS.register_module()
class TQG1FinetuneLrUpdaterHook(LrUpdaterHook):
    def __init__(self, quality_lr=1e-5, start_epoch=8, end_epoch=12,
                 steps_per_epoch=454,
                 source_checkpoint='work_dirs/quality_choice/g1_s0/epoch_8.pth',
                 by_epoch=True, **kwargs):
        if not by_epoch or kwargs.get('warmup') is not None:
            raise ValueError('G1 continuation uses an epoch-wise phase without warmup')
        super().__init__(by_epoch=True, **kwargs)
        self.quality_lr = float(quality_lr)
        self.start_epoch, self.end_epoch = int(start_epoch), int(end_epoch)
        self.steps_per_epoch = int(steps_per_epoch)
        self.source = Path(source_checkpoint).resolve()
        if (not math.isfinite(self.quality_lr) or self.quality_lr <= 0
                or self.start_epoch != 8 or self.end_epoch != 12 or self.steps_per_epoch != 454):
            raise ValueError('Expected the declared G1 8->12 / 454-step continuation')

    def before_run(self, runner):
        if isinstance(runner.optimizer, dict):
            raise RuntimeError('Expected the original single AdamW optimizer')
        if not self.start_epoch <= runner.epoch < self.end_epoch or runner.max_epochs != self.end_epoch:
            raise RuntimeError('Resume epoch8, or epoch9..11 of this continuation; max_epochs=12')
        if runner.iter != runner.epoch*self.steps_per_epoch:
            raise RuntimeError('Expected a single-GPU batch2 epoch-boundary checkpoint')
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        q = model.candidate_quality
        if (model.integrated_build_id != 'TQ-F2-signed-geometry-goodset-v1'
                or model.quality_arch != 'attention' or not model.quality_detach
                or model.selector_quality_target != 'coco_all_iou'
                or not model.selector_learned_gate or not model.selector_gate_trainable
                or not model.selector_quality_rank_only
                or model.selector_tb_fusion != 'max' or model.selector_rank_weight != .1
                or model.selector_sigma != .7 or model.selector_fusion_loss_weight != 0
                or model.selector_edge_weight != 0
                or not getattr(q, 'paired_boundary', False) or q.roi_pool_size != 5
                or q.hidden != 96 or q.heads != 6 or not q.texture or not q.detail
                or not q.lbp_enabled or not q.gabor_enabled or not hasattr(q, 'signed_aux')
                or model.active_rank_name != 'loss_winner_set'):
            raise RuntimeError('Keep the completed G1 architecture and recipe unchanged')
        geometry_contract = dict(geometry_weight=.05, geometry_beta=.05, ambiguity_gap=.1,
            winners_max=4, winner_temperature=.5, rank='goodset_replaces_pair_rank',
            count_normalized=True, rival_cap=None, unchosen_equivalent='neutral',
            geometry_only_training=True, output_boxes_unchanged=True)
        settings = (model.geometry_weight, model.geometry_beta, model.geometry_ambiguity_gap,
                    model.winner_max, model.winner_temperature)
        if settings != (.05, .05, .1, 4, .5):
            raise RuntimeError('Do not change geometry, winner settings or rival coverage')
        if any(p.requires_grad for n, p in model.named_parameters()
               if not n.startswith('candidate_quality.')):
            raise RuntimeError('The detector must stay frozen')
        if sum(p.numel() for p in q.parameters() if p.requires_grad) != 2580065:
            raise RuntimeError('G1 trainable parameter count changed')
        alpha = float((.1+.8*q.fusion_gate_logit.detach().sigmoid()).cpu())
        if not q.fusion_gate_logit.requires_grad or not math.isfinite(alpha) or not .1 < alpha < .9:
            raise RuntimeError('Resume the learned scalar alpha, do not freeze/reinitialize it')
        meta = runner.meta or {}
        if meta.get('seed') != 0:
            raise RuntimeError('Expected seed0 source and continuation')
        original = meta.get('roi_query_contract', {})
        if (original.get('model') != model.integrated_build_id
                or original.get('trainable') != 2580065 or not original.get('alpha_trainable')
                or original.get('active_rank_loss') != 'loss_winner_set'
                or meta.get('geometry_winner_contract') != geometry_contract):
            raise RuntimeError('Source checkpoint lacks matching G1 provenance')
        if meta.get('tq_initialization', {}).get('detector_checkpoint_sha256') != (
                '4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74'):
            raise RuntimeError('Source G1 used a different main detector')
        required_state = (q.fusion_gate_logit, q.high_head.weight,
                          q.visual_projection[0].weight,
                          q.roi_reader.level_gate[0].weight, q.pair_reader.level_gate[0].weight,
                          q.signed_aux[1].weight, q.signed_aux[-1].weight)
        for parameter in required_state:
            state = runner.optimizer.state.get(parameter, {})
            if (not {'step', 'exp_avg', 'exp_avg_sq'} <= set(state)
                    or float(state['step']) < 1
                    or state['exp_avg'].shape != parameter.shape
                    or state['exp_avg_sq'].shape != parameter.shape):
                raise RuntimeError('AdamW state was not restored for quality/Query/auxiliary/alpha; use resume_from')
        if not self.source.is_file() or self.source.stat().st_size == 0:
            raise RuntimeError('Missing/empty source checkpoint: use the SERVER original epoch_8.pth')
        source_hash = checkpoint_sha256(self.source)
        contract = dict(start_epoch=8, end_epoch=12, quality_lr=self.quality_lr, steps_per_epoch=454)
        previous = meta.get('tq_g1_continuation')
        if previous is None and runner.epoch != self.start_epoch:
            raise RuntimeError('Later resumed epochs require continuation provenance')
        if previous is None and abs(alpha-.5371760725975037) > 1e-6:
            raise RuntimeError('Source epoch8 alpha was not restored (expected about .53717607)')
        if previous is not None and (previous['schedule'] != contract
                or previous['source_sha256'] != source_hash
                or Path(previous['source_checkpoint']).resolve() != self.source):
            raise RuntimeError('Continuation schedule/source changed')
        output = Path(runner.work_dir).resolve()
        if output == self.source.parent:
            raise RuntimeError('Use a NEW output directory; preserve the eight-epoch run')
        if previous is not None and Path(previous['work_dir']).resolve() != output:
            raise RuntimeError('Resume an interrupted continuation in its existing output directory')
        if previous is None and (output/'g1_continuation.json').exists():
            raise RuntimeError('Continuation exists: resume its checkpoint or choose a new output directory')
        messages = meta.setdefault('hook_msgs', {})
        if previous is None and (not math.isfinite(float(messages.get('best_score', float('nan'))))
                                 or abs(float(messages['best_score'])-.401) > 1e-7):
            raise RuntimeError('Expected the G1 epoch8 state with inherited best AP=.401')
        removed = {}
        for key in ('best_ckpt', 'last_ckpt'):
            old = messages.get(key)
            if old and output not in Path(old).resolve().parents:
                removed[key] = messages.pop(key)
        for hook in runner.hooks:
            old = getattr(hook, 'best_ckpt_path', None)
            if old and output not in Path(old).resolve().parents:
                hook.best_ckpt_path = None
        if any('initial_lr' not in g for g in runner.optimizer.param_groups):
            raise RuntimeError('Missing original optimizer initial_lr values')
        active = [g for g in runner.optimizer.param_groups if any(p.requires_grad for p in g['params'])]
        if not active or any(abs(float(g['initial_lr'])-1e-4) > 1e-10 for g in active):
            raise RuntimeError('Original quality, auxiliary and alpha LR must be 1e-4')
        super().before_run(runner)
        self.base_lr = [float(g['initial_lr'])*(self.quality_lr/1e-4)
                        for g in runner.optimizer.param_groups]
        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)
        record = dict(schedule=contract, work_dir=str(output), source_checkpoint=str(self.source),
            source_sha256=source_hash, resumed_at_epoch=runner.epoch, alpha_at_resume=alpha,
            original_best_ap=.401, current_best_comparator=messages.get('best_score'),
            optimizer_state_restored=True, exact_rng_state_restored=False,
            removed_external_checkpoint_references=removed,
            scope='Extra-budget four-epoch low-LR phase; not a matched 12-epoch-from-scratch comparison')
        meta['tq_g1_continuation'] = record
        runner.meta = meta
        output.mkdir(parents=True, exist_ok=True)
        (output/'g1_continuation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        runner.logger.info('[G1-CONTINUE] %s', json.dumps(record))

    def get_lr(self, runner, base_lr):
        progress = (runner.epoch-self.start_epoch)/(self.end_epoch-self.start_epoch)
        if not 0. <= progress <= 1.:
            raise RuntimeError('Continuation epoch is outside the declared phase')
        return .5*base_lr*(1.+math.cos(math.pi*progress))

    def before_train_epoch(self, runner):
        if len(runner.data_loader) != self.steps_per_epoch:
            raise RuntimeError('Keep original sampling and single-GPU batch2 (454 steps/epoch)')
        super().before_train_epoch(runner)
        rates = sorted({g['lr'] for g in runner.optimizer.param_groups
                        if any(p.requires_grad for p in g['params'])})
        runner.logger.info('[G1-CONTINUE-LR] epoch=%d qualityAuxAndAlphaLRs=%s', runner.epoch+1, rates)
