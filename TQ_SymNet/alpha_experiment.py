"""Original 599/200 cohort audit and scalar-alpha diagnostics.

Separate from the 479/120/200 hooks: never silently bypass their contract.
No scoring or candidate-selection formula is changed by these hooks.
"""
import json
from pathlib import Path
import torch
from mmcv.runner import HOOKS, Hook
from .checkpoints import load_detector_checkpoint
from .data_contract import sha256_file, verify_dataset_contract


@HOOKS.register_module()
class TQOriginalCohortInitHook(Hook):
    def __init__(self, checkpoint, checkpoint_sha256, train_annotation,
                 val_annotation, train_sha256, val_sha256):
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.paths = dict(train=train_annotation, val=val_annotation)
        self.hashes = dict(train=train_sha256, val=val_sha256)

    def before_run(self, runner):
        identities = []
        for phase, count, boxes in (('train', 599, 902), ('val', 200, 309)):
            if sha256_file(self.paths[phase]) != self.hashes[phase]:
                raise RuntimeError('Original-cohort annotation hash mismatch: ' + phase)
            doc = json.loads(Path(self.paths[phase]).read_text(encoding='utf-8'))
            names = {x['file_name'] for x in doc['images']}
            if len(names) != count or len(doc['images']) != count or len(doc['annotations']) != boxes:
                raise RuntimeError('Unexpected original-cohort size: ' + phase)
            identities.append(names)
        if identities[0] & identities[1]:
            raise RuntimeError('Train and validation share filenames')
        if sha256_file(self.checkpoint) != self.checkpoint_sha256:
            raise RuntimeError('Wrong detector checkpoint; expected original M2 epoch 13, not a new split')
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if any(p.requires_grad for n, p in model.named_parameters()
               if not n.startswith('candidate_quality.')):
            raise RuntimeError('Detector must remain frozen')
        contract = dict(protocol='original_TBX11K_TB_599_200', **self.hashes)
        if runner.meta is None:
            runner.meta = {}
        previous = runner.meta.get('tq_original_cohort')
        if previous is not None and previous != contract:
            raise RuntimeError('Resume cohort mismatch')
        if runner.epoch == 0 and runner.iter == 0:
            _, audit = load_detector_checkpoint(model, self.checkpoint)
            runner.meta['tq_initialization'] = dict(
                detector_checkpoint=self.checkpoint,
                detector_checkpoint_sha256=self.checkpoint_sha256,
                quality_initialization='random_from_command_seed', **audit)
        elif previous is None:
            raise RuntimeError('Resume requires original-cohort metadata')
        runner.meta['tq_original_cohort'] = contract
        runner.logger.info('[ORIGINAL-COHORT] train=599/902 val=200/309 '
                           'detectorSHA=%s resumed=%s', self.checkpoint_sha256,
                           bool(runner.epoch or runner.iter))
        Path(runner.work_dir).mkdir(parents=True, exist_ok=True)
        (Path(runner.work_dir) / 'original_cohort.json').write_text(
            json.dumps(dict(contract=contract, initialization=runner.meta.get('tq_initialization')), indent=2),
            encoding='utf-8')

    def before_train_epoch(self, runner):
        verify_dataset_contract(runner.data_loader.dataset, self.hashes['train'], 'train')
        for hook in runner.hooks:
            if hasattr(hook, 'dataloader'):
                verify_dataset_contract(hook.dataloader.dataset, self.hashes['val'], 'val')


@HOOKS.register_module()
class TQScalarAlphaAuditHook(Hook):
    """Validate optimizer wiring; log actual gradients and parameter movement."""
    def __init__(self, interval=50):
        self.interval = int(interval)
        if self.interval < 1:
            raise ValueError('interval must be positive')

    def before_run(self, runner):
        self.model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.gate = self.model.candidate_quality.fusion_gate_logit
        groups = [g for g in runner.optimizer.param_groups if any(p is self.gate for p in g['params'])]
        if self.gate.requires_grad and len(groups) != 1:
            raise RuntimeError('Trainable scalar gate is not in exactly one optimizer group')
        if not self.model.selector_learned_gate or not self.model.selector_quality_rank_only:
            raise RuntimeError('Expected bounded scalar fusion and quality-only rank')
        if self.model.selector_fusion_loss_weight != 0:
            raise RuntimeError('Do not double-count fusion calibration; existing residual loss trains alpha')
        self.group = groups[0] if groups else None
        self.path = Path(runner.work_dir) / 'alpha_history.jsonl'
        self.score_stats = []
        original_loss = self.model._residual_loss
        def observed_loss(base_z, quality_z, target, actual, matched):
            # Scalar reductions only; no extra forward, RNG draw or supervision.
            with torch.no_grad():
                a, b = base_z.detach().float(), quality_z.detach().float()
                self.score_stats.append(torch.stack((a.sum(), a.square().sum(),
                    b.sum(), b.square().sum(), a.new_tensor(float(a.numel())))))
            return original_loss(base_z, quality_z, target, actual, matched)
        self.model._residual_loss = observed_loss
        runner.logger.info('[ALPHA-BUILD] trainable=%s initial=%.8f range=(.1,.9) '
            'gateLoss=.1*balancedBCE+.01*(alpha-.5)^2 qualityInputDetached=True '
            'qualityRankOnly=True optimizerGroups=%d lr=%s', self.gate.requires_grad,
            float((.1 + .8 * self.gate.detach().sigmoid()).cpu()), len(groups),
            self.group['lr'] if self.group else None)

    def before_train_epoch(self, runner):
        self.start = float((.1 + .8 * self.gate.detach().sigmoid()).cpu())
        self.grad_sum = self.grad_abs = 0.
        self.grad_steps = self.nonzero = 0
        self.score_stats = []

    def after_train_iter(self, runner):
        grad = self.gate.grad
        if grad is not None:
            value = float(grad.detach().cpu())
            if not torch.isfinite(grad).all():
                raise RuntimeError('Nonfinite alpha gradient')
            self.grad_sum += value
            self.grad_abs += abs(value)
            self.grad_steps += 1
            self.nonzero += int(value != 0.)
        alpha = float((.1 + .8 * self.gate.detach().sigmoid()).cpu())
        if not .1 <= alpha <= .9:
            raise RuntimeError('Nonfinite/out-of-range alpha')
        if (runner.iter + 1) % self.interval == 0:
            runner.logger.info('[ALPHA-ITER] epoch=%d iter=%d alpha=%.8f '
                'gradSteps=%d nonzeroGradSteps=%d meanAbsGrad=%.6g lr=%s',
                runner.epoch + 1, runner.iter + 1, alpha, self.grad_steps,
                self.nonzero, self.grad_abs / max(1, self.grad_steps),
                self.group['lr'] if self.group else None)

    def after_train_epoch(self, runner):
        alpha = float((.1 + .8 * self.gate.detach().sigmoid()).cpu())
        row = dict(epoch=runner.epoch + 1, trainable=self.gate.requires_grad,
                   alpha=alpha, alpha_start=self.start, delta=alpha-self.start,
                   gradient_steps=self.grad_steps, nonzero_gradient_steps=self.nonzero,
                   mean_gradient=self.grad_sum/max(1,self.grad_steps),
                   mean_abs_gradient=self.grad_abs/max(1,self.grad_steps),
                   near_bound=alpha < .12 or alpha > .88)
        if self.score_stats:
            sm, ss, qm, qs, n = torch.stack(self.score_stats).sum(0).cpu().tolist()
            n = max(n, 1.)
            row['sampled_logits'] = dict(count=int(n), sym_mean=sm/n, quality_mean=qm/n,
                sym_std=max(ss/n-(sm/n)**2,0.)**.5,
                quality_std=max(qs/n-(qm/n)**2,0.)**.5)
        self.score_stats.clear()
        runner.logger.info('[ALPHA-EPOCH] %s', json.dumps(row))
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        runner.meta['tq_scalar_alpha'] = row
        if self.gate.requires_grad and not self.nonzero:
            runner.logger.warning('[ALPHA-NO-UPDATE] No nonzero gradients observed this epoch')
