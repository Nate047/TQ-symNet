"""Resolved config checks; optional real MMCV/CUDA two-update + one-val smoke.

The smoke uses temporary models and work dirs. It never saves/overwrites an
experiment checkpoint. CPU unit tests remain in test_roi_query/test_threshold_distribution.
"""
import argparse
import copy
import gc
import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from check_structure import load_config

FILES = ('f1_roi_content_query.py', 'f2_roi_paired_query.py', 'a0_distribution_threshold.py')


def check_configs():
    control = load_config(ROOT / 'configs/roi_role_query/a0_original_learned_alpha.py')
    for name in FILES:
        config = load_config(ROOT / 'configs/roi_role_query' / name)
        for key in ('data', 'optimizer', 'optimizer_config', 'lr_config', 'runner', 'evaluation',
                    'custom_hooks', 'fp16', 'load_from', 'resume_from'):
            assert config[key] == control[key], (name, key)
        assert config['model']['selector_gate_trainable']
        assert config['model']['quality_detach']
        assert config['model']['selector_quality_target'] == 'coco_all_iou'
        assert config['data']['samples_per_gpu'] == 2
        assert config['runner']['max_epochs'] == 8
        assert config['model']['selector_fusion_init'] == .5
        assert config['model']['quality_rank_weight'] == config['model']['selector_rank_weight'] == .1
        if name == FILES[2]:
            assert config['model']['type'] == 'TQSymNetThresholdDistribution'
            assert not any(k.startswith('roi_') for k in config['model'])
        print('PASS resolved config:', name)


def oracle_validation_smoke(core, runner, validation, val_loader, eval_hook_type):
    """Real EvalHook path with simulated epoch4 after two temporary updates.

    The supplied hook type is mmdet.core.evaluation.eval_hooks.EvalHook on the
    server. One ordinary evaluation + pre/soft128/final Oracle evaluations must
    occur. No mocked model/GT filtering is used in the server invocation.
    """
    old_epoch, old_diagnostics = runner.epoch, core.diagnostics
    original_evaluate = validation.evaluate
    records = []
    def observed_evaluate(*args, **kwargs):
        metrics = original_evaluate(*args, **kwargs)
        record = {k:float(metrics[k]) for k in ('official_tb_bbox_mAP',
            'official_tb_bbox_mAP_50', 'official_tb_bbox_mAP_75')}
        if any(not math.isfinite(v) for v in record.values()):
            raise RuntimeError('Nonfinite actual/Oracle evaluation smoke metric')
        records.append(record)
        return metrics
    try:
        core.diagnostics = True
        runner.epoch = 3  # trigger epoch4 diagnostics, NOT four trained epochs
        runner.log_buffer = SimpleNamespace(output={}, ready=False)
        validation.evaluate = observed_evaluate
        hook = eval_hook_type(val_loader, interval=1, metric='bbox', classwise=False,
            tb_official_agnostic=True, tb_class_agnostic=False, save_best=None)
        hook.before_run(runner)
        hook._do_evaluate(runner)
        if len(records) != 4:
            raise RuntimeError('Expected actual + three Oracle evaluations, got %d' % len(records))
        if (core.quality_collect or core.quality_cache or getattr(core,'distribution_val_cache',[])
                or getattr(core,'geometry_val',[])):
            raise RuntimeError('Oracle/cache cleanup did not complete')
        paths = ['selector_stages_epoch_04.json', 'selector_candidates_epoch_04.json',
                 'selector_hard_samples_epoch_04.json']
        if getattr(core.candidate_quality, 'threshold_distribution', False):
            paths.append('threshold_val_epoch_04.json')
        if hasattr(core.candidate_quality,'signed_aux'):
            paths.append('geometry_winner_val_epoch_04.json')
        if any(not (Path(runner.work_dir)/name).is_file() for name in paths):
            raise RuntimeError('Expected Oracle diagnostic export is missing')
        report = dict(simulated_epoch=4, trained_updates=2, validation_images=len(validation),
            benchmark_result=False, evaluations=dict(zip(('actual','pre','soft128','final'),records)))
        (Path(runner.work_dir)/'oracle_smoke_report.json').write_text(
            json.dumps(report,indent=2),encoding='utf-8')
        runner.logger.info('[ROI-THRESHOLD-ORACLE-SMOKE] %s',json.dumps(report))
        return report
    finally:
        validation.evaluate = original_evaluate
        core.diagnostics = old_diagnostics
        runner.epoch = old_epoch


def build_smoke(names, device, config_directory='roi_role_query'):
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import HOOKS, build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from mmdet.core.evaluation.eval_hooks import EvalHook
    from TQ_SymNet.supplement_smoke import checked_optimizer_update, state_digest

    target = torch.device(device)
    if target.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('--build-smoke needs the server CUDA/MMCV environment')
    gpu = target.index if target.index is not None else 0
    logger = logging.getLogger('tq-roi-threshold-smoke')
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    for name in names:
        set_random_seed(0, deterministic=True)
        cfg = Config.fromfile(str(ROOT / 'configs' / config_directory / name))
        import_modules_from_strings(**cfg.custom_imports)
        cfg.model.backbone.pretrained = None
        cfg.model.diagnostics = False
        core = build_detector(cfg.model)
        core.init_weights()
        core.to(device)
        model = MMDataParallel(core, device_ids=[gpu])
        optimizer = build_optimizer(core, cfg.optimizer)
        training = build_dataset(cfg.data.train)
        validation = build_dataset(dict(cfg.data.val, test_mode=True))
        train_loader = build_dataloader(training, samples_per_gpu=2, workers_per_gpu=0,
                                       dist=False, shuffle=False, seed=0)
        val_loader = build_dataloader(validation, samples_per_gpu=1, workers_per_gpu=0,
                                     dist=False, shuffle=False)
        with tempfile.TemporaryDirectory(prefix='tq_roi_threshold_smoke_') as work:
            runner = SimpleNamespace(model=model, optimizer=optimizer, meta=dict(seed=0),
                epoch=0, iter=0, max_iters=len(train_loader)*8, work_dir=work, logger=logger)
            selected = []
            for item in cfg.custom_hooks:
                if item['type'] == 'TQSymNetDebugHook':
                    continue  # full-epoch statistical logger is not needed for two updates
                options = copy.deepcopy(item); options.pop('priority', None)
                selected.append(HOOKS.build(options))
            runner.hooks = selected + [SimpleNamespace(dataloader=val_loader)]
            for hook in selected:
                hook.before_run(runner)
            model.train()
            for hook in selected:
                hook.before_train_epoch(runner)
            before = state_digest(core)
            iterator = iter(train_loader)
            for step in range(2):
                batch = next(iterator)
                outputs = model.train_step(batch, optimizer)
                update = checked_optimizer_update(core, optimizer, outputs['loss'])
                gate_grad = core.candidate_quality.fusion_gate_logit.grad
                if gate_grad is None or not torch.isfinite(gate_grad).all():
                    raise RuntimeError('Invalid alpha gradient')
                for hook in selected:
                    hook.after_train_iter(runner)
                runner.iter += 1
            projection_grad = core.candidate_quality.visual_projection[0].weight.grad
            if projection_grad is None or not bool((projection_grad != 0).any()):
                raise RuntimeError('No quality gradient reached shared visual projection')
            if state_digest(core) != before:
                raise RuntimeError('Temporary quality update changed frozen detector state')
            for hook in selected:
                hook.after_train_epoch(runner)
            validation.img_ids = validation.img_ids[:1]
            validation.data_infos = validation.data_infos[:1]
            short_loader = build_dataloader(validation, samples_per_gpu=1, workers_per_gpu=0,
                                           dist=False, shuffle=False)
            oracle_validation_smoke(core,runner,validation,short_loader,EvalHook)
            if state_digest(core) != before:
                raise RuntimeError('Oracle validation changed frozen detector state')
            print('PASS CUDA/MMCV pipeline:', name, update,
                  'detector unchanged; actual + three Oracle evaluations; NOT benchmark results')
        del iterator, outputs, batch, optimizer, model, core
        del training, validation, train_loader, val_loader, short_loader, selected, runner
        del hook, projection_grad, gate_grad
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-smoke', action='store_true')
    parser.add_argument('--config', action='append', choices=FILES)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    check_configs()
    if args.build_smoke:
        build_smoke(args.config or FILES, args.device)
