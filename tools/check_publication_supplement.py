"""Offline contract checks; --build --smoke uses the actual server MMDet API."""
import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from check_structure import load_config

FILES = ('a10_roi_original.py', 'a12_full_original.py', 'a13_roi_capacity.py',
         'gfl_joint.py', 'gflv2_joint.py', 'vfnet_joint.py')


def check_configs():
    configs = {}
    for name in FILES:
        path = ROOT/'configs/publication_supplement'/name
        cfg = load_config(path)
        model, data = cfg['model'], cfg['data']
        assert data['train']['datasets'][1]['times'] == 3
        leaves = [data['train']['datasets'][0], data['train']['datasets'][1]['dataset']]
        assert all(x['ann_file'].endswith('/TBX11K_train_only_tb.json') for x in leaves)
        assert data['val']['ann_file'].endswith('/TBX11K_val_only_tb.json')
        assert data['test']['ann_file'] == data['val']['ann_file']
        assert data['samples_per_gpu'] == 2
        assert model['test_cfg']['nms_pre'] == 1000
        assert model['test_cfg']['max_per_img'] == 100
        assert model['bbox_head']['anchor_generator']['strides'] == [4, 8, 16, 32]
        assert cfg['load_from'] is None and cfg['resume_from'] is None
        assert 'LoadLungMask' not in repr(data) and cfg['fp16'] is None
        if name.startswith('a'):
            assert cfg['runner']['max_epochs'] == 8
            assert model['selector_quality_target'] == 'direct_iou'
            assert model['quality_rank_weight'] == model['selector_rank_weight'] == 0.
            assert model['quality_hidden'] == 96 and model['quality_chunk'] == 32
            assert not model['selector_gate_trainable']
            assert model['selector_fusion_init'] == .5
            init = next(h for h in cfg['custom_hooks'] if h['type'] == 'TQOriginalCohortInitHook')
            assert init['checkpoint_sha256'].startswith('4c7ee5c68dc8')
        else:
            assert cfg['runner']['max_epochs'] == 16
            assert model['type'] == 'TQDenseHeadBaseline'
            assert not any(k.startswith(('quality_', 'selector_')) for k in model)
            assert cfg['optimizer']['lr'] == 1e-4
            assert model['train_cfg']['assigner'] == dict(type='ATSSAssigner', topk=9)
        configs[name] = cfg
        print('PASS contract', name, cfg['runner']['max_epochs'], 'epochs')
    a, b = configs['a10_roi_original.py'], configs['a13_roi_capacity.py']
    import copy
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for cfg in (a, b):
        cfg['model'].pop('experiment_tag', None)
        cfg['model'].pop('quality_pooled_mlp_hidden', None)
    assert a == b, 'A13 differs from A10 in something other than MLP width/tag'
    print('PASS A10/A13 resolved difference = MLP width + tag only')
    for method in ('gfl','gflv2','vfnet'):
        cfg = load_config(ROOT/'configs/publication_supplement'/('test_online_'+method+'.py'))
        assert cfg['data']['test']['ann_file'].endswith('/all_test.json')
        assert cfg['data']['test']['test_mode'] and cfg['custom_hooks'] == []
        assert cfg['model']['backbone']['pretrained'] is None
        assert not cfg['model']['diagnostics']
        print('PASS GT-free online config',method)
    return configs


def build_and_smoke(configs, smoke, device):
    import gc
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet.models import build_detector
    from mmdet.models.builder import build_head
    counts = {}
    for name in FILES:
        cfg = Config.fromfile(str(ROOT/'configs/publication_supplement'/name))
        # Check the real MMCV resolver agrees with the dependency-free audit.
        assert cfg.model.type == configs[name]['model']['type']
        import_modules_from_strings(**cfg.custom_imports)
        cfg.model.backbone.pretrained = None
        if name.startswith('a'):
            core = build_detector(cfg.model)
            counts[name] = sum(p.numel() for p in core.candidate_quality.parameters())
            print('PASS build', name, 'qualityParameters', counts[name])
            if smoke and name != 'a12_full_original.py':
                from TQ_SymNet.supplement_smoke import quality_update_smoke
                print('PASS supervised ROI optimizer smoke',name,quality_update_smoke(core,device))
            del core
        else:
            head_cfg = cfg.model.bbox_head.copy()
            head_cfg.update(train_cfg=cfg.model.train_cfg, test_cfg=cfg.model.test_cfg)
            head = build_head(head_cfg).to(device); head.init_weights(); head.train()
            print('PASS build', name, type(head).__name__)
            if smoke:
                feats = [torch.rand(2,256,s,s,device=device) for s in (16,8,4,2)]
                metas = [dict(img_shape=(64,64,3), pad_shape=(64,64,3),
                              ori_shape=(128,128,3), scale_factor=[.5]*4) for _ in range(2)]
                gt = [torch.tensor([[8.,10.,36.,49.]],device=device),
                      torch.empty((0,4),device=device)]
                labels = [torch.tensor([0],device=device), torch.empty(0,dtype=torch.long,device=device)]
                outs = head(feats)
                losses = head.loss(*outs, gt, labels, metas)
                total = sum(x.mean() for v in losses.values()
                            for x in (v if isinstance(v, (list,tuple)) else [v]))
                assert torch.isfinite(total), (name, losses)
                total.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
                if name == 'gflv2_joint.py':
                    assert head.reg_conf[0].weight.grad is not None
                    assert head.reg_conf[0].weight.grad.abs().sum() > 0
                with torch.no_grad():
                    raw = head.get_bboxes(*outs, metas, with_nms=False, rescale=False)
                    scaled = head.get_bboxes(*outs, metas, with_nms=False, rescale=True)
                for (box, score), (rescaled, _) in zip(raw, scaled):
                    assert score.shape[1] == 3 and torch.isfinite(score).all()
                    torch.testing.assert_close(rescaled, box*2, rtol=1e-5, atol=1e-5)
                print('PASS smoke', name, 'loss/backward/empty-GT/raw-decode/rescale')
            del head
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    ratio = counts['a13_roi_capacity.py']/counts['a12_full_original.py']
    print('A13/full allocated quality parameter ratio:', ratio)
    if not .9 <= ratio <= 1.1:
        raise RuntimeError('A13 is not within 10% of the actual full quality parameter count')


def real_batch_smoke(names, device):
    """Actual pipeline -> train_step -> optimizer -> EMA -> validation COCO.

    This is a temporary model/optimizer and a one-image validation smoke,
    not a new experiment, checkpoint, or publishable metric.
    """
    import copy
    import gc
    import logging
    import tempfile
    from types import SimpleNamespace
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import HOOKS, build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from TQ_SymNet.supplement_smoke import checked_optimizer_update, state_digest
    target_device = torch.device(device)
    if target_device.type != 'cuda':
        raise ValueError('--data-smoke requires the actual CUDA/MMCV server')
    gpu = target_device.index if target_device.index is not None else 0
    for name in names:
        set_random_seed(0, deterministic=True)
        cfg = Config.fromfile(str(ROOT/'configs/publication_supplement'/name))
        import_modules_from_strings(**cfg.custom_imports)
        quality = cfg.model.type == 'TQSymNet'
        if quality:
            cfg.model.backbone.pretrained = None  # actual M2 init hook loads the audited detector
        cfg.model.diagnostics = False
        core = build_detector(cfg.model); core.init_weights(); core.to(device)
        model = MMDataParallel(core,device_ids=[gpu])
        optimizer = build_optimizer(core,cfg.optimizer)
        training = build_dataset(cfg.data.train)
        validation = build_dataset(dict(cfg.data.val,test_mode=True))
        train_loader = build_dataloader(training,samples_per_gpu=2,workers_per_gpu=0,
                                       dist=False,shuffle=False,seed=0)
        val_loader = build_dataloader(validation,samples_per_gpu=1,workers_per_gpu=0,
                                     dist=False,shuffle=False)
        with tempfile.TemporaryDirectory(prefix='tq_publication_smoke_') as work:
            runner = SimpleNamespace(model=model,optimizer=optimizer,meta=dict(seed=0),
                epoch=0,iter=0,max_iters=len(train_loader)*cfg.runner.max_epochs,
                work_dir=work,logger=logging.getLogger('tq-real-batch-smoke'))
            selected_hooks = []
            for item in cfg.custom_hooks:
                if item['type'] in ('TQOriginalCohortInitHook','TQSupplementAuditHook',
                                    'ExpMomentumEMAHook'):
                    options = copy.deepcopy(item); options.pop('priority',None)
                    selected_hooks.append(HOOKS.build(options))
            runner.hooks = selected_hooks+[SimpleNamespace(dataloader=val_loader)]
            for hook in selected_hooks: hook.before_run(runner)
            model.train()
            for hook in selected_hooks: hook.before_train_epoch(runner)
            before = state_digest(core)
            batch = next(iter(train_loader))
            outputs = model.train_step(batch,optimizer)
            update = checked_optimizer_update(core,optimizer,outputs['loss'])
            if quality and state_digest(core) != before:
                raise RuntimeError('Real-batch update changed the frozen detector')
            if not quality and not any(p.grad is not None and bool((p.grad!=0).any())
                                       for p in core.backbone.parameters()):
                raise RuntimeError('No dense-head gradient reached the backbone')
            for hook in selected_hooks:
                if hasattr(hook,'after_train_iter'): hook.after_train_iter(runner)
                if hasattr(hook,'after_train_epoch'): hook.after_train_epoch(runner)
            # Restrict this temporary dataset to one image, retaining its GT.
            # Never report this partial validation as a benchmark score.
            validation.img_ids = validation.img_ids[:1]
            validation.data_infos = validation.data_infos[:1]
            validation_loader = build_dataloader(validation,samples_per_gpu=1,
                workers_per_gpu=0,dist=False,shuffle=False)
            model.eval()
            with torch.no_grad():
                prediction = model(return_loss=False,rescale=True,
                                   **next(iter(validation_loader)))
            values = validation.evaluate(prediction,metric='bbox',classwise=False,
                tb_official_agnostic=True,tb_class_agnostic=False,bbox_max_dets=(1,10,100))
            for key in ('official_tb_bbox_mAP','official_tb_bbox_mAP_50','official_tb_bbox_mAP_75'):
                if key not in values or not torch.isfinite(torch.tensor(values[key])):
                    raise RuntimeError('Invalid real-pipeline evaluation output')
            print('PASS REAL-BATCH',name,'actual hooks/train_step/update/EMA/val/COCO',
                  update,'partial validation only; NOT experiment results')
        del model,core,optimizer,training,validation,train_loader,val_loader,validation_loader
        del outputs,batch,prediction
        gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--data-smoke', action='store_true',
                   help='Use real server data/checkpoints for one train/update/validation batch')
    p.add_argument('--config', action='append', choices=FILES,
                   help='Select real-data smoke configs; default A13 and GFLv2')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    for path in (ROOT/'configs/publication_supplement').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'))
    configs = check_configs()
    if args.build or args.smoke:
        build_and_smoke(configs, args.smoke, args.device)
    if args.data_smoke:
        real_batch_smoke(args.config or ['a13_roi_capacity.py','gflv2_joint.py'],args.device)
    print('PASS supplement checks; head/ROI runtime=%s real-data pipeline=%s' %
          (bool(args.build or args.smoke),args.data_smoke))


if __name__ == '__main__':
    main()
