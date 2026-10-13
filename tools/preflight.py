"""Audit resolved experimental contracts; optional server file/build validation."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_structure import load_config
from TQ_SymNet.data_contract import load_split_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--build', action='store_true', help='Requires the server MMCV/CUDA environment')
    parser.add_argument('--config', action='append', default=[])
    args = parser.parse_args()
    folder = ROOT / 'configs'
    paths = [Path(p) for p in args.config] if args.config else sorted(
        list(folder.glob('a[0-9]*.py')) + list(folder.glob('b[0-2]_*.py')) + list(folder.glob('m[0-3]_*.py')))
    rows = []
    annotations = {}
    for path in paths:
        cfg = load_config(path)
        reference = load_config(ROOT / cfg.get('matched_reference', 'configs/a0_full.py'))
        model = cfg['model']
        quality = model['type'] == 'TQSymNet'
        contract, split_paths = load_split_contract(ROOT / cfg['split_manifest'])
        assert cfg['split_seed'] == contract['split_seed'] == 0
        assert Path(cfg['data']['val']['ann_file']).resolve() == split_paths['val']
        assert Path(cfg['data']['test']['ann_file']).resolve() == split_paths['test']
        train = cfg['data']['train']
        train_leaves = (train['datasets'][0], train['datasets'][1]['dataset'])
        assert all(Path(d['ann_file']).resolve() == split_paths['train'] for d in train_leaves)
        assert 'LoadLungMask' not in repr(cfg['data']), path
        assert model['test_cfg']['nms_pre'] == 1000 and model['test_cfg']['max_per_img'] == 100
        assert model['bbox_head']['anchor_generator']['strides'] == [4,8,16,32]
        if model['neck']['type'] == 'FPN':
            forbidden = ('residual_init','semantic_residual_max','window_size','use_top_down',
                         'learnable_fusion_weights','freqfusion_top_down','reg_detail_bypass')
            assert not any(k in model['neck'] for k in forbidden), path
        if quality:
            assert cfg['load_from'] is None, 'Matched groups must not load a pretrained quality network'
            init_hook = next(h for h in cfg['custom_hooks'] if h['type'] == 'TQDetectorInitHook')
            ref_hook = next(h for h in reference['custom_hooks'] if h['type'] == 'TQDetectorInitHook')
            assert cfg['custom_hooks'][0]['type'] == 'TQSplitContractHook'
            assert init_hook['checkpoint'] == ref_hook['checkpoint']
            for key in ('data','optimizer','lr_config','runner'):
                assert cfg[key] == reference[key], (path.name,key)
        if args.server_files:
            needed = []
            def visit(value):
                if isinstance(value, dict):
                    needed.extend(value[k] for k in ('ann_file','img_prefix') if k in value)
                    for child in value.values(): visit(child)
                elif isinstance(value, list):
                    for child in value: visit(child)
            visit(cfg['data'])
            if quality: needed.append(init_hook['checkpoint'])
            elif model['backbone'].get('pretrained'): needed.append(model['backbone']['pretrained'])
            for name in set(needed):
                if not Path(name).exists(): raise FileNotFoundError(name)
                if str(name).endswith('.json') and name not in annotations:
                    document = json.loads(Path(name).read_text(encoding='utf-8-sig'))
                    if 'images' in document:
                        categories = {c['id']: c['name'] for c in document.get('categories', [])}
                        annotations[name] = dict(
                            images=len(document['images']), boxes=len(document.get('annotations', [])),
                            boxes_by_category=dict(Counter(categories.get(a['category_id'], str(a['category_id']))
                                for a in document.get('annotations', []))))
        row = dict(config=path.name, role=cfg.get('experiment_role'),
                   split_seed=contract['split_seed'], split_images=contract['images'],
                   backbone=model['backbone']['type'], neck=model['neck']['type'],
                   quality=quality, epochs=cfg['runner']['max_epochs'],
                   switches={k:v for k,v in model.items() if k in (
                       'quality_arch','quality_texture','quality_lbp','quality_gabor','quality_gaussian',
                       'selector_edges','selector_detail','selector_rank_weight','selector_quality_target',
                       'selector_gate_trainable','selector_fusion_init','inference_nms')})
        if args.build:
            import gc
            from mmcv import Config
            from mmcv.utils import import_modules_from_strings
            from mmdet.models import build_detector
            resolved = Config.fromfile(str(path))
            import_modules_from_strings(**resolved.custom_imports)
            check_p2t = args.server_files and model['backbone']['type'] == 'TQP2TSmall'
            if not check_p2t:
                resolved.model.backbone.pretrained = None
                if 'init_cfg' in resolved.model.backbone: resolved.model.backbone.init_cfg = None
            built = build_detector(resolved.model)
            if check_p2t:
                import torch
                initial = {k:v.detach().cpu().clone() for k,v in built.backbone.state_dict().items()}
                built.init_weights()
                changed = [k for k,v in built.backbone.state_dict().items()
                           if not torch.equal(initial[k],v.detach().cpu())]
                if changed:
                    raise RuntimeError('Detector init overwrote P2T pretraining: '+repr(changed[:12]))
                row['pretrained_audit'] = built.backbone.pretrained_audit
                if row['pretrained_audit'] is None:
                    raise RuntimeError('P2T build completed without pretrained weights')
                row['pretrained_preserved_after_detector_init'] = True
                del initial
            row['constructed_parameters'] = sum(p.numel() for p in built.parameters())
            del built
            gc.collect()
        rows.append(row)
    print(json.dumps(rows, indent=2))
    if annotations:
        print('DATASET_COUNTS', json.dumps(annotations, ensure_ascii=False, indent=2))
    print('PASS: resolved groups, equal quality data/budget, checkpoint isolation, FPN keys, and candidate contracts')
    if not args.build: print('Full model construction was not tested; use --build on the server.')


if __name__ == '__main__':
    main()
