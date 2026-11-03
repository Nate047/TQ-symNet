"""Dependency-free config contracts; optional real server build/forward."""
import argparse
import copy
import gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from check_structure import load_config

FILES = ['b0_transnext_small.py','b1_transnext_tiny.py','b2_swin_tiny.py']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Real CUDA pretrained load + forward')
    args = parser.parse_args()
    configs = []
    for name in FILES:
        path = ROOT/'configs/backbone_capacity'/name
        cfg = load_config(path)
        model = cfg['model']
        assert model['type'] == 'TQDetector'
        assert not any(k.startswith('quality_') or k.startswith('selector_') for k in model)
        assert cfg['load_from'] is None and cfg['resume_from'] is None
        assert cfg['runner']['max_epochs'] == 16
        assert cfg['data']['samples_per_gpu'] == 2 and cfg['fp16'] is None
        assert cfg['data']['test'] == cfg['data']['val']
        assert cfg['data']['val']['ann_file'].endswith('/TBX11K_val_only_tb.json')
        assert cfg['data']['train']['datasets'][1]['times'] == 3
        assert [h['type'] for h in cfg['custom_hooks']] == ['TQSupplementAuditHook','ExpMomentumEMAHook']
        assert cfg['evaluation']['save_best'] == 'official_tb_bbox_mAP'
        assert all(k not in model['backbone'] for k in ('candidate_quality','side_refine'))
        invariant = copy.deepcopy(cfg)
        for key in ('_base_','work_dir'):
            invariant.pop(key, None)
        invariant['model'].pop('backbone')
        invariant['model'].pop('experiment_tag')
        invariant['model']['neck'].pop('in_channels')
        configs.append((path, cfg, invariant))
        assert invariant == configs[0][2], 'Unintended non-backbone change: '+name
        print('PASS config',name)
    assert configs[1][1]['model']['backbone']['pretrain_size'] == 224
    assert configs[2][1]['model']['backbone']['convert_weights'] is True
    assert configs[2][1]['model']['neck']['in_channels'] == [96,192,384,768]
    if not args.smoke:
        return
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet.models import build_detector
    for path, _, _ in configs:
        cfg = Config.fromfile(str(path))
        import_modules_from_strings(**cfg.custom_imports)
        model = build_detector(cfg.model)
        model.init_weights()
        model.cuda().eval()
        with torch.no_grad():
            features = model.backbone(torch.zeros(1,3,512,512,device='cuda'))
            assert [x.shape[1] for x in features] == cfg.model.neck.in_channels
            pyramid = model.neck(features)
            assert [tuple(x.shape[1:]) for x in pyramid] == [
                (256,128,128),(256,64,64),(256,32,32),(256,16,16)]
            cls, box = model.bbox_head(pyramid,c2_feat=features[0])[:2]
            assert len(cls) == len(box) == 4
            assert all(x.shape[1] == 18 for x in cls)
            assert all(x.shape[1] == 36 for x in box)
        print('PASS CUDA pretrained/build/forward',path.name,
              'backbone_params=',sum(p.numel() for p in model.backbone.parameters()))
        del features,pyramid,cls,box,model
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
