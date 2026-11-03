"""Matched recipe check + optional real CUDA/MMCV/Oracle smoke for new groups."""
import argparse
from pathlib import Path
from check_structure import load_config
from check_roi_threshold import build_smoke

ROOT=Path(__file__).resolve().parents[1]
FILES=('g1_f2_geometry_winner.py','r0_roi_complete_recipe.py')


def check_configs():
    old=load_config(ROOT/'configs/roi_role_query/f2_roi_paired_query.py')
    for name in FILES:
        config=load_config(ROOT/'configs/quality_choice'/name)
        for key in ('data','optimizer','optimizer_config','lr_config','runner','evaluation'):
            assert old[key]==config[key],(name,key)
        assert config['data']['samples_per_gpu']==2 and config['runner']['max_epochs']==8
        assert config['fp16'] is None and config['load_from'] is None and config['resume_from'] is None
        assert config['model']['quality_detach'] and config['model']['selector_gate_trainable']
        assert config['model']['selector_quality_target']=='coco_all_iou'
        assert config['model']['selector_rank_weight']==.1
        print('PASS complete-recipe contract:',name)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--build-smoke',action='store_true')
    parser.add_argument('--config',action='append',choices=FILES)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    check_configs()
    if args.build_smoke:build_smoke(args.config or FILES,args.device,config_directory='quality_choice')
