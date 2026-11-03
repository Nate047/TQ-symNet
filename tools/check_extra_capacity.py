"""Check all six extra configs; optional real pretrained CUDA forward/backward.

Run from the standalone project root. --smoke is per-config so one missing
custom operator does not prevent testing unrelated families.
"""
import argparse
import ast
import copy
import hashlib
import json
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from check_structure import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke',help='Config basename, e.g. b7_efficientvim_m3')
    p.add_argument('--backward',action='store_true',help='Also test FP32 training gradients')
    p.add_argument('--environment',action='store_true',help='Report actual server stack without building')
    args = p.parse_args()
    folder = ROOT/'configs/backbone_capacity'
    expected = None
    configs = {}
    for path in sorted(folder.glob('b[3-8]_*.py')):
        cfg=load_config(path)
        model=cfg['model']
        assert model['type']=='TQDetector'
        assert model['backbone']['type']=='TQEfficientBackbone'
        assert cfg['runner']['max_epochs']==16 and cfg['data']['samples_per_gpu']==2
        assert cfg['load_from'] is None and cfg['resume_from'] is None
        assert cfg['data']['test']==cfg['data']['val']
        assert cfg['evaluation']['save_best']=='official_tb_bbox_mAP'
        invariant=copy.deepcopy(cfg)
        for key in ('_base_','work_dir'): invariant.pop(key,None)
        invariant['model'].pop('backbone')
        invariant['model'].pop('experiment_tag')
        invariant['model']['neck'].pop('in_channels')
        if expected is None: expected=invariant
        assert invariant==expected,path.name
        configs[path.stem]=(path,cfg)
        print('PASS config',path.name)
    assert len(configs)==6
    source_root=ROOT/'third_party/capacity_backbones'
    lock=json.loads((source_root/'SOURCES.json').read_text(encoding='utf-8'))
    for rel,digest in lock['sha256'].items():
        path=source_root/rel
        if not path.is_file(): raise FileNotFoundError(path)
        text=path.read_text(encoding='utf-8-sig')
        assert hashlib.sha256(text.encode('utf-8')).hexdigest()==digest,rel
        ast.parse(text,filename=rel)
    print('PASS pinned official source hashes',len(lock['sha256']))
    if args.environment or args.smoke:
        import torch
        from importlib import metadata
        print('SERVER Python=%s torch=%s CUDA=%s' %
              (sys.version.split()[0],torch.__version__,torch.version.cuda))
        for dist in ('mmcv-full','mmcv','timm','triton','natten','einops','PyWavelets'):
            try: print('SERVER',dist,metadata.version(dist))
            except metadata.PackageNotFoundError: print('SERVER',dist,'NOT INSTALLED')
        amp=getattr(torch,'amp',None)
        if not hasattr(amp,'custom_fwd'):
            print('BLOCKED LSNet: official SKA requires torch.amp.custom_fwd; '
                  'do not upgrade the working server environment in place.')
    if not args.smoke: return
    if args.smoke not in configs: raise ValueError('Unknown --smoke config')
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet.models import build_detector
    cfg=Config.fromfile(str(configs[args.smoke][0]))
    import_modules_from_strings(**cfg.custom_imports)
    model=build_detector(cfg.model)
    model.init_weights()
    model.cuda().eval()
    with torch.no_grad():
        raw=model.backbone(torch.randn(1,3,512,512,device='cuda'))
        assert [x.shape[1] for x in raw]==cfg.model.neck.in_channels
        pyramid=model.neck(raw)
        cls,box=model.bbox_head(pyramid,c2_feat=raw[0])[:2]
        assert len(cls)==len(box)==4
        assert all(x.shape[1]==18 for x in cls)
        assert all(x.shape[1]==36 for x in box)
    print('PASS full CUDA 512 forward',args.smoke,[tuple(x.shape) for x in raw])
    del raw,pyramid,cls,box
    if args.backward:
        model.train()
        image=torch.randn(2,3,512,512,device='cuda')
        metas=[dict(img_shape=(512,512,3),ori_shape=(512,512,3),
            pad_shape=(512,512,3),scale_factor=1.,flip=False) for _ in range(2)]
        gt=[image.new_tensor([[100,90,240,210]]) for _ in range(2)]
        labels=[torch.zeros(1,dtype=torch.long,device='cuda') for _ in range(2)]
        losses=model.forward_train(image,metas,gt,labels)
        total=sum(v.mean() if isinstance(v,torch.Tensor) else sum(t.mean() for t in v)
                  for k,v in losses.items() if 'loss' in k)
        assert torch.isfinite(total)
        total.backward()
        grads=[param.grad for param in model.backbone.parameters() if param.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        print('PASS FP32 batch2 full-detector backward',float(total.detach()))


if __name__=='__main__': main()
