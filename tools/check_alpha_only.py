"""Dependency-light checks of the actual alpha-only forward and freeze path."""
import ast
import hashlib
import importlib.util
import types
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def extract(file, owner, names, ns):
    tree = ast.parse(file.read_text(encoding='utf-8'))
    body = tree.body if owner is None else next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == owner).body
    nodes = [x for x in body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), 'exec'), ns)


def overlaps(a, b):
    wh = (torch.minimum(a[:, None, 2:], b[None, :, 2:])-torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0)
    inter = wh.prod(2)
    return inter / (((a[:, 2:]-a[:, :2]).prod(1)[:, None]+(b[:, 2:]-b[:, :2]).prod(1)[None])-inter).clamp_min(1e-12)


def main():
    ns = dict(torch=torch, F=F, hashlib=hashlib, GATE='candidate_quality.fusion_gate_logit', bbox_overlaps=overlaps)
    extract(ROOT/'TQ_SymNet/models/selection.py', None, {'balanced_mean','all_iou_target','gaussian_soft_select'}, ns)
    extract(ROOT/'TQ_SymNet/models/selection.py', 'RetinaNAMHighQualitySelector', {'_selector_sample'}, ns)
    extract(ROOT/'TQ_SymNet/models/detail.py', 'RetinaNAMDetailSelector', {'_residual_loss'}, ns)
    extract(ROOT/'TQ_SymNet/alpha_only.py', None, {'frozen_state_hash'}, ns)
    extract(ROOT/'TQ_SymNet/alpha_only.py', 'TQSymNetAlphaOnly', {'train','forward_train'}, ns)

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(.8))
            self.candidate_quality = torch.nn.Linear(4, 2)
            self.candidate_quality.register_parameter('fusion_gate_logit', torch.nn.Parameter(torch.tensor(.18)))
            for n,p in self.named_parameters():p.requires_grad_(n == ns['GATE'])
            self.selector_learned_gate = True
            self.quality_samples_per_bin = 96
            self.bbox_head = types.SimpleNamespace(test_cfg=types.SimpleNamespace(score_thr=.05))
            self.boxes = torch.tensor([[0.,0.,1.,1.],[0.,0.,2.,2.]])
            self.sym = torch.tensor([[.6,.1,0.],[.7,.1,0.]])
        def _main(self,*args):
            self.backbone(torch.ones(2,4))
            return None,[(self.boxes,self.sym)],None
        def _v330_unwrap_original_images(self,*args):return [None]
        def _image_memory(self,*args):return self.candidate_quality(torch.ones(2,4))
        def _normalized(self,b,m):return b
        def _score_candidates(self,b,m,s):return torch.tensor([[3.,3.],[-2.,-2.]])[:len(b)],None,None
        def _quality_target(self,u):return ns['all_iou_target'](u)
        def _final_score(self,s,z,r):
            a=.1+.8*self.candidate_quality.fusion_gate_logit.sigmoid()
            return ((1-a)*torch.logit(s[:,:-1].max(1)[0])+a*z).sigmoid()

    for name in ['train','forward_train','_selector_sample','_residual_loss']:
        setattr(Fake,name,ns[name])
    model=Fake();model.train()
    assert not any(m.training for m in model.modules())
    before=ns['frozen_state_hash'](model);start=model.candidate_quality.fusion_gate_logit.detach().clone()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.)
    for _ in range(3):
        optimizer.zero_grad()
        loss=model.forward_train(None,[{}],[torch.tensor([[0.,0.,1.,1.]])],[None])['loss_alpha_calibration']
        loss.backward();assert model.candidate_quality.fusion_gate_logit.grad.abs()>0
        assert all(p.grad is None for n,p in model.named_parameters() if n!=ns['GATE'])
        optimizer.step()
    assert ns['frozen_state_hash'](model)==before
    assert not torch.equal(start,model.candidate_quality.fusion_gate_logit)
    optimizer.zero_grad();model.forward_train(None,[{}],[torch.empty(0,4)],[None])['loss_alpha_calibration'].backward()
    model.boxes=torch.empty(0,4);model.sym=torch.empty(0,3)
    optimizer.zero_grad();empty=model.forward_train(None,[{}],[torch.empty(0,4)],[None])['loss_alpha_calibration']
    empty.backward();assert model.candidate_quality.fusion_gate_logit.grad.item()==0
    assert ns['frozen_state_hash'](model)==before
    # Same production Soft-NMS function must accept an empty set and preserve scores.
    ids,values=ns['gaussian_soft_select'](torch.empty(0,4),torch.empty(0));assert not len(ids)
    ids,values=ns['gaussian_soft_select'](torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.],[2.,2.,3.,3.]]),torch.tensor([.9,.8,.7]))
    assert ids.tolist()==[0,2,1] and values[2]<.8
    spec=importlib.util.spec_from_file_location('structure',ROOT/'tools/check_structure.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=module.load_config(ROOT/'configs/publication_alpha/a2_alpha_only.py')
    assert cfg['model']['type']=='TQSymNetAlphaOnly'
    assert cfg['optimizer']==dict(type='AdamW',lr=1e-4,weight_decay=0.)
    assert cfg['lr_config']==dict(policy='Fixed')
    assert cfg['load_from'] is None and cfg['resume_from'] is None
    assert cfg['custom_hooks'][0]['expected_epoch']==7
    for file in ['TQ_SymNet/alpha_only.py','configs/publication_alpha/a2_alpha_only.py','tools/analysis_tools/sweep_scalar_alpha.py']:
        ast.parse((ROOT/file).read_text(),feature_version=(3,8))
    print('PASS actual alpha-only forward: nonzero scalar update, frozen weights/BN buffers, eval mode, empty/no-GT, config and Py3.8 syntax')
    print('Full MMCV/CUDA checkpoint build remains a server runtime check.')


if __name__=='__main__':main()
