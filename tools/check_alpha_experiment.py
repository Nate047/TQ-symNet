"""CPU regression checks for actual scalar fusion/loss methods; no MMCV needed."""
import ast
import importlib.util
import types
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]


def extract(path, owner, names, namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    body=tree.body if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner).body
    nodes=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=len(names):raise AssertionError('Missing actual source method')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def main():
    ns=dict(torch=torch,F=F)
    extract(ROOT/'TQ_SymNet/models/selection.py',None,{'balanced_mean'},ns)
    extract(ROOT/'TQ_SymNet/models/detail.py','RetinaNAMDetailSelector',{'_final_logit','_residual_loss'},ns)
    gate=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
    model=types.SimpleNamespace(selector_adaptive_subtype_gate=False,
        selector_tb_fusion='max',selector_learned_gate=True,
        candidate_quality=types.SimpleNamespace(fusion_gate_logit=gate))
    sym=torch.tensor([[.8,.2,0.],[.4,.7,0.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([2.,-2.],dtype=torch.float64,requires_grad=True)
    base=torch.logit(sym[:,:-1]).max(1)[0]
    actual=torch.tensor([.9,.2],dtype=torch.float64)
    matched=torch.tensor([0,0]);target=torch.tensor([.9,.1],dtype=torch.float64)
    actual_score=ns['_final_logit'](model,sym,q,q)
    assert torch.equal(actual_score,.5*base+.5*q), 'Fixed .5 parity'
    loss=ns['_residual_loss'](model,base,q,target,actual,matched)
    optimizer=torch.optim.AdamW([dict(params=[gate],lr=1e-4,weight_decay=0.)])
    loss.backward()
    assert gate.grad is not None and gate.grad.abs()>0
    assert q.grad is None and sym.grad is None, 'Calibration must only update gate'
    before=gate.detach().clone();optimizer.step()
    assert not torch.equal(before,gate.detach()), 'Actual optimizer update'
    gate.requires_grad_(False)
    fixed=ns['_residual_loss'](model,base,q,target,actual,matched)
    assert not fixed.requires_grad, 'Fixed control must not train quality with calibration'
    spec=importlib.util.spec_from_file_location('structure',ROOT/'tools/check_structure.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a=module.load_config(ROOT/'configs/publication_alpha/a0_fixed.py')
    b=module.load_config(ROOT/'configs/publication_alpha/a1_learned.py')
    a['model'].pop('experiment_tag');b['model'].pop('experiment_tag')
    assert not a['model'].pop('selector_gate_trainable')
    assert b['model'].pop('selector_gate_trainable')
    assert a==b, 'A0/A1 must differ only in gate trainability and tag'
    assert a['load_from'] is None and a['resume_from'] is None
    assert '479' not in a['detector_checkpoint']
    assert a['model']['selector_fusion_loss_weight']==0
    assert a['data']['train']['datasets'][1]['times']==3
    assert a['runner']['max_epochs']==8
    assert a['model']['selector_quality_target']=='coco_all_iou'
    assert a['model']['quality_detach']
    control=module.load_config(ROOT/'configs/publication_alpha/b1_soft.py')
    assert control['data']['val']['ann_file']==a['data']['val']['ann_file']
    assert control['model']['baseline_sigma']==a['model']['selector_sigma']
    assert control['model']['baseline_topk']==128
    for name in ('benchmark_cost.py','check_alpha_experiment.py'):
        ast.parse((ROOT/'tools'/name).read_text(encoding='utf-8'),feature_version=(3,8))
    ast.parse((ROOT/'TQ_SymNet/alpha_experiment.py').read_text(encoding='utf-8'),feature_version=(3,8))
    print('PASS actual scalar fusion .5 parity, gate-only gradients, optimizer movement, matched configs and Python 3.8 syntax')
    print('NOT TESTED: MMCV/CUDA full-model build, actual server training or latency.')


if __name__=='__main__':main()
