"""CPU regression tests for the real retained-checkpoint API, without MMCV/CUDA.

The stub only replaces checkpoint file reading; loading/audit logic is real.
Position-cache fixtures use the actual backbone's deterministic CPB function.
"""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tq_checkpoint_contract',ROOT/'TQ_SymNet/checkpoints.py')
contract=importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
tree=ast.parse((ROOT/'mmdet/models/backbones/transnext_cuda.py').read_text(encoding='utf-8-sig'))
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
              and node.name=='get_relative_position_cpb')
namespace=dict(torch=torch,F=F)
exec(compile(ast.Module(body=[function],type_ignores=[]),'<real-cpb-function>','exec'),namespace)
make_cpb=namespace['get_relative_position_cpb']


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Module()
        self.backbone.weight=nn.Parameter(torch.zeros(2,2))
        for level in range(1,5):
            index,coords=make_cpb((4,4),(2,2),(4,4),device=torch.device('cpu'))
            self.backbone.register_buffer('relative_pos_index%d'%level,index,persistent=False)
            self.backbone.register_buffer('relative_coords_table%d'%level,coords,persistent=False)
        self.neck=nn.Linear(2,2)
        self.bbox_head=nn.Linear(2,2)
        self.candidate_quality=nn.Module()
        self.candidate_quality.quality=nn.Linear(2,1)
        self.candidate_quality.tb_lse_temperature_raw=nn.Parameter(torch.zeros(()),requires_grad=False)
        self.selector_tb_fusion='max'


def saved_state(model):
    state={name:value.clone()+1 for name,value in model.state_dict().items()}
    for name,buffer in model.named_buffers():
        state[name]=buffer.clone()
    return state


def load_public(model,state):
    checkpoint=dict(state_dict=state,meta=dict(epoch=7))
    runner=types.ModuleType('mmcv.runner')
    reader=types.ModuleType('mmcv.runner.checkpoint')
    reader._load_checkpoint=lambda *args,**kwargs:checkpoint
    def old_load(model,*args,**kwargs):
        normalized={key[7:] if key.startswith('module.') else key:value for key,value in state.items()}
        model.load_state_dict(normalized,strict=False)
        return checkpoint
    runner.load_checkpoint=old_load
    mmcv=types.ModuleType('mmcv');mmcv.runner=runner;runner.checkpoint=reader
    with patch.dict(sys.modules,{'mmcv':mmcv,'mmcv.runner':runner,'mmcv.runner.checkpoint':reader}):
        return contract.load_retained_checkpoint(model,'fixture.pth')


class Checks(unittest.TestCase):
    def test_exact_user_failure_old_persistent_cache(self):
        model=Toy();state=saved_state(model)
        before={name:value.clone() for name,value in model.named_buffers()}
        _,audit=load_public(model,state)
        for name,value in model.named_parameters():
            torch.testing.assert_close(value,state[name])
        for name,value in model.named_buffers():
            torch.testing.assert_close(value,before[name])
        self.assertEqual(len(audit['ignored_rebuilt_position_caches']),8)

    def test_clean_checkpoint(self):
        model=Toy();state={name:value.clone()+1 for name,value in model.state_dict().items()}
        _,audit=load_public(model,state)
        self.assertEqual(audit['ignored_rebuilt_position_caches'],[])

    def test_module_prefix(self):
        model=Toy();state={'module.'+key:value for key,value in saved_state(model).items()}
        _,audit=load_public(model,state)
        self.assertEqual(len(audit['ignored_rebuilt_position_caches']),8)

    def test_missing_learned_weight_rejected(self):
        model=Toy();state=saved_state(model);del state['candidate_quality.quality.weight']
        with self.assertRaises(RuntimeError):load_public(model,state)

    def test_unknown_backbone_parameter_rejected(self):
        model=Toy();state=saved_state(model);state['backbone.unexpected_weight']=torch.ones(1)
        with self.assertRaises(RuntimeError):load_public(model,state)

    def test_active_shape_mismatch_rejected_before_loading(self):
        model=Toy();state=saved_state(model);state['backbone.weight']=torch.ones(3,3)
        original=model.neck.weight.detach().clone()
        with self.assertRaises(RuntimeError):load_public(model,state)
        torch.testing.assert_close(model.neck.weight,original)

    def test_different_rebuilt_coordinates_rejected(self):
        model=Toy();state=saved_state(model);state['backbone.relative_coords_table1']+=.1
        with self.assertRaises(RuntimeError):load_public(model,state)

    def test_unregistered_cache_like_key_rejected(self):
        model=Toy();state=saved_state(model);state['backbone.relative_pos_index5']=torch.ones(1)
        with self.assertRaises(RuntimeError):load_public(model,state)

    def test_inactive_lse_and_ema_compatibility(self):
        model=Toy();state=saved_state(model);del state['candidate_quality.tb_lse_temperature_raw']
        state['ema_backbone.weight']=torch.ones(2,2)
        _,audit=load_public(model,state)
        self.assertIn('candidate_quality.tb_lse_temperature_raw',audit['allowed_inactive_missing'])
        self.assertIn('ema_backbone.weight',audit['ignored_inactive_keys'])


if __name__=='__main__':unittest.main()
