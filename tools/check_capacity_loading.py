"""CPU-only regression checks for strict official-weight loading; no MMCV."""
import ast
import inspect
import logging
import math
from pathlib import Path
import sys
import tempfile
import unittest
from argparse import Namespace
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from TQ_SymNet.capacity_pretrained import load_official_checkpoint, strict_load_backbone


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.net=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4))
        self.state={k:v.clone() for k,v in self.net.state_dict().items()}

    def test_complete(self):
        audit=strict_load_backbone(self.net,self.state)
        self.assertEqual(audit['loaded_parameters'],audit['total_parameters'])

    def test_missing_mean_and_variance_rejected(self):
        for suffix in ('running_mean','running_var'):
            state=dict(self.state); state.pop('1.'+suffix)
            with self.assertRaisesRegex(RuntimeError,'missing='):
                strict_load_backbone(self.net,state)

    def test_old_batch_counter_reset(self):
        self.net[1].num_batches_tracked.fill_(17)
        self.state.pop('1.num_batches_tracked')
        audit=strict_load_backbone(self.net,self.state)
        self.assertEqual(self.net[1].num_batches_tracked.item(),0)
        self.assertEqual(audit['reset_batch_counters'],['1.num_batches_tracked'])

    def test_bad_bn_rejected(self):
        for value in (-1.,float('nan'),float('inf')):
            state=dict(self.state); state['1.running_var']=torch.full((4,),value)
            with self.assertRaisesRegex(RuntimeError,'invalid_bn='):
                strict_load_backbone(self.net,state)

    def test_bad_shape_rejected(self):
        self.state['0.weight']=torch.zeros(5,3,1,1)
        with self.assertRaisesRegex(RuntimeError,'mismatched='):
            strict_load_backbone(self.net,self.state)

    def test_missing_parameter_rejected(self):
        self.state.pop('0.weight')
        with self.assertRaises(RuntimeError): strict_load_backbone(self.net,self.state)

    def test_allowed_new_output_module(self):
        net=nn.Module(); net.extra_norm=nn.LayerNorm(4)
        strict_load_backbone(net,{},allowed_new=('extra_norm.',))
        with self.assertRaises(RuntimeError): strict_load_backbone(net,{})

    def test_legacy_loader_signature(self):
        def old_load(path,map_location=None):
            self.assertEqual(map_location,'cpu'); return 42
        self.assertEqual(load_official_checkpoint('unused',loader=old_load),42)

    def test_modern_loader_signature(self):
        def new_load(path,map_location=None,weights_only=True):
            self.assertFalse(weights_only); return 42
        self.assertEqual(load_official_checkpoint('unused',loader=new_load),42)

    def test_trusted_metadata_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'official.pth'
            torch.save(dict(model=self.state,args=Namespace(seed=0)),str(path))
            obj=load_official_checkpoint(path)
            self.assertEqual(obj['args'].seed,0)

    def test_lsnet_official_bias_resize_and_indices(self):
        tree=ast.parse((ROOT/'TQ_SymNet/models/capacity_backbones.py').read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_load_pretrained')
        env=dict(torch=torch,Path=Path,F=F,math=math,logging=logging,
                 load_official_checkpoint=load_official_checkpoint,
                 strict_load_backbone=strict_load_backbone)
        exec(compile(ast.Module(body=[func],type_ignores=[]),'adapter_loader','exec'),env)
        net=nn.Module();net.attention_biases=nn.Parameter(torch.zeros(2,196))
        net.register_buffer('attention_bias_idxs',torch.zeros(196,196,dtype=torch.long))
        old=torch.arange(32.).reshape(2,16)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'lsnet.pth'
            torch.save(dict(model=dict(attention_biases=old,
                attention_bias_idxs=torch.ones(16,16,dtype=torch.long))),str(path))
            audit=env['_load_pretrained'](net,path,'lsnet_s')
        expected=F.interpolate(old.reshape(1,2,4,4),size=(14,14),
                               mode='bicubic',align_corners=False).reshape(2,196)
        self.assertTrue(torch.equal(net.attention_biases,expected))
        self.assertEqual(net.attention_bias_idxs.sum().item(),0)
        self.assertEqual(audit['resized_attention_biases'],['attention_biases'])


if __name__=='__main__': unittest.main()
