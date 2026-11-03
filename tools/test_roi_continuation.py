"""CPU/config regression tests; no checkpoint is unpickled or modified."""
import ast
import copy
import json
import logging
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_structure import load_config


class LRInterface:
    """The small MMCV 1.7.1 epoch-LR interface used by this hook."""
    def __init__(self,by_epoch=True,**kwargs): self.by_epoch=by_epoch
    def before_run(self,runner):
        self.base_lr=[g['initial_lr'] for g in runner.optimizer.param_groups]
    def get_regular_lr(self,runner):
        return [self.get_lr(runner,lr) for lr in self.base_lr]
    def _set_lr(self,runner,rates):
        for group,rate in zip(runner.optimizer.param_groups,rates): group['lr']=rate
    def before_train_epoch(self,runner):
        self.regular_lr=self.get_regular_lr(runner); self._set_lr(runner,self.regular_lr)


def hook_class():
    path=ROOT/'TQ_SymNet/roi_continuation.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    for n in tree.body: n.decorator_list=[]
    scope=dict(LrUpdaterHook=LRInterface,math=math,Path=Path,json=json)
    exec(compile(tree,str(path),'exec'),scope)
    return scope['TQROIFinetuneLrUpdaterHook']


def make_runner(folder):
    source=folder/'a13_s0'; source.mkdir(exist_ok=True)
    old=source/'best_official_tb_bbox_mAP_epoch_8.pth'; old.write_bytes(b'preserve-me')
    core=nn.Module(); core.backbone=nn.Linear(2,2).requires_grad_(False)
    core.candidate_quality=nn.Linear(2,1)
    q=core.candidate_quality; q.hidden=96; q.mlp_hidden=880
    q.fusion_gate_logit=nn.Parameter(torch.zeros(()),requires_grad=False)
    core.quality_arch='pooled'; core.selector_quality_target='direct_iou'
    core.quality_rank_weight=core.selector_rank_weight=0.
    group=dict(params=[p for p in q.parameters() if p.requires_grad],
               initial_lr=1e-4,lr=3.806e-6)
    optimizer=SimpleNamespace(param_groups=[group],state={next(q.parameters()):dict(step=3632)})
    meta=dict(tq_supplement=dict(parameters=dict(trainable=2341713)),
        tq_initialization=dict(detector_checkpoint_sha256=
            '4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74'),
        hook_msgs=dict(best_ckpt=str(old),best_score=.396,last_ckpt=str(old)))
    runner=SimpleNamespace(model=core,optimizer=optimizer,meta=meta,epoch=8,iter=3632,
        max_epochs=12,work_dir=str(folder/'a13_continue_e8_e12_s0'),
        logger=logging.getLogger('roi-continuation-test'),
        hooks=[SimpleNamespace(best_ckpt_path=str(old))],data_loader=range(454))
    return runner,source,old


class ContinuationTests(unittest.TestCase):
    def test_config_changes_only_continuation_controls(self):
        base=load_config(ROOT/'configs/publication_supplement/a13_roi_capacity.py')
        new=load_config(ROOT/'configs/publication_supplement/a13_roi_continue_e8_e12.py')
        for key in ('data','optimizer','optimizer_config','evaluation','custom_hooks'):
            self.assertEqual(base[key],new[key],key)
        a,b=copy.deepcopy(base['model']),copy.deepcopy(new['model'])
        a.pop('experiment_tag'); b.pop('experiment_tag'); self.assertEqual(a,b)
        self.assertEqual(new['runner']['max_epochs'],12)
        self.assertIn('best_official_tb_bbox_mAP_epoch_8.pth',new['resume_from'])
        self.assertIsNone(new['load_from'])

    def test_small_lr_schedule_and_old_best_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,source,old=make_runner(Path(tmp))
            hook=hook_class()(source_dir=str(source)); hook.before_run(runner)
            self.assertAlmostEqual(runner.optimizer.param_groups[0]['lr'],1e-5)
            self.assertEqual(runner.optimizer.param_groups[0]['initial_lr'],1e-4)
            self.assertEqual(runner.meta['hook_msgs']['best_score'],.396)
            self.assertNotIn('best_ckpt',runner.meta['hook_msgs'])
            self.assertIsNone(runner.hooks[0].best_ckpt_path)
            self.assertEqual(old.read_bytes(),b'preserve-me')
            rates=[]
            for e in range(8,12):
                runner.epoch=e; hook.before_train_epoch(runner)
                rates.append(runner.optimizer.param_groups[0]['lr'])
            for actual,expected in zip(rates,[1e-5,8.5355339059e-6,5e-6,1.4644660941e-6]):
                self.assertAlmostEqual(actual,expected,places=12)

    def test_resume_phase_does_not_reduce_lr_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,source,old=make_runner(Path(tmp))
            hook_class()(source_dir=str(source)).before_run(runner)
            runner.epoch=10; runner.iter=4540
            hook_class()(source_dir=str(source)).before_run(runner)
            self.assertAlmostEqual(runner.optimizer.param_groups[0]['lr'],5e-6)
            self.assertEqual(old.read_bytes(),b'preserve-me')

    def test_reject_cold_start_or_original_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,source,_=make_runner(Path(tmp))
            runner.epoch=0; runner.iter=0
            with self.assertRaises(RuntimeError): hook_class()(source_dir=str(source)).before_run(runner)
            runner.epoch=8; runner.iter=3632; runner.work_dir=str(source)
            with self.assertRaises(RuntimeError): hook_class()(source_dir=str(source)).before_run(runner)
            runner.work_dir=str(Path(tmp)/'new'); runner.optimizer.state={}
            with self.assertRaises(RuntimeError): hook_class()(source_dir=str(source)).before_run(runner)


if __name__=='__main__': unittest.main(verbosity=2)
