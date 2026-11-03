"""CPU checks for F2 continuation; source experiment checkpoints are not opened."""
import ast
import copy
import hashlib
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
from test_roi_query import classes
from test_roi_continuation import LRInterface

torch.set_num_threads(2)


def hook_class():
    path=ROOT/'TQ_SymNet/roi_query_continuation.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    for node in tree.body: node.decorator_list=[]
    scope=dict(LrUpdaterHook=LRInterface,math=math,Path=Path,json=json,
               torch=torch,hashlib=hashlib)
    exec(compile(tree,str(path),'exec'),scope)
    return scope['TQF2FinetuneLrUpdaterHook']


def make_runner(folder):
    source=folder/'f2_s0'; source.mkdir()
    checkpoint=source/'best_official_tb_bbox_mAP_epoch_8.pth'
    checkpoint.write_bytes(b'unit-test-source-placeholder-not-a-pytorch-file')
    core=nn.Module()
    core.backbone=nn.Linear(2,2).requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        core.candidate_quality=classes()['ROIContentQualityAttention'](
            hidden=96,heads=6,edges=True,detail=True,texture=True,paired_boundary=True)
    q=core.candidate_quality
    probability=(.5364386439323425-.1)/.8
    q.fusion_gate_logit=nn.Parameter(torch.tensor(math.log(probability/(1-probability))))
    q.tb_lse_temperature_raw=nn.Parameter(torch.tensor(0.),requires_grad=False)
    core.integrated_build_id='TQ-ROI-content-role-paired-querygate-v2'
    core.quality_arch='attention'
    core.quality_detach=core.selector_learned_gate=core.selector_gate_trainable=True
    core.selector_quality_target='coco_all_iou'; core.selector_tb_fusion='max'
    core.selector_rank_weight=.1; core.selector_sigma=.7
    core.selector_fusion_loss_weight=core.selector_edge_weight=0.
    parameters=(q.fusion_gate_logit,q.high_head.weight,q.visual_projection[0].weight,
                q.roi_reader.level_gate[0].weight,q.pair_reader.level_gate[0].weight)
    groups=[dict(params=[p for p in q.parameters() if p.requires_grad],
                 initial_lr=1e-4,lr=3.806e-6)]
    state={p:dict(step=3632,exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
           for p in parameters}
    meta=dict(seed=0,roi_query_contract=dict(model=core.integrated_build_id,
        trainable=2465096,alpha_trainable=True),tq_initialization=dict(
        detector_checkpoint_sha256='4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74'),
        hook_msgs=dict(best_ckpt=str(checkpoint),last_ckpt=str(checkpoint),best_score=.402))
    runner=SimpleNamespace(model=core,optimizer=SimpleNamespace(param_groups=groups,state=state),
        meta=meta,epoch=8,iter=3632,max_epochs=12,work_dir=str(folder/'f2_continue'),
        hooks=[SimpleNamespace(best_ckpt_path=str(checkpoint))],
        logger=logging.getLogger('f2-continuation-test'),data_loader=range(454))
    return runner,checkpoint


class F2ContinuationTests(unittest.TestCase):
    def test_resolved_config_only_budget_schedule_tag_and_resume_change(self):
        a=load_config(ROOT/'configs/roi_role_query/f2_roi_paired_query.py')
        b=load_config(ROOT/'configs/roi_role_query/f2_continue_e8_e12.py')
        for key in ('data','optimizer','optimizer_config','evaluation','custom_hooks'):
            self.assertEqual(a[key],b[key])
        x,y=copy.deepcopy(a['model']),copy.deepcopy(b['model'])
        x.pop('experiment_tag');y.pop('experiment_tag')
        self.assertEqual(x,y)
        self.assertEqual(b['runner']['max_epochs'],12)
        self.assertEqual(b['resume_from'],b['lr_config']['source_checkpoint'])
        self.assertIsNone(b['load_from'])

    def test_schedule_optimizer_alpha_and_external_checkpoint_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,source=make_runner(Path(tmp))
            saved=source.read_bytes()
            alpha=runner.model.candidate_quality.fusion_gate_logit.detach().clone()
            optimizer_state=runner.optimizer.state
            hook=hook_class()(source_checkpoint=str(source))
            hook.before_run(runner)
            self.assertIs(runner.optimizer.state,optimizer_state)
            torch.testing.assert_close(runner.model.candidate_quality.fusion_gate_logit,alpha,rtol=0,atol=0)
            self.assertEqual(runner.meta['hook_msgs']['best_score'],.402)
            self.assertNotIn('best_ckpt',runner.meta['hook_msgs'])
            self.assertNotIn('last_ckpt',runner.meta['hook_msgs'])
            self.assertIsNone(runner.hooks[0].best_ckpt_path)
            rates=[]
            for epoch in range(8,12):
                runner.epoch=epoch;hook.before_train_epoch(runner)
                rates.append(runner.optimizer.param_groups[0]['lr'])
            for actual,expected in zip(rates,[1e-5,8.5355339059e-6,5e-6,1.4644660941e-6]):
                self.assertAlmostEqual(actual,expected,places=12)
            self.assertEqual(source.read_bytes(),saved)
            self.assertEqual(runner.optimizer.param_groups[0]['initial_lr'],1e-4)

    def test_interrupted_resume_no_double_scaling_or_external_path_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,source=make_runner(Path(tmp))
            hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.epoch=10;runner.iter=4540
            inside=str(Path(runner.work_dir)/'best_official_tb_bbox_mAP_epoch_10.pth')
            runner.meta['hook_msgs'].update(best_ckpt=inside,best_score=.404)
            runner.hooks[0].best_ckpt_path=inside
            hook_class()(source_checkpoint=str(source)).before_run(runner)
            self.assertAlmostEqual(runner.optimizer.param_groups[0]['lr'],5e-6)
            self.assertEqual(runner.meta['hook_msgs']['best_ckpt'],inside)
            self.assertEqual(runner.hooks[0].best_ckpt_path,inside)

    def test_reject_missing_gate_state_wrong_model_and_cold_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,source=make_runner(Path(tmp))
            gate=runner.model.candidate_quality.fusion_gate_logit
            state=runner.optimizer.state.pop(gate)
            with self.assertRaisesRegex(RuntimeError,'AdamW state'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.optimizer.state[gate]=state
            runner.model.integrated_build_id='pre-fix-f2'
            with self.assertRaises(RuntimeError): hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.model.integrated_build_id='TQ-ROI-content-role-paired-querygate-v2'
            runner.epoch=runner.iter=0
            with self.assertRaises(RuntimeError): hook_class()(source_checkpoint=str(source)).before_run(runner)

    def test_reject_source_output_reuse_and_schedule_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,source=make_runner(Path(tmp))
            output=runner.work_dir
            runner.work_dir=str(source.parent)
            with self.assertRaisesRegex(RuntimeError,'NEW output'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.work_dir=output
            hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.epoch=9;runner.iter=4086
            with self.assertRaisesRegex(RuntimeError,'schedule/source changed'):
                hook_class()(source_checkpoint=str(source),quality_lr=2e-5).before_run(runner)

    def test_reject_empty_source_and_reinitialized_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,source=make_runner(Path(tmp))
            saved=source.read_bytes();source.write_bytes(b'')
            with self.assertRaisesRegex(RuntimeError,'Missing/empty source'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)
            source.write_bytes(saved)
            with torch.no_grad(): runner.model.candidate_quality.fusion_gate_logit.zero_()
            with self.assertRaisesRegex(RuntimeError,'alpha was not restored'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)


if __name__=='__main__': unittest.main(verbosity=2)
