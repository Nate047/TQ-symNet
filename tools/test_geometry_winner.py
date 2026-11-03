"""CPU tests for actual geometry/winner math, reader and inherited train loop."""
import copy
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from pathlib import Path

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from check_structure import load_config
from test_roi_query import classes,load_definitions
from test_threshold_distribution import implementation_scope,harness_class,overlaps
from TQ_SymNet.geometry_winner_math import (winner_sets,winner_set_loss,
    signed_geometry_targets,signed_geometry_loss)

torch.set_num_threads(2)


def scope_with_models():
    scope=implementation_scope()
    scope.update(winner_sets=winner_sets,winner_set_loss=winner_set_loss,
        signed_geometry_targets=signed_geometry_targets,signed_geometry_loss=signed_geometry_loss,
        bbox_overlaps=overlaps,json=json,Path=Path)
    path=ROOT/'TQ_SymNet/models/geometry_winner.py'
    load_definitions(path,{'SignedGeometryQualityAttention','geometry_summary',
                          'geometry_record','winner_record'},scope)
    base=harness_class(scope)
    class F2Harness(base):
        def __init__(self,**kwargs):
            super().__init__()
            previous=self.candidate_quality
            q=scope['ROIContentQualityAttention'](hidden=24,heads=4,edges=True,
                detail=True,texture=False,paired_boundary=True,chunk=2)
            q.fusion_gate_logit=nn.Parameter(previous.fusion_gate_logit.detach().clone())
            q.tb_lse_temperature_raw=nn.Parameter(previous.tb_lse_temperature_raw.detach().clone(),requires_grad=False)
            q.load_state_dict(previous.state_dict(),strict=False)
            self.candidate_quality=q
        def _score_candidates(self,boxes,memory,sym,training=False):
            return self.candidate_quality(boxes,memory,sym,training_checkpoint=training)
    scope['TQSymNetROIQuery']=F2Harness
    load_definitions(path,{'TQSymNetGeometryWinner'},scope)
    return scope


class GeometryWinnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.scope=scope_with_models()

    def test_winner_selection_with_legacy_argsort_api(self):
        original_argsort = torch.Tensor.argsort

        def legacy_argsort(tensor, *args, **kwargs):
            if 'stable' in kwargs:
                raise TypeError('argsort() received unsupported keyword: stable')
            return original_argsort(tensor, *args, **kwargs)

        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                # Equal IoUs must keep candidate order; close but unequal IoUs
                # must not be reversed by an epsilon-based tie-breaking hack.
                actual = torch.tensor([.86, .865, .86, .86, .865, .86, .6],
                                      device=device)
                matched = torch.zeros(len(actual), dtype=torch.long, device=device)
                boxes = torch.tensor([[0., 0., 10., 10.]] * len(actual), device=device)
                before = actual.clone()
                with patch.object(torch.Tensor, 'argsort', legacy_argsort):
                    group = winner_sets(actual, matched, boxes)[0]
                self.assertEqual(group['high'].tolist(), [1, 4, 0, 2])
                self.assertEqual(group['rivals'].tolist(), [6])
                self.assertEqual(group['neutral_equivalent'], 2)
                self.assertEqual(group['high'].device, actual.device)
                torch.testing.assert_close(actual, before, rtol=0, atol=0)

    def test_plateau_neutral_and_other_gt_protection(self):
        u=torch.tensor([.86,.855,.852,.851,.8505,.84,.74,.2,.92])
        g=torch.tensor([0,0,0,0,0,0,0,-1,1])
        boxes=torch.tensor([[0.,0.,10.,10.]]*len(u))
        groups=winner_sets(u,g,boxes)
        zero=next(x for x in groups if int(x['gt'])==0)
        self.assertEqual(len(zero['high']),4)
        self.assertTrue(bool((u[zero['high']]>=.85).all()))
        self.assertEqual(zero['neutral_equivalent'],1)
        self.assertNotIn(4,zero['rivals'].tolist())
        self.assertNotIn(8,zero['rivals'].tolist())
        self.assertIn(5,zero['rivals'].tolist())
        self.assertIn(7,zero['rivals'].tolist())

    def test_rivals_not_capped_and_loss_focuses_on_good_set(self):
        u=torch.cat((torch.tensor([.9]),torch.full((120,),.6)))
        g=torch.zeros(len(u),dtype=torch.long)
        boxes=torch.tensor([[0.,0.,10.,10.]]*len(u))
        groups=winner_sets(u,g,boxes)
        self.assertEqual(len(groups[0]['rivals']),120)
        z=torch.zeros(len(u),requires_grad=True)
        loss,count,_=winner_set_loss(z,groups)
        self.assertAlmostEqual(float(loss),1.,places=6)
        self.assertEqual(count,120)
        loss.backward()
        self.assertLess(float(z.grad[0]),0.)
        self.assertTrue(bool((z.grad[1:]>0).all()))
        better=z.detach().clone();better[0]=2.
        self.assertLess(float(winner_set_loss(better,groups)[0]),float(loss))
        extreme=torch.linspace(-1000,1000,len(u),requires_grad=True)
        a=winner_set_loss(extreme,groups)[0];a.backward()
        self.assertTrue(bool(torch.isfinite(extreme.grad).all()))

    def test_geometry_xyxy_scale_and_ambiguous_gt(self):
        boxes=torch.tensor([[10.,20.,30.,60.],[0.,0.,20.,20.]])
        gt=torch.tensor([[8.,24.,34.,56.]])
        target,valid,ambiguous=signed_geometry_targets(boxes,gt,torch.tensor([0,0]),torch.tensor([.7,.1]))
        torch.testing.assert_close(target[0],torch.tensor([-.1,.1,.2,-.1]))
        self.assertEqual(valid.tolist(),[True,False])
        scaled,valid2,_=signed_geometry_targets(boxes*torch.tensor([2.,3.,2.,3.]),
            gt*torch.tensor([2.,3.,2.,3.]),torch.tensor([0,0]),torch.tensor([.7,.1]))
        torch.testing.assert_close(scaled,target)
        gt2=torch.tensor([[0.,0.,10.,10.],[.1,0.,10.1,10.]])
        _,v,a=signed_geometry_targets(torch.tensor([[0.,0.,10.,10.]]),gt2,torch.tensor([0]),torch.tensor([1.]))
        self.assertFalse(bool(v[0]));self.assertTrue(bool(a[0]))
        pred=torch.zeros_like(target,requires_grad=True)
        loss=signed_geometry_loss(pred,target,valid,torch.tensor([0,0]));loss.backward()
        self.assertGreater(float(pred.grad[0].abs().sum()),0.)
        self.assertEqual(float(pred.grad[1].abs().sum()),0.)

    def test_reader_matches_f2_score_before_training_and_keeps_boxes(self):
        scope=self.scope
        torch.manual_seed(4)
        old=scope['ROIContentQualityAttention'](hidden=24,heads=4,edges=True,detail=True,
            texture=False,paired_boundary=True,chunk=2)
        torch.manual_seed(4)
        new=scope['SignedGeometryQualityAttention'](hidden=24,heads=4,edges=True,detail=True,
            texture=False,paired_boundary=True,chunk=2)
        for key,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        with torch.no_grad():
            old.high_head.weight.normal_(std=.05)
            new.high_head.load_state_dict(old.high_head.state_dict())
        maps=[torch.randn(1,256,s,s) for s in (8,4,2,1)];gray=torch.rand(1,1,32,32)
        boxes=torch.tensor([[.1,.2,.7,.8],[.2,.3,.8,.9]])
        original=boxes.clone();sym=torch.rand(2,3)
        old.eval();new.eval()
        with torch.no_grad():
            a=old(boxes,old.prepare(maps,gray,(32,32),(32,32)),sym)
            memory=new.prepare(maps,gray,(32,32),(32,32))
            b=new(boxes,memory,sym,return_geometry=True)
            torch.testing.assert_close(a[0][:,0],b[0][:,0],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(a[1],b[1],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(boxes,original,rtol=0,atol=0)
            new.chunk=16
            c=new(boxes,memory,sym,return_geometry=True)
            for x,y in zip(b,c):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
            self.assertEqual(new(boxes[:0],memory,sym[:0],return_geometry=True)[3].shape,(0,4))

    def test_production_train_loop_new_losses_alpha_and_frozen_main(self):
        model=self.scope['TQSymNetGeometryWinner']()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        image=torch.rand(1,3,32,32);gray=torch.rand(1,1,32,32)
        gt=[torch.tensor([[4.,4.,18.,18.]])]
        before={k:v.detach().clone() for k,v in model.backbone.state_dict().items()}
        for _ in range(2):
            optimizer.zero_grad()
            losses=model.forward_train(image,[dict(img_shape=(32,32,3))],gt,
                [torch.zeros(1,dtype=torch.long)],original_img=gray)
            self.assertIn('loss_signed_geometry',losses);self.assertIn('loss_winner_set',losses)
            self.assertNotIn('loss_selector_competition',losses)
            self.assertNotIn('loss_selector_edge_pair',losses)
            sum(losses.values()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertGreater(float(model.candidate_quality.signed_aux[-1].weight.grad.abs().sum()),0.)
            self.assertGreater(float(model.candidate_quality.fusion_gate_logit.grad.abs()),0.)
            optimizer.step()
        for k,v in model.backbone.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(float(model.candidate_quality.roi_reader.v.weight.grad.abs().sum()),0.)
        model._main=lambda img,metas:([],[(img.new_empty((0,4)),img.new_empty((0,3)))],{})
        optimizer.zero_grad()
        sum(model.forward_train(image,[{}],gt,[torch.zeros(1)],original_img=gray).values()).backward()

    def test_validation_geometry_is_posthoc_and_caches_clear(self):
        model=self.scope['TQSymNetGeometryWinner']().eval()
        model.v50_begin_oracle(4)
        img,gray=torch.rand(1,3,32,32),torch.rand(1,1,32,32)
        feats,raw,_=model._main(img,[{}]);boxes,sym=raw[0]
        memory=model.candidate_quality.prepare(feats,gray,(32,32),(32,32))
        with torch.no_grad():out,_,_=model._score_candidates(boxes/32.,memory,sym)
        model.quality_cache.append(dict(boxes=boxes,logit=out[:,0],corrected=out[:,0].sigmoid()))
        dataset=SimpleNamespace(get_ann_info=Mock(return_value=dict(bboxes=[[4.,4.,18.,18.]])))
        runner=SimpleNamespace(work_dir=str(ROOT),logger=Mock())
        with patch.object(Path,'write_text') as write:
            model.v50_finish_oracle(dataset,runner)
            record=json.loads(write.call_args.args[0])
        self.assertEqual(record['geometry']['observations'],5)
        self.assertEqual(record['high85_positive_pool_top1']['groups'],1)
        self.assertEqual(model.geometry_val,[])
        self.assertEqual(model.quality_cache,[])
        self.assertTrue(model.finished)

    def test_config_controls_no_calibration_or_quality_warmstart(self):
        old=load_config(ROOT/'configs/roi_role_query/f2_roi_paired_query.py')
        main=load_config(ROOT/'configs/quality_choice/g1_f2_geometry_winner.py')
        roi=load_config(ROOT/'configs/quality_choice/r0_roi_complete_recipe.py')
        for new in (main,roi):
            for key in ('data','optimizer','optimizer_config','lr_config','runner','evaluation'):
                self.assertEqual(old[key],new[key])
            self.assertIsNone(new['load_from']);self.assertIsNone(new['resume_from'])
            self.assertTrue(new['model']['selector_gate_trainable'])
            self.assertEqual(new['model']['selector_quality_target'],'coco_all_iou')
            self.assertEqual(new['model']['selector_rank_weight'],.1)
        self.assertEqual(roi['model']['quality_pooled_mlp_hidden'],880)
        self.assertEqual(main['model']['geometry_weight'],.05)

    def test_roi_complete_recipe_real_pair_rank_and_alpha(self):
        scope=self.scope
        load_definitions(ROOT/'TQ_SymNet/models/light_quality.py',{'PooledQualityScorer'},scope)
        base=harness_class(scope)
        class ROIModel(base):
            def __init__(self):
                super().__init__()
                self.candidate_quality=scope['PooledQualityScorer'](mlp_hidden=880)
                self.candidate_quality.fusion_gate_logit=nn.Parameter(torch.zeros(()))
                self.candidate_quality.tb_lse_temperature_raw=nn.Parameter(torch.tensor(0.),requires_grad=False)
            def _score_candidates(self,boxes,memory,sym,training=False):
                return self.candidate_quality(boxes,memory,sym,training_checkpoint=training)
        model=ROIModel()
        self.assertEqual(sum(p.numel() for p in model.candidate_quality.parameters() if p.requires_grad),2341714)
        image,gray=torch.rand(1,3,32,32),torch.rand(1,1,32,32)
        losses=model.forward_train(image,[dict(img_shape=(32,32,3))],
            [torch.tensor([[4.,4.,18.,18.]])],[torch.zeros(1,dtype=torch.long)],original_img=gray)
        self.assertGreater(float(losses['loss_selector_competition']),0.)
        sum(losses.values()).backward()
        self.assertGreater(float(model.candidate_quality.fusion_gate_logit.grad.abs()),0.)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_auxiliary_alone_reaches_visual_roi_and_native_features(self):
        torch.manual_seed(8)
        q=self.scope['SignedGeometryQualityAttention'](hidden=24,heads=4,edges=True,
            detail=True,texture=False,paired_boundary=True,chunk=1)
        # After the first head update, its nonzero weights must deliver spatial
        # supervision into shared reading, not just train an isolated MLP.
        with torch.no_grad():q.signed_aux[-1].weight.normal_(std=.03)
        maps=[torch.randn(1,256,s,s) for s in (8,4,2,1)]
        gray=torch.rand(1,1,32,32)
        boxes=torch.tensor([[.1,.2,.7,.8],[.2,.3,.8,.9]])
        pred=q(boxes,q.prepare(maps,gray,(32,32),(32,32)),torch.rand(2,3),
               training_checkpoint=True,return_geometry=True)[3]
        target=torch.tensor([[-.1,.1,.2,-.1],[.1,-.2,.1,.2]])
        signed_geometry_loss(pred,target,torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.long)).backward()
        for p in (q.visual_projection[0].weight,q.roi_reader.v.weight,q.detail_encoder.stem[0].weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.abs().sum()),0.)
        self.assertIsNone(q.high_head.weight.grad)

    def test_model_sampler_includes_all_competitors_not_just_math_helper(self):
        model=self.scope['TQSymNetGeometryWinner']()
        actual=torch.cat((torch.tensor([.9]),torch.full((120,),.6)))
        matched=torch.zeros(121,dtype=torch.long)
        boxes=torch.tensor([[0.,0.,6.,10.]]*121);boxes[0,2]=9.
        target=model._quality_target(actual)
        selected=model._selector_sample(actual,matched,target,torch.full_like(actual,.5),
                                        torch.full_like(actual,.5),boxes)
        self.assertEqual(len(selected),121)
        self.assertLess(model.winner_sample[-1][1],121)
        self.assertGreater(model.winner_sample[-1][3],0)


if __name__=='__main__':unittest.main(verbosity=2)
