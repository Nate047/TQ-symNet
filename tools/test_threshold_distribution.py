"""CPU regressions: real distribution head, reader, and inherited train loop."""
import ast
import copy
import json
import math
import sys
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from check_structure import load_config
from test_roi_query import classes, load_definitions
from TQ_SymNet.threshold_distribution_math import (distribution_predictions,
    distribution_element_loss, soft_threshold_targets, ThresholdDistributionHead,
    threshold_statistics, format_threshold_statistics)

torch.set_num_threads(2)


def overlaps(boxes, gt):
    lt = torch.maximum(boxes[:, None, :2], gt[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    a = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    b = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1)
    return inter / (a[:, None] + b[None] - inter).clamp_min(1e-8)


def implementation_scope():
    scope = classes()
    scope.update(ThresholdDistributionHead=ThresholdDistributionHead,
        distribution_element_loss=distribution_element_loss,
        threshold_statistics=threshold_statistics,
        format_threshold_statistics=format_threshold_statistics,
        bbox_overlaps=overlaps, Path=Path, json=json)
    load_definitions(ROOT / 'TQ_SymNet/models/threshold_distribution.py',
                     {'DistributionQualityAttention'}, scope)
    load_definitions(ROOT / 'TQ_SymNet/models/selection.py',
                     {'all_iou_target', 'balanced_mean'}, scope)
    scope['pairwise_box_iou'] = lambda boxes: overlaps(boxes, boxes)
    # Execute production methods without the unavailable MMDet registration.
    for filename, class_name, names in (
        ('selection.py', 'RetinaNAMHighQualitySelector',
         {'forward_train', '_training_quality_predictions', '_quality_element_loss',
          '_additional_supervised_loss', '_replacement_rank_loss',
          '_selector_sample', '_full_candidate_training_loss', '_extra_training_loss',
          '_trainable_zero', '_quality_target'}),
        ('detail.py', 'RetinaNAMDetailSelector',
         {'_residual_loss', '_final_score', '_final_logit'})):
        path = ROOT / 'TQ_SymNet/models' / filename
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def harness_class(scope):
    class OriginalHarness(nn.Module):
        """Tiny frozen feature source; production quality/training methods."""
        def __init__(self, **kwargs):
            super().__init__()
            self.backbone = nn.Conv2d(3,256,1).requires_grad_(False)
            self.candidate_quality = scope['DetailQualityAttention'](
                hidden=24, heads=4, texture=False, edges=True, detail=True, chunk=2)
            self.candidate_quality.fusion_gate_logit = nn.Parameter(torch.zeros(()))
            self.candidate_quality.tb_lse_temperature_raw = nn.Parameter(torch.tensor(.5), requires_grad=False)
            self.quality_arch = 'attention'
            self.quality_detach = self.selector_learned_gate = self.selector_gate_trainable = True
            self.selector_quality_trainable = self.selector_quality_rank_only = self.selector_list_rank = True
            self.selector_quality_target, self.selector_quality_target_temperature = 'coco_all_iou', .03
            self.selector_tb_fusion, self.selector_adaptive_subtype_gate = 'max', False
            self.selector_beta, self.selector_rank_weight = .5, .1
            self.selector_fusion_loss_weight, self.selector_edge_weight = 0., 0.
            self.selector_temperature, self.quality_samples_per_bin = .03, 3
            self.selector_train_stats, self.quality_cache = [], []
            self.quality_collect = False
            self.bbox_head = SimpleNamespace(test_cfg=SimpleNamespace(score_thr=.05))

        def _main(self, img, metas):
            with torch.no_grad():
                feature = self.backbone(img)
                features = [F.avg_pool2d(feature, stride) for stride in (4,8,16,32)]
            boxes = img.new_tensor([[4,4,18,18], [4,4,17,18], [6,6,18,18],
                                   [24,24,31,31], [4,4,10,10]])
            sym = img.new_tensor([[.9,.1,0.], [.8,.2,0.], [.7,.1,0.],
                                 [.6,.1,0.], [.5,.1,0.]])
            return features, [(boxes,sym)], {}

        def _v330_unwrap_original_images(self, original, count):
            return [original]

        def _image_memory(self, features, img, meta, original, index):
            return self.candidate_quality.prepare(features, original, (32,32), (32,32))

        def _normalized(self, boxes, meta):
            return boxes / 32.

        def v50_begin_oracle(self, epoch):
            self.quality_epoch = epoch
            self.quality_cache = []
            self.quality_collect = epoch % 4 == 0

        def v50_finish_oracle(self, dataset, runner):
            self.finished = True
            self.quality_cache = []
            self.quality_collect = False

    for name in ('forward_train', '_training_quality_predictions', '_quality_element_loss',
                 '_additional_supervised_loss', '_replacement_rank_loss',
                 '_selector_sample', '_full_candidate_training_loss', '_extra_training_loss',
                 '_trainable_zero', '_quality_target', '_residual_loss', '_final_score', '_final_logit'):
        setattr(OriginalHarness, name, scope[name])
    return OriginalHarness


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = implementation_scope()
        cls.scope['TQSymNet'] = harness_class(cls.scope)
        load_definitions(ROOT / 'TQ_SymNet/models/threshold_distribution.py',
                         {'TQSymNetThresholdDistribution'}, cls.scope)

    def test_monotonicity_scalar_equivalence_and_extremes(self):
        for raw in (torch.randn(9,11), torch.randn(9,11) * 1000, torch.zeros(0,11)):
            raw.requires_grad_()
            quality, logits = distribution_predictions(raw)
            p = logits.sigmoid()
            self.assertTrue(bool((p[:,1:] <= p[:,:-1] + 1e-6).all()))
            torch.testing.assert_close(quality.sigmoid(), p.mean(1), rtol=1e-5, atol=1e-6)
            loss = distribution_element_loss(logits, torch.linspace(0,1,len(raw)))
            self.assertTrue(bool(torch.isfinite(loss).all()))
            loss.sum().backward()
            self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        head = ThresholdDistributionHead(24)
        q = head(torch.randn(4,24)).squeeze(1)
        torch.testing.assert_close(q, torch.zeros_like(q), rtol=0, atol=1e-6)

    def test_targets_match_existing_mean_and_loss_not_mean_collapse(self):
        actual = torch.linspace(0,1,101)
        y = soft_threshold_targets(actual)
        torch.testing.assert_close(y.mean(1), self.scope['all_iou_target'](actual))
        raw = torch.full((2,11), -6.)
        raw[0,5] = 6.
        raw[1,0] = raw[1,10] = 6.
        q, logits = distribution_predictions(raw)
        torch.testing.assert_close(q[0], q[1], atol=2e-5, rtol=0)
        loss = distribution_element_loss(logits, torch.tensor([.725,.725]))
        self.assertGreater(float((loss[0] - loss[1]).abs()), .01)

    def test_production_train_loop_supervision_and_frozen_main(self):
        torch.manual_seed(0)
        model = self.scope['TQSymNetThresholdDistribution']()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        img, gray = torch.rand(1,3,32,32), torch.rand(1,1,32,32)
        gt = [torch.tensor([[4.,4.,18.,18.]])]
        before = {n:p.detach().clone() for n,p in model.backbone.named_parameters()}
        quality_before = model.candidate_quality.visual_projection[0].weight.detach().clone()
        for step in range(2):
            optimizer.zero_grad()
            losses = model.forward_train(img, [dict(img_shape=(32,32,3))], gt,
                                         [torch.zeros(1,dtype=torch.long)], original_img=gray)
            self.assertEqual(set(losses), {'loss_selector_high_quality', 'loss_selector_residual',
                'loss_selector_fusion', 'loss_selector_competition', 'loss_selector_edge_pair',
                'loss_selector_head_competition'})
            self.assertEqual(float(losses['loss_selector_fusion']), 0.)
            sum(losses.values()).backward()
            for n,p in model.named_parameters():
                if p.grad is not None:
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), n)
            self.assertGreater(float(model.candidate_quality.high_head.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(model.candidate_quality.fusion_gate_logit.grad.abs()), 0.)
            optimizer.step()
        self.assertGreater(float((quality_before - model.candidate_quality.visual_projection[0].weight).abs().sum()), 0.)
        for n,p in model.backbone.named_parameters():
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        self.assertEqual(float(model.distribution_train_stats[0,0]), 10.)
        # No candidates must remain a legal, differentiable training step.
        model._main = lambda img, metas: ([], [(img.new_empty((0,4)),img.new_empty((0,3)))], {})
        optimizer.zero_grad()
        sum(model.forward_train(img,[{}],gt,[torch.zeros(1)],original_img=gray).values()).backward()

    def test_reader_state_chunk_empty_and_no_roi(self):
        torch.manual_seed(5); old = self.scope['TQSymNet']()
        torch.manual_seed(5); model = self.scope['TQSymNetThresholdDistribution']()
        for key, value in old.state_dict().items():
            if not key.startswith('candidate_quality.high_head.'):
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        q = model.candidate_quality
        self.assertFalse(hasattr(q, 'roi_reader'))
        self.assertFalse(hasattr(q, 'pair_reader'))
        model.eval()
        img, gray = torch.rand(1,3,32,32), torch.rand(1,1,32,32)
        features, raw, _ = model._main(img, [{}])
        memory = q.prepare(features, gray, (32,32), (32,32))
        boxes, sym = raw[0]
        with torch.no_grad():
            a = q(boxes/32.,memory,sym,return_thresholds=True)
            q.chunk = 16
            b = q(boxes/32.,memory,sym,return_thresholds=True)
            for x,y in zip(a,b):
                torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
            self.assertEqual(q(boxes[:0],memory,sym[:0],return_thresholds=True)[3].shape,(0,10))

    def test_control_loss_hook_is_unchanged(self):
        z = torch.randn(6,requires_grad=True)
        t = torch.rand(6)
        actual = torch.rand(6)
        original = F.binary_cross_entropy_with_logits(z,t,reduction='none')
        hooked = self.scope['_quality_element_loss'](None,z,t,actual,None)
        torch.testing.assert_close(original,hooked,rtol=0,atol=0)

    def test_config_no_mixed_structural_ablation(self):
        config = load_config(ROOT/'configs/roi_role_query/a0_distribution_threshold.py')
        control = load_config(ROOT/'configs/roi_role_query/a0_original_learned_alpha.py')
        a, b = dict(config['model']), dict(control['model'])
        a.pop('type'); b.pop('type'); a.pop('experiment_tag'); b.pop('experiment_tag')
        self.assertEqual(a,b)
        for key in ('data','optimizer','lr_config','runner','evaluation','custom_hooks'):
            self.assertEqual(config[key],control[key])
        self.assertEqual(config['data']['samples_per_gpu'],2)
        self.assertEqual(config['runner']['max_epochs'],8)
        self.assertTrue(config['model']['selector_gate_trainable'])

    def test_diagnostic_counts_and_strata(self):
        actual = torch.tensor([0.,.3,.5,.8,.9])
        _, logits = distribution_predictions(torch.zeros(5,11))
        stats = threshold_statistics(logits,actual)
        self.assertEqual(stats.shape,(6,51))
        rows = format_threshold_statistics(stats)
        self.assertEqual(rows['all']['count'],5)
        self.assertEqual(sum(v['count'] for k,v in rows.items() if k!='all'),5)
        self.assertEqual(rows['iou_ge_85']['actual_pass_rate'][7],1.)
        self.assertEqual(rows['iou_ge_85']['actual_pass_rate'][9],0.)

    def test_val_cache_and_gt_only_posthoc_diagnostics(self):
        # Exercise production validation summary with both nonempty/empty images.
        # Inference below receives no GT; GT is queried ONLY when finishing logs.
        model = self.scope['TQSymNetThresholdDistribution']().eval()
        model.v50_begin_oracle(4)
        img, gray = torch.rand(1,3,32,32), torch.rand(1,1,32,32)
        features, raw, _ = model._main(img,[{}])
        memory = model.candidate_quality.prepare(features,gray,(32,32),(32,32))
        boxes, sym = raw[0]
        with torch.no_grad():
            model._score_candidates(boxes/32.,memory,sym)
            model.quality_cache.append(dict(boxes=boxes))
            model._score_candidates(boxes[:0],memory,sym[:0])
            model.quality_cache.append(dict(boxes=boxes[:0]))
        dataset = SimpleNamespace(get_ann_info=Mock(return_value=dict(
            bboxes=torch.tensor([[4.,4.,18.,18.]]))))
        runner = SimpleNamespace(logger=Mock(),work_dir=str(ROOT))
        with patch.object(Path,'write_text') as write:
            model.v50_finish_oracle(dataset,runner)
            import json
            record = json.loads(write.call_args.args[0])
        self.assertEqual(dataset.get_ann_info.call_count,2)
        self.assertEqual(record['stats']['all']['count'],5)
        self.assertTrue(model.finished)
        self.assertEqual(model.distribution_val_cache,[])
        self.assertEqual(model.quality_cache,[])


if __name__ == '__main__':
    unittest.main(verbosity=2)
