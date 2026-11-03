"""Actual reader CPU/autograd + inherited config checks; no MMCV required.

AST loading only removes detector registrations/imports. Neural class/method
bodies executed below are the production implementations, not test replicas.
"""
import ast
import copy
import json
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from check_structure import load_config
from TQ_SymNet.roi_query_math import LocalRoleReader, PairedBoundaryReader, sample_points

torch.set_num_threads(2)


def load_definitions(path, names, scope):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in names]
    for node in tree.body:
        node.decorator_list = []
    exec(compile(tree, str(path), 'exec'), scope)


def classes():
    scope = dict(torch=torch, nn=nn, F=F, math=math, checkpoint=checkpoint,
                 LocalRoleReader=LocalRoleReader, PairedBoundaryReader=PairedBoundaryReader)
    folder = ROOT / 'TQ_SymNet/models'
    load_definitions(folder / 'attention.py',
        {'grid_positions', 'gaussian_bias', 'texture_encoder', 'LearnableLBP',
         'LearnableGabor', 'CandidateQualityAttention'}, scope)
    load_definitions(folder / 'edges.py', {'edge_bias', 'EdgeQualityAttention'}, scope)
    load_definitions(folder / 'detail.py',
        {'DetailResidualBlock', 'NativeDetailEncoder', 'edge_anchor_points',
         'DetailQualityAttention'}, scope)
    load_definitions(folder / 'roi_query.py', {'ROIContentQualityAttention'}, scope)
    return scope


class ROIQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = classes()

    def build(self, paired=False, detail=True, texture=False, chunk=2):
        return self.scope['ROIContentQualityAttention'](
            channels=256, hidden=24, heads=4, chunk=chunk, edges=True,
            detail=detail, texture=texture, paired_boundary=paired)

    def inputs(self):
        features = [torch.randn(1, 256, h, w) for h, w in ((8, 8), (4, 4), (2, 2), (1, 1))]
        gray = torch.rand(1, 1, 28, 24)
        boxes = torch.tensor([[.05,.05,.4,.4], [.4,.2,.9,.85], [.01,.7,.2,1.]])
        sym = torch.tensor([[.8,.1,0.], [.4,.7,0.], [.3,.2,0.]])
        return features, gray, boxes, sym

    def test_configs_only_intended_changes(self):
        folder = ROOT / 'configs/roi_role_query'
        configs = [load_config(folder / name) for name in (
            'a0_original_learned_alpha.py', 'f1_roi_content_query.py', 'f2_roi_paired_query.py')]
        original = load_config(ROOT / 'configs/publication_alpha/_original_a0.py')
        for config in configs:
            for key in ('data', 'optimizer', 'optimizer_config', 'runner', 'evaluation', 'lr_config'):
                self.assertEqual(config[key], original[key], key)
            self.assertTrue(config['model']['selector_gate_trainable'])
            self.assertTrue(config['model']['quality_detach'])
            self.assertEqual(config['model']['selector_fusion_init'], .5)
            self.assertEqual(config['model']['selector_quality_target'], 'coco_all_iou')
            delta = {k for k in set(config['model']) | set(original['model'])
                     if config['model'].get(k) != original['model'].get(k)}
            self.assertLessEqual(delta, {'type', 'experiment_tag', 'selector_gate_trainable',
                                        'roi_pool_size', 'roi_paired_boundary'})
            self.assertIsNone(config['load_from'])
            self.assertIsNone(config['resume_from'])
        self.assertFalse(configs[1]['model']['roi_paired_boundary'])
        self.assertTrue(configs[2]['model']['roi_paired_boundary'])

    def test_padding_coordinates_and_invalid_mask(self):
        # Valid shape 8x4, padded 8x8: valid x=.5 maps to padded x=.25.
        feature = torch.arange(8.).reshape(1,1,1,8).expand(1,1,8,8)
        points = torch.tensor([[[[.5,.5], [1.5,.5]]]])
        values, valid = sample_points(feature, points, torch.tensor([.5,1.]))
        torch.testing.assert_close(values[0,0,0,0], torch.tensor(1.5))
        self.assertFalse(bool(valid[0,0,1]))
        self.assertEqual(float(values[0,0,1,0]), 0.)

    def test_historical_query_and_common_initialization(self):
        torch.manual_seed(42)
        old = self.scope['DetailQualityAttention'](hidden=24, heads=4, texture=False, edges=True, detail=True)
        # Inheritance plus fork_rng leaves all common weights and global RNG identical.
        torch.manual_seed(42); new = self.build()
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        geometry = torch.randn(3,24)
        expected = geometry[:,None] + old.query_tokens[None]
        torch.testing.assert_close(old._initial_query(torch.rand(3,4), geometry, ()), expected,
                                   rtol=0, atol=0)

    def test_forward_backward_chunk_empty_and_native_layout(self):
        for paired, detail, texture in ((False,True,False), (True,True,True), (True,False,False)):
            with self.subTest(paired=paired, detail=detail, texture=texture):
                torch.manual_seed(4)
                model = self.build(paired, detail, texture)
                features, gray, boxes, sym = self.inputs()
                # Zero head is historical initialization. Check supervised updates
                # across two steps, not a zero-output-squared pseudo-smoke test.
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                before = model.roi_reader.v.weight.detach().clone()
                for _ in range(2):
                    optimizer.zero_grad()
                    memory = model.prepare(features, gray, (28,24), (32,32))
                    if detail:
                        self.assertEqual(memory[-2][0].shape[-2:], (14,12))
                        self.assertEqual(memory[-1][0].shape[-2:], (7,6))
                    output = model(boxes, memory, sym, training_checkpoint=True)[0]
                    self.assertEqual(output.shape, (3,2))
                    loss = F.binary_cross_entropy_with_logits(output[:,0], torch.tensor([.9,.3,.7]))
                    loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                        if p.grad is not None))
                    optimizer.step()
                self.assertGreater(float((before-model.roi_reader.v.weight).abs().max()), 0.)
                self.assertGreater(float(model.visual_projection[0].weight.grad.abs().sum()), 0.)
                if paired:
                    self.assertGreater(float(model.pair_reader.compare[-1].weight.grad.abs().sum()), 0.)
                model.eval()
                with torch.no_grad():
                    memory = model.prepare(features, gray, (28,24), (32,32))
                    result = model(boxes, memory, sym)[0]
                    model.chunk = 8
                    other = model(boxes, memory, sym)[0]
                    torch.testing.assert_close(result, other, rtol=1e-4, atol=1e-6)
                    order = torch.tensor([2,0,1])
                    shuffled = model(boxes[order], memory, sym[order])[0]
                    torch.testing.assert_close(other[order], shuffled, rtol=1e-4, atol=1e-6)
                    self.assertEqual(model(boxes[:0], memory, sym[:0])[0].shape, (0,2))

    def test_local_content_sensitivity_and_pair_mask(self):
        model = self.build(paired=True, detail=False)
        _, _, boxes, _ = self.inputs()
        maps = [torch.randn(1,24,h,w,requires_grad=True) for h,w in ((8,8),(4,4),(2,2),(1,1))]
        query = torch.randn(3,28,24)
        scale = torch.tensor([.75,.875])
        a = model.roi_reader(boxes, query, maps, scale)[0]
        b = model.roi_reader(boxes, query, [v * 0. for v in maps], scale)[0]
        self.assertGreater(float((a-b).abs().sum()), 0.)
        output, message, _, valid = model.pair_reader(boxes, a, maps, scale)
        self.assertEqual(message.shape, (3,24,24))
        self.assertEqual(valid.shape, (3,4,3))
        torch.testing.assert_close(output[:,:4], a[:,:4])
        output.square().mean().backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in maps))

    def test_scale_gate_query_cannot_cancel_in_softmax(self):
        for reader in (LocalRoleReader(24,4), PairedBoundaryReader(24)):
            with self.subTest(reader=type(reader).__name__):
                torch.manual_seed(11)
                stack = torch.randn(3,24,4,24)
                query = torch.randn(3,24,24)
                gate = reader.level_gate
                def weights(q):
                    return gate(torch.cat((stack,q[:,:,None].expand_as(stack)),-1)).squeeze(-1).softmax(-1)
                a, b = weights(query), weights(query*5.+3.)
                self.assertGreater(float((a-b).abs().max()),1e-4)
                (a*torch.randn_like(a)).sum().backward()
                first = gate if isinstance(gate,nn.Linear) else gate[0]
                self.assertGreater(float(first.weight.grad[:,24:].abs().max()),1e-5)

    def test_pair_message_uses_visual_query_not_only_coordinates(self):
        torch.manual_seed(3)
        reader = PairedBoundaryReader(24)
        boxes = torch.tensor([[.2,.2,.8,.8],[.1,.1,.6,.7]])
        maps = [torch.randn(1,24,s,s) for s in (8,4,2,1)]
        query = torch.randn(2,28,24,requires_grad=True)
        a = reader(boxes,query,maps,torch.ones(2))[1]
        b = reader(boxes,query.detach()*5.+2.,maps,torch.ones(2))[1]
        self.assertGreater(float((a-b).abs().max()),1e-4)
        a.square().sum().backward()
        self.assertGreater(float(query.grad.abs().max()),1e-5)

    def test_existing_alpha_loss_only_trains_alpha(self):
        path = ROOT / 'TQ_SymNet/models/detail.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RetinaNAMDetailSelector')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_residual_loss')
        scope = dict(torch=torch, F=F,
                     balanced_mean=lambda x, actual, matched: x.mean())
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        from types import SimpleNamespace
        gate = nn.Parameter(torch.zeros(()))
        obj = SimpleNamespace(selector_learned_gate=True,
                              candidate_quality=SimpleNamespace(fusion_gate_logit=gate))
        base = torch.tensor([1.,-.5], requires_grad=True)
        q = torch.tensor([-.3,2.], requires_grad=True)
        actual = torch.tensor([.9,.2])
        scope['_residual_loss'](obj, base, q, actual, actual, torch.tensor([0,0])).backward()
        self.assertGreater(float(gate.grad.abs()), 0.)
        self.assertIsNone(q.grad)
        self.assertIsNone(base.grad)

    def test_audit_distinguishes_active_and_legacy_rank(self):
        scope = dict(self.scope,Hook=object,Path=Path,json=json)
        load_definitions(ROOT/'TQ_SymNet/models/roi_query.py',{'TQROIQueryAuditHook'},scope)
        model = nn.Module()
        model.backbone=nn.Linear(2,2).requires_grad_(False)
        model.candidate_quality=self.build(paired=True)
        model.candidate_quality.fusion_gate_logit=nn.Parameter(torch.zeros(()))
        model.quality_detach=model.selector_learned_gate=model.selector_gate_trainable=True
        model.selector_tb_fusion='max'; model.selector_quality_target='coco_all_iou'
        model.selector_edge_weight=model.selector_fusion_loss_weight=0.
        model.selector_rank_weight=.1
        model.quality_rank_weight=.9  # legacy flag is recorded, not an active loss
        model.selector_sigma=.7; model.oracle_interval=4
        model.integrated_build_id='test-querygate-v2'
        runner=SimpleNamespace(model=model,meta=None,logger=Mock(),work_dir=str(ROOT),
            optimizer=SimpleNamespace(param_groups=[dict(params=list(model.candidate_quality.parameters()))]))
        with patch.object(Path,'write_text'):
            scope['TQROIQueryAuditHook']().before_run(runner)
        row=runner.meta['roi_query_contract']
        self.assertEqual(row['active_rank_loss'],'loss_selector_competition')
        self.assertEqual(row['active_rank_weight'],.1)
        self.assertEqual(row['legacy_quality_rank_weight'],.9)
        self.assertFalse(row['legacy_quality_rank_active'])
        self.assertNotIn('quality_rank',row)

    def test_actual_model_replacement_preserves_state_and_optimizer_flags(self):
        detail_class = self.scope['DetailQualityAttention']
        class RetainedModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.quality_arch, self.quality_detach = 'attention', True
                self.selector_learned_gate = self.selector_gate_trainable = True
                self.selector_quality_trainable = True
                self.backbone = nn.Linear(2,2).requires_grad_(False)
                self.candidate_quality = detail_class(hidden=24, heads=4, edges=True,
                                                     detail=True, texture=False)
                self.candidate_quality.fusion_gate_logit = nn.Parameter(torch.zeros(()))
                self.candidate_quality.tb_lse_temperature_raw = nn.Parameter(
                    torch.tensor(.5), requires_grad=False)
        scope = dict(self.scope, TQSymNet=RetainedModel)
        load_definitions(ROOT / 'TQ_SymNet/models/roi_query.py', {'TQSymNetROIQuery'}, scope)
        torch.manual_seed(10); control = RetainedModel()
        torch.manual_seed(10); f1 = scope['TQSymNetROIQuery']()
        torch.manual_seed(10); f2 = scope['TQSymNetROIQuery'](roi_paired_boundary=True)
        for model in (f1, f2):
            for key, value in control.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
            self.assertTrue(model.candidate_quality.fusion_gate_logit.requires_grad)
            self.assertFalse(model.candidate_quality.tb_lse_temperature_raw.requires_grad)
        for key, value in f1.state_dict().items():
            torch.testing.assert_close(value, f2.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
