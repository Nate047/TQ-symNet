"""Numerical CPU tests. CUDA/MMCV integration is a separate server check."""
import ast
import contextlib
import copy
import io
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from TQ_SymNet.supplement_math import dgqp_joint_logit, fused_probability
from TQ_SymNet.bootstrap_ap import ap_from_records, coco_all_area_records, paired_bootstrap


def definitions(path, names, scope):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(tree, str(path), 'exec'), scope)


class SupplementTests(unittest.TestCase):
    def test_alpha_endpoints(self):
        p = torch.tensor([.01, .2, .6, .99])
        q = torch.tensor([-2., 1., .3, 3.])
        torch.testing.assert_close(fused_probability(p, q, 0), p)
        torch.testing.assert_close(fused_probability(p, q, 1), q.sigmoid())
        torch.testing.assert_close(fused_probability(p, q, .5),
                                   (.5*torch.logit(p)+.5*q).sigmoid())
        with self.assertRaises(ValueError):
            fused_probability(p, q, 1.1)

    def test_roi_width_and_empty(self):
        scope = dict(torch=torch, nn=nn, F=F)
        definitions(ROOT/'TQ_SymNet/models/light_quality.py', {'PooledQualityScorer'}, scope)
        cls = scope['PooledQualityScorer']
        torch.manual_seed(0); a = cls()
        torch.manual_seed(0); b = cls(mlp_hidden=96)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        wide = cls(mlp_hidden=880, chunk=1)
        self.assertEqual(sum(p.numel() for p in wide.parameters()), 2341713)
        self.assertEqual(wide.visual_projection[0].out_channels, 96)
        with torch.no_grad(): wide.high_head.weight.normal_(std=.01)
        maps = [torch.randn(1,256,s,s) for s in (8,4,2,1)]
        boxes = torch.tensor([[.1,.2,.8,.9],[.3,.3,.5,.6]])
        memory = wide.prepare(maps, None, (28,30), (32,32))
        out = wide(boxes, memory, boxes.new_zeros((2,3)))[0]
        out.sum().backward()
        self.assertTrue(torch.isfinite(wide.visual_projection[0].weight.grad).all())
        self.assertEqual(wide(boxes[:0], memory, boxes.new_zeros((0,3)))[0].shape, (0,2))
        other = copy.deepcopy(wide); other.chunk = 32
        out2 = other(boxes, other.prepare(maps,None,(28,30),(32,32)), boxes.new_zeros((2,3)))[0]
        torch.testing.assert_close(out, out2, rtol=1e-5, atol=1e-6)

    def test_dgqp_preserved_forward_and_gradients(self):
        # Execute the actual preserved forward_single method, not a duplicate
        # hand-coded DGQP oracle. Full MMDet inheritance is checked on server.
        path = ROOT/'third_party/gfocalv2_reference/gfocal_head.py.reference'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GFocalHead')
        method = next(n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_single')
        module = ast.Module(body=[method], type_ignores=[])
        scope = dict(torch=torch, F=F)
        exec(compile(module, str(path), 'exec'), scope)
        torch.manual_seed(5)
        layer = nn.Module()
        layer.cls_convs = nn.ModuleList([nn.Conv2d(8,8,3,padding=1)])
        layer.reg_convs = nn.ModuleList([nn.Conv2d(8,8,3,padding=1)])
        layer.gfl_cls = nn.Conv2d(8,2,3,padding=1)
        layer.gfl_reg = nn.Conv2d(8,68,3,padding=1)
        layer.reg_conf = nn.Sequential(nn.Conv2d(20,64,1), nn.ReLU(), nn.Conv2d(64,1,1), nn.Sigmoid())
        layer.reg_max, layer.reg_topk, layer.add_mean = 16, 4, True
        reference = copy.deepcopy(layer)
        x = torch.randn(2,8,4,4)
        official, _ = scope['forward_single'](reference, x, lambda v:v)
        cls_logit = layer.gfl_cls(layer.cls_convs[0](x))
        distribution = layer.gfl_reg(layer.reg_convs[0](x))
        adapted = dgqp_joint_logit(cls_logit, distribution, layer.reg_conf).sigmoid()
        torch.testing.assert_close(adapted, official, rtol=1e-5, atol=1e-7)
        target = torch.rand_like(official)
        F.binary_cross_entropy(official, target).backward()
        F.binary_cross_entropy(adapted, target).backward()
        for (name,p), (other,q) in zip(layer.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-7)

    def test_coco_accumulation_and_repeated_image_bootstrap(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        def evaluate(doc, predictions):
            with contextlib.redirect_stdout(io.StringIO()):
                gt = COCO(); gt.dataset = doc; gt.createIndex()
                dt = gt.loadRes(predictions)
                ev = COCOeval(gt, dt, 'bbox')
                ev.params.useCats = 0; ev.params.maxDets = [1,10,100]
                ev.evaluate(); ev.accumulate(); ev.summarize()
            return ev
        doc = dict(info={}, images=[dict(id=i,width=200,height=200) for i in (1,2,3)],
            categories=[dict(id=1,name='TB'),dict(id=2,name='Obsolete')],
            annotations=[dict(id=1,image_id=1,category_id=1,bbox=[10,10,50,50],area=2500,iscrowd=0),
                         dict(id=2,image_id=2,category_id=2,bbox=[20,30,100,100],area=10000,iscrowd=0)])
        predictions = [dict(image_id=1,category_id=2,bbox=[10,10,50,50],score=.8),
                       dict(image_id=1,category_id=1,bbox=[11,12,50,50],score=.7),
                       dict(image_id=2,category_id=1,bbox=[20,30,95,100],score=.6),
                       dict(image_id=3,category_id=1,bbox=[1,1,20,20],score=.9)]
        ev = evaluate(doc, predictions)
        records = coco_all_area_records(ev)
        curve = ap_from_records(records)
        np.testing.assert_allclose([curve.mean(), curve[0], curve[5]], ev.stats[:3]*100, atol=1e-10)
        # Repeated image 1 must also duplicate its GT and its detection list.
        sample = [0,0,2]
        copied = dict(info={}, categories=doc['categories'], images=[], annotations=[])
        detections = []
        for new, old in enumerate(sample, 10):
            copied['images'].append(dict(id=new,width=200,height=200))
            for ann in doc['annotations']:
                if ann['image_id'] == old+1:
                    copied['annotations'].append(dict(ann,id=len(copied['annotations'])+1,image_id=new))
            detections.extend(dict(p,image_id=new) for p in predictions if p['image_id'] == old+1)
        repeated = evaluate(copied, detections)
        np.testing.assert_allclose(ap_from_records(records,sample),
                                   ap_from_records(coco_all_area_records(repeated)), atol=1e-10)
        result = paired_bootstrap(records,records,samples=20,seed=0)
        for value in result['differences_pp'].values():
            self.assertEqual(value['ci95'], [0.,0.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
