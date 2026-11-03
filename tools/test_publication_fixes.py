"""Regression tests for the review findings; exercise actual evaluator code."""
import ast
import contextlib
import copy
import importlib.util
import io
import itertools
import logging
import sys
import types
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_definitions(path, names, scope):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in names]
    exec(compile(tree, str(path), 'exec'), scope)


class ReviewRegressionTests(unittest.TestCase):
    def evaluator(self, predictions, *, official=True, single=False, classwise=False):
        path = ROOT/'mmdet/datasets/coco.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        owner = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CocoDataset')
        method = next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        helpers = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_load_coco_results']
        # Only external IO/printing is stubbed. The production evaluate method,
        # real COCO matching, accumulation and all three branches execute.
        tb = {}
        source_definitions(ROOT/'mmdet/datasets/tbx11k_eval.py',
            {'_xywh_iou','merge_tb_predictions','make_single_tb_ground_truth'},
            tb := dict(np=np, deepcopy=copy.deepcopy, TB_CATEGORY_ID=1))
        scope = dict(np=np, logging=logging, itertools=itertools, OrderedDict=OrderedDict,
            copy=copy, COCO=COCO, COCOeval=COCOeval,
            mmcv=types.SimpleNamespace(load=lambda _:copy.deepcopy(predictions)),
            print_log=lambda *a,**k:None,
            AsciiTable=lambda rows:types.SimpleNamespace(table=''),
            merge_tb_predictions=tb['merge_tb_predictions'],
            make_single_tb_ground_truth=tb['make_single_tb_ground_truth'])
        exec(compile(ast.Module(body=helpers+[method],type_ignores=[]),str(path),'exec'),scope)
        gt = COCO()
        gt.dataset = dict(info={}, images=[dict(id=1,width=256,height=256)],
            categories=[dict(id=1,name='Active'),dict(id=2,name='Obsolete')],
            annotations=[dict(id=1,image_id=1,category_id=1,bbox=[10,10,40,40],area=1600,iscrowd=0),
                         dict(id=2,image_id=1,category_id=2,bbox=[100,100,100,100],area=10000,iscrowd=0)])
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            obj = types.SimpleNamespace(format_results=lambda *a:({'bbox':'memory.json'},None),
                                        coco=gt,cat_ids=[1,2],img_ids=[1])
            result = scope['evaluate'](obj,[[np.empty((0,5)),np.empty((0,5))]],
                tb_official_agnostic=official, tb_class_agnostic=single, classwise=classwise)
        return result, gt

    def test_empty_predictions_all_evaluation_branches(self):
        result, gt = self.evaluator([], single=True, classwise=True)
        for prefix in ('bbox','official_tb_bbox','tb_bbox'):
            for metric in ('mAP','mAP_50','mAP_75','mAP_m','mAP_l'):
                self.assertEqual(result[prefix+'_'+metric],0.)
            self.assertEqual(result[prefix+'_mAP_s'],-1.)  # no small GT, not fabricated zero
        self.assertEqual(len(gt.dataset['annotations']),2)
        self.assertEqual(result['bbox_Active_mAP'],0.)
        self.assertEqual(result['bbox_Obsolete_mAP'],0.)

    def test_nonempty_predictions_unchanged_official_coco(self):
        predictions = [dict(image_id=1,category_id=1,bbox=[10,10,40,40],score=.9),
                       dict(image_id=1,category_id=2,bbox=[100,100,100,100],score=.8)]
        before = copy.deepcopy(predictions)
        result, gt = self.evaluator(predictions)
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(copy.deepcopy(predictions))
            ev = COCOeval(gt,dt,'bbox'); ev.params.useCats=0
            ev.params.catIds=[1,2,3]; ev.params.imgIds=[1]; ev.params.maxDets=[1,10,100]
            ev.evaluate(); ev.accumulate(); ev.summarize()
        self.assertEqual(result['official_tb_bbox_mAP'],float('%.3f'%ev.stats[0]))
        self.assertEqual(predictions,before)

    def test_roi_smoke_has_real_update_and_freezes_detector(self):
        from TQ_SymNet.supplement_smoke import quality_update_smoke
        scope = dict(torch=torch,nn=nn,F=F)
        source_definitions(ROOT/'TQ_SymNet/models/light_quality.py',{'PooledQualityScorer'},scope)
        core = nn.Module()
        core.backbone = nn.Linear(3,4).requires_grad_(False)
        core.candidate_quality = scope['PooledQualityScorer'](hidden=16,mlp_hidden=32)
        result = quality_update_smoke(core,'cpu')
        self.assertGreater(result['nonzero_gradient_parameters'],0)
        self.assertGreater(result['changed_quality_parameters'],0)
        self.assertTrue(result['detector_unchanged'])
        self.assertGreater(result['first_loss'],0.)

    def test_smoke_rejects_vacuous_zero_gradient(self):
        from TQ_SymNet.supplement_smoke import checked_optimizer_update
        model = nn.Linear(3,1)
        optimizer = torch.optim.SGD(model.parameters(),lr=.1)
        loss = model(torch.zeros(2,3)).sum()*0.
        with self.assertRaisesRegex(RuntimeError,'no nonzero gradients'):
            checked_optimizer_update(model,optimizer,loss)

    def test_online_export_whitelist_preserves_safety(self):
        spec = importlib.util.spec_from_file_location('online_for_test',ROOT/'tools/export_online.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for kind in ('TQDetector','TQDenseHeadBaseline'):
            cfg = dict(type=kind)
            self.assertFalse(module.configure_export_model(cfg,'tb'))
            with self.assertRaises(ValueError): module.configure_export_model(cfg,'subtype')
        cfg = dict(type='TQSymNet')
        self.assertTrue(module.configure_export_model(cfg,'subtype'))
        self.assertEqual(cfg['inference_mode'],'subtype')
        self.assertFalse(cfg['diagnostics'])
        with self.assertRaises(ValueError): module.configure_export_model(dict(type='Unknown'),'tb')


if __name__ == '__main__':
    unittest.main(verbosity=2)
