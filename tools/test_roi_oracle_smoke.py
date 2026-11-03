"""CPU control-flow regression for the REAL local EvalHook implementation.

Only the unavailable MMCV base and the GPU inference callback are substituted.
Full models, actual pipeline, COCO and exports are checked by --build-smoke
ON SERVER, not claimed completed by this test.
"""
import ast
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_roi_threshold import oracle_validation_smoke


def eval_hook_class():
    class Base:
        def __init__(self,loader,**kwargs):
            self.dataloader, self.save_best = loader, kwargs['save_best']
            self.kwargs=kwargs
        def before_run(self,runner): pass
        def _should_evaluate(self,runner): return runner.epoch==3
        def evaluate(self,runner,results):
            metrics=self.dataloader.dataset.evaluate(results,metric='bbox')
            runner.log_buffer.output.update(metrics)
            runner.log_buffer.ready=True
    path=ROOT/'mmdet/core/evaluation/eval_hooks.py'
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EvalHook']
    scope=dict(BaseEvalHook=Base)
    exec(compile(tree,str(path),'exec'),scope)
    return scope['EvalHook']


class Dataset:
    def __init__(self): self.calls=0
    def __len__(self): return 1
    def evaluate(self,*args,**kwargs):
        self.calls+=1
        return dict(official_tb_bbox_mAP=.2, official_tb_bbox_mAP_50=.5,
                    official_tb_bbox_mAP_75=.1)


class OracleSmokeTests(unittest.TestCase):
    def fixture(self,clear=True):
        trace=[]
        core=SimpleNamespace(diagnostics=False,quality_collect=False,quality_cache=[],
            distribution_val_cache=[],candidate_quality=SimpleNamespace(threshold_distribution=True))
        def begin(epoch):
            trace.append(('begin',epoch))
            core.quality_collect=core.diagnostics and epoch==4
        def finish(dataset,runner):
            trace.append(('finish',core.quality_collect))
            for _ in range(3): dataset.evaluate([],metric='bbox')
            if clear:
                core.quality_collect=False
                core.quality_cache=[]
                core.distribution_val_cache=[]
        core.v50_begin_oracle, core.v50_finish_oracle=begin, finish
        model=SimpleNamespace(module=core)
        dataset=Dataset()
        loader=SimpleNamespace(dataset=dataset,__len__=lambda:1)
        class Loader:
            def __init__(self): self.dataset=dataset
            def __len__(self): return 1
        loader=Loader()
        def infer(model,loader,show):
            trace.append(('inference',core.quality_collect))
            core.quality_cache.append('one_image')
            core.distribution_val_cache.append('thresholds')
            return [[]]
        apis=types.ModuleType('mmdet.apis')
        apis.single_gpu_test=infer
        runner=SimpleNamespace(model=model,epoch=0,work_dir=str(ROOT),logger=Mock())
        return core,runner,dataset,loader,apis,trace

    def test_real_eval_hook_calls_oracle_begin_and_finish(self):
        core,runner,dataset,loader,apis,trace=self.fixture()
        original=dataset.evaluate
        with patch.dict(sys.modules,{'mmdet.apis':apis}), \
             patch.object(Path,'is_file',return_value=True), \
             patch.object(Path,'write_text') as write:
            report=oracle_validation_smoke(core,runner,dataset,loader,eval_hook_class())
        self.assertEqual(trace,[('begin',4),('inference',True),('finish',True)])
        self.assertEqual(dataset.calls,4)
        self.assertEqual(list(report['evaluations']),['actual','pre','soft128','final'])
        self.assertFalse(report['benchmark_result'])
        self.assertEqual(report['trained_updates'],2)
        self.assertEqual(json.loads(write.call_args.args[0]),report)
        self.assertEqual(runner.epoch,0)
        self.assertFalse(core.diagnostics)
        self.assertEqual(dataset.evaluate,original)

    def test_missing_export_fails(self):
        core,runner,dataset,loader,apis,_=self.fixture()
        with patch.dict(sys.modules,{'mmdet.apis':apis}), \
             patch.object(Path,'is_file',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'export is missing'):
                oracle_validation_smoke(core,runner,dataset,loader,eval_hook_class())
        self.assertEqual(runner.epoch,0)
        self.assertFalse(core.diagnostics)

    def test_incomplete_cache_cleanup_fails(self):
        core,runner,dataset,loader,apis,_=self.fixture(clear=False)
        with patch.dict(sys.modules,{'mmdet.apis':apis}):
            with self.assertRaisesRegex(RuntimeError,'cleanup did not complete'):
                oracle_validation_smoke(core,runner,dataset,loader,eval_hook_class())
        self.assertEqual(runner.epoch,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
