"""G1 continuation guards: no real checkpoint loading, small CPU model fixtures."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from check_structure import load_config
from test_geometry_winner import scope_with_models
from test_roi_continuation import LRInterface

torch.set_num_threads(2)


def checkpoint_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hook_class():
    path = ROOT/'TQ_SymNet/geometry_winner_continuation.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for node in tree.body:
        node.decorator_list = []
    scope = dict(LrUpdaterHook=LRInterface, math=math, Path=Path, json=json,
                 checkpoint_sha256=checkpoint_sha256)
    exec(compile(tree, str(path), 'exec'), scope)
    return scope['TQG1FinetuneLrUpdaterHook']


def make_runner(folder):
    source = folder/'g1_s0'
    source.mkdir()
    checkpoint = source/'epoch_8.pth'
    checkpoint.write_bytes(b'unit-test-epoch8-placeholder-not-a-pytorch-file')
    best = source/'best_official_tb_bbox_mAP_epoch_5.pth'
    best.write_bytes(b'preserve-original-best-epoch5')
    model = nn.Module()
    model.backbone = nn.Linear(2, 2).requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model.candidate_quality = scope_with_models()['SignedGeometryQualityAttention'](
            hidden=96, heads=6, edges=True, detail=True, texture=True, paired_boundary=True)
    q = model.candidate_quality
    probability = (.5371760725975037-.1)/.8
    q.fusion_gate_logit = nn.Parameter(torch.tensor(math.log(probability/(1-probability))))
    q.tb_lse_temperature_raw = nn.Parameter(torch.zeros(()), requires_grad=False)
    model.integrated_build_id = 'TQ-F2-signed-geometry-goodset-v1'
    model.quality_arch = 'attention'
    model.quality_detach = model.selector_learned_gate = model.selector_gate_trainable = True
    model.selector_quality_rank_only = True
    model.selector_quality_target, model.selector_tb_fusion = 'coco_all_iou', 'max'
    model.selector_rank_weight, model.selector_sigma = .1, .7
    model.selector_fusion_loss_weight = model.selector_edge_weight = 0.
    model.geometry_weight = model.geometry_beta = .05
    model.geometry_ambiguity_gap, model.winner_max, model.winner_temperature = .1, 4, .5
    model.active_rank_name = 'loss_winner_set'
    parameters = (q.fusion_gate_logit, q.high_head.weight, q.visual_projection[0].weight,
        q.roi_reader.level_gate[0].weight, q.pair_reader.level_gate[0].weight,
        q.signed_aux[1].weight, q.signed_aux[-1].weight)
    groups = [dict(params=[p for p in q.parameters() if p.requires_grad],
                   initial_lr=1e-4, lr=3.806e-6)]
    state = {p: dict(step=3632, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
             for p in parameters}
    meta = dict(seed=0, roi_query_contract=dict(model=model.integrated_build_id,
        trainable=2580065, alpha_trainable=True, active_rank_loss='loss_winner_set'),
        geometry_winner_contract=dict(geometry_weight=.05, geometry_beta=.05, ambiguity_gap=.1,
            winners_max=4, winner_temperature=.5, rank='goodset_replaces_pair_rank',
            count_normalized=True, rival_cap=None, unchosen_equivalent='neutral',
            geometry_only_training=True, output_boxes_unchanged=True),
        tq_initialization=dict(detector_checkpoint_sha256=
            '4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74'),
        hook_msgs=dict(best_ckpt=str(best), last_ckpt=str(checkpoint), best_score=.401))
    runner = SimpleNamespace(model=model, optimizer=SimpleNamespace(param_groups=groups, state=state),
        meta=meta, epoch=8, iter=3632, max_epochs=12, work_dir=str(folder/'g1_continue'),
        hooks=[SimpleNamespace(best_ckpt_path=str(best))], data_loader=range(454),
        logger=logging.getLogger('g1-continuation-test'))
    return runner, checkpoint, best


class G1ContinuationTests(unittest.TestCase):
    def test_config_preserves_recipe_and_resumes_last_not_best(self):
        old = load_config(ROOT/'configs/quality_choice/g1_f2_geometry_winner.py')
        new = load_config(ROOT/'configs/quality_choice/g1_continue_e8_e12.py')
        for key in ('data', 'optimizer', 'optimizer_config', 'evaluation', 'custom_hooks', 'fp16'):
            self.assertEqual(old[key], new[key], key)
        a, b = copy.deepcopy(old['model']), copy.deepcopy(new['model'])
        a.pop('experiment_tag'); b.pop('experiment_tag')
        self.assertEqual(a, b)
        self.assertEqual(new['runner']['max_epochs'], 12)
        self.assertEqual(new['resume_from'], 'work_dirs/quality_choice/g1_s0/epoch_8.pth')
        self.assertEqual(new['resume_from'], new['lr_config']['source_checkpoint'])
        self.assertIsNone(new['load_from'])
        self.assertEqual(new['lr_config']['quality_lr'], 1e-5)

    def test_schedule_alpha_moments_and_original_files_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, source, best = make_runner(Path(tmp))
            originals = {p: p.read_bytes() for p in (source, best)}
            alpha = runner.model.candidate_quality.fusion_gate_logit.detach().clone()
            states = runner.optimizer.state
            hook = hook_class()(source_checkpoint=str(source))
            hook.before_run(runner)
            self.assertIs(runner.optimizer.state, states)
            torch.testing.assert_close(runner.model.candidate_quality.fusion_gate_logit, alpha, rtol=0, atol=0)
            self.assertEqual(runner.meta['hook_msgs']['best_score'], .401)
            self.assertNotIn('best_ckpt', runner.meta['hook_msgs'])
            self.assertNotIn('last_ckpt', runner.meta['hook_msgs'])
            self.assertIsNone(runner.hooks[0].best_ckpt_path)
            for epoch, expected in zip(range(8, 12), (1e-5, 8.5355339059e-6, 5e-6, 1.4644660941e-6)):
                runner.epoch = epoch
                hook.before_train_epoch(runner)
                self.assertAlmostEqual(runner.optimizer.param_groups[0]['lr'], expected, places=12)
            self.assertEqual(runner.optimizer.param_groups[0]['initial_lr'], 1e-4)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)

    def test_interrupted_resume_keeps_internal_best_and_no_double_lr_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, source, _ = make_runner(Path(tmp))
            hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.epoch, runner.iter = 10, 4540
            inside = str(Path(runner.work_dir)/'best_official_tb_bbox_mAP_epoch_10.pth')
            runner.meta['hook_msgs'].update(best_ckpt=inside, best_score=.404)
            runner.hooks[0].best_ckpt_path = inside
            hook_class()(source_checkpoint=str(source)).before_run(runner)
            self.assertAlmostEqual(runner.optimizer.param_groups[0]['lr'], 5e-6)
            self.assertEqual(runner.meta['hook_msgs']['best_ckpt'], inside)
            self.assertEqual(runner.hooks[0].best_ckpt_path, inside)

    def test_reject_best_epoch5_cold_start_and_missing_auxiliary_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, source, _ = make_runner(Path(tmp))
            runner.epoch, runner.iter = 5, 2270
            with self.assertRaisesRegex(RuntimeError, 'Resume epoch8'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.epoch = runner.iter = 0
            with self.assertRaisesRegex(RuntimeError, 'Resume epoch8'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.epoch, runner.iter = 8, 3632
            runner.optimizer.state.pop(runner.model.candidate_quality.signed_aux[-1].weight)
            with self.assertRaisesRegex(RuntimeError, 'AdamW state'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)

    def test_reject_reset_alpha_and_wrong_geometry_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, source, _ = make_runner(Path(tmp))
            runner.meta['geometry_winner_contract']['rival_cap'] = 32
            with self.assertRaisesRegex(RuntimeError, 'provenance'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.meta['geometry_winner_contract']['rival_cap'] = None
            with torch.no_grad():
                runner.model.candidate_quality.fusion_gate_logit.zero_()
            with self.assertRaisesRegex(RuntimeError, 'alpha was not restored'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)

    def test_reject_source_reuse_changed_schedule_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, source, _ = make_runner(Path(tmp))
            output = runner.work_dir
            runner.work_dir = str(source.parent)
            with self.assertRaisesRegex(RuntimeError, 'NEW output'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.work_dir = output
            hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.epoch, runner.iter = 9, 4086
            with self.assertRaisesRegex(RuntimeError, 'schedule/source changed'):
                hook_class()(source_checkpoint=str(source), quality_lr=2e-5).before_run(runner)
            runner.data_loader = range(227)
            with self.assertRaisesRegex(RuntimeError, '454 steps'):
                hook_class()(source_checkpoint=str(source)).before_train_epoch(runner)

    def test_reject_empty_source_or_restarting_completed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, source, _ = make_runner(Path(tmp))
            saved = source.read_bytes()
            source.write_bytes(b'')
            with self.assertRaisesRegex(RuntimeError, 'Missing/empty'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)
            source.write_bytes(saved)
            hook_class()(source_checkpoint=str(source)).before_run(runner)
            runner.meta.pop('tq_g1_continuation')
            with self.assertRaisesRegex(RuntimeError, 'Continuation exists'):
                hook_class()(source_checkpoint=str(source)).before_run(runner)


if __name__ == '__main__':
    unittest.main(verbosity=2)
