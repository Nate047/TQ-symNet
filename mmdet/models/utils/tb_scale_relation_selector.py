"""V61: per-edge scale selection and optional full-pool candidate relations.

Both variants keep V59's frozen detector, quality target and Soft-NMS.  The
relation memory is always built from the complete pre-NMS pool, including when
the loss is computed on a mined subset of that pool.
"""
import math
import random
import json
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import HOOKS, Hook

from mmdet.models.builder import DETECTORS
from mmdet.core import bbox_overlaps
from .tb_detail_selector import (DetailQualityAttention, DetailSelectorDebugHook,
                                 RetinaNAMDetailSelector)
from .tb_quality_attention import CandidateQualityAttention
from .tb_high_quality_selector import HighQualitySelectorDebugHook


@contextmanager
def preserve_rng():
    """Diagnostics must not change subsequent training samples or augmentation."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def ranking_diagnostics(cache, dataset):
    """Same full-pool, GT85 and suppression-competitor definitions on both splits.

    One highest-true-IoU reference per GT85-eligible lesion is compared with
    same-GT/low-IoU competitors, overlap > .5 and true-IoU gap > .05.
    A tie receives half credit. No random pair subsampling is used.
    """
    totals = dict(images=len(cache), candidates=0, gt85_available=0,
                  final_gt85=0, pairs=0)
    for name in ('sym', 'quality', 'fused'):
        totals[name + '_gt85_top1'] = 0
        totals[name + '_pair_correct'] = 0.
    for index, item in enumerate(cache):
        boxes = item['boxes']
        totals['candidates'] += len(boxes)
        gt = torch.as_tensor(dataset.get_ann_info(index)['bboxes'],
                             dtype=boxes.dtype).reshape(-1, 4)
        if not len(boxes) or not len(gt):
            continue
        overlap = bbox_overlaps(boxes, gt)
        actual, matched = overlap.max(1)
        totals['final_gt85'] += int((overlap[item['source']] >= .85).any(0).sum())
        scores = dict(sym=item['sym'][:, :-1].max(1)[0],
                      quality=item['logit'], fused=item['corrected'])
        for g in range(len(gt)):
            eligible = (matched == g) & (actual >= .5)
            good = eligible & (actual >= .85)
            if not good.any():
                continue
            totals['gt85_available'] += 1
            ids = eligible.nonzero(as_tuple=False).squeeze(1)
            best = actual.masked_fill(~good, -1.).argmax()
            neighbors = bbox_overlaps(boxes[best:best+1], boxes)[0] > .5
            rivals = neighbors & ((matched == g) | (actual < .5))
            rivals &= actual < actual[best] - .05
            totals['pairs'] += int(rivals.sum())
            for name, score in scores.items():
                winner = ids[score[ids].argmax()]
                totals[name + '_gt85_top1'] += int(actual[winner] >= .85)
                gap = score[best] - score[rivals]
                totals[name + '_pair_correct'] += float((gap > 0).sum())
                totals[name + '_pair_correct'] += .5 * float((gap == 0).sum())
    for name in ('sym', 'quality', 'fused'):
        totals[name + '_top1_rate'] = (totals[name + '_gt85_top1'] /
                                     max(totals['gt85_available'], 1))
        totals[name + '_pair_accuracy'] = (totals[name + '_pair_correct'] /
                                         max(totals['pairs'], 1))
    return totals


class PerEdgeScaleQuality(DetailQualityAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, detail=True, **kwargs)
        hidden = self.hidden
        with torch.random.fork_rng(devices=[]):
            self.scale_q = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(6))
            self.scale_k = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(6))
            self.scale_v = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(6))
            self.scale_out = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(6))
            self.scale_selector = nn.Sequential(nn.Linear(hidden * 2, hidden),
                                                nn.SiLU(), nn.Linear(hidden, 1))
            self.scale_bias = nn.Parameter(torch.zeros(6))
            self.scale_fusion = nn.Sequential(
                nn.LayerNorm(hidden * 25), nn.Linear(hidden * 25, hidden * 2),
                nn.SiLU(), nn.Linear(hidden * 2, hidden))
            self.scale_gain = nn.Parameter(torch.tensor(.1))
        self.scale_debug = []

    def prepare(self, features, gray, valid_shape, batch_shape):
        memory = super().prepare(features, gray, valid_shape, batch_shape)
        # Use exactly the detector feature maps from the V59 path. The four
        # extra projections are shared by every candidate in this image.
        memory.extend((self.visual_projection[level](features[level]),)
                      for level in range(4))
        # Convert valid-image normalized coordinates to padded-map coordinates.
        ratio = gray.new_tensor([valid_shape[1] / batch_shape[1],
                                 valid_shape[0] / batch_shape[0]]).repeat(2)
        memory.append((ratio,))
        return memory

    def _edge_read(self, descriptor, boxes, feature, level):
        patches = self._sample_edge_patches(feature, boxes)
        offsets = boxes.new_tensor([[-1., -1.], [0., -1.], [1., -1.],
                                    [-1., 0.], [0., 0.], [1., 0.],
                                    [-1., 1.], [0., 1.], [1., 1.]])
        patches = patches + self.detail_position(offsets)[None, None]
        query = descriptor[:, None] + self.detail_tokens[None]
        n, heads, dim = len(boxes), self.heads, self.hidden // self.heads
        q = self.scale_q[level](query).reshape(n, 24, heads, dim)
        k = self.scale_k[level](patches).reshape(n, 24, 9, heads, dim)
        v = self.scale_v[level](patches).reshape(n, 24, 9, heads, dim)
        weights = (torch.einsum('nqhd,nqthd->nqht', q, k) /
                   math.sqrt(dim)).softmax(-1)
        attended = torch.einsum('nqht,nqthd->nqhd', weights, v).reshape(
            n, 24, self.hidden)
        return query + self.scale_out[level](attended)

    def _score(self, boxes, *inputs):
        # Four padded P maps plus a coordinate ratio follow V59's memories.
        descriptor, gate, entropy = super()._score(boxes, *inputs[:-5])
        maps = list(inputs[-5:-1]) + [inputs[-7], inputs[-6]]
        reads = torch.stack([self._edge_read(descriptor,
                             boxes * inputs[-1] if level < 4 else boxes, feature, level)
                             for level, feature in enumerate(maps)], 2)
        semantic = descriptor[:, None, None].expand(-1, 24, 6, -1)
        logits = self.scale_selector(torch.cat((semantic, reads), -1)).squeeze(-1)
        weights = (logits + self.scale_bias).softmax(-1)
        if not self.training:
            self.scale_debug.append(weights.detach().reshape(
                len(boxes), 4, 6, 6).sum((0, 2)).cpu())
        selected = (reads * weights[..., None]).sum(2)
        fused = self.scale_fusion(torch.cat((descriptor, selected.flatten(1)), 1))
        return descriptor + self.scale_gain.tanh() * fused, gate, entropy

    def forward_with_embedding(self, boxes, memories, sym,
                               training_checkpoint=False):
        # Same chunking/checkpointing as V59, but expose the visual descriptor
        # to the F2 relation head without re-running the expensive encoder.
        if not len(boxes):
            empty = boxes.new_empty((0, self.hidden))
            return (boxes.new_empty((0, 2)), empty,
                    boxes.new_empty((0, 7)), boxes.new_empty((0, 3)))
        descriptor, gate, entropy = CandidateQualityAttention.forward(
            self, boxes, memories, training_checkpoint=training_checkpoint)
        sym_logits = torch.logit(sym[:, :2].clamp(1e-6, 1. - 1e-6)).detach()
        residual = self.residual_head(torch.cat((descriptor, sym_logits), 1))
        return torch.cat((self.high_head(descriptor), residual), 1), descriptor, gate, entropy

    def forward(self, boxes, memories, sym, training_checkpoint=False):
        output, _, gate, entropy = self.forward_with_embedding(
            boxes, memories, sym, training_checkpoint)
        return output, gate, entropy


class LocalCandidateRelation(nn.Module):
    """A learned residual correction using nearby candidates."""
    def __init__(self, hidden, neighbors=16, min_overlap=.30):
        super().__init__()
        self.neighbors = int(neighbors)
        self.min_overlap = float(min_overlap)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.geometry = nn.Linear(5, hidden)
        self.output = nn.Sequential(nn.LayerNorm(hidden * 2),
                                    nn.Linear(hidden * 2, hidden), nn.SiLU(),
                                    nn.Linear(hidden, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, boxes, embedding, reference_boxes, reference_embedding):
        if not len(boxes) or not len(reference_boxes):
            return embedding.new_zeros((len(boxes),)), 0., 0.
        corrections, counts, valid_counts = [], 0, 0
        reference_embedding = reference_embedding.detach()
        for box_part, query_part in zip(boxes.split(64), embedding.split(64)):
            overlap = bbox_overlaps(box_part, reference_boxes)
            count = min(self.neighbors, len(reference_boxes))
            # Ignore exact self-matches: they provide no candidate relation.
            selectable = overlap.masked_fill(overlap >= 1. - 1e-6, -1.)
            _, ids = selectable.topk(count, dim=1)
            iou = overlap.gather(1, ids)
            near = (iou >= self.min_overlap) & (iou < 1. - 1e-6)
            ref_box = reference_boxes[ids]
            query_center = (box_part[:, :2] + box_part[:, 2:]) * .5
            ref_center = (ref_box[..., :2] + ref_box[..., 2:]) * .5
            query_size = (box_part[:, 2:] - box_part[:, :2]).clamp_min(1e-4)
            ref_size = (ref_box[..., 2:] - ref_box[..., :2]).clamp_min(1e-4)
            relative = torch.cat(((ref_center - query_center[:, None]) /
                                  query_size[:, None],
                                  (ref_size / query_size[:, None]).log(),
                                  iou[..., None]), -1)
            key = self.key(reference_embedding[ids]) + self.geometry(relative)
            score = ((self.query(query_part)[:, None] * key).sum(-1) /
                     math.sqrt(query_part.shape[-1]))
            score = score.masked_fill(~near, -1e4)
            attention = score.softmax(-1) * near.float()
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
            message = (attention[..., None] * self.value(reference_embedding[ids])).sum(1)
            correction = self.output(torch.cat((query_part, message), -1)).squeeze(-1)
            corrections.append(correction * near.any(1).float())
            counts += near.numel()
            valid_counts += int(near.sum())
        return self.gain.tanh() * torch.cat(corrections), valid_counts, counts


@DETECTORS.register_module()
class RetinaNAMScaleRelationSelector(RetinaNAMDetailSelector):
    def __init__(self, *args, selector_candidate_relation=False,
                 selector_relation_neighbors=16, **kwargs):
        kwargs.update(selector_detail=True, selector_learned_gate=True)
        super().__init__(*args, **kwargs)
        previous = self.candidate_quality
        with torch.random.fork_rng(devices=[]):
            self.candidate_quality = PerEdgeScaleQuality(
                hidden=previous.hidden, heads=previous.heads,
                texture=previous.texture, gaussian=previous.gaussian,
                chunk=previous.chunk,
                checkpoint_candidates=previous.checkpoint_candidates,
                edges=True)
            self.candidate_quality.fusion_gate_logit = nn.Parameter(torch.zeros(()))
            # Inherit the same optimizer multiplier, train/eval mode and
            # frozen-detector hash exclusion as all quality-network parameters.
            self.candidate_quality.relation = LocalCandidateRelation(
                previous.hidden, selector_relation_neighbors)
        self.selector_candidate_relation = bool(selector_candidate_relation)
        if not self.selector_candidate_relation:
            for parameter in self.candidate_relation.parameters():
                parameter.requires_grad_(False)
        self._relation_context = None
        self._relation_debug = [0, 0]
        self.integrated_build_id = 'v61.1-coordinate-rng-relation-scope-fixed'

    @property
    def candidate_relation(self):
        return self.candidate_quality.relation

    def _image_memory(self, features, img, meta, original, index):
        self._relation_context = None
        memory = super()._image_memory(features, img, meta, original, index)
        # Parent assumes D2/D4 are the last entries; V61 appends P maps/ratio.
        self.detail_debug[-1].update(d2=tuple(memory[-7][0].shape[-2:]),
                                     d4=tuple(memory[-6][0].shape[-2:]))
        return memory

    def v50_begin_oracle(self, epoch):
        self._relation_debug = [0, 0]
        self.candidate_quality.scale_debug = []
        self.detail_debug = []
        return super().v50_begin_oracle(epoch)

    def _score_candidates(self, boxes, memory, sym, training=False):
        output, embedding, gate, entropy = self.candidate_quality.forward_with_embedding(
            boxes, memory, sym, training_checkpoint=training)
        if self.selector_candidate_relation and len(boxes):
            if self._relation_context is None:
                # Full pre-NMS scan in both training and validation. Training
                # queries later use this same complete, detached context.
                self._relation_context = (boxes.detach(), embedding.detach())
            correction, valid, total = self.candidate_relation(
                boxes, embedding, *self._relation_context)
            output = output.clone()
            output[:, 0] = output[:, 0] + correction
            self._relation_debug[0] += valid
            self._relation_debug[1] += total
        high = output[:, 0]
        return torch.stack((high, high), 1), gate, entropy

    def v50_finish_oracle(self, dataset, runner):
        self.val_rank_diagnostics = ranking_diagnostics(self.quality_cache, dataset)
        runner.logger.info('[SCALE-RANK-EVAL] split=val epoch=%d stats=%s',
                           self.quality_epoch, json.dumps(self.val_rank_diagnostics))
        stats = self.candidate_quality.scale_debug
        if stats:
            weights = torch.stack(stats).sum(0)
            weights = (weights / weights.sum(1, keepdim=True).clamp_min(1e-8)).tolist()
            runner.logger.info('[SCALE-EDGE] rows=L/T/R/B cols=P2/P3/P4/P5/D2/D4 '
                               'weights=%s '
                               'gain=%.4f', weights,
                               float(self.candidate_quality.scale_gain.tanh().detach()))
        self.candidate_quality.scale_debug = []
        valid, total = self._relation_debug
        runner.logger.info('[SCALE-RELATION] enabled=%s validNeighborFraction=%.5f '
                           'neighbors=%d gain=%.4f', self.selector_candidate_relation,
                           valid / max(total, 1), self.candidate_relation.neighbors,
                           float(self.candidate_relation.gain.tanh().detach()))
        self._relation_debug = [0, 0]
        return super().v50_finish_oracle(dataset, runner)


@HOOKS.register_module()
class ScaleRelationDebugHook(DetailSelectorDebugHook):
    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        runner.logger.info('[SCALE-BUILD] perEdge=True levels=P2/P3/P4/P5/D2/D4 '
            'relation=%s relationFullPool=True detectorFrozen=True '
            'loss=V59-D3 gaussianSoftNMS=unchanged', model.selector_candidate_relation)

    def after_train_epoch(self, runner):
        HighQualitySelectorDebugHook.after_train_epoch(self, runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        alpha = float((.1 + .8 * model.candidate_quality.fusion_gate_logit.sigmoid()).detach())
        runner.logger.info('[SCALE-SCORE] final=sigmoid((1-alpha)*logitSym+'
                           'alpha*qualityLogit) alpha=%.5f rank=qualityOnly', alpha)


@HOOKS.register_module()
class ScaleTrainValEvaluationHook(Hook):
    """Evaluate unique training images with the validation inference pipeline.

    Runs after the ordinary EvalHook, before logging. It never selects or
    saves a checkpoint, so validation remains the sole model-selection set.
    """
    def __init__(self, train_ann_file, interval=2):
        self.train_ann_file = train_ann_file
        self.interval = int(interval)
        if self.interval < 1:
            raise ValueError('interval must be positive')
        self.train_loader = None

    def before_run(self, runner):
        from mmdet.core.evaluation.eval_hooks import EvalHook
        from mmdet.datasets import build_dataset, build_dataloader
        evaluation_hooks = [hook for hook in runner.hooks if isinstance(hook, EvalHook)]
        if len(evaluation_hooks) != 1:
            raise RuntimeError('Train/val diagnostic requires exactly one single-GPU EvalHook')
        val_dataset = evaluation_hooks[0].dataloader.dataset
        train_dataset = build_dataset(dict(
            type='CocoDataset', ann_file=self.train_ann_file,
            img_prefix=val_dataset.img_prefix, classes=val_dataset.CLASSES,
            pipeline=val_dataset.pipeline.transforms, test_mode=True))
        # The TBX11K train/val JSONs renumber COCO image IDs independently;
        # compare filenames, not numeric IDs (which overlap by design).
        train_files = {item['filename'] for item in train_dataset.data_infos}
        val_files = {item['filename'] for item in val_dataset.data_infos}
        overlap = train_files & val_files
        if overlap:
            raise RuntimeError('Train/val diagnostic filenames overlap: %d' % len(overlap))
        loader_generator = torch.Generator().manual_seed(torch.initial_seed())
        self.train_loader = build_dataloader(
            train_dataset, samples_per_gpu=1, workers_per_gpu=2,
            num_gpus=1, dist=False, shuffle=False,
            seed=torch.initial_seed() % (2**32), generator=loader_generator)
        runner.logger.info('[SCALE-TRAINVAL-SETUP] trainUnique=%d val=%d '
            'overlap=0 sharedValidationPipeline=True officialProtocol=True '
            'interval=%d checkpointSelection=valOnly', len(train_dataset),
            len(val_dataset), self.interval)

    def after_train_epoch(self, runner):
        if (runner.epoch + 1) % self.interval:
            return
        val_keys = ('official_tb_bbox_mAP', 'official_tb_bbox_mAP_50',
                    'official_tb_bbox_mAP_75')
        val_output = runner.log_buffer.output
        if any(key not in val_output for key in val_keys):
            runner.logger.warning('[SCALE-TRAINVAL] epoch=%d skipped: '
                'validation metrics not yet available', runner.epoch + 1)
            return
        val = [float(val_output[key]) for key in val_keys]
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        modes = [(module, module.training) for module in runner.model.modules()]
        model.v50_begin_oracle(runner.epoch + 1)
        from mmdet.apis import single_gpu_test
        try:
            with preserve_rng():
                predictions = single_gpu_test(runner.model, self.train_loader, show=False)
                result = self.train_loader.dataset.evaluate(
                    predictions, metric='bbox', tb_official_agnostic=True,
                    tb_class_agnostic=False)
                train_rank = ranking_diagnostics(model.quality_cache, self.train_loader.dataset)
        finally:
            model.candidate_quality.scale_debug = []
            model._relation_debug = [0, 0]
            model.detail_debug = []
            model.quality_collect = False
            model.quality_cache = []
            model.candidate_quality.attention_stats = []
            model._relation_context = None
            for module, training in modes:
                module.training = training
        train = [float(result[key]) for key in val_keys]
        gap = [a-b for a, b in zip(train, val)]
        runner.logger.info('[SCALE-TRAINVAL] epoch=%d protocol=official_tb_bbox '
            'trainUnique AP/AP50/AP75=%.3f/%.3f/%.3f '
            'val AP/AP50/AP75=%.3f/%.3f/%.3f '
            'gap(train-val)=%.3f/%.3f/%.3f '
            'samePipeline=True sameNMS=True noOversample=True',
            runner.epoch + 1, *(train + val + gap))
        runner.logger.info('[SCALE-RANK-EVAL] split=train epoch=%d stats=%s',
                           runner.epoch + 1, json.dumps(train_rank))
        val_rank = getattr(model, 'val_rank_diagnostics', {})
        rank_gaps = {key: train_rank[key] - val_rank[key] for key in train_rank
                     if key.endswith(('_top1_rate', '_pair_accuracy')) and key in val_rank}
        runner.logger.info('[SCALE-RANK-GAP] epoch=%d trainMinusVal=%s '
                           'RNGRestored=True', runner.epoch + 1, json.dumps(rank_gaps))
