"""Frozen-detector ROI-content role queries, with optional paired semantics.

Import explicitly in the new configs. Old model/config registrations do not
change. Same candidate set, quality target, rank, final alpha and Soft-NMS.
"""
import json
from pathlib import Path

import torch
from torch import nn
from mmcv.runner import HOOKS, Hook
from mmdet.models.builder import DETECTORS

from ..roi_query_math import LocalRoleReader, PairedBoundaryReader
from .detail import DetailQualityAttention
from .network import TQSymNet


class ROIContentQualityAttention(DetailQualityAttention):
    def __init__(self, *args, paired_boundary=False, roi_pool_size=5, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.edges:
            raise ValueError('ROI role experiment retains all 28 roles')
        self.paired_boundary = bool(paired_boundary)
        self.roi_pool_size = int(roi_pool_size)
        with torch.random.fork_rng(devices=[]):
            self.roi_reader = LocalRoleReader(self.hidden, self.heads, self.roi_pool_size)
            self.pair_reader = PairedBoundaryReader(self.hidden) if self.paired_boundary else None
        self.roi_debug = {}

    def prepare(self, features, gray, valid_shape, batch_shape):
        memories = super().prepare(features, gray, valid_shape, batch_shape)
        scale = features[0].new_tensor((valid_shape[1] / batch_shape[1],
                                       valid_shape[0] / batch_shape[0]))
        # Share the retained 1x1 projection WEIGHTS; never backprop into the
        # frozen input features. No detached copy of these quality projections.
        extra = [(projection(feature),) for projection, feature
                 in zip(self.visual_projection, features[:4])] + [(scale,)]
        if self.detail:
            # Native detail maps must remain last: existing detail/debug code
            # deliberately consumes memory[-2:] and strips them before _score.
            memories[-2:-2] = extra
        else:
            memories.extend(extra)
        return memories

    def _initial_query(self, boxes, geometry, memories):
        query = super()._initial_query(boxes, geometry, memories)
        maps, scale = memories[-5:-1], memories[-1]
        result, update, level_weights = self.roi_reader(boxes, query, maps, scale)
        message = update.new_zeros((len(boxes), 24, self.hidden))
        cosine = update.new_zeros((len(boxes), 4, 3))
        valid = torch.zeros_like(cosine, dtype=torch.bool)
        if self.pair_reader is not None:
            result, message, cosine, valid = self.pair_reader(boxes, result, maps, scale)
        # No mining-pass duplication; legacy reentrant checkpoint records the
        # supervised recomputation once. Constant-size GPU reductions only.
        if not self.training or torch.is_grad_enabled():
            with torch.no_grad():
                values = torch.cat((torch.stack((
                    boxes.new_tensor(float(len(boxes))),
                    query.norm(dim=-1).mean(1).sum(),
                    update.norm(dim=-1).mean(1).sum(),
                    message.norm(dim=-1).mean(1).sum(),
                    (cosine * valid).sum(), valid.sum().to(boxes.dtype))),
                    level_weights.mean(1).sum(0)))
                split = 'train' if self.training else 'val'
                self.roi_debug[split] = self.roi_debug.get(split, 0.) + values.detach()
        return result


@DETECTORS.register_module()
class TQSymNetROIQuery(TQSymNet):
    def __init__(self, *args, roi_paired_boundary=False, roi_pool_size=5, **kwargs):
        super().__init__(*args, **kwargs)
        if self.quality_arch != 'attention' or not self.quality_detach:
            raise ValueError('ROI Query experiment requires attention and frozen detector')
        if not self.selector_learned_gate or not self.selector_gate_trainable:
            raise ValueError('These experiments retain trainable scalar alpha')
        previous = self.candidate_quality
        with torch.random.fork_rng(devices=[]):
            replacement = ROIContentQualityAttention(
                hidden=previous.hidden, heads=previous.heads,
                texture=previous.texture, gaussian=previous.gaussian,
                chunk=previous.chunk, checkpoint_candidates=previous.checkpoint_candidates,
                edges=previous.edges, detail=previous.detail,
                lbp_enabled=previous.lbp_enabled, gabor_enabled=previous.gabor_enabled,
                paired_boundary=roi_paired_boundary, roi_pool_size=roi_pool_size)
        replacement.fusion_gate_logit = nn.Parameter(
            previous.fusion_gate_logit.detach().clone(), requires_grad=True)
        replacement.tb_lse_temperature_raw = nn.Parameter(
            previous.tb_lse_temperature_raw.detach().clone(), requires_grad=False)
        result = replacement.load_state_dict(previous.state_dict(), strict=False)
        if result.unexpected_keys or any(not k.startswith(('roi_reader.', 'pair_reader.'))
                                         for k in result.missing_keys):
            raise RuntimeError('Unexpected retained-reader transfer: %s' % (result,))
        for parameter in replacement.parameters():
            parameter.requires_grad_(self.selector_quality_trainable)
        replacement.tb_lse_temperature_raw.requires_grad_(False)
        replacement.fusion_gate_logit.requires_grad_(self.selector_gate_trainable)
        self.candidate_quality = replacement
        self.integrated_build_id = ('TQ-ROI-content-role-paired-querygate-v2' if roi_paired_boundary
                                    else 'TQ-ROI-content-role-querygate-v2')


@HOOKS.register_module()
class TQROIQueryAuditHook(Hook):
    """Same guard for control/F1/F2; old diagnostics and alpha audit remain."""
    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        q = model.candidate_quality
        if (not model.quality_detach or not model.selector_learned_gate
                or not model.selector_gate_trainable or model.selector_tb_fusion != 'max'
                or model.selector_quality_target != 'coco_all_iou'
                or model.selector_edge_weight != 0 or model.selector_fusion_loss_weight != 0):
            raise RuntimeError('ROI experiment target/fusion/frozen contract changed')
        if model.selector_rank_weight != .1:
            raise RuntimeError('ROI experiment must retain the active quality-only rank weight')
        if any(p.requires_grad for n, p in model.named_parameters()
               if not n.startswith('candidate_quality.')):
            raise RuntimeError('Detector received a trainable parameter')
        trainable = [p for p in q.parameters() if p.requires_grad]
        grouped = [p for g in runner.optimizer.param_groups for p in g['params']]
        if any(sum(p is item for item in grouped) != 1 for p in trainable):
            raise RuntimeError('Each trainable quality parameter needs one optimizer entry')
        row = dict(model=model.integrated_build_id,
            roi_content=isinstance(q, ROIContentQualityAttention),
            paired_boundary=getattr(q, 'paired_boundary', False),
            roi_grid=getattr(q, 'roi_pool_size', None),
            parameters=sum(p.numel() for p in q.parameters()),
            trainable=sum(p.numel() for p in trainable),
            frozen_detector=True, alpha_trainable=q.fusion_gate_logit.requires_grad,
            target=model.selector_quality_target,
            active_rank_loss=getattr(model,'active_rank_name','loss_selector_competition'), active_rank_weight=model.selector_rank_weight,
            legacy_quality_rank_weight=model.quality_rank_weight, legacy_quality_rank_active=False,
            softnms_sigma=model.selector_sigma, oracle_every=model.oracle_interval)
        if runner.meta is None:
            runner.meta = {}
        old = runner.meta.get('roi_query_contract')
        if old is not None and old != row:
            raise RuntimeError('Resume architecture/contract mismatch')
        runner.meta['roi_query_contract'] = row
        Path(runner.work_dir).mkdir(parents=True, exist_ok=True)
        (Path(runner.work_dir) / 'roi_query_contract.json').write_text(
            json.dumps(row, indent=2), encoding='utf-8')
        runner.logger.info('[ROI-QUERY-BUILD] %s', json.dumps(row))
        self.model = model

    def before_train_epoch(self, runner):
        if hasattr(self.model.candidate_quality, 'roi_debug'):
            self.model.candidate_quality.roi_debug.clear()

    def after_train_epoch(self, runner):
        q = self.model.candidate_quality
        for split, values in getattr(q, 'roi_debug', {}).items():
            n, base, local, paired, cosine, valid, *weights = values.cpu().tolist()
            row = dict(split=split, epoch=runner.epoch + 1, observations=int(n),
                base_query_norm=base / max(n, 1), local_update_norm=local / max(n, 1),
                pair_update_norm=paired / max(n, 1),
                inner_outer_cosine=cosine / max(valid, 1), valid_boundary_pairs=int(valid),
                local_level_weight=[x / max(n, 1) for x in weights])
            runner.logger.info('[ROI-QUERY-INTERNAL] %s', json.dumps(row))
        if hasattr(q, 'roi_debug'):
            q.roi_debug.clear()
