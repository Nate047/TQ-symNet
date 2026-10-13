"""V59 high-quality selector with optional native-image detail memory."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import HOOKS

from mmdet.models.builder import DETECTORS
from mmdet.core import bbox2result

from .tb_edge_selector import EdgeQualityAttention, RetinaNAMEdgeSelector
from .tb_high_quality_selector import (HighQualitySelectorDebugHook,
                                       gaussian_soft_select)
from .tb_high_quality_selector import balanced_mean


class DetailResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels))

    def forward(self, value):
        return F.silu(value + self.body(value))


class NativeDetailEncoder(nn.Module):
    """One shared shallow pass over the native-resolution grayscale CXR."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            DetailResidualBlock(24), DetailResidualBlock(24))
        self.d2 = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32), nn.SiLU(), DetailResidualBlock(32))
        self.d4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(), DetailResidualBlock(64))

    def forward(self, gray):
        detail = self.stem(gray)
        d2 = self.d2(detail)
        return d2, self.d4(d2)


def edge_anchor_points(boxes):
    """Same ordering as edge_bias: edge, inner/outer, three along slots."""
    center = (boxes[:, :2] + boxes[:, 2:]) * .5
    size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-4)
    points = []
    for edge in range(4):
        axis, sign = edge % 2, (-1. if edge < 2 else 1.)
        for offset in (-.06, .06):
            for along in (-.30, 0., .30):
                relative = boxes.new_zeros((len(boxes), 2))
                relative[:, axis] = sign * (.5 + offset)
                relative[:, 1-axis] = along
                points.append(center + relative * size)
    return torch.stack(points, 1)


class DetailQualityAttention(EdgeQualityAttention):
    def __init__(self, *args, detail=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.detail = bool(detail)
        hidden = self.hidden
        with torch.random.fork_rng(devices=[]):
            self.detail_encoder = NativeDetailEncoder()
            self.detail_projection = nn.ModuleList([
                nn.Conv2d(32, hidden, 1), nn.Conv2d(64, hidden, 1)])
            self.detail_tokens = nn.Parameter(torch.randn(24, hidden) * .02)
            self.detail_position = nn.Linear(2, hidden)
            self.detail_q = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
            self.detail_k = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
            self.detail_v = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
            self.detail_out = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
            self.detail_fusion = nn.Sequential(
                nn.LayerNorm(hidden * 49), nn.Linear(hidden * 49, hidden * 2),
                nn.SiLU(), nn.Linear(hidden * 2, hidden))

    def prepare(self, features, gray, valid_shape, batch_shape):
        memories = super().prepare(features, gray, valid_shape, batch_shape)
        if self.detail:
            d2, d4 = self.detail_encoder(gray)
            memories.extend([(self.detail_projection[0](d2),),
                             (self.detail_projection[1](d4),)])
        return memories

    def _sample_edge_patches(self, feature, boxes):
        anchors = edge_anchor_points(boxes)
        size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-4)
        offsets = boxes.new_tensor([
            [-.025,-.025],[0.,-.025],[.025,-.025],
            [-.025,0.],[0.,0.],[.025,0.],
            [-.025,.025],[0.,.025],[.025,.025]])
        grid = anchors[:, :, None] + offsets[None, None] * size[:, None, None]
        grid = grid.mul(2.).sub(1.).reshape(len(boxes), 24*9, 1, 2)
        sampled = F.grid_sample(feature.expand(len(boxes), -1, -1, -1),
                                grid, mode='bilinear', padding_mode='border',
                                align_corners=False)
        return sampled[:, :, :, 0].transpose(1, 2).reshape(
            len(boxes), 24, 9, self.hidden)

    def _local_attention(self, descriptor, boxes, feature, level):
        memory = self._sample_edge_patches(feature, boxes)
        offsets=boxes.new_tensor([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]])
        memory=memory+self.detail_position(offsets)[None,None]
        query = descriptor[:, None] + self.detail_tokens[None]
        heads, dim = self.heads, self.hidden // self.heads
        q = self.detail_q[level](query).reshape(len(boxes), 24, heads, dim)
        k = self.detail_k[level](memory).reshape(len(boxes), 24, 9, heads, dim)
        v = self.detail_v[level](memory).reshape(len(boxes), 24, 9, heads, dim)
        attention = torch.einsum('nqhd,nqthd->nqht', q, k) / math.sqrt(dim)
        attention = attention.softmax(-1)
        selected = torch.einsum('nqht,nqthd->nqhd', attention, v).reshape(
            len(boxes), 24, self.hidden)
        return query + self.detail_out[level](selected)

    def _score(self, boxes, *inputs):
        descriptor, gate, entropy = super()._score(boxes, *(inputs[:-2] if self.detail else inputs))
        if not self.detail:
            return descriptor, gate, entropy
        d2, d4 = inputs[-2:]
        edge_d2 = self._local_attention(descriptor, boxes, d2, 0)
        edge_d4 = self._local_attention(descriptor, boxes, d4, 1)
        # Preserve edge and inner/outer identity; pool only three along-edge slots.
        # Preserve all along-edge slots through the final projection.
        fused = self.detail_fusion(torch.cat((descriptor, edge_d2.flatten(1),
                                               edge_d4.flatten(1)), 1))
        return descriptor + fused, gate, entropy


@DETECTORS.register_module()
class RetinaNAMDetailSelector(RetinaNAMEdgeSelector):
    def __init__(self, *args, selector_detail=False, selector_learned_gate=False,
                 selector_tb_fusion='max', selector_lse_temperature=.5,
                 selector_lse_trainable=False,
                 inference_mode='tb', subtype_topk=128,
                 selector_adaptive_subtype_gate=False,
                 selector_fusion_loss_weight=0.,
                 selector_high_iou_aux_weight=.2,
                 selector_quality_trainable=True,
                 selector_gate_trainable=True, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.candidate_quality
        with torch.random.fork_rng(devices=[]):
            self.candidate_quality = DetailQualityAttention(
                hidden=previous.hidden, heads=previous.heads,
                texture=previous.texture, gaussian=previous.gaussian,
                chunk=previous.chunk,
                checkpoint_candidates=previous.checkpoint_candidates,
                edges=previous.edges, detail=bool(selector_detail))
        self.selector_detail = bool(selector_detail)
        self.selector_tb_fusion = str(selector_tb_fusion)
        self.selector_lse_trainable = bool(selector_lse_trainable)
        if self.selector_tb_fusion not in ('max', 'lse'):
            raise ValueError('selector_tb_fusion must be max or lse')
        if not math.isfinite(float(selector_lse_temperature)) or selector_lse_temperature <= .02:
            raise ValueError('selector_lse_temperature must be finite and greater than .02')
        # Store the temperature inside candidate_quality so optimizer grouping
        # and the frozen-selector contract remain explicit.
        tau = float(selector_lse_temperature)
        raw_tau = math.log(math.expm1(max(tau - .02, 1e-4)))
        self.candidate_quality.tb_lse_temperature_raw = nn.Parameter(
            torch.tensor(raw_tau), requires_grad=self.selector_lse_trainable)
        self.inference_mode = str(inference_mode)
        self.subtype_topk = int(subtype_topk)
        if self.inference_mode not in ('tb', 'subtype'):
            raise ValueError('inference_mode must be tb or subtype')
        if self.subtype_topk <= 0:
            raise ValueError('subtype_topk must be positive')
        self.selector_quality_rank_only=True
        self.selector_learned_gate=bool(selector_learned_gate)
        self.selector_quality_trainable = bool(selector_quality_trainable)
        self.selector_gate_trainable = bool(selector_gate_trainable)
        for parameter in self.candidate_quality.parameters():
            parameter.requires_grad_(self.selector_quality_trainable)
        # Apply the temperature switch AFTER the general quality freeze flag.
        self.candidate_quality.tb_lse_temperature_raw.requires_grad_(
            self.selector_tb_fusion == 'lse' and self.selector_lse_trainable)
        self.selector_adaptive_subtype_gate = bool(selector_adaptive_subtype_gate)
        self.selector_fusion_loss_weight = float(selector_fusion_loss_weight)
        self.selector_high_iou_aux_weight = float(selector_high_iou_aux_weight)
        if not 0. <= self.selector_high_iou_aux_weight <= 1.:
            raise ValueError('selector_high_iou_aux_weight must be in [0,1]')
        if self.selector_fusion_loss_weight < 0:
            raise ValueError('selector_fusion_loss_weight must be nonnegative')
        if self.selector_tb_fusion == 'lse' and self.selector_adaptive_subtype_gate:
            raise ValueError('LSE and adaptive subtype gates are exclusive ablations')
        if (self.selector_tb_fusion == 'lse' and self.selector_lse_trainable
                and self.selector_fusion_loss_weight <= 0):
            raise ValueError('Learnable LSE requires selector_fusion_loss_weight > 0')
        # A small temperature would make this nearly-hard argmax and starve
        # the losing class gate of gradient.  T=1 keeps both subtype logits
        # learnable; A1 is a new scorer and is compared empirically with D3.
        self.selector_gate_temperature = 1.0
        # D3's learned scalar gate in the selected checkpoint.
        self.selector_gate_alpha_prior = .5202
        self.subtype_gate_stats = []
        self.subtype_gate_train_stats = []
        if self.selector_adaptive_subtype_gate:
            # Candidate-dependent gates choose Active vs Obsolete evidence;
            # a separate gate decides how much localization quality to use.
            # Zero output weights initialize alpha to D3's global value;
            # subtype mixing remains intentionally learnable from step one.
            with torch.random.fork_rng(devices=[]):
                self.candidate_quality.subtype_fusion_gate = nn.Sequential(
                    nn.LayerNorm(4), nn.Linear(4, 16), nn.SiLU(),
                    nn.Linear(16, 3))
            nn.init.zeros_(self.candidate_quality.subtype_fusion_gate[-1].weight)
            nn.init.zeros_(self.candidate_quality.subtype_fusion_gate[-1].bias)
            alpha_prior = (self.selector_gate_alpha_prior - .1) / .8
            nn.init.constant_(self.candidate_quality.subtype_fusion_gate[-1].bias[2],
                              math.log(alpha_prior / (1. - alpha_prior)))
        # Global scalar gate is intentionally modest for the small dataset.
        # Kept inside candidate_quality so freeze/hash/optimizer rules agree.
        self.candidate_quality.fusion_gate_logit=nn.Parameter(torch.zeros(()),
            requires_grad=(self.selector_learned_gate and
                           self.selector_gate_trainable))
        self.integrated_build_id = 'v59.1-quality-only-rank-optional-detached-gate'
        self.detail_debug = []

    def _score_candidates(self, boxes, memory, sym, training=False):
        outputs, gate, entropy = self.candidate_quality(
            boxes, memory, sym, training_checkpoint=training)
        # Both downstream slots intentionally carry the single high-quality
        # logit. The second is an interface slot, not a separate GT75/85 head.
        high = outputs[:, 0]
        return torch.stack((high, high), 1), gate, entropy

    def _final_score(self, sym, high_logit, residual):
        return self._final_logit(sym, high_logit, residual).sigmoid()

    def _final_logit(self, sym, high_logit, residual):
        subtype_logits = torch.logit(sym[:, :-1].clamp(1e-6, 1.-1e-6))
        if self.selector_adaptive_subtype_gate:
            gate_input = torch.cat((subtype_logits, high_logit[:, None],
                subtype_logits[:, :1] - subtype_logits[:, 1:2]), dim=1)
            gate_logits = self.candidate_quality.subtype_fusion_gate(gate_input)
            # Temperature-softmax is a smooth, learnable generalization of max:
            # as T approaches zero it selects the larger subtype logit.
            class_weights = torch.softmax(
                subtype_logits / self.selector_gate_temperature + gate_logits[:, :2],
                dim=1)
            subtype_score = (class_weights * subtype_logits).sum(1)
            quality_weight = .1 + .8 * torch.sigmoid(gate_logits[:, 2])
            if len(high_logit):
                # The outer detector stays in eval mode during frozen-detector
                # training. candidate_quality.training is the actual phase.
                if not self.candidate_quality.training:
                    stats = self.subtype_gate_stats
                elif torch.is_grad_enabled():
                    # Record the supervised pass, not the no-grad mining scan.
                    stats = self.subtype_gate_train_stats
                else:
                    stats = None
                if stats is not None:
                    stats.append(torch.stack((
                        class_weights[:, 0].detach().sum(),
                        class_weights[:, 1].detach().sum(),
                        quality_weight.detach().sum(),
                        high_logit.new_tensor(float(len(high_logit))))))
            return (1. - quality_weight) * subtype_score + quality_weight * high_logit
        if self.selector_tb_fusion == 'lse':
            tau = .02 + F.softplus(
                self.candidate_quality.tb_lse_temperature_raw)
            base = tau * torch.logsumexp(subtype_logits / tau, dim=1)
        else:
            base = subtype_logits.max(1)[0]
        if self.selector_learned_gate:
            alpha=.1+.8*self.candidate_quality.fusion_gate_logit.sigmoid()
            return (1-alpha)*base+alpha*high_logit
        return base + self.selector_beta * high_logit

    def _subtype_scores(self, sym, high_logit):
        """D3 quality fusion applied separately to each subtype logit."""
        logits = torch.logit(sym[:, :-1].clamp(1e-6, 1. - 1e-6))
        if self.selector_learned_gate:
            alpha = .1 + .8 * self.candidate_quality.fusion_gate_logit.sigmoid()
            return ((1. - alpha) * logits + alpha * high_logit[:, None]).sigmoid()
        return (logits + self.selector_beta * high_logit[:, None]).sigmoid()

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        if self.inference_mode == 'tb':
            return super().simple_test(img, img_metas, rescale=rescale, **kwargs)
        features, raw, _ = self._main(img, img_metas)
        originals = self._v330_unwrap_original_images(
            kwargs.get('original_img'), len(raw))
        results = []
        score_threshold = float(self.bbox_head.test_cfg.score_thr)
        max_per_img = int(self.bbox_head.test_cfg.max_per_img)
        for index, ((boxes, sym), meta, original) in enumerate(
                zip(raw, img_metas, originals)):
            # Preserve the trained D3 quality input domain. The max condition
            # is equivalent to "either subtype passes" at this threshold.
            keep = (sym[:, :-1] > score_threshold).any(1)
            boxes, sym = boxes[keep], sym[keep]
            memory = self._image_memory(features, img, meta, original, index)
            outputs, _, _ = self._score_candidates(
                self._normalized(boxes, meta), memory, sym)
            high = outputs[:, 0] if len(boxes) else boxes.new_empty((0,))
            scores = self._subtype_scores(sym, high)
            selected_ids, selected_scores, selected_labels = [], [], []
            for label in range(self.bbox_head.num_classes):
                eligible = (scores[:, label] > score_threshold).nonzero(
                    as_tuple=False).squeeze(1)
                if not len(eligible):
                    continue
                source, decayed = gaussian_soft_select(
                    boxes[eligible], scores[eligible, label],
                    self.subtype_topk, self.selector_sigma)
                selected_ids.append(eligible[source])
                selected_scores.append(decayed)
                selected_labels.append(torch.full_like(source, label))
            if selected_ids:
                ids = torch.cat(selected_ids)
                ranks = torch.cat(selected_scores)
                labels = torch.cat(selected_labels)
                order = ranks.argsort(descending=True)[:max_per_img]
                ids, ranks, labels = ids[order], ranks[order], labels[order]
                det = torch.cat((boxes[ids], ranks[:, None]), 1)
            else:
                labels = boxes.new_empty((0,), dtype=torch.long)
                det = boxes.new_empty((0, 5))
            if rescale and len(det):
                det[:, :4] /= det.new_tensor(meta['scale_factor'])
            results.append(bbox2result(
                det.cpu().numpy(), labels.cpu().numpy(),
                self.bbox_head.num_classes))
        return results

    def v50_begin_oracle(self, epoch):
        self.subtype_gate_stats = []
        if self.inference_mode == 'subtype':
            self.quality_collect = False
            self.quality_cache = []
            return
        return super().v50_begin_oracle(epoch)

    def _log_subtype_gates(self, logger, split):
        stats = (self.subtype_gate_train_stats if split == 'train'
                 else self.subtype_gate_stats)
        if stats:
            total = torch.stack(stats).sum(0).cpu().tolist()
            count = int(total[3])
            logger.info('[DETAIL-SUBTYPE-GATE] split=%s '
                'classWeightMean(active/obsolete)=%.4f/%.4f '
                'qualityWeightMean=%.4f candidateObservations=%d',
                split, total[0] / max(count, 1), total[1] / max(count, 1),
                total[2] / max(count, 1), count)
        stats.clear()

    def _residual_loss(self, base_z, residual, target, actual, matched):
        # The high-quality BCE and final-score rank already supervise this
        # shared logit. Do not add the saturated V58 residual objective.
        if not self.selector_learned_gate:
            return residual.sum() * 0.
        alpha=.1+.8*self.candidate_quality.fusion_gate_logit.sigmoid()
        # Fusion calibration cannot force the quality network to compensate
        # for Sym: only the gate receives this gradient.
        fused=(1-alpha)*base_z.detach()+alpha*residual.detach()
        loss=balanced_mean(F.binary_cross_entropy_with_logits(fused,
            target.detach(),reduction='none'),actual,matched)
        return .1*loss+.01*(alpha-.5).square()

    def _image_memory(self, features, img, meta, original, index):
        memory = super()._image_memory(features, img, meta, original, index)
        if self.selector_detail and len(memory) >= 8:
            self.detail_debug.append(dict(
                original=tuple(original.shape[-2:]),
                detector=tuple(img.shape[-2:]),
                d2=tuple(memory[-2][0].shape[-2:]),
                d4=tuple(memory[-1][0].shape[-2:])))
        return memory

    def v50_finish_oracle(self, dataset, runner):
        if self.selector_tb_fusion == 'lse':
            tau = float((.02 + F.softplus(
                self.candidate_quality.tb_lse_temperature_raw)).detach().cpu())
            runner.logger.info('[DETAIL-TB-FUSION] mode=lse temperature=%.5f '
                'trainable=%s', tau, self.selector_lse_trainable)
        else:
            runner.logger.info('[DETAIL-TB-FUSION] mode=max trainable=False')
        if self.inference_mode == 'subtype':
            runner.logger.info('[DETAIL-INFERENCE] mode=subtype '
                'qualityFusion=perClass softNMS=perClass topk=%d '
                'globalMaxDet=%d; mainDetectorFrozen=%s',
                self.subtype_topk, self.bbox_head.test_cfg.max_per_img,
                self.quality_detach)
            self.detail_debug = []
            return
        if self.selector_adaptive_subtype_gate and self.subtype_gate_stats:
            self._log_subtype_gates(runner.logger, 'val')
        elif self.selector_learned_gate:
            alpha=float((.1+.8*self.candidate_quality.fusion_gate_logit.sigmoid()).detach())
            runner.logger.info('[DETAIL-GATE] learned=%s qualityWeight=%.5f range=.1..9 '
                'gateLossDetachedFromQuality=True qualityRankOnly=True',
                self.selector_learned_gate,alpha)
        else:
            runner.logger.info('[DETAIL-GATE] learned=False qualityWeight=%.4f',
                self.selector_beta)
        self.subtype_gate_stats = []
        if self.selector_detail and self.detail_debug:
            runner.logger.info('[DETAIL-SHAPES] examples=%s normalizedCoordinates=True '
                'flipAppliedOnce=True', self.detail_debug[:3])
        self.detail_debug = []
        runner.logger.info('[DETAIL-SEMANTICS] detail=%s learnedGate=%s '
            'noTanh=True noSplit75_85=True',self.selector_detail,self.selector_learned_gate)
        return super().v50_finish_oracle(dataset, runner)


@HOOKS.register_module()
class DetailSelectorDebugHook(HighQualitySelectorDebugHook):
    def before_train_epoch(self, runner):
        super().before_train_epoch(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.subtype_gate_train_stats = []
        self.lse_grad_stats = None

    def after_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        temperature = model.candidate_quality.tb_lse_temperature_raw
        if temperature.requires_grad and temperature.grad is not None:
            g = temperature.grad.detach().float()
            row = torch.stack((g, g.abs(), g.new_tensor(1.)))
            if self.lse_grad_stats is None:
                self.lse_grad_stats = row.clone()
            else:
                self.lse_grad_stats.add_(row)

    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        runner.logger.info('[DETAIL-BUILD] nativeDetail=%s D2D4=True '
            'wholeImageEncodeOnce=True localEdgeQKV=True scoreUsesHighHead=True '
            'adaptiveSubtypeGate=%s fusionLossWeight=%.3f qualityTarget=%s '
            'fusionQualityDetached=True qualityTrainable=%s '
            'rankScope=qualityOnly targetTemperature=%.4f',
            model.selector_detail, model.selector_adaptive_subtype_gate,
            model.selector_fusion_loss_weight,
            model.selector_quality_target, model.selector_quality_trainable,
            model.selector_quality_target_temperature)

    def after_train_epoch(self, runner):
        super().after_train_epoch(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model._log_subtype_gates(runner.logger, 'train')
        if model.selector_tb_fusion == 'lse':
            tau = float((.02 + F.softplus(
                model.candidate_quality.tb_lse_temperature_raw)).detach().cpu())
            stats = (self.lse_grad_stats.cpu().tolist()
                     if self.lse_grad_stats is not None else [0., 0., 0.])
            runner.logger.info('[LSE-TRAIN] epoch=%d temperature=%.7f '
                'gradSteps=%d meanGrad=%.7e meanAbsGrad=%.7e '
                'fusionBCEQualityDetached=True auxMix=%.3f', runner.epoch + 1,
                tau, int(stats[2]), stats[0] / max(stats[2], 1),
                stats[1] / max(stats[2], 1), model.selector_high_iou_aux_weight)
        runner.logger.info('[DETAIL-SCORE] source=singleHighQualityLogit '
            'adaptiveSubtypeGate=%s fusionLossWeight=%.3f target=%s',
            model.selector_adaptive_subtype_gate,
            model.selector_fusion_loss_weight,
            model.selector_quality_target)
