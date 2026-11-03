"""Frozen V55 visual evaluator trained to select high-quality candidates.

No PQA, box refinement, or candidate deletion before one Gaussian Soft-NMS.
"""
import torch
import os.path as osp
import mmcv
from torch.nn import functional as F
from mmdet.core import bbox_overlaps, bbox2result
from mmdet.models.builder import DETECTORS
from .candidates import RetinaNAMCandidateQuality, coverage, bins, oracle_result
from .attention import pairwise_box_iou
from mmcv.runner import HOOKS, Hook
from .candidates import main_hash


def high_quality_target(iou, temperature=.03):
    return .5 * (torch.sigmoid((iou - .75) / temperature) +
                 torch.sigmoid((iou - .85) / temperature))


def all_iou_target(iou, temperature=.03):
    """Smooth fraction of COCO IoU thresholds passed by a candidate."""
    thresholds = iou.new_tensor([.50 + .05 * index for index in range(10)])
    return torch.sigmoid(
        (iou[..., None] - thresholds) / float(temperature)).mean(-1)


def bounded_score(sym, residual, beta):
    return torch.sigmoid(torch.logit(sym.clamp(1e-6, 1.-1e-6)) +
                         beta * residual.tanh())


def balanced_mean(element, actual, matched):
    """IoU<.5 is a separate low-quality group, not assigned to GT zero."""
    terms = []
    low = actual < .5
    if low.any():
        terms.append(element[low].mean())
    for g in matched[~low].unique():
        terms.append(element[(matched == g) & ~low].mean())
    return torch.stack(terms).mean() if terms else element.sum() * 0.


def gaussian_soft_select(boxes, scores, topk=128, sigma=.7):
    """GPU-resident dynamic Soft-NMS; scores remain associated with source IDs."""
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1 or len(boxes) != len(scores):
        raise ValueError('Soft-NMS expects boxes[N,4] and scores[N] with matching lengths')
    active = torch.arange(len(boxes), device=boxes.device)
    if not len(active) or topk <= 0:
        return active[:0], scores[:0]
    dynamic = scores.clone()
    selected, values = [], []
    for _ in range(min(topk, len(boxes))):
        position = dynamic[active].argmax()
        winner = active[position]
        selected.append(winner)
        values.append(dynamic[winner])
        active = active[torch.arange(len(active), device=active.device) != position]
        if not len(active):
            break
        overlap = bbox_overlaps(boxes[winner].reshape(1, 4), boxes[active])[0]
        dynamic[active] *= torch.exp(-overlap.square() / sigma)
    if not selected:
        return active, scores[:0]
    return torch.stack(selected), torch.stack(values)


def hard_select(boxes, scores, topk=128, iou_threshold=.5):
    """Class-agnostic hard NMS over exactly the same eligible candidates."""
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1 or len(boxes) != len(scores):
        raise ValueError('Hard-NMS expects boxes[N,4] and scores[N] with matching lengths')
    active = torch.arange(len(boxes), device=boxes.device)
    if not len(active) or topk <= 0:
        return active[:0], scores[:0]
    selected = []
    for _ in range(min(topk, len(boxes))):
        # Suppression in the preceding iteration can empty the active set.
        # The iteration budget is based on the ORIGINAL number of candidates.
        if not len(active):
            break
        winner = active[scores[active].argmax()]
        selected.append(winner)
        active = active[active != winner]
        if not len(active):
            break
        overlap = bbox_overlaps(boxes[winner].reshape(1, 4), boxes[active])[0]
        active = active[overlap <= iou_threshold]
    ids = torch.stack(selected) if selected else active
    return ids, scores[ids]


class RetinaNAMHighQualitySelector(RetinaNAMCandidateQuality):
    def _score_candidates(self, boxes, memory, sym, training=False):
        return self.candidate_quality(boxes, memory, training_checkpoint=training)

    def _training_quality_predictions(self, boxes, memory, sym):
        outputs, gate, entropy = self._score_candidates(boxes, memory, sym, True)
        return outputs, gate, entropy, None

    def _quality_element_loss(self, logit, target, actual, supervision):
        # Historical scalar BCE; optional distribution experiments override
        # this hook without duplicating sampling, rank or the training loop.
        return F.binary_cross_entropy_with_logits(logit, target, reduction='none')

    def _additional_supervised_loss(self, boxes, sym, gt, matched, actual, memory, meta, supervision):
        return self._extra_training_loss(boxes, sym, gt, matched, actual, memory, meta)

    def _replacement_rank_loss(self, logits, actual, matched, boxes):
        # None selects the unchanged historical pair-rank implementation.
        return None

    def _extra_training_loss(self, boxes, sym, gt, matched, actual, memory, meta):
        return self.candidate_quality.quality[-1].weight.sum() * 0.

    def _full_candidate_training_loss(self, boxes, sym, gt, matched, actual,
                                      memory, meta, scan=None,
                                      corrected_scan=None):
        return self.candidate_quality.quality[-1].weight.sum() * 0.

    def _final_score(self, sym, high_logit, residual):
        return bounded_score(sym[:, :-1].max(1)[0], residual,
                             self.selector_beta)

    def _residual_loss(self, base_z, residual, target, actual, matched):
        desired = (torch.logit(target.clamp(.02, .98)) - base_z).clamp(
            -.8 * self.selector_beta, .8 * self.selector_beta).detach()
        return balanced_mean(F.smooth_l1_loss(self.selector_beta * residual.tanh(),
            desired, reduction='none'), actual, matched)

    def __init__(self, *args, selector_beta=1., selector_temperature=.03,
                 selector_rank_weight=0., selector_sigma=.7,
                 selector_quality_target='high_75_85',
                 selector_quality_target_temperature=None, **kwargs):
        kwargs.update(quality_detach=True, quality_score_mode='sym')
        super().__init__(*args, **kwargs)
        if selector_beta <= 0 or selector_temperature <= 0 or selector_sigma <= 0:
            raise ValueError('Selector beta/temperature/sigma must be positive')
        self.selector_beta = float(selector_beta)
        self.selector_temperature = float(selector_temperature)
        self.selector_rank_weight = float(selector_rank_weight)
        self.selector_sigma = float(selector_sigma)
        self.selector_quality_target = str(selector_quality_target)
        self.selector_quality_target_temperature = float(
            selector_temperature if selector_quality_target_temperature is None
            else selector_quality_target_temperature)
        if self.selector_quality_target_temperature <= 0:
            raise ValueError('selector_quality_target_temperature must be positive')
        if self.selector_quality_target not in (
                'high_75_85', 'direct_iou', 'coco_all_iou',
                'coco_all_iou_mix'):
            raise ValueError('Unknown selector_quality_target')
        old = self.candidate_quality.quality[-1]
        with torch.random.fork_rng(devices=[]):
            self.candidate_quality.quality[-1] = torch.nn.Linear(old.in_features, 2)
        torch.nn.init.zeros_(self.candidate_quality.quality[-1].weight)
        torch.nn.init.zeros_(self.candidate_quality.quality[-1].bias)
        self.integrated_build_id = 'v57.1-separated-quality-residual-aligned-mining'
        self.selector_train_stats = []

    def _quality_target(self, iou):
        if self.selector_quality_target == 'direct_iou':
            return iou.clamp(0., 1.)
        if self.selector_quality_target == 'coco_all_iou':
            return all_iou_target(iou, self.selector_quality_target_temperature)
        if self.selector_quality_target == 'coco_all_iou_mix':
            all_target = all_iou_target(iou, self.selector_quality_target_temperature)
            high_target = high_quality_target(iou, self.selector_temperature)
            weight = float(getattr(self, 'selector_high_iou_aux_weight', .2))
            return (1. - weight) * all_target + weight * high_target
        return high_quality_target(iou, self.selector_quality_target_temperature)

    def _trainable_zero(self):
        # In gate-only training the old quality head is frozen. Connect an
        # empty-batch loss to every trainable parameter, including the gate.
        terms = [p.reshape(-1)[0] * 0. for p in self.candidate_quality.parameters()
                 if p.requires_grad and p.numel()]
        if not terms:
            raise RuntimeError('Selector training has no trainable parameters')
        return torch.stack(terms).sum()

    def _selector_sample(self, actual, matched, target, prediction, corrected, boxes):
        selected = []
        for mask in (actual < .5, (actual >= .5) & (actual < .75), actual >= .75):
            ids = mask.nonzero(as_tuple=False).squeeze(1)
            n = min(len(ids), self.quality_samples_per_bin)
            if not n:
                continue
            # Errors use the SAME target as the classification objective.
            error = (target[ids] - prediction[ids]).abs()
            hard_n = int(n * .75)
            hard = ids[error.topk(hard_n).indices]
            available = mask.clone(); available[hard] = False
            rest = available.nonzero(as_tuple=False).squeeze(1)
            selected.extend((hard, rest[torch.randperm(len(rest), device=rest.device)[:n-hard_n]]))
        for g in matched[actual >= .5].unique():
            ids = (matched == g).nonzero(as_tuple=False).squeeze(1)
            best = ids[actual[ids].argmax()]
            selected.append(best.reshape(1))
            # Mine competitors from ALL candidates, not only the BCE sample.
            overlap = bbox_overlaps(boxes[best].reshape(1, 4), boxes)[0]
            valid = ((matched == g) | (actual < .5)) & (overlap > .5)
            valid &= actual < actual[best] - .05
            rivals = valid.nonzero(as_tuple=False).squeeze(1)
            selected.append(rivals[corrected[rivals].topk(min(32, len(rivals))).indices])
        return torch.cat(selected).unique() if selected else matched[:0]

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, **kwargs):
        if gt_bboxes_ignore is not None and any(len(x) for x in gt_bboxes_ignore):
            raise ValueError('Ignored GT requires explicit selector supervision')
        features, raw, _ = self._main(img, img_metas)
        originals = self._v330_unwrap_original_images(kwargs.get('original_img'), len(raw))
        classification, ranking, residual_losses, extra_losses = [], [], [], []
        fusion_losses = []
        full_candidate_losses = []
        counts = dict(low=0, positive=0, pairs=0, unreachable=0)
        for index, ((boxes, sym), gt, meta, original) in enumerate(
                zip(raw, gt_bboxes, img_metas, originals)):
            keep = (sym[:, :-1] > self.bbox_head.test_cfg.score_thr).any(1)
            boxes, sym = boxes[keep].detach(), sym[keep].detach()
            if not len(boxes):
                continue
            memory = self._image_memory(features, img, meta, original, index)
            normalized = self._normalized(boxes, meta)
            with torch.no_grad():
                if len(gt):
                    actual, matched = bbox_overlaps(boxes, gt).max(1)
                else:
                    actual = boxes.new_zeros(len(boxes))
                    matched = torch.full((len(boxes),), -1, device=boxes.device, dtype=torch.long)
                scan = self._score_candidates(normalized, memory, sym)[0]
                all_target = self._quality_target(actual)
                corrected_scan = self._final_score(sym, scan[:, 0], scan[:, 1])
                ids = self._selector_sample(actual, matched, all_target,
                    scan[:, 0].sigmoid(), corrected_scan, boxes)
            full_candidate_losses.append(self._full_candidate_training_loss(
                boxes, sym, gt, matched, actual, memory, meta,
                scan.detach(), corrected_scan.detach()))
            outputs, _, _, supervision = self._training_quality_predictions(
                normalized[ids], memory, sym[ids])
            extra_losses.append(self._additional_supervised_loss(boxes[ids], sym[ids], gt,
                matched[ids], actual[ids], memory, meta, supervision))
            z, residual = outputs.unbind(1)
            target = self._quality_target(actual[ids])
            element = self._quality_element_loss(z, target, actual[ids], supervision)
            classification.append(balanced_mean(element, actual[ids], matched[ids]))
            base_z = torch.logit(sym[ids, :-1].max(1)[0].clamp(1e-6, 1.-1e-6))
            # Explicit residual supervision; no BCE-driven saturation of the
            # correction output. Interior targets leave headroom for ranking.
            residual_losses.append(self._residual_loss(base_z, residual, target,
                                                       actual[ids], matched[ids]))
            fusion_loss_weight = float(getattr(
                self, 'selector_fusion_loss_weight', 0.))
            if fusion_loss_weight > 0:
                # Same detached quality input in A1 and A3: fusion supervision
                # updates gates only. Quality retains its own BCE/rank losses.
                if hasattr(self, '_final_logit'):
                    final_logit = self._final_logit(
                        sym[ids].detach(), z.detach(), residual.detach())
                else:
                    final_probability = self._final_score(
                        sym[ids].detach(), z.detach(), residual.detach())
                    final_logit = torch.logit(final_probability.clamp(1e-6, 1.-1e-6))
                fusion_losses.append(balanced_mean(
                    F.binary_cross_entropy_with_logits(
                        final_logit, target, reduction='none'),
                    actual[ids], matched[ids]))
            counts['low'] += int((actual[ids] < .5).sum())
            counts['positive'] += int((actual[ids] >= .5).sum())
            replacement_rank = self._replacement_rank_loss(z, actual[ids], matched[ids], normalized[ids])
            if self.selector_rank_weight > 0 and replacement_rank is not None:
                ranking.append(replacement_rank[0])
                counts['pairs'] += replacement_rank[1]
            elif self.selector_rank_weight > 0:
                quality_only = getattr(self,'selector_quality_rank_only',False)
                final_z = z if quality_only else torch.logit(
                    self._final_score(sym[ids],z,residual).clamp(1e-6,1.-1e-6))
                overlap = pairwise_box_iou(normalized[ids])
                per_gt = []
                for g in matched[ids][actual[ids] >= .5].unique():
                    same = matched[ids] == g
                    # Compare best available candidates against the most
                    # dangerous current competitors, using final ranking score.
                    eligible = same & (actual[ids] >= .5)
                    winner = (actual[ids].masked_fill(~eligible, -1.)).argmax()
                    if getattr(self,'selector_list_rank',False):
                        qualified = eligible & (actual[ids] >= actual[ids][winner]-.02)
                        good_ids=qualified.nonzero(as_tuple=False).squeeze(1)
                        winner=good_ids[final_z[good_ids].argmax()]
                    valid = (same | (actual[ids] < .5)) & (actual[ids] < actual[ids][winner] - .05)
                    valid &= overlap[winner] > .5
                    rivals = valid.nonzero(as_tuple=False).squeeze(1)
                    if len(rivals):
                        rivals = rivals[final_z[rivals].topk(min(32, len(rivals))).indices]
                        gap = actual[ids][winner] - actual[ids][rivals]
                        counts['pairs'] += len(rivals)
                        if not quality_only:
                            counts['unreachable'] += int(((base_z[rivals] - base_z[winner]) >=
                                                         2 * self.selector_beta).sum())
                            feasible = (base_z[rivals] - base_z[winner] + .5 * gap) < 2 * self.selector_beta
                            rivals, gap = rivals[feasible], gap[feasible]
                        if not len(rivals):
                            continue
                        per_gt.append(F.softplus(.5 * gap -
                            (final_z[winner] - final_z[rivals])).mean())
                if per_gt:
                    ranking.append(torch.stack(per_gt).mean())
        zero = (self.candidate_quality.quality[-1].weight.sum() * 0.
                if classification else self._trainable_zero())
        self.selector_train_stats.append(counts)
        return dict(loss_selector_high_quality=torch.stack(classification).mean()
                    if classification else zero,
                    loss_selector_residual=torch.stack(residual_losses).mean()
                    if residual_losses else zero,
                    loss_selector_fusion=float(getattr(
                        self, 'selector_fusion_loss_weight', 0.)) *
                    (torch.stack(fusion_losses).mean() if fusion_losses else zero),
                    loss_selector_competition=self.selector_rank_weight *
                    (torch.stack(ranking).mean() if ranking else zero),
                    loss_selector_edge_pair=torch.stack(extra_losses).mean()
                    if extra_losses else zero,
                    loss_selector_head_competition=torch.stack(
                        full_candidate_losses).mean() if full_candidate_losses else zero)

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        features, raw, _ = self._main(img, img_metas)
        originals = self._v330_unwrap_original_images(kwargs.get('original_img'), len(raw))
        results = []
        for index, ((boxes, sym), meta, original) in enumerate(zip(raw, img_metas, originals)):
            raw_count = len(boxes)
            keep = (sym[:, :-1] > self.bbox_head.test_cfg.score_thr).any(1)
            boxes, sym = boxes[keep], sym[keep]
            memory = self._image_memory(features, img, meta, original, index)
            outputs, gate, entropy = self._score_candidates(self._normalized(boxes, meta), memory, sym)
            if len(boxes):
                z, residual = outputs.unbind(1)
            else:
                z = residual = boxes.new_empty((0,))
            original_score = sym[:, :-1].max(1)[0]
            corrected = self._final_score(sym, z, residual)
            soft_source, soft_score = gaussian_soft_select(boxes, corrected, 128, self.selector_sigma)
            source, decayed = soft_source, soft_score
            source, decayed = source[:self.bbox_head.test_cfg.max_per_img], decayed[:self.bbox_head.test_cfg.max_per_img]
            labels = sym[:, :-1].argmax(1)[source]
            det = torch.cat((boxes[source], decayed[:, None]), 1)
            if self.quality_collect:
                if meta.get('flip', False):
                    raise ValueError('Selector diagnostics require unflipped validation')
                self.quality_cache.append(dict(boxes=(boxes / boxes.new_tensor(meta['scale_factor'])).cpu(),
                    sym=sym.cpu(), quality=z.sigmoid().cpu(), logit=z.cpu(),
                    gate=gate.cpu(), entropy=entropy.cpu(), source=source.cpu(),
                    labels=labels.cpu(), score=decayed.cpu(), raw_count=raw_count,
                    corrected=corrected.cpu(), soft_source=soft_source.cpu(),
                    residual=residual.cpu(),
                    soft_score=soft_score.cpu(),
                    filename=meta.get('filename')))
            if rescale:
                det[:, :4] /= det.new_tensor(meta['scale_factor'])
            results.append(bbox2result(det.cpu().numpy(), labels.cpu().numpy(), self.bbox_head.num_classes))
        return results

    def v50_finish_oracle(self, dataset, runner):
        runner.logger.info('[SELECTOR-SEMANTICS] qualityTarget=%s '
            'directIoU=%s beta=%.2f rankWeight=%.3f sigma=%.2f '
            'qualityRankOnly=%s frozen=True', self.selector_quality_target,
            self.selector_quality_target == 'direct_iou', self.selector_beta,
            self.selector_rank_weight, self.selector_sigma,
            getattr(self,'selector_quality_rank_only',False))
        summaries, exported, hard = {}, [], []
        oracle = {name: [] for name in ('pre', 'soft128', 'final')}
        regret = {'sym': [], 'corrected': [], 'sym_all': [], 'corrected_all': []}
        traces = []
        complement = {t: dict(both=0,sym_only=0,network_only=0,corrected_only=0,
                              corrected_lost=0)
                      for t in (.5,.75,.85)}
        shared_candidates = candidate_budget = 0
        bound_pairs = bound_blocked = 0
        for image_index, item in enumerate(self.quality_cache):
            boxes = item['boxes']
            gt = torch.as_tensor(dataset.get_ann_info(image_index)['bboxes'],
                                 dtype=torch.float32).reshape(-1, 4)
            if len(gt) and len(boxes):
                actual, matched = bbox_overlaps(boxes, gt).max(1)
            else:
                actual = torch.zeros(len(boxes))
                matched = torch.full((len(boxes),), -1, dtype=torch.long)
            sym = item['sym'][:, :-1].max(1)[0]
            full_overlap = bbox_overlaps(boxes, gt)
            stages = dict(pre=torch.arange(len(boxes)),
                sym128=sym.topk(min(128, len(boxes))).indices,
                network128=item['quality'].topk(min(128,len(boxes))).indices,
                corrected128=item['corrected'].topk(min(128, len(boxes))).indices,
                soft128=item['soft_source'], final=item['source'])
            shared_candidates += len(set(stages['sym128'].tolist()) &
                                     set(stages['network128'].tolist()))
            candidate_budget += len(stages['sym128'])
            for threshold in complement:
                sets = {}
                for name in ('sym128','network128','corrected128'):
                    ids = stages[name]
                    sets[name] = set((full_overlap[ids] >= threshold).any(0).nonzero(
                        as_tuple=False).squeeze(1).tolist())
                a,b,c = sets['sym128'],sets['network128'],sets['corrected128']
                complement[threshold]['both'] += len(a&b)
                complement[threshold]['sym_only'] += len(a-b)
                complement[threshold]['network_only'] += len(b-a)
                complement[threshold]['corrected_only'] += len(c-a)
                complement[threshold]['corrected_lost'] += len(a-c)
            for stage, ids in stages.items():
                st = summaries.setdefault(stage, dict(count=0, bins=[0]*5, gt=[0]*3))
                st['count'] += len(ids)
                st['bins'] = [a+b for a,b in zip(st['bins'], bins(actual[ids]))]
                st['gt'] = [a+b for a,b in zip(st['gt'], coverage(boxes[ids], gt))]
                if stage in oracle and (self.quality_epoch % getattr(self, 'oracle_interval', 4) == 0
                                        or getattr(self, 'force_oracle', False)):
                    oracle[stage].append(oracle_result(boxes[ids], item['sym'][ids], gt,
                                                       self.bbox_head.num_classes))
            for g in matched[actual >= .5].unique():
                ids = ((matched == g) & (actual >= .5)).nonzero(as_tuple=False).squeeze(1)
                best = ids[actual[ids].argmax()]
                related = (matched == g) | (bbox_overlaps(
                    boxes[best].reshape(1, 4), boxes)[0] > .5)
                all_ids = related.nonzero(as_tuple=False).squeeze(1)
                for name, score in (('sym', sym), ('corrected', item['corrected'])):
                    chosen = ids[score[ids].argmax()]
                    regret[name].append(float(actual[ids].max() - actual[chosen]))
                    all_chosen = all_ids[score[all_ids].argmax()]
                    # Evaluate relative to THIS GT, not the competitor's own GT.
                    relative = bbox_overlaps(boxes[all_chosen].reshape(1,4), gt[g].reshape(1,4))[0,0]
                    regret[name+'_all'].append(float(actual[best] - relative))
                    if name == 'corrected' and actual[ids].max() >= .85 and actual[chosen] < .85:
                        best = ids[actual[ids].argmax()]
                        hard.append(dict(image_index=image_index, filename=item['filename'],
                            best_bbox=boxes[best].tolist(), chosen_bbox=boxes[chosen].tolist(),
                            best_iou=float(actual[best]), chosen_iou=float(actual[chosen]),
                            best_sym=float(sym[best]), chosen_sym=float(sym[chosen]),
                            best_corrected=float(score[best]), chosen_corrected=float(score[chosen])))
                base_z = torch.logit(sym.clamp(1e-6, 1.-1e-6))
                rivals = all_ids[actual[all_ids] < actual[best] - .05]
                bound_pairs += len(rivals)
                if not getattr(self,'selector_quality_rank_only',False):
                    bound_blocked += int((base_z[rivals]-base_z[best] >= 2*self.selector_beta).sum())
                good = ids[actual[ids] >= .85]
                retained = set(item['soft_source'].tolist())
                if len(good) and not any(int(i) in retained for i in good):
                    selected = item['soft_source']
                    contribution = bbox_overlaps(boxes[best].reshape(1,4), boxes[selected])[0].square()/self.selector_sigma
                    top = contribution.topk(min(5,len(contribution))).indices
                    traces.append(dict(image_index=image_index, gt_index=int(g),
                        best_bbox=boxes[best].tolist(), best_iou=float(actual[best]),
                        initial_score=float(item['corrected'][best]),
                        cumulative_log_decay=float(contribution.sum()),
                        final_score=float(item['corrected'][best]*torch.exp(-contribution.sum())),
                        suppressors=[dict(step=int(k), bbox=boxes[selected[k]].tolist(),
                            true_iou=float(actual[selected[k]]),
                            matched_gt=int(matched[selected[k]]),
                            same_gt=bool(matched[selected[k]]==g),
                            log_decay=float(contribution[k])) for k in top.tolist()]))
            exported.append(dict(image_index=image_index, filename=item['filename'],
                boxes=boxes.tolist(), true_iou=actual.tolist(), matched_gt=matched.tolist(),
                sym=sym.tolist(), high_probability=item['quality'].tolist(),
                corrected_score=item['corrected'].tolist(),
                residual_logit=item['residual'].tolist(),
                soft_source=item['soft_source'].tolist(), soft_score=item['soft_score'].tolist()))
        for stage, st in summaries.items():
            runner.logger.info('[SELECTOR-STAGE] epoch=%d stage=%s count=%d bins=%s GT50/75/85=%s',
                self.quality_epoch, stage, st['count'], st['bins'], st['gt'])
        runner.logger.info('[SELECTOR-REGRET] sym=%.5f corrected=%.5f missedBest85=%d',
            sum(regret['sym'])/max(len(regret['sym']),1),
            sum(regret['corrected'])/max(len(regret['corrected']),1), len(hard))
        runner.logger.info('[SELECTOR-COMPLEMENT] candidateOverlap=%.5f '
            'GTthresholds=%s counts=%s (network=highProbability,notIoU)',
            shared_candidates/max(candidate_budget,1),list(complement),complement)
        runner.logger.info('[SELECTOR-ALL-COMPETITORS] regretSym=%.5f regretCorrected=%.5f '
            'boundBlocked=%d/%d lostGT85Soft=%d',
            sum(regret['sym_all'])/max(len(regret['sym_all']),1),
            sum(regret['corrected_all'])/max(len(regret['corrected_all']),1),
            bound_blocked, bound_pairs, len(traces))
        for name, rows in (('candidates', exported), ('hard_samples', hard),
                           ('soft_traces', traces), ('stages', summaries),
                           ('complementarity',complement)):
            path = osp.join(runner.work_dir, 'selector_%s_epoch_%02d.json' % (name, self.quality_epoch))
            mmcv.dump(rows, path)
            runner.logger.info('[SELECTOR-EXPORT] %s', path)
        if self.quality_epoch % getattr(self, 'oracle_interval', 4) == 0 or getattr(self, 'force_oracle', False):
            for stage, rows in oracle.items():
                result = dataset.evaluate(rows, metric='bbox', tb_official_agnostic=True,
                                           tb_class_agnostic=False)
                runner.logger.info('[SELECTOR-ORACLE] stage=%s AP/AP50/AP75=%.3f/%.3f/%.3f',
                    stage, result['official_tb_bbox_mAP'], result['official_tb_bbox_mAP_50'],
                    result['official_tb_bbox_mAP_75'])
        self.quality_cache = []
        self.quality_collect = False


class HighQualitySelectorDebugHook(Hook):
    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.initial_hash = main_hash(model)
        runner.logger.info('[SELECTOR-BUILD] id=%s frozen=True qualityTarget=%s '
            'beta=%.2f rank=%.3f softSigma=%.2f zeroResidualInit=True mainSHA256=%s',
            model.integrated_build_id, model.selector_quality_target, model.selector_beta,
            model.selector_rank_weight, model.selector_sigma, self.initial_hash)

    def before_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model._nam_epoch = runner.epoch

    def after_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        current = main_hash(model)
        runner.logger.info('[SELECTOR-MAIN] mainChanged=%s mainSHA256=%s',
                           current != self.initial_hash, current)
        if current != self.initial_hash:
            raise RuntimeError('Frozen detector changed during selector training')
        totals = {key: sum(x[key] for x in model.selector_train_stats)
                  for key in ('low','positive','pairs','unreachable')}
        runner.logger.info('[SELECTOR-TRAIN] low/positive=%d/%d rankPairs=%d '
            'boundUnreachable=%d targetAlignedMining=True alphaCalibrationSeparate=True '
            'activeRankLoss=%s',
            totals['low'],totals['positive'],totals['pairs'],totals['unreachable'],
            getattr(model,'active_rank_name','loss_selector_competition'))
        model.selector_train_stats = []
        if hasattr(model,'edge_stats'):
            count=sum(x['count'] for x in model.edge_stats)
            correct=sum(x['correct'] for x in model.edge_stats)
            directions=[sum(x['directions'][k] for x in model.edge_stats) for k in range(4)]
            correct_directions=[sum(x.get('correct_directions',[0]*4)[k] for x in model.edge_stats) for k in range(4)]
            runner.logger.info('[SELECTOR-EDGE] validPairs=%d directionCounts=%s '
                'pairOrder=%.5f edges=%s symExplicit=True edgeWeight=%.3f',
                count,directions,correct/max(count,1),model.candidate_quality.edges,
                model.selector_edge_weight)
            runner.logger.info('[SELECTOR-EDGE-DIRECTIONS] left/top/right/bottom=%s',
                [a/max(b,1) for a,b in zip(correct_directions,directions)])
            runner.logger.info('[SELECTOR-EDGE-ACCEPT] attempted/accepted=%d/%d '
                'GT85parents attempted/accepted=%d/%d',
                sum(x.get('attempted',0) for x in model.edge_stats),count,
                sum(x.get('attempted85',0) for x in model.edge_stats),
                sum(x.get('accepted85',0) for x in model.edge_stats))
            model.edge_stats=[]
        if hasattr(model,'residual_stats'):
            values=[sum(x[k] for x in model.residual_stats) for k in range(4)]
            runner.logger.info('[SELECTOR-RESIDUAL-TARGET] loss=finalLogitContinuousIoUBCE '
                'unreachable/all=%d/%d unreachable85/all85=%d/%d noClippedTarget=True',*values)
            model.residual_stats=[]
