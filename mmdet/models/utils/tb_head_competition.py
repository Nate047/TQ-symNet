"""V60 complete-candidate mining and per-GT head competition loss."""
import torch
from torch.nn import functional as F
from mmcv.runner import HOOKS

from mmdet.core import bbox_overlaps
from mmdet.models.builder import DETECTORS

from .tb_detail_selector import RetinaNAMDetailSelector, DetailSelectorDebugHook
from .tb_high_quality_selector import gaussian_soft_select


def preceding_suppressors(boxes, good, bad_mask, order, sigma):
    """Actual log-decay contributions BEFORE the first acceptable selection.

    If no acceptable candidate survived the 128-step replay, use all steps.
    No later selected box is called a suppressor of an already selected winner.
    """
    hits = (order[:, None] == good[None]).any(1)
    positions = hits.nonzero(as_tuple=False).squeeze(1)
    stop = int(positions[0]) if len(positions) else len(order)
    earlier = order[:stop]
    earlier = earlier[bad_mask[earlier]]
    if not len(earlier):
        return earlier, boxes.new_empty((0,))
    decay = bbox_overlaps(boxes[good], boxes[earlier]).square() / sigma
    return earlier, decay.mean(0)


def winner_loss(good_logits, bad_logits, margins):
    # A lower-scoring acceptable winner must not penalize a successful one.
    # Every mined rival participates, including low-q/high-Sym suppressors.
    return F.softplus(margins + bad_logits - good_logits.max()).mean()


@DETECTORS.register_module()
class RetinaNAMHeadCompetition(RetinaNAMDetailSelector):
    def __init__(self, *args, head_competition_weight=.10,
                 head_competition_topk=4, head_competition_temperature=.10,
                 **kwargs):
        # Disable the historical sampled pair rank. E1 has one ranking loss.
        kwargs['selector_rank_weight'] = 0.
        kwargs['selector_edge_weight'] = 0.
        super().__init__(*args, **kwargs)
        self.head_competition_weight = float(head_competition_weight)
        self.head_competition_topk = int(head_competition_topk)
        self.head_competition_temperature = float(head_competition_temperature)
        if (self.head_competition_weight < 0 or self.head_competition_topk < 1 or
                self.head_competition_temperature <= 0):
            raise ValueError('Invalid head-competition configuration')
        self.head_competition_stats = []
        self.integrated_build_id = 'v60.1-actual-prefix-suppressors-protected-rivals'

    @staticmethod
    def _top(ids, score, count):
        if not len(ids):
            return ids
        return ids[score[ids].topk(min(count, len(ids))).indices]

    def _full_candidate_training_loss(self, boxes, sym, gt, matched, actual,
                                      memory, meta, scan=None,
                                      corrected_scan=None):
        zero = self.candidate_quality.high_head.weight.sum() * 0.
        if (self.head_competition_weight <= 0 or not len(gt) or
                not len(boxes)):
            return zero
        quality_scan = scan[:, 0]
        sym_score = sym[:, :-1].max(1)[0]
        soft_source, _ = gaussian_soft_select(
            boxes, corrected_scan, 128, self.selector_sigma)
        selected, specifications = [], []
        stats = dict(gt=0, gt85=0, no85=0, quality=0, sym=0,
                     fused=0, soft=0, competitors=0,
                     correct_before=0, winner85=0)
        for gt_index in matched[actual >= .50].unique():
            same_positive = (matched == gt_index) & (actual >= .50)
            members = same_positive.nonzero(as_tuple=False).squeeze(1)
            best_iou = actual[members].max()
            has85 = bool(best_iou >= .85)
            if has85:
                good_mask = same_positive & (actual >= .85)
            else:
                good_mask = same_positive & (actual >= best_iou - .02)
            good = good_mask.nonzero(as_tuple=False).squeeze(1)
            good_overlap = bbox_overlaps(boxes[good], boxes).max(0)[0]
            bad_mask = ((same_positive & ~good_mask) | (actual < .50))
            # Keep a neutral band: .85+ is never trained as bad merely because
            # another candidate reaches .96. No hard .849/.851 opposition.
            good_floor = actual[good].min()
            bad_mask &= actual < good_floor - .03
            bad_mask &= good_overlap > .50
            bad = bad_mask.nonzero(as_tuple=False).squeeze(1)
            if not len(good) or not len(bad):
                continue
            by_quality = self._top(bad, quality_scan,
                                   self.head_competition_topk)
            by_sym = self._top(bad, sym_score, self.head_competition_topk)
            by_fused = self._top(bad, corrected_scan,
                                 self.head_competition_topk)
            soft_bad, contribution = preceding_suppressors(
                boxes, good, bad_mask, soft_source, self.selector_sigma)
            soft_bad = soft_bad[contribution.topk(
                min(self.head_competition_topk,len(soft_bad))).indices]
            rivals = torch.cat((by_quality, by_sym, by_fused,
                                soft_bad)).unique()
            all_ids = torch.cat((good, rivals)).unique()
            offset = len(selected)
            selected.extend(all_ids.tolist())
            local = {int(value): offset + position
                     for position, value in enumerate(all_ids.tolist())}
            specifications.append((
                [local[int(value)] for value in good],
                [local[int(value)] for value in rivals],
                float(good_floor), actual[rivals].detach(), has85))
            stats['gt'] += 1
            stats['gt85'] += int(has85)
            stats['no85'] += int(not has85)
            stats['quality'] += len(by_quality)
            stats['sym'] += len(by_sym)
            stats['fused'] += len(by_fused)
            stats['soft'] += len(soft_bad)
            stats['competitors'] += len(rivals)
            stats['correct_before'] += int(
                quality_scan[good].max() > quality_scan[rivals].max())
        if not selected:
            self.head_competition_stats.append(stats)
            return zero
        # De-duplicate across GTs, then remap each list to one gradient pass.
        unique = torch.as_tensor(selected, device=boxes.device).unique()
        reverse = {int(value): position for position, value in enumerate(
            unique.tolist())}
        outputs = self._score_candidates(
            self._normalized(boxes[unique], meta), memory, sym[unique], True)[0]
        logits = outputs[:, 0]
        losses = []
        # Rebuild mapping from concatenated local specification to global ID.
        concatenated = selected
        for good_local, bad_local, good_floor, bad_iou, has85 in specifications:
            good_ids = torch.as_tensor([reverse[concatenated[index]]
                                        for index in good_local],
                                       device=boxes.device)
            bad_ids = torch.as_tensor([reverse[concatenated[index]]
                                       for index in bad_local],
                                      device=boxes.device)
            margins = (.5 * (good_floor - bad_iou)).clamp(.015,.30)
            losses.append(winner_loss(logits[good_ids],logits[bad_ids],margins))
            stats['winner85'] += int(has85 and
                logits[good_ids].max().detach() > logits[bad_ids].max().detach())
        self.head_competition_stats.append(stats)
        return self.head_competition_weight * torch.stack(losses).mean()

    def v50_finish_oracle(self, dataset, runner):
        total=0
        hit=dict(sym=0,quality=0,fused=0)
        for index,item in enumerate(self.quality_cache):
            gt=torch.as_tensor(dataset.get_ann_info(index)['bboxes'],dtype=torch.float32).reshape(-1,4)
            boxes=item['boxes']
            if not len(gt) or not len(boxes):
                continue
            overlap=bbox_overlaps(boxes,gt)
            actual,matched=overlap.max(1)
            for g in matched[actual>=.85].unique():
                ids=((matched==g)&(actual>=.5)).nonzero(as_tuple=False).squeeze(1)
                total+=1
                for name,score in (('sym',item['sym'][:,:-1].max(1)[0]),
                                   ('quality',item['quality']),('fused',item['corrected'])):
                    hit[name]+=int(actual[ids[score[ids].argmax()]]>=.85)
        runner.logger.info('[HEAD-COMPETE-VAL] completeCandidates=True '
            'GTwith85=%d top1GT85(sym/quality/fused)=%d/%d/%d '
            'matching=maxIoUGT positivePool=IoUge50',total,hit['sym'],hit['quality'],hit['fused'])
        return super().v50_finish_oracle(dataset,runner)


@HOOKS.register_module()
class HeadCompetitionDebugHook(DetailSelectorDebugHook):
    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        runner.logger.info('[HEAD-COMPETE-BUILD] completeCandidates=True '
            'sources=quality/sym/fused/actualPrefixDecay topk=%d legacyTemperatureUnused=%.3f '
            'weight=%.3f rankTarget=qualityLogit onlyOneRankLoss=True',
            model.head_competition_topk,
            model.head_competition_temperature,
            model.head_competition_weight)

    def after_train_epoch(self, runner):
        super().after_train_epoch(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        keys = ('gt','gt85','no85','quality','sym','fused','soft',
                'competitors','correct_before','winner85')
        total = {key: sum(item[key] for item in model.head_competition_stats)
                 for key in keys}
        runner.logger.info('[HEAD-COMPETE-TRAIN] GT=%d GT85/no85=%d/%d '
            'mined(q/sym/fused/soft)=%d/%d/%d/%d uniqueCompetitors=%d '
            'scanWinnerAllGT=%d forwardWinner85_preUpdate=%d', total['gt'],
            total['gt85'], total['no85'], total['quality'], total['sym'],
            total['fused'], total['soft'], total['competitors'],
            total['correct_before'], total['winner85'])
        model.head_competition_stats = []
