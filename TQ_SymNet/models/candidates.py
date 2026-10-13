"""Sym-threshold candidates -> shared visual/LBP/Gabor quality evaluator."""
import hashlib
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import HOOKS, Hook
from mmdet.models.builder import DETECTORS
from mmdet.core import bbox_overlaps, bbox2result, multiclass_nms
from .detector import TQDetector as RetinaNAMAttV44
from .attention import CandidateQualityAttention, competition_rank


def coverage(boxes, gt):
    if not len(boxes) or not len(gt):
        return [0, 0, 0]
    best = bbox_overlaps(boxes, gt).max(0)[0]
    return [int((best >= t).sum()) for t in (.50, .75, .85)]


def bins(iou):
    return [int((iou < .10).sum()), int(((iou >= .10)&(iou < .50)).sum()),
            int(((iou >= .50)&(iou < .75)).sum()),
            int(((iou >= .75)&(iou < .85)).sum()), int((iou >= .85).sum())]


def oracle_result(boxes, sym, gt, classes):
    if not len(boxes) or not len(gt):
        return [np.empty((0, 5), np.float32) for _ in range(classes)]
    overlap = bbox_overlaps(boxes, gt)
    used_boxes, used_gt, selected = set(), set(), []
    for flat in overlap.flatten().argsort(descending=True).tolist():
        i, j = divmod(flat, len(gt))
        if i not in used_boxes and j not in used_gt:
            used_boxes.add(i); used_gt.add(j)
            selected.append((i, float(overlap[i, j])))
        if len(used_gt) == len(gt):
            break
    det = torch.stack([torch.cat((boxes[i], boxes.new_tensor([q]))) for i,q in selected])
    labels = sym[:, :-1].argmax(1)[[i for i,_ in selected]]
    return bbox2result(det.numpy(), labels.numpy(), classes)


class RetinaNAMCandidateQuality(RetinaNAMAttV44):
    def __init__(self, *args, quality_detach=True, quality_texture=True,
                 quality_lbp=True, quality_gabor=True,
                 quality_gaussian=True, quality_hidden=64, quality_heads=4,
                 quality_chunk=32, quality_checkpoint=True,
                 quality_samples_per_bin=64, quality_rank_weight=.10,
                 quality_loss_weight=1., quality_score_mode='sym',
                 quality_underestimate_weight=2., **kwargs):
        kwargs['local_refine'] = False
        super().__init__(*args, **kwargs)
        self.quality_detach = bool(quality_detach)
        self.quality_samples_per_bin = int(quality_samples_per_bin)
        self.quality_rank_weight = float(quality_rank_weight)
        self.quality_loss_weight = float(quality_loss_weight)
        self.quality_score_mode = quality_score_mode
        self.quality_underestimate_weight = float(quality_underestimate_weight)
        if quality_score_mode not in ('sym', 'quality', 'fusion') or self.quality_samples_per_bin < 1:
            raise ValueError('Invalid quality score mode or sample budget')
        if self.region_contrast_loss_weight > 0 or self.bbox_head.use_boxalign or self.bbox_head.use_bbox_contrastive:
            raise ValueError('Quality cascade currently requires historical region/BoxAlign/bbox contrast auxiliaries off')
        with torch.random.fork_rng(devices=[]):
            self.candidate_quality = CandidateQualityAttention(hidden=quality_hidden,
                heads=quality_heads, texture=quality_texture, gaussian=quality_gaussian,
                lbp_enabled=quality_lbp, gabor_enabled=quality_gabor,
                chunk=quality_chunk, checkpoint_candidates=quality_checkpoint)
        if self.quality_detach:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith('candidate_quality.'))
        self.quality_epoch = 0
        self.quality_collect = False
        self.quality_cache = []
        self.quality_train_stats = []
        self.best_quality_key = None
        self.quality_timing = None
        self.integrated_build_id = 'v55.1-levelwise-hardpositive-spatial-quality'

    def _sample_quality_candidates(self, actual, matched, scan):
        """Cover every GT and mine current over/under-estimation failures."""
        selected, detail = [], [0, 0, 0]
        groups = (actual < .50, (actual >= .50)&(actual < .75), actual >= .75)
        for tier, mask in enumerate(groups):
            ids = mask.nonzero(as_tuple=False).squeeze(1)
            budget = min(len(ids), self.quality_samples_per_bin)
            if not budget:
                selected.append(ids)
                continue
            if tier == 0:
                hard_count = int(round(budget * .75))
                hard = ids[scan[ids].topk(min(hard_count, len(ids))).indices]
                available = mask.clone(); available[hard] = False
                rest = available.nonzero(as_tuple=False).squeeze(1)
                random = rest[torch.randperm(len(rest), device=rest.device)[:budget-len(hard)]]
                chosen = torch.cat((hard, random))
            else:
                seeds = []
                for gt_index in matched[ids].unique():
                    candidates = ids[matched[ids] == gt_index]
                    seeds.append(candidates[(actual[candidates]-scan[candidates]).argmax()])
                seeds = torch.stack(seeds) if seeds else ids[:0]
                if len(seeds) > budget:
                    error = actual[seeds] - scan[seeds]
                    seeds = seeds[error.topk(budget).indices]
                available = mask.clone(); available[seeds] = False
                rest = available.nonzero(as_tuple=False).squeeze(1)
                remaining = budget-len(seeds)
                hard_count = min(len(rest), int(round(remaining*.75)))
                error = actual[rest]-scan[rest]
                hard = rest[error.topk(hard_count).indices] if hard_count else rest[:0]
                available[hard] = False
                random = available.nonzero(as_tuple=False).squeeze(1)
                random = random[torch.randperm(len(random),device=random.device)[:remaining-hard_count]]
                chosen = torch.cat((seeds,hard,random))
            detail[tier] = len(chosen)
            selected.append(chosen)
        return torch.cat(selected), detail

    def train(self, mode=True):
        nn.Module.train(self, mode and not self.quality_detach)
        self.candidate_quality.train(mode)
        return self

    def _main(self, img, metas, gt_bboxes=None, gt_labels=None, gt_ignore=None):
        shape = tuple(img.shape[-2:])
        with torch.set_grad_enabled(self.training and not self.quality_detach):
            features, c2 = self._build_pyramid(self.backbone(img), lbp_lab=None)
            outs = self.bbox_head.forward(features, nam_graphs=None, lbp_lab=None,
                c2_feat=c2, region_feat=None,
                gt_bboxes=gt_bboxes, gt_labels=gt_labels,
                img_shapes=[shape]*len(metas),
                lung_masks=self._collect_lung_masks(metas, img.device, shape),
                reg_detail_feats=getattr(self.neck, 'regression_detail_feats', None))
            detection_losses = {}
            if self.training and not self.quality_detach:
                # Run loss immediately after its training forward. No eval
                # forward is inserted here to overwrite D-FINE caches.
                detection_losses = self.bbox_head.loss(outs[0], outs[1], gt_bboxes,
                    gt_labels, metas, gt_bboxes_ignore=gt_ignore)
        with torch.no_grad():
            raw = self.bbox_head.get_bboxes(outs[0], outs[1], metas,
                cfg=self.bbox_head.test_cfg, rescale=False, with_nms=False)
        return features, raw, detection_losses

    def _image_memory(self, features, img, meta, original, index):
        gray = self._v330_gray_original(original).to(img.device, non_blocking=True)
        # Rotate already updates original_img in the pipeline; Resize does
        # not. Texture stays native resolution. Flip must be applied ONCE.
        if meta.get('flip', False):
            direction = meta.get('flip_direction', 'horizontal')
            axes = {'horizontal':[-1], 'vertical':[-2], 'diagonal':[-2,-1]}
            if direction not in axes:
                raise ValueError('Unsupported texture flip direction')
            gray = gray.flip(axes[direction])
        maps = [f[index:index+1].detach() if self.quality_detach else f[index:index+1]
                for f in features[:4]]
        return self.candidate_quality.prepare(maps, gray[None],
            meta['img_shape'][:2], img.shape[-2:])

    @staticmethod
    def _normalized(boxes, meta):
        h,w = meta['img_shape'][:2]
        return boxes / boxes.new_tensor((w,h,w,h))

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, **kwargs):
        if gt_bboxes_ignore is not None and any(len(v) for v in gt_bboxes_ignore):
            raise ValueError('Candidate quality needs explicit ignored-GT labels')
        features, raw, losses = self._main(img, img_metas, gt_bboxes,
                                          gt_labels, gt_bboxes_ignore)
        if self.quality_timing is not None:
            self.quality_timing['detector_done'].record()
        originals = self._v330_unwrap_original_images(kwargs.get('original_img'), len(raw))
        q_losses, rank_losses = [], []
        stats = dict(sample=[0,0,0], error=[0.,0.,0.], pred=[0.,0.,0.], rank_pairs=0,
                     gate=[0.]*7, gate_n=0, sampled_by_tier=[0,0,0],
                     gt_available=[0,0], gt_sampled=[0,0])
        for index, ((boxes,sym),gt,meta,original) in enumerate(zip(raw, gt_bboxes,img_metas,originals)):
            keep = (sym[:, :-1] > self.bbox_head.test_cfg.score_thr).any(1)
            boxes = boxes[keep].detach()
            if not len(boxes):
                continue
            memories = self._image_memory(features, img, meta, original, index)
            normalized = self._normalized(boxes, meta).detach()
            with torch.no_grad():
                scan = self.candidate_quality(normalized, memories)[0].sigmoid()
                overlap = bbox_overlaps(boxes, gt)
                if len(gt):
                    actual, matched = overlap.max(1)
                else:
                    actual = boxes.new_zeros(len(boxes))
                    matched = torch.full((len(boxes),), -1, device=boxes.device, dtype=torch.long)
                selected, sampled = self._sample_quality_candidates(actual, matched, scan)
                stats['sampled_by_tier'] = [a+b for a,b in zip(
                    stats['sampled_by_tier'], sampled)]
                for out_index, threshold in enumerate((.50, .75)):
                    available_gt = matched[actual >= threshold].unique()
                    available_gt = available_gt[available_gt >= 0]
                    sampled_gt = matched[selected][actual[selected] >= threshold].unique()
                    sampled_gt = sampled_gt[sampled_gt >= 0]
                    stats['gt_available'][out_index] += len(available_gt)
                    stats['gt_sampled'][out_index] += len(sampled_gt)
            logit, gate, _ = self.candidate_quality(normalized[selected], memories, training_checkpoint=True)
            prediction, target = logit.sigmoid(), actual[selected].detach()
            group_losses = []
            for tier, mask in enumerate((target < .50, (target >= .50)&(target < .75), target >= .75)):
                if mask.any():
                    item = F.binary_cross_entropy_with_logits(
                        logit[mask], target[mask], reduction='none')
                    if tier == 2:
                        item = item * (1. + self.quality_underestimate_weight *
                            (target[mask]-prediction[mask]).detach().clamp_min(0.))
                    group_losses.append(item.mean())
                    stats['sample'][tier] += int(mask.sum())
                    stats['error'][tier] += float((prediction[mask]-target[mask]).abs().sum().detach())
                    stats['pred'][tier] += float(prediction[mask].sum().detach())
            q_losses.append(torch.stack(group_losses).mean())
            rank, pairs = competition_rank(logit, target, matched[selected], normalized[selected],
                overlap_threshold=float(self.bbox_head.test_cfg.nms.get('iou_threshold', .50)))
            rank_losses.append(rank)
            stats['rank_pairs'] += pairs
            stats['gate'] = [a+b for a,b in zip(stats['gate'], gate.detach().sum(0).tolist())]
            stats['gate_n'] += len(gate)
        zero = self.candidate_quality.quality[-1].weight.sum()*0.
        losses['loss_candidate_iou'] = self.quality_loss_weight * (torch.stack(q_losses).mean() if q_losses else zero)
        losses['loss_candidate_rank'] = self.quality_rank_weight * (torch.stack(rank_losses).mean() if rank_losses else zero)
        self.quality_train_stats.append(stats)
        if self.quality_timing is not None:
            self.quality_timing['forward_done'].record()
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        features, raw, _ = self._main(img, img_metas)
        originals = self._v330_unwrap_original_images(kwargs.get('original_img'), len(raw))
        results = []
        for index, ((boxes,sym),meta,original) in enumerate(zip(raw,img_metas,originals)):
            keep = (sym[:, :-1] > self.bbox_head.test_cfg.score_thr).any(1)
            raw_count = len(boxes)
            boxes, sym = boxes[keep], sym[keep]
            memories = self._image_memory(features,img,meta,original,index)
            logit, gate, entropy = self.candidate_quality(self._normalized(boxes,meta), memories)
            quality = logit.sigmoid()
            score = sym.clone()
            threshold = self.bbox_head.test_cfg.score_thr
            if self.quality_score_mode != 'sym':
                valid = score[:, :-1] > threshold
                value = quality[:, None].expand_as(score[:, :-1])
                if self.quality_score_mode == 'fusion':
                    value = (value*score[:, :-1]).clamp_min(0.).sqrt()
                score[:, :-1] = torch.where(valid, value, value.new_full((), -2.))
                threshold = -1.
            test = self.bbox_head.test_cfg
            det, labels, ids = multiclass_nms(boxes,score,threshold,test.nms,test.max_per_img,return_inds=True)
            if self.quality_collect:
                source = ids // self.bbox_head.num_classes
                scale = boxes.new_tensor(meta['scale_factor'])
                if meta.get('flip', False):
                    raise ValueError('Validation diagnostics require unflipped single augmentation')
                self.quality_cache.append(dict(boxes=(boxes/scale).cpu(),sym=sym.cpu(),
                    quality=quality.cpu(),logit=logit.cpu(),gate=gate.cpu(),entropy=entropy.cpu(),
                    source=source.cpu(),labels=labels.cpu(),score=det[:,4].cpu(),
                    raw_count=raw_count, filename=meta.get('filename')))
            if rescale:
                det[:, :4] /= det.new_tensor(meta['scale_factor'])
            results.append(bbox2result(det.cpu().numpy(),labels.cpu().numpy(),self.bbox_head.num_classes))
        return results

    def v50_begin_oracle(self, epoch):
        self.quality_epoch = int(epoch)
        self.quality_collect = True
        self.quality_cache = []
        return True

    def v50_finish_oracle(self, dataset, runner):
        summaries, errors = {}, np.zeros((5,3))
        candidates, final, hard = [], [], []
        oracle_pre, oracle_post = [], []
        branch_gate, branch_entropy = np.zeros(7), np.zeros(3)
        branch_n, correct, pair_n = 0, 0, 0
        for index,item in enumerate(self.quality_cache):
            gt = torch.as_tensor(dataset.get_ann_info(index)['bboxes'],dtype=torch.float32).reshape(-1,4)
            boxes = item['boxes']
            overlap = bbox_overlaps(boxes,gt)
            if len(gt):
                actual, assigned = overlap.max(1)
            else:
                actual, assigned = torch.zeros(len(boxes)), torch.full((len(boxes),),-1,dtype=torch.long)
            selected = item['source'].unique()
            stages = dict(sym_threshold=torch.arange(len(boxes)),
                quality_top64=item['logit'].topk(min(64,len(boxes))).indices,
                quality_top256=item['logit'].topk(min(256,len(boxes))).indices, final_nms=selected)
            for stage,ids in stages.items():
                summary = summaries.setdefault(stage,dict(count=0,bins=[0]*5,gt=[0]*3))
                summary['count'] += len(ids)
                summary['bins'] = [a+b for a,b in zip(summary['bins'],bins(actual[ids]))]
                summary['gt'] = [a+b for a,b in zip(summary['gt'],coverage(boxes[ids],gt))]
            edges = [0.,.1,.5,.75,.85,1.0001]
            for b in range(5):
                mask = (actual >= edges[b]) & (actual < edges[b+1])
                errors[b] += [int(mask.sum()),float((item['quality'][mask]-actual[mask]).abs().sum()),float(item['quality'][mask].sum())]
            for g in assigned[actual >= .50].unique():
                ids = ((assigned == g)&(actual >= .50)).nonzero(as_tuple=False).squeeze(1)
                if len(ids)>256:
                    ids=ids[torch.linspace(0,len(ids)-1,256).long()]
                gap = actual[ids,None]-actual[ids][None]
                valid=gap>.05
                correct += int(((item['logit'][ids,None]-item['logit'][ids][None]>0)&valid).sum())
                pair_n += int(valid.sum())
            branch_gate += item['gate'].sum(0).numpy(); branch_entropy += item['entropy'].sum(0).numpy()
            branch_n += len(boxes)
            retained=torch.zeros(len(boxes),dtype=torch.bool); retained[selected]=True
            for kind,mask in (('overestimated_low',(actual<.50)&(item['quality']>=.75)),
                              ('underestimated_good',(actual>=.75)&(item['quality']<.50))):
                ids=mask.nonzero(as_tuple=False).squeeze(1)
                discrepancy=(item['quality'][ids]-actual[ids]).abs()
                ids=ids[discrepancy.topk(min(20,len(ids))).indices]
                for i in ids.tolist():
                    hard.append(dict(image_index=index,filename=item['filename'],kind=kind,
                        bbox=boxes[i].tolist(),true_iou=float(actual[i]),qIoU=float(item['quality'][i]),
                        logit=float(item['logit'][i]),sym=item['sym'][i,:-1].tolist(),nms_kept=bool(retained[i])))
            candidates.append(dict(image_index=index,filename=item['filename'],raw_count=item['raw_count'],
                boxes=boxes.tolist(),true_iou=actual.tolist(),matched_gt=assigned.tolist(),
                qIoU=item['quality'].tolist(),rank_logit=item['logit'].tolist(),
                sym=item['sym'][:,:-1].tolist(),branch_weights=item['gate'].tolist()))
            for position,i in enumerate(item['source'].tolist()):
                final.append(dict(image_index=index,filename=item['filename'],bbox=boxes[i].tolist(),
                    subtype=int(item['labels'][position]),true_iou=float(actual[i]),
                    qIoU=float(item['quality'][i]),rank_logit=float(item['logit'][i]),
                    sym=item['sym'][i,:-1].tolist(),nms_score=float(item['score'][position])))
            if self.quality_epoch%4==0:
                oracle_pre.append(oracle_result(boxes,item['sym'],gt,self.bbox_head.num_classes))
                oracle_post.append(oracle_result(boxes[selected],item['sym'][selected],gt,self.bbox_head.num_classes))
        for stage,values in summaries.items():
            runner.logger.info('[CQ-STAGE] epoch=%d %s count=%d IoUbins=%s GT50/75/85=%s',
                self.quality_epoch,stage,values['count'],values['bins'],values['gt'])
        runner.logger.info('[CQ-PREDICTION] counts=%s MAE=%s predictedMeans=%s '
            'sameGTorder=%.5f pairs=%d', errors[:,0].astype(int).tolist(),
            (errors[:,1]/np.maximum(errors[:,0],1)).tolist(),
            (errors[:,2]/np.maximum(errors[:,0],1)).tolist(),correct/max(pair_n,1),pair_n)
        gate_mean=(branch_gate/max(branch_n,1)).tolist()
        runner.logger.info('[CQ-BRANCH] visual/LBP/Gabor=%s P2/P3/P4/P5=%s '
            'entropyVisual/LBP/Gabor=%s detach=%s',gate_mean[:3],gate_mean[3:],
            (branch_entropy/max(branch_n,1)).tolist(),self.quality_detach)
        spatial=(2.*self.candidate_quality.gaussian_raw.detach().sigmoid()).cpu().tolist()
        runner.logger.info('[CQ-SPATIAL] gaussianEnabled=%s strengthsVisual/LBP/Gabor=%s '
            'sigmaWidthHeightFactors=0.35/0.85 queryRoles=core/ring/context/global',
            self.candidate_quality.gaussian,spatial)
        import mmcv
        for label,rows in (('candidates',candidates),('final_scores',final),('hard_samples',hard),('stages',summaries)):
            path=osp.join(runner.work_dir,'cq_%s_epoch_%02d.json'%(label,self.quality_epoch))
            mmcv.dump(rows,path)
            runner.logger.info('[CQ-EXPORT] %s',path)
        if oracle_pre:
            for label,result in (('sym_threshold',oracle_pre),('final_nms',oracle_post)):
                metric=dataset.evaluate(result,metric='bbox',tb_official_agnostic=True,tb_class_agnostic=False)
                runner.logger.info('[CQ-ORACLE] epoch=%d stage=%s AP/AP50/AP75=%.3f/%.3f/%.3f',
                    self.quality_epoch,label,metric['official_tb_bbox_mAP'],metric['official_tb_bbox_mAP_50'],metric['official_tb_bbox_mAP_75'])
        if summaries:
            top=summaries['quality_top64']
            macro_mae=float(np.mean(errors[errors[:,0]>0,1]/errors[errors[:,0]>0,0]))
            key=(top['gt'][1],top['gt'][2],correct/max(pair_n,1),-macro_mae)
            improved=self.best_quality_key is None or key>self.best_quality_key
            runner.logger.info('[CQ-SELECT] Top64_GT75/85=%d/%d order=%.5f macroMAE=%.5f improved=%s notMAP=True',
                key[0],key[1],key[2],macro_mae,improved)
            if improved:
                self.best_quality_key=key
                runner.save_checkpoint(runner.work_dir,filename_tmpl='best_candidate_quality.pth',
                    save_optimizer=True,create_symlink=False,meta=dict(candidate_quality_key=key))
        self.quality_cache=[]; self.quality_collect=False


def main_hash(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if not name.startswith('candidate_quality.'):
            digest.update(name.encode()); digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class CandidateQualityDebugHook(Hook):
    def before_train_iter(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        model.quality_timing=None
        if (runner.iter+1)%50==0 and torch.cuda.is_available():
            model.quality_timing={key:torch.cuda.Event(enable_timing=True)
                for key in ('start','detector_done','forward_done','end')}
            model.quality_timing['start'].record()

    def after_train_iter(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        timing=model.quality_timing
        if timing is not None:
            timing['end'].record(); timing['end'].synchronize()
            runner.logger.info('[CQ-TIME] iter=%d detector=%.3fs qualityForward=%.3fs '
                'backwardUpdate=%.3fs total=%.3fs',runner.iter+1,
                timing['start'].elapsed_time(timing['detector_done'])/1000.,
                timing['detector_done'].elapsed_time(timing['forward_done'])/1000.,
                timing['forward_done'].elapsed_time(timing['end'])/1000.,
                timing['start'].elapsed_time(timing['end'])/1000.)
            model.quality_timing=None

    def before_run(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        self.initial_hash=main_hash(model)
        runner.logger.info('[CQ-PARAMS] qualityHead=%d perBin=%d chunk=%d '
            'updates=onePerIteration fp32=True',sum(p.numel() for p in model.candidate_quality.parameters()),
            model.quality_samples_per_bin,model.candidate_quality.chunk)
        runner.logger.info('[CQ-BUILD] id=%s detach=%s texture=%s gaussian=%s hidden=%d '
            'noPTB=True noH2=True noResNet=True NMS=%s mainSHA256=%s',model.integrated_build_id,
            model.quality_detach,model.candidate_quality.texture,model.candidate_quality.gaussian,
            model.candidate_quality.hidden,model.quality_score_mode,self.initial_hash)

    def before_train_epoch(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        model._nam_epoch=runner.epoch

    def after_train_epoch(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        samples=np.zeros(3); sampled=np.zeros(3); error=np.zeros(3); pred=np.zeros(3); gate=np.zeros(7); gn=0; pairs=0
        gt_available=np.zeros(2); gt_sampled=np.zeros(2)
        for values in model.quality_train_stats:
            samples+=values['sample']; sampled+=values['sampled_by_tier']; error+=values['error']; pred+=values['pred']
            gate+=values['gate']; gn+=values['gate_n']; pairs+=values['rank_pairs']
            gt_available+=values['gt_available']; gt_sampled+=values['gt_sampled']
        gate_mean=(gate/max(gn,1)).tolist()
        runner.logger.info('[CQ-TRAIN] low/mid/high=%s sampled=%s qMAE=%s '
            'predictedMeans=%s rankPairs=%d visual/LBP/Gabor=%s P2/P3/P4/P5=%s',
            samples.astype(int).tolist(),sampled.astype(int).tolist(),
            (error/np.maximum(samples,1)).tolist(),(pred/np.maximum(samples,1)).tolist(),
            pairs,gate_mean[:3],gate_mean[3:])
        runner.logger.info('[CQ-HARD-MINING] available/sampledGT50=%d/%d '
            'available/sampledGT75=%d/%d highTierIncludesMostUnderestimatedPerGT=True '
            'lowTierHardRatio=0.75 loss=softIoUBCE highUnderWeight=%.2f',
            gt_available[0],gt_sampled[0],gt_available[1],gt_sampled[1],
            model.quality_underestimate_weight)
        current=main_hash(model)
        runner.logger.info('[CQ-MAIN] detach=%s mainChanged=%s mainSHA256=%s',
            model.quality_detach,current!=self.initial_hash,current)
        if model.quality_detach and current!=self.initial_hash:
            raise RuntimeError('Frozen main detector weights changed')
        model.quality_train_stats=[]
