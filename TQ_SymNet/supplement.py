"""Isolated publication controls. Never imported by the retained recipe."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mmcv.runner import HOOKS, Hook
from mmdet.core import bbox2result, bbox_overlaps
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector

from .models.gflv2_adapter import TQGFLV2Head  # register, no force=True
from .models.selection import gaussian_soft_select
from .models.candidates import bins, coverage, oracle_result
from .data_contract import sha256_file, verify_dataset_contract


def frozen_state_hash(model):
    """Audit persistent buffers as well as parameters; no RNG or forward."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith('candidate_quality.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@DETECTORS.register_module()
class TQDenseHeadBaseline(SingleStageDetector):
    """Normal dense-head training, identical CrossScale, no Sym/quality cascade.

    The official MMDet GFL/VFNet head forward/loss/decode paths are untouched.
    Only post-decode, overall-TB suppression is shared with TQ-SymNet.
    """
    def __init__(self, *args, inference_nms='tb_soft', baseline_sigma=.7,
                 baseline_topk=128, diagnostics=True, oracle_interval=4,
                 experiment_tag='SUPPLEMENT_DENSE_HEAD', **kwargs):
        super().__init__(*args, **kwargs)
        if inference_nms not in ('tb_soft', 'head'):
            raise ValueError('Use tb_soft or the native classwise head NMS')
        self.inference_nms = inference_nms
        self.baseline_sigma, self.baseline_topk = float(baseline_sigma), int(baseline_topk)
        self.diagnostics, self.oracle_interval = bool(diagnostics), int(oracle_interval)
        if self.baseline_sigma <= 0 or self.baseline_topk < 100 or self.oracle_interval < 1:
            raise ValueError('Invalid suppression/diagnostic settings')
        self.experiment_tag = experiment_tag
        self.quality_collect = False
        self.baseline_cache = []

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, original_img=None, **kwargs):
        if kwargs:
            raise TypeError('Unexpected training fields: '+str(sorted(kwargs)))
        # The original-image field is collected by the common pipeline but
        # is intentionally unused by official dense heads.
        return super().forward_train(img, img_metas, gt_bboxes, gt_labels,
                                     gt_bboxes_ignore=gt_bboxes_ignore)

    def simple_test(self, img, img_metas, rescale=False, original_img=None, **kwargs):
        if kwargs:
            raise TypeError('Unexpected inference fields: '+str(sorted(kwargs)))
        outs = self.bbox_head(self.extract_feat(img))
        if self.inference_nms == 'head':
            detections = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
            return [bbox2result(box, label, self.bbox_head.num_classes)
                    for box, label in detections]
        raw = self.bbox_head.get_bboxes(*outs, img_metas, rescale=False, with_nms=False)
        results = []
        for (boxes, scores), meta in zip(raw, img_metas):
            if scores.shape[1] != self.bbox_head.num_classes+1:
                raise RuntimeError('Expected MMDet scores with dummy background')
            probability, labels = scores[:, :-1].max(1)
            eligible = probability > self.bbox_head.test_cfg.score_thr
            boxes, probability, labels = boxes[eligible], probability[eligible], labels[eligible]
            ids, values = gaussian_soft_select(boxes, probability,
                                               self.baseline_topk, self.baseline_sigma)
            ids, values = ids[:self.test_cfg.max_per_img], values[:self.test_cfg.max_per_img]
            original_boxes = boxes / boxes.new_tensor(meta['scale_factor'])
            if self.quality_collect:
                self.baseline_cache.append(dict(boxes=original_boxes.detach().cpu(),
                    selected=ids.detach().cpu(), score=values.detach().cpu(),
                    filename=meta.get('filename')))
            chosen = original_boxes[ids] if rescale else boxes[ids]
            det = torch.cat((chosen, values[:, None]), 1)
            results.append(bbox2result(det, labels[ids], self.bbox_head.num_classes))
        return results

    def v50_begin_oracle(self, epoch):
        self.supplement_epoch = int(epoch)
        self.baseline_cache = []
        self.quality_collect = (self.diagnostics and self.inference_nms == 'tb_soft'
                                and epoch % self.oracle_interval == 0)

    def v50_finish_oracle(self, dataset, runner):
        if not self.quality_collect:
            return
        if len(self.baseline_cache) != len(dataset):
            raise RuntimeError('Incomplete dense-head validation diagnostic')
        stages = {k: dict(count=0, iou_bins=[0]*5, covered_gt=[0]*3)
                  for k in ('eligible', 'final')}
        oracle = []
        for index, row in enumerate(self.baseline_cache):
            box = row['boxes']
            gt = torch.as_tensor(dataset.get_ann_info(index)['bboxes']).float().reshape(-1, 4)
            actual = bbox_overlaps(box, gt).max(1).values if len(gt) else box.new_zeros(len(box))
            for stage, ids in (('eligible', torch.arange(len(box))), ('final', row['selected'])):
                entry = stages[stage]
                entry['count'] += len(ids)
                entry['iou_bins'] = [a+b for a, b in zip(entry['iou_bins'], bins(actual[ids]))]
                entry['covered_gt'] = [a+b for a, b in zip(entry['covered_gt'], coverage(box[ids], gt))]
            # GT-dependent one-to-one candidate diagnostic only, never inference.
            dummy = box.new_zeros((len(box), self.bbox_head.num_classes+1))
            oracle.append(oracle_result(box, dummy, gt, self.bbox_head.num_classes))
        metrics = dataset.evaluate(oracle, metric='bbox', tb_official_agnostic=True,
                                   tb_class_agnostic=False, bbox_max_dets=(1, 10, 100))
        report = dict(epoch=self.supplement_epoch, stages=stages, oracle=metrics,
                      note='GT-dependent diagnostic; coverage is not AP; not proof of achievable AP.')
        Path(runner.work_dir, 'dense_candidates_epoch_%02d.json' % self.supplement_epoch).write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        runner.logger.info('[SUPPLEMENT-DENSE-STAGES] %s', json.dumps(report))
        self.baseline_cache = []
        self.quality_collect = False


@HOOKS.register_module()
class TQSupplementAuditHook(Hook):
    """Fail closed on wrong cohort/optimizer wiring; record scientific scope."""
    def __init__(self, train_annotation, val_annotation, role,
                 train_sha256='9d1b19eb8985e281c4767148fac466445844b764db3dab2f7ca635503470971c',
                 val_sha256='bdec993506af15bf72010b32ec82ac279615ccc4c51ad9ca9438a7b9638d6e9d'):
        self.paths = dict(train=train_annotation, val=val_annotation)
        self.hashes = dict(train=train_sha256, val=val_sha256)
        self.role = role

    def before_run(self, runner):
        self.model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        cohorts = []
        for phase, count, boxes in (('train', 599, 902), ('val', 200, 309)):
            path = Path(self.paths[phase])
            if sha256_file(path) != self.hashes[phase]:
                raise RuntimeError('Wrong publication annotation: '+phase)
            doc = json.loads(path.read_text(encoding='utf-8-sig'))
            if len(doc['images']) != count or len(doc['annotations']) != boxes:
                raise RuntimeError('Unexpected publication cohort size')
            cohorts.append({x['file_name'] for x in doc['images']})
        if cohorts[0] & cohorts[1]:
            raise RuntimeError('Overlapping train/val filenames')
        quality = getattr(self.model, 'candidate_quality', None)
        if self.role in ('matched_roi', 'matched_full'):
            expected_arch = 'pooled' if self.role == 'matched_roi' else 'attention'
            if quality is None or self.model.quality_arch != expected_arch:
                raise RuntimeError('Unexpected matched quality architecture')
            if any(p.requires_grad for n, p in self.model.named_parameters()
                   if not n.startswith('candidate_quality.')):
                raise RuntimeError('Matched ROI experiment must freeze detector')
            if self.model.selector_quality_target != 'direct_iou' or (
                    self.model.quality_rank_weight or self.model.selector_rank_weight):
                raise RuntimeError('ROI comparison needs direct IoU and no rank')
            alpha = .1+.8*quality.fusion_gate_logit.detach().sigmoid()
            if quality.fusion_gate_logit.requires_grad or abs(float(alpha)-.5) > 1e-6:
                raise RuntimeError('ROI matched controls must keep alpha=0.5')
            self.detector_hash = frozen_state_hash(self.model)
        elif self.role == 'dense_joint':
            if quality is not None or not any(p.requires_grad for p in self.model.backbone.parameters()):
                raise RuntimeError('Dense baseline must train backbone and have no quality cascade')
        else:
            raise ValueError('Unknown supplement role')
        params = dict(total=sum(p.numel() for p in self.model.parameters()),
            trainable=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            quality=sum(p.numel() for p in quality.parameters()) if quality is not None else 0)
        optimizer_ids = {id(p) for g in runner.optimizer.param_groups for p in g['params']}
        if any(id(p) not in optimizer_ids for p in self.model.parameters() if p.requires_grad):
            raise RuntimeError('A trainable parameter is missing from optimizer')
        row = dict(role=self.role, protocol='original_599_200', hashes=self.hashes,
            parameters=params, seed=(runner.meta or {}).get('seed'),
            bbox_head=type(self.model.bbox_head).__name__,
            optimizer_lrs=sorted({g['lr'] for g in runner.optimizer.param_groups}),
            frozen_detector=self.role in ('matched_roi', 'matched_full'),
            comparison_scope=('same detector and candidate domain' if quality is not None
                              else 'different jointly trained detector; NOT same-candidate evidence'))
        runner.logger.info('[SUPPLEMENT-BUILD] %s', json.dumps(row))
        runner.meta = runner.meta or {}
        runner.meta['tq_supplement'] = row
        Path(runner.work_dir, 'supplement_contract.json').write_text(json.dumps(row, indent=2), encoding='utf-8')

    def before_train_epoch(self, runner):
        if len(runner.data_loader) != 454:
            raise RuntimeError('Expected original cohort batch2: 454 updates/epoch; check sampling/batch')
        verify_dataset_contract(runner.data_loader.dataset, self.hashes['train'], 'train')
        for hook in runner.hooks:
            if hasattr(hook, 'dataloader'):
                verify_dataset_contract(hook.dataloader.dataset, self.hashes['val'], 'val')

    def after_train_epoch(self, runner):
        if self.role in ('matched_roi', 'matched_full'):
            unchanged = frozen_state_hash(self.model) == self.detector_hash
            runner.logger.info('[SUPPLEMENT-FROZEN] epoch=%d mainUnchanged=%s', runner.epoch+1, unchanged)
            if not unchanged:
                raise RuntimeError('Frozen detector parameters/buffers changed')
