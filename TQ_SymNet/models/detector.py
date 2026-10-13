"""TQ-SymNet: extracted measured implementations; see PROVENANCE.json."""
import math
import logging
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector

@DETECTORS.register_module()
class TQDetector(SingleStageDetector):
    def __init__(self, *args, experiment_tag='TQ_SYMNET',
                 inference_nms='head', inference_only=False,
                 baseline_sigma=.7, baseline_topk=128, baseline_temperature=1., **kwargs):
        # Selector base historically passes local_refine=False. No refiner is instantiated.
        if kwargs.pop('local_refine', False):
            raise ValueError('TQ-SymNet has no H2/local refiner')
        super().__init__(*args, **kwargs)
        self.experiment_tag = str(experiment_tag)
        self.integrated_build_id = 'TQ-SymNet-mask-free-split'
        self.region_contrast_loss_weight = self.boxalign_loss_weight = 0.
        self.use_region_auxiliaries = self.use_pc_branch = self.use_local_refine = False
        self._nam_epoch = 0
        if inference_nms not in ('head', 'tb_hard', 'tb_soft'):
            raise ValueError('Unknown detector inference_nms')
        self.inference_nms = inference_nms
        self.inference_only = bool(inference_only)
        self.baseline_sigma = float(baseline_sigma)
        self.baseline_topk = int(baseline_topk)
        self.baseline_temperature = float(baseline_temperature)
        if self.baseline_temperature <= 0:
            raise ValueError('baseline_temperature must be positive')
        self.capture_candidates = False
        self.baseline_cache = []

    def _build_pyramid(self, backbone_feats, lbp_lab=None):
        if lbp_lab is not None:
            raise ValueError('The historical PC adapter is disabled')
        inputs = list(backbone_feats)
        return tuple(self.neck(inputs)), inputs[0]

    def _collect_lung_masks(self, *args, **kwargs):
        return None

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, **kwargs):
        if self.inference_only:
            raise RuntimeError('B0/B1 are inference controls; evaluate the same detector checkpoint')
        for meta in img_metas:
            meta['batch_input_shape'] = tuple(img.shape[-2:])
        features, c2 = self._build_pyramid(self.backbone(img))
        outputs = self.bbox_head(features, c2_feat=c2)
        return self.bbox_head.loss(outputs[0], outputs[1], gt_bboxes,
            gt_labels, img_metas, gt_bboxes_ignore=gt_bboxes_ignore)

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        from mmdet.core import bbox2result
        features, c2 = self._build_pyramid(self.backbone(img))
        outputs = self.bbox_head(features, c2_feat=c2)
        if self.inference_nms != 'head':
            # Import lazily: selection's base class depends on TQDetector.
            from .selection import gaussian_soft_select, hard_select
            raw = self.bbox_head.get_bboxes(*outputs[:2], img_metas,
                                           rescale=False, with_nms=False)
            results = []
            for (boxes, sym), meta in zip(raw, img_metas):
                keep = (sym[:, :-1] > self.bbox_head.test_cfg.score_thr).any(1)
                boxes, sym = boxes[keep], sym[keep]
                scores, labels = sym[:, :-1].max(1)
                if self.baseline_temperature != 1.:
                    scores = torch.sigmoid(torch.logit(scores.clamp(1e-6, 1.-1e-6))
                                           / self.baseline_temperature)
                if self.inference_nms == 'tb_soft':
                    ids, values = gaussian_soft_select(
                        boxes, scores, self.baseline_topk, self.baseline_sigma)
                else:
                    ids, values = hard_select(boxes, scores, self.baseline_topk,
                        float(self.bbox_head.test_cfg.nms.get('iou_threshold', .5)))
                limit = self.bbox_head.test_cfg.max_per_img
                ids, values = ids[:limit], values[:limit]
                det = torch.cat((boxes[ids], values[:, None]), 1)
                if self.capture_candidates:
                    self.baseline_cache.append(dict(
                        boxes=(boxes / boxes.new_tensor(meta['scale_factor'])).cpu(),
                        selected=ids.cpu(), score=values.cpu(),
                        filename=meta.get('filename')))
                if rescale:
                    det[:, :4] /= det.new_tensor(meta['scale_factor'])
                results.append(bbox2result(det.cpu().numpy(), labels[ids].cpu().numpy(),
                                          self.bbox_head.num_classes))
            return results
        detections = self.bbox_head.get_bboxes(*outputs[:2], img_metas, rescale=rescale)
        return [bbox2result(b, labels, self.bbox_head.num_classes) for b, labels in detections]

    @staticmethod
    def _v330_unwrap_original_images(original_img, batch_size):
        if original_img is None:
            raise KeyError(
                'local_refine=True requires original_img in the data '
                'pipeline. Add PreserveOriginalImage, ImageToTensor and '
                'ToDataContainer(cpu_only=True), then Collect original_img.')
        value = original_img
        while isinstance(value, (list, tuple)) and len(value) == 1:
            nested = value[0]
            if isinstance(nested, (list, tuple)):
                value = nested
            else:
                break
        if isinstance(value, torch.Tensor):
            if value.dim() == 4:
                images = [item for item in value]
            elif value.dim() == 3:
                images = [value]
            else:
                raise ValueError(
                    f'original_img tensor must be CHW/NCHW, got {value.shape}')
        elif isinstance(value, (list, tuple)):
            images = list(value)
        else:
            raise TypeError(
                f'Unsupported original_img container {type(value)!r}')
        if len(images) != batch_size:
            raise ValueError(
                f'original_img batch {len(images)} != img_metas '
                f'batch {batch_size}')
        output = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)
            if image.dim() != 3:
                raise ValueError(
                    f'each original image must be CHW, got {image.shape}')
            output.append(image)
        return output

    @staticmethod
    def _v330_flip_boxes(boxes, width, height, direction):
        output = boxes.clone()
        if direction == 'horizontal':
            output[:, 0] = width - boxes[:, 2]
            output[:, 2] = width - boxes[:, 0]
        elif direction == 'vertical':
            output[:, 1] = height - boxes[:, 3]
            output[:, 3] = height - boxes[:, 1]
        elif direction == 'diagonal':
            output[:, 0] = width - boxes[:, 2]
            output[:, 2] = width - boxes[:, 0]
            output[:, 1] = height - boxes[:, 3]
            output[:, 3] = height - boxes[:, 1]
        else:
            raise ValueError(f'Unsupported flip direction {direction!r}')
        return output

    @classmethod
    def _v330_boxes_to_original(cls, boxes, meta, already_rescaled=False):
        output = boxes.clone()
        ori_h, ori_w = [int(v) for v in meta['ori_shape'][:2]]
        img_h, img_w = [int(v) for v in meta['img_shape'][:2]]
        if bool(meta.get('flip', False)):
            width = ori_w if already_rescaled else img_w
            height = ori_h if already_rescaled else img_h
            output = cls._v330_flip_boxes(
                output, width, height,
                meta.get('flip_direction', 'horizontal'))
        if not already_rescaled:
            scale = output.new_tensor(meta.get('scale_factor', 1.0))
            if scale.numel() == 1:
                output = output / scale
            else:
                output = output / scale.flatten()[:4]
        output[:, 0::2].clamp_(0, ori_w)
        output[:, 1::2].clamp_(0, ori_h)
        return output

    @classmethod
    def _v330_boxes_from_original(cls, boxes, meta):
        output = boxes.clone()
        scale = output.new_tensor(meta.get('scale_factor', 1.0))
        if scale.numel() == 1:
            output = output * scale
        else:
            output = output * scale.flatten()[:4]
        if bool(meta.get('flip', False)):
            img_h, img_w = [int(v) for v in meta['img_shape'][:2]]
            output = cls._v330_flip_boxes(
                output, img_w, img_h,
                meta.get('flip_direction', 'horizontal'))
        return output

    @staticmethod
    def _v330_gray_original(image):
        # original_img is intentionally cpu_only. Crop/resize on CPU and move
        # only the compact patch batch to GPU; this scales to native CXR sizes.
        image = image.to(dtype=torch.float32)
        if image.shape[0] == 1:
            gray = image
        elif image.shape[0] >= 3:
            # LoadImageFromFile uses BGR before Normalize(to_rgb=True).
            gray = (0.114 * image[0:1]
                    + 0.587 * image[1:2]
                    + 0.299 * image[2:3])
        else:
            raise ValueError(
                f'original image has unsupported channels {image.shape[0]}')
        if gray.max().item() > 1.5:
            gray = gray / 255.0
        return gray.clamp(0.0, 1.0)
