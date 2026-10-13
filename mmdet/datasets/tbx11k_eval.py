"""TBX11K evaluation helpers independent from MMDetection runtime.

The detector remains two-class (Active/Obsolete), while this module builds a
single ``tuberculosis`` evaluation view for comparison with work that reports
category-collapsed TBX11K COCO AP.  Cross-class duplicate boxes are removed
before COCOeval; simply setting ``useCats=0`` does not do that.
"""

from copy import deepcopy

import numpy as np


TB_CATEGORY_ID = 1


def _xywh_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    box_x2, box_y2 = box[0] + box[2], box[1] + box[3]
    boxes_x2 = boxes[:, 0] + boxes[:, 2]
    boxes_y2 = boxes[:, 1] + boxes[:, 3]
    inter_w = np.maximum(
        0.0, np.minimum(box_x2, boxes_x2) - np.maximum(box[0], boxes[:, 0]))
    inter_h = np.maximum(
        0.0, np.minimum(box_y2, boxes_y2) - np.maximum(box[1], boxes[:, 1]))
    inter = inter_w * inter_h
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter
    return inter / np.maximum(union, np.finfo(np.float32).eps)


def merge_tb_predictions(predictions, iou_threshold=0.5, max_dets=100):
    """Collapse all TB subclasses and apply deterministic cross-class NMS."""
    by_image = {}
    for prediction in predictions:
        item = deepcopy(prediction)
        item['category_id'] = TB_CATEGORY_ID
        by_image.setdefault(item['image_id'], []).append(item)

    merged = []
    for image_id in sorted(by_image):
        candidates = by_image[image_id]
        # Stable secondary index makes equal-score results reproducible.
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(candidates[index]['score']), index))
        kept = []
        while order and len(kept) < max_dets:
            best = order.pop(0)
            kept.append(best)
            if not order:
                break
            ious = _xywh_iou(
                candidates[best]['bbox'],
                [candidates[index]['bbox'] for index in order])
            order = [
                index for index, iou in zip(order, ious)
                if iou <= iou_threshold
            ]
        merged.extend(candidates[index] for index in kept)
    return merged


def make_single_tb_ground_truth(coco_dataset):
    """Return a deep-copied COCO dataset dictionary with one TB category."""
    dataset = deepcopy(coco_dataset)
    dataset['categories'] = [dict(id=TB_CATEGORY_ID, name='tuberculosis')]
    for annotation in dataset.get('annotations', []):
        annotation['category_id'] = TB_CATEGORY_ID
    return dataset
