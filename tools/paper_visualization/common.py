"""Offline diagnostics only. XYXY uses continuous coordinates (no '+1')."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

BIN_LABELS = ['<0.10', '0.10-0.50', '0.50-0.75', '0.75-0.85', '>=0.85']
BIN_EDGES = [0., .10, .50, .75, .85, 1.000001]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def as_boxes(value):
    array = np.asarray(value, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(array).all():
        raise ValueError('Non-finite box coordinates')
    if ((array[:, 2:] - array[:, :2]) < 0).any():
        raise ValueError('Inverted XYXY box')
    return array


def iou_matrix(boxes, gt):
    boxes, gt = as_boxes(boxes), as_boxes(gt)
    wh = np.maximum(0., np.minimum(boxes[:, None, 2:], gt[None, :, 2:])
                    - np.maximum(boxes[:, None, :2], gt[None, :, :2]))
    inter = wh.prod(2)
    area_b = np.maximum(0., boxes[:, 2:] - boxes[:, :2]).prod(1)
    area_g = np.maximum(0., gt[:, 2:] - gt[:, :2]).prod(1)
    return inter / np.maximum(area_b[:, None] + area_g[None] - inter, 1e-12)


def box_stats(boxes, gt):
    overlap = iou_matrix(boxes, gt)
    best = overlap.max(1) if overlap.shape[1] else np.zeros(len(overlap))
    coverage = [(overlap >= threshold).any(0).sum().item()
                for threshold in (.50, .75, .85)]
    return dict(count=len(best), iou_bins=np.histogram(best, BIN_EDGES)[0].tolist(),
                covered_gt=coverage), best


def match_predictions(predictions, gt, threshold=.50):
    """Score-ordered, one-to-one, category-agnostic matching for plots.

    Not a substitute for COCOeval: this helper does not implement crowd/ignore
    matching or interpolated AP. Exported GT has already passed dataset parsing.
    """
    boxes = [p['bbox_xyxy'] for p in predictions]
    overlap = iou_matrix(boxes, gt)
    best = overlap.max(1) if overlap.shape[1] else np.zeros(len(predictions))
    used = np.zeros(overlap.shape[1], dtype=bool)
    status = np.full(len(predictions), 'background', dtype=object)
    matched = np.full(len(predictions), -1, dtype=int)
    order = np.argsort(-np.array([p['score'] for p in predictions]), kind='stable')
    for index in order:
        available = np.flatnonzero(~used)
        if len(available):
            target = available[np.argmax(overlap[index, available])]
            if overlap[index, target] >= threshold:
                used[target] = True
                status[index], matched[index] = 'TP', target
                continue
        if best[index] >= threshold:
            status[index] = 'duplicate'
        elif best[index] >= .10:
            status[index] = 'localization'
    return dict(status=status.tolist(), matched_gt=matched.tolist(),
                max_iou=best.tolist(), tp=int((status == 'TP').sum()),
                fp=int((status != 'TP').sum()), fn=int((~used).sum()))


def validate_comparison(runs):
    """Fail closed if cohorts, detector weights or GT coordinates differ."""
    reference = runs[0]
    ids = [item['image_id'] for item in reference['images']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate image IDs')
    for run in runs[1:]:
        for key in ('ann_sha256', 'detector_sha256', 'detector_state_sha256',
                    'nms_pre','eligibility_score_thr','max_per_img','classes'):
            if run['run'].get(key) != reference['run'].get(key):
                raise ValueError('Unmatched comparison: ' + key)
        if [item['image_id'] for item in run['images']] != ids:
            raise ValueError('Cohort/order mismatch; export the same complete split')
        for a, b in zip(reference['images'], run['images']):
            if (a['gt'] != b['gt'] or a.get('ignored_gt') != b.get('ignored_gt')
                    or a['width'] != b['width'] or a['height'] != b['height']):
                raise ValueError('GT/image-size mismatch')
            if 'candidates' in a and 'candidates' in b:
                boxes_a=as_boxes(a['candidates']['boxes_xyxy'])
                boxes_b=as_boxes(b['candidates']['boxes_xyxy'])
                if boxes_a.shape != boxes_b.shape or not np.allclose(
                        boxes_a,boxes_b,rtol=0.,atol=.001):
                    raise ValueError('Eligible candidate pools differ for image '+str(a['image_id']))

