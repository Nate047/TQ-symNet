"""Paired image bootstrap from official COCO per-image matching records.

Matching is done by COCOeval, not by a new IoU matcher. Repeated images
contribute repeated detections AND repeated GT; score accumulation remains
global. This estimates image-sampling uncertainty, NOT training variability.
"""
import numpy as np


def coco_all_area_records(evaluator):
    p = evaluator.params
    if p.useCats != 0 or p.maxDets[-1] != 100:
        raise ValueError('Require category-agnostic COCO evaluation, maxDets=100')
    target_area = p.areaRng[p.areaRngLbl.index('all')]
    rows = {row['image_id']: row for row in evaluator.evalImgs
            if row is not None and list(row['aRng']) == list(target_area)
            and row['maxDet'] == 100}
    return [rows.get(image_id) for image_id in p.imgIds]


def ap_from_records(records, indices=None):
    chosen = records if indices is None else [records[int(i)] for i in indices]
    chosen = [x for x in chosen if x is not None]
    count = sum(int(np.logical_not(x['gtIgnore']).sum()) for x in chosen)
    if not count:
        return np.full(10, np.nan)
    scores = np.concatenate([np.asarray(x['dtScores'])[:100] for x in chosen])
    matches = np.concatenate([x['dtMatches'][:, :100] for x in chosen], axis=1)
    ignored = np.concatenate([x['dtIgnore'][:, :100] for x in chosen], axis=1)
    order = np.argsort(-scores, kind='mergesort')
    matches, ignored = matches[:, order], ignored[:, order]
    tp = np.cumsum(np.logical_and(matches > 0, ~ignored), axis=1, dtype=float)
    fp = np.cumsum(np.logical_and(matches == 0, ~ignored), axis=1, dtype=float)
    recall_levels = np.linspace(0., 1., 101)
    result = []
    for t, f in zip(tp, fp):
        recall = t/count
        precision = t/(t+f+np.spacing(1))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        positions = np.searchsorted(recall, recall_levels, side='left')
        sampled = np.zeros(101)
        valid = positions < len(precision)
        sampled[valid] = precision[positions[valid]]
        result.append(sampled.mean()*100.)
    return np.asarray(result)


def paired_bootstrap(reference, candidate, samples=2000, seed=0):
    if len(reference) != len(candidate) or samples < 1:
        raise ValueError('Bootstrap needs matching full image cohorts')
    rng = np.random.RandomState(seed)
    differences = []
    for _ in range(samples):
        ids = rng.randint(0, len(reference), size=len(reference))
        a, b = ap_from_records(reference, ids), ap_from_records(candidate, ids)
        if np.isfinite(a).all() and np.isfinite(b).all():
            differences.append([float((b-a).mean()), float(b[0]-a[0]), float(b[5]-a[5])])
    if not differences:
        raise RuntimeError('No nonempty bootstrap samples')
    values = np.asarray(differences)
    return dict(resamples=samples, valid_resamples=len(values), seed=seed,
        differences_pp={name: dict(mean=float(values[:, i].mean()),
                         ci95=np.percentile(values[:, i], [2.5, 97.5]).tolist())
                        for i, name in enumerate(('AP', 'AP50', 'AP75'))},
        scope='Paired image resampling of fixed selected predictions; not seed variability or selection-bias correction.')
