"""Training-only signed geometry and good-set choice. No box modifications."""
import math
import torch
from torch.nn import functional as F


def cross_iou(a, b):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb-lt).clamp_min(0).prod(-1)
    aa = (a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    bb = (b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return inter/(aa[:,None]+bb[None]-inter).clamp_min(1e-8)


def passing_count(actual):
    thresholds = actual.new_tensor([.50+.05*k for k in range(10)])
    return (actual[:,None] >= thresholds).sum(1)


def winner_sets(actual, matched, boxes, max_winners=4, overlap_threshold=.5):
    """Each GT: up to4 best-IoU representatives on its best COCO plateau.

    Unchosen EQUIVALENT-plateau boxes are neutral, never negatives. Rivals are
    lower-plateau overlapping boxes from this GT, or max-IoU<.5 candidates.
    Positives assigned to ANOTHER GT are not negatives. NO rival-count cap.
    These labels are constructed ONLY during training/posthoc GT diagnostics.
    """
    if max_winners < 1 or not 0. < overlap_threshold < 1.:
        raise ValueError('Invalid winner-set parameters')
    levels = passing_count(actual)
    result = []
    for g in matched[(actual >= .5) & (matched >= 0)].unique():
        own = (matched == g) & (actual >= .5)
        ids = own.nonzero(as_tuple=False).squeeze(1)
        best_level = levels[ids].max()
        equivalent = ids[levels[ids] == best_level]
        # PyTorch 1.10 supports stable sort on CPU/CUDA, but not stable
        # Tensor.argsort. Keep identical tie ordering without a CPU fallback.
        ordered = torch.sort(actual[equivalent], descending=True, stable=True).indices
        high = equivalent[ordered[:max_winners]]
        overlap = cross_iou(boxes,boxes[high]).max(1)[0]
        rival_mask = ((matched == g) | (actual < .5)) & (levels < best_level)
        rival_mask &= overlap > overlap_threshold
        rivals = rival_mask.nonzero(as_tuple=False).squeeze(1)
        result.append(dict(gt=g, high=high, rivals=rivals,
                           neutral_equivalent=len(equivalent)-len(high),best_level=best_level))
    return result


def winner_set_loss(logits, groups, temperature=.5):
    """Normalized multi-positive choice NLL; optimum unchanged by count scaling.

    Division by log(|C|/|H|) makes uniform-score initial loss=1 independently of
    duplicate density. Keep the external rank weight .1; no extra pair-rank.
    """
    if temperature <= 0 or not math.isfinite(float(temperature)):
        raise ValueError('Winner temperature must be finite and positive')
    terms=[]; observations=0; stats=[]
    for group in groups:
        h, r = group['high'], group['rivals']
        if not len(r):
            continue
        qh, qr = logits[h].float()/temperature, logits[r].float()/temperature
        denominator = torch.logsumexp(torch.cat((qh,qr)),0)
        numerator = torch.logsumexp(qh,0)
        scale = math.log((len(h)+len(r))/len(h))
        terms.append((denominator-numerator)/scale)
        observations += len(r)
        with torch.no_grad():
            stats.append(torch.stack((logits.new_tensor(float(len(h))),
                logits.new_tensor(float(len(r))),
                logits.new_tensor(float(group['neutral_equivalent'])),
                (numerator-denominator).exp(),(qh.max() >= qr.max()).to(logits.dtype))))
    return (torch.stack(terms).mean() if terms else logits.sum()*0.),observations,stats


def signed_geometry_targets(boxes, gt, matched, actual, ambiguity_gap=.1):
    if not 0. <= ambiguity_gap <= 1.:
        raise ValueError('Geometry ambiguity gap must be in [0,1]')
    target = torch.zeros_like(boxes)
    valid = (actual >= .5) & (matched >= 0) & (matched < len(gt))
    valid &= ((boxes[:,2:]-boxes[:,:2]) > 1e-5).all(1)
    ambiguous = torch.zeros_like(valid)
    if not len(gt) or not len(boxes):
        return target,valid,ambiguous
    if len(gt) > 1:
        values = cross_iou(boxes,gt).topk(2,dim=1).values
        ambiguous = valid & ((values[:,0]-values[:,1]) < ambiguity_gap)
        valid &= ~ambiguous
    ids = valid.nonzero(as_tuple=False).squeeze(1)
    size = (boxes[ids,2:]-boxes[ids,:2]).clamp_min(1e-5)
    scale = torch.cat((size,size),1)  # XYXY ordering: w,h,w,h
    target[ids] = (gt[matched[ids]]-boxes[ids])/scale
    return target,valid,ambiguous


def signed_geometry_loss(prediction, target, valid, matched, beta=.05):
    if beta <= 0:
        raise ValueError('Geometry SmoothL1 beta must be positive')
    if not bool(valid.any()):
        return prediction.sum()*0.
    element = F.smooth_l1_loss(prediction[valid],target[valid],reduction='none',beta=beta).mean(1)
    ids=matched[valid]
    return torch.stack([element[ids==g].mean() for g in ids.unique()]).mean()
