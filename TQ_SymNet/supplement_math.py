"""Small numerical helpers; no registries, model imports, or GT at scoring."""
import math

import torch


def fused_probability(sym_probability, quality_logit, alpha):
    """Use true endpoints, not a saturated learned-gate approximation."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0. <= alpha <= 1.:
        raise ValueError('alpha must be finite and in [0,1]')
    if sym_probability.shape != quality_logit.shape:
        raise ValueError('score shapes differ')
    base = torch.logit(sym_probability.clamp(1e-6, 1.-1e-6))
    return ((1.-alpha)*base + alpha*quality_logit).sigmoid()


def dgqp_joint_logit(cls_logit, distribution, reg_conf, reg_max=16,
                     reg_topk=4, add_mean=True):
    """Preserved GFocalV2 statistics/product, adapted to local logits API.

    The upstream GFocalHead emits probabilities; this MMDet GFLHead consumes
    logits in BOTH QFL and decoding. Convert the product once, preventing a
    second sigmoid of the already activated joint score.
    """
    n, _, h, w = distribution.shape
    prob = distribution.reshape(n, 4, reg_max+1, h, w).softmax(2)
    top = prob.topk(reg_topk, dim=2).values
    stat = torch.cat((top, top.mean(2, keepdim=True)), dim=2) if add_mean else top
    quality = reg_conf(stat.reshape(n, -1, h, w))
    joint = cls_logit.sigmoid() * quality
    # FP32 numerical guard only; no GT, ranking target, or score reweighting.
    return torch.logit(joint.clamp(1e-7, 1.-1e-7))
