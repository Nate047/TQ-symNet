"""Monotone threshold survival prediction, numerically stable in FP32.

Eleven ordered bins are delimited by .50,.55,...,.95. Ten survival probabilities
come from ONE normalized bin distribution. Their mean is the sole final quality
score, NOT ten scores passed to NMS. No GT is read by the head at inference.
"""
import torch
from torch import nn
from torch.nn import functional as F


def evaluation_thresholds(reference):
    return reference.new_tensor([.50 + .05 * k for k in range(10)])


def soft_threshold_targets(actual, temperature=.03):
    if temperature <= 0:
        raise ValueError('Threshold temperature must be positive')
    return ((actual.detach().float()[:, None] - evaluation_thresholds(actual.float()))
            / float(temperature)).sigmoid()


def distribution_predictions(bin_logits):
    if bin_logits.ndim != 2 or bin_logits.shape[1] != 11:
        raise ValueError('Expected eleven ordered bin logits per candidate')
    raw = bin_logits.float()
    # q_k = sum(p_{k+1:}); logit(q_k) is logsumexp(tail)-logsumexp(head).
    # Avoid sigmoid/softmax -> clamp -> logit, which loses gradients in tails.
    threshold_logits = torch.stack([
        torch.logsumexp(raw[:, k+1:], 1) - torch.logsumexp(raw[:, :k+1], 1)
        for k in range(10)], 1)
    # logit(mean(q_k)); factors 1/10 cancel in numerator/denominator.
    quality_logit = (torch.logsumexp(F.logsigmoid(threshold_logits), 1)
                     - torch.logsumexp(F.logsigmoid(-threshold_logits), 1))
    return quality_logit, threshold_logits


class ThresholdDistributionHead(nn.Linear):
    def __init__(self, hidden):
        super().__init__(hidden, 11)
        # Uniform bins -> mean survival=.5, matching original scalar initial q.
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def predict(self, descriptor):
        return distribution_predictions(F.linear(descriptor, self.weight, self.bias))

    def forward(self, descriptor):
        return self.predict(descriptor)[0].unsqueeze(1)


def distribution_element_loss(threshold_logits, actual, temperature=.03):
    if threshold_logits.shape != (len(actual), 10):
        raise ValueError('Threshold logits and supervised candidates are misaligned')
    target = soft_threshold_targets(actual, temperature)
    return F.binary_cross_entropy_with_logits(threshold_logits, target,
                                              reduction='none').mean(1)


def threshold_statistics(threshold_logits, actual, temperature=.03):
    """Six strata x [count, sum(pred), sum(soft), sum(hard), BCE, Brier]."""
    with torch.no_grad():
        p = threshold_logits.detach().float().sigmoid()
        actual = actual.detach().float()
        soft = soft_threshold_targets(actual, temperature)
        hard = (actual[:, None] >= evaluation_thresholds(actual)).float()
        bce = F.binary_cross_entropy_with_logits(threshold_logits.detach().float(),
                                                soft, reduction='none')
        masks = (torch.ones_like(actual, dtype=torch.bool), actual < .1,
                 (actual >= .1) & (actual < .5), (actual >= .5) & (actual < .75),
                 (actual >= .75) & (actual < .85), actual >= .85)
        return torch.stack([torch.cat((mask.sum().float().reshape(1), p[mask].sum(0),
            soft[mask].sum(0), hard[mask].sum(0), bce[mask].sum(0),
            (p[mask] - hard[mask]).square().sum(0))) for mask in masks])


def format_threshold_statistics(stats):
    names = ('all', 'iou_lt_10', 'iou_10_50', 'iou_50_75', 'iou_75_85', 'iou_ge_85')
    rows = {}
    for name, values in zip(names, stats.cpu().tolist()):
        n, data = int(values[0]), values[1:]
        rows[name] = dict(count=n,
            predicted=[x / max(n, 1) for x in data[:10]],
            soft_target=[x / max(n, 1) for x in data[10:20]],
            actual_pass_rate=[x / max(n, 1) for x in data[20:30]],
            soft_bce=[x / max(n, 1) for x in data[30:40]],
            hard_brier=[x / max(n, 1) for x in data[40:50]])
    return rows
