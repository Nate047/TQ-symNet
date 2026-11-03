"""Box-local content readers. No MMDet, GT, box updates or score selection.

Boxes use normalized VALID-image coordinates; feature maps cover the PADDED
image. Multiplying by valid/batch scale exactly once handles non-square images.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def sample_points(feature, points, scale):
    """N x R x S x 2 points -> N x R x S x C, with explicit valid mask.

    Use the one shared image map (batch=1), NOT expand it per candidate. This
    avoids allocating/processing hundreds of duplicate image memories.
    """
    n, roles, slots = points.shape[:3]
    valid = ((points >= 0.) & (points <= 1.)).all(-1)
    grid = (points * scale).mul(2.).sub(1.).reshape(1, n * roles, slots, 2)
    sampled = F.grid_sample(feature, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    values = sampled[0].permute(1, 2, 0).reshape(n, roles, slots, -1)
    return values * valid[..., None], valid


def masked_softmax(logits, valid):
    """A wholly out-of-image role produces zero, never NaN or padding evidence."""
    weights = logits.masked_fill(~valid, -1e4).softmax(-1) * valid
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


class LocalRoleReader(nn.Module):
    """Preserve 5x5 ROI spatial cells until role-specific local attention.

    Independent local reads per P2..P5, then candidate/role-dependent scale
    fusion. The output seeds the EXISTING global/context QKV; it is not a
    replacement detector or a second quality objective.
    """
    def __init__(self, hidden, heads, pool_size=5):
        super().__init__()
        if pool_size < 3 or hidden % heads:
            raise ValueError('pool_size >= 3 and hidden divisible by heads required')
        self.hidden, self.heads, self.pool_size = hidden, heads, pool_size
        self.position = nn.Linear(2, hidden)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.query_norm, self.memory_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        # A single Linear([feature,query]) makes the query a common additive
        # term, which cancels in the across-level softmax. Nonlinearity couples
        # feature/query before scoring. A final shared scalar bias is redundant.
        self.level_gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                        nn.Linear(hidden, 1, bias=False))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))

    def forward(self, boxes, query, maps, scale):
        n, roles, c = query.shape
        axis = (torch.arange(self.pool_size, device=boxes.device,
                             dtype=boxes.dtype) + .5) / self.pool_size
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        relative = torch.stack((xx, yy), -1).reshape(-1, 2)
        size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-5)
        points = boxes[:, None, None, :2] + relative[None, None] * size[:, None, None]
        q = self.q(self.query_norm(query)).reshape(n, roles, self.heads, c // self.heads)
        reads = []
        for feature in maps:
            cells, valid = sample_points(feature, points, scale)
            cells = self.memory_norm(cells[:, 0] + self.position(relative * 2. - 1.)[None])
            k = self.k(cells).reshape(n, -1, self.heads, c // self.heads)
            v = self.v(cells).reshape(n, -1, self.heads, c // self.heads)
            logits = torch.einsum('nrhd,nshd->nrhs', q, k) / math.sqrt(c // self.heads)
            weights = masked_softmax(logits, valid[:, :, None])
            reads.append(torch.einsum('nrhs,nshd->nrhd', weights, v).reshape(n, roles, c))
        stack = torch.stack(reads, 2)
        weights = self.level_gate(torch.cat((stack, query[:, :, None].expand_as(stack)), -1))
        weights = weights.squeeze(-1).softmax(-1)
        content = (stack * weights[..., None]).sum(2)
        update = self.out(content)
        return query + update, update, weights


class PairedBoundaryReader(nn.Module):
    """Twelve explicit semantic pairs (edge x along slot), never RGB edges.

    Each pair samples a 3x3 patch on both sides, keeps sign/order, and injects
    different inside/outside messages into the corresponding 24 role queries.
    Sampling offsets are box-relative; no candidate coordinates are modified.
    """
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.level_gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                        nn.Linear(hidden, 1, bias=False))
        self.compare = nn.Sequential(nn.LayerNorm(4 * hidden + 8),
            nn.Linear(4 * hidden + 8, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, boxes, query, maps, scale):
        if query.shape[1] != 28:
            raise ValueError('Paired reader requires the retained 28-role layout')
        center = (boxes[:, :2] + boxes[:, 2:]) * .5
        size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-5)
        anchors = []
        for edge in range(4):
            axis, sign = edge % 2, (-1. if edge < 2 else 1.)
            for offset in (-.06, .06):
                for along in (-.30, 0., .30):
                    relative = boxes.new_zeros((len(boxes), 2))
                    relative[:, axis] = sign * (.5 + offset)
                    relative[:, 1-axis] = along
                    anchors.append(center + relative * size)
        anchors = torch.stack(anchors, 1)
        offsets = boxes.new_tensor([[x, y] for y in (-.025, 0., .025)
                                   for x in (-.025, 0., .025)])
        points = anchors[:, :, None] + offsets[None, None] * size[:, None, None]
        reads, validity = [], []
        for feature in maps:
            sampled, valid = sample_points(feature, points, scale)
            reads.append(sampled.sum(2) / valid.sum(2, keepdim=True).clamp_min(1))
            validity.append(valid.any(2))
        stack = torch.stack(reads, 2)
        level_weights = self.level_gate(torch.cat(
            (stack, query[:, 4:, None].expand_as(stack)), -1)).squeeze(-1).softmax(-1)
        edges = (stack * level_weights[..., None]).sum(2).reshape(len(boxes), 4, 2, 3, -1)
        inner, outer = edges[:, :, 0], edges[:, :, 1]
        geometry = torch.cat((boxes, center, size), 1)[:, None, None].expand(-1, 4, 3, -1)
        message = self.compare(torch.cat((inner, outer, inner - outer,
                                          inner * outer, geometry), -1))
        # N,edge,along,inside/outside,C -> N,edge,inside/outside,along,C.
        message = message.reshape(len(boxes), 4, 3, 2, -1).permute(0, 1, 3, 2, 4)
        # A pair clipped completely outside the valid image is not evidence.
        valid = validity[0].reshape(len(boxes), 4, 2, 3)
        valid_pair = (valid[:, :, 0] & valid[:, :, 1])[:, :, None, :, None]
        message = (message * valid_pair).reshape(len(boxes), 24, -1)
        result = torch.cat((query[:, :4], query[:, 4:] + message), 1)
        cosine = F.cosine_similarity(inner, outer, dim=-1)
        return result, message, cosine, valid_pair[:, :, 0, :, 0]
