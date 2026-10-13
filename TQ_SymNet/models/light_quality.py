"""Simple shared-feature IoU scorer for the Q0 strong control."""
import torch
from torch import nn
from torch.nn import functional as F


class PooledQualityScorer(nn.Module):
    def __init__(self, hidden=96, chunk=32, pool_size=5):
        super().__init__()
        self.hidden, self.chunk, self.pool_size = hidden, chunk, pool_size
        self.heads = 6
        self.texture = self.lbp_enabled = self.gabor_enabled = False
        self.gaussian = self.edges = self.detail = self.checkpoint_candidates = False
        self.attention_stats = []
        self.visual_projection = nn.ModuleList([nn.Conv2d(256, hidden, 1) for _ in range(4)])
        self.quality = nn.Sequential(nn.LayerNorm(4*hidden+8),
            nn.Linear(4*hidden+8, 2*hidden), nn.SiLU(),
            nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    @property
    def high_head(self):
        return self.quality[-1]

    def prepare(self, features, gray, valid_shape, batch_shape):
        scale = features[0].new_tensor((valid_shape[1]/batch_shape[1],
                                       valid_shape[0]/batch_shape[0]))
        return [(projection(feature), scale) for projection, feature
                in zip(self.visual_projection, features[:4])]

    def forward(self, boxes, memories, sym, training_checkpoint=False):
        if not len(boxes):
            return boxes.new_empty((0, 2)), boxes.new_empty((0, 7)), boxes.new_empty((0, 3))
        axis = (torch.arange(self.pool_size, device=boxes.device, dtype=boxes.dtype)+.5)/self.pool_size
        yy, xx = torch.meshgrid(axis, axis)
        unit = torch.stack((xx, yy), -1)
        values = []
        for part in boxes.split(self.chunk):
            center = (part[:, :2]+part[:, 2:])*.5
            size = (part[:, 2:]-part[:, :2]).clamp_min(1e-5)
            points = part[:, None, None, :2] + unit[None] * size[:, None, None]
            pooled = []
            for feature, scale in memories:
                grid = (points * scale).mul(2).sub(1)
                sampled = F.grid_sample(feature.expand(len(part), -1, -1, -1),
                    grid, mode='bilinear', padding_mode='border', align_corners=False)
                pooled.append(sampled.mean((2, 3)))
            descriptor = torch.cat(pooled + [part, center, size], 1)
            values.append(self.quality(descriptor).squeeze(1))
        logit = torch.cat(values)
        gate = boxes.new_zeros((len(boxes), 7))
        gate[:, 0] = 1.; gate[:, 3:] = .25
        return torch.stack((logit, logit), 1), gate, boxes.new_zeros((len(boxes), 3))
