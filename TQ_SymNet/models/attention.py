"""Whole-image visual/LBP/Gabor attention for candidate localization quality.

No detector registrations or external CUDA extensions: CPU-testable math.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def grid_positions(h, w, device, dtype, extent=(1., 1.)):
    y = (torch.arange(h, device=device, dtype=dtype) + .5) / h * extent[0]
    x = (torch.arange(w, device=device, dtype=dtype) + .5) / w * extent[1]
    yy, xx = torch.meshgrid(y, x)
    return torch.stack((xx, yy), -1).reshape(-1, 2)


def gaussian_bias(boxes, positions, strength):
    """Center-radiating Gaussian, soft boundary ring, broad context, global.

    Both boxes and positions use the SAME normalized valid-image coordinates.
    No GT-derived masks. Global query is not constrained to lesion center.
    """
    center = (boxes[:, :2] + boxes[:, 2:]) * .5
    size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-4)
    delta = (positions[None] - center[:, None]) / size[:, None]
    radius = delta.square().sum(-1)
    core = -.5 * radius / (.35 ** 2)
    broad = -.5 * radius / (.85 ** 2)
    ring = (broad.exp() - core.exp()).clamp_min(1e-8).log()
    bias = torch.stack((core, ring, broad, torch.zeros_like(core)), 1).clamp_min(-20.)
    bias = bias * strength
    valid = ((positions >= 0.) & (positions <= 1.)).all(1)
    return bias.masked_fill(~valid[None, None], -1e4)


class LearnableLBP(nn.Module):
    def __init__(self, embedding=8):
        super().__init__()
        mapping = torch.full((256,), 58, dtype=torch.long)
        uniform = []
        for code in range(256):
            bits = [(code >> bit) & 1 for bit in range(8)]
            if sum(bits[i] != bits[(i+1) % 8] for i in range(8)) <= 2:
                uniform.append(code)
        for index, code in enumerate(uniform):
            mapping[code] = index
        self.register_buffer('mapping', mapping)
        self.embedding = nn.Embedding(59, embedding)
        self.gate = nn.Parameter(torch.zeros(59))

    def forward(self, gray):
        h, w = gray.shape[-2:]
        padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        code = torch.zeros_like(gray, dtype=torch.long)
        for bit, (dy, dx) in enumerate(((-1,-1),(-1,0),(-1,1),(0,1),
                                       (1,1),(1,0),(1,-1),(0,-1))):
            neighbor = padded[:, :, 1+dy:1+dy+h, 1+dx:1+dx+w]
            code = code | ((neighbor >= gray).long() << bit)
        bins = self.mapping[code[:, 0]]
        value = self.embedding(bins) * self.gate.sigmoid()[bins, None]
        return value.permute(0, 3, 1, 2).contiguous()


class LearnableGabor(nn.Module):
    def __init__(self, count=8, kernel=9):
        super().__init__()
        self.count, self.kernel = count, kernel
        self.theta = nn.Parameter(torch.arange(count).float() * math.pi / count)
        self.log_wavelength = nn.Parameter(torch.full((count,), math.log(6.)))
        self.log_sigma = nn.Parameter(torch.full((count,), math.log(3.)))

    def forward(self, gray):
        axis = torch.arange(-(self.kernel//2), self.kernel//2+1,
                            device=gray.device, dtype=gray.dtype)
        yy, xx = torch.meshgrid(axis, axis)
        theta = self.theta[:, None, None]
        xr = xx[None] * theta.cos() + yy[None] * theta.sin()
        yr = -xx[None] * theta.sin() + yy[None] * theta.cos()
        wavelength = self.log_wavelength.exp().clamp(2., 16.)[:, None, None]
        sigma = self.log_sigma.exp().clamp(1., 8.)[:, None, None]
        kernel = torch.exp(-(xr.square()+yr.square())/(2*sigma.square())) * torch.cos(2*math.pi*xr/wavelength)
        kernel = kernel - kernel.mean((1, 2), keepdim=True)
        kernel = kernel / kernel.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-6)
        return F.conv2d(gray, kernel[:, None], padding=self.kernel//2).abs()


def texture_encoder(channels, hidden):
    return nn.Sequential(nn.Conv2d(channels, hidden//2, 3, stride=2, padding=1),
        nn.GroupNorm(4, hidden//2), nn.SiLU(),
        nn.Conv2d(hidden//2, hidden, 3, stride=2, padding=1),
        nn.GroupNorm(4, hidden), nn.SiLU())


class CandidateQualityAttention(nn.Module):
    def __init__(self, channels=256, hidden=64, heads=4, texture=True,
                 gaussian=True, chunk=32, checkpoint_candidates=True,
                 lbp_enabled=True, gabor_enabled=True):
        super().__init__()
        if hidden % heads or hidden % 8 or chunk < 1:
            raise ValueError('hidden must divide heads/8; chunk must be positive')
        self.hidden, self.heads = hidden, heads
        self.texture, self.gaussian = bool(texture), bool(gaussian)
        self.lbp_enabled = bool(lbp_enabled) and self.texture
        self.gabor_enabled = bool(gabor_enabled) and self.texture
        self.chunk, self.checkpoint_candidates = int(chunk), bool(checkpoint_candidates)
        self.visual_projection = nn.ModuleList([nn.Conv2d(channels, hidden, 1) for _ in range(4)])
        self.level_embedding = nn.Parameter(torch.randn(4, hidden) * .02)
        self.position = nn.Linear(6, hidden)
        self.geometry = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.query_tokens = nn.Parameter(torch.randn(4, hidden) * .02)
        # Four visual levels remain separate until candidate-conditioned level
        # fusion. Texture memories are the final two entries.
        self.memory_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(6)])
        self.query_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(6)])
        self.q_projection = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(6)])
        self.k_projection = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(6)])
        self.v_projection = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(6)])
        self.output_projection = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(6)])
        self.visual_level_gate = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, 4))
        with torch.random.fork_rng(devices=[]):
            self.lbp = LearnableLBP() if texture else None
            self.gabor = LearnableGabor() if texture else None
            self.lbp_encoder = texture_encoder(8, hidden) if texture else None
            self.gabor_encoder = texture_encoder(8, hidden) if texture else None
        self.fusion_gate = nn.Sequential(nn.Linear(hidden*4, hidden*2), nn.SiLU(),
                                         nn.Linear(hidden*2, 3))
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.zeros_(self.fusion_gate[-1].bias)
        self.quality = nn.Sequential(nn.LayerNorm(hidden*5),
            nn.Linear(hidden*5, hidden*2), nn.SiLU(),
            nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.gaussian_raw = nn.Parameter(torch.zeros(3))

    def _memory(self, feature, positions, branch):
        xy = positions
        position = torch.cat((xy, (xy*math.pi).sin(), (xy*math.pi).cos()), 1)
        memory = self.memory_norm[branch](feature + self.position(position))
        return self.k_projection[branch](memory), self.v_projection[branch](memory), positions

    def prepare(self, features, gray, valid_shape, batch_shape):
        """Projection/texture extraction once per image, reused by all boxes."""
        result = []
        extent = (float(batch_shape[0])/valid_shape[0], float(batch_shape[1])/valid_shape[1])
        for level, feature in enumerate(features[:4]):
            value = self.visual_projection[level](feature)[0]
            h, w = value.shape[-2:]
            position = grid_positions(h, w, value.device, value.dtype, extent)
            result.append(self._memory(value.flatten(1).t() +
                                       self.level_embedding[level], position, level))
        if self.texture:
            # Keep fixed memory slots and parameter names across all ablations.
            for branch, enabled, extractor in (
                    (4, self.lbp_enabled, lambda: self.lbp_encoder(self.lbp(gray))),
                    (5, self.gabor_enabled, lambda: self.gabor_encoder(self.gabor(gray)))):
                if not enabled:
                    # Tensor placeholders work with legacy reentrant checkpointing.
                    empty = gray.new_empty((0, self.hidden))
                    result.append((empty, empty, gray.new_empty((0, 2))))
                    continue
                value = extractor()[0]
                h, w = value.shape[-2:]
                result.append(self._memory(value.flatten(1).t(),
                    grid_positions(h, w, value.device, value.dtype), branch))
        return result

    def _attention(self, query, boxes, key, value, positions, projection,
                   gaussian_branch):
        n, queries, hidden = query.shape
        head_dim = hidden // self.heads
        # Visual levels have independent Q/K/V projections. LBP and Gabor do
        # too; no level is compressed into a shared token pool beforehand.
        q = self.q_projection[projection](self.query_norm[projection](query)).reshape(n, queries, self.heads, head_dim)
        k, v = key.reshape(-1, self.heads, head_dim), value.reshape(-1, self.heads, head_dim)
        similarity = torch.einsum('nqhd,thd->nhqt', q, k) / math.sqrt(head_dim)
        strength = (2. * self.gaussian_raw[gaussian_branch].sigmoid()) if self.gaussian else query.new_tensor(0.)
        bias = gaussian_bias(boxes, positions, strength)
        attention = (similarity + bias[:, None]).softmax(-1)
        selected = torch.einsum('nhqt,thd->nqhd', attention, v).reshape(n, queries, hidden)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean((1, 2))
        return query + self.output_projection[projection](selected), entropy

    def _score(self, boxes, *memories):
        center, size = (boxes[:, :2]+boxes[:, 2:])*.5, boxes[:, 2:]-boxes[:, :2]
        geometry = self.geometry(torch.cat((boxes, center, size), 1))
        query = geometry[:, None] + self.query_tokens[None]
        visual_levels, visual_entropy = [], []
        offset = 0
        for level in range(4):
            memory = memories[offset:offset+3]; offset += 3
            selected, entropy = self._attention(
                query, boxes, *memory, projection=level, gaussian_branch=0)
            visual_levels.append(selected); visual_entropy.append(entropy)
        visual_stack = torch.stack(visual_levels, 2)  # N, roles, levels, C
        gate_input = torch.cat(
            [value.mean(1) for value in visual_levels] + [geometry], 1)
        level_gate = self.visual_level_gate(gate_input).softmax(1)
        # Preserve core/ring/context/global role tokens; only fuse the level
        # axis with candidate-conditioned weights.
        visual = (visual_stack * level_gate[:, None, :, None]).sum(2)
        if self.texture:
            # Both texture routes query the SAME visual-conditioned descriptor.
            e0 = torch.stack(visual_entropy, 1).mean(1)
            if self.lbp_enabled:
                lbp, e1 = self._attention(visual, boxes, *memories[offset:offset+3],
                                          projection=4, gaussian_branch=1)
            else:
                lbp, e1 = visual, e0 * 0.
            offset += 3
            if self.gabor_enabled:
                gabor, e2 = self._attention(visual, boxes, *memories[offset:offset+3],
                                            projection=5, gaussian_branch=2)
            else:
                gabor, e2 = visual, e0 * 0.
            gate_logits = self.fusion_gate(torch.cat(
                (visual.mean(1), lbp.mean(1), gabor.mean(1), geometry), 1))
            if not self.lbp_enabled:
                gate_logits[:, 1] = -1e4
            if not self.gabor_enabled:
                gate_logits[:, 2] = -1e4
            gate = gate_logits.softmax(1)
            fused = visual + gate[:, 1, None, None] * (lbp-visual) + gate[:, 2, None, None] * (gabor-visual)
            entropy = torch.stack((torch.stack(visual_entropy, 1).mean(1),
                                   e1, e2), 1)
        else:
            fused = visual
            gate = boxes.new_zeros((len(boxes), 3)); gate[:, 0] = 1.
            e0 = torch.stack(visual_entropy, 1).mean(1)
            entropy = torch.stack((e0, e0*0., e0*0.), 1)
        logit = self.quality(torch.cat((fused.flatten(1), geometry), 1)).squeeze(1)
        diagnostics = torch.cat((gate, level_gate), 1)
        return logit, diagnostics, entropy

    def forward(self, boxes, memories, training_checkpoint=False):
        if not len(boxes):
            return (boxes.new_empty((0,)), boxes.new_empty((0, 7)),
                    boxes.new_empty((0, 3)))
        inputs = tuple(value for memory in memories for value in memory)
        outputs = []
        for chunk in boxes.split(self.chunk):
            if training_checkpoint and self.checkpoint_candidates:
                outputs.append(checkpoint(self._score, chunk, *inputs))
            else:
                outputs.append(self._score(chunk, *inputs))
        return tuple(torch.cat([result[i] for result in outputs]) for i in range(3))


def competition_rank(logits, quality, matched, boxes, min_gap=.05, max_pairs=128, overlap_threshold=.5):
    """Finite-margin ranking for candidate pairs that can compete in NMS."""
    pair_overlap = pairwise_box_iou(boxes)
    losses, count = [], 0
    for gt_index in matched[quality >= .50].unique():
        positive = (matched == gt_index) & (quality >= .50)
        gap = quality[:, None] - quality[None]
        difference = logits[:, None] - logits[None]
        # Positives for different GT must not be trained to suppress each other.
        right = (matched == gt_index) | (quality < .50)
        valid = positive[:, None] & right[None] & (gap > min_gap) & (pair_overlap > overlap_threshold)
        # Margin vanishes with quality gap. Avoid a fixed minimum separation
        # that conflicts with calibrated, nearly equal IoU targets.
        violation = F.relu(.50 * gap - difference)
        values = violation[valid]
        if len(values):
            values = values.topk(min(max_pairs, len(values))).values
            losses.append(values.mean()); count += len(values)
    return torch.stack(losses).mean() if losses else logits.sum()*0., count


def pairwise_box_iou(boxes):
    lt, rb = torch.maximum(boxes[:, None, :2], boxes[None, :, :2]), torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (rb-lt).clamp_min(0.).prod(-1)
    area = (boxes[:, 2:]-boxes[:, :2]).clamp_min(0.).prod(-1)
    return intersection/(area[:, None]+area[None]-intersection).clamp_min(1e-8)
