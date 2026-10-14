"""Classification-only priors. No detector weights, boxes, masks or subtype labels.

The fresh symmetry module is a mirror-neighbourhood attention adaptation, NOT
the released SymFormer's deformable CUDA operator. Gabor energy is inspired by
IJCV 2026, doi:10.1007/s11263-025-02658-2; learnable filters and query fusion are
our adaptation. LBP uses the 58 uniform 8-bit patterns plus one nonuniform bin.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .classification_heads import CrossScaleClassQueryHead, _PerPositionLayerNorm


class MirrorNeighbourAttention(nn.Module):
    """Read a 3x3 neighbourhood around the horizontal mirror, not equality loss."""
    def __init__(self, channels, dimension):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(channels, dimension, 1),
                                     _PerPositionLayerNorm(dimension))
        self.q = nn.Conv2d(dimension, dimension, 1)
        self.k = nn.Conv2d(dimension, dimension, 1)
        self.v = nn.Conv2d(dimension, dimension, 1)
        self.relation = nn.Sequential(nn.Conv2d(4 * dimension, dimension, 1),
                                      _PerPositionLayerNorm(dimension), nn.GELU())

    def forward(self, feature):
        local = self.project(feature)
        opposite = local.flip(-1)
        b, d, h, w = local.shape
        # Replicate padding avoids treating a padded zero token as contralateral tissue.
        keys = F.unfold(F.pad(self.k(opposite), (1, 1, 1, 1), mode='replicate'), 3)
        values = F.unfold(F.pad(self.v(opposite), (1, 1, 1, 1), mode='replicate'), 3)
        keys, values = keys.reshape(b, d, 9, h*w), values.reshape(b, d, 9, h*w)
        weights = ((self.q(local).flatten(2).unsqueeze(2) * keys).sum(1)
                   / math.sqrt(d)).softmax(1)
        opposite = (values * weights[:, None]).sum(2).reshape(b, d, h, w)
        delta = local - opposite
        return self.relation(torch.cat([local, opposite, delta, delta.abs()], 1))


class UniformLBP(nn.Module):
    """Categorical lookup -> spatial embedding; NEVER interpolate ordinal codes."""
    def __init__(self, dimension):
        super().__init__()
        mapping, next_bin = [], 0
        for code in range(256):
            bits = [(code >> bit) & 1 for bit in range(8)]
            transitions = sum(bits[i] != bits[(i+1) % 8] for i in range(8))
            if transitions <= 2:
                mapping.append(next_bin)
                next_bin += 1
            else:
                mapping.append(58)
        assert next_bin == 58
        self.register_buffer('mapping', torch.tensor(mapping, dtype=torch.long))
        self.embedding = nn.Embedding(59, dimension)
        self.bin_logits = nn.Parameter(torch.zeros(59))
        self.norm = _PerPositionLayerNorm(dimension)

    def forward(self, gray, grid):
        with torch.no_grad():
            h, w = gray.shape[-2:]
            padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
            code = torch.zeros_like(gray[:, 0], dtype=torch.long)
            for bit, (y, x) in enumerate(((0, 0), (0, 1), (0, 2), (1, 2),
                                          (2, 2), (2, 1), (2, 0), (1, 0))):
                code += (padded[:, 0, y:y+h, x:x+w] >= gray[:, 0]).long() << bit
            bins = self.mapping[code]
        # Pool sparse bin occupancies first: same as pooling full-resolution
        # embeddings, but avoids B x H x W x D activation materialization.
        counts = torch.cat([F.adaptive_avg_pool2d((bins == i).float()[:, None], grid)
                            for i in range(59)], 1)
        weights = self.embedding.weight * self.bin_logits.sigmoid()[:, None]
        result = F.conv2d(counts, weights.t()[:, :, None, None])
        # Apply gates before mixing bins, not before a per-bin normalization
        # that would cancel the gates.
        return self.norm(result)


class GaborEnergy(nn.Module):
    def __init__(self, dimension, filters=8, kernel_size=15):
        super().__init__()
        self.theta = nn.Parameter(torch.arange(filters).float() * math.pi / filters)
        # Bounded wavelengths [3,12] and widths [1.5,5] pixels. No unbounded
        # negative widths or frequencies above Nyquist during optimization.
        self.wave_raw = nn.Parameter(torch.linspace(-1., 1., filters))
        self.sigma_raw = nn.Parameter(torch.zeros(filters))
        axis = torch.arange(kernel_size).float() - kernel_size // 2
        self.register_buffer('yy', axis[:, None].expand(kernel_size, kernel_size).clone())
        self.register_buffer('xx', axis[None, :].expand(kernel_size, kernel_size).clone())
        self.project = nn.Sequential(nn.Conv2d(filters, dimension, 1),
                                     _PerPositionLayerNorm(dimension))
        self.padding = kernel_size // 2

    def kernels(self):
        theta = self.theta[:, None, None]
        x = self.xx * theta.cos() + self.yy * theta.sin()
        y = -self.xx * theta.sin() + self.yy * theta.cos()
        wave = (3. + 9. * self.wave_raw.sigmoid())[:, None, None]
        sigma = (1.5 + 3.5 * self.sigma_raw.sigmoid())[:, None, None]
        envelope = torch.exp(-(x*x + y*y) / (2. * sigma*sigma))
        phase = 2. * math.pi * x / wave
        pair = torch.stack([envelope * phase.cos(), envelope * phase.sin()], 1)
        pair = pair - pair.mean((-2, -1), keepdim=True)
        pair = pair / pair.square().sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        return pair.flatten(0, 1)[:, None]

    def forward(self, gray, grid):
        response = F.conv2d(F.pad(gray, (self.padding,)*4, mode='reflect'), self.kernels())
        b, _, h, w = response.shape
        response = response.reshape(b, len(self.theta), 2, h, w)
        energy = (response.square().sum(2) + 1e-8).sqrt()
        # Average energy BEFORE spatial decimation; no full-res semantic CNN.
        energy = F.adaptive_avg_pool2d(energy, grid)
        return self.project(torch.log1p(energy))


class PriorClassQueryHead(CrossScaleClassQueryHead):
    uses_images = True

    def __init__(self, use_sym=False, use_lbp=False, use_gabor=False,
                 texture_grid=32, image_mean=(123.675, 116.28, 103.53),
                 image_std=(58.395, 57.12, 57.375), **kwargs):
        super().__init__(**kwargs)
        self.use_sym, self.use_lbp, self.use_gabor = use_sym, use_lbp, use_gabor
        self.texture_grid = int(texture_grid)
        if self.texture_grid <= 0:
            raise ValueError('texture_grid must be positive')
        d, channels = self.embed_dim, kwargs.get('in_channels', 256)
        self.branch_names = ['semantic'] + (['sym'] if use_sym else []) + (
            ['lbp'] if use_lbp else []) + (['gabor'] if use_gabor else [])
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        # Each optional module leaves the baseline RNG unchanged, including
        # later dropout draws. Shared modules start identically across ablations.
        with torch.random.fork_rng(devices=[]):
            self.visual_query = nn.LayerNorm(d)
            self.branch_gate = nn.Sequential(nn.LayerNorm(3*d), nn.Linear(3*d, d),
                                             nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.branch_gate[-1].weight)
            nn.init.zeros_(self.branch_gate[-1].bias)
        with torch.random.fork_rng(devices=[]):
            if use_sym:
                self.sym = nn.ModuleList([MirrorNeighbourAttention(channels, d) for _ in range(3)])
                self.sym_read = nn.ModuleList([nn.MultiheadAttention(d, self.heads, dropout=0.,
                                               batch_first=True) for _ in range(3)])
                self.sym_level_gate = nn.Linear(2*d, 1)
        with torch.random.fork_rng(devices=[]):
            if use_lbp:
                self.lbp = UniformLBP(d)
                self.lbp_read = nn.MultiheadAttention(d, self.heads, dropout=0., batch_first=True)
        with torch.random.fork_rng(devices=[]):
            if use_gabor:
                self.gabor = GaborEnergy(d)
                self.gabor_read = nn.MultiheadAttention(d, self.heads, dropout=0., batch_first=True)

    def read(self, query, feature, attention):
        h, w = feature.shape[-2:]
        tokens = feature.flatten(2).transpose(1, 2)
        keys = tokens + self._position(h, w, self.embed_dim, tokens.device, tokens.dtype)
        response, weights = attention(query, keys, tokens, need_weights=self.collect_debug)
        entropy = None
        if self.collect_debug:
            weights = weights.detach().clamp_min(1e-8)
            entropy = (-(weights * weights.log()).sum(-1) / math.log(max(h*w, 2))).mean(0)
        return response, entropy

    def enhance_local(self, local, global_feature, pyramid, images):
        query = self.visual_query(local + self.queries[None])
        responses, debug = [local], {}
        if self.use_sym:
            levels = [self.read(query, module(feature), reader)[0]
                      for feature, module, reader in zip(pyramid[1:], self.sym, self.sym_read)]
            stacked = torch.stack(levels, 2)
            context = global_feature[:, None, None].expand_as(stacked)
            level_gate = self.sym_level_gate(torch.cat([stacked, context], -1)).squeeze(-1).softmax(-1)
            responses.append((stacked * level_gate[..., None]).sum(2))
            if self.collect_debug:
                debug['sym_level_gate_by_class'] = level_gate.detach().mean(0).cpu().tolist()
        if self.use_lbp or self.use_gabor:
            if images is None:
                raise ValueError('Texture branches require normalized classification images')
            # Invert ImageNet normalization before extracting grayscale priors.
            rgb = (images.detach().float() * self.image_std + self.image_mean) / 255.
            gray = (rgb[:, 0:1] * .299 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114).clamp(0, 1)
            grid = (min(self.texture_grid, gray.shape[-2]), min(self.texture_grid, gray.shape[-1]))
            for name, enabled in (('lbp', self.use_lbp), ('gabor', self.use_gabor)):
                if enabled:
                    response, entropy = self.read(query, getattr(self, name)(gray, grid),
                                                   getattr(self, name + '_read'))
                    responses.append(response)
                    if self.collect_debug:
                        debug[name + '_attention_entropy'] = entropy.cpu().tolist()
        stacked = torch.stack(responses, 2)
        semantic = local[:, :, None].expand_as(stacked)
        context = global_feature[:, None, None].expand_as(stacked)
        gate_logits = self.branch_gate(torch.cat([stacked, semantic, context], -1)).squeeze(-1)
        # Initial semantic share 0.8, other branches total 0.2, all trainable
        # immediately. Not an identity initialization; C3 remains separate.
        prior = stacked.new_zeros(len(responses))
        if len(responses) > 1:
            prior[0] = math.log(4. * (len(responses)-1))
        gate = (gate_logits + prior).softmax(-1)
        if self.collect_debug:
            debug['prior_gate_by_class'] = gate.detach().mean(0).cpu().tolist()
            debug['prior_response_norm_by_class'] = stacked.detach().norm(dim=-1).mean(0).cpu().tolist()
            if self.use_lbp:
                debug['lbp_bin_gate'] = self.lbp.bin_logits.detach().sigmoid().cpu().tolist()
            if self.use_gabor:
                debug['gabor_theta'] = self.gabor.theta.detach().cpu().tolist()
                debug['gabor_wavelength'] = (3+9*self.gabor.wave_raw.detach().sigmoid()).cpu().tolist()
        return (stacked * gate[..., None]).sum(2), debug
