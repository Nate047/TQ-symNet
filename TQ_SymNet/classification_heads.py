"""Image classification read-outs for a frozen, detector-trained CrossScale pyramid."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class _PerPositionLayerNorm(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, feature):
        return self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CrossScaleHybridPoolHead(nn.Module):
    """Small control: GAP and GMP on each of P3, P4 and P5."""

    def __init__(self, in_channels=256, hidden=512, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(6 * in_channels),
            nn.Linear(6 * in_channels, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.last_debug = {}
        self.collect_debug = False

    def forward(self, pyramid):
        if len(pyramid) != 4:
            raise ValueError('Expected frozen CrossScale P2/P3/P4/P5')
        levels = pyramid[1:]
        representation = torch.cat(
            [pool(x, 1).flatten(1) for x in levels
             for pool in (F.adaptive_avg_pool2d, F.adaptive_max_pool2d)], dim=1)
        self.last_debug = {'pooled_levels': 3} if self.collect_debug else {}
        return self.classifier(representation)


class CrossScaleClassQueryHead(nn.Module):
    """Read each level with three class queries, then retain global context."""

    def __init__(self, in_channels=256, embed_dim=128, heads=4,
                 scale_fusion='fixed', dropout=0.1,
                 attention_temperature=False):
        super().__init__()
        if embed_dim <= 0 or embed_dim % heads or embed_dim % 4:
            raise ValueError('embed_dim must be positive and divisible by heads and four')
        if scale_fusion not in ('fixed', 'learned'):
            raise ValueError('scale_fusion must be fixed or learned')
        self.scale_fusion = scale_fusion
        self.embed_dim = embed_dim
        self.heads = heads
        self.attention_temperature = bool(attention_temperature)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, embed_dim, 1),
                          _PerPositionLayerNorm(embed_dim)) for _ in range(3)])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=0., batch_first=True)
            for _ in range(3)])
        self.queries = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.global_projection = nn.Linear(in_channels, embed_dim)
        self.readout = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, 1))
        # Shared modules are initialized first. Optional gate initialization
        # must not shift later dropout/RNG draws in the fixed-vs-learned run.
        if scale_fusion == 'learned':
            with torch.random.fork_rng(devices=[]):
                self.level_embeddings = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
                self.scale_gate = nn.Sequential(
                    nn.LayerNorm(3 * embed_dim),
                    nn.Linear(3 * embed_dim, embed_dim), nn.GELU(),
                    nn.Linear(embed_dim, 1))
                # Uniform initial gate gives the same output as the control.
                nn.init.zeros_(self.scale_gate[-1].weight)
                nn.init.zeros_(self.scale_gate[-1].bias)
        else:
            self.register_parameter('level_embeddings', None)
            self.scale_gate = None
        if self.attention_temperature:
            # Optional parameters must not advance the RNG shared with C3.
            with torch.random.fork_rng(devices=[]):
                self.temperature_head = nn.Sequential(
                    nn.LayerNorm(2 * embed_dim),
                    nn.Linear(2 * embed_dim, embed_dim // 2), nn.GELU(),
                    nn.Linear(embed_dim // 2, 1))
                nn.init.zeros_(self.temperature_head[-1].weight)
                nn.init.zeros_(self.temperature_head[-1].bias)
        else:
            self.temperature_head = None
        self.last_debug = {}
        self.collect_debug = False

    @staticmethod
    def _position(h, w, channels, device, dtype):
        # Position is computed from the current feature shape, never a fixed
        # training-grid lookup. Each coordinate gets sine and cosine channels.
        y = torch.linspace(0., 1., h, device=device, dtype=dtype)
        x = torch.linspace(0., 1., w, device=device, dtype=dtype)
        # The explicit indexing argument is unavailable in older server Torch.
        try:
            yy, xx = torch.meshgrid(y, x, indexing='ij')
        except TypeError:
            yy, xx = torch.meshgrid(y, x)
        frequencies = torch.arange(channels // 4, device=device, dtype=dtype)
        frequencies = 2. * math.pi * torch.exp(-math.log(10000.) * frequencies
                                              / max(channels // 4 - 1, 1))
        return torch.cat([torch.sin(xx[..., None] * frequencies),
                          torch.cos(xx[..., None] * frequencies),
                          torch.sin(yy[..., None] * frequencies),
                          torch.cos(yy[..., None] * frequencies)], dim=-1).reshape(1, h * w, channels)

    def _temperature_attention(self, layer, query, keys, values, temperature):
        """MHA with image/class-specific temperature on projected QK logits.

        Scaling input query before nn.MultiheadAttention would not scale its
        learned projection bias, so it would not implement QK/temperature.
        Reuse the same layer weights and scale the actual attention logits.
        """
        batch, classes, dimension = query.shape
        locations = keys.shape[1]
        head_dim = dimension // self.heads
        wq, wk, wv = layer.in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = layer.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, wq, bq).reshape(batch, classes, self.heads, head_dim)
        k = F.linear(keys, wk, bk).reshape(batch, locations, self.heads, head_dim)
        v = F.linear(values, wv, bv).reshape(batch, locations, self.heads, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores / temperature[:, None, :, None]
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).permute(0, 2, 1, 3)
        attended = attended.reshape(batch, classes, dimension)
        output = F.linear(attended, layer.out_proj.weight, layer.out_proj.bias)
        return output, weights.mean(1) if self.collect_debug else None

    def forward(self, pyramid):
        if len(pyramid) != 4:
            raise ValueError('Expected frozen CrossScale P2/P3/P4/P5')
        selected = pyramid[1:]
        batch = selected[0].shape[0]
        global_feature = self.global_projection(selected[-1].mean(dim=(2, 3)))
        query = self.query_norm(self.queries)[None].expand(batch, -1, -1)
        if self.temperature_head is not None:
            context = global_feature[:, None].expand_as(query)
            raw_temperature = self.temperature_head(
                torch.cat([query, context], dim=-1)).squeeze(-1)
            temperature = torch.exp(torch.tanh(raw_temperature))
        else:
            temperature = None
        responses, entropies = [], []
        for level, (feature, projection, attention) in enumerate(
                zip(selected, self.projections, self.attentions)):
            if feature.shape[0] != batch:
                raise ValueError('Mismatched CrossScale batch dimensions')
            projected = projection(feature)
            h, w = projected.shape[-2:]
            tokens = projected.flatten(2).transpose(1, 2)
            keys = tokens + self._position(h, w, self.embed_dim,
                                           tokens.device, tokens.dtype)
            if temperature is None:
                response, weights = attention(query, keys, tokens,
                                              need_weights=self.collect_debug)
            else:
                response, weights = self._temperature_attention(
                    attention, query, keys, tokens, temperature)
            responses.append(response)
            if self.collect_debug:
                with torch.no_grad():
                    weights = weights.detach().float().clamp_min(1e-8)
                    entropies.append((-(weights * weights.log()).sum(-1)
                                      / math.log(max(h * w, 2))).mean(0).cpu().tolist())
        stacked = torch.stack(responses, dim=2)  # B, three classes, three levels, D
        if self.scale_gate is None:
            gate = stacked.new_full(stacked.shape[:3], 1. / 3.)
        else:
            context = global_feature[:, None, None].expand_as(stacked)
            # Scale identity belongs in the gate. Adding a constant vector to
            # every spatial key cancels in softmax and cannot identify a level.
            level_identity = self.level_embeddings[None, None].expand_as(stacked)
            gate_input = torch.cat([stacked, context, level_identity], -1)
            gate = torch.softmax(self.scale_gate(gate_input).squeeze(-1), -1)
        local = (stacked * gate[..., None]).sum(dim=2)
        global_for_class = global_feature[:, None].expand(-1, 3, -1)
        logits = self.readout(torch.cat([local, global_for_class], dim=-1)).squeeze(-1)
        self.last_debug = (
            {'scale_gate_by_class': gate.detach().mean(0).cpu().tolist(),
             'attention_entropy_by_level': entropies,
             **({'attention_temperature_by_class':
                 temperature.detach().mean(0).cpu().tolist()}
                if temperature is not None else {})}
            if self.collect_debug else {})
        return logits
