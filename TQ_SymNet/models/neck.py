"""TQ-SymNet: extracted measured implementations; see PROVENANCE.json."""
import math
import logging
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.utils.deterministic_ops import deterministic_bilinear_interpolate
from mmdet.models.builder import NECKS
from .checkpoint_compat import UnsupportedBranch
_FreqFusion2x = _C5GatedC2RegressionDetail = UnsupportedBranch

class _LayerNorm2d(nn.Module):
    """LayerNorm over channels for 2D feature maps.

    nn.LayerNorm(C) on (B,C,H,W) input is only supported in PyTorch >= 1.10.
    This wrapper permutes to (B,H,W,C), applies LN, and permutes back,
    providing identical behaviour across all PyTorch versions.
    Train/eval behaviour is identical — no running statistics.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → LN over C → (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class _DetailPreservingDownsample(nn.Module):
    """Move a shallow map to the next level without discarding 2x2 phases.

    Pixel-unshuffle first stores every spatial phase in channels.  A pointwise
    projection and depthwise refinement then adapt it to the target level.
    This follows PtbNet's small-object preservation principle more closely
    than a stride-2 convolution.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            _LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels, channels, 3, padding=1,
                groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.xavier_uniform_(self.project[0].weight)
        nn.init.kaiming_normal_(
            self.project[3].weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.project[-1].weight, gain=0.5)
        nn.init.zeros_(self.project[-1].bias)

    @staticmethod
    def _local_variation(x: torch.Tensor) -> torch.Tensor:
        """Mean first-order spatial variation, normalized by feature RMS."""
        x = x.detach().float()
        terms = []
        if x.shape[-2] > 1:
            terms.append((x[..., 1:, :] - x[..., :-1, :]).abs().mean())
        if x.shape[-1] > 1:
            terms.append((x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        if not terms:
            return x.new_zeros(())
        variation = torch.stack(terms).mean()
        return variation / x.square().mean().sqrt().clamp_min(1e-8)

    def forward(self, x: torch.Tensor, target_size) -> torch.Tensor:
        source = x
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        x = F.pixel_unshuffle(x, downscale_factor=2)
        if x.shape[-2:] != tuple(target_size):
            x = F.interpolate(
                x, size=target_size, mode='bilinear',
                align_corners=False)
        output = self.project(x)
        if getattr(self, '_collect_debug', True):
            with torch.no_grad():
                source_rms = source.detach().float().square().mean().sqrt()
                output_rms = output.detach().float().square().mean().sqrt()
                self._dbg_rms_ratio = (
                    output_rms / source_rms.clamp_min(1e-8))
                source_var = self._local_variation(source)
                output_var = self._local_variation(output)
                self._dbg_variation_ratio = (
                    output_var / source_var.clamp_min(1e-8))
        return output

class _WindowedSpatialCrossAttn(nn.Module):
    """Local cross-position attention with bounded memory at P2/P3.

    ``query_feat`` selects spatial evidence from ``value_feat``.  Unlike the
    old same-location cosine gate, every location can attend to other
    positions inside its window.
    """

    def __init__(self, channels: int = 256, window_size: int = 8):
        super().__init__()
        if window_size < 1:
            raise ValueError('window_size must be positive')
        hidden = max(channels // 8, 8)
        self.window_size = int(window_size)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.xavier_uniform_(self.out_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _partition(self, x: torch.Tensor):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.reshape(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.reshape(-1, ws * ws, C)

    def _restore(self, x: torch.Tensor, batch_size: int,
                 padded_h: int, padded_w: int):
        ws = self.window_size
        C = x.shape[-1]
        x = x.reshape(
            batch_size, padded_h // ws, padded_w // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(batch_size, C, padded_h, padded_w)

    def forward(self, query_feat: torch.Tensor,
                value_feat: torch.Tensor) -> torch.Tensor:
        if value_feat.shape[-2:] != query_feat.shape[-2:]:
            value_feat = F.interpolate(
                value_feat, size=query_feat.shape[-2:],
                mode='bilinear', align_corners=False)
        B, _, H, W = query_feat.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        q = self.query(query_feat)
        k = self.key(value_feat)
        v = self.value(value_feat)
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            q = F.pad(q, padding)
            k = F.pad(k, padding)
            v = F.pad(v, padding)

        q = F.normalize(self._partition(q), p=2, dim=-1, eps=1e-6)
        k = F.normalize(self._partition(k), p=2, dim=-1, eps=1e-6)
        v = self._partition(v)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        attn = torch.softmax(
            logit_scale * (q @ k.transpose(-2, -1)), dim=-1)
        if getattr(self, '_collect_debug', True):
            with torch.no_grad():
                # Sampling keeps diagnostics cheap at high-resolution P2/P3.
                query_stride = max(1, attn.shape[-2] // 16)
                probs = attn.detach().float()[
                    :min(attn.shape[0], 8), ::query_stride]
                probs = probs.clamp_min(1e-8)
                normalizer = math.log(max(2, probs.shape[-1]))
                self._dbg_entropy = (
                    -(probs * probs.log()).sum(dim=-1) / normalizer
                ).mean()
                self._dbg_peak = probs.max(dim=-1).values.mean()
                self._dbg_logit_scale = logit_scale.detach().float()
        enhanced = attn @ v
        enhanced = self._restore(enhanced, B, H + pad_h, W + pad_w)
        enhanced = enhanced[:, :, :H, :W]
        return self.out_proj(enhanced)

class _ChannelCrossAttnMatrix(nn.Module):
    """TB-FusionNet-style CxC cross-channel relation, not an SE gate."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.query_fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.key_fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.value = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query_feat: torch.Tensor,
                value_feat: torch.Tensor) -> torch.Tensor:
        if value_feat.shape[-2:] != query_feat.shape[-2:]:
            value_feat = F.interpolate(
                value_feat, size=query_feat.shape[-2:],
                mode='bilinear', align_corners=False)
        query = self.query_fc(query_feat.mean(dim=(2, 3)))
        key = self.key_fc(value_feat.mean(dim=(2, 3)))
        channel_attn = torch.softmax(
            query.unsqueeze(-1) * key.unsqueeze(-2), dim=-1)
        if getattr(self, '_collect_debug', True):
            with torch.no_grad():
                probs = channel_attn.detach().float().clamp_min(1e-8)
                channels = probs.shape[-1]
                normalizer = math.log(max(2, channels))
                self._dbg_entropy = (
                    -(probs * probs.log()).sum(dim=-1) / normalizer
                ).mean()
                self._dbg_peak = probs.max(dim=-1).values.mean()
                # Uniform attention has diag_gain=1. Values above one indicate
                # preferential same-channel transport.
                self._dbg_diag_gain = (
                    probs.diagonal(dim1=-2, dim2=-1).mean() * channels)
        value = self.value(value_feat).flatten(2)
        enhanced = torch.bmm(channel_attn, value)
        enhanced = enhanced.reshape_as(value_feat)
        return self.out_proj(enhanced)

class _CrossScaleOutputRefine(nn.Module):
    """Bounded local refinement; the old unscaled residual could rewrite P2."""

    def __init__(self, channels: int = 256,
                 residual_init: float = 1e-3,
                 residual_max: float = 0.02):
        super().__init__()
        if not 0 < residual_init < residual_max:
            raise ValueError(
                'refine residual_init must be between zero and residual_max')
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1,
            groups=channels, bias=False)
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.residual_max = float(residual_max)
        self.residual_raw = nn.Parameter(torch.tensor(
            math.atanh(residual_init / residual_max),
            dtype=torch.float32))
        nn.init.kaiming_normal_(
            self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def effective_scale(self) -> torch.Tensor:
        return self.residual_max * torch.tanh(self.residual_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.pointwise(self.act(self.depthwise(x)))
        scaled_delta = self.effective_scale() * delta
        if getattr(self, '_collect_debug', True):
            with torch.no_grad():
                base = x.detach().float().flatten(1)
                change = scaled_delta.detach().float().flatten(1)
                self._dbg_rel_delta = (
                    change.norm(dim=1)
                    / base.norm(dim=1).clamp_min(1e-8)
                ).mean()
                self._dbg_delta_cos = F.cosine_similarity(
                    base, change, dim=1, eps=1e-8).mean()
        return x + scaled_delta

class _TBFusionCrossScale(nn.Module):
    """Direct FPN replacement with detail-first bidirectional fusion.

    Main path (source-faithful): P2→P3→P4→P5.  Pixel-unshuffle preserves
    small-lesion phases before windowed spatial and CxC channel cross-attention.

    Optional semantic return: P5→P4→P3→P2.  Its maximum contribution is
    separately bounded and decreases toward P2, while raw P2 bypasses output
    refinement.  Rows returned by :meth:`effective_scales` are the three
    bottom-up hops followed by the three top-down hops.
    """

    def __init__(self, channels: int = 256,
                 residual_init: float = 1e-3,
                 residual_max: float = 0.1,
                 semantic_residual_init: float = 5e-4,
                 semantic_residual_max: float = 0.02,
                 refine_residual_init: float = 1e-3,
                 refine_residual_max: float = 0.02,
                 window_size: int = 8,
                 use_top_down: bool = True,
                 learnable_fusion_weights: bool = False,
                 detail_fusion_init: float = 0.75,
                 semantic_fusion_init: float = 0.25,
                 freqfusion_top_down: bool = False,
                 freqfusion_compressed_channels: int = 64):
        super().__init__()
        self._validate_scale(
            'detail', residual_init, residual_max)
        self._validate_scale(
            'semantic', semantic_residual_init, semantic_residual_max)
        self.use_top_down = bool(use_top_down)
        self.learnable_fusion_weights = bool(
            learnable_fusion_weights)
        self.freqfusion_top_down = bool(freqfusion_top_down)
        self.detail_residual_max = float(residual_max)
        self.semantic_residual_max = float(semantic_residual_max)
        self._validate_fusion_fraction(
            'detail_fusion_init', detail_fusion_init)
        self._validate_fusion_fraction(
            'semantic_fusion_init', semantic_fusion_init)

        self.bottom_up_downsamples = nn.ModuleList([
            _DetailPreservingDownsample(channels) for _ in range(3)
        ])
        self.bottom_up_sca = nn.ModuleList([
            _WindowedSpatialCrossAttn(channels, window_size)
            for _ in range(3)
        ])
        self.bottom_up_cca = nn.ModuleList([
            _ChannelCrossAttnMatrix(channels) for _ in range(3)
        ])
        self.top_down_sca = nn.ModuleList([
            _WindowedSpatialCrossAttn(channels, window_size)
            for _ in range(3)
        ])
        self.top_down_cca = nn.ModuleList([
            _ChannelCrossAttnMatrix(channels) for _ in range(3)
        ])
        self.freqfusion_blocks = nn.ModuleList([
            _FreqFusion2x(
                channels=channels,
                compressed_channels=freqfusion_compressed_channels)
            for _ in range(2)
        ] if self.freqfusion_top_down else [])

        detail_raw_init = math.atanh(residual_init / residual_max)
        semantic_raw_init = math.atanh(
            semantic_residual_init / semantic_residual_max)
        self.detail_residual_raw = nn.Parameter(torch.full(
            (3, 2), detail_raw_init, dtype=torch.float32))
        self.semantic_residual_raw = nn.Parameter(torch.full(
            (3, 2), semantic_raw_init, dtype=torch.float32))
        self.detail_weight_raw = nn.Parameter(
            torch.zeros((3, 2), dtype=torch.float32))
        self.semantic_weight_raw = nn.Parameter(
            torch.zeros((3, 2), dtype=torch.float32))
        self.register_buffer(
            'semantic_level_factors',
            torch.tensor([[1.0], [0.75], [0.5]], dtype=torch.float32))
        detail_logits = torch.log(torch.tensor(
            [1.0 - detail_fusion_init, detail_fusion_init],
            dtype=torch.float32)).repeat(3, 1)
        semantic_logits = torch.log(torch.tensor(
            [1.0 - semantic_fusion_init, semantic_fusion_init],
            dtype=torch.float32)).repeat(3, 1)
        if self.learnable_fusion_weights:
            self.detail_fusion_logits = nn.Parameter(detail_logits)
            self.semantic_fusion_logits = nn.Parameter(semantic_logits)
        else:
            # Buffers preserve checkpoint/config compatibility without adding
            # unused optimizer parameters to the P0U0 baseline.
            self.register_buffer(
                'detail_fusion_logits', detail_logits)
            self.register_buffer(
                'semantic_fusion_logits', semantic_logits)

        # P2 deliberately has no post-fusion convolution.  This keeps its raw
        # high-resolution route intact for small-object anchors.
        self.output_refines = nn.ModuleList([
            _CrossScaleOutputRefine(
                channels, refine_residual_init, refine_residual_max)
            for _ in range(3)
        ])
        self._dbg_hops = []
        self._dbg_output_grad_rms = [None] * 4
        self._dbg_top_down_modes = ['bilinear'] * 3
        # The detector enables these only around periodic debug iterations;
        # direct unit use defaults to full collection.
        self.collect_forward_debug = True
        self.collect_gradient_debug = True

    @staticmethod
    def _validate_scale(name: str, initial: float, maximum: float):
        if not 0 < initial < maximum:
            raise ValueError(
                f'{name} residual_init must satisfy '
                '0 < residual_init < residual_max')

    @staticmethod
    def _validate_fusion_fraction(name: str, value: float):
        if not 0.0 < float(value) < 1.0:
            raise ValueError(f'{name} must be in (0, 1)')

    @staticmethod
    def _branch_scales(raw: torch.Tensor, weights: torch.Tensor,
                       maximum: float) -> torch.Tensor:
        branch_weights = 2.0 * torch.softmax(weights, dim=-1)
        residual_scales = maximum * torch.tanh(raw)
        return branch_weights * residual_scales

    def effective_scales(self) -> torch.Tensor:
        detail = self._branch_scales(
            self.detail_residual_raw,
            self.detail_weight_raw,
            self.detail_residual_max)
        semantic = self._branch_scales(
            self.semantic_residual_raw,
            self.semantic_weight_raw,
            self.semantic_residual_max)
        semantic = semantic * self.semantic_level_factors
        if not self.use_top_down:
            semantic = torch.zeros_like(semantic)
        return torch.cat([detail, semantic], dim=0)

    def effective_fusion_weights(self) -> torch.Tensor:
        """Return ``[base, cross-scale proposal]`` weights for six hops."""
        if not self.learnable_fusion_weights:
            # Disabled means byte-for-byte feature algebra from v3.16:
            # accept the complete residual proposal returned by _fuse.
            disabled = self.detail_fusion_logits.new_tensor(
                [0.0, 1.0]).repeat(3, 1)
            detail = disabled
            semantic = disabled.clone()
        else:
            detail = F.softmax(self.detail_fusion_logits, dim=-1)
            semantic = F.softmax(self.semantic_fusion_logits, dim=-1)
        if not self.use_top_down:
            semantic = semantic.new_tensor([1.0, 0.0]).repeat(3, 1)
        return torch.cat([detail, semantic], dim=0)

    @staticmethod
    def _weighted_feature_fusion(
            base: torch.Tensor,
            proposal: torch.Tensor,
            weights: torch.Tensor) -> torch.Tensor:
        return weights[0] * base + weights[1] * proposal

    def _fuse(self, target: torch.Tensor, source: torch.Tensor,
              sca: nn.Module, cca: nn.Module,
              scales: torch.Tensor) -> torch.Tensor:
        sca_raw = sca(query_feat=target, value_feat=source)
        cca_raw = cca(query_feat=target, value_feat=source)
        sca_delta = scales[0] * sca_raw
        cca_delta = scales[1] * cca_raw
        total_delta = sca_delta + cca_delta
        if self.collect_forward_debug:
            with torch.no_grad():
                target_flat = target.detach().float().flatten(1)
                sca_flat = sca_delta.detach().float().flatten(1)
                cca_flat = cca_delta.detach().float().flatten(1)
                total_flat = total_delta.detach().float().flatten(1)
                target_norm = target_flat.norm(dim=1).clamp_min(1e-8)

                def relative_norm(value):
                    return (value.norm(dim=1) / target_norm).mean()

                self._dbg_hops.append({
                    'rel_sca': relative_norm(sca_flat),
                    'rel_cca': relative_norm(cca_flat),
                    'rel_total': relative_norm(total_flat),
                    'delta_cos': F.cosine_similarity(
                        target_flat, total_flat, dim=1, eps=1e-8).mean(),
                    'branch_cos': F.cosine_similarity(
                        sca_flat, cca_flat, dim=1, eps=1e-8).mean(),
                })
        return target + total_delta

    @staticmethod
    def _debug_value(module: nn.Module, name: str,
                     default: float = 0.0) -> float:
        value = getattr(module, name, default)
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        value = float(value)
        return value if math.isfinite(value) else float(default)

    def debug_stats(self) -> dict:
        """Return forward-path evidence used by the periodic debug logger.

        The residual scales show intent; ``hop_rel_*`` shows the actual
        feature change after attention, which is the stronger effectiveness
        test. Attention entropy is normalized to [0, 1], where values near
        one indicate an almost uniform transport matrix.
        """
        scales = self.effective_scales().detach().float().cpu()
        hops = list(self._dbg_hops)
        while len(hops) < 6:
            hops.append({})

        spatial_modules = list(self.bottom_up_sca) + list(self.top_down_sca)
        channel_modules = list(self.bottom_up_cca) + list(self.top_down_cca)
        return {
            'scale_sca': scales[:, 0].tolist(),
            'scale_cca': scales[:, 1].tolist(),
            'fusion_base_weight':
                self.effective_fusion_weights()[:, 0]
                .detach().float().cpu().tolist(),
            'fusion_proposal_weight':
                self.effective_fusion_weights()[:, 1]
                .detach().float().cpu().tolist(),
            'hop_rel_sca': [
                self._debug_value_from_mapping(hop, 'rel_sca')
                for hop in hops],
            'hop_rel_cca': [
                self._debug_value_from_mapping(hop, 'rel_cca')
                for hop in hops],
            'hop_rel_total': [
                self._debug_value_from_mapping(hop, 'rel_total')
                for hop in hops],
            'hop_delta_cos': [
                self._debug_value_from_mapping(hop, 'delta_cos')
                for hop in hops],
            'hop_branch_cos': [
                self._debug_value_from_mapping(hop, 'branch_cos')
                for hop in hops],
            'sca_entropy': [
                self._debug_value(module, '_dbg_entropy')
                for module in spatial_modules],
            'sca_peak': [
                self._debug_value(module, '_dbg_peak')
                for module in spatial_modules],
            'cca_entropy': [
                self._debug_value(module, '_dbg_entropy')
                for module in channel_modules],
            'cca_peak': [
                self._debug_value(module, '_dbg_peak')
                for module in channel_modules],
            'cca_diag_gain': [
                self._debug_value(module, '_dbg_diag_gain')
                for module in channel_modules],
            'downsample_rms_ratio': [
                self._debug_value(module, '_dbg_rms_ratio')
                for module in self.bottom_up_downsamples],
            'downsample_variation_ratio': [
                self._debug_value(module, '_dbg_variation_ratio')
                for module in self.bottom_up_downsamples],
            'refine_rel_delta': [
                self._debug_value(module, '_dbg_rel_delta')
                for module in self.output_refines],
            'refine_delta_cos': [
                self._debug_value(module, '_dbg_delta_cos')
                for module in self.output_refines],
            'freq_alpf_entropy': [
                self._debug_value(module, '_dbg_alpf_entropy')
                for module in self.freqfusion_blocks],
            'freq_ahpf_entropy': [
                self._debug_value(module, '_dbg_ahpf_entropy')
                for module in self.freqfusion_blocks],
            'freq_offset_abs': [
                self._debug_value(module, '_dbg_offset_abs')
                for module in self.freqfusion_blocks],
            'freq_lr_vs_bilinear': [
                self._debug_value(module, '_dbg_lr_vs_bilinear')
                for module in self.freqfusion_blocks],
            'freq_hr_detail_rel': [
                self._debug_value(module, '_dbg_hr_detail_rel')
                for module in self.freqfusion_blocks],
            'freq_local_similarity': [
                self._debug_value(module, '_dbg_local_similarity')
                for module in self.freqfusion_blocks],
        }

    @staticmethod
    def _debug_value_from_mapping(mapping: dict, name: str,
                                  default: float = 0.0) -> float:
        value = mapping.get(name, default)
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        value = float(value)
        return value if math.isfinite(value) else float(default)

    @staticmethod
    def _gradient_group(parameters) -> tuple:
        sum_sq = 0.0
        seen_numel = 0
        total_numel = 0
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            total_numel += parameter.numel()
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            sum_sq += grad.square().sum().item()
            seen_numel += parameter.numel()
        rms = math.sqrt(sum_sq / max(1, seen_numel))
        coverage = seen_numel / max(1, total_numel)
        return rms, coverage

    def gradient_stats(self) -> dict:
        """Report whether detection gradients reach every CrossScale branch."""
        parameter_groups = {
            'bottom_up_downsample':
                self.bottom_up_downsamples.parameters(),
            'bottom_up_sca': self.bottom_up_sca.parameters(),
            'bottom_up_cca': self.bottom_up_cca.parameters(),
            'top_down_sca': self.top_down_sca.parameters(),
            'top_down_cca': self.top_down_cca.parameters(),
            'output_refine': self.output_refines.parameters(),
            'detail_scale': [
                self.detail_residual_raw, self.detail_weight_raw],
            'semantic_scale': [
                self.semantic_residual_raw, self.semantic_weight_raw],
        }
        if self.learnable_fusion_weights:
            parameter_groups['fusion_weight'] = [
                self.detail_fusion_logits,
                self.semantic_fusion_logits]
        if self.freqfusion_top_down:
            parameter_groups['freqfusion'] = (
                self.freqfusion_blocks.parameters())
        grad_rms = {}
        grad_coverage = {}
        for name, parameters in parameter_groups.items():
            rms, coverage = self._gradient_group(parameters)
            grad_rms[name] = rms
            grad_coverage[name] = coverage

        output_grad_rms = []
        for value in self._dbg_output_grad_rms:
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().item()
            value = 0.0 if value is None else float(value)
            output_grad_rms.append(
                value if math.isfinite(value) else 0.0)
        return {
            'output_grad_rms': output_grad_rms,
            'group_grad_rms': grad_rms,
            'group_grad_coverage': grad_coverage,
        }

    def _capture_output_gradient(self, index: int, grad: torch.Tensor):
        self._dbg_output_grad_rms[index] = (
            grad.detach().float().square().mean().sqrt())
        return grad

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError('CrossScale requires P2/P3/P4/P5 inputs')
        P2, P3, P4, P5 = feats
        scales = self.effective_scales()
        fusion_weights = self.effective_fusion_weights()
        self._dbg_hops = []
        self._dbg_output_grad_rms = [None] * 4
        debug_modules = (
            list(self.bottom_up_downsamples)
            + list(self.bottom_up_sca)
            + list(self.bottom_up_cca)
            + list(self.top_down_sca)
            + list(self.top_down_cca)
            + list(self.freqfusion_blocks)
            + list(self.output_refines))
        for module in debug_modules:
            module._collect_debug = self.collect_forward_debug

        # Detail-first path: source-faithful low→high fusion.
        P2d = self.bottom_up_downsamples[0](P2, P3.shape[-2:])
        P3_proposal = self._fuse(
            P3, P2d, self.bottom_up_sca[0],
            self.bottom_up_cca[0], scales[0])
        P3b = self._weighted_feature_fusion(
            P3, P3_proposal, fusion_weights[0])
        P3d = self.bottom_up_downsamples[1](P3b, P4.shape[-2:])
        P4_proposal = self._fuse(
            P4, P3d, self.bottom_up_sca[1],
            self.bottom_up_cca[1], scales[1])
        P4b = self._weighted_feature_fusion(
            P4, P4_proposal, fusion_weights[1])
        P4d = self.bottom_up_downsamples[2](P4b, P5.shape[-2:])
        P5_proposal = self._fuse(
            P5, P4d, self.bottom_up_sca[2],
            self.bottom_up_cca[2], scales[2])
        P5b = self._weighted_feature_fusion(
            P5, P5_proposal, fusion_weights[2])

        if self.use_top_down:
            # Lightweight semantic return.  Here the deep feature is VALUE,
            # so semantic content, rather than another shallow reweighting,
            # is actually injected into the target level.
            if self.freqfusion_top_down:
                P4_target, P5u = self.freqfusion_blocks[0](
                    P4b, P5b)
                self._dbg_top_down_modes = [
                    'freqfusion', 'freqfusion', 'bilinear']
            else:
                P4_target = P4b
                P5u = deterministic_bilinear_interpolate(
                    P5b, size=P4b.shape[-2:], align_corners=False)
                self._dbg_top_down_modes = ['bilinear'] * 3
            P4_proposal = self._fuse(
                P4_target, P5u, self.top_down_sca[0],
                self.top_down_cca[0], scales[3])
            P4o = self._weighted_feature_fusion(
                P4_target, P4_proposal, fusion_weights[3])
            if self.freqfusion_top_down:
                P3_target, P4u = self.freqfusion_blocks[1](
                    P3b, P4o)
            else:
                P3_target = P3b
                P4u = deterministic_bilinear_interpolate(
                    P4o, size=P3b.shape[-2:], align_corners=False)
            P3_proposal = self._fuse(
                P3_target, P4u, self.top_down_sca[1],
                self.top_down_cca[1], scales[4])
            P3o = self._weighted_feature_fusion(
                P3_target, P3_proposal, fusion_weights[4])
            P3u = deterministic_bilinear_interpolate(
                P3o, size=P2.shape[-2:], align_corners=False)
            P2_proposal = self._fuse(
                P2, P3u, self.top_down_sca[2],
                self.top_down_cca[2], scales[5])
            P2o = self._weighted_feature_fusion(
                P2, P2_proposal, fusion_weights[5])
        else:
            P2o, P3o, P4o = P2, P3b, P4b
            self._dbg_top_down_modes = ['disabled'] * 3

        outputs = [
            P2o,
            self.output_refines[0](P3o),
            self.output_refines[1](P4o),
            self.output_refines[2](P5b),
        ]
        if (self.collect_gradient_debug and self.training
                and torch.is_grad_enabled()):
            for index, output in enumerate(outputs):
                if output.requires_grad:
                    output.register_hook(
                        lambda grad, i=index:
                        self._capture_output_gradient(i, grad))
        return outputs

@NECKS.register_module()
class TQCrossScaleNeck(nn.Module):
    """Four-level CrossScale neck that directly replaces FPN.

    Interface:
        inputs: C2/C3/C4/C5 backbone tensors
        returns: P2/P3/P4/P5 tensors with ``out_channels`` channels

    PC enhancement is deliberately outside this module and is applied only
    in the bbox regression tower, so this neck can be ablated independently
    against a four-level FPN.
    """

    def __init__(self,
                 in_channels=(72, 144, 288, 576),
                 out_channels: int = 256,
                 residual_init: float = 1e-3,
                 residual_max: float = 0.1,
                 semantic_residual_init: float = 5e-4,
                 semantic_residual_max: float = 0.02,
                 refine_residual_init: float = 1e-3,
                 refine_residual_max: float = 0.02,
                 window_size: int = 8,
                 use_top_down: bool = True,
                 learnable_fusion_weights: bool = False,
                 detail_fusion_init: float = 0.75,
                 semantic_fusion_init: float = 0.25,
                 freqfusion_top_down: bool = False,
                 freqfusion_compressed_channels: int = 64,
                 reg_detail_bypass: bool = False,
                 reg_detail_compressed_channels: int = 32):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(
                'TBFusionCrossScaleNeck requires four input levels')
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.use_reg_detail_bypass = bool(reg_detail_bypass)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in self.in_channels
        ])
        self.cross_scale = _TBFusionCrossScale(
            channels=out_channels,
            residual_init=residual_init,
            residual_max=residual_max,
            semantic_residual_init=semantic_residual_init,
            semantic_residual_max=semantic_residual_max,
            refine_residual_init=refine_residual_init,
            refine_residual_max=refine_residual_max,
            window_size=window_size,
            use_top_down=use_top_down,
            learnable_fusion_weights=learnable_fusion_weights,
            detail_fusion_init=detail_fusion_init,
            semantic_fusion_init=semantic_fusion_init,
            freqfusion_top_down=freqfusion_top_down,
            freqfusion_compressed_channels=(
                freqfusion_compressed_channels))
        self.reg_detail_generator = (
            _C5GatedC2RegressionDetail(
                channels=out_channels,
                compressed_channels=(
                    reg_detail_compressed_channels))
            if self.use_reg_detail_bypass else None)
        self.regression_detail_feats = (None, None, None, None)

        for conv in self.lateral_convs:
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, inputs):
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f'expected {len(self.in_channels)} backbone levels, '
                f'got {len(inputs)}')
        laterals = [
            conv(feat)
            for conv, feat in zip(self.lateral_convs, inputs)
        ]
        if self.use_reg_detail_bypass:
            detail_p4, detail_p5 = self.reg_detail_generator(
                laterals[0], laterals[-1],
                target_sizes=(
                    laterals[2].shape[-2:],
                    laterals[3].shape[-2:]))
            self.regression_detail_feats = (
                None, None, detail_p4, detail_p5)
        else:
            self.regression_detail_feats = (
                None, None, None, None)
        return tuple(self.cross_scale(laterals))
