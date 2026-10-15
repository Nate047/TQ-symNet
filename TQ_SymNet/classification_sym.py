"""Fresh official SymFormer block for C12 classification.

Reuses bundled author classes and original RetinaGuideAttHead settings:
256 channels, one encoder layer, 8 heads, 4 deformable points, 1024 FFN,
GuidePositionalEncoding(normalize=True, offset=-.5, left=True).
The bundled position class retains the project's equivalent deterministic
sampling helper; no lung mask or upper-zone prior is enabled.
"""
import torch
from torch import nn


class OriginalSymPyramid(nn.Module):
    def __init__(self, channels=256, dimension=128):
        super().__init__()
        if channels != 256:
            raise ValueError('Official GuidePositionalEncoding requires 256 input channels')
        from mmdet.models.utils.transformer import DeformableDetrTransformerConv
        from mmdet.models.utils.positional_encoding import GuidePositionalEncoding
        from .classification_heads import _PerPositionLayerNorm

        self.position = GuidePositionalEncoding(num_feats=128, normalize=True,
                                                offset=-.5, left=True)
        self.transformer = DeformableDetrTransformerConv(encoder=dict(
            type='DetrTransformerEncoder', num_layers=1,
            transformerlayers=dict(type='SymDetrTransformerEncoderLayer',
                attn_cfgs=dict(type='SymMultiScaleDeformableAttention',
                               embed_dims=256, num_levels=1),
                feedforward_channels=1024, ffn_dropout=.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))))
        # Original conv-only transformer requires a non-None query embedding,
        # although it does not use decoder queries in its encoder forward.
        self.query_embedding = nn.Embedding(100, 512)
        self.transformer.init_weights()
        # Match RetinaGuideAttHead: the same Sym block is used at each scale.
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, dimension, 1),
                          _PerPositionLayerNorm(dimension)) for _ in range(3)])

    def forward(self, features):
        if len(features) != 3:
            raise ValueError('Expected P3/P4/P5')
        outputs = []
        for feature, projection in zip(features, self.projections):
            if not feature.is_cuda:
                raise RuntimeError('Official Sym requires the MMCV CUDA deformable-attention operator')
            b, _, h, w = feature.shape
            mask = torch.zeros((b, h, w), device=feature.device, dtype=torch.bool)
            position = self.position(mask, feature)
            enhanced = self.transformer((feature,), [mask],
                                        self.query_embedding.weight, [position])
            outputs.append(projection(enhanced))
        return outputs
