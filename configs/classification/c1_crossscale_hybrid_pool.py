"""Frozen detector CrossScale; image head pools P3/P4/P5 by mean and max."""
_base_ = ['./c0_backbone_sym.py']
neck = dict(
    type='TQCrossScaleNeck',
    in_channels=[72, 144, 288, 576], out_channels=256,
    residual_init=0.001, residual_max=0.1,
    semantic_residual_init=0.0005, semantic_residual_max=0.02,
    refine_residual_init=0.001, refine_residual_max=0.02,
    window_size=8, use_top_down=True, learnable_fusion_weights=False,
    detail_fusion_init=0.75, semantic_fusion_init=0.25,
    freqfusion_top_down=False, freqfusion_compressed_channels=64,
    reg_detail_bypass=False, reg_detail_compressed_channels=32)
classification_head = 'hybrid_pool'
classification_options = dict(in_channels=256, hidden=512, dropout=0.1)
