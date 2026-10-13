"""Ablation: remove LBP and Gabor; visual and spatial routes remain."""
_base_ = ['./_matched_quality.py']
model = dict(quality_texture=False, quality_lbp=False, quality_gabor=False)
