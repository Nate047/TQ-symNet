"""Ablation: remove only the LBP memory route."""
_base_ = ['./_matched_quality.py']
model = dict(quality_texture=True, quality_lbp=False, quality_gabor=True)
