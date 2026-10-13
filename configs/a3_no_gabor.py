"""Ablation: remove only the Gabor memory route."""
_base_ = ['./_matched_quality.py']
model = dict(quality_texture=True, quality_lbp=True, quality_gabor=False)
