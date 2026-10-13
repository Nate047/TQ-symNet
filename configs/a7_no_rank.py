"""Ablation: remove quality ranking terms while retaining quality BCE."""
_base_ = ['./_matched_quality.py']
model = dict(quality_rank_weight=0., selector_rank_weight=0.)
