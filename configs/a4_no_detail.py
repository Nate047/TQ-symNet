"""Ablation: remove native-image side-detail attention."""
_base_ = ['./_matched_quality.py']
model = dict(selector_detail=False)
