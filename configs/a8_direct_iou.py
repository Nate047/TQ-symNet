"""Objective ablation: continuous IoU target with the full representation."""
_base_ = ['./_matched_quality.py']
model = dict(selector_quality_target='direct_iou',
             selector_quality_target_temperature=.03)
