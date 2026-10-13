"""Objective reference: previous smoothed 75/85 target."""
_base_ = ['./_matched_quality.py']
model = dict(selector_quality_target='high_75_85')
