"""Matched control: original role QKV, alpha trainable from .5."""
_base_ = ['./_common.py']
model = dict(type='TQSymNet', experiment_tag='ROI_QUERY_A0_ORIGINAL_LEARNED_ALPHA')
