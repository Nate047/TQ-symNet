"""A0 original reading + 11-bin cumulative threshold supervision, learned alpha.

No F1/F2 local reader. Only the last quality projection and its BCE change.
"""
_base_ = ['./_common.py']
model = dict(type='TQSymNetThresholdDistribution',
             experiment_tag='A0_CUMULATIVE_THRESHOLD_LEARNED_ALPHA')
