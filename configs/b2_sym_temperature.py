"""Optional inference control: same score transform as alpha=.5 and constant quality logit=0.

This separates learned quality benefit from a fixed score-temperature change
that can also affect Gaussian Soft-NMS's sequential choices.
"""
_base_ = ['./b1_sym_soft.py']
model = dict(experiment_tag='TQ_B2_SYM_TEMPERATURE', baseline_temperature=2.)
