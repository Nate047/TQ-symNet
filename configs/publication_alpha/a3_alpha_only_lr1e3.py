"""Faster scalar calibration from the SAME selected epoch-7 source as A2.

Only alpha is optimized. Choose the resulting checkpoint by validation AP;
calibration-loss convergence does not imply AP optimality.
"""
_base_ = ['./a2_alpha_only.py']
model = dict(experiment_tag='PUB_ALPHA_ONLY_LR1E3')
optimizer = dict(lr=1e-3)
