"""Remove Gaussian/ring/side attention priors; keep coordinates and padding masks.

Native local side sampling remains, so this ablates attention biases rather
than all geometric information.
"""
_base_ = ['./_matched_quality.py']
model = dict(quality_gaussian=False)
