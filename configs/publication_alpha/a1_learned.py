"""Only switch: learn existing bounded scalar alpha with detached calibration."""
_base_ = ['./_original_a0.py']
model = dict(experiment_tag='PUB_ALPHA_LEARNED', selector_gate_trainable=True)
