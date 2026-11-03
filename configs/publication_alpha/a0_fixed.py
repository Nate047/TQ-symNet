"""Same-round control on original 599/200 cohort; fresh quality, fixed alpha .5."""
_base_ = ['./_original_a0.py']
model = dict(experiment_tag='PUB_ALPHA_FIXED', selector_gate_trainable=False)
