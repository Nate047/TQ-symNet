"""Optional same-runtime bridge to archived A10, not a required new run."""
_base_ = ['./_roi.py']
model = dict(experiment_tag='PUB_A10_ROI_BRIDGE', quality_pooled_mlp_hidden=96)
