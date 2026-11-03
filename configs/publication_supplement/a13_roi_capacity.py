"""A13: widen ONLY MLP 392->1760->880->1; features remain 4x96."""
_base_ = ['./_roi.py']
model = dict(experiment_tag='PUB_A13_ROI_CAPACITY', quality_pooled_mlp_hidden=880)
