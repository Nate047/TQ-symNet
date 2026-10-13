"""120-image model-selection validation. Final 200-image test uses each model's own cfg."""
_base_ = ['./_common.py']
model = dict(diagnostics=False)
custom_hooks = []
evaluation = dict(save_best=None)
data = dict(test=dict(ann_file='data/splits/tbx11k_479_120_200_balanced_v2_seed0/seed_0/val.json'))
