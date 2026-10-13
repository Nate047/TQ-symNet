"""Same checkpoint, classwise quality fusion and classwise Soft-NMS."""
_base_ = ['./_common.py']
model = dict(inference_mode='subtype')
custom_hooks = []
evaluation = dict(save_best=None)
