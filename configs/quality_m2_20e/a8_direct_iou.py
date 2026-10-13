"""Direct IoU supervision with rank, frozen M2-20e epoch 17."""
_base_ = ['./_base.py']
model = dict(selector_quality_target='direct_iou', selector_quality_target_temperature=.03)
work_dir = 'work_dirs/balanced_v2_quality_m2_20e_seed0/a8_s0'
