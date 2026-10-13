"""Remove native-image detail encoder, frozen M2-20e epoch 17."""
_base_ = ['./_base.py']
model = dict(selector_detail=False)
work_dir = 'work_dirs/balanced_v2_quality_m2_20e_seed0/a4_s0'
