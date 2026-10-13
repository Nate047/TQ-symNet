"""Remove rank losses, frozen M2-20e epoch 17."""
_base_ = ['./_base.py']
model = dict(quality_rank_weight=0., selector_rank_weight=0.)
work_dir = 'work_dirs/balanced_v2_quality_m2_20e_seed0/a7_s0'
