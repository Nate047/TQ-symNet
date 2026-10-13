"""Remove LBP/Gabor, frozen M2-20e epoch 17."""
_base_ = ['./_base.py']
model = dict(quality_texture=False, quality_lbp=False, quality_gabor=False)
work_dir = 'work_dirs/balanced_v2_quality_m2_20e_seed0/a1_s0'
