"""Independent training-budget control: ~7280 updates versus historical 7264.

Do not replace the existing 16-epoch selected detector before comparing on val.
"""
_base_ = ['./m2_transnext_crossscale.py']
model = dict(experiment_tag='TQ_M2_TRANSNEXT_CROSSSCALE_20E')
runner = dict(max_epochs=20)
lr_config = dict(step=[14,19])
work_dir = 'work_dirs/balanced_v2_seed0_479_120_200/m2_20e_s0'
