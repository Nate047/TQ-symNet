"""Frozen A0, expert-boxed positive Shenzhen images; read official_tb_bbox only."""
_base_ = ['../a0_full.py']
model = dict(diagnostics=False)
custom_hooks = []
data = dict(test=dict(
    ann_file='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/shenzhen_extra_trainval_only_tb.json',
    classes=('tuberculosis', 'unused_model_slot'), test_mode=True))
