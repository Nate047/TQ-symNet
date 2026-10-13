"""Frozen B1, matched image list and Soft-NMS; read official_tb_bbox only."""
_base_ = ['../b1_sym_soft.py']
custom_hooks = []
data = dict(test=dict(
    ann_file='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/montgomery_extra_trainval_only_tb.json',
    classes=('tuberculosis', 'unused_model_slot'), test_mode=True))
