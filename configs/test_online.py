"""Official hidden test export. No GT evaluation, filtering, or Oracle."""
_base_ = ['./a0_full.py']
model = dict(diagnostics=False)
custom_hooks = []
data = dict(test=dict(ann_file='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/all_test.json', test_mode=True))
