"""GT-free hidden-index export of the selected GFL adaptation."""
_base_ = ['./gfl_joint.py']
model = dict(diagnostics=False, backbone=dict(pretrained=None))
custom_hooks = []
data = dict(test=dict(
    ann_file='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/all_test.json',
    test_mode=True))
