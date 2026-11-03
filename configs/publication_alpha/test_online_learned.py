"""Official test index; identical scorer to publication_alpha/a1_learned.

The learned scalar is restored from the selected checkpoint, not overwritten
with the validation-sweep alpha. Reuse the exact previous C20 TXT/manifest.
"""
_base_ = ['./a1_learned.py']
model = dict(diagnostics=False, inference_mode='tb')
custom_hooks = []
data = dict(test=dict(
    ann_file='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/all_test.json',
    test_mode=True))
