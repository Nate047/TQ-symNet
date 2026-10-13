"""Backbone/neck control: TransNeXt+native FPN+Sym, no quality scorer."""
_base_ = ['./train_detector.py']
model = dict(
    experiment_tag='TQ_FOUNDATION_TRANSNEXT_FPN',
    inference_nms='tb_hard',
    neck=dict(_delete_=True, type='FPN', in_channels=[72, 144, 288, 576],
              out_channels=256, start_level=0, num_outs=4, add_extra_convs=False),
    bbox_head=dict(anchor_generator=dict(strides=[4, 8, 16, 32])))
experiment_role = 'foundation_training'
