"""Optional official bundled VFNetHead: no hand-written VFNet surrogate."""
_base_ = ['./_dense.py']
model = dict(experiment_tag='PUB_VFNET_JOINT', bbox_head=dict(_delete_=True,
    type='VFNetHead', num_classes=2, in_channels=256, feat_channels=256,
    stacked_convs=3, strides=[4,8,16,32], gradient_mul=.1, bbox_norm_type='reg_denom',
    regress_ranges=((-1,64),(64,128),(128,256),(256,100000000)),
    use_atss=True, use_vfl=True,
    anchor_generator=dict(type='AnchorGenerator', ratios=[1.], octave_base_scale=8,
                          scales_per_octave=1, center_offset=0., strides=[4,8,16,32]),
    loss_cls=dict(type='VarifocalLoss', use_sigmoid=True, alpha=.75, gamma=2., iou_weighted=True, loss_weight=1.),
    loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
    loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.)))
