_base_ = [
    './_base_/models/mask_rcnn_efficientvit_fpn.py',
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]
pretrained = None
model = dict(
    pretrained=None,
    backbone=dict(
        type='EfficientViM_M4',
        pretrained=pretrained,
        frozen_stages=-1,
        in_channels=[224, 320, 512],
        ),
    neck=dict(
        type='EfficientViTFPN',
        in_channels=[224, 320, 512],
        out_channels=256,
        num_outs=5,
        start_level=0,
        num_extra_trans_convs=2,
        pre_norm=True),
    )

# weight_Deca
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'attention_biases': dict(decay_mult=0.),
                                                 'attention_bias_idxs': dict(decay_mult=0.),
                                                 }))

optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
version = "TEST"

# optimizer.lr=0.0002
# optimizer.weight_decay=0.0001
# lr_config.step=[11]
#