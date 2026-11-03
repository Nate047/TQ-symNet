custom_imports = dict(imports=['TQ_SymNet.models', 'TQ_SymNet.alpha_experiment'], allow_failed_imports=False)
model = dict(
    type='TQSymNet',
    experiment_tag='TQ_SYMNET_MAX_ALL_IOU',
    backbone=dict(
        type='transnext_small',
        pretrain_size=384,
        img_size=512,
        pretrained=None,
        scales=5),
    neck=dict(
        type='TQCrossScaleNeck',
        in_channels=[72, 144, 288, 576],
        out_channels=256,
        residual_init=0.001,
        residual_max=0.1,
        semantic_residual_init=0.0005,
        semantic_residual_max=0.02,
        refine_residual_init=0.001,
        refine_residual_max=0.02,
        window_size=8,
        use_top_down=True,
        learnable_fusion_weights=False,
        detail_fusion_init=0.75,
        semantic_fusion_init=0.25,
        freqfusion_top_down=False,
        freqfusion_compressed_channels=64,
        reg_detail_bypass=False,
        reg_detail_compressed_channels=32),
    bbox_head=dict(
        type='TQSymHead',
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        drop_path_rate=0.1,
        token_dropout=0.0,
        spatial_temp=1.5,
        max_regions=200,
        contrast_temperature=0.07,
        purity_weight_floor=0.25,
        hard_negative_ratio=3.0,
        hard_negative_expansion=1.5,
        min_hard_negatives=16,
        head_pc_modulation=False,
        bbox_contrastive=False,
        boxalign=False,
        reg_detail_bypass=False,
        reg_detail_levels=(2, 3),
        reg_detail_residual_init=0.01,
        reg_detail_residual_max=0.1,
        d_fine_style=True,
        d_fine_reg_max=16,
        d_fine_refine_init=0.25,
        d_fine_refine_max=1.0,
        d_fine_loss_weight=0.05,
        edge_supervision=False,
        edge_loss_weight=0.1,
        debug_interval=50,
        deep_debug_interval=500,
        ema_momentum=0.0,
        num_classes=2,
        num_query=500,
        dims_radio=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        reg_decoded_bbox=True,
        transformer=dict(
            type='DeformableDetrTransformerConv',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='SymDetrTransformerEncoderLayer',
                    attn_cfgs=dict(
                        type='SymMultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='GuidePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5,
            left=True,
            use_upper_prior=False,
            upper_prior_mode='image',
            upper_prior_boundary=0.3333333333333333,
            upper_prior_temperature=0.08,
            upper_prior_init=0.02,
            upper_prior_max=0.15),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        lung_anatomy_injection='off',
        lung_anatomy_levels=(3, ),
        lung_anatomy_residual_init=0.01,
        lung_anatomy_residual_max=0.05,
        nms_oracle_debug=False,
        nms_debug_interval=250,
        sym_regression_source='crossscale',
        sym_reg_fusion='adapted_sym',
        sym_reg_residual_max=0.1,
        sym_reg_residual_trainable=True,
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100),
    quality_detach=True,
    quality_texture=True,
    quality_lbp=True,
    quality_gabor=True,
    quality_gaussian=True,
    quality_hidden=96,
    quality_heads=6,
    quality_chunk=32,
    quality_checkpoint=True,
    quality_samples_per_bin=96,
    quality_loss_weight=1.0,
    quality_rank_weight=0.1,
    quality_score_mode='sym',
    quality_underestimate_weight=2.0,
    selector_beta=0.5,
    selector_temperature=0.03,
    selector_rank_weight=0.1,
    selector_sigma=0.7,
    selector_edges=True,
    selector_edge_weight=0.0,
    selector_detail=True,
    selector_learned_gate=True,
    selector_gate_trainable=False,
    selector_quality_trainable=True,
    selector_adaptive_subtype_gate=False,
    selector_fusion_loss_weight=0.0,
    selector_quality_target='coco_all_iou',
    selector_quality_target_temperature=0.03,
    selector_tb_fusion='max',
    diagnostics=True,
    quality_arch='attention',
    selector_fusion_init=0.5,
    oracle_interval=4,
    inference_mode='tb')
optimizer = dict(
    type='AdamW',
    lr=1e-06,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(candidate_quality=dict(lr_mult=100.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0.0, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=8)
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='official_tb_bbox_mAP',
    rule='greater',
    classwise=True,
    tb_official_agnostic=True,
    tb_class_agnostic=False,
    tb_nms_iou=0.5)
checkpoint_config = dict(interval=1, max_keep_ckpts=4)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = []
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
find_unused_parameters = True
fp16 = None
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoDataset',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='PreserveOriginalImage', key='original_img'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='ImageToTensor', keys=['original_img']),
                    dict(
                        type='ToDataContainer',
                        fields=[dict(key='original_img', cpu_only=True)]),
                    dict(
                        type='Collect',
                        keys=['img', 'original_img', 'gt_bboxes', 'gt_labels'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'original_img_shape'))
                ],
                ann_file=
                '/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_train_only_tb.json',
                img_prefix='/home/data2/LiYingxian/Dataset_TB/TBX11K/imgs',
                filter_empty_gt=True,
                classes=('ActiveTuberculosis',
                         'ObsoletePulmonaryTuberculosis')),
            dict(
                type='RepeatDataset',
                times=3,
                dataset=dict(
                    type='CocoCategorySubsetDataset',
                    subset_classes=('ObsoletePulmonaryTuberculosis', ),
                    exclude_classes=('ActiveTuberculosis', ),
                    log_stats=True,
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='PreserveOriginalImage', key='original_img'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='Resize',
                            img_scale=(512, 512),
                            keep_ratio=True),
                        dict(type='RandomFlip', flip_ratio=0.5),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=32),
                        dict(type='DefaultFormatBundle'),
                        dict(type='ImageToTensor', keys=['original_img']),
                        dict(
                            type='ToDataContainer',
                            fields=[dict(key='original_img', cpu_only=True)]),
                        dict(
                            type='Collect',
                            keys=[
                                'img', 'original_img', 'gt_bboxes', 'gt_labels'
                            ],
                            meta_keys=('filename', 'ori_filename', 'ori_shape',
                                       'img_shape', 'pad_shape',
                                       'scale_factor', 'flip',
                                       'flip_direction', 'img_norm_cfg',
                                       'original_img_shape'))
                    ],
                    ann_file=
                    '/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_train_only_tb.json',
                    img_prefix='/home/data2/LiYingxian/Dataset_TB/TBX11K/imgs',
                    filter_empty_gt=True,
                    classes=('ActiveTuberculosis',
                             'ObsoletePulmonaryTuberculosis')))
        ]),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json',
        img_prefix='/home/data2/LiYingxian/Dataset_TB/TBX11K/imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='PreserveOriginalImage', key='original_img'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img', 'original_img']),
                    dict(
                        type='ToDataContainer',
                        fields=[dict(key='original_img', cpu_only=True)]),
                    dict(
                        type='Collect',
                        keys=['img', 'original_img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'original_img_shape'))
                ])
        ],
        classes=('ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis'),
        test_mode=True),
    test=dict(
        type='CocoDataset',
        test_mode=True,
        ann_file=
        '/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json',
        img_prefix='/home/data2/LiYingxian/Dataset_TB/TBX11K/imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='PreserveOriginalImage', key='original_img'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img', 'original_img']),
                    dict(
                        type='ToDataContainer',
                        fields=[dict(key='original_img', cpu_only=True)]),
                    dict(
                        type='Collect',
                        keys=['img', 'original_img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'original_img_shape'))
                ])
        ],
        classes=('ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis')))
detector_checkpoint = 'work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth'
experiment_role = 'matched_quality_training'


# Original publication cohort, not the later 479/120/200 split.
custom_hooks = [
    dict(type='TQOriginalCohortInitHook', priority='VERY_HIGH',
         checkpoint=detector_checkpoint,
         checkpoint_sha256='4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74',
         train_annotation=data['train']['datasets'][0]['ann_file'],
         val_annotation=data['val']['ann_file'],
         train_sha256='9d1b19eb8985e281c4767148fac466445844b764db3dab2f7ca635503470971c',
         val_sha256='bdec993506af15bf72010b32ec82ac279615ccc4c51ad9ca9438a7b9638d6e9d'),
    dict(type='TQSymNetDebugHook'),
    dict(type='TQScalarAlphaAuditHook', interval=50, priority='NORMAL')]
# Explicit longest-key override: same LR as quality, no gate weight decay.
optimizer['paramwise_cfg']['custom_keys']['candidate_quality.fusion_gate_logit'] = dict(lr_mult=100., decay_mult=0.)
