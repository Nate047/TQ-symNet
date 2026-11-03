"""Jointly trained official-head controls. Different candidates from TQ-SymNet.
Same TransNeXt/CrossScale, original cohort and 16-epoch M2 optimizer recipe.
The 8 extra TQ quality epochs are NOT hidden in compute comparisons.
"""
_base_ = ['../publication_alpha/_original_a0.py']
custom_imports = dict(imports=['TQ_SymNet.models', 'TQ_SymNet.supplement'], allow_failed_imports=False)
model = dict(_delete_=True, type='TQDenseHeadBaseline',
    backbone={"type": "transnext_small",
"pretrain_size": 384,
"img_size": 512,
"pretrained": "/home/data2/LiYingxian/TransNeXt-main/transnext_small_384_1k_ft_1k.pth",
"scales": 5},
    neck={"type": "TQCrossScaleNeck",
"in_channels": [72, 144, 288, 576],
"out_channels": 256,
"residual_init": 0.001,
"residual_max": 0.1,
"semantic_residual_init": 0.0005,
"semantic_residual_max": 0.02,
"refine_residual_init": 0.001,
"refine_residual_max": 0.02,
"window_size": 8,
"use_top_down": True,
"learnable_fusion_weights": False,
"detail_fusion_init": 0.75,
"semantic_fusion_init": 0.25,
"freqfusion_top_down": False,
"freqfusion_compressed_channels": 64,
"reg_detail_bypass": False,
"reg_detail_compressed_channels": 32},
    bbox_head=dict(type='GFLHead', num_classes=2, in_channels=256,
        stacked_convs=4, feat_channels=256, reg_max=16,
        anchor_generator=dict(type='AnchorGenerator', ratios=[1.], octave_base_scale=8,
                              scales_per_octave=1, strides=[4,8,16,32]),
        loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2., loss_weight=1.),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=.25),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.)),
    train_cfg=dict(assigner=dict(type='ATSSAssigner', topk=9), allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=.05,
                  nms=dict(type='nms', iou_threshold=.6), max_per_img=100),
    inference_nms='tb_soft', baseline_sigma=.7, baseline_topk=128,
    diagnostics=True, oracle_interval=4)
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=1e-4)
lr_config = dict(_delete_=True, policy='step', warmup='linear', warmup_iters=500, warmup_ratio=.001, step=[11,15])
runner = dict(type='EpochBasedRunner', max_epochs=16)
load_from = None
resume_from = None
custom_hooks = [dict(type='TQSupplementAuditHook', role='dense_joint', train_annotation="/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_train_only_tb.json", val_annotation="/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json", priority=45), dict(type='ExpMomentumEMAHook', momentum=.001, total_iter=1000, interval=1, skip_buffers=True, priority=49)]
experiment_role = 'joint_head_comparison_not_matched_candidates'
