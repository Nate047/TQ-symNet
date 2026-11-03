"""Original 599/200 matched ROI controls; never inherit the newer split."""
_base_ = ['../publication_alpha/_original_a0.py']
custom_imports = dict(imports=['TQ_SymNet.models', 'TQ_SymNet.alpha_experiment', 'TQ_SymNet.supplement'], allow_failed_imports=False)
model = dict(quality_arch='pooled', quality_hidden=96,
    quality_texture=False, quality_lbp=False, quality_gabor=False,
    quality_gaussian=False, selector_edges=False, selector_detail=False,
    quality_rank_weight=0., selector_rank_weight=0.,
    selector_quality_target='direct_iou', selector_fusion_init=.5,
    selector_gate_trainable=False)
custom_hooks = [dict(type='TQOriginalCohortInitHook', priority='VERY_HIGH', checkpoint='work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth', checkpoint_sha256='4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74', train_annotation="/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_train_only_tb.json", val_annotation="/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json", train_sha256='9d1b19eb8985e281c4767148fac466445844b764db3dab2f7ca635503470971c', val_sha256='bdec993506af15bf72010b32ec82ac279615ccc4c51ad9ca9438a7b9638d6e9d'), dict(type='TQSymNetDebugHook'), dict(type='TQSupplementAuditHook', role='matched_roi', train_annotation="/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_train_only_tb.json", val_annotation="/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json", priority=45)]
