"""Original 599/200 M2; new quality training, NOT old quality continuation."""
_base_ = ['../publication_alpha/_original_a0.py']
custom_imports = dict(imports=['TQ_SymNet.models', 'TQ_SymNet.alpha_experiment',
                              'TQ_SymNet.models.roi_query',
                              'TQ_SymNet.models.threshold_distribution'], allow_failed_imports=False)
model = dict(selector_gate_trainable=True, selector_fusion_init=.5)
# Copy the original hooks exactly, adding the structural contract at VERY_LOW
# so internal val diagnostics are printed AFTER the ordinary evaluation hook.
custom_hooks = [
    dict(type='TQOriginalCohortInitHook', priority='VERY_HIGH',
         checkpoint='work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth',
         checkpoint_sha256='4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74',
         train_annotation='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_train_only_tb.json',
         val_annotation='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json',
         train_sha256='9d1b19eb8985e281c4767148fac466445844b764db3dab2f7ca635503470971c',
         val_sha256='bdec993506af15bf72010b32ec82ac279615ccc4c51ad9ca9438a7b9638d6e9d'),
    dict(type='TQSymNetDebugHook'),
    dict(type='TQScalarAlphaAuditHook', interval=50, priority='NORMAL'),
    dict(type='TQROIQueryAuditHook', priority='VERY_LOW'),
    dict(type='TQThresholdDistributionAuditHook', priority='VERY_LOW')]
load_from = None
resume_from = None
