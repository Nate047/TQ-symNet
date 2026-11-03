"""Eight NEW epochs, constant LR; retain learned epoch-7 quality weights."""
_base_ = ['./_original_a0.py']
custom_imports = dict(imports=['TQ_SymNet.models', 'TQ_SymNet.alpha_only'],
                      allow_failed_imports=False)
model = dict(type='TQSymNetAlphaOnly', experiment_tag='PUB_ALPHA_ONLY',
             selector_quality_trainable=False, selector_gate_trainable=True)
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.)
optimizer_config = dict(grad_clip=None)
lr_config = dict(_delete_=True, policy='Fixed')
runner = dict(type='EpochBasedRunner', max_epochs=8)
load_from = None
resume_from = None
custom_hooks = [
    dict(type='TQAlphaContinuationInitHook', priority='VERY_HIGH',
         quality_checkpoint='work_dirs/publication_alpha/a1_learned_s0/best_official_tb_bbox_mAP_epoch_7.pth',
         expected_epoch=7,
         checkpoint='work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth',
         checkpoint_sha256='4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74',
         train_annotation='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_train_only_tb.json',
         val_annotation='/home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json',
         train_sha256='9d1b19eb8985e281c4767148fac466445844b764db3dab2f7ca635503470971c',
         val_sha256='bdec993506af15bf72010b32ec82ac279615ccc4c51ad9ca9438a7b9638d6e9d'),
    dict(type='TQScalarAlphaAuditHook', interval=50, priority='NORMAL')]
