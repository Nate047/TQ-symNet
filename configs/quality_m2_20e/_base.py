"""Matched quality experiments on the locked 20-epoch M2 VAL-best epoch 17."""
_base_ = ['../_matched_quality.py']
detector_checkpoint = 'work_dirs/balanced_v2_seed0_479_120_200/m2_20e_s0/best_official_tb_bbox_mAP_epoch_17.pth'
matched_reference = 'configs/quality_m2_20e/a0_full.py'
enforce_detector_provenance = True
custom_hooks = [
    dict(type='TQSplitContractHook',
         manifest='data/splits/tbx11k_479_120_200_balanced_v2_seed0/seed_0/manifest.json', priority=9),
    dict(type='TQDetectorInitHook', checkpoint=detector_checkpoint, priority='VERY_HIGH'),
    dict(type='TQSymNetDebugHook')]
