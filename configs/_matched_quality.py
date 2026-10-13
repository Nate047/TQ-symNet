"""Matched ablations: shared detector only, fresh quality network per command seed.

All groups load the detector trained on this exact 479-image cohort. Run the
selection helper after M2 training; its stable file points to the VAL-selected model.
"""
_base_ = ['./_common.py']
detector_checkpoint = 'work_dirs/balanced_v2_seed0_479_120_200/m2_s0/selected_detector.pth'
load_from = None
resume_from = None
model = dict(
    backbone=dict(pretrained=None),
    quality_arch='attention', quality_detach=True,
    quality_texture=True, quality_lbp=True, quality_gabor=True,
    quality_gaussian=True, selector_edges=True, selector_detail=True,
    selector_quality_target='coco_all_iou',
    quality_rank_weight=.10, selector_rank_weight=.10,
    selector_fusion_init=.5, selector_learned_gate=True,
    selector_gate_trainable=False, selector_quality_trainable=True,
    selector_edge_weight=0., selector_fusion_loss_weight=0.,
    diagnostics=True, oracle_interval=4)
custom_hooks = [
    dict(type='TQSplitContractHook',
         manifest='data/splits/tbx11k_479_120_200_balanced_v2_seed0/seed_0/manifest.json', priority=9),
    dict(type='TQDetectorInitHook', checkpoint=detector_checkpoint, priority='VERY_HIGH'),
    dict(type='TQSymNetDebugHook')]
experiment_role = 'matched_quality_training'
