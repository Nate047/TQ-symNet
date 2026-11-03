"""Resume the completed F2-v2 epoch8 best, four explicit small-LR epochs."""
_base_ = ['./f2_roi_paired_query.py']
custom_imports = dict(imports=[
    'TQ_SymNet.models','TQ_SymNet.alpha_experiment','TQ_SymNet.models.roi_query',
    'TQ_SymNet.models.threshold_distribution','TQ_SymNet.roi_query_continuation'],
    allow_failed_imports=False)
model = dict(experiment_tag='F2_QUERYGATE_V2_CONTINUE_E8_E12_LR1E5')
runner = dict(type='EpochBasedRunner',max_epochs=12)
lr_config = dict(_delete_=True,policy='TQF2Finetune',by_epoch=True,
    quality_lr=1e-5,start_epoch=8,end_epoch=12,steps_per_epoch=454,
    source_checkpoint='work_dirs/roi_threshold/f2_s0/best_official_tb_bbox_mAP_epoch_8.pth')
resume_from = 'work_dirs/roi_threshold/f2_s0/best_official_tb_bbox_mAP_epoch_8.pth'
load_from = None
