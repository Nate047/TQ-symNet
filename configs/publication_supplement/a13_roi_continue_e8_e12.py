"""A13 additional-budget continuation: resume epoch8, train epochs9..12.

Keep the original eight-epoch controlled result unchanged. This is a declared
small-LR restart, not a retrospectively extended 12-epoch cosine schedule.
"""
_base_ = ['./a13_roi_capacity.py']
custom_imports = dict(imports=[
    'TQ_SymNet.models','TQ_SymNet.alpha_experiment','TQ_SymNet.supplement',
    'TQ_SymNet.roi_continuation'],allow_failed_imports=False)
model = dict(experiment_tag='PUB_A13_CONTINUE_E8_E12_LR1E5')
runner = dict(type='EpochBasedRunner',max_epochs=12)
lr_config = dict(_delete_=True,policy='TQROIFinetune',by_epoch=True,
                 quality_lr=1e-5,start_epoch=8,end_epoch=12,steps_per_epoch=454,
                 source_dir='work_dirs/publication_supplement/a13_s0')
resume_from = 'work_dirs/publication_supplement/a13_s0/best_official_tb_bbox_mAP_epoch_8.pth'
load_from = None
