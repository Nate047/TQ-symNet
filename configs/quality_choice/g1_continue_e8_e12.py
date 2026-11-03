"""Resume LAST G1 epoch8 (not best epoch5), then train epochs9..12."""
_base_ = ['./g1_f2_geometry_winner.py']
custom_imports = dict(imports=[
    'TQ_SymNet.models', 'TQ_SymNet.alpha_experiment', 'TQ_SymNet.models.roi_query',
    'TQ_SymNet.models.threshold_distribution', 'TQ_SymNet.models.geometry_winner',
    'TQ_SymNet.geometry_winner_continuation'], allow_failed_imports=False)
model = dict(experiment_tag='G1_GEOMETRY_GOODSET_CONTINUE_E8_E12_LR1E5')
runner = dict(type='EpochBasedRunner', max_epochs=12)
lr_config = dict(_delete_=True, policy='TQG1Finetune', by_epoch=True,
    quality_lr=1e-5, start_epoch=8, end_epoch=12, steps_per_epoch=454,
    source_checkpoint='work_dirs/quality_choice/g1_s0/epoch_8.pth')
resume_from = 'work_dirs/quality_choice/g1_s0/epoch_8.pth'
load_from = None
