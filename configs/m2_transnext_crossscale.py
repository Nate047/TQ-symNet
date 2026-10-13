"""M2: retained foundation, no quality network or H2, trained from ImageNet initialization."""
_base_ = ['./train_detector.py']
model = dict(experiment_tag='TQ_M2_TRANSNEXT_CROSSSCALE', inference_nms='tb_hard')
experiment_role = 'foundation_training'
