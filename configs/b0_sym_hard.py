"""Inference only: same detector checkpoint/candidates as A, MAX-Sym hard NMS."""
_base_ = ['./train_detector.py']
model = dict(experiment_tag='TQ_B0_SYM_HARD', backbone=dict(pretrained=None),
             inference_nms='tb_hard', inference_only=True, baseline_topk=128)
custom_hooks = []
evaluation = dict(save_best=None)
experiment_role = 'inference_only'
