"""Inference only: B0 candidates, exact TQ Gaussian Soft-NMS (no extra score floor)."""
_base_ = ['./b0_sym_hard.py']
model = dict(experiment_tag='TQ_B1_SYM_SOFT', inference_nms='tb_soft',
             baseline_sigma=.7, baseline_topk=128)
