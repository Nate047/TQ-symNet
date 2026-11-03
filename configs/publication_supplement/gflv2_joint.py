"""GFocalV2 mirrored-source adapter; four P2-P5 levels, not COCO replay."""
_base_ = ['./_dense.py']
model = dict(experiment_tag='PUB_GFLV2_JOINT', bbox_head=dict(type='TQGFLV2Head', reg_topk=4, reg_channels=64, add_mean=True))
