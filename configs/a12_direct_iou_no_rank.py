"""Direct-IoU/full-representation control without rank (Q1 in the paper)."""
_base_ = ['./a8_direct_iou.py']
model = dict(quality_rank_weight=0., selector_rank_weight=0.)
