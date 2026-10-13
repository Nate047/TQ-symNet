"""Q0: 4-level 5x5 ROI mean pooling + geometry + MLP, direct IoU, no rank.

Compare representation against A12 (full representation, direct IoU, no rank).
This inexpensive comparator is not an implementation of IoU-Net/GFL.
"""
_base_ = ['./_matched_quality.py']
model = dict(quality_arch='pooled', quality_texture=False,
             quality_lbp=False, quality_gabor=False, quality_gaussian=False,
             selector_edges=False, selector_detail=False,
             quality_rank_weight=0., selector_rank_weight=0.,
             selector_quality_target='direct_iou')
