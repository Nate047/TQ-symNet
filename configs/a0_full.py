"""Primary TQ-SymNet quality run: all retained quality routes enabled."""
_base_ = ['./_matched_quality.py']
model = dict(quality_texture=True, quality_lbp=True, quality_gabor=True,
             quality_gaussian=True, selector_edges=True, selector_detail=True,
             quality_rank_weight=.10, selector_rank_weight=.10,
             selector_quality_target='coco_all_iou')
