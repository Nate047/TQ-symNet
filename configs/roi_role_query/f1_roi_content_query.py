"""F1: visual 5x5 ROI cells seed role Query, before retained global reading."""
_base_ = ['./_common.py']
model = dict(type='TQSymNetROIQuery', experiment_tag='ROI_QUERY_F1_CONTENT',
             roi_pool_size=5, roi_paired_boundary=False)
