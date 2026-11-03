"""Capacity-matched ROI with old all-threshold target/pair rank/learned alpha."""
_base_ = ['./_common.py']
model = dict(type='TQSymNet',experiment_tag='R0_ROI_COMPLETE_RECIPE',
    quality_arch='pooled',quality_pooled_mlp_hidden=880,
    selector_detail=False,selector_edges=False)
