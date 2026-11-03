"""Main combined probe: F2 + signed geometry auxiliary + good-set choice."""
_base_ = ['./_common.py']
model = dict(type='TQSymNetGeometryWinner',experiment_tag='G1_F2_GEOMETRY_GOODSET',
    roi_pool_size=5,roi_paired_boundary=True,geometry_weight=.05,geometry_beta=.05,
    geometry_ambiguity_gap=.1,winner_temperature=.5,winner_max=4)
