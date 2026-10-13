"""Optional fourth cell of the backbone x neck 2x2 study."""
_base_ = ['./m2_transnext_crossscale.py']
model = dict(
    experiment_tag='TQ_M3_P2T_CROSSSCALE',
    backbone=dict(_delete_=True, type='TQP2TSmall', out_indices=(0, 1, 2, 3),
                  pretrained='pretrained/p2t_small.pth'),
    neck=dict(in_channels=[64, 128, 320, 512]))
