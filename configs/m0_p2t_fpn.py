"""P2T-S/FPN foundation control with the SAME TQ Sym/FDR head and P2-P5 anchors.

This is a controlled backbone comparison, not a byte-identical official Sym
replay (the official model uses a different head/anchor/training recipe).
"""
_base_ = ['./foundation_transnext_fpn.py']
model = dict(
    experiment_tag='TQ_M0_P2T_FPN',
    backbone=dict(_delete_=True, type='TQP2TSmall', out_indices=(0, 1, 2, 3),
                  pretrained='pretrained/p2t_small.pth'),
    neck=dict(in_channels=[64, 128, 320, 512]))
