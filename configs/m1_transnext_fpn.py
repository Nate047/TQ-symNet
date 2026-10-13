"""M1: compare to M0 for backbone/pretraining package, M2 for the neck."""
_base_ = ['./foundation_transnext_fpn.py']
model = dict(experiment_tag='TQ_M1_TRANSNEXT_FPN')
