"""Optional official OpenMMLab GFL-v1 control for the DGQP comparison."""
_base_ = ['./_dense.py']
model = dict(experiment_tag='PUB_GFL_JOINT')
