"""Both independent classification encoders trainable."""
_base_ = ['./c14_original_sym_frozen.py']
encoder_training = dict(train_backbone=True, train_neck=True)
