"""Trainable classification backbone; frozen neck transmits input gradients."""
_base_ = ['./c14_original_sym_frozen.py']
encoder_training = dict(train_backbone=True)
