"""C14 with only the independent classification CrossScale neck trainable."""
_base_ = ['./c14_original_sym_frozen.py']
encoder_training = dict(train_neck=True)
