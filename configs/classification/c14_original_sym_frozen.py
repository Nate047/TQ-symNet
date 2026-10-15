"""C12 priors with the official fresh Sym block; both encoders frozen."""
_base_ = ['./c12_query_sym_texture.py']
classification_options = dict(sym_mode='original')
encoder_training = dict(train_backbone=False, train_neck=False)
save_encoder = True
# Same effective image batch and optimizer-update count for all C14-C21.
# No AMP. Activations are released after each microbatch backward.
data = dict(batch_size=2, val_batch_size=2)
training = dict(accumulation_steps=4)
optimizer = dict(backbone_lr_mult=.01, neck_lr_mult=.1)
