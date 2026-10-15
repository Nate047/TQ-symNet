"""Trainable classification neck plus CLAHE."""
_base_ = ['./c16_original_sym_neck.py']
data = dict(clahe=dict(clip_limit=2., tile_grid_size=(8, 8)))
