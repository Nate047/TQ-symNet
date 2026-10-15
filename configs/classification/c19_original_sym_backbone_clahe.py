"""Trainable classification backbone plus CLAHE."""
_base_ = ['./c18_original_sym_backbone.py']
data = dict(clahe=dict(clip_limit=2., tile_grid_size=(8, 8)))
