"""Both independent classification encoders trainable plus CLAHE."""
_base_ = ['./c20_original_sym_both.py']
data = dict(clahe=dict(clip_limit=2., tile_grid_size=(8, 8)))
