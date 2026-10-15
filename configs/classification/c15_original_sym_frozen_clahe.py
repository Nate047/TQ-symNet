"""Frozen original-Sym control with matched CLAHE input."""
_base_ = ['./c14_original_sym_frozen.py']
data = dict(clahe=dict(clip_limit=2., tile_grid_size=(8, 8)))
