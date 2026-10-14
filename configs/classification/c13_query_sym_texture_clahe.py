"""Full classification priors + deterministic CLAHE; compare directly to C12."""
_base_ = ['./c12_query_sym_texture.py']
data = dict(clahe=dict(clip_limit=2., tile_grid_size=(8, 8)))
