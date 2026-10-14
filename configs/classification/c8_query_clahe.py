"""C3 only; deterministic CLAHE in classification train/val/test, no detector change."""
_base_ = ['./c3_crossscale_query_adaptive.py']
data = dict(clahe=dict(clip_limit=2., tile_grid_size=(8, 8)))
