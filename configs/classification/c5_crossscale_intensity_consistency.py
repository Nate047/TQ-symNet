"""C3 plus mild brightness/contrast consistency, without changing val/test."""
_base_ = ['./c3_crossscale_query_adaptive.py']
consistency = dict(weight=0.1, brightness_pixels=5., contrast=0.05)
