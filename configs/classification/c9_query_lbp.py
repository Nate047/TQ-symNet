"""C3 + spatial LBP59 embeddings with learnable bin weights."""
_base_ = ['./c7_query_fresh_sym.py']
classification_options = dict(use_sym=False, use_lbp=True, use_gabor=False)
