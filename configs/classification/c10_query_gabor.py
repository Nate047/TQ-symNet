"""C3 + learnable quadrature Gabor energy; no LBP or Sym."""
_base_ = ['./c7_query_fresh_sym.py']
classification_options = dict(use_sym=False, use_lbp=False, use_gabor=True)
