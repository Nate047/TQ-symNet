"""C3 + separately queried LBP and Gabor; no Sym or CLAHE."""
_base_ = ['./c7_query_fresh_sym.py']
classification_options = dict(use_sym=False, use_lbp=True, use_gabor=True)
