"""C3 + fresh Sym + separate LBP/Gabor query branches, raw classifier input."""
_base_ = ['./c7_query_fresh_sym.py']
classification_options = dict(use_sym=True, use_lbp=True, use_gabor=True)
