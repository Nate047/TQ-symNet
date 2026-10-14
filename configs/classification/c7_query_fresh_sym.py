"""C3 + fresh three-class mirror attention; detector and neck remain frozen."""
_base_ = ['./c3_crossscale_query_adaptive.py']
classification_head = 'prior_query'
classification_options = dict(use_sym=True, use_lbp=False, use_gabor=False,
                              texture_grid=32)
