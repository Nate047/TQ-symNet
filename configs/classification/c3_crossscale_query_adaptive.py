"""C2 with a per-image, per-class learned scale gate; same frozen source."""
_base_ = ['./c2_crossscale_query_fixed.py']
classification_options = dict(scale_fusion='learned')
selection_metric = 'accuracy'
diagnostic_interval = 2
