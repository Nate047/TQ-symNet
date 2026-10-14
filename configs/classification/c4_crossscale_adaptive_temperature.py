"""C3 plus image/class-dependent spatial attention temperature."""
_base_ = ['./c3_crossscale_query_adaptive.py']
classification_options = dict(attention_temperature=True)
