"""Frozen detector CrossScale; class queries with equal P3/P4/P5 weights."""
_base_ = ['./c1_crossscale_hybrid_pool.py']
classification_head = 'class_query'
classification_options = dict(_delete_=True, in_channels=256, embed_dim=128,
                              heads=4, scale_fusion='fixed', dropout=0.1)
