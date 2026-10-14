"""Combination control: C4 temperature and C5 consistency."""
_base_ = ['./c4_crossscale_adaptive_temperature.py']
consistency = dict(weight=0.1, brightness_pixels=5., contrast=0.05)
