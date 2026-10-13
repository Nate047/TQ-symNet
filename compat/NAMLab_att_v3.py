"""OPTIONAL server import shim. Back up the old monolith before copying this.

Destination: mmdet/models/utils/NAMLab_att_v3.py.
Only bootstrap exports used by this MMDetection tree are provided. Old NAM
experiment configs are intentionally not supported by this shim. Run numerical
comparison with the original monolith first; the shim is NOT installed locally.
"""
from TQ_SymNet.models.head import TQSymHead as RetinaNAMGuideAttHeadV3
from TQ_SymNet.models.detector import TQDetector as RetinaNAMAttV3

__all__ = ['RetinaNAMGuideAttHeadV3', 'RetinaNAMAttV3']
