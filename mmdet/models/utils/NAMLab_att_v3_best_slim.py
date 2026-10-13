"""Strict, rollback-safe profiles for the measured NAMLab v3.18 module.

This module intentionally contains no second implementation of CrossScale,
SymFormer, or FDR.  Duplicating those modules would make historical weights
and numerics drift.  Instead, it provides a shallow experiment interface over
``NAMLab_att_v3.py`` and rejects combinations that have already failed the
controlled ablations.

Deployment contract:
    1. ``NAMLab_att_v3.py`` must be the v3.18 implementation.
    2. Import this module with ``custom_imports`` from the experiment config.
    3. Use the V319 registry names below, never the unrestricted V3 names.

The default profile is the reproducible G2 stack:
    original CrossScale + left/right Sym prior + FDR(weight=0.05)

Only two unresolved seams remain independently testable:
    * FreqFusion top-down resampling
    * C5-gated C2 regression detail bypass
"""

import logging
import math

from mmdet.models.builder import DETECTORS, HEADS, NECKS

try:
    from .NAMLab_att_v3 import (RetinaNAMAttV3, RetinaNAMGuideAttHeadV3,
                                TBFusionCrossScaleNeck)
except ImportError as error:
    raise ImportError(
        'NAMLab_att_v3_best_slim requires the v3.18 implementation at '
        'mmdet/models/utils/NAMLab_att_v3.py. Deploy '
        'NAMLab_att_v3_dropin.py there before importing the V319 config.'
    ) from error


NAMLAB_SLIM_BUILD_ID = 'v3.19-best-g2-strict-slim-20260804'


def _reject_enabled(kwargs, names, owner):
    enabled = [name for name in names if bool(kwargs.get(name, False))]
    if enabled:
        raise ValueError(
            f'{owner} rejects failed auxiliary modules: {enabled}. '
            'Use the archived unrestricted v3.18 implementation for old '
            'protocol replays.')


def _require_zero(kwargs, names, owner):
    nonzero = [
        name for name in names
        if kwargs.get(name) is not None
        and not math.isclose(float(kwargs[name]), 0.0, abs_tol=1e-12)
    ]
    if nonzero:
        raise ValueError(
            f'{owner} requires disabled auxiliary losses, got non-zero '
            f'values for {nonzero}.')


@NECKS.register_module(force=True)
class TBFusionCrossScaleNeckV319(TBFusionCrossScaleNeck):
    """Original CrossScale with two isolated, unresolved test switches.

    Learnable proposal/base weights are deliberately unavailable: G3 and H0
    show that they erase the gain of FDR.  FreqFusion and the regression-detail
    bypass may each be tested, but not together until either is positive alone.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('use_top_down', True)
        kwargs.setdefault('learnable_fusion_weights', False)
        kwargs.setdefault('freqfusion_top_down', False)
        kwargs.setdefault('reg_detail_bypass', False)

        if not bool(kwargs['use_top_down']):
            raise ValueError('V319 requires the measured bidirectional path')
        if bool(kwargs['learnable_fusion_weights']):
            raise ValueError(
                'V319 forbids learnable_fusion_weights: the controlled '
                'G3 ablation was worse than G2 in AP and AP50.')
        if (bool(kwargs['freqfusion_top_down'])
                and bool(kwargs['reg_detail_bypass'])):
            raise ValueError(
                'Test FreqFusion and regression detail independently. Their '
                'H3 combination was negative and is intentionally blocked.')

        super().__init__(*args, **kwargs)
        self.slim_build_id = NAMLAB_SLIM_BUILD_ID


@HEADS.register_module(force=True)
class RetinaNAMGuideAttHeadV319(RetinaNAMGuideAttHeadV3):
    """Sym head locked to G2 FDR, with an optional regression-detail seam."""

    _REMOVED_SWITCHES = (
        'head_pc_modulation',
        'bbox_contrastive',
        'boxalign',
        'edge_supervision',
    )

    def __init__(self, *args, **kwargs):
        _reject_enabled(kwargs, self._REMOVED_SWITCHES, self.__class__.__name__)
        kwargs.setdefault('head_pc_modulation', False)
        kwargs.setdefault('bbox_contrastive', False)
        kwargs.setdefault('boxalign', False)
        kwargs.setdefault('edge_supervision', False)
        kwargs.setdefault('token_dropout', 0.0)
        kwargs.setdefault('d_fine_style', True)
        kwargs.setdefault('d_fine_reg_max', 16)
        kwargs.setdefault('d_fine_refine_init', 0.25)
        kwargs.setdefault('d_fine_refine_max', 1.0)
        kwargs.setdefault('d_fine_loss_weight', 0.05)
        kwargs.setdefault('reg_detail_bypass', False)

        if not bool(kwargs['d_fine_style']):
            raise ValueError('V319 G2 requires d_fine_style=True')
        if int(kwargs['d_fine_reg_max']) != 16:
            raise ValueError('V319 G2 was measured with d_fine_reg_max=16')
        if not math.isclose(
                float(kwargs['d_fine_loss_weight']), 0.05,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('V319 G2 requires d_fine_loss_weight=0.05')

        positional = kwargs.get('positional_encoding')
        if isinstance(positional, dict) and bool(
                positional.get('use_upper_prior', False)):
            raise ValueError(
                'V319 keeps only the original left/right Sym prior; the '
                'upper-location prior was negative in the P/U ablation.')

        super().__init__(*args, **kwargs)
        self.slim_build_id = NAMLAB_SLIM_BUILD_ID


@DETECTORS.register_module(force=True)
class RetinaNAMAttV319(RetinaNAMAttV3):
    """Detector profile that removes every disproven NAM/PC objective.

    The unrestricted v3.18 module remains the deep implementation and rollback
    seam.  This shallow class exists to keep a training config from silently
    drifting away from the measured G2 experiment.
    """

    _REMOVED_SWITCHES = ('pc_branch',)
    _REMOVED_LOSSES = (
        'contrast_loss_weight',
        'region_contrast_loss_weight',
        'boxalign_loss_weight',
    )

    def __init__(self, *args, **kwargs):
        _reject_enabled(kwargs, self._REMOVED_SWITCHES, self.__class__.__name__)
        _require_zero(kwargs, self._REMOVED_LOSSES, self.__class__.__name__)
        kwargs.setdefault('pc_branch', False)
        kwargs.setdefault('contrast_loss_weight', 0.0)
        kwargs.setdefault('region_contrast_loss_weight', 0.0)
        kwargs.setdefault('boxalign_loss_weight', 0.0)
        kwargs.setdefault('input_preprocess', 'raw_imagenet')
        kwargs.setdefault('experiment_tag', 'V319_G2_STRICT_SLIM')

        if kwargs['input_preprocess'] != 'raw_imagenet':
            raise ValueError(
                'V319 reproduces raw 512x512 CXR + ImageNet normalization; '
                'ROI, CLAHE and histogram remapping are not allowed.')

        super().__init__(*args, **kwargs)
        if not isinstance(self.neck, TBFusionCrossScaleNeckV319):
            raise TypeError('RetinaNAMAttV319 requires the V319 neck')
        if not isinstance(self.bbox_head, RetinaNAMGuideAttHeadV319):
            raise TypeError('RetinaNAMAttV319 requires the V319 head')

        cross_scale = self.neck.cross_scale
        logging.getLogger('mmdet').info(
            '[NAM-SLIM] id=%s profile=G2 cs_weighted=%s freqfusion=%s '
            'reg_detail=%s dfine=%s dfine_w=%.3f pc=%s region_aux=%s',
            NAMLAB_SLIM_BUILD_ID,
            cross_scale.learnable_fusion_weights,
            cross_scale.freqfusion_top_down,
            self.neck.use_reg_detail_bypass,
            self.bbox_head.use_d_fine_style,
            self.bbox_head.d_fine_loss_weight,
            self.use_pc_branch,
            self.use_region_auxiliaries)


__all__ = [
    'NAMLAB_SLIM_BUILD_ID',
    'TBFusionCrossScaleNeckV319',
    'RetinaNAMGuideAttHeadV319',
    'RetinaNAMAttV319',
]
