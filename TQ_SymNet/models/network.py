"""Public model and training diagnostics for the retained MAX/all-IoU recipe."""
import math
import torch
from mmcv.runner import HOOKS
from mmdet.models.builder import DETECTORS
from .detail import RetinaNAMDetailSelector, DetailSelectorDebugHook
from .light_quality import PooledQualityScorer


@DETECTORS.register_module()
class TQSymNet(RetinaNAMDetailSelector):
    def __init__(self, *args, diagnostics=True, oracle_interval=4,
                 quality_arch='attention', selector_fusion_init=None,
                 quality_pooled_mlp_hidden=None, **kwargs):
        if kwargs.get('selector_tb_fusion', 'max') != 'max':
            raise ValueError('This release retains MAX fusion, not LSE')
        if kwargs.get('selector_adaptive_subtype_gate', False):
            raise ValueError('Adaptive subtype mixing is not in the retained recipe')
        super().__init__(*args, **kwargs)
        if quality_arch not in ('attention', 'pooled'):
            raise ValueError('quality_arch must be attention or pooled')
        self.quality_arch = quality_arch
        if quality_arch != 'pooled' and quality_pooled_mlp_hidden is not None:
            raise ValueError('quality_pooled_mlp_hidden is only valid for pooled heads')
        if quality_arch == 'pooled':
            previous = self.candidate_quality
            with torch.random.fork_rng(devices=[]):
                self.candidate_quality = PooledQualityScorer(
                    previous.hidden, previous.chunk, mlp_hidden=quality_pooled_mlp_hidden)
            for parameter in self.candidate_quality.parameters():
                parameter.requires_grad_(self.selector_quality_trainable)
            self.candidate_quality.fusion_gate_logit = torch.nn.Parameter(
                previous.fusion_gate_logit.detach().clone(),
                requires_grad=previous.fusion_gate_logit.requires_grad)
            self.candidate_quality.tb_lse_temperature_raw = torch.nn.Parameter(
                previous.tb_lse_temperature_raw.detach().clone(), requires_grad=False)
            self.selector_detail = False
        if selector_fusion_init is not None:
            alpha = float(selector_fusion_init)
            if not .1 < alpha < .9:
                raise ValueError('Fusion initialization must be inside (.1,.9)')
            value = (alpha-.1)/.8
            with torch.no_grad():
                self.candidate_quality.fusion_gate_logit.fill_(math.log(value/(1-value)))
        self.diagnostics = bool(diagnostics)
        self.oracle_interval = int(oracle_interval)
        if self.oracle_interval < 1:
            raise ValueError('oracle_interval must be positive')
        self.integrated_build_id = 'TQ-SymNet-MAX-all-IoU-mask-free'

    def _image_memory(self, features, img, meta, original, index):
        if self.quality_arch == 'pooled':
            maps = [value[index:index+1].detach() for value in features[:4]]
            return self.candidate_quality.prepare(maps, None,
                meta['img_shape'][:2], img.shape[-2:])
        return super()._image_memory(features, img, meta, original, index)

    def v50_begin_oracle(self, epoch):
        self.quality_epoch = int(epoch)
        self.quality_cache = []
        self.detail_debug = []
        self.candidate_quality.attention_stats = []
        self.subtype_gate_stats = []
        self.quality_collect = (self.diagnostics and self.inference_mode == 'tb'
                                and (epoch % self.oracle_interval == 0
                                     or getattr(self, 'force_oracle', False)))

    def v50_finish_oracle(self, dataset, runner):
        if self.quality_collect:
            return super().v50_finish_oracle(dataset, runner)
        runner.logger.info('[TQ-VAL] epoch=%d mode=%s oracleDeferred=True '
                           'oracleEvery=%d; normalCOCOEvaluationUnchanged=True',
                           self.quality_epoch, self.inference_mode, self.oracle_interval)
        self.quality_cache = []
        self.detail_debug = []
        self.candidate_quality.attention_stats = []
        self.subtype_gate_stats = []

    def simple_test(self, *args, **kwargs):
        result = super().simple_test(*args, **kwargs)
        # CLI/hidden-test inference must not accumulate per-chunk diagnostic tensors.
        if not self.quality_collect:
            self.detail_debug = []
            self.candidate_quality.attention_stats = []
            self.subtype_gate_stats = []
        return result


@HOOKS.register_module()
class TQSymNetDebugHook(DetailSelectorDebugHook):
    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        alpha = .1 + .8 * model.candidate_quality.fusion_gate_logit.sigmoid()
        runner.logger.info('[TQ-SYMNET-BUILD] id=%s masks=False H2=False '
            'mainFrozen=%s qualityTarget=%s rankScope=qualityOnly '
            'score=MAX-logit+quality alpha=%.6f texture=%s lbp=%s gabor=%s '
            'gaussian=%s sideQuery=%s detail=%s legacyQualityRankInactive=%.3f activeQualityRank=%.3f '
            'inferenceMode=%s oracleEvery=%d architecture=%s', model.integrated_build_id,
            model.quality_detach, model.selector_quality_target,
            float(alpha.detach()), model.candidate_quality.texture,
            model.candidate_quality.lbp_enabled, model.candidate_quality.gabor_enabled,
            model.candidate_quality.gaussian, getattr(model.candidate_quality, 'edges', False),
            model.selector_detail, model.quality_rank_weight,
            model.selector_rank_weight, model.inference_mode, model.oracle_interval,
            model.quality_arch)
        if not model.quality_detach:
            raise RuntimeError('The retained TQ-SymNet quality stage requires a frozen detector')
