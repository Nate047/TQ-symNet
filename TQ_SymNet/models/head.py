"""TQ-SymNet: extracted measured implementations; see PROVENANCE.json."""
import math
import logging
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.retina_guide_att_head import RetinaGuideAttHead
from .distribution import (_distribution_refinement_expectation,
                           _symmetric_distribution_focal_loss)
from .checkpoint_compat import (_RegionTransformerBranchV3_3,
    _LogGaborPhaseCongruency, _reset_boxalign_debug_state,
    _PerPixelLBPBranch, _BoundaryResponsiveGate, _RegressionDetailInjector)

class _HeadCore(RetinaGuideAttHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.1,
                 token_dropout: float = 0.15,
                 spatial_temp: float = 1.5,
                 max_regions: int = 200,
                 ema_momentum: float = 0.0,
                 positive_purity_threshold: float = 0.5,
                 negative_purity_threshold: float = 0.05,
                 minimum_positive_purity: float = math.pi / 4,
                 ensure_gt_positive: bool = True,
                 contrast_temperature: float = 0.07,
                 purity_weight_floor: float = 0.25,
                 hard_negative_ratio: float = 3.0,
                 hard_negative_expansion: float = 1.5,
                 min_hard_negatives: int = 16,
                 far_negative_weight: float = 0.25,
                 boundary_rank_margin: float = 0.10,
                 boundary_rank_weight: float = 0.0,
                 region_in_channels: int = 72,
                 boxalign: bool = False,
                 boxalign_dim: int = 128,
                 boxalign_temperature: float = 0.10,
                 boxalign_min_iou: float = 0.30,
                 boxalign_levels=(0, 1, 2),
                 boxalign_max_anchors_per_level: int = 2,
                 boxalign_loss_ratio: float = 0.5,
                 debug_interval: int = 50,
                 deep_debug_interval: int = 500,
                 cross_scale: bool = False,
                 pc_branch: bool = False,
                 head_pc_modulation: bool = False,
                 head_pc_residual_init: float = 1e-2,
                 head_pc_residual_max: float = 0.1,
                 bbox_contrastive: bool = False,
                 bbox_contrast_loss_weight: float = 0.01,
                 bbox_contrast_dim: int = 128,
                 bbox_contrast_temperature: float = 0.10,
                 bbox_contrast_pos_iou: float = 0.5,
                 bbox_contrast_neg_iou: float = 0.4,
                 bbox_contrast_max_pos_per_gt: int = 16,
                 bbox_contrast_hard_negative_ratio: float = 3.0,
                 bbox_contrast_min_negatives: int = 32,
                 reg_detail_bypass: bool = False,
                 reg_detail_levels=(2, 3),
                 reg_detail_residual_init: float = 1e-2,
                 reg_detail_residual_max: float = 0.1,
                 d_fine_style: bool = False,
                 d_fine_reg_max: int = 16,
                 d_fine_refine_init: float = 0.25,
                 d_fine_refine_max: float = 1.0,
                 d_fine_loss_weight: float = 0.25,
                 feat_channels: int = 256,
                 embed_dims: int = 256,
                 dims_radio: float = 2.0,
                 num_query: int = 100,
                 num_reg_fcs: int = 2,
                 transformer=None,
                 positional_encoding=None,
                 stacked_convs: int = 4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=None,
                 bbox_coder=None,
                 reg_decoded_bbox: bool = False,
                 loss_cls=None,
                 loss_bbox=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.v3_dropout = dropout
        self.v3_drop_path_rate = drop_path_rate
        self.v3_token_dropout = token_dropout
        self.v3_spatial_temp = spatial_temp
        self.v3_max_regions = max_regions
        self.v3_ema_momentum = ema_momentum
        self.v3_positive_purity_threshold = positive_purity_threshold
        self.v3_negative_purity_threshold = negative_purity_threshold
        self.v3_minimum_positive_purity = minimum_positive_purity
        self.v3_ensure_gt_positive = ensure_gt_positive
        self.v3_contrast_temperature = contrast_temperature
        self.v3_purity_weight_floor = purity_weight_floor
        self.v3_hard_negative_ratio = hard_negative_ratio
        self.v3_hard_negative_expansion = hard_negative_expansion
        self.v3_min_hard_negatives = min_hard_negatives
        self.v3_far_negative_weight = far_negative_weight
        self.v3_boundary_rank_margin = boundary_rank_margin
        self.v3_boundary_rank_weight = boundary_rank_weight
        self.region_in_channels = int(region_in_channels)
        if self.region_in_channels <= 0:
            raise ValueError('region_in_channels must be positive')
        self.use_boxalign = bool(boxalign)
        self.boxalign_dim = int(boxalign_dim)
        self.boxalign_temperature = float(boxalign_temperature)
        self.boxalign_min_iou = float(boxalign_min_iou)
        self.boxalign_levels = tuple(int(level) for level in boxalign_levels)
        self.boxalign_max_anchors_per_level = int(
            boxalign_max_anchors_per_level)
        self.boxalign_loss_ratio = float(boxalign_loss_ratio)
        if self.boxalign_dim <= 0:
            raise ValueError('boxalign_dim must be positive')
        if self.boxalign_dim != 128:
            raise ValueError(
                'boxalign_dim must match the 128-D NAM contrast head')
        if self.boxalign_temperature <= 0:
            raise ValueError('boxalign_temperature must be positive')
        if not 0 <= self.boxalign_min_iou <= 1:
            raise ValueError('boxalign_min_iou must be in [0, 1]')
        if not self.boxalign_levels or min(self.boxalign_levels) < 0:
            raise ValueError('boxalign_levels must contain valid levels')
        if self.boxalign_max_anchors_per_level < 1:
            raise ValueError(
                'boxalign_max_anchors_per_level must be positive')
        if self.boxalign_loss_ratio < 0:
            raise ValueError('boxalign_loss_ratio must be non-negative')
        self.debug_interval = int(debug_interval)
        self.deep_debug_interval = int(deep_debug_interval)
        if self.debug_interval < 0 or self.deep_debug_interval < 0:
            raise ValueError(
                'debug intervals must be non-negative')
        self._debug_step = 1
        legacy_edge_supervision = bool(
            kwargs.pop('edge_supervision', False))
        kwargs.pop('edge_loss_weight', None)
        if legacy_edge_supervision:
            raise ValueError(
                'edge_supervision was removed in NAMLab v3.9 because '
                'its LBP-derived target duplicated PC texture cues')
        self.use_head_pc_modulation = bool(head_pc_modulation)
        self.head_pc_residual_init = float(head_pc_residual_init)
        self.head_pc_residual_max = float(head_pc_residual_max)
        if not (0.0 < self.head_pc_residual_init
                < self.head_pc_residual_max):
            raise ValueError(
                'head_pc_residual_init must satisfy 0 < init < max')
        self.use_bbox_contrastive = bool(bbox_contrastive)
        if self.use_bbox_contrastive and self.use_boxalign:
            raise ValueError(
                'boxalign replaces the harmful anchor-anchor '
                'bbox_contrastive objective; do not enable both')
        self.bbox_contrast_loss_weight = float(
            bbox_contrast_loss_weight)
        self.bbox_contrast_dim = int(bbox_contrast_dim)
        self.bbox_contrast_temperature = float(
            bbox_contrast_temperature)
        self.bbox_contrast_pos_iou = float(bbox_contrast_pos_iou)
        self.bbox_contrast_neg_iou = float(bbox_contrast_neg_iou)
        self.bbox_contrast_max_pos_per_gt = int(
            bbox_contrast_max_pos_per_gt)
        self.bbox_contrast_hard_negative_ratio = float(
            bbox_contrast_hard_negative_ratio)
        self.bbox_contrast_min_negatives = int(
            bbox_contrast_min_negatives)
        if self.bbox_contrast_loss_weight < 0:
            raise ValueError(
                'bbox_contrast_loss_weight must be non-negative')
        if self.bbox_contrast_dim <= 0:
            raise ValueError('bbox_contrast_dim must be positive')
        if self.bbox_contrast_temperature <= 0:
            raise ValueError(
                'bbox_contrast_temperature must be positive')
        if not (0.0 <= self.bbox_contrast_neg_iou
                < self.bbox_contrast_pos_iou <= 1.0):
            raise ValueError(
                'Expected 0 <= bbox_contrast_neg_iou < '
                'bbox_contrast_pos_iou <= 1')
        if self.bbox_contrast_max_pos_per_gt < 2:
            raise ValueError(
                'bbox_contrast_max_pos_per_gt must be at least 2')
        if self.bbox_contrast_hard_negative_ratio <= 0:
            raise ValueError(
                'bbox_contrast_hard_negative_ratio must be positive')
        if self.bbox_contrast_min_negatives < 1:
            raise ValueError(
                'bbox_contrast_min_negatives must be positive')
        self.use_reg_detail_bypass = bool(reg_detail_bypass)
        self.reg_detail_levels = tuple(
            int(level) for level in reg_detail_levels)
        self.reg_detail_residual_init = float(
            reg_detail_residual_init)
        self.reg_detail_residual_max = float(
            reg_detail_residual_max)
        if not 0.0 < self.reg_detail_residual_init \
                < self.reg_detail_residual_max:
            raise ValueError(
                'reg_detail_residual_init must satisfy 0 < init < max')
        # D-FINE-style anchor adapter. This intentionally implements only
        # fine-grained distribution refinement (FDR): GO-LSD requires
        # iterative DETR decoder layers and must not be approximated by
        # distilling across semantically different FPN levels.
        self.use_d_fine_style = bool(d_fine_style)
        self.d_fine_reg_max = int(d_fine_reg_max)
        self.d_fine_refine_init = float(d_fine_refine_init)
        self.d_fine_refine_max = float(d_fine_refine_max)
        self.d_fine_loss_weight = float(d_fine_loss_weight)
        if self.d_fine_reg_max < 2:
            raise ValueError('d_fine_reg_max must be at least 2')
        if not (0.0 < self.d_fine_refine_init
                < self.d_fine_refine_max):
            raise ValueError(
                'd_fine_refine_init must satisfy 0 < init < max')
        if self.d_fine_loss_weight < 0:
            raise ValueError(
                'd_fine_loss_weight must be non-negative')
        self._last_d_fine_logits = None
        self._last_d_fine_base_preds = None
        self._dbg_d_fine_entropy = [0.0] * 4
        self._dbg_d_fine_loss_raw = 0.0
        self._dbg_d_fine_saturation = 0.0
        self._dbg_d_fine_target_abs = 0.0
        self._dbg_d_fine_expected_abs = [0.0] * 4
        self._last_reg_features = None
        # Deprecated compatibility arguments.  CrossScale now lives at
        # the neck seam. Detector-level pc_branch no longer injects C2.
        self._legacy_cross_scale = cross_scale
        self._legacy_pc_branch = pc_branch
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            embed_dims=embed_dims,
            dims_radio=dims_radio,
            num_query=num_query,
            num_reg_fcs=num_reg_fcs,
            transformer=transformer,
            positional_encoding=positional_encoding,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            reg_decoded_bbox=reg_decoded_bbox,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        super()._init_layers()

        # --- Region Transformer branch (dynamic token pooling from C2) ---
        self.nam_branch = _RegionTransformerBranchV3_3(
            feat_channels=self.region_in_channels,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.v3_dropout,
            drop_path_rate=self.v3_drop_path_rate,
            token_dropout=self.v3_token_dropout,
            spatial_temp=self.v3_spatial_temp,
            max_regions=self.v3_max_regions,
            ema_momentum=self.v3_ema_momentum,
            positive_purity_threshold=(
                self.v3_positive_purity_threshold),
            negative_purity_threshold=(
                self.v3_negative_purity_threshold),
            minimum_positive_purity=(
                self.v3_minimum_positive_purity),
            ensure_gt_positive=self.v3_ensure_gt_positive,
            contrast_temperature=self.v3_contrast_temperature,
            purity_weight_floor=self.v3_purity_weight_floor,
            hard_negative_ratio=self.v3_hard_negative_ratio,
            hard_negative_expansion=(
                self.v3_hard_negative_expansion),
            min_hard_negatives=self.v3_min_hard_negatives,
            far_negative_weight=self.v3_far_negative_weight,
            boundary_rank_margin=self.v3_boundary_rank_margin,
            boundary_rank_weight=self.v3_boundary_rank_weight)

        # PC is fused only into the bbox regression tower after SymFormer.
        # A bounded scale protects the pretrained detector at startup.
        if self.use_head_pc_modulation:
            self.lbp_branch = _PerPixelLBPBranch(
                in_channels=576, feat_channels=self.feat_channels)
            self.br_gate = _BoundaryResponsiveGate(
                channels=self.feat_channels,
                mod_scale_init=self.head_pc_residual_init,
                mod_scale_max=self.head_pc_residual_max,
                boundary_groups=18)
        else:
            self.lbp_branch = None
            self.br_gate = None

        self.reg_detail_injector = (
            _RegressionDetailInjector(
                channels=self.feat_channels,
                num_levels=self.anchor_generator.num_levels,
                levels=self.reg_detail_levels,
                residual_init=self.reg_detail_residual_init,
                residual_max=self.reg_detail_residual_max)
            if self.use_reg_detail_bypass else None)

        if self.use_d_fine_style:
            bins = self.d_fine_reg_max + 1
            # RetinaGuideAttHead invokes _init_layers before constructing
            # SymFormer. Preserve the global RNG so merely enabling this
            # zero-start branch does not silently change transformer
            # initialization and confound the ablation.
            with torch.random.fork_rng(devices=[]):
                self.d_fine_reg = nn.Conv2d(
                    self.feat_channels,
                    self.num_anchors * 4 * bins,
                    kernel_size=3,
                    padding=1)
            # A uniform symmetric distribution has zero expectation, so
            # the first forward is exactly the Retina/GIoU baseline.
            nn.init.zeros_(self.d_fine_reg.weight)
            nn.init.zeros_(self.d_fine_reg.bias)
            raw_init = math.atanh(
                self.d_fine_refine_init
                / self.d_fine_refine_max)
            self.d_fine_refine_raw = nn.Parameter(torch.full(
                (self.anchor_generator.num_levels,),
                raw_init,
                dtype=torch.float32))
        else:
            self.d_fine_reg = None
            self.register_parameter(
                'd_fine_refine_raw', None)

        # Detection-aligned auxiliary branch. It is evaluated only during
        # training: high-IoU anchors matched to the same GT instance are
        # positives; bbox-neighbouring background anchors are negatives.
        # No convolution or sampling is executed during inference.
        if self.use_bbox_contrastive:
            self.bbox_contrast_proj = nn.Conv2d(
                self.feat_channels,
                self.bbox_contrast_dim,
                kernel_size=1)
            self.bbox_anchor_embed = nn.Embedding(
                self.num_anchors,
                self.bbox_contrast_dim)
            nn.init.kaiming_normal_(
                self.bbox_contrast_proj.weight,
                mode='fan_out',
                nonlinearity='linear')
            nn.init.zeros_(self.bbox_contrast_proj.bias)
            nn.init.normal_(
                self.bbox_anchor_embed.weight, std=0.02)
        else:
            self.bbox_contrast_proj = None
            self.bbox_anchor_embed = None

        # C2/NAM instance prototypes are one view; matched P2-P4
        # regression-tower cells are the detection view. Region prototypes
        # are detached in the alignment loss so this branch updates
        # CrossScale/Sym/Retina without destabilising the C2 prototype.
        if self.use_boxalign:
            self.boxalign_proj = nn.Conv2d(
                self.feat_channels, self.boxalign_dim, kernel_size=1)
            self.boxalign_anchor_embed = nn.Embedding(
                self.num_anchors, self.boxalign_dim)
            nn.init.kaiming_normal_(
                self.boxalign_proj.weight,
                mode='fan_out',
                nonlinearity='linear')
            nn.init.zeros_(self.boxalign_proj.bias)
            nn.init.normal_(
                self.boxalign_anchor_embed.weight, std=0.02)
        else:
            self.boxalign_proj = None
            self.boxalign_anchor_embed = None
        _reset_boxalign_debug_state(
            self, self.anchor_generator.num_levels)

        # --- Learned LBP embedding: 59 uniform patterns → 32-dim dense vector ---
        # Replaces sparse one-hot (98.3% zeros) with dense learnable embedding.
        # 6 directions × 3 scales × 32 dims = 576 output channels (was 1062).
        self.lbp_embed = nn.Embedding(59, 32)

        # --- Source-faithful PC front-end for spatial BPPC ---
        # Six orientations match the paper (0,30,...,150 degrees). Three
        # log-Gabor scales supply odd/even quadrature responses, amplitude
        # normalization and noise-thresholded local phase congruency.
        self.phase_congruency = _LogGaborPhaseCongruency(
            num_scales=3,
            num_orientations=6,
            min_wavelength=3.0,
            scale_multiplier=2.1,
            sigma_on_frequency=0.55,
            noise_k=2.0)

        # Temperature hook: amplify gradient for spatial bias learning.
        self.nam_branch.spatial_bias_fn.temperature.register_hook(
            lambda g: g * 20.)

    def init_weights(self):
        super().init_weights()
        if self.use_d_fine_style:
            # Protect the zero-residual start even if a parent init_cfg
            # initializes all Conv2d layers.
            nn.init.zeros_(self.d_fine_reg.weight)
            nn.init.zeros_(self.d_fine_reg.bias)

    def effective_d_fine_scales(self) -> torch.Tensor:
        if not self.use_d_fine_style:
            return self.retina_reg.weight.new_zeros(
                self.anchor_generator.num_levels)
        return (
            self.d_fine_refine_max
            * torch.tanh(self.d_fine_refine_raw))

    def _fine_grained_distribution_loss(
            self,
            cls_scores,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore=None):
        """DFL-supervise the residual distribution on positive anchors."""
        logits_per_level = self._last_d_fine_logits
        base_per_level = self._last_d_fine_base_preds
        self._last_d_fine_logits = None
        self._last_d_fine_base_preds = None
        if logits_per_level is None or base_per_level is None:
            raise RuntimeError(
                'D-FINE-style loss requires a preceding training forward')

        featmap_sizes = [
            featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = (
            self.cls_out_channels if self.use_sigmoid_cls else 1)
        targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if targets is None:
            return [
                logits.sum() * 0.0
                for logits in logits_per_level]
        (_, _, bbox_targets_list, bbox_weights_list,
         num_total_pos, _) = targets

        losses = []
        saturation_count = 0
        coordinate_count = 0
        target_abs_sum = 0.0
        scales = self.effective_d_fine_scales()
        for level_index, (logits, base_pred) in enumerate(zip(
                logits_per_level, base_per_level)):
            batch, _, height, width = logits.shape
            bins = self.d_fine_reg_max + 1
            shaped_logits = logits.permute(
                0, 2, 3, 1).reshape(
                    batch, height, width,
                    self.num_anchors, 4, bins).reshape(
                        -1, 4, bins)
            shaped_base = base_pred.permute(
                0, 2, 3, 1).reshape(-1, 4)
            shaped_targets = bbox_targets_list[
                level_index].reshape(-1, 4)
            shaped_weights = bbox_weights_list[
                level_index].reshape(-1, 4)
            level_anchors = torch.stack([
                per_image[level_index]
                for per_image in anchor_list], dim=0).reshape(-1, 4)
            positive = shaped_weights.sum(dim=1) > 0
            if not positive.any():
                losses.append(shaped_logits.sum() * 0.0)
                continue

            if self.reg_decoded_bbox:
                target_delta = self.bbox_coder.encode(
                    level_anchors[positive],
                    shaped_targets[positive])
            else:
                target_delta = shaped_targets[positive]
            scale = scales[level_index].detach().float().clamp_min(
                1e-6)
            normalized_target = (
                target_delta.float()
                - shaped_base[positive].detach().float()
            ) / scale
            positive_weights = shaped_weights[positive].float()
            saturation_count += int(
                (normalized_target.abs() >= 1.0).sum().item())
            coordinate_count += int(normalized_target.numel())
            target_abs_sum += float(
                normalized_target.abs().sum().item())
            raw_loss = _symmetric_distribution_focal_loss(
                shaped_logits[positive].float(),
                normalized_target,
                positive_weights,
                self.d_fine_reg_max,
                avg_factor=max(num_total_pos * 4, 1))
            losses.append(self.d_fine_loss_weight * raw_loss)

        self._dbg_d_fine_loss_raw = float(
            sum(
                loss.detach().float().item()
                for loss in losses)
            / max(self.d_fine_loss_weight, 1e-12))
        self._dbg_d_fine_saturation = (
            saturation_count / max(coordinate_count, 1))
        self._dbg_d_fine_target_abs = (
            target_abs_sum / max(coordinate_count, 1))
        return losses

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
             img_metas, gt_bboxes_ignore=None):
        losses = super().loss(
            cls_scores, bbox_preds, gt_bboxes, gt_labels,
            img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        if losses is None or not self.use_d_fine_style:
            return losses
        losses['loss_dfr'] = self._fine_grained_distribution_loss(
            cls_scores,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

class _InitCompatibility(_HeadCore):
    def __init__(self, *args, lung_anatomy_injection='off',
                 lung_anatomy_levels=(3,), lung_anatomy_residual_init=.01,
                 lung_anatomy_residual_max=.05, **kwargs):
        disabled = ('head_pc_modulation', 'bbox_contrastive', 'boxalign',
                    'edge_supervision', 'reg_detail_bypass', 'cross_scale', 'pc_branch')
        for name in disabled:
            if kwargs.get(name, False):
                raise ValueError(name + ' is not supported in the retained network')
        if lung_anatomy_injection != 'off':
            raise ValueError('TQ-SymNet does not consume external lung masks')
        if kwargs.get('positional_encoding', {}).get('use_upper_prior', False):
            raise ValueError('Only the original left/right Sym prior is retained')
        if kwargs.get('token_dropout', 0.) != 0.:
            raise ValueError('Inactive region slots require token_dropout=0')
        kwargs.setdefault('token_dropout', 0.)
        kwargs.setdefault('d_fine_style', True)
        kwargs.setdefault('d_fine_reg_max', 16)
        kwargs.setdefault('d_fine_loss_weight', .05)
        self.v321_lung_anatomy_mode = 'off'
        self.v321_lung_anatomy_levels = tuple(lung_anatomy_levels)
        self.v321_lung_anatomy_residual_init = float(lung_anatomy_residual_init)
        self.v321_lung_anatomy_residual_max = float(lung_anatomy_residual_max)
        if not 0. < self.v321_lung_anatomy_residual_init < self.v321_lung_anatomy_residual_max:
            raise ValueError('Invalid inactive checkpoint-slot initialization')
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        # Conv2d constructors consume the CPU RNG even when their weights are
        # immediately zeroed.  Preserve that RNG across both dynamic calls to
        # _init_layers so all V321 ablations share identical G2 parameters.
        with torch.random.fork_rng(devices=[]):
            self.v321_lung_anatomy_proj = nn.ModuleList([
                nn.Conv2d(3, self.feat_channels, 3, padding=1)
                for _ in range(self.anchor_generator.num_levels)
            ])
        raw = math.atanh(
            self.v321_lung_anatomy_residual_init
            / self.v321_lung_anatomy_residual_max)
        self.v321_lung_anatomy_scale_raw = nn.Parameter(torch.full(
            (self.anchor_generator.num_levels,), raw,
            dtype=torch.float32))
        self._v321_reset_anatomy_layers()

    def _v321_reset_anatomy_layers(self):
        for projection in self.v321_lung_anatomy_proj:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def init_weights(self):
        super().init_weights()
        self._v321_reset_anatomy_layers()
        self.v321_core_sha256 = self._v321_core_parameter_sha256()

    def _v321_core_parameter_sha256(self):
        """Hash non-V321 parameters to prove initialization equivalence."""
        digest = hashlib.sha256()
        excluded = (
            'v321_lung_anatomy_proj',
            'v321_lung_anatomy_scale_raw')
        with torch.no_grad():
            for name, parameter in sorted(self.named_parameters()):
                if name.startswith(excluded):
                    continue
                value = parameter.detach().cpu().contiguous()
                digest.update(name.encode('utf-8'))
                digest.update(str(tuple(value.shape)).encode('ascii'))
                digest.update(str(value.dtype).encode('ascii'))
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

@HEADS.register_module()
class TQSymHead(_InitCompatibility):
    def __init__(self, *args, sym_regression_source='crossscale',
                 sym_reg_residual_max=0.10,
                 sym_reg_residual_trainable=True,
                 sym_reg_fusion='direct_delta',
                 nms_oracle_debug=False, nms_debug_interval=250,
                 **kwargs):
        self.sym_regression_source = str(sym_regression_source).lower()
        self.sym_reg_residual_max = float(sym_reg_residual_max)
        self.sym_reg_residual_trainable = bool(sym_reg_residual_trainable)
        self.sym_reg_fusion = str(sym_reg_fusion).lower()
        self.nms_oracle_debug = bool(nms_oracle_debug)
        self.nms_debug_interval = int(nms_debug_interval)
        if self.sym_regression_source not in ('sym', 'crossscale'):
            raise ValueError(
                "sym_regression_source must be 'sym' or 'crossscale'")
        if self.sym_reg_fusion not in (
                'direct_delta', 'adapted_delta', 'adapted_sym'):
            raise ValueError(
                'sym_reg_fusion must be direct_delta, adapted_delta, or '
                'adapted_sym')
        if not 0.0 < self.sym_reg_residual_max <= 1.0:
            raise ValueError('sym_reg_residual_max must be in (0, 1]')
        if self.nms_debug_interval < 0:
            raise ValueError('nms_debug_interval must be non-negative')
        super().__init__(*args, **kwargs)
        # Zero is an identity initialisation for the regression route.  It is
        # intentionally created after all historical layers, so this module
        # consumes no RNG and cannot shift the legacy model initialisation.
        self.sym_reg_residual_raw = nn.Parameter(
            torch.zeros(self.anchor_generator.num_levels, dtype=torch.float32),
            requires_grad=(self.sym_regression_source == 'crossscale'
                           and self.sym_reg_residual_trainable))
        # Adapters are created only for the two task-adapted routes and only
        # after all historical layers.  Conv construction is RNG-isolated;
        # the explicit identity reset makes A_l initially transparent without
        # shifting any legacy core-layer initialization.
        self.sym_reg_adapters = nn.ModuleList()
        if (self.sym_regression_source == 'crossscale'
                and self.sym_reg_fusion in ('adapted_delta', 'adapted_sym')):
            with torch.random.fork_rng(devices=[]):
                self.sym_reg_adapters = nn.ModuleList([
                    nn.Conv2d(self.feat_channels, self.feat_channels, 1)
                    for _ in range(self.anchor_generator.num_levels)
                ])
            self._reset_sym_reg_adapters()
        self._v44_decouple_stats = [None] * self.anchor_generator.num_levels
        self._v44_nms_stats = {}

    def _reset_sym_reg_adapters(self):
        """Set every per-level 1x1 adapter to an exact identity map."""
        for adapter in self.sym_reg_adapters:
            nn.init.zeros_(adapter.weight)
            diagonal = min(adapter.out_channels, adapter.in_channels)
            indices = torch.arange(diagonal, device=adapter.weight.device)
            adapter.weight.data[indices, indices, 0, 0] = 1.0
            if adapter.bias is not None:
                nn.init.zeros_(adapter.bias)

    def init_weights(self):
        super().init_weights()
        # Parent init_cfg may visit every Conv2d.  Reassert identity after it.
        self._reset_sym_reg_adapters()

    def effective_sym_reg_scales(self):
        return self.sym_reg_residual_max * torch.tanh(
            self.sym_reg_residual_raw)

    def forward_single(self, x, lbp_lab=None, lung_masks=None,
                       level_index=0, reg_detail_feat=None):
        bs, _, h, w = x.shape
        masks = x.new_zeros((bs, h, w)).to(torch.bool)
        pos_embed = self.positional_encoding(
            masks, x, lung_mask=lung_masks)
        sym_feat = self.transformer(
            tuple([x]), [masks], self.query_embedding.weight, [pos_embed])

        # Classification intentionally uses the semantic/symmetry route.
        cls_feat = sym_feat
        sym_delta = sym_feat - x
        if self.sym_regression_source == 'sym':
            # Coupled control: exact historical no-PC V321/V331 feature path.
            sym_scale = x.new_tensor(1.0)
            reg_feat = sym_feat
            adapter_input = sym_feat
            adapter_output = sym_feat
        else:
            # Decoupled path. This is not a detach: bbox gradients still
            # improve CrossScale/backbone; Sym receives regression gradients
            # only through this bounded residual when it is trainable.
            sym_scale = self.effective_sym_reg_scales()[level_index].to(
                dtype=x.dtype)
            if self.sym_reg_fusion == 'adapted_sym':
                adapter_input = sym_feat
            else:
                adapter_input = sym_delta
            if self.sym_reg_fusion.startswith('adapted_'):
                adapter_output = self.sym_reg_adapters[level_index](
                    adapter_input)
            else:
                adapter_output = adapter_input
            reg_feat = x + sym_scale * adapter_output

        gate_stats = {
            'gate_mean': 0.0, 'gate_active': 0.0, 'edge_mean': 0.0,
            'spatial_gate_mean': 0.0, 'edge_temp': 0.0,
            'mod_scale': 0.0, 'boundary_weight_entropy': 0.0,
            'boundary_weight_peak': 0.0, 'pc_feat_norm': 0.0,
            'pc_contribution_ratio': 0.0,
        }
        # Head PC was invalidated by the P/U ablations.  Do not silently route
        # it into this new geometry study.
        if self.use_head_pc_modulation and lbp_lab is not None:
            raise RuntimeError(
                'V44 geometry ablation requires head_pc_modulation=False')
        if self.use_reg_detail_bypass:
            reg_feat = self.reg_detail_injector(
                reg_feat, reg_detail_feat, level_index=level_index)

        with torch.no_grad():
            native = x.detach().float().flatten(1)
            delta = sym_delta.detach().float().flatten(1)
            adapter_input_flat = adapter_input.detach().float().flatten(1)
            adapter_output_flat = adapter_output.detach().float().flatten(1)
            reg_flat = reg_feat.detach().float().flatten(1)
            # Measure the actual change relative to CrossScale, not merely
            # alpha*A(input).  They are equal for the two soft routes, while
            # coupled control must correctly report |S-C|/|C| rather than
            # the misleading feature magnitude |S|/|C|.
            contribution = reg_flat - native
            self._v44_decouple_stats[level_index] = {
                'scale': float(sym_scale.detach().float().item()),
                'sym_delta_rel': float((
                    delta.norm(dim=1) / native.norm(dim=1).clamp_min(1e-8)
                ).mean().item()),
                'sym_cos': float(F.cosine_similarity(
                    native, sym_feat.detach().float().flatten(1),
                    dim=1, eps=1e-8).mean().item()),
                'adapter_in_rel': float((
                    adapter_input_flat.norm(dim=1)
                    / native.norm(dim=1).clamp_min(1e-8)).mean().item()),
                'adapter_out_rel': float((
                    adapter_output_flat.norm(dim=1)
                    / native.norm(dim=1).clamp_min(1e-8)).mean().item()),
                'actual_contribution': float((
                    contribution.norm(dim=1)
                    / native.norm(dim=1).clamp_min(1e-8)).mean().item()),
                'reg_vs_c_cos': float(F.cosine_similarity(
                    reg_flat, native, dim=1, eps=1e-8).mean().item()),
            }

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_base = self.retina_reg(reg_feat)
        distribution_logits = None
        if self.use_d_fine_style:
            distribution_logits = self.d_fine_reg(reg_feat)
            normalized_residual, entropy = _distribution_refinement_expectation(
                distribution_logits, self.num_anchors, self.d_fine_reg_max)
            refine_scale = self.effective_d_fine_scales()[level_index].to(
                dtype=bbox_base.dtype)
            bbox_pred = bbox_base + refine_scale * normalized_residual
            self._dbg_d_fine_entropy[level_index] = float(
                entropy.detach().float().item())
            self._dbg_d_fine_expected_abs[level_index] = float(
                normalized_residual.detach().float().abs().mean().item())
        else:
            bbox_pred = bbox_base
        return (
            cls_score, bbox_pred, sym_feat, sym_feat, reg_feat, gate_stats,
            bbox_base, distribution_logits)

    def _debug_log(self, *args, **kwargs):
        pass

    def forward(self, feats, nam_graphs=None, gt_bboxes=None, gt_labels=None,
                lbp_lab=None, c2_feat=None, region_feat=None, img_shapes=None,
                lung_masks=None, reg_detail_feats=None):
        if nam_graphs is not None or lbp_lab is not None or lung_masks is not None:
            raise ValueError('TQSymHead uses visual features without external masks/NAM maps')
        cls_scores, bbox_preds, dfl, base, reg_features = [], [], [], [], []
        for level, feature in enumerate(feats):
            output = self.forward_single(feature, level_index=level)
            cls_scores.append(output[0]); bbox_preds.append(output[1])
            reg_features.append(output[4]); base.append(output[6]); dfl.append(output[7])
        self._last_reg_features = None  # No inactive contrast/BoxAlign graph retention.
        self._last_d_fine_logits = tuple(dfl) if self.training and self.use_d_fine_style else None
        self._last_d_fine_base_preds = tuple(base) if self.training and self.use_d_fine_style else None
        if self.training:
            self._debug_step += 1
        zero = feats[0].new_zeros(())
        return tuple(cls_scores), tuple(bbox_preds), zero, zero
