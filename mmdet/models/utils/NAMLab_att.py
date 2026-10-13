"""
NAMLab_att: NAM 色块图引导的双分支注意力模块。

对接 NAMLab.py 输出的色块分层图：
  - SymFormer 分支: FPN feat → Deformable Transformer (x→1-x 对称) → feat1
  - NAM 分支:     色块图 → NAMEncoder → nam_feat → residual → feat2
  - 融合:         fused = feat1 + alpha * feat2

NAM 色块图 = 解剖平滑图（抹除纹理）。residual = 原始feat - 平滑feat = 纹理异常信号。
TB 病灶正是纹理异常区域，NAM 分支告诉 attention "哪里值得看"。

组件:
    LoadNAMMap            - 数据管线: 加载预计算的 NAM 色块图
    NAMEncoder            - 将色块图编码为与 FPN 对齐的特征
    NAMBranch             - NAM 分支: 编码→残差→融合
    RetinaNAMGuideAttHead - 双分支 Head (继承 RetinaGuideAttHead)
    RetinaNAMAtt          - 单阶段检测器 (继承 SingleStageDetector)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.datasets.builder import PIPELINES

# ---------- 预定义解剖区域调色板 (25 类足以覆盖胸部解剖结构) ----------
# 每个区域一种颜色 (RGB, 0-255)，用于 label_map → RGB 色块图转换
ANATOMY_PALETTE = np.array([
    [0,   0,   0  ],   # 0: 背景
    [128, 0,   0  ],   # 1: 左肺
    [0,   128, 0  ],   # 2: 右肺
    [128, 128, 0  ],   # 3: 心脏
    [0,   0,   128],   # 4: 气管
    [128, 0,   128],   # 5: 锁骨
    [0,   128, 128],   # 6: 膈肌
    [64,  0,   0  ],   # 7: 纵隔
    [0,   64,  0  ],   # 8: 主动脉
    [64,  64,  0  ],   # 9: 脊柱
    [0,   0,   64 ],   # 10-24: 扩展区域
    [64,  0,   64 ], [0,   64,  64 ], [192, 0,   0  ], [0,   192, 0  ],
    [192, 192, 0  ], [0,   0,   192], [192, 0,   192], [0,   192, 192],
    [64,  128, 0  ], [128, 64,  0  ], [64,  0,   128], [128, 0,   64 ],
    [0,   64,  128], [0,   128, 64 ],
], dtype=np.uint8)


# ============================================================================
# LoadNAMMap: 训练时从磁盘加载预计算的 NAM 色块图
# ============================================================================
@PIPELINES.register_module()
class LoadNAMMap:
    """
    数据管线 transform：加载预计算的 NAM 色块图。

    支持两种输入模式:
      - mode='rgb':   直接加载 RGB 色块图 (由 NAMLab.py 生成的三通道图像)
      - mode='label': 加载 label map → 用调色板转为 RGB 色块图

    用法 — 在 config 的 train_pipeline / test_pipeline 中加入:

        dict(type='LoadNAMMap',
             nam_dir='data/TBX11K/nam_maps',    # NAMLab.py 输出目录
             mode='rgb',                         # 'rgb' 或 'label'
             suffix='_nam_rgb.png',              # 文件名后缀
             key='nam_map')

    RGB 色块图命名约定 (mode='rgb'):
        {原图basename}_nam_rgb.png
    Label 色块图命名约定 (mode='label'):
        {原图basename}_nam_label.npy

    加载后，detector.forward_train 提取 nam_map 传入 NAMBranch。
    """

    def __init__(self, nam_dir: str,
                 mode: str = 'rgb',
                 suffix: str = '_nam_rgb.png',
                 key: str = 'nam_map',
                 palette: np.ndarray = None):
        self.nam_dir = nam_dir
        self.mode = mode
        self.suffix = suffix
        self.key = key
        self.palette = palette if palette is not None else ANATOMY_PALETTE

    def __call__(self, results):
        """加载 NAM 色块图并存入 results[key] 为 (3, H, W) float32 RGB tensor."""
        # 从 img 路径推导 NAM 色块图路径
        img_path = results.get('filename', results.get('img_prefix', ''))
        if not img_path:
            img_meta = results.get('img_metas', {})
            if isinstance(img_meta, dict):
                img_path = img_meta.get('filename', '')
            elif isinstance(img_meta, list) and len(img_meta) > 0:
                img_path = img_meta[0].get('filename', '')

        basename = os.path.splitext(os.path.basename(img_path))[0]
        nam_file = os.path.join(self.nam_dir, f'{basename}{self.suffix}')

        if self.mode == 'rgb':
            nam_map = self._load_rgb(nam_file, results)
        elif self.mode == 'label':
            nam_map = self._load_label(nam_file, results)
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'. Use 'rgb' or 'label'.")

        # 存入 (H, W, 3) → 转为 (3, H, W) float32 [0, 255]
        if nam_map.ndim == 2:
            nam_map = self._label_to_rgb(nam_map)
        if nam_map.shape[-1] == 3:  # HWC → CHW
            nam_map = nam_map.transpose(2, 0, 1)
        results[self.key] = nam_map.astype(np.float32)
        return results

    def _load_rgb(self, nam_file, results):
        """加载 RGB 色块图。"""
        if os.path.exists(nam_file):
            from mmcv.image.io import imread
            nam_map = imread(nam_file)  # (H, W, 3) uint8 BGR
            nam_map = nam_map[..., ::-1]  # BGR → RGB
        else:
            h, w = results.get('img_shape', (512, 512))[:2]
            nam_map = np.zeros((h, w, 3), dtype=np.uint8)
        return nam_map

    def _load_label(self, nam_file, results):
        """加载 label map (.npy) 并转为 RGB。"""
        if os.path.exists(nam_file):
            label_map = np.load(nam_file)  # (H, W) int32
        else:
            h, w = results.get('img_shape', (512, 512))[:2]
            label_map = np.zeros((h, w), dtype=np.int32)
        return self._label_to_rgb(label_map)

    def _label_to_rgb(self, label_map):
        """label map (H, W) int → RGB image (H, W, 3) uint8."""
        label_map = np.clip(label_map.astype(np.int32), 0, len(self.palette) - 1)
        rgb_map = self.palette[label_map]  # (H, W, 3) uint8
        return rgb_map

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'nam_dir={self.nam_dir}, mode={self.mode}, '
                f'suffix={self.suffix})')


# ============================================================================
# InvertedResidualBlock: MobileNetV2 风格的倒残差块
# ============================================================================
class InvertedResidualBlock(nn.Module):
    """
    倒残差块: 1×1 expand → 3×3 DWConv → 1×1 compress + skip.

    轻量高效，用深度可分离卷积捕获局部细节模式。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 expand_ratio: int = 4, stride: int = 1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_skip = (stride == 1) and (in_channels == out_channels)

        layers = []
        # Expand
        if expand_ratio > 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        # Depthwise
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ]
        # Compress
        layers += [
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = out + x
        return out


# ============================================================================
# MultiScaleConv: 多尺度空洞卷积并行
# ============================================================================
class MultiScaleConv(nn.Module):
    """
    同一输入并行跑多个 dilation 的 depthwise conv，concat 融合。

    小的 dilation 捕获细粒度纹理，大的 dilation 捕获局部上下文。
    """

    def __init__(self, channels: int, dilations: tuple = (1, 2, 4)):
        super().__init__()
        n_branches = len(dilations)
        self.branches = nn.ModuleList()
        for d in dilations:
            # DWConv 输出全通道(每分支 channels 个)，再靠 merge 1×1 压缩
            self.branches.append(
                nn.Conv2d(channels, channels, 3,
                          padding=d, dilation=d, groups=channels, bias=False))
        self.merge = nn.Sequential(
            nn.Conv2d(channels * n_branches, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)  # (N, C*n_branches, H, W)
        return self.merge(out)        # → (N, C, H, W)



# ============================================================================
# NAMEncoder: 色块图 → 特征空间 (增强局部细节版)
# ============================================================================
class NAMEncoder(nn.Module):
    """
    将 NAM 色块图编码为与 FPN 对齐的特征空间。

    架构:
      stem: 1×1 → 64ch
      stage1: InvertedResidual 64→128 (expand=4)
      stage2: InvertedResidual 128→256 (expand=4)
      stage3: MultiScaleConv 256 → 多 dilation 并行
      project: 1×1 256→out_channels

    相比旧版纯 3 层卷积:
      - 深度增加了 3 个倒残差块，每块有 skip connection
      - stage3 用多尺度空洞卷积(1,2,4) 同时捕获细粒度和上下文
      - 感受野从 3×3 扩展到 ~17×17

    Args:
        in_channels (int): 输入通道数 (3=RGB)，默认 3。
        out_channels (int): 输出通道数（需 = FPN feat_channels），默认 256。
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels

        # Stem: 轻量投影
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 倒残差块堆叠
        self.stage1 = InvertedResidualBlock(64, 128, expand_ratio=4)
        self.stage2 = InvertedResidualBlock(128, 256, expand_ratio=4)

        # 多尺度空洞卷积
        self.ms_conv = MultiScaleConv(256, dilations=(1, 2, 4))

        # 输出投影
        self.project = nn.Conv2d(256, out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == 1 or m.groups == m.in_channels:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # 零初始化输出投影: α=0 时整个 NAM 分支输出为 0
        nn.init.zeros_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    def forward(self, nam_map: torch.Tensor) -> torch.Tensor:
        """nam_map: (N, 3, H, W) [0,255] → nam_feat: (N, C, H, W)"""
        x = self.stem(nam_map / 255.0)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.ms_conv(x)
        return self.project(x)


# ============================================================================
# RegionAdaptiveResidual: NAM区域感知的残差调制
# ============================================================================
class RegionAdaptiveResidual(nn.Module):
    """
    让 NAM 的区域身份信息指导残差的局部放大/抑制。

    核心思路:
      NAM 色块图的每个纯色区域 = 一种解剖结构。
      不同解剖区域的"正常纹理水平"不同:
        - 肺野: 本该平滑 → 微小残差也可能是病灶 → 放大
        - 心脏边缘: 天然有纹理 → 残差可能只是正常边界 → 抑制
        - 骨骼: 高密度自然变化 → 中等

    实现:
      region_modulation = 1 + tanh(region_conv(nam_feat))
        → 每个像素、每个通道的调制系数在 (0, 2) 之间
      contrast = residual_raw - avgpool(residual_raw, k=3)
        → 局部突出程度
      residual = region_modulation * (residual_raw + contrast)
        → 区域敏感 + 局部峰值增强
    """

    def __init__(self, channels: int):
        super().__init__()
        # 从 nam_feat 预测区域调制系数
        self.region_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Tanh(),
        )
        # 零初始化 → 训练初期 modulation = 1 (不做调制)
        nn.init.zeros_(self.region_conv[-1].weight)
        if self.region_conv[-1].bias is not None:
            nn.init.zeros_(self.region_conv[-1].bias)

    def forward(self, residual_raw: torch.Tensor,
                nam_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual_raw: (N, C, H, W) feat_raw - nam_feat (纹理异常)
            nam_feat:     (N, C, H, W) NAM 编码的解剖平滑特征 (区域身份)

        Returns:
            residual: (N, C, H, W) 区域自适应增强后的残差
        """
        # 区域调制: 每个解剖区域对残差的敏感度不同
        region_modulation = 1.0 + self.region_conv(nam_feat)  # (0, 2)

        # 局部对比度: 残差在 3×3 邻域里的突出程度
        local_avg = F.avg_pool2d(residual_raw, 3, stride=1, padding=1)
        contrast = residual_raw - local_avg

        # 区域自适应增强
        return region_modulation * (residual_raw + contrast)


# ============================================================================
# NAM Branch: 色块图 → 纹理异常特征 (增强局部感知版)
# ============================================================================
class NAMBranch(nn.Module):
    """
    NAM 分支 v2 — 加深编码 + 区域自适应残差。

    完整流程:
      nam_map (N,3,H,W)                feat_raw (N,256,H,W)
           │                                │
           ├─ NAMEncoder ─→ nam_feat ───┐   │
           │    stem: 1×1 3→64          │   │
           │    IRB₁: 64→128            │   ├─ residual_raw = feat_raw - nam_feat
           │    IRB₂: 128→256           │   │
           │    MConv(d=1,2,4)          │   └── RegionAdaptiveResidual
           │    proj: 1×1→256           │        ├ region_mod(nam_feat): 哪区的残差该放大
           │                            │        ├ contrast(residual_raw): 局部峰值
           │                            │        └→ residual = mod * (raw + contrast)
           │                            │
           └── concat(nam_feat, residual) ─→ fusion_conv → feat2

    关键改进 vs v1:
      - NAMEncoder: 3层CNN → InvertedResidual + MultiScaleConv (感受野 3→17)
      - 残差不再是全局减法 → RegionAdaptiveResidual: 不同解剖区域不同敏感度
      - 色块图提供"区域身份"，不提供纹理——纹理信息全在残差里

    Args:
        feat_channels (int): FPN 特征通道数, 默认 256
        nam_in_channels (int): NAM 输入通道数 (3=RGB), 默认 3
    """

    def __init__(self, feat_channels: int = 256, nam_in_channels: int = 3):
        super().__init__()

        # 编码器: 色块→解剖平滑特征 (加深版)
        self.nam_encoder = NAMEncoder(in_channels=nam_in_channels,
                                       out_channels=feat_channels)

        # 区域自适应残差: NAM 区域身份 → 指导残差调制
        self.region_residual = RegionAdaptiveResidual(channels=feat_channels)

        # 融合: nam_feat + enhanced_residual → feat2
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1),
        )

        # 可学习融合系数
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # 零初始化最后一层: alpha=0 时整个 NAM 分支贡献为 0
        nn.init.zeros_(self.fusion_conv[-1].weight)
        if self.fusion_conv[-1].bias is not None:
            nn.init.zeros_(self.fusion_conv[-1].bias)

    def forward(self, feat_raw: torch.Tensor, nam_map: torch.Tensor):
        """
        Args:
            feat_raw: (N, C, H, W) 原始 FPN 特征
            nam_map:  (N, 3, H_nam, W_nam) NAM 色块图, [0,255]

        Returns:
            feat2: (N, C, H, W) NAM 分支输出
        """
        # 色块图 resize 到与 feat 对齐
        if nam_map.shape[-2:] != feat_raw.shape[-2:]:
            nam_map = F.interpolate(nam_map, size=feat_raw.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # 1. 色块 → 解剖平滑特征
        nam_feat = self.nam_encoder(nam_map)

        # 2. 残差 + 区域自适应调制 (nam_feat提供区域身份上下文)
        residual_raw = feat_raw - nam_feat
        residual = self.region_residual(residual_raw, nam_feat)

        # 3. 融合 nam_feat + enhanced_residual
        feat2 = self.fusion_conv(
            torch.cat([nam_feat, residual], dim=1))
        return feat2


# ============================================================================
# RetinaNAMGuideAttHead: 双分支 Head (继承 RetinaGuideAttHead)
# ============================================================================
import sys as _sys
# 绕过 dense_heads/__init__.py 的循环导入: 从 sys.modules 直接取已加载的模块
_retina_guide = _sys.modules.get('mmdet.models.dense_heads.retina_guide_att_head')
if _retina_guide is not None:
    RetinaGuideAttHead = _retina_guide.RetinaGuideAttHead
else:
    from ..dense_heads.retina_guide_att_head import RetinaGuideAttHead  # noqa: F811

from ..builder import HEADS


@HEADS.register_module()
class RetinaNAMGuideAttHead(RetinaGuideAttHead):
    """
    双分支 RetinaNet Head: SymFormer + NAM 分支并行。

    继承自 RetinaGuideAttHead，仅覆盖:
      - __init__:        增加 NAM 参数
      - _init_layers:    增加 NAM 分支构建
      - forward_single:  双分支前向 + 融合
      - forward:         支持 nam_maps 参数

    === 完整 Config 示例 ===

    # ── 数据管线: 加载 NAM 色块图 ──
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='LoadNAMMap',                          # ← 新增
             nam_dir='data/TBX11K/nam_maps',
             mode='rgb',
             suffix='_nam_rgb.png',
             key='nam_map'),
        dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
                                   'nam_map']),         # ← nam_map 加入 keys
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadNAMMap',                          # ← 测试也要加载
             nam_dir='data/TBX11K/nam_maps',
             mode='rgb',
             suffix='_nam_rgb.png',
             key='nam_map'),
        dict(type='MultiScaleFlipAug',
             img_scale=(512, 512), flip=False,
             transforms=[
                 dict(type='Resize', keep_ratio=True),
                 dict(type='RandomFlip'),
                 dict(type='Normalize', **img_norm_cfg),
                 dict(type='Pad', size_divisor=32),
                 dict(type='ImageToTensor', keys=['img', 'nam_map']),
                 dict(type='Collect', keys=['img', 'nam_map']),
             ]),
    ]

    # ── 模型: 双分支 Head ──
    model = dict(
        type='RetinaNAMAtt',
        backbone=dict(...),
        neck=dict(...),
        bbox_head=dict(
            type='RetinaNAMGuideAttHead',               # ← 新 Head
            nam_in_channels=3,                           # NAM 输入通道(RGB=3)
            num_classes=2,
            num_query=500,
            dims_radio=1,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(...),
            bbox_coder=dict(...),
            transformer=dict(...),
            positional_encoding=dict(...),
            loss_cls=dict(...),
            loss_bbox=dict(...)),
        train_cfg=dict(...),
        test_cfg=dict(...))

    Args:
        nam_in_channels (int): NAM 色块图输入通道数(3=RGB)。默认 3。
        (其余参数同 RetinaGuideAttHead)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 nam_in_channels: int = 3,
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
        # 先存 NAM 特有参数，再调父类 __init__（父类会调用 _init_layers）
        self.nam_in_channels = nam_in_channels
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
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
        """初始化卷积头 + NAM 分支。"""
        # 调用父类: 构建 cls_convs, reg_convs, retina_cls, retina_reg, query_embedding
        super()._init_layers()

        # 构建 NAM 分支
        self.nam_branch = NAMBranch(
            feat_channels=self.feat_channels,
            nam_in_channels=self.nam_in_channels)

    # ========== Forward ==========
    def forward_single(self, x, nam_map=None):
        """
        单个 FPN 层级的双分支前向。

        Args:
            x (Tensor): FPN 特征 (bs, C, H, W)
            nam_map (Tensor or None): NAM 色块图 (bs, 3, H_nam, W_nam)

        Returns:
            tuple: cls_score (bs, A*cls, H, W), bbox_pred (bs, A*4, H, W)
        """
        bs, _, h, w = x.shape
        masks = x.new_zeros((bs, h, w)).to(torch.bool)

        # ---- Branch 1: SymFormer ----
        pos_embed = self.positional_encoding(masks, x)
        feat1 = self.transformer(
            tuple([x]), [masks], self.query_embedding.weight, [pos_embed])

        # ---- Branch 2: NAM (色块图 → 纹理异常) ----
        if nam_map is not None:
            feat2 = self.nam_branch(x, nam_map)
            fused = feat1 + self.nam_branch.alpha * feat2
        else:
            fused = feat1  # 无 NAM 色块图时退化为原版 SymFormer

        # ---- 共享分类/回归头 ----
        cls_feat, reg_feat = fused, fused
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def forward(self, feats, nam_maps=None):
        """
        多层级前向。

        Args:
            feats (tuple[Tensor]): FPN 多层级特征
            nam_maps (list[Tensor] or None): 每层 NAM 色块图，长度与 feats 一致

        Returns:
            tuple: cls_scores (list), bbox_preds (list)
        """
        from mmdet.core import multi_apply
        if nam_maps is None:
            nam_maps = [None] * len(feats)
        return multi_apply(self.forward_single, feats, nam_maps)

    def simple_test(self, feats, img_metas, rescale=False, nam_maps=None):
        """覆盖父类 simple_test, 将 nam_maps 传入 forward。"""
        outs = self.forward(feats, nam_maps=nam_maps)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    # loss / get_targets / get_anchors 全部继承自 AnchorHead，
    # 无需重写。输入输出接口完全兼容。

# ============================================================================
# RetinaNAMAtt: 带 NAM 色块图传递的 Detector (继承 SingleStageDetector)
# ============================================================================
# 注: 此处在 dense_heads 加载时执行。SingleStageDetector 在 detectors 包中，
# detectors 尚未加载 → 用 sys.modules 检查，未命中则触发 detectors 加载（无循环依赖）。
_single_stage = _sys.modules.get('mmdet.models.detectors.single_stage')
if _single_stage is not None:
    SingleStageDetector = _single_stage.SingleStageDetector
else:
    from ..detectors.single_stage import SingleStageDetector  # noqa: F811

from ..builder import DETECTORS


@DETECTORS.register_module()
class RetinaNAMAtt(SingleStageDetector):
    """
    单阶段检测器，将 NAM 色块图从 data dict 传递到双分支 Head。

    继承 SingleStageDetector，仅覆盖 forward_train 和 simple_test
    以提取并传递 nam_map。

    Config 中使用:
        model = dict(
            type='RetinaNAMAtt',
            backbone=dict(...),
            neck=dict(...),
            bbox_head=dict(type='RetinaNAMGuideAttHead', ...),
            train_cfg=dict(...),
            test_cfg=dict(...),
        )
    """

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      nam_map=None,
                      **kwargs):
        """
        Args:
            img (Tensor): 输入图像 (N, C, H, W)
            nam_map (Tensor or None): NAM 色块图 (N, 3, H, W)，由 LoadNAMMap 加载。

        Returns:
            dict[str, Tensor]: 损失字典。
        """
        x = self.extract_feat(img)

        # 构建每层 NAM maps — 逐层 resize 到对应 FPN 空间尺寸
        nam_maps = None
        if nam_map is not None:
            nam_maps = []
            for xi in x:
                nh, nw = xi.shape[-2:]
                nmi = F.interpolate(nam_map, size=(nh, nw),
                                    mode='bilinear', align_corners=False)
                nam_maps.append(nmi)

        # Head 前向
        outs = self.bbox_head(feats=x, nam_maps=nam_maps)

        # 损失计算（继承自 AnchorHead）
        losses = self.bbox_head.loss(
            *outs, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, nam_map=None, rescale=False, **kwargs):
        """
        Args:
            img (Tensor): 输入图像 (N, C, H, W)
            nam_map (Tensor or None): NAM 色块图 (N, 3, H, W)

        Returns:
            list[list[np.ndarray]]: 每张图每类的 bbox 结果。
        """
        x = self.extract_feat(img)

        nam_maps = None
        if nam_map is not None:
            nam_maps = []
            for xi in x:
                nh, nw = xi.shape[-2:]
                nmi = F.interpolate(nam_map, size=(nh, nw),
                                    mode='bilinear', align_corners=False)
                nam_maps.append(nmi)

        from mmdet.core import bbox2result
        results_list = self.bbox_head.simple_test(
            x, img_metas, rescale=rescale, nam_maps=nam_maps)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
