"""
NAMLab_att v2: 图注意力 NAM 分支（Graph Attention）。

v1 → v2 核心变化:
  - v1: 色块图 RGB → CNN Encoder → 全局像素残差
  - v2: 区域 WLD + 邻接图 → EdgeGAT → 区域异常分数 → 像素残差

设计思路:
  TB 结节 = 纹理异常 + 邻域高对比。NAM 区域天然携带:
    - 自身纹理 (WLD): "这个区域长什么样"
    - 邻域对比 (DC/DT/DBC): "和邻居差多少"
    - 邻接关系: "谁是解剖上下文"

  图注意力让区域在相邻区域间交换信息，显式利用邻接结构。

组件:
    LoadNAMGraph           - 数据管线: 加载 .npz 图数据
    EdgeGATLayer           - 边条件图注意力层
    NAMGraphBranch         - 图注意力 NAM 分支
    RetinaNAMGuideAttHeadV2 - 双分支 Head
    RetinaNAMAttV2         - 检测器

数据格式 (.npz):
    label_map  : (H, W) int32       像素→区域ID
    node_wld   : (N, 3, 59) float32  区域LBP纹理 (uniform P=8)
    node_stat  : (N, 6) float32      区域一阶灰度统计
    node_glcm  : (N, 4) float32      区域GLCM Haralick特征
    node_area  : (N,) float32        区域面积
    node_lab   : (N, 3) float32     区域Lab均值
    edge_index : (2, E) int32       邻接边
    edge_dc    : (E,) float32        颜色距离
    edge_dt    : (E,) float32        纹理距离
    edge_dbc   : (E,) float32        边界对比度

Config 示例:
    bbox_head=dict(
        type='RetinaNAMGuideAttHeadV2',
        num_classes=2, num_query=500, ...
        gat_hidden=256, gat_heads=4, gat_layers=2,
        loss_cls=..., loss_bbox=...)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.datasets.builder import PIPELINES


# ============================================================================
# CLAHE: OpenCV CLAHE pipeline transform (替代 albumentations)
# ============================================================================
@PIPELINES.register_module()
class CLAHETransform:
    """OpenCV CLAHE 对比度增强。

    用法:
        dict(type='CLAHETransform', clip_limit=3.0, tile_grid_size=(8, 8))
    """

    def __init__(self, clip_limit: float = 3.0,
                 tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, results):
        import cv2
        img = results['img']  # (H, W, 3) uint8 BGR
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        results['img'] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'clip_limit={self.clip_limit}, '
                f'tile_grid_size={self.tile_grid_size})')


# ============================================================================
# LoadNAMGraph: 从 .npz 加载 NAM 图数据
# ============================================================================
@PIPELINES.register_module()
class LoadNAMGraph:
    """加载 NAMLab 预计算的图数据 (.npz)。

    用法 — 在 config pipeline 中加入:
        dict(type='LoadNAMGraph',
             nam_dir='data/TBX11K/nam_graphs/',
             suffix='_nam_graph.npz',
             key='nam_graph')

    存入 results[key] = {
        'label_map':  (H, W) torch.LongTensor,
        'node_wld':   (N, 3, 48) torch.FloatTensor,
        'node_area':  (N,) torch.FloatTensor,
        'node_lab':   (N, 3) torch.FloatTensor,
        'edge_index': (2, E) torch.LongTensor,
        'edge_dc':    (E,) torch.FloatTensor,
        'edge_dt':    (E,) torch.FloatTensor,
        'edge_dbc':   (E,) torch.FloatTensor,
    }
    """

    def __init__(self, nam_dir: str,
                 suffix: str = '_nam_graph.npz',
                 key: str = 'nam_graph'):
        self.nam_dir = nam_dir
        self.suffix = suffix
        self.key = key
        self._subdir_cache = None  # 延迟扫描

    def _scan_subdirs(self):
        """扫描 nam_dir 下的子目录，缓存结果。"""
        if self._subdir_cache is not None:
            return self._subdir_cache
        dirs = []
        if os.path.isdir(self.nam_dir):
            for sub in sorted(os.listdir(self.nam_dir)):
                sub_path = os.path.join(self.nam_dir, sub)
                if os.path.isdir(sub_path):
                    npz_sub = os.path.join(sub_path, 'npz')
                    dirs.append((sub, sub_path, npz_sub if os.path.isdir(npz_sub) else None))
        self._subdir_cache = dirs
        return dirs

    def __call__(self, results):
        img_path = results.get('filename', results.get('img_prefix', ''))
        if not img_path:
            img_meta = results.get('img_metas', {})
            if isinstance(img_meta, dict):
                img_path = img_meta.get('filename', '')
            elif isinstance(img_meta, list) and len(img_meta) > 0:
                img_path = img_meta[0].get('filename', '')

        basename = os.path.splitext(os.path.basename(img_path))[0]

        # 路径1: 从图片父目录推断 (img在 health/ → npz在 nam_dir/health/npz/)
        img_parent = os.path.basename(os.path.dirname(img_path))
        nam_file = os.path.join(self.nam_dir, img_parent, 'npz',
                                f'{basename}{self.suffix}')

        # 路径2: nam_dir/npz/ (扁平布局)
        if not os.path.exists(nam_file):
            nam_file = os.path.join(self.nam_dir, 'npz',
                                    f'{basename}{self.suffix}')

        # 路径3: nam_dir/ 根目录
        if not os.path.exists(nam_file):
            nam_file = os.path.join(self.nam_dir, f'{basename}{self.suffix}')

        # 路径4: 扫描所有子目录 (首次使用缓存)
        if not os.path.exists(nam_file):
            for _, sub_path, npz_sub in self._scan_subdirs():
                if npz_sub:
                    cand = os.path.join(npz_sub, f'{basename}{self.suffix}')
                    if os.path.exists(cand):
                        nam_file = cand
                        break
                cand = os.path.join(sub_path, f'{basename}{self.suffix}')
                if os.path.exists(cand):
                    nam_file = cand
                    break

        if os.path.exists(nam_file):
            data = np.load(nam_file, allow_pickle=False)
            graph_data = {
                'label_map':  torch.from_numpy(data['label_map']).long(),
                'node_wld':   torch.from_numpy(data['node_wld']).float(),
                'node_area':  torch.from_numpy(data['node_area']).float(),
                'node_lab':   torch.from_numpy(data['node_lab']).float(),
                'node_stat':  torch.from_numpy(data['node_stat']).float() if 'node_stat' in data else torch.zeros(data['node_wld'].shape[2], 6),
                'node_glcm':  torch.from_numpy(data['node_glcm']).float() if 'node_glcm' in data else torch.zeros(data['node_wld'].shape[2], 4),
                'edge_index': torch.from_numpy(data['edge_index']).long(),
                'edge_dc':    torch.from_numpy(data['edge_dc']).float(),
                'edge_dt':    torch.from_numpy(data['edge_dt']).float(),
                'edge_dbc':   torch.from_numpy(data['edge_dbc']).float(),
            }
        else:
            # Fallback: empty graph
            h, w = results.get('img_shape', (512, 512))[:2]
            graph_data = {
                'label_map':  torch.zeros((h, w), dtype=torch.long),
                'node_wld':   torch.zeros((1, 3, 59), dtype=torch.float32),
                'node_area':  torch.zeros((1,), dtype=torch.float32),
                'node_lab':   torch.zeros((1, 3), dtype=torch.float32),
                'node_stat':  torch.zeros((1, 6), dtype=torch.float32),
                'node_glcm':  torch.zeros((1, 4), dtype=torch.float32),
                'edge_index': torch.zeros((2, 0), dtype=torch.long),
                'edge_dc':    torch.zeros((0,), dtype=torch.float32),
                'edge_dt':    torch.zeros((0,), dtype=torch.float32),
                'edge_dbc':   torch.zeros((0,), dtype=torch.float32),
            }

        # 存入 img_metas 避免 collate 批处理冲突 (不同图N/E不同)
        results[self.key] = graph_data
        if 'img_metas' in results and isinstance(results['img_metas'], dict):
            results['img_metas'][self.key] = graph_data
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'nam_dir={self.nam_dir}, suffix={self.suffix})')


# ============================================================================
# EdgeGATLayer: 边条件图注意力层
# ============================================================================
class EdgeGATLayer(nn.Module):
    """单层边条件图注意力 (Edge-conditioned Graph Attention)。

    对于每条边 (i→j):
      attn_ij = LeakyReLU( a^T [W_q·h_i || W_k·h_j] + w_e^T·e_ij )
      h_i'     = Σ_j softmax_j(attn_ij) · W_v·h_j

    边特征 e_ij = [DC, DT, DBC] 直接调制注意力分数：
      - DC 大 → 颜色差异大 → 可能是解剖边界（attention ↓）或病灶边界（attention ↑）
      - DT 大 → 纹理差异大 → 病灶纹理突变信号（attention ↑）
      - DBC 大 → 边界锐利 → 病灶边缘特征（attention ↑）

    Args:
        in_dim (int): 节点输入维度
        out_dim (int): 节点输出维度
        edge_dim (int): 边特征维度 (默认 3 = DC/DT/DBC)
        heads (int): 多头注意力头数
        dropout (float): attention dropout
    """

    def __init__(self, in_dim: int, out_dim: int,
                 edge_dim: int = 3, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = dropout

        # QKV 投影
        self.W_q = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim * heads, bias=False)

        # 边特征 → 注意力偏置
        self.W_e = nn.Sequential(
            nn.Linear(edge_dim, heads),
            nn.LeakyReLU(0.2),
        )

        # 输出
        self.out_proj = nn.Linear(out_dim * heads, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 节点特征 (N, in_dim)
            edge_index (LongTensor): 边索引 (2, E)，无向边
            edge_attr (Tensor): 边特征 (E, edge_dim) [DC, DT, DBC]
        Returns:
            Tensor: 更新后的节点特征 (N, out_dim)
        """
        N = x.size(0)
        E = edge_index.size(1)
        H = self.heads
        D = self.out_dim

        # === QKV ===
        q = self.W_q(x).view(N, H, D)   # (N, heads, out_dim)
        k = self.W_k(x).view(N, H, D)
        v = self.W_v(x).view(N, H, D)

        src, dst = edge_index[0], edge_index[1]  # (E,)

        # Q·K per head: (E, heads)
        qk = (q[src] * k[dst]).sum(dim=-1)   # dot product

        # Edge bias: (E, heads)
        e_bias = self.W_e(edge_attr)          # (E, heads)

        # Attention score
        attn = F.leaky_relu(qk + e_bias, 0.2)  # (E, heads)

        # Softmax per destination node
        # 对每个 dst 节点，其入边的 attn 做 softmax
        attn_exp = attn.exp()
        # Scatter sum of exp per dst
        sum_per_dst = torch.zeros(N, H, device=x.device)
        sum_per_dst = sum_per_dst.index_add(0, dst, attn_exp)
        sum_per_dst = sum_per_dst.clamp(min=1e-8)  # avoid div by 0
        attn_norm = attn_exp / sum_per_dst[dst]     # (E, heads)

        attn_norm = F.dropout(attn_norm, p=self.dropout, training=self.training)

        # Aggregate: weighted sum of V per destination
        v_weighted = v[src] * attn_norm.unsqueeze(-1)  # (E, heads, out_dim)
        out = torch.zeros(N, H, D, device=x.device)
        out = out.index_add(0, dst, v_weighted)         # (N, heads, out_dim)

        # Concat heads + project
        out = out.view(N, H * D)
        out = self.out_proj(out)
        out = self.norm(out + x[..., :self.out_dim]) if x.size(-1) == self.out_dim else self.norm(out)
        return out


# ============================================================================
# NAMGraphBranch: 图注意力 NAM 分支
# ============================================================================
class NAMGraphBranch(nn.Module):
    """图注意力 NAM 分支 v2。

    输入: NAM 区域图 (.npz 数据)
    流程:
      区域特征 (WLD+area) ──MLP──→ node_emb
      边特征 (DC+DT+DBC)  ──MLP──→ edge_emb
         ↓
      EdgeGAT × 2 层 ─→ node_out (每个区域的"异常感知"特征)
         ↓
      label_map 回投 ─→ pixel_feat (H, W, C)
         ↓
      residual = backbone_feat - smooth(pixel_feat)

    Args:
        feat_channels (int): 输出特征通道数 (= FPN feat_channels)
        gat_hidden (int): GAT 中间维度
        gat_heads (int): 注意力头数
        gat_layers (int): GAT 层数
    """

    def __init__(self, feat_channels: int = 256,
                 gat_hidden: int = 256, gat_heads: int = 4,
                 gat_layers: int = 2):
        super().__init__()

        # === 节点编码: LBP(177) + stat(6) + glcm(4) + lab(3) + dc_agg(2) + area(1) ===
        # LBP cross/outside=0.76x: TB边界DT偏低→弱但可学习
        # LBP(354) + stat(6) + glcm(4) + lab(3) + dc_agg(2) + area(1)
        # TODO: re-generate npz with 59-bin after ComputeRegion.cpp fix
        wld_dim = 3 * 118
        node_dim = wld_dim + 1 + 3 + 2 + 6 + 4  # 370
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(node_dim),  # 多源特征值域差异大(0.01~227), 归一化防NaN
            nn.Linear(node_dim, gat_hidden * 2),
            nn.LayerNorm(gat_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(gat_hidden * 2, gat_hidden),
            nn.LayerNorm(gat_hidden),
        )

        # === 边特征归一化: DC/DT/DBC 量纲统一 ===
        self.edge_scale = nn.Parameter(torch.tensor([0.02, 1.0, 0.02]))

        # === 边编码: DC + DT + DBC → gat_hidden ===
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, gat_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gat_hidden // 2, 3),  # 3-dim for GAT edge bias
        )

        # === GAT 堆叠 ===
        self.gat_layers = nn.ModuleList()
        for _ in range(gat_layers):
            self.gat_layers.append(
                EdgeGATLayer(in_dim=gat_hidden, out_dim=gat_hidden,
                             edge_dim=3, heads=gat_heads, dropout=0.1))

        # === 回投: node feature → 像素空间平滑特征 ===
        self.pixel_proj = nn.Sequential(
            nn.Linear(gat_hidden, feat_channels),
            nn.LayerNorm(feat_channels),
        )

        # === 像素残差融合 ===
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 1),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))

        # 零初始化最后一层
        nn.init.zeros_(self.fusion_conv[-1].weight)
        if self.fusion_conv[-1].bias is not None:
            nn.init.zeros_(self.fusion_conv[-1].bias)

    def _process_single_graph(self, graph_data: dict,
                               feat_raw: torch.Tensor) -> torch.Tensor:
        """处理单个样本的图数据。

        Args:
            graph_data: LoadNAMGraph 加载的 dict, 所有 tensor 在目标 device 上
            feat_raw: (C, H, W) 单张图 FPN 特征
        Returns:
            (C, H, W) NAM 分支输出
        """
        node_wld = graph_data['node_wld']       # (N, 3, 48)
        node_area = graph_data['node_area']      # (N,)
        edge_index = graph_data['edge_index']    # (2, E)
        edge_dc = graph_data['edge_dc']          # (E,)
        edge_dt = graph_data['edge_dt']          # (E,)
        edge_dbc = graph_data['edge_dbc']        # (E,)
        label_map = graph_data['label_map']      # (H, W)

        N = node_wld.size(0)
        if N <= 1:
            return feat_raw.new_zeros(feat_raw.shape)

        # === 1. 节点编码: LBP纹理 + 密度 + DC聚合 + 面积 ===
        wld_flat = node_wld.view(N, -1)                     # (N, 177)
        area_log = torch.log1p(node_area).unsqueeze(-1)      # (N, 1)
        # 密度信号: node_lab (已有的Lab颜色均值, L通道=区域亮度)
        node_lab = graph_data.get('node_lab',
                    node_wld.new_zeros(N, 3))                # (N, 3)
        # DC聚合: 每条边的颜色距离 → 聚合到目标节点
        # 让 DC 信号(已知TB边界≈15) 首次进入节点特征
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            dc_sum = torch.zeros(N, device=feat_raw.device)
            dc_cnt = torch.zeros(N, device=feat_raw.device)
            dc_sum = dc_sum.index_add(0, dst, edge_dc)
            dc_cnt = dc_cnt.index_add(0, dst, torch.ones_like(edge_dc))
            dc_mean = dc_sum / dc_cnt.clamp(min=1)           # (N,)
            dc_max = torch.zeros(N, device=feat_raw.device)
            for e in range(edge_index.size(1)):
                d, dc = dst[e].item(), edge_dc[e].item()
                if dc > dc_max[d]: dc_max[d] = dc
            dc_feat = torch.stack([dc_mean, dc_max], dim=-1)  # (N, 2)
        else:
            dc_feat = node_wld.new_zeros(N, 2)

        # 一阶灰度统计 + GLCM (新增, 旧npz无则补零)
        node_stat = graph_data.get('node_stat',
                     node_wld.new_zeros(N, 6))           # (N, 6)
        node_glcm = graph_data.get('node_glcm',
                     node_wld.new_zeros(N, 4))           # (N, 4)

        node_x = torch.cat([wld_flat, node_lab, dc_feat, node_stat,
                           node_glcm, area_log], dim=-1)
        node_x = self.node_encoder(node_x)                   # (N, gat_hidden)

        # === 2. 边编码: 量纲归一化 ===
        if edge_index.size(1) > 0:
            dc_norm = (edge_dc - 15.0) * self.edge_scale[0]
            dt_norm = torch.log1p(edge_dt * 1000) * self.edge_scale[1]
            dbc_norm = (edge_dbc - 8.0) * self.edge_scale[2]
            edge_attr = torch.stack([dc_norm, dt_norm, dbc_norm], dim=-1)
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = edge_dc.new_zeros(0, 3)

        # === 3. GAT ===
        for gat in self.gat_layers:
            node_x = gat(node_x, edge_index, edge_attr)

        # === 4. 回投像素空间 ===
        pixel_emb = self.pixel_proj(node_x)                     # (N, C)
        pixel_feat = pixel_emb[label_map.long()]                # (H, W, C)
        pixel_feat = pixel_feat.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        # === 5. 区域平滑 → 纹理异常 ===
        # 把区域特征视为"区域平滑表示"
        # residual = 原始特征 - 平滑特征 = 纹理异常
        pixel_feat = F.interpolate(pixel_feat, size=feat_raw.shape[-2:],
                                   mode='bilinear', align_corners=False)
        residual_raw = feat_raw.unsqueeze(0) - pixel_feat
        feat2 = self.fusion_conv(
            torch.cat([pixel_feat, residual_raw], dim=1))      # (1, C, H, W)
        return feat2.squeeze(0)

    def forward(self, feat_raw: torch.Tensor,
                nam_graphs: list) -> torch.Tensor:
        """批量前向。

        Args:
            feat_raw: (B, C, H, W) FPN 特征
            nam_graphs: list[dict], 每张图的图数据 (长度 = B)
        Returns:
            (B, C, H, W) NAM 分支输出
        """
        B = feat_raw.size(0)
        outputs = []
        for b in range(B):
            g = {k: v.to(feat_raw.device) if isinstance(v, torch.Tensor) else v
                 for k, v in nam_graphs[b].items()}
            out = self._process_single_graph(g, feat_raw[b])
            outputs.append(out.unsqueeze(0))
        return torch.cat(outputs, dim=0)


# ============================================================================
# RetinaNAMGuideAttHeadV2: 双分支 Head (Graph 版本)
# ============================================================================
import sys as _sys
_retina_guide = _sys.modules.get('mmdet.models.dense_heads.retina_guide_att_head')
if _retina_guide is not None:
    RetinaGuideAttHead = _retina_guide.RetinaGuideAttHead
else:
    from ..dense_heads.retina_guide_att_head import RetinaGuideAttHead  # noqa: F811

from ..builder import HEADS


@HEADS.register_module()
class RetinaNAMGuideAttHeadV2(RetinaGuideAttHead):
    """双分支 RetinaNet Head v2: SymFormer + NAM Graph 分支。

    与 RetinaNAMGuideAttHead 唯一的区别:
      NAM 分支从 CNN (v1) 换成 EdgeGAT (v2)

    Config:
        bbox_head=dict(
            type='RetinaNAMGuideAttHeadV2',
            gat_hidden=256, gat_heads=4, gat_layers=2,
            num_classes=2, num_query=500, ...
        )
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 gat_hidden: int = 256,
                 gat_heads: int = 4,
                 gat_layers: int = 2,
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
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads
        self.gat_layers = gat_layers
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
        super()._init_layers()
        self.nam_branch = NAMGraphBranch(
            feat_channels=self.feat_channels,
            gat_hidden=self.gat_hidden,
            gat_heads=self.gat_heads,
            gat_layers=self.gat_layers)

    def forward_single(self, x, nam_graphs=None):
        """单个 FPN 层级前向。

        Args:
            x (Tensor): (B, C, H, W) FPN 特征
            nam_graphs (list[dict] or None): 图数据列表
        """
        bs, _, h, w = x.shape
        masks = x.new_zeros((bs, h, w)).to(torch.bool)

        # Branch 1: SymFormer
        pos_embed = self.positional_encoding(masks, x)
        feat1 = self.transformer(
            tuple([x]), [masks], self.query_embedding.weight, [pos_embed])

        # Branch 2: NAM Graph
        if nam_graphs is not None and any(g is not None for g in nam_graphs):
            feat2 = self.nam_branch(x, nam_graphs)
            fused = feat1 + self.nam_branch.alpha * feat2
        else:
            fused = feat1

        cls_feat, reg_feat = fused, fused
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def forward(self, feats, nam_graphs=None):
        """多层级前向。nam_graphs 是 per-image 的图数据，各 FPN 层级共享。"""
        cls_scores, bbox_preds = [], []
        for feat in feats:
            cls_score, bbox_pred = self.forward_single(feat, nam_graphs)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return tuple(cls_scores), tuple(bbox_preds)

    def simple_test(self, feats, img_metas, rescale=False, nam_graphs=None):
        """覆盖父类 simple_test, 将 nam_graphs 传入 forward。"""
        outs = self.forward(feats, nam_graphs=nam_graphs)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list


# ============================================================================
# RetinaNAMAttV2: 检测器 (Graph 版本)
# ============================================================================
_single_stage = _sys.modules.get('mmdet.models.detectors.single_stage')
if _single_stage is not None:
    SingleStageDetector = _single_stage.SingleStageDetector
else:
    from ..detectors.single_stage import SingleStageDetector  # noqa: F811

_single_stage_cls = _sys.modules.get('mmdet.models.detectors.single_stage_class')
if _single_stage_cls is not None:
    SingleStageClsDetector = _single_stage_cls.SingleStageClsDetector
else:
    from ..detectors.single_stage_class import SingleStageClsDetector  # noqa: F811

from ..builder import DETECTORS


@DETECTORS.register_module()
class RetinaNAMAttV2(SingleStageDetector):
    """单阶段检测器，传递 nam_graph 数据到 Graph Head。

    Config:
        model=dict(
            type='RetinaNAMAttV2',
            backbone=dict(...),
            neck=dict(...),
            bbox_head=dict(type='RetinaNAMGuideAttHeadV2', ...),
        )
    """

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        x = self.extract_feat(img)

        # 从 img_metas 提取 nam_graph (LoadNAMGraph 存入 img_metas['nam_graph'])
        nam_graphs = [meta.get('nam_graph') for meta in img_metas
                      if isinstance(meta, dict) and 'nam_graph' in meta]
        if not nam_graphs:
            nam_graphs = None

        outs = self.bbox_head(feats=x, nam_graphs=nam_graphs)
        losses = self.bbox_head.loss(
            *outs, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        x = self.extract_feat(img)
        nam_graphs = [meta.get('nam_graph') for meta in img_metas
                      if isinstance(meta, dict) and 'nam_graph' in meta]
        if not nam_graphs:
            nam_graphs = None

        from mmdet.core import bbox2result
        results_list = self.bbox_head.simple_test(
            x, img_metas, rescale=rescale, nam_graphs=nam_graphs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


# ============================================================================
# RetinaNAMAttV2Cls: 分类阶段检测器 (Graph 版本)
# ============================================================================
@DETECTORS.register_module()
class RetinaNAMAttV2Cls(SingleStageClsDetector):
    """分类阶段检测器，接收 nam_graph 但不使用 (cls 阶段只跑 classifier)。

    Config:
        model=dict(
            type='RetinaNAMAttV2Cls',
            backbone=dict(...),
            neck=dict(...),
            bbox_head=dict(...),
            classifier=dict(input_dim=576),
        )
    """

    def forward_train(self,
                      img,
                      img_metas,
                      gt_classes,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """与 SingleStageClsDetector 一致，额外接收 nam_graph kwargs。"""
        return super().forward_train(img, img_metas, gt_classes,
                                     gt_bboxes, gt_labels, gt_bboxes_ignore)
