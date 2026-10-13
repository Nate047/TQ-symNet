# 历史命令归档：旧数据划分

当前cfg已更新。请使用`RUN_SEED0_479.md`和本次新训练checkpoint，不执行下方旧路径命令。

## 原TQ-SymNet 实验顺序

当前采用单seed、质量网络重点方案，首轮优先按`RUN_CORE_SINGLE_SEED.md`运行。
下方保留全配置命令目录；多seed与主干/neck细分不是当前首轮必跑项。

所有质量组都冻结同一个检测器，并使用相同的训练JSON、采样策略、候选预算、Soft-NMS、batch和8个质量训练epoch。每组先跑seed 0做主效应，再对保留组跑seed 999和seed 2024。

## 第一批：主效应和必要消融

```bash
cd /home/data2/LiYingxian/Tuberculosis-main/TQ_SymNet

# Foundation controls: train M2/M1 first; B0/B1/B2 are inference-only replay controls.
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/m2_transnext_crossscale.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/m2_transnext_crossscale_s0
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/m1_transnext_fpn.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/m1_transnext_fpn_s0
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/m0_p2t_fpn.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/m0_p2t_fpn_s0

CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/b0_sym_hard.py \
  work_dirs/TQ_SymNet/m2_transnext_crossscale_s0/latest.pth --out-dir work_dirs/TQ_SymNet/b0_sym_hard_eval --diagnostics
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/b1_sym_soft.py \
  work_dirs/TQ_SymNet/m2_transnext_crossscale_s0/latest.pth --out-dir work_dirs/TQ_SymNet/b1_sym_soft_eval --diagnostics
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/b2_sym_temperature.py \
  work_dirs/TQ_SymNet/m2_transnext_crossscale_s0/latest.pth --out-dir work_dirs/TQ_SymNet/b2_sym_temperature_eval --diagnostics

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/a0_full.py \
  --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a0_full_s0

CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/a1_no_texture.py \
  --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a1_no_texture_s0

CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/a4_no_detail.py \
  --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a4_no_detail_s0

CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/a7_no_rank.py \
  --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a7_no_rank_s0

CUDA_VISIBLE_DEVICES=4 python tools/train.py configs/a8_direct_iou.py \
  --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a8_direct_iou_s0

CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/a9_high_threshold.py \
  --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a9_high_threshold_s0

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/a10_light_iou.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a10_light_iou_s0
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/a11_learned_fusion.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a11_learned_fusion_s0
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/a12_direct_iou_no_rank.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a12_direct_iou_no_rank_s0
```

## 第二批：纹理和空间细分

只在第一批证明完整网络有效后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/a2_no_lbp.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a2_no_lbp_s0
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/a3_no_gabor.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a3_no_gabor_s0
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/a5_no_side_query.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a5_no_side_query_s0
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/a6_no_gaussian.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a6_no_gaussian_s0
```

## 每个checkpoint的评测

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/test_val.py \
  work_dirs/TQ_SymNet/a0_full_s0/best_official_tb_bbox_mAP_epoch_*.pth --eval bbox

CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/test_subtype.py \
  work_dirs/TQ_SymNet/a0_full_s0/best_official_tb_bbox_mAP_epoch_*.pth --eval bbox
```

实际运行时将`*`替换成明确的checkpoint文件名。质量组应保存：overall TB AP/AP50/AP75、Active/Obsolete AP/AP50、每epoch质量统计、rank统计、Soft-NMS前后候选分布和每4epoch Oracle诊断。B组保存metrics.json、candidate_stages.json和predictions。

## 主干和neck对照

这三组与质量消融分开做：

1. 原Sym + 原FPN（原仓库配置）；
2. TransNeXt + 原FPN（`foundation_transnext_fpn.py`）；
3. TransNeXt + CrossScale（`b0_sym_hard.py`或`train_detector.py`）。

三组均关闭TQ质量网络，使用相同检测训练数据、EMA、epoch和输入尺寸；B0/B1只作为同一M2 checkpoint的后处理对照。第3组是`m2_transnext_crossscale.py`，第2组是`m1_transnext_fpn.py`，第1组仍使用仓库原Sym/FPN配置。只有M2固定后，才运行质量网络消融。

## 最终online

在开发集和三seed结果冻结后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_online.py \
  configs/test_online.py \
  work_dirs/TQ_SymNet/a0_full_s0/best_official_tb_bbox_mAP_epoch_*.pth \
  --json-prefix work_dirs/TQ_SymNet/submission/tq_tb
```

导出后只检查JSON的image_id、xywh、score和category_id，不在隐藏test上调参。

