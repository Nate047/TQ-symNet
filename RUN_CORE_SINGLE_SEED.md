# 历史命令归档：旧599张训练划分

当前479/120/200协议必须使用`RUN_SEED0_479.md`，以下旧命令和checkpoint不要执行。

## 原首轮：单seed、质量网络重点实验

本文件为当前首轮命令。seed统一为0；主检测器来源在`configs/_matched_quality.py`中的
`detector_checkpoint`只设置一次，全部A/B组必须一致。

目前默认来源是：
`work_dirs/NAM_v59r1/d3_gate_seed0/best_official_tb_bbox_mAP_epoch_4.pth`。
A组只加载其中主干/neck/head，丢弃全部质量权重并重新初始化，固定alpha=0.5，8epoch。

## 迁移后检查

```bash
cd /home/data2/LiYingxian/Tuberculosis-main/TQ_SymNet
python tools/check_structure.py
python tools/check_standalone.py --runtime
python tools/preflight.py --server-files --build
```

肺mask不需要。`compare_legacy.py`仅为可选历史对照，需显式提供旧配置；独立项目不依赖旧项目目录。

## B组：同一来源权重，只评测

```bash
DET=/home/data2/LiYingxian/Tuberculosis-main/work_dirs/NAM_v59r1/d3_gate_seed0/best_official_tb_bbox_mAP_epoch_4.pth
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/b0_sym_hard.py "$DET" --out-dir work_dirs/TQ_SymNet/b0_s0 --diagnostics
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/b1_sym_soft.py "$DET" --out-dir work_dirs/TQ_SymNet/b1_s0 --diagnostics
```

## A组：6卡首批，各自单卡batch2

每条在独立终端运行。若卡号不同，修改CUDA_VISIBLE_DEVICES。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/a0_full.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a0_s0
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/a1_no_texture.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a1_s0
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/a4_no_detail.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a4_s0
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/a7_no_rank.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a7_s0
CUDA_VISIBLE_DEVICES=4 python tools/train.py configs/a8_direct_iou.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a8_s0
CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/a10_light_iou.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a10_s0
```

有卡释放后补第7组（A10的同目标、同rank强对照）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/a12_direct_iou_no_rank.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/a12_s0
```

如有第7卡，把该条CUDA_VISIBLE_DEVICES改为6即可同时运行。

## 各组按自己的cfg评测

下面以A0为例。将QCKPT设为该组实际选定的best权重文件；不能用通用完整模型cfg加载去组件的权重。

```bash
QCKPT=/实际路径/best_official_tb_bbox_mAP_epoch_实际轮次.pth
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/a0_full.py "$QCKPT" --mode tb --out-dir work_dirs/TQ_SymNet/a0_s0/eval_tb --diagnostics
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/a0_full.py "$QCKPT" --mode subtype --out-dir work_dirs/TQ_SymNet/a0_s0/eval_subtype
```

对其他A组替换cfg和QCKPT。同一组的TB和亚型使用同一权重，不各自挑最好轮次。

## 如果主网络也要从头训练

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/m2_transnext_crossscale.py --seed 0 --deterministic --work-dir work_dirs/TQ_SymNet/m2_s0
```

随后将`_matched_quality.py`的detector_checkpoint及B组DET统一指向选定M2权重，再启动全部A组。
M2与历史D3来源的结果不得交叉拼接；M0/M1/M3主干neck细分暂缓。

## 最终online

只对最终选定版本运行导出。组件变体须先用它自己的cfg构建模型，再明确改test索引为官方TBX11K_test.json。
默认`test_online.py`适用于完整attention架构；不能直接导出去组件或轻量架构权重。
最终提交规则和checkpoint固定后才导出，不自动消耗提交次数。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_online.py configs/test_online.py "$QCKPT" --json-prefix work_dirs/TQ_SymNet/submission/tq_tb
```

online返回指标、效率测量和病例可视化另行补齐，单seed不报告跨seed标准差。

