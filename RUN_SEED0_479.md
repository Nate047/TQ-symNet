# TQ-SymNet：固定seed0，479/120/200重新训练

> 本页为已归档v1命令。当前cfg已切换balanced-v2，请使用
> [RUN_BALANCED_V2_SINGLE_GPU.md](RUN_BALANCED_V2_SINGLE_GPU.md)，不要执行下列旧路径命令。

仅两张卡时，优先使用同级`experiment_runner/README_TWO_GPU.md`自动队列，避免手动分配六张卡。

## 数据与模型约束

- 唯一活动划分：`data/splits/tbx11k_479_120_200_seed0/seed_0/`。
- train：479图/728框；val：120图/183框；test：200图/300框。
- 双类JSON保留Active/Obsolete；同名`*_tb.json`合成category_id=1，用于GLIP类单类复现，图像/框坐标完全相同。
- 训练使用479张全量一次＋纯Obsolete83张重复3次，有效728图/epoch。val/test不重复。
- 主网络16epoch、batch2、EMA；质量阶段冻结主网络，8epoch、batch2；命令统一seed0＋deterministic。
- 保留MAX＋全IoU质量监督，融合固定0.5；无肺mask、无H2、无pTB/ResNet后过滤器。
- 主指标`official_tb_bbox_mAP`；保存最佳只看120张val。200张test不用于调参或择优。
- 新test中157图曾属于旧599张训练，禁止加载旧任务checkpoint；ImageNet预训练不受此限制。

从项目根目录运行；图像路径仍为`/home/data2/LiYingxian/Dataset_TB/TBX11K/imgs`。
若服务器路径不同，只修改`configs/_common.py`的`img_prefix`与主干预训练路径。

## 第一步：重新训练主网络

```bash
cd /home/data2/LiYingxian/TQ_SymNet
python tools/check_structure.py
python tools/check_standalone.py --runtime
python tools/preflight.py --config configs/m2_transnext_crossscale.py --server-files --build
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/m2_transnext_crossscale.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/m2_s0
python tools/select_detector_checkpoint.py
```

最后一条只复制VAL选出的最佳M2，生成`m2_s0/selected_detector.pth`和来源记录。
所有质量组读取此稳定路径，只加载backbone/neck/head，质量网重新随机初始化。
训练日志和checkpoint保存train/val/test哈希；旧划分checkpoint会被拒绝。

## 第二步：质量网络核心消融

先跑A0完整模型；其余组可并行，每条命令独占一张卡，卡号可自行改。

```bash
python tools/preflight.py --config configs/a0_full.py --server-files --build
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/a0_full.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/a0_s0
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/a1_no_texture.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/a1_s0
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/a4_no_detail.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/a4_s0
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/a7_no_rank.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/a7_s0
CUDA_VISIBLE_DEVICES=4 python tools/train.py configs/a8_direct_iou.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/a8_s0
CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/a12_direct_iou_no_rank.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/a12_s0
```

|组|相对A0的改动|回答的问题|
|---|---|---|
|A0|完整质量网络|最终方法|
|A1|关闭LBP/Gabor纹理分支|纹理先验是否有效|
|A4|关闭原图细节编码|原图信息是否必要|
|A7|关闭rank|排序监督是否有增益|
|A8|全阈值质量目标改为直接IoU|目标设计是否有效|
|A12|直接IoU且无rank|避免把目标和rank效应混为一谈|

现有其他A组也已切到同一划分，可补做LBP/Gabor单独、高斯、四边query、轻量质量头消融。
`A11`是可学习融合门控；不是A0固定0.5的同义组。

## 主干和neck的贡献（从头训练，非质量阶段）

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/m0_p2t_fpn.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/m0_s0
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/m1_transnext_fpn.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/m1_s0
```

M0→M1主要比较主干，M1→M2比较neck；三者都不启用质量网络。
M0是本项目架构控制，不冒充原官方Sym完整复现。

## 最终锁定后的200张test

先根据val锁定方法和checkpoint，再运行这些命令，不能看test结果后重选最佳epoch。
B0/B1只推理，不重训。B0/B1共用本次M2；A组使用各自VAL最佳quality checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/b0_sym_hard.py work_dirs/seed0_479_120_200/m2_s0/selected_detector.pth --mode tb --out-dir work_dirs/seed0_479_120_200/test_b0
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/b1_sym_soft.py work_dirs/seed0_479_120_200/m2_s0/selected_detector.pth --mode tb --out-dir work_dirs/seed0_479_120_200/test_b1
```

质量组把下面`实际epoch`换成该组val选出的epoch（不要用test挑选）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/a0_full.py work_dirs/seed0_479_120_200/a0_s0/best_official_tb_bbox_mAP_epoch_实际epoch.pth --mode tb --out-dir work_dirs/seed0_479_120_200/test_a0
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/a0_full.py work_dirs/seed0_479_120_200/a0_s0/best_official_tb_bbox_mAP_epoch_实际epoch.pth --mode subtype --out-dir work_dirs/seed0_479_120_200/test_a0_subtype
```

普通A/B cfg默认测试200张。`test_val.py`明确只测试120张val。
官方3302张online test入口仍保留，使用`test_online.py`，与这次公开数据再划分test分开报告。

## 迁移与检查范围

更新包按项目根目录覆盖，包含cfg、数据契约hook、检查/选模/评测工具及划分JSON。
不覆盖MMDet/CUDA算子，不打包图片和权重。旧日志/权重不删除，使用新work-dir隔离。
本机完成静态配置、源码保真和划分哈希检查；完整MMCV/CUDA构建与训练须在服务器执行。
