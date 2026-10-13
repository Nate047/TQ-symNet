# TQ-SymNet 独立MMDetection项目

本目录是独立项目根目录，不依赖外层项目的mmdet或tools，保留MMDetection 2.x结构。
模型算法、CUDA算子、数据处理和评测协议不因本次打包改变。

2026-10-09起，当前cfg采用公开TB池的固定seed0划分：479训练/120验证/200测试。
活动命令和迁移说明见 [RUN_SEED0_479.md](RUN_SEED0_479.md)。旧599/200结果不属于这套划分。

```text
TQ_SymNet/                 # 独立项目根目录，可迁移到任意路径
  mmdet/                   # 本地MMDetection：训练、数据集、变换、评测、Sym/TransNeXt
  tools/train.py           # 原训练入口
  tools/test.py            # 原测试入口
  tools/evaluate.py        # TQ各组自身cfg评测
  configs/                 # 自包含配置链
  TQ_SymNet/               # 拆分后的Python模型包
  requirements/
  setup.py
```

## 环境和资产

沿用服务器已能训练的PyTorch/torchvision/mmcv-full/timm/CUDA环境，不盲目升级。
Sym依赖已编译的mmcv._ext，TransNeXt依赖swattention；它们是环境依赖，
不是外层项目源码依赖，本次不改写或重新编译CUDA算子。
图片、checkpoint和预训练权重不打包；本次划分JSON在data/splits/tbx11k_479_120_200_seed0。
图片和预训练权重仍沿用cfg中的服务器绝对路径；不需要肺mask。

历史NAM单文件仅作可选数值对照档案，不由mmdet注册入口自动导入。
实际模型通过custom_imports=['TQ_SymNet.models']注册。
直接从根目录启动，无需pip安装；如安装本地mmdet需先确认环境替换，不重装CUDA扩展。

## 启动

```bash
cd /你的服务器目录/TQ_SymNet
python tools/check_structure.py
python tools/check_standalone.py
python tools/check_standalone.py --runtime
python tools/preflight.py --config configs/m2_transnext_crossscale.py --server-files --build
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/m2_transnext_crossscale.py --seed 0 --deterministic --work-dir work_dirs/seed0_479_120_200/m2_s0
python tools/select_detector_checkpoint.py
```

活动消融命令见RUN_SEED0_479.md，开关在各cfg。
默认A组共享本次479张训练所得M2的主干/neck/head，质量网随机初始化，8epoch、seed0、融合固定0.5。
不得加载旧M2/D3/H2的任务训练权重：新测试集中157张曾属于旧训练集。

## 测试

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/a0_full.py /实际权重.pth --mode tb --out-dir work_dirs/a0_s0/eval_tb --diagnostics
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py configs/a0_full.py /实际权重.pth --mode subtype --out-dir work_dirs/a0_s0/eval_subtype
```

各组使用自身cfg，两种模式使用同一权重。整体TB为COCO bbox AP .50:.05:.95、
useCats=0、maxDets=100；亚型单独类别感知评测。同划分同任务同口径才能直接比较。

原作者tools/test_sym_original.py保持字节不变，独立项目用tools/test_sym.py包装调用，
保证优先导入本项目mmdet。官方ALL需要真实Health/Sick/TB图像分类器；
质量网不是该分类器。export_online.py只导出bbox JSON，不自动提交。

## 验证范围

静态配置、源码哈希和数值组件检查不等于服务器全模型验证。
完整MMCV/CUDA构建、实际checkpoint及数据评测仍需服务器执行。
compare_legacy.py仅为可选历史对照，需要显式旧cfg，不是项目训练依赖。
