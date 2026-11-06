# TQ-SymNet

胸部X线肺结核病灶检测的独立 MMDetection 2.x 实现。主检测器采用
TransNeXt / CrossScale / Sym，冻结后训练候选质量网络，并学习全局融合系数，
在 Gaussian Soft-NMS 前融合亚型置信度与质量分。质量网络不修改候选坐标，
推理不使用真实标注，也不需要肺部分割 mask。

本页只介绍论文主实验。历史消融与探索配置保留在仓库中，不属于下面的运行流程。

## 主实验与结果

- 检测队列：599张TB阳性训练图像 / 200张验证图像，902 / 309个框。
- 质量网络：冻结同一M2检测器，训练8轮，按验证集整体TB AP选择checkpoint。
- 最终模型：原质量读取结构 + 全阈值监督 + rank + 可学习融合系数。
- 论文选定模型为质量训练第7轮，α约0.53571，验证AP/AP50/AP75为40.1/77.5/35.7。
- 官方隐藏ALL测试AP/AP50/AP75为35.11/73.53/31.54，论文记录提交ID为1002246。

本地验证集仅含TB阳性图像，不能与包含整图筛查的隐藏ALL分数直接比较。
固定α控制、离线α扫描和最终可学习融合模型不是同一个实验。
新训练需按自身验证结果选模，不能保证复现同一峰值或同一最佳轮次。

## 1. 环境与文件准备

建议使用原Linux环境：Python 3.8、PyTorch 1.10、CUDA 11.3、兼容的torchvision、
MMCV 1.7.1（含CUDA扩展）及timm。TransNeXt的swattention扩展需与PyTorch/CUDA匹配。
requirements.txt不负责安装PyTorch、mmcv-full或编译CUDA算子，不要盲目升级已有环境。

```bash
git clone https://github.com/Nate047/TQ-symNet.git
cd TQ-symNet
# 先激活已配置好的MMDetection 2.x环境
python -m pip install -r requirements.txt
python -c "import mmdet; print(mmdet.__file__)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

所有命令从仓库根目录执行；mmdet应指向本仓库。下面均为单卡运行。

按TBX11K原始使用条款准备图像、检测标注及整图分类索引：
`TBX11K_train_only_tb.json`、`TBX11K_val_only_tb.json`、
`all_train.json`、`all_val.json`和无GT的`all_test.json`。
整图分类还使用官方文件列表中的MC/Shenzhen及DA/DB补充图像。

修改以下位置的服务器示例路径，使其指向自己的文件：

| 用途 | 路径设置位置 |
|---|---|
| 检测训练/验证图像和JSON | `configs/publication_alpha/_original_a0.py`中的`data`及初始化hook |
| 冻结M2 checkpoint | 同文件中的`detector_checkpoint` |
| C20分类数据和初始M2 | `configs/classification/c0_backbone_sym.py`，由C20配置继承 |
| 官方测试图像索引 | `configs/publication_alpha/test_online_learned.py` |

主实验要求原始M2第13轮checkpoint，默认放置于：
`work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth`。

其SHA-256为：
`4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74`。

初始化hook还核验训练/验证JSON哈希。可以修改文件位置，不应绕过内容校验后
将不同队列或不同检测器的结果称为相同实验。
当前以下流程从指定的冻结M2开始；仓库中的后续479/120/200检测训练配置
不能替代原599/200实验的检测器重训流程。完整从头重训该历史M2仍需对应配置与资产核验。

## 2. 训练主质量网络

```bash
python tools/train.py configs/publication_alpha/a1_learned.py \
  --seed 0 --deterministic \
  --work-dir work_dirs/publication_alpha/a1_learned_s0
```

主检测器保持冻结，质量网络重新初始化，融合系数参与训练。
不要复用其他实验的非空work-dir。训练种子及确定化设置不承诺CUDA逐位一致。

## 3. 本地验证

将QCKPT替换为本次训练按验证AP选出的实际文件：

```bash
QCKPT=work_dirs/publication_alpha/a1_learned_s0/best_official_tb_bbox_mAP_epoch_7.pth

python tools/test.py configs/publication_alpha/a1_learned.py "$QCKPT" \
  --eval bbox --work-dir work_dirs/main_eval_tb \
  --cfg-options model.inference_mode=tb model.diagnostics=False

python tools/test.py configs/publication_alpha/a1_learned.py "$QCKPT" \
  --eval bbox --work-dir work_dirs/main_eval_subtype \
  --cfg-options model.inference_mode=subtype model.diagnostics=False
```

使用主配置自带的验证索引，不改成后续自划分test。
整体TB采用COCO IoU 0.50:0.05:0.95、类别无关匹配；不是两个亚型AP的平均值。
分类别模式保留双亚型假设并逐类抑制，与整体TB输出上的argmax亚型结果分开报告。
`tools/evaluate.py`绑定另一套split-contract流程，不作为上述原599/200主实验的入口。

## 4. 官方ALL测试的独立C20筛查器

官方ALL输出需要真实的Healthy / Sick non-TB / TB整图分类概率。
质量网络不是整图分类器。C20训练的是独立编码器副本，不改动检测器及质量网络权重。

```bash
M2=work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth

python tools/train_classification.py \
  configs/classification/c20_original_sym_both.py \
  --backbone-checkpoint "$M2" \
  --seed 0 --deterministic --work-dir work_dirs/c20_main_s0

python tools/export_classification.py \
  work_dirs/c20_main_s0/best_cls_accuracy.pth \
  --backbone-checkpoint "$M2" \
  --out work_dirs/main_submission/TQSymNet.txt
```

C20训练12轮，微批次2、累积4次，有效批次8；按验证准确率选模。
导出会恢复该检查点的完整分类编码器，并生成概率TXT及来源manifest；
必须保留它们，不能手动打乱图像顺序。

## 5. 导出最终检测结果

```bash
python tools/export_online.py \
  configs/publication_alpha/test_online_learned.py "$QCKPT" \
  --classification-txt work_dirs/main_submission/TQSymNet.txt \
  --cls-filter \
  --json-prefix work_dirs/main_submission/TQSymNet
```

官方测试索引需包含全部3,302张图像，且不含隐藏GT。
仅对C20预测为TB的图像导出检测。该命令只生成本地文件，不自动提交到平台。
保存检测JSON、分类TXT、相关manifest及评分返回记录。
最终模型只按验证集选择，不使用隐藏测试返回结果调参。

## 权重与公开提交记录

| 资产 | 公开状态 |
|---|---|
| 主实验M2检测器checkpoint | 待提供真实权重后发布 |
| 最终可学习融合质量模型checkpoint | 待提供真实权重后发布 |
| 独立C20分类器checkpoint | 待提供真实权重后发布 |
| 2026-10-19隐藏测试预测包 | [已公开：检测JSON和分类TXT](results/hidden_all_20261019/README.md) |
| 官方评分回执/平台导出记录 | 待补充可公开的原始返回文件 |

[下载预测包](results/hidden_all_20261019/TQSymNet_submission.zip)。
它包含63,802个预测框（643张导出图像）和3,302行分类概率，不包含测试GT或模型权重。
该包与归档导出的哈希一致，只证明文件身份；不能单独证明平台提交ID或独立复算隐藏分数。
权重公开后将在此补充下载链接、对应配置和SHA-256，目前不提供占位或无效下载链接。

## 复现边界

数据与预训练资源须遵守各自许可。验证集用于选型与分析，局部结果具有选型依赖；
图像级bootstrap不代表训练种子方差。仓库静态检查不等于完整CUDA训练验证。
现有`coco.py`与旧`STANDALONE_SOURCE_MANIFEST.json`哈希不同，
因此旧standalone检查会报告core drift；未通过重写哈希掩盖该差异。
新运行应核验实际评测与论文协议一致。模型、配置及实现来源见[PROVENANCE.json](PROVENANCE.json)。
