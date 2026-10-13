# TQ-SymNet论文可视化

只增加图形与诊断脚本，不修改模型、loss、训练cfg或NMS。
迁移整个 `tools/paper_visualization/` 到服务器独立项目的同名目录。
以下服务器命令均在 `/home/data2/LiYingxian/TQ_SymNet` 执行。

## 1. 框架图（本地或服务器均可）

```bash
python tools/paper_visualization/draw_framework.py --out-dir paper_figures
```

生成 `framework` 和 `quality_network`，各提供PDF、SVG、300dpi PNG。
默认按接近IEEE双栏宽度绘图；SVG文字保持可编辑。
总体图：TransNeXt → CrossScale → Sym分类/回归 → 候选质量 → 固定0.5 logit融合 → Gaussian Soft-NMS。
细节图：几何token、P2-P5分层读取、LBP/Gabor独立注意力、原图四边细节、质量目标及rank。
没有pTB、H2、ResNet或GT依赖推理；图中的原图编码是每图共享，不是每框裁剪backbone。

## 2. 用真实checkpoint导出基线与完整模型

M2权重路径已由日志确认。A0路径采用该轮日志中的 `work_dirs/a0_s0`，最佳轮为7；
如果服务器上实际目录不同，只修改下面A0命令的checkpoint路径。
这是测试/诊断，不需要重新训练，不消耗online提交额度。

```bash
cd /home/data2/LiYingxian/TQ_SymNet

CUDA_VISIBLE_DEVICES=0 python tools/paper_visualization/export_predictions.py \
  configs/b0_sym_hard.py \
  work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth \
  --tag B0 --out-dir paper_visualization/b0 --candidates

CUDA_VISIBLE_DEVICES=0 python tools/paper_visualization/export_predictions.py \
  configs/b1_sym_soft.py \
  work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth \
  --tag B1 --out-dir paper_visualization/b1 --candidates

CUDA_VISIBLE_DEVICES=0 python tools/paper_visualization/export_predictions.py \
  configs/a0_full.py \
  work_dirs/a0_s0/best_official_tb_bbox_mAP_epoch_7.pth \
  --tag A0 --out-dir paper_visualization/a0 --candidates
```

- 使用各自cfg的val pipeline及实际simple_test，不重实现NMS；GT在forward之后才读取。
- 原图XYXY，连续面积公式，没有额外+1，不暗改候选数量、阈值或框坐标。
- 禁止翻转/多尺度TTA，避免缓存坐标与绘图坐标产生不一致。
- 每个模型单独运行，不同时把两个模型放入显存；按图清理诊断缓存。
- 如数据目录不同，加 `--ann-file /绝对路径/val.json --img-root /绝对路径/imgs`。
- `--max-images 2`可先冒烟。部分队列不输出AP，不可混入整套论文统计。
- 完整导出保存 `visualization.json.gz`、实际 `predictions.bbox.json`、metrics、配置和权重哈希。
- 预期当前checkpoint验证AP约为B0=.380、B1=.387、A0=.400；若明显不符，先核对配置/权重/数据，勿把图当成原模型结果。

## 3. 一次生成检测图、候选组成和分数图

```bash
python tools/paper_visualization/plot_predictions.py \
  --runs paper_visualization/b0/visualization.json.gz \
         paper_visualization/b1/visualization.json.gz \
         paper_visualization/a0/visualization.json.gz \
  --out-dir paper_figures --cases 4
```

生成：

| 文件 | 用途 |
|---|---|
| `qualitative_detections.png/svg` | 同一原图的GT、B0、B1、A0对比；包括增益、少数亚型、失败病例 |
| `candidate_quality_distribution.png/svg` | 候选/输出IoU组成、独立GT覆盖、一对一TP/FP |
| `score_vs_true_iou.png/svg` | Sym/质量/融合logit与真实IoU的关系 |
| `sym_quality_joint_distribution.png/svg` | Sym和质量logit联合分布，按真实IoU分层 |
| `candidate_scores.csv` | 每个候选的原图框、亚型概率、质量logit、质量代理概率、融合/最终分数、真实IoU、选中状态 |
| `per_gt_ranking.json` | 每个GT可用最佳IoU及三种得分在IoU≥.5候选内选中的框IoU |
| `figure_audit.json` / `case_manifest.json` | 数据来源、匹配统计、绘图病例及选择理由 |

默认检测图只显示top5且score≥.05，减少100个重叠框造成的遮挡；这是**显示过滤**，AP和分布统计仍用全部原输出。
可改 `--display-topk 10 --display-min-score 0.01`，或者指定 `--image-ids 12,37,68,105`。
论文中的真实图使用 `--image-ids 88,1,150 --paper-style --display-topk 3 --display-min-score .05`。
paper-style在框内标记分数排名，预测亚型与真实IoU移至图像下方，减少文字遮挡；框和NMS结果不变。
自动选择是事后诊断例图，不是随机临床样本；选图理由会记录，正文不能以这些案例估计总体性能。
统计图不需要原图，使用 `--no-images` 即可。

本地绘图只需numpy/matplotlib/Pillow，不需Torch/MMDetection。拷回三个导出目录后：

```powershell
& 'C:/Users/18313/anaconda3/python.exe' 'C:/Users/18313/Desktop/Path_to_Free/TQ_SymNet/tools/paper_visualization/plot_predictions.py' --runs 'C:/Users/18313/Desktop/paper_visualization/b0/visualization.json.gz' 'C:/Users/18313/Desktop/paper_visualization/b1/visualization.json.gz' 'C:/Users/18313/Desktop/paper_visualization/a0/visualization.json.gz' --img-root 'C:/Users/18313/Desktop/TBX11K_img' --out-dir 'C:/Users/18313/Desktop/TIP_Latex/figures'
```

本地原图目录应能直接拼接JSON的file_name，如 `TBX11K_img/tb/xxx.png`；不按basename模糊匹配。

## 4. Precision-recall及各IoU阈值AP曲线（可选）

```bash
python tools/paper_visualization/plot_coco_pr.py \
  --ann-file /home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json \
  --predictions paper_visualization/b0/predictions.bbox.json \
                paper_visualization/b1/predictions.bbox.json \
                paper_visualization/a0/predictions.bbox.json \
  --labels B0 B1 A0 --out-dir paper_figures
```

使用完整原始预测和COCOeval：useCats=0、catIds=[1,2,3]、maxDets=[1,10,100]，不加NMS或显示阈值。
输出 `precision_recall.png/svg` 和 `coco_curve_audit.json`，应与相应验证日志的数值一致（舍入差异除外）。
已有逐epoch AP/覆盖图的生成脚本位于论文目录 `scripts/plot_results_20261006.py`，不必重新推理。

## 质量网络改善选择结果的专用机制图（建议正文保留一张）

公平对比优先B1对A0：两者使用同一检测器、候选、Gaussian Soft-NMS和输出预算。
B0对A0同时改变NMS，不适合把全部收益归因于质量网络。

```bash
python tools/paper_visualization/plot_quality_effect.py \
  --baseline paper_visualization/b1/visualization.json.gz \
  --quality paper_visualization/a0/visualization.json.gz \
  --out-dir paper_figures
```

- `quality_selection_effect.png/svg`：同一GT首次分数匹配的IoU组成、最终GT覆盖、配对改善/恶化。
- `final_error_composition.png/svg`：TP、重复FP、定位不足FP及IoU<.10 FP组成。
- `paired_gt_selection.csv`、`quality_effect_audit.json`：逐GT对应、改善/恶化/新增/漏失数量及可核验来源。
- 不使用原图，不增加推理，不需要CUDA；直接复用导出文件。
- “改善”是排序选出了更好的框，**不是质量网络修改了同个框的坐标**。
- 首次匹配是按最终分数、一对一IoU.50匹配；0表示未匹配，不表示真实IoU为0。
- GT覆盖是至少一个最终框命中，不能与首次匹配或AP混为一谈。匹配图不替代COCOeval。
- 图中包含全部GT及全部原输出，不只统计成功框，不丢弃未匹配的GT。
- 当前导出器增加NMS设置记录；旧导出若缺该记录会注明suppression_verified=False。

### 正文与补充材料的取舍

建议正文保留一个紧凑的机制图（GT覆盖＋配对选择改善），配合AP/AP75对照表；
候选分布、错误组成或联合分数散点按论证需要选一张，其余放补充材料，不必全部塞进正文。
没有“每一种诊断图都必须提交”的要求。候选纯度提高本身不能证明AP提升：
重复框、漏检、分数先后仍有影响，需要最终检测指标和同GT结果一起说明。

## 5. 定义和边界

- 某候选IoU≥.5只说明它与GT重叠，不说明它是检测TP；同一GT的第二个重复检测仍是FP。
- 颜色：青色GT，绿色一对一IoU.50 TP，黄色重复FP，红色其他FP；A/O为预测亚型。
- 匹配诊断为类别无关、按实际输出分数排序；不是COCO AP替代实现。若有crowd/ignore GT，绘图匹配会拒绝，不静默处理。
- 质量概率是全阈值目标代理，不能标成“预测真实IoU”或“TB概率”。
- 散点最多显示20000点（固定抽样种子）；完整统计/CSV不抽样。
- 候选相关性有重复样本依赖，不能当成患者独立统计证据。每GT的排序数据另行输出。
- 脚本自动检查同一划分/GT/检测器哈希，防止拼接不匹配的基线和质量模型。
- 没有生成伪造检测图或把GT最高IoU框替换成模型输出。导出数据可用于检查论文结果。

### 隐藏test没有GT时：只画实际预测

先用原有测试/提交导出流程得到COCO格式的实际bbox结果，不额外提交online。
用下面代码绘图；它只读取image元数据，不读取GT annotations，不标TP/FP或真实IoU：

```bash
python tools/paper_visualization/plot_prediction_only.py \
  --images-json /实际路径/test_images.json \
  --predictions /实际路径/predictions.bbox.json \
  --img-root /实际路径/imgs --out-dir paper_figures/test_predictions
```

输入bbox必须是COCO的原图XYWH，不是诊断缓存中的XYXY；脚本不自动猜测坐标格式。

## 6. 填入LaTeX

框架原占位的figure*内部可换成：

```latex
\centering
\includegraphics[width=\textwidth]{framework.pdf}
\caption{Overview of TQ-SymNet. Detector parameters are frozen during quality training.}
```

质量网络细节可作为方法中的第二幅figure*：

```latex
\centering
\includegraphics[width=\textwidth]{quality_network.pdf}
\caption{Candidate-conditioned visual, texture, and side-detail quality evaluation.}
```

真实检测图生成后用 `qualitative_detections.png` 替换定性占位；整页多列图建议figure*，不要缩成单栏导致文字不可读。
本次仅提供脚本/图形资产，未改动已交付论文正文或模型代码。

## 验证状态

`python tools/paper_visualization/test_visualization.py`：8项CPU检查（空框、连续IoU、重复FP、tie rank、来源匹配、临时绘图流程、配对GT实际IoU、同NMS设置验证）。
导出器通过语法检查，但本机没有服务器MMCV/CUDA模型环境；完整checkpoint推理应先在服务器用2图冒烟，再全量运行。
