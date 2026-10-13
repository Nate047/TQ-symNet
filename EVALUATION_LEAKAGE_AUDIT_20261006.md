# TQ-SymNet评测与泄露审查（2026-10-06）

## 结论和范围

审查当前独立项目源码、10月4–5日M/A组日志及本地TBX11K图片/JSON，未发现验证GT进入实际评分/NMS、Oracle替换正常AP或相同图像跨训练验证集合。
这不是服务器文件逐字核验，也不代表排除了患者级重叠、近重复影像或长期验证集选择偏差。
本次仅新增只读数据审查工具和记录，未修改模型、数据变换和评测算法。

## 数据证据

本地图片根目录为C:/Users/18313/Desktop/TBX11K_img。

| 项目 | train | val |
|---|---:|---:|
| 图像 | 599 | 200 |
| bbox | 902 | 309 |
| Active / Obsolete bbox | 724 / 178 | 248 / 61 |
| 集合内部重复文件名/图像ID | 0 / 0 | 0 / 0 |
| 集合内部相同解码RGB像素 | 0 | 0 |
| 缺失图像 | 0 | 0 |
| crowd / ignore标注 | 0 / 0 | 0 / 0 |

跨集合重名文件、同SHA256图像文件、同尺寸RGB解码像素均为0。JSON中的image_id分别从1开始属于独立ID空间，不能把跨JSON的数字ID重合当作图像泄露。

JSON原始文件SHA256：
- train：9d1b19eb8985e281c4767148fac466445844b764db3dab2f7ca635503470971c
- val：bdec993506af15bf72010b32ec82ac279615ccc4c51ad9ca9438a7b9638d6e9d

训练ConcatDataset两个来源均为train_only_tb.json；纯Obsolete重复三次只在训练集选取。val/test均为val_only_tb.json。当前tools/evaluate.py默认是在开发验证集评估，不能称为隐藏test结果。

## GT流向

- mmdet/datasets/custom.py的prepare_test_img只传img_info，不传ann_info。
- 验证pipeline没有LoadAnnotations，Collect只返回img、original_img及尺寸/翻转等元信息。原图为直接像素复制，不含GT绘制。
- mmdet/apis/test.py先model.eval()，在torch.no_grad()内调用return_loss=False。
- 质量网simple_test用主检测器生成的boxes和sym分、图像特征计算质量分，随后融合与Soft-NMS，完成返回。没有GT参数参与筛选或排名。
- 原图与候选坐标通过scale_factor对齐，实际输出只反缩放一次，COCO导出再由xyxy转换为xywh。
- Oracle只在预测完成后，从quality_cache另取框并读取dataset.get_ann_info；正常EvalHook先评测实际结果得到key_score，再执行Oracle。Oracle的局部result没有写回runner.log_buffer或key_score。
- 调用COCO.evaluate本身不更新网络参数、梯度、优化器或融合门控。主网络冻结日志与哈希一致，门控固定0.5。

## 评测口径

新旧项目mmdet/datasets/coco.py的SHA256相同：
dbb041f0c28ca6546d1d3bc99c4b273903475b33f492886076662b413a245ee1

整体TB指标为COCO bbox、IoU=.50:.05:.95、catIds=[1,2,3]、useCats=0、maxDets=[1,10,100]，200张val图全部进入评测。类别3在当前JSON没有标注；类别1/2包含全部309框。输出类别来自模型预测，GT类别不参与预测分配。
实际使用pycocotools.COCOeval；没有删除低IoU GT、按预测成功筛图或把AP50当AP。
亚型bbox_mAP采用useCats=1，与official_tb_bbox_mAP分开。当前A0的0.400不是双亚型平均AP。

## 不能忽略的比较差异

- M2已经使用类别无关Hard-NMS；不能与旧head的分类别NMS结果直接做单因素归因。
- A0使用质量融合加Soft-NMS。B1缺失时，M2 0.380→A0 0.400包含评分和抑制两部分变化。
- 历史D3和本次新M2不是相同主网络checkpoint，初始化和训练历史不同。
- 反复用同一200张val挑epoch、模块和超参数会产生选择偏差；独立最终测试才能评估未参与开发的泛化性能。
- 本地精确图片内容检查不能排除同患者不同影像或近重复；需要患者标识或另做影像相似性核查。

## 服务器复核

迁移tools/audit_split_content.py后，在服务器执行：

```bash
cd /home/data2/LiYingxian/TQ_SymNet
python tools/audit_split_content.py \
  --train-json /home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_train_only_tb.json \
  --val-json /home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_val_only_tb.json \
  --image-root /home/data2/LiYingxian/Dataset_TB/TBX11K/imgs
```

用同一A0 checkpoint分别开启/关闭诊断评测，核对metrics.json中实际AP以及predictions.bbox.json。两次运行时不能更换cfg或checkpoint；这可以进一步验证服务器上的诊断路径不影响预测。
