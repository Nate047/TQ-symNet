# 官方Sym整图分类器：冻结TQ-SymNet主干

分类器结构与Sym公开实现一致，仅将首层输入通道设为TransNeXt的576。
Backbone保持eval/no_grad/参数冻结，只训练分类器。SGD lr=.001、momentum=.9、weight decay=.0001，12epoch，batch8，warmup500，LR step8/11；CE权重Healthy/Sick/TB=1/1/4。

分类数据使用TBX11K_train.json（6600）和TBX11K_val.json（1800）；不使用带288张extra的all_train/all_val。数据集保留无bbox图像，标签由明确的health/sick/tb目录生成。
每轮验证报告Accuracy、TB AUC、Sensitivity、Specificity、macro AP/Recall和混淆矩阵；按验证TB AUC保存best_cls_auc.pth。每轮记录backbone哈希并断言没有梯度和权重/持久buffer变化。

默认加载此前已确认的M2第13轮checkpoint，它与冻结A0的主干相同。若改用A0完整checkpoint，可用--backbone-checkpoint指定，仅加载backbone.*，严格检查主干参数及重建的位置buffer。

## 迁移

在服务器TQ_SymNet根目录覆盖/新增：
- TQ_SymNet/classification.py
- configs/classification/c0_backbone_sym.py
- tools/train_classification.py
- tools/export_classification.py
- tools/export_online.py
- tools/package_online.py
- configs/test_online.py
- tools/check_structure.py

## 训练

```bash
cd /home/data2/LiYingxian/TQ_SymNet
CUDA_VISIBLE_DEVICES=0 python tools/train_classification.py \
  configs/classification/c0_backbone_sym.py \
  --seed 0 --deterministic --work-dir work_dirs/cls_backbone_sym_s0
```

不改主检测器、质量网络或检测checkpoint文件。新增分类权重单独保存；不会在已有latest.pth的work-dir覆盖训练。

## 隐藏test导出及打包

TXT为3302行Healthy/Sick/TB概率，按all_test.json的images顺序。JSON是COCO检测实例列表，使用原image_id，不生成每图空记录或重编号。
以下启用Sym官方的整图argmax==TB过滤：只对判为TB的图像运行TQ检测，其余图在JSON中没有框；框坐标/融合分/Soft-NMS与A0相同。TXT和JSON导出会核对测试索引、文件哈希和同一冻结主干。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_classification.py \
  work_dirs/cls_backbone_sym_s0/best_cls_auc.pth \
  --out work_dirs/online_test/TQSymNet.txt

CUDA_VISIBLE_DEVICES=0 python tools/export_online.py \
  configs/test_online.py \
  work_dirs/a0_s0/best_official_tb_bbox_mAP_epoch_7.pth \
  --mode tb --classification-txt work_dirs/online_test/TQSymNet.txt \
  --cls-filter --json-prefix work_dirs/online_test/TQSymNet

python tools/package_online.py \
  --txt work_dirs/online_test/TQSymNet.txt \
  --json work_dirs/online_test/TQSymNet.bbox.json \
  --out work_dirs/online_test/TQSymNet_submission.zip
```

生成ZIP根目录仅含TQSymNet.txt和TQSymNet.json，不自动上传。
分类旁证保存为TQSymNet.classification.manifest.json，检测旁证保存为TQSymNet.detection.manifest.json，防止相互覆盖。打包器兼容早期被覆盖的检测manifest并核对保留下来的完整索引与分类过滤一致性。
需要比较分类过滤影响时，保持同一检测checkpoint，另行不带--cls-filter导出到不同json-prefix；这仍处理全部3302张隐藏test图像，不能用GT筛选Only-TB。
尚未执行服务器CUDA全模型训练；本地验证覆盖分类器梯度、冻结行为、标签/顺序、概率指标、配置解析与提交打包。
