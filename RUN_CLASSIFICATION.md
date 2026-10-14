# 官方Sym整图分类器：冻结TQ-SymNet主干

分类器结构与Sym公开实现一致，仅将首层输入通道设为TransNeXt的576。
Backbone保持eval/no_grad/参数冻结，只训练分类器。SGD lr=.001、momentum=.9、weight decay=.0001，12epoch，batch8，warmup500，LR step8/11；CE权重Healthy/Sick/TB=1/1/4。

分类训练使用官方数据包的all_train.json（6888张，含288张extra），验证使用all_val.json（2088张，含288张extra）。不合并all_trainval训练，保留val挑选最佳模型；这与官方最终trainval合并训练不同，不能称为完全相同训练条件。
数据集保留无bbox图像。TBX11K标签按health/sick/tb目录生成；MC/Shenzhen按CHNCXR/MCUCXR文件名末尾_0/_1映射Healthy/TB；DA/DB按n/p前缀映射Healthy/TB。外部负例不标为Sick。未知路径明确报错，绝不默认猜标签。
启动时检查train/val/test文件名不重叠，记录各来源的类别数量、标注哈希到日志和data_audit.json。仅用test索引做重叠检查，不使用test预测或标签选模型。
训练启动前拒绝缺失/零字节图片，绝不跳过样本改变划分。若检查失败，修复官方图片文件后再训练，不删除相应JSON记录。
每轮验证报告Accuracy、TB AUC、Sensitivity、Specificity、macro AP/Recall和混淆矩阵；按验证TB AUC保存best_cls_auc.pth。每轮记录backbone哈希并断言没有梯度和权重/持久buffer变化。

默认加载此前已确认的M2第13轮checkpoint，它与冻结A0的主干相同。若改用A0完整checkpoint，可用--backbone-checkpoint指定，仅加载backbone.*，严格检查主干参数及重建的位置buffer。

## 迁移

本次在服务器TQ_SymNet根目录覆盖/新增：
- TQ_SymNet/classification.py
- TQ_SymNet/classification_heads.py
- TQ_SymNet/classification_consistency.py
- configs/classification/c0_backbone_sym.py
- configs/classification/c1_crossscale_hybrid_pool.py
- configs/classification/c2_crossscale_query_fixed.py
- configs/classification/c3_crossscale_query_adaptive.py
- configs/classification/c4_crossscale_adaptive_temperature.py
- configs/classification/c5_crossscale_intensity_consistency.py
- configs/classification/c6_crossscale_temperature_consistency.py
- tools/train_classification.py
- tools/export_classification.py
- tools/check_classification_data.py
- tools/check_classification_heads.py
- tools/check_classification_runtime.py
- RUN_CLASSIFICATION.md

数据目录需已有annotations/json/all_train.json、all_val.json、all_test.json，以及imgs/extra/da+db和imgs/extra/mc+shenzhen（沿用官方划分，不生成新JSON）。已有导出工具不改。

## 训练

```bash
cd /home/data2/LiYingxian/TQ_SymNet
python tools/check_classification_data.py \
  --json-dir /home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json \
  --image-root /home/data2/LiYingxian/Dataset_TB/TBX11K/imgs
CUDA_VISIBLE_DEVICES=0 python tools/train_classification.py \
  configs/classification/c0_backbone_sym.py \
  --seed 0 --deterministic --work-dir work_dirs/cls_backbone_sym_extra_s0
```

不改主检测器、质量网络或检测checkpoint文件。新增分类权重单独保存；不会在已有latest.pth的work-dir覆盖训练。

## 冻结CrossScale分类头消融

四组共享同一M2检测checkpoint、all_train/all_val数据、12轮SGD和按val TB AUC选模。
c0为原Sym头，只读取backbone末层；c1冻结并读取完整CrossScale P3/P4/P5，
比较平均与最大池化；c2使用三类别query分层读取、固定1/3尺度融合；
c3仅将c2的融合改为每图每类别可学习权重，其融合器初始等于c2。
分类投影采用逐空间位置的通道LayerNorm。C3尺度编码显式进入尺度门控，
不加到整层所有Key上；C2/C3共享参数初值和构造后的全局RNG状态相同。
本修复改变C2/C3分类头参数结构，旧分类头checkpoint不兼容，需用新work-dir从头训练；
冻结的M2检测checkpoint继续沿用，C0/C1不受影响。
类别query由整图三分类标签训练，不使用GT框。三个CrossScale头还融合P5全局特征。
日志中[CLS-READOUT]汇总约九个val批次的类别尺度权重和注意力熵；
注意力不解释为病灶概率。检测backbone与neck均以eval/no_grad运行，
每轮核对哈希和梯度；导出时重新校验两部分权重与训练时一致。

```bash
cd /home/data2/LiYingxian/TQ_SymNet
python tools/check_classification_heads.py
python tools/check_classification_runtime.py configs/classification/c1_crossscale_hybrid_pool.py
python tools/check_classification_runtime.py configs/classification/c2_crossscale_query_fixed.py
python tools/check_classification_runtime.py configs/classification/c3_crossscale_query_adaptive.py
for GROUP in c0_backbone_sym c1_crossscale_hybrid_pool c2_crossscale_query_fixed c3_crossscale_query_adaptive; do
  CUDA_VISIBLE_DEVICES=0 python tools/train_classification.py \
    "configs/classification/${GROUP}.py" --seed 0 --deterministic \
    --work-dir "work_dirs/cls_${GROUP}_s0" || exit 1
done
```

比较best_metrics.json中的Accuracy、TB_AUC、Sensitivity、Specificity和混淆矩阵。
胜出组使用各自目录的best_cls_auc.pth通过同一个export_classification.py导出3302行test概率。
该消融验证的是分类读出层，不能把多尺度query的收益归因于上游检测器。

## 下一轮：注意力温度与成像一致性

沿用C3的全冻结架构、同一M2检测权重、all_train/all_val、12轮SGD与val TB AUC选模。
C4只增加每图/每类别的空间注意力温度，初始温度严格为1，
和C3共享参数及初始化后全局RNG状态；温度作用在QK注意力分数上，
不作用在最终三分类Softmax。C5只增加train图像的轻度亮度(±5像素)
与对比度(±5%)扰动，采用原图预测stop-gradient的KL约束，权重0.1。
val与test完全使用原始单视图。C6同时增加两者，判断能否叠加。
这些数值是首轮消融起点，不声称已调到最优；所有组从同一个冻结M2权重
重新初始化分类头，不加载C3已训练权重。
日志同时记录原始CE、未乘权的KL、总体loss、每类温度、尺度权重和注意力熵。

```bash
cd /home/data2/LiYingxian/TQ_SymNet
python tools/check_classification_heads.py
for GROUP in c4_crossscale_adaptive_temperature c5_crossscale_intensity_consistency c6_crossscale_temperature_consistency; do
  CFG="configs/classification/${GROUP}.py"
  CUDA_VISIBLE_DEVICES=0 python tools/check_classification_runtime.py "$CFG" || exit 1
  CUDA_VISIBLE_DEVICES=0 python tools/train_classification.py "$CFG" \
    --seed 0 --deterministic \
    --work-dir "work_dirs/cls_temp_consistency_seed0/${GROUP}_s0" || exit 1
done
```

比较C3及C4/C5/C6的best_metrics.json和epoch9–12均值；不要用test挑选设置。
分类头权重结构在C4/C6增加了温度分支，须严格匹配本组config导出。

## 隐藏test导出及打包

TXT为3302行Healthy/Sick/TB概率，按all_test.json的images顺序。JSON是COCO检测实例列表，使用原image_id，不生成每图空记录或重编号。
以下启用Sym官方的整图argmax==TB过滤：只对判为TB的图像运行TQ检测，其余图在JSON中没有框；框坐标/融合分/Soft-NMS与A0相同。TXT和JSON导出会核对测试索引、文件哈希和同一冻结主干。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_classification.py \
  work_dirs/cls_backbone_sym_extra_s0/best_cls_auc.pth \
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
