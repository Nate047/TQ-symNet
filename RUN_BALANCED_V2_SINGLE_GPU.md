# Balanced-v2：单卡顺序消融

> 当前新增的质量消融采用M2-20e最佳epoch17，运行命令见
> [RUN_QUALITY_M2_20E_SINGLE_GPU.md](RUN_QUALITY_M2_20E_SINGLE_GPU.md)。
> 本页根目录A配置仍对应之前的16epoch底座，不要与新一轮混用。

保持479/120/200，固定原test名单，只重分train/val。主网络16epoch、质量网络8epoch，
与上一轮相比仅改变划分及输出目录，未同时改LR/轮数。总更新量是否不足是另一个控制实验。
有效训练图像仍为479 + 83×3 = 728，每epoch364次更新（batch2）。

## 文件迁移

从本地TQ_SymNet根目录复制到服务器同名项目根目录，保持相对路径：

```text
data/splits/tbx11k_479_120_200_balanced_v2_seed0/  整个目录
configs/_common.py
configs/_matched_quality.py
configs/train_detector.py
configs/test_val.py
tools/check_structure.py
tools/select_detector_checkpoint.py
tools/generate_balanced_tb_split.py
TQ_SymNet/pretrained.py
TQ_SymNet/models/p2t_checked.py
TQ_SymNet/models/__init__.py
configs/m0_p2t_fpn.py
configs/m3_p2t_crossscale.py
configs/m2_transnext_crossscale_20e.py
tools/preflight.py
RUN_BALANCED_V2_SINGLE_GPU.md
```

已有的data_contract.py、models/initialization.py、preflight.py等v1迁移文件必须仍在。
六份标注和三份图像列表、manifest.json都在新目录的seed_0下。
不删除旧文件/权重，也不将本次输出写入旧work-dir。旧ZIP包不包含本轮更新。

## 单卡完整命令（Bash，不需要.sh文件）

以下工作目录须尚未用于另一轮实验。若M2已完成本次v2训练，跳过M2训练块，
运行select_detector_checkpoint.py后再跑未完成的组，不能重复混跑。

若训练时用了其他--work-dir，指定实际位置：

```bash
python tools/select_detector_checkpoint.py --work-dir /实际/M2训练目录
# 或指定该目录下VAL选出的best文件：
python tools/select_detector_checkpoint.py --checkpoint /实际路径/best_official_tb_bbox_mAP_epoch_N.pth
```

脚本验证划分哈希后，将权重发布到当前质量cfg要求的固定selected_detector路径。
源目录和目标目录可以不同；不会跨目录自动挑选模型，也不会采用latest.pth。

```bash
cd /home/data2/LiYingxian/TQ_SymNet
PY=/home/data2/LiYingxian/Tuberculosis-main/conda_env/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0

"$PY" tools/check_structure.py || exit 1
"$PY" tools/check_standalone.py --runtime || exit 1
"$PY" tools/preflight.py --config configs/m2_transnext_crossscale.py --server-files --build || exit 1
"$PY" tools/train.py configs/m2_transnext_crossscale.py \
  --seed 0 --deterministic \
  --work-dir work_dirs/balanced_v2_seed0_479_120_200/m2_s0 || exit 1
"$PY" tools/select_detector_checkpoint.py || exit 1

for CFG in m0_p2t_fpn m1_transnext_fpn a0_full a1_no_texture a4_no_detail a7_no_rank a8_direct_iou a12_direct_iou_no_rank
do
  GROUP="${CFG%%_*}"
  "$PY" tools/preflight.py --config "configs/${CFG}.py" --server-files --build || exit 1
  "$PY" tools/train.py "configs/${CFG}.py" --seed 0 --deterministic \
    --work-dir "work_dirs/balanced_v2_seed0_479_120_200/${GROUP}_s0" || exit 1
done

for CFG in b0_sym_hard b1_sym_soft
do
  GROUP="${CFG%%_*}"
  "$PY" tools/evaluate.py "configs/${CFG}.py" \
    work_dirs/balanced_v2_seed0_479_120_200/m2_s0/selected_detector.pth \
    --split val --mode tb \
    --out-dir "work_dirs/balanced_v2_seed0_479_120_200/${GROUP}_val" || exit 1
done
```

本命令不测test。质量组共用本次M2且冻结，重新初始化质量网络。

## 独立20epoch预算对照

本组只改变M2训练轮数和LR里程碑，不自动替换16epoch消融的固定底座。
训练从ImageNet初始化开始，不从16epoch checkpoint续训。

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/m2_transnext_crossscale_20e.py \
  --seed 0 --deterministic \
  --work-dir work_dirs/balanced_v2_seed0_479_120_200/m2_20e_s0
```

M0/M3增加P2T严格预训练适配器，forward仍继承原P2T。启动应打印
`[TQ-PRETRAIN] P2T-Small loaded=.../... (100%)`及文件SHA。
`preflight --server-files --build`也验证完整detector初始化后P2T权重未被覆盖。
这能排除静默漏加载，不能仅凭第1epoch AP=0确认故障已解决。

## MIU/GLIP同步划分

复现目录中的prepare_recipes.py、check_preparation.py、audit_structural_runtime.py、
audit_sym_runtime.py及shared_data/balanced_v2_seed0同步迁移。服务器再次执行：

```bash
cd /home/data2/LiYingxian/TBX_reproduction
python prepare_recipes.py \
  --split-dir /home/data2/LiYingxian/TQ_SymNet/data/splits/tbx11k_479_120_200_balanced_v2_seed0/seed_0 \
  --image-root /home/data2/LiYingxian/Dataset_TB/TBX11K/imgs
python check_preparation.py
```

生成的MIU/GLIP输出在work_dirs/balanced_v2_seed0/下；旧recipe启动训练所得权重不混用。
