# C14-C21: C12 priors with the original Sym block and independent encoder tuning

Retains C12's semantic Query, separate LBP/Gabor attention and learned branch
fusion. Only replaces the simplified mirror-neighbourhood block with the bundled
original SymFormer call chain: GuidePositionalEncoding ->
DeformableDetrTransformerConv -> SymMultiScaleDeformableAttention + FFN.

The same 256-channel Sym block is applied independently at P3/P4/P5 as in the
original RetinaGuideAttHead, then each map is projected to C12's 128 dimensions.
One encoder layer, 8 heads, 4 learned sampling points, 1024 FFN and .1 dropout;
GuidePositionalEncoding has 128 features, normalize=True, offset=-.5, left=True.
No upper-zone prior or lung segmentation. The all-zero boolean masks are padding
masks required by the original operator, not lung masks. Original mmdet source
files are imported without editing them; their existing equivalent deterministic
position sampling helpers remain. There is no CPU deformable-attention fallback.

All new Sym parameters are freshly initialized. No subtype Sym weights loaded.
All eight groups initialize from the same original M2 backbone/neck, and the same
seed initializes the same classifier. No historical C12 checkpoint is loaded:
its simplified Sym architecture is incompatible. C0-C13 configs remain available
with their original architectures and checkpoint formats.

## Configs

| Config | Backbone | CrossScale neck | CLAHE |
|---|---|---|---|
| c14_original_sym_frozen | frozen | frozen | off |
| c15_original_sym_frozen_clahe | frozen | frozen | on |
| c16_original_sym_neck | frozen | trainable | off |
| c17_original_sym_neck_clahe | frozen | trainable | on |
| c18_original_sym_backbone | trainable | frozen | off |
| c19_original_sym_backbone_clahe | trainable | frozen | on |
| c20_original_sym_both | trainable | trainable | off |
| c21_original_sym_both_clahe | trainable | trainable | on |

C14/16/18/20 isolate the two encoder switches. Each odd-numbered config differs
from the preceding even-numbered one only in classification CLAHE preprocessing.
The fully frozen CLAHE group is included so that the CLAHE/fine-tuning interaction
can be compared within the SAME original-Sym architecture.

## Training and checkpoint behavior

- Independent classification copy only; no mutation of the bbox detector file.
- Frozen encoders: requires_grad=False and eval mode. When only backbone trains,
  the frozen neck stays in eval but transmits gradients to backbone inputs.
- Trainable encoders: train mode, included in optimizer. Base LR .001 for classifier,
  neck LR .0001, backbone LR .00001. These are controlled starting values, not
  a claim of optimal LR. All groups retain 12 epochs and steps (8,11).
- No AMP. Every group uses microbatch 2, accumulation 4, effective batch 8.
  6888 images -> 3444 microbatches -> 861 optimizer updates per epoch, matching
  the old batch-8 update budget. Weighted CE is normalized across all 8 images,
  not by averaging separately normalized microbatch losses. Dropout and floating
  reduction mean microbatching is not promised to match batch-8 bit for bit.
- Source all_train/all_val/all_test and source M2 paths remain inherited from C0.
- Save full backbone/neck/classifier for ALL new groups. VAL-best replay restores
  ALL components from that epoch, not the best head on the latest encoder.
- best_cls_accuracy.pth is primary; ties use Sensitivity then Specificity, then
  earlier epoch. best_cls_auc.pth is retained as secondary.
- Every epoch validates; every 2 epochs exports per-image logits/probabilities,
  FN/FP IDs and filenames, source statistics and sampled-epoch error history.
  Training finish also exports best-checkpoint branch dependency and attention.
- [CLS-GRAD] records classifier/backbone/neck and Sym/LBP/Gabor gradient norms. Frozen encoder
  states are checked for invariance. [CLS-ENCODER] records policy and current
  hashes; trainable encoder hashes are allowed to change.
- Changing encoder weights requires using the updated export_classification.py
  and export_online.py. Classification TXT export restores the tuned encoder;
  detection export audits against its ORIGINAL source hashes. It never substitutes
  the tuned classification backbone into the bbox detector.

## Migration files relative to server TQ_SymNet root

```
TQ_SymNet/classification.py
TQ_SymNet/classification_priors.py
TQ_SymNet/classification_sym.py
tools/train_classification.py
tools/export_classification.py
tools/export_online.py
tools/check_classification_runtime.py
tools/check_classification_priors.py
tools/check_classification_finetune.py
configs/classification/c14_original_sym_frozen.py
configs/classification/c15_original_sym_frozen_clahe.py
configs/classification/c16_original_sym_neck.py
configs/classification/c17_original_sym_neck_clahe.py
configs/classification/c18_original_sym_backbone.py
configs/classification/c19_original_sym_backbone_clahe.py
configs/classification/c20_original_sym_both.py
configs/classification/c21_original_sym_both_clahe.py
RUN_CLASSIFICATION_ORIGINAL_SYM.md
```

Keep previously migrated classification_heads.py, classification_diagnostics.py,
classification_consistency.py and C0-C13 config bases. Retain the standalone
mmdet/models/utils/sym_attention.py, transformer.py and positional_encoding.py,
and the installed mmcv-full CUDA extension. No new external package is needed.

## One-GPU sequential run

```bash
cd /home/data2/LiYingxian/Tuberculosis-main
conda activate ./conda_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /home/data2/LiYingxian/TQ_SymNet
python tools/check_classification_finetune.py || exit 1
python tools/check_classification_priors.py --require-clahe || exit 1

for GROUP in \
  c14_original_sym_frozen \
  c15_original_sym_frozen_clahe \
  c16_original_sym_neck \
  c17_original_sym_neck_clahe \
  c18_original_sym_backbone \
  c19_original_sym_backbone_clahe \
  c20_original_sym_both \
  c21_original_sym_both_clahe
do
  CFG="configs/classification/${GROUP}.py"
  python tools/check_classification_runtime.py "$CFG" || exit 1
  python tools/train_classification.py "$CFG" --seed 0 --deterministic \
    --work-dir "work_dirs/cls_original_sym_seed0/${GROUP}_s0" || exit 1
done
```

Choose fresh work directories. If memory is insufficient on the target GPU,
change ALL compared groups to batch_size=1 and accumulation_steps=8; do not
change effective batch size or optimizer-update count for a single group.

## Export selected classifier

Example path only; select the actual winner by val Accuracy before online test:

```bash
python tools/export_classification.py \
  work_dirs/cls_original_sym_seed0/c20_original_sym_both_s0/best_cls_accuracy.pth \
  --out work_dirs/online_original_sym/TQSymNet.txt
```

Use the updated export_online.py with your existing quality-detector config and
checkpoint plus --classification-txt and --cls-filter. Detection source weights
remain the same M2 ancestry. Do not train or select using hidden-test labels.

Local checks cover all four gradient policies, weighted accumulation equivalence
in a deterministic toy model, full encoder restore and config inheritance.
The server runtime check verifies original Sym's CUDA forward/backward and direct
position->transformer->projection parity. Full CUDA integration is NOT claimed
verified by CPU-only tests.
