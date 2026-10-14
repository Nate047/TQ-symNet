# C3 classification priors: C7-C13

This is classification-only research, not a change to bbox training, selection,
NMS or evaluation. All groups retain the C3 frozen M2 backbone/CrossScale source,
6888 training images, 2088 validation images, 12 epochs, seed 0, CE weights
(1,1,4), SGD and the original schedule. No C4 temperature or C5 consistency.
The new classifier starts from scratch, not a trained C3 or subtype Sym head.

## Groups

| Config | Fresh symmetry | LBP59 | Gabor energy | CLAHE |
|---|---|---|---|---|
| c3_crossscale_query_adaptive | no | no | no | no |
| c7_query_fresh_sym | yes | no | no | no |
| c8_query_clahe | no | no | no | yes |
| c9_query_lbp | no | yes | no | no |
| c10_query_gabor | no | no | yes | no |
| c11_query_texture | no | yes | yes | no |
| c12_query_sym_texture | yes | yes | yes | no |
| c13_query_sym_texture_clahe | yes | yes | yes | yes |

C7/9/10 versus C3 isolate each new prior. C11 tests texture combination;
C12 versus C11 tests symmetry with texture. C13 versus C12 tests CLAHE on the
combined model; C8 versus C3 tests CLAHE alone. Run C7-C10 first if budget limited.

## Exact implementation and boundaries

- C3 continues reading P3/P4/P5, with learned class-specific scale weights.
- Fresh Sym is **our mirror-neighbourhood attention adaptation**, not a copy of
  SymFormer's deformable-attention CUDA implementation. Each of P3/P4/P5 has its
  own projection and Q/K/V. Each position reads a 3x3 neighbourhood around the
  horizontally mirrored position. Original, attended, signed difference and
  absolute difference features form a learned relation representation. The
  three visual class queries read each level, with learned scale fusion. No
  left/right equality constraint, segmentation masks, bbox labels, old Sym weights
  or additional Sym auxiliary loss. Three-class CE trains all new branches.
- LBP computes a categorical uniform 8-neighbour code at input image resolution:
  58 uniform codes plus a nonuniform bin. Sparse per-bin occupancy is pooled to
  32x32, multiplied by learned 59-bin sigmoid gates and embeddings. This is
  equivalent to averaging gated embeddings, NOT interpolating ordinal code IDs.
- Gabor uses 8 learnable directions, wavelengths in [3,12] pixels, widths in
  [1.5,5], kernel 15x15. Each has zero-mean, unit-norm cosine/sine kernels. Energy
  sqrt(cos_response^2+sin_response^2+1e-8) is pooled before decimation and projected
  to 128 channels. No separately pretrained CNN. Texture branches invert the
  classifier's actual ImageNet normalization before extracting grayscale priors.
- C3 visual queries independently attend to LBP and Gabor spatial maps with
  positional encoding. An image/class-dependent gate fuses semantic and enabled
  prior responses before the existing three-class readout. Initial semantic share
  is 0.8, all other branches together 0.2; these are starting values, not limits.
  Optional initialization preserves shared C3 parameters and global RNG; enabled
  branches are NOT promised to be functionally identical to C3 at initialization.
- CLAHE: clipLimit=2, tileGridSize=(8,8), deterministic grayscale preprocessing
  BEFORE normalization. Same operation in train, val and test export. Only the
  classification dataset changes: detection always retains its own original
  input preprocessing. No noise, bilateral filter or additional geometric aug.
  This isolates CLAHE; it is not a full reproduction of the RegNetY-ViT pipeline.

## Evidence, not claims of guaranteed benefit

- Gabor magnitude representation: IJCV 2026, *A Lightweight Hybrid Gabor Deep
  Learning Approach and its Application to Medical Image Classification*,
  https://doi.org/10.1007/s11263-025-02658-2 . Our learnable parameterization and
  disease-query fusion differ from that paper; its CXR data are not TBX11K.
- CLAHE starting parameters: Frontiers in Medicine 2026,
  https://doi.org/10.3389/fmed.2026.1908751 . Different five-class/custom-split
  protocol; neither its accuracy nor its full preprocessing effect is ours.
- FDAM (ICCV 2025) and FDConv (CVPR 2025) motivate checking frequency preservation
  and filter diversity. Neither is implemented or claimed as TBX11K evidence.
- LBP59 learnable gates are our ablation hypothesis, not claimed as a recent
  top-tier validated TBX11K classification method.

## Migration files (relative to server TQ_SymNet root)

1. TQ_SymNet/classification.py
2. TQ_SymNet/classification_heads.py
3. TQ_SymNet/classification_priors.py (new)
4. tools/train_classification.py
5. tools/export_classification.py
6. tools/check_classification_priors.py (new)
7. configs/classification/c7_query_fresh_sym.py
8. configs/classification/c8_query_clahe.py
9. configs/classification/c9_query_lbp.py
10. configs/classification/c10_query_gabor.py
11. configs/classification/c11_query_texture.py
12. configs/classification/c12_query_sym_texture.py
13. configs/classification/c13_query_sym_texture_clahe.py
14. RUN_CLASSIFICATION_PRIORS.md

Keep existing C0-C6 config bases, classification_consistency.py, runtime/head
checkers and all detection files. No dataset annotation or M2 checkpoint change.

## Single-GPU sequential run

```bash
cd /home/data2/LiYingxian/TQ_SymNet
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python tools/check_classification_priors.py --require-clahe || exit 1
python tools/check_classification_heads.py || exit 1

for GROUP in \
  c3_crossscale_query_adaptive \
  c7_query_fresh_sym \
  c8_query_clahe \
  c9_query_lbp \
  c10_query_gabor \
  c11_query_texture \
  c12_query_sym_texture \
  c13_query_sym_texture_clahe
do
  CFG="configs/classification/${GROUP}.py"
  python tools/check_classification_runtime.py "$CFG" || exit 1
  python tools/train_classification.py "$CFG" --seed 0 --deterministic \
    --work-dir "work_dirs/cls_priors_seed0/${GROUP}_s0" || exit 1
done
```

C3 can be omitted if reusing the completed matching run; rerunning produces the
new per-image/subset diagnostics. Choose a fresh work-dir; no automatic resume
or overwrite. AMP is not enabled. Preserve one optimizer update per image batch.

## Logs and outputs

- [CLS-FROZEN]: backbone/neck hashes and frozen state checked each epoch.
- [CLS-PRIORS]: all enabled flags, branch order, CLAHE configuration.
- [CLS-READOUT]: class-specific scale/branch gates, attention entropy, response
  norms, all LBP bin weights, Gabor directions and wavelengths. Gate magnitudes
  alone do not prove branch usefulness; use matched ablations and per-image errors.
- [CLS-VAL]: Accuracy, AUC, Sensitivity, Specificity, confusion matrix, loss.
- [CLS-VAL-TBX11K]: same metrics restricted to original TBX11K val; avoids mixing
  external MC/Shenzhen and DA/DB improvements with benchmark improvements.
- val_predictions_epoch_XX.npz: image IDs, filenames, labels and three-class
  probabilities in validation index order; supports FN/FP and complementarity.
- latest.pth, best_cls_auc.pth (existing primary policy), and additionally
  best_cls_accuracy.pth. Do not choose a checkpoint based on online-test results.

The classification checkpoint stores all new modules and input config. Existing
export_classification.py reconstructs them strictly and applies the recorded
CLAHE automatically. The following is an example only, not a selected winner:

```bash
python tools/export_classification.py \
  work_dirs/cls_priors_seed0/c13_query_sym_texture_clahe_s0/best_cls_accuracy.pth \
  --out work_dirs/online_test_priors/TQSymNet.txt
```

Local tests check module gradients, frozen encoder contracts, optional RNG,
config inheritance, categorical LBP, Gabor normalization and strict state reload.
Actual MMCV/CUDA checkpoint integration must pass the server runtime check.
