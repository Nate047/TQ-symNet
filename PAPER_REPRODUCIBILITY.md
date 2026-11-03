# TQ-SymNet paper experiment: 599/200 protocol

This guide describes the paper's original 599/200 cohort. Commands are single-GPU examples and
must be run from this directory in an already compatible MMDetection 2.x environment.
No lung segmentation mask is used. A score rounded to 40.1 AP is a single selected
validation checkpoint, not a guarantee that another training run will reproduce it.

## 1. Environment and required assets

- Historical environment: Linux, Python 3.8, PyTorch 1.10, CUDA 11.3,
  MMCV 1.7.1 with the compiled CUDA extensions, compatible torchvision and timm;
  the TransNeXt `swattention` extension must be built for the installed CUDA/PyTorch.
  Reuse the working environment rather than blindly installing newer versions.
- Obtain TBX11K images and annotations under their original licensing conditions.
  The paper's TB-positive detector train/val sets have 599/200 images and
  902/309 boxes respectively. Place the original train-only-TB and val-only-TB
  JSON files plus images at the paths in
  `configs/publication_alpha/_original_a0.py`, or edit those paths to match your system.
- Obtain the ImageNet-pretrained TransNeXt-Small weights for detector training,
  the original frozen M2 detector checkpoint (epoch 13), and the independent
  image-classifier checkpoint for the hidden-test export. These files, datasets,
  submission predictions and GPU-built binaries are **not included** in this repository.
- The original-cohort initialization hook checks the exact detector and annotation
  SHA-256 values in `_original_a0.py`. Moving files is permitted; changing their
  contents fails the check. Do not bypass hashes and report the run as the same
  experiment. Pure-Obsolete images are repeated three additional times by the
  config; the mixed-subtype images are not in that supplement.

The independent detector checkpoint is expected at
`work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth`, with SHA-256
`4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74`.
Train/val annotation hashes are recorded in the config. The working server paths
are examples, not downloadable resources. Before executing, locate the actual
files and adjust paths consistently in the config and hook arguments.

**Detector from scratch:** A separately verified original-cohort detector training
config and its corresponding assets are required to regenerate the historical
detector checkpoint. The quality-stage command below reproduces the paper's
specified training procedure only when that exact frozen detector and the exact
original annotations are available.

## 2. Inspect configuration and train quality scoring

```bash
cd /path/to/tb/TQ_SymNet
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python tools/train.py configs/publication_alpha/a1_learned.py \
  --seed 0 --deterministic \
  --work-dir work_dirs/publication_alpha/a1_learned_s0
```

The final paper model is the original quality reader at epoch 7, with learned
global alpha approximately 0.53571 and validation AP/AP50/AP75 40.1/77.5/35.7.
Each new training run must select its own checkpoint on validation AP; do not
assume the same epoch, replace the learned scalar with 0.5, or use the hidden test
for selection. `a0_fixed.py` is a distinct fixed-alpha control. Its 40.00 AP and
the offline alpha sweep's 40.05 AP are not the separately trained 40.1 model.

## 3. Evaluate on local validation

```bash
QCKPT=work_dirs/publication_alpha/a1_learned_s0/best_official_tb_bbox_mAP_epoch_7.pth
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/publication_alpha/a1_learned.py "$QCKPT" \
  --mode tb --out-dir work_dirs/publication_alpha/a1_learned_s0/eval_tb --diagnostics
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/publication_alpha/a1_learned.py "$QCKPT" \
  --mode subtype --out-dir work_dirs/publication_alpha/a1_learned_s0/eval_subtype
```

Change `QCKPT` to the actual selected file. Pooled TB AP uses COCO IoUs
0.50:0.05:0.95 and category-agnostic matching; it is not the mean of subtype APs.
The classwise subtype replay and neck ablation are local validation results.
The validation set also selects checkpoints, so those results are selection-conditioned.

## 4. Hidden ALL test export

The official online test is scored on ALL; no separate Only-TB result is available
for the current submission. The image-level Healthy/Sick/TB classifier is a
separate network. The quality scorer is **not** that classifier. Train/export the
independent C20 classifier using `RUN_CLASSIFICATION_ORIGINAL_SYM.md`, including
its matching image index and provenance manifest. Then use the quality checkpoint
and matching classifier TXT with `configs/publication_alpha/test_online_learned.py`.
See `tools/export_online.py --help` for `--classification-txt`, `--cls-filter`,
and required manifest checks. Export is local; it does not submit to the platform.
Test images/annotations are never used for model selection. Record the submitted
JSON/TXT, checkpoint hashes and returned scores independently.

## 5. Reproducibility and limitations

The paper reports hidden ALL AP/AP50/AP75 35.11/73.53/31.54 for its selected
model. It also reports a fixed-checkpoint, image-paired bootstrap comparison on
the 200 validation images; those intervals are not uncertainty over training
seeds and are not confidence intervals for hidden-test results. Baseline GFL,
GFLv2 and VFNet adaptations use different detection heads/candidates and are
system-level comparisons. Check `PROVENANCE.json`, existing configuration files
and the publication configuration for implementation details and audit constraints.

The supplied local `mmdet/datasets/coco.py` differs from the legacy hash in
`STANDALONE_SOURCE_MANIFEST.json`; consequently `tools/check_standalone.py`
currently reports core drift. The legacy manifest has not been silently rewritten.
Audit the evaluator change and verify the server's COCO evaluation against the
paper protocol before treating new measurements as a reproduction. Full CUDA
runtime, data loading and model training have not been verified in this package.
