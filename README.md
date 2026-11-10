# TQ-SymNet

A standalone MMDetection 2.x implementation for tuberculosis (TB) lesion detection
in chest radiographs. A TransNeXt/CrossScale/Sym detector generates candidates.
With the detector frozen, a candidate-conditioned quality evaluator and a global
fusion coefficient are trained to combine localization-quality evidence with
subtype confidence before Gaussian Soft-NMS.

The quality evaluator does not change box coordinates. Inference uses no ground-truth
boxes and requires no lung segmentation masks. This README covers only the paper's
main experiment. Historical ablation and exploratory configurations remain in the repository.

## Main experiment

- **Detection cohort:** 599 TB-positive training images / 200 validation images,
  with 902 / 309 annotated boxes.
- **Quality training:** eight epochs using the same frozen M2 detector.
  Checkpoints are selected by category-agnostic validation TB AP.
- **Final recipe:** the original quality reader, threshold-passing supervision,
  within-lesion ranking, and a learned global fusion coefficient.
- **Selected paper checkpoint:** quality epoch 7, alpha approximately 0.53571;
  validation AP/AP50/AP75: **40.1/77.5/35.7**.
- **Reported official hidden ALL results:** AP/AP50/AP75:
  **35.11/73.53/31.54**; paper-recorded submission ID: **1002246**.

The local validation cohort contains only TB-positive images; its scores are not
directly comparable with hidden ALL scores, which also depend on image screening.
The fixed-alpha control, offline alpha sweep, and separately trained learned-fusion
model are different experiments. A new run must select its own checkpoint on
validation AP; the same peak score or selected epoch is not guaranteed.

## 1. Environment and required assets

The historical environment is Linux, Python 3.8, PyTorch 1.10, CUDA 11.3,
MMCV 1.7.1 with compiled CUDA extensions, and compatible torchvision and timm.
TransNeXt's `swattention` extension must match the installed PyTorch/CUDA versions.
`requirements.txt` does not install PyTorch or mmcv-full, or compile CUDA operators.
Do not blindly upgrade an existing compatible environment.

```bash
git clone https://github.com/Nate047/TQ-symNet.git
cd TQ-symNet
# Activate your compatible MMDetection 2.x environment first.
python -m pip install -r requirements.txt
python -c "import mmdet; print(mmdet.__file__)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Run every command from the repository root. The printed `mmdet` path should point
to this repository. All examples use one GPU.

Obtain TBX11K images and annotations under their original data-use terms:
`TBX11K_train_only_tb.json`, `TBX11K_val_only_tb.json`,
`all_train.json`, `all_val.json`, and `all_test.json` without hidden annotations.
Image screening also uses the MC/Shenzhen and DA/DB examples in the official file lists.

Update the example server paths to match your files:

| Asset | Configuration |
|---|---|
| Detector training/validation images and JSON files | `data` and initialization-hook paths in `configs/publication_alpha/_original_a0.py` |
| Frozen M2 checkpoint | `detector_checkpoint` in the same file |
| C20 screening data and initial M2 source | `configs/classification/c0_backbone_sym.py`, inherited by C20 |
| Official test image index | `configs/publication_alpha/test_online_learned.py` |

The main experiment requires the original M2 epoch-13 checkpoint, expected at:
`work_dirs/m2_s0/best_official_tb_bbox_mAP_epoch_13.pth`.

Expected SHA-256:

```text
4c7ee5c68dc8dc10ee4fc938dad48404bbad6d3b67a6de3b3ea97756f6eaad74
```

The initialization hook also checks the training/validation JSON hashes.
File locations may change, but do not bypass content checks and describe a different
cohort or detector as the same experiment.

**Starting point:** the workflow below starts from the specified frozen M2.
The later 479/120/200 detector configurations are not replacements for the original
599/200 experiment. Regenerating the historical M2 from scratch still requires
verification of its corresponding training configuration and assets.

## 2. Train the main quality evaluator

```bash
python tools/train.py configs/publication_alpha/a1_learned.py \
  --seed 0 --deterministic \
  --work-dir work_dirs/publication_alpha/a1_learned_s0
```

The detector stays frozen, the quality evaluator is freshly initialized, and the
fusion coefficient is trainable. Do not reuse another experiment's nonempty work
directory. The seed and deterministic settings do not guarantee bitwise-identical CUDA runs.

## 3. Evaluate on local validation

Replace `QCKPT` with the actual checkpoint selected by validation AP in your run:

```bash
QCKPT=work_dirs/publication_alpha/a1_learned_s0/best_official_tb_bbox_mAP_epoch_7.pth

python tools/test.py configs/publication_alpha/a1_learned.py "$QCKPT" \
  --eval bbox --work-dir work_dirs/main_eval_tb \
  --cfg-options model.inference_mode=tb model.diagnostics=False

python tools/test.py configs/publication_alpha/a1_learned.py "$QCKPT" \
  --eval bbox --work-dir work_dirs/main_eval_subtype \
  --cfg-options model.inference_mode=subtype model.diagnostics=False
```

Keep the validation index supplied by the main configuration; do not substitute
the later self-partitioned test set. Overall TB AP uses COCO IoU thresholds
0.50:0.05:0.95 and category-agnostic matching. It is not the mean of subtype APs.
Classwise inference retains both subtype hypotheses and suppresses them separately;
report it separately from argmax subtype labels attached to overall-TB selections.
`tools/evaluate.py` is tied to another split-contract workflow and is not the
entry point for this original 599/200 experiment.

## 4. Train and export the independent C20 image classifier

Official ALL evaluation requires genuine Healthy / Sick non-TB / TB image-level
probabilities. The quality evaluator is not an image classifier. C20 fine-tunes
an independent encoder copy without modifying the detector or quality-model weights.

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

C20 trains for 12 epochs with microbatch size 2 and four accumulation steps
(effective batch size 8). Select its checkpoint by validation accuracy.
Export restores the complete selected classifier, including its tuned encoder,
and writes a probability TXT plus a provenance manifest. Keep both files and
preserve the official image order.

## 5. Export the final hidden-test detections

```bash
python tools/export_online.py \
  configs/publication_alpha/test_online_learned.py "$QCKPT" \
  --classification-txt work_dirs/main_submission/TQSymNet.txt \
  --cls-filter \
  --json-prefix work_dirs/main_submission/TQSymNet
```

The official index must include all 3,302 images without hidden ground truth.
Detections are exported only for images classified as TB by C20.
This command generates local files; it does not submit to the challenge platform.
Archive the detection JSON, classification TXT, manifests, and returned scoring
records. Select models on validation data, not hidden-test feedback.

## Model weights and submission records

| Asset | Availability |
|---|---|
| Main M2 detector checkpoint | Pending release of the actual weight file |
| Final learned-fusion quality checkpoint | Pending release of the actual weight file |
| Independent C20 classifier checkpoint | Pending release of the actual weight file |
| 2026-10-19 hidden-test predictions | [Available: detection JSON and classification TXT](results/hidden_all_20261019/README.md) |
| Official scoring response / platform export | Pending publication of the original response |

[Download the prediction archive](results/hidden_all_20261019/TQSymNet_submission.zip).
It contains 63,802 predicted boxes for 643 exported images and 3,302 rows of
classification probabilities, not hidden annotations or model weights.
Checksums establish file identity with the archived export; they do not independently
verify a platform submission ID or reproduce its hidden score.
Weight downloads, matching configurations, and SHA-256 values will be added after
the actual files are provided. No placeholder download links are supplied.

## Reproducibility notes

Data and pretrained resources remain subject to their respective licenses.
Validation supports both selection and analysis, so local results are
selection-conditioned. Image-level bootstrap does not estimate training-seed
variability. Static checks do not replace full CUDA runtime verification.

The current `coco.py` differs from the legacy `STANDALONE_SOURCE_MANIFEST.json`
hash, so the legacy standalone check reports core drift. The manifest has not
been rewritten to hide that difference. Verify the actual evaluator against the
paper protocol before treating new measurements as a reproduction.
Implementation provenance is documented in [PROVENANCE.json](PROVENANCE.json).
