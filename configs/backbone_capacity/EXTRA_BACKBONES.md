# Six additional official-backbone screens

## What changes, and what does not

Configs b3..b8 inherit the SAME historical 599/200 stage-1 control as b0..b2:
batch2, seed from CLI, 16 epochs, AdamW1e-4/decay1e-4, EMA, CrossScale math,
Sym/FDR head, stride4/8/16/32 anchors, input512, official TB AP selection.
No quality/pTB/H2 is built, and no old detector checkpoint is loaded.

| cfg | family | adapter output channels at strides4/8/16/32 |
|---|---|---|
| b3_lsnet_s | LSNet-S | 48,96,192,768 |
| b4_lsnet_b | LSNet-B | 64,128,256,896 |
| b5_overlock_xt | OverLoCK-XT | 56,112,340,420 |
| b6_mobilemamba_b1 | MobileMamba-B1 | 50,100,200,824 |
| b7_efficientvim_m3 | EfficientViM-M3 | 56,112,224,832 |
| b8_efficientvim_m4 | EfficientViM-M4 | 56,112,224,832 |

IMPORTANT: except OverLoCK, these are **four-level adaptations**, not original
paper architectures run without changes. Official coarse stems start at8 or16.
We expose genuine intermediate stem features at4/8 (no upsampled fake P2),
retain all original deep blocks, and concatenate the stride64 output (bilinear
upsampled) into the stride32 output. LSNet uses its native8/16/32 blocks plus
the stem4 tap. MobileMamba/EfficientViM use stem4/8 plus native16/32 blocks.
The deepest native64 branch is NOT discarded. No new learned projection is
inserted in the adapter; CrossScale's normal input projections accept the new
channel dimensions. Report this adaptation in any architecture comparison.

BN running statistics are frozen for batch2 (`norm_eval=True`); BN affine
parameters still learn, and every backbone block remains trainable. This follows
the LSNet/EfficientViM detection convention; it is an additional difference from
the LN-based TransNeXt control. Native stochastic-depth and pretraining recipes
differ. These runs cannot isolate parameter count as the sole causal variable.

### Scientific and engineering boundaries

- EfficientViM **does have an official RetinaNet implementation**, contrary to
  the earlier classification-only assessment. M4 is officially evaluated there;
  M3 here instantiates the same official detector class with official M3 depths
  and state dimensions. Both retain the published HSMSSD calculation.
- Official sources are unmodified, pinned by commit and 66 normalized source
  hashes in `third_party/capacity_backbones/SOURCES.json`. The adapter changes
  import/registry namespaces only and extracts the pyramid explicitly.
- Upstream licenses remain in their repositories; they are not relicensed.
- Pretrained loading fails if ANY original learned backbone weight is missing
  or shape-incompatible. Only OverLoCK's newly initialized detection output
  adapters (`extra_norm`, `h_proj`) may be absent from a classification ckpt.
  Classifier weights and geometry-index buffers are not backbone parameters.
- No replacement NATTEN, SKA or scan operator is supplied. MobileMamba refuses
  the upstream slow Python selective-scan fallback if no CUDA extension exists.
- Local validation: all config invariants and source hashes pass. EfficientViM
  M3/M4 real CPU forward passes at128-square,128x160,512-square and finite
  backward into the deepest stage pass (Torch2.7.1, timm1.0.22 test-only).
  No real pretrained checkpoint was downloaded locally. Other CUDA families,
  full MMDet training, checkpoint loading and old-server compatibility remain
  server smoke-test requirements. Do not skip them.

## Server files and upstream source

Work in your standalone TQ_SymNet root, NOT the older parent project.
Manually migrate:

1. `TQ_SymNet/models/capacity_backbones.py`
2. `configs/backbone_capacity/b3_*.py` through `b8_*.py` and this document
3. `tools/check_extra_capacity.py`
4. `third_party/capacity_backbones/SOURCES.json`
5. `TQ_SymNet/capacity_pretrained.py` (strict loader, required by adapter)
6. `tools/check_capacity_loading.py` (CPU loader regression tests)

Loader repair: also update the previously supplied
`tools/prepare_capacity_weights.py`, which uses the same version-aware loader.
Existing already-converted Tiny state_dict files do not need regeneration.

Keep the previously supplied `configs/backbone_capacity/_common.py` and its
base chain, `TQ_SymNet/supplement.py`, `tools/check_structure.py`. No edit to
backbones/__init__, mmdet/coco, NMS, existing model source, or existing configs
is required. The new cfg's custom_imports registers the adapter lazily.

Instead of uploading full upstream trees, fetch directly from GitHub below.
Commands assume these four destination directories do not already exist.

```bash
mkdir -p third_party/capacity_backbones pretrained
git clone --no-checkout https://github.com/THU-MIG/lsnet.git third_party/capacity_backbones/lsnet
git -C third_party/capacity_backbones/lsnet checkout --detach cbe737c92b7c43ecf02d08545a07f03f1010177c
git clone --no-checkout https://github.com/LMMMEng/OverLoCK.git third_party/capacity_backbones/overlock
git -C third_party/capacity_backbones/overlock checkout --detach fb6f22bef264c01be737ff90e88b4c16109e249b
git clone --no-checkout https://github.com/lewandofskee/MobileMamba.git third_party/capacity_backbones/mobilemamba
git -C third_party/capacity_backbones/mobilemamba checkout --detach 0980f48a23d232e0f2e86e740eccab038c8fd760
git clone --no-checkout https://github.com/mlvlab/EfficientViM.git third_party/capacity_backbones/efficientvim
git -C third_party/capacity_backbones/efficientvim checkout --detach 304340cb9c339b61669250d058525c9cdadd5e93
python tools/check_extra_capacity.py
python tools/check_capacity_loading.py
python tools/check_extra_capacity.py --environment
```

## Official classification checkpoint downloads (no proxy)

Non-distilled weights are used consistently here. Do not use a fused/deploy
OverLoCK checkpoint or a COCO detector checkpoint as an interchangeable input.
Official Google Drive links can have quotas; verify downloads are not HTML.

```bash
curl -fL https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_s.pth -o pretrained/lsnet_s.pth
curl -fL https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_b.pth -o pretrained/lsnet_b.pth
curl -fL https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_xt_in1k_224.pth -o pretrained/overlock_xt_in1k_224.pth
python -m pip install gdown==5.2.0
python -m gdown 1ODO1XM2luMfczbDLJxHrEl-ie7yUscAC -O pretrained/mobilemamba_b1.pth
python -m gdown 1z8dxkVLPbZdHaran9pDeuMmILsz5xUlf -O pretrained/EfficientViM_M3_e450.pth
python -m gdown 1rVnX2FT9AVJdU6xSPhEUllXZlhUYv28z -O pretrained/EfficientViM_M4_e450.pth
```

## Dependencies: DO NOT upgrade the working conda_env in place

The server's installed stack is the authority. Historical logs indicate
Python3.8 / torch1.10 / CUDA11.3; use `--environment` above to report the
actual current stack before deciding which optional families can run. This
repair does NOT require a torch/MMCV upgrade and does not promise SKA/NATTEN
support on that old stack. Keep blocked families pending instead of replacing
their operators or using an unreported CPU fallback.

The shared loader inspects torch.load's signature: torch1.10 receives only
map_location; versions exposing weights_only explicitly receive False for
trusted official files containing training metadata. Never load untrusted
pickle checkpoints. Learned weights and all persistent buffers must load;
missing BN means/variances, NaN/Inf or negative variances fail before training.
Only an old BN batch counter may be explicitly reset to zero. LSNet geometry
indices are regenerated locally and its learned bias tables use the original
detector's bicubic resizing rule; all such exceptions are logged.

The historical PyTorch1.10/MMDet2 environment is **not established compatible**
with all six backbones. Code import shims do not solve binary ABI differences.
Never install mmcv2 over the existing mmcv-full1.x used by this project.

- **EfficientViM** uses standard torch ops, timm, and no Mamba CUDA extension.
  Try its smoke first in the existing environment. Upstream detection recipe
  specifies Torch2.0.1, MMCV1.6 and MMDet2.25, not the historical project stack.
- **LSNet**: upstream requirements include Torch2.4.1, Triton, timm; its SKA
  source imports torch.amp.custom_fwd/custom_bwd. This is NOT drop-in on1.10.
- **OverLoCK**: upstream recommends Torch2.3.1/CUDA12.1, timm0.6.12,
  einops, NATTEN0.17.1. GPU neighborhood attention remains mandatory; its
  optional large-kernel iGEMM path is disabled in favor of upstream torch Conv2d.
- **MobileMamba**: requires Triton, PyWavelets, einops, fvcore and its official
  selective_scan extension. The repository's detector uses MMDet3/MMCV2, but
  our adapter intentionally bypasses only its registry, NOT its neural ops.
  Install the extension in a separately validated CUDA environment, e.g.:

```bash
python -m pip install ./third_party/capacity_backbones/mobilemamba/downstream/det/backbones/lib_mamba/kernels/selective_scan
```

If a new environment is necessary, rebuild the project's own MMCV/Sym/TransNeXt
CUDA extensions there, and re-run b0 in that environment before comparing AP.
Do not interpret a build failure as a negative model result. Dependency setup
must be validated on the actual server; no single universal install line is
claimed for these different upstream stacks.

## Smoke and sequential single-card training

First do FP32 forward and batch2 backward, one config at a time:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_extra_capacity.py --smoke b7_efficientvim_m3 --backward
CUDA_VISIBLE_DEVICES=0 python tools/check_extra_capacity.py --smoke b8_efficientvim_m4 --backward
# Use the same command with b3_lsnet_s, b4_lsnet_b, b5_overlock_xt,
# b6_mobilemamba_b1 ONLY once that family's dependencies are ready.
```

The loop below is sequential, requires just ONE card, and stops at the first
failed smoke or training. It is appropriate only in an environment validated
for every listed family; otherwise run family-specific subsets separately.

```bash
for cfg in b7_efficientvim_m3 b8_efficientvim_m4 b3_lsnet_s b4_lsnet_b b5_overlock_xt b6_mobilemamba_b1; do
  CUDA_VISIBLE_DEVICES=0 python tools/check_extra_capacity.py --smoke "$cfg" --backward &&
  CUDA_VISIBLE_DEVICES=0 python tools/train.py "configs/backbone_capacity/${cfg}.py" --seed 0 --deterministic --work-dir "work_dirs/backbone_capacity/${cfg}_seed0" || break
done
```

Inspect `[CAPACITY-PRETRAIN]`, `[CAPACITY-ADAPTER]`, `[SUPPLEMENT-BUILD]`, actual
AP/AP50/AP75, per-subtype AP, and late means. The source hashes and loading
audit make it possible to distinguish operator/environment errors from weak
generalization. No speed superiority or optimal TB capacity is assumed.
