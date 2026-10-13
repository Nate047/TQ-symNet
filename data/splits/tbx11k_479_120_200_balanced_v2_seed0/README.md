# Active balanced-v2 split, seed 0

| Phase | Images | Boxes | Active | Obsolete | Small/medium/large |
|---|---:|---:|---:|---:|---|
| Train | 479 | 730 | 587 | 143 | 5 / 190 / 535 |
| Val | 120 | 181 | 146 | 35 | 1 / 47 / 133 |
| Test | 200 | 300 | 239 | 61 | 2 / 78 / 220 |

The reserved 200 test image identities and all their original annotations are
unchanged from v1. Only train and val membership was revised, using whole-image
exchanges within active-only / obsolete-only / mixed strata. Original image IDs,
annotation IDs, categories, coordinates and area values are preserved.

The fixed annotation-only objective balances subtype-specific area, width,
height, aspect-ratio, 3x3 position histograms, and boxes per image. Category box
counts are constrained to floor/ceil of proportional counts; val retains 1–2
small boxes. The deterministic optimizer stops at a pair-swap local optimum or
250 exchanges. It does not read predictions, scores, checkpoints, or AP.

This is a documented protocol revision after a distribution audit, not an
untouched original holdout or the authors' unpublished MIU/GLIP partition.
Patient independence cannot be established without patient IDs. All 799 source
image dimensions and exact pixel hashes were checked; no exact duplicates were
found. Histograms do not guarantee equal clinical difficulty.

`balance_report.json` records every exchange, the declared objective, bin edges,
and pre/post histograms. The v1 files remain in the sibling original directory.
`seed_0/*.json` are the actual labels. `*_tb.json` merge subtype labels to TB=1
without changing images or boxes; ordinary train/val/test.json retain subtypes.

Reproduce from the TQ project root:

```bash
python tools/generate_balanced_tb_split.py \
  --source-json /home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_trainval_only_tb.json \
  --image-root /home/data2/LiYingxian/Dataset_TB/TBX11K/imgs \
  --parent-split data/splits/tbx11k_479_120_200_seed0/seed_0 \
  --out-dir data/splits/tbx11k_479_120_200_balanced_v2_seed0
```

Train every baseline anew on balanced-v2. V1 and old 599-image task checkpoints
are not valid initializations. Generic ImageNet/GLIP pretraining remains allowed.
Use val for model selection; the reserved test is for the locked final experiment.
