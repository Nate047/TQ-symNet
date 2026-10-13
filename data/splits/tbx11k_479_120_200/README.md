# TBX11K public-box split: 479 / 120 / 200

This is our declared stratified split of the public 799-image/1211-box TB pool.
It follows the **sizes** in ConES, MIU-VL and StructuralGLIP; it is not a claim
to reproduce their unpublished filenames. No new lesion labels are invented.

| Image stratum | Train | Val | Test |
|---|---:|---:|---:|
| Active only | 378 | 94 | 158 |
| Obsolete only | 83 | 21 | 35 |
| Mixed | 18 | 5 | 7 |
| Total | 479 | 120 | 200 |

For each seed 0, 999, 2022:

- `train.json`, `val.json`, `test.json` retain source subtype categories, image
  and annotation IDs, boxes, areas and all source metadata.
- `train_tb.json`, `val_tb.json`, `test_tb.json` use one `tuberculosis` category
  with COCO category ID 1. Original subtype IDs are retained in the auxiliary
  `source_category_id` annotation field.
- `*_images.txt` and `manifest.json` record the exact cohort and all file hashes.

Partitioning is keyed to the split seed and image filename using SHA256, making
it independent of global training RNG and Python/NumPy shuffle versions. The
source image audit checks stored sizes and exact decoded-image duplicates.
Patient identity cannot be checked without patient metadata. Seeds are three
paired repeated holdouts; results should not be pooled as 600 independent images.

Both branches of the current train ConcatDataset (full train and pure-Obsolete
repeat) must use the same **new train.json**. The validation loader uses val.json;
the final test loader uses test.json. Keep oversampling inside train only.
GLIP single-TB configurations can use the corresponding *_tb.json files.
Use the same filenames for every baseline and TQ experiment at a given seed.

Every newly partitioned run starts from generic pretrained weights, rather than
a detector or quality checkpoint trained on the old 599-image cohort. Choose
checkpoints on val; reserve test for final evaluation. Split seed and optimizer
seed are recorded separately, even if both use the same number.

Rebuild on the server (from the standalone TQ_SymNet root):

```bash
python tools/generate_tb_only_splits.py \
  --source-json /home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_trainval_only_tb.json \
  --image-root /home/data2/LiYingxian/Dataset_TB/TBX11K/imgs \
  --out-dir data/splits/tbx11k_479_120_200 \
  --seeds 0 999 2022
```

The generator refuses to overwrite any different existing cohort. Migrating the
already generated JSONs/manifests is sufficient; image files need not be copied
again when the server uses the same TBX11K image collection.
