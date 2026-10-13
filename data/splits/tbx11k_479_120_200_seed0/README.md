# Active experiment: one fixed TBX11K split, seed 0

All reproduced baselines and TQ-SymNet use this same image partition. Report
single-run AP/AP50/AP75 and separate subtype metrics. No multi-seed mean/std
is claimed for this experiment.

| Partition | Images | Boxes | Active boxes | Obsolete boxes |
|---|---:|---:|---:|---:|
| Train | 479 | 728 | 586 | 142 |
| Val | 120 | 183 | 147 | 36 |
| Test | 200 | 300 | 239 | 61 |

`seed_0/train.json`, `val.json`, `test.json` preserve subtype labels and original
box geometry. Their *_tb.json counterparts merge both labels to TB category 1,
for single-class GLIP-family training/evaluation. Both forms use identical images.
Exact names, annotation identities and hashes are recorded in manifest.json.

The model optimizer seed is also 0 for the initial reproduced runs. Model
selection uses val; fixed final checkpoints are evaluated on test. Each task
model is trained from generic pretrained weights on these 479 images. The old
599-image detector checkpoint has seen 157 images now assigned to test and
cannot serve as the starting detector for the new experiment.

To regenerate on the server from the TQ_SymNet root:

```bash
python tools/generate_tb_only_splits.py \
  --source-json /home/data2/LiYingxian/Dataset_TB/TBX11K/annotations/json/TBX11K_trainval_only_tb.json \
  --image-root /home/data2/LiYingxian/Dataset_TB/TBX11K/imgs \
  --out-dir data/splits/tbx11k_479_120_200_seed0 \
  --seeds 0
```

The image root contains `tb/*.png`. Only the JSONs/manifests and generator need
migration when the server already has the same image collection. Partition sizes
follow published GLIP-family protocols; the actual image lists are our declared
partition and are not claimed to be identical to unpublished author splits.
