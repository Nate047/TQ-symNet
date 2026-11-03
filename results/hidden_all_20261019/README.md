# TBX11K hidden ALL submission predictions

`TQSymNet_submission.zip` is the user-provided 2026-10-19 submission bundle
for the learned-fusion TQ-SymNet model. It is provided for result auditing,
not as ground-truth annotation or a training dataset. The archive contains:

- `TQSymNet.json`: 63,802 model-predicted boxes for 643 images, in COCO result
  format (`image_id`, `bbox` as `[x,y,width,height]`, `score`, `category_id`).
- `TQSymNet.txt`: 3,302 rows of image-level classification probabilities,
  one row per test image in the official index order, with columns Healthy,
  Sick non-TB and TB. The independent image classifier gated which images
  produced detections. Row ordering must be interpreted with the official index.

The archived export manifest identifies the detector checkpoint as
`publication_alpha/a1_learned_s0/best_official_tb_bbox_mAP_epoch_7.pth`.
Both files' SHA-256 values match the separately archived 2026-10-19 learned-alpha
export. This checksum match verifies file identity with that export; it does not
independently prove the platform submission ID or reproduce the hidden score.
The scoring server retains the hidden ground truth. The paper reports the ALL
score returned for its selected submission; no Only-TB score is claimed.

SHA-256:

```text
bc9ae6fbc94c7c6f3c02c85147343bbc4065ea2bcc0be62f4ea3d48ea9e9edac  TQSymNet_submission.zip
77313047d7999dfb17a228d4da061907d4c98670900454a65f9f30c9e3633559  TQSymNet.json
08828b8a2e8cd2bf73a51584c8240e48feaf3685a75a2f3b234d97dac7187dfa  TQSymNet.txt
```

No test images, labels, model checkpoints or official scoring response are
included. Dataset access and reuse remain governed by the original dataset terms.
