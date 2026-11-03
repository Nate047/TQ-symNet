# Stage-1 backbone capacity screen

These configs belong inside the standalone `TQ_SymNet` server project.
They train the main detector, not a frozen quality network. No H2/pTB/quality
module is constructed. Existing implementation files are unchanged.

| Config | Backbone | Pretraining |
|---|---|---|
| b0 | current TransNeXt Small control | ImageNet-1K, fine-tuned at 384 |
| b1 | TransNeXt Tiny | ImageNet-1K 224 |
| b2 | Swin-T | ImageNet-1K 224, original Microsoft format |

Fixed: historical train 599/902 boxes, val 200/309 boxes, pure-Obsolete x3
additional repeats; 512 input; single GPU batch2; AdamW 1e-4/decay1e-4;
16 epochs; LR steps11/15; EMA.001; same CrossScale algorithm and Sym/FDR
head; same evaluator and native head NMS. Swin changes only the neck's
required input channels to 96/192/384/768; output stays 256.

The default 479/120/200 split hook is REPLACED, not retained. The existing
supplement audit validates original JSON hashes, no train/val overlap and
454 optimizer updates/epoch. `data.test` points to local VAL; nothing here
submits to the online test platform. Compare to a same-run b0, not to an
old quality-scored Soft-NMS result.

Limitations: Tiny/Small differ in native stochastic depth (.3/.5), and
Swin uses .2; pretraining resolution and architecture also differ. This
is not an isolated proof that parameter count causes overfitting. Read
train losses alongside val AP/AP50/AP75 and late-epoch means. No quality
checkpoint can be reused across these backbones without retraining.

## Server preparation (project root)

Small keeps the path already used by `configs/train_detector.py`.
Download only from official repositories; no proxy is used.

```bash
mkdir -p pretrained
curl -fL 'https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true' -o pretrained/transnext_tiny_224_official.pth
curl -fL 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth' -o pretrained/swin_tiny_patch4_window7_224.pth
python tools/prepare_capacity_weights.py --tiny pretrained/transnext_tiny_224_official.pth
python tools/check_capacity_configs.py
CUDA_VISIBLE_DEVICES=0 python tools/check_capacity_configs.py --smoke
```

`tools/prepare_capacity_weights.py` also requires
`TQ_SymNet/capacity_pretrained.py`. This loader supports torch1.10 and newer
torch metadata checkpoints without changing the installed environment.

Tiny preparation strips the original classifier/wrapper, checks stage depths
and saves a separate MMDet-compatible state_dict. It never overwrites a file.
Swin must remain in original format: its loader uses `convert_weights=True`
including patch-merging weight order conversion. Check loading warnings:
missing output stage norms can be expected for classification checkpoints;
missing stage/block convolution or attention weights is NOT acceptable.

## Single GPU, sequential, stop on failure

```bash
for cfg in b0_transnext_small b1_transnext_tiny b2_swin_tiny; do
  CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/backbone_capacity/${cfg}.py --seed 0 --deterministic --work-dir work_dirs/backbone_capacity/${cfg}_seed0 || break
done
```

To run only the two new backbones, remove `b0_transnext_small` from the loop,
but do not claim strict capacity attribution without a comparable Small run.

Migrate this entire config directory plus `tools/prepare_capacity_weights.py`
and `tools/check_capacity_configs.py`; the rest is already in the standalone
project (train_detector, _common, models, supplement and MMDet Swin code).
