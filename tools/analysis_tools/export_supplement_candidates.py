"""One inference of the selected quality checkpoint; no training or GT scoring."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.utils import import_modules_from_strings
    from mmdet.apis import single_gpu_test, set_random_seed
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from TQ_SymNet.checkpoints import load_retained_checkpoint
    from TQ_SymNet.data_contract import sha256_file
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config'); p.add_argument('checkpoint')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--expected-epoch', type=int, default=7)
    p.add_argument('--expected-checkpoint-sha256', default='a567dc35dea116d0c9ffd7ab0cb0c597293e127e5fc732478c71e08c22b92532')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    set_random_seed(args.seed, deterministic=True)
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError('Wrong selected checkpoint hash; do not silently substitute another run')
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.backbone.pretrained = None
    cfg.model.diagnostics = True
    cfg.model.inference_mode = 'tb'
    data = cfg.data.val.copy(); data.test_mode = True
    expected_val = 'bdec993506af15bf72010b32ec82ac279615ccc4c51ad9ca9438a7b9638d6e9d'
    if sha256_file(data.ann_file) != expected_val:
        raise RuntimeError('Expected the original 200-image validation cohort')
    dataset = build_dataset(data)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                             dist=False, shuffle=False)
    core = build_detector(cfg.model)
    checkpoint, audit = load_retained_checkpoint(core, args.checkpoint)
    epoch = int(checkpoint.get('meta', {}).get('epoch', -1))
    if epoch != args.expected_epoch:
        raise RuntimeError('Unexpected selected checkpoint epoch')
    core.force_oracle = True; core.v50_begin_oracle(epoch)
    core.CLASSES = dataset.CLASSES
    output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        predictions = single_gpu_test(MMDataParallel(core.cuda().eval(), device_ids=[0]), loader)
    if len(core.quality_cache) != len(dataset):
        raise RuntimeError('Candidate export is incomplete')
    rows = []
    for row in core.quality_cache:
        # Explicit quality logit, not a possibly historical residual alias.
        rows.append(dict(filename=row['filename'], boxes=row['boxes'].tolist(),
            sym=row['sym'][:, :-1].max(1).values.tolist(),
            residual_logit=row['logit'].tolist(), corrected_score=row['corrected'].tolist(),
            soft_source=row['soft_source'].tolist(), soft_score=row['soft_score'].tolist()))
    candidate_path = output/'selected_candidates.json'
    candidate_path.write_text(json.dumps(rows), encoding='utf-8')
    dataset.format_results(predictions, jsonfile_prefix=str(output/'selected_original'))
    alpha = float(.1+.8*core.candidate_quality.fusion_gate_logit.detach().sigmoid().cpu())
    meta = dict(config=str(Path(args.config).resolve()), config_sha256=sha256_file(args.config),
        resolved_config_sha256=__import__('hashlib').sha256(cfg.pretty_text.encode()).hexdigest(),
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256_file(args.checkpoint),
        checkpoint_epoch=epoch, checkpoint_audit=audit, annotation_sha256=expected_val,
        candidates_sha256=sha256_file(candidate_path), images=len(rows),
        candidates=sum(len(r['boxes']) for r in rows), original_alpha=alpha,
        sigma=core.selector_sigma, topk=128, max_dets=100,
        quality_field='residual_logit stores row[logit], the actual quality logit',
        scope='Selected checkpoint, original 200-image VAL; no train/test fitting')
    candidate_path.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    cfg.dump(str(output/'resolved_config.py'))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
