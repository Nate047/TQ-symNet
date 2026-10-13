"""Evaluate B controls or any A checkpoint with its OWN config and fixed COCO protocol."""
import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import mmcv
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.utils import import_modules_from_strings
    from mmdet.apis import single_gpu_test
    from mmdet.core import bbox_overlaps
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from TQ_SymNet.checkpoints import load_detector_checkpoint, load_retained_checkpoint
    from TQ_SymNet.data_contract import (load_split_contract, verify_checkpoint_contract,
                                        sha256_file)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--mode', choices=('tb', 'subtype'), default='tb')
    parser.add_argument('--split', choices=('val', 'test'), default=None,
                        help='Explicit cohort partition; default preserves config test annotations')
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.split:
        _, paths = load_split_contract(cfg.split_manifest)
        cfg.data.test.ann_file = str(paths[args.split])
    import_modules_from_strings(**cfg.custom_imports)
    quality_model = cfg.model.type == 'TQSymNet'
    if not quality_model and args.mode != 'tb':
        raise ValueError('B controls are category-agnostic TB comparisons')
    if quality_model:
        cfg.model.inference_mode = args.mode
        cfg.model.diagnostics = args.diagnostics
    cfg.model.backbone.pretrained = None
    if 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                              dist=False, shuffle=False)
    model = build_detector(cfg.model)
    load = load_retained_checkpoint if quality_model else load_detector_checkpoint
    checkpoint, audit = load(model, args.checkpoint)
    contract, _ = load_split_contract(cfg.split_manifest)
    verify_checkpoint_contract(checkpoint.get('meta', {}).get('tq_data_contract'), contract)
    if quality_model and cfg.get('enforce_detector_provenance', False):
        expected_init = next(h for h in cfg.custom_hooks if h['type'] == 'TQDetectorInitHook')
        expected_file = Path(expected_init['checkpoint'])
        if not expected_file.is_absolute():
            expected_file = ROOT / expected_file
        saved_init = checkpoint.get('meta', {}).get('tq_initialization', {})
        if saved_init.get('detector_checkpoint_sha256') != sha256_file(expected_file):
            raise RuntimeError('Quality checkpoint was trained on a different detector; '
                               'do not mix the 16-epoch and 20-epoch ablation families')
    if sha256_file(cfg.data.test.ann_file) not in (contract['val_sha256'], contract['test_sha256']):
        raise RuntimeError('Evaluation must use this cohort\'s validation or reserved test JSON')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('tq_eval')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(out / 'evaluation.log', encoding='utf-8'))
    logger.info('[TQ-EVAL] checkpoint=%s mode=%s audit=%s', args.checkpoint, args.mode, audit)
    if quality_model and args.diagnostics:
        model.force_oracle = True
        model.v50_begin_oracle(int(checkpoint.get('meta', {}).get('epoch', 0)))
    if not quality_model:
        model.capture_candidates = args.diagnostics
    parallel = MMDataParallel(model.cuda(), device_ids=[0])
    outputs = single_gpu_test(parallel, loader)
    options = dict(metric='bbox', classwise=True, tb_official_agnostic=True,
                   tb_class_agnostic=False, bbox_max_dets=(1, 10, 100))
    metrics = dataset.evaluate(outputs, **options)
    result = dict(config=args.config, checkpoint=args.checkpoint, output_mode=args.mode,
                  images=len(dataset), metrics=metrics, checkpoint_audit=audit,
                  checkpoint_epoch=checkpoint.get('meta', {}).get('epoch'),
                  initialization=checkpoint.get('meta', {}).get('tq_initialization'))
    mmcv.dump(result, str(out / 'metrics.json'))
    cfg.dump(str(out / 'resolved_config.py'))
    dataset.format_results(outputs, jsonfile_prefix=str(out / 'predictions'))
    if quality_model and args.diagnostics:
        model.v50_finish_oracle(dataset, SimpleNamespace(
            logger=logger, work_dir=str(out), epoch=model.quality_epoch-1))
    if not quality_model and args.diagnostics:
        summary = {s: dict(count=0, iou_bins=[0]*5, covered_gt=[0]*3) for s in ('pre', 'final')}
        for index, row in enumerate(model.baseline_cache):
            boxes = row['boxes']
            gt = torch.as_tensor(dataset.get_ann_info(index)['bboxes']).float().reshape(-1, 4)
            overlap = bbox_overlaps(boxes, gt)
            actual = overlap.max(1).values if len(gt) else torch.zeros(len(boxes))
            for stage, ids in (('pre', torch.arange(len(boxes))), ('final', row['selected'])):
                value = actual[ids]
                counts = [int(m.sum()) for m in (value < .1, (value >= .1)&(value < .5),
                    (value >= .5)&(value < .75), (value >= .75)&(value < .85), value >= .85)]
                coverage = [int((overlap[ids] >= t).any(0).sum()) for t in (.5,.75,.85)]
                entry = summary[stage]
                entry['count'] += len(ids)
                entry['iou_bins'] = [a+b for a,b in zip(entry['iou_bins'],counts)]
                entry['covered_gt'] = [a+b for a,b in zip(entry['covered_gt'],coverage)]
        mmcv.dump(summary, str(out / 'candidate_stages.json'))
        logger.info('[TQ-BASELINE-STAGES] %s', summary)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
