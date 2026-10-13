"""Export ACTUAL simple_test outputs + existing diagnostic cache; no new NMS.

Run in the server's existing MMDetection/CUDA environment. GT is accessed only
after each forward and is never supplied to model inference.
"""
import argparse
import copy
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import sha256, write_json


def detector_state_hash(model):
    """Include buffers, dtypes and shapes as well as main-network parameters."""
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        if not key.startswith(('backbone.','neck.','bbox_head.')):continue
        array=value.detach().cpu().contiguous().numpy()
        digest.update((key+'|'+str(array.dtype)+'|'+str(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--ann-file')
    parser.add_argument('--img-root')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--candidates', action='store_true')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Smoke test only. Partial runs do not calculate AP.')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('--max-images must be nonnegative')

    import mmcv
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.utils import import_modules_from_strings
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from TQ_SymNet.checkpoints import load_detector_checkpoint, load_retained_checkpoint
    from TQ_SymNet.models.candidates import main_hash

    if not torch.cuda.is_available():
        raise RuntimeError('Use the server CUDA environment; no CPU model fallback.')
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    quality = cfg.model.type == 'TQSymNet'
    if quality:
        cfg.model.inference_mode = 'tb'
        cfg.model.diagnostics = args.candidates
    elif cfg.model.type != 'TQDetector' or cfg.model.get('inference_nms', 'head') == 'head':
        raise ValueError('Use a0_full or b0_sym_hard/b1_sym_soft, not a training-head config.')
    cfg.model.backbone.pretrained = None
    if 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    data_cfg = copy.deepcopy(cfg.data[args.split])
    if args.ann_file:
        data_cfg.ann_file = args.ann_file
    if args.img_root:
        data_cfg.img_prefix = args.img_root
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)
    if not hasattr(dataset, 'coco') or not hasattr(dataset, 'data_infos'):
        raise ValueError('Export expects a single CocoDataset, not Repeat/ConcatDataset.')
    if not len(dataset):
        raise ValueError('Empty dataset: check annotation paths and category names')
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
                              dist=False, shuffle=False)
    model = build_detector(cfg.model)
    load = load_retained_checkpoint if quality else load_detector_checkpoint
    checkpoint, audit = load(model, args.checkpoint)
    model.CLASSES = dataset.CLASSES
    detector_digest = main_hash(model)
    detector_state_digest = detector_state_hash(model)
    if quality:
        model.quality_collect = args.candidates
        model.quality_epoch = int(checkpoint.get('meta', {}).get('epoch', 0))
    else:
        model.capture_candidates = args.candidates
    parallel = MMDataParallel(model.cuda(), device_ids=[0])
    parallel.eval()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, outputs = [], []
    progress = mmcv.ProgressBar(len(dataset))
    for index, batch in enumerate(loader):
        if any(key.startswith('gt_') for key in batch):
            raise ValueError('Inference pipeline must not collect GT tensors')
        # Validate metadata before inference. This exporter supports one scale,
        # no flip: output and cache are then BOTH native-image coordinates.
        metas = batch['img_metas'][0].data[0]
        if len(metas) != 1 or metas[0].get('flip', False):
            raise ValueError('Expected batch=1 and unflipped validation')
        if len(batch['img']) != 1:
            raise ValueError('Multi-scale test-time augmentation is not supported')
        with torch.no_grad():
            predictions = parallel(return_loss=False, rescale=True, **batch)
        if len(predictions) != 1 or len(predictions[0]) != len(dataset.CLASSES):
            raise RuntimeError('Unexpected simple_test output structure')
        result = predictions[0]
        outputs.append(result)
        info, ann = dataset.data_infos[index], dataset.get_ann_info(index)
        filename = str(Path(dataset.img_prefix) / info['filename'])
        row = dict(image_id=int(info['id']), filename=filename,
                   file_name=info['filename'], width=int(info['width']),
                   height=int(info['height']), gt=[dict(bbox_xyxy=b.tolist(),
                       label=int(label), category_id=int(dataset.cat_ids[label]))
                       for b, label in zip(ann['bboxes'], ann['labels'])],
                   ignored_gt=ann['bboxes_ignore'].tolist(), predictions=[])
        for label, array in enumerate(result):
            array = np.asarray(array).reshape(-1, 5)
            if not np.isfinite(array).all():
                raise RuntimeError('Non-finite model prediction')
            for detection in array:
                row['predictions'].append(dict(bbox_xyxy=detection[:4].tolist(),
                    score=float(detection[4]), label=label,
                    category_id=int(dataset.cat_ids[label])))
        row['predictions'].sort(key=lambda item: -item['score'])
        if args.candidates:
            cache = model.quality_cache if quality else model.baseline_cache
            if len(cache) != 1:
                raise RuntimeError('Diagnostic cache must contain exactly this image')
            item = cache[0]
            if item.get('filename') != metas[0].get('filename'):
                raise RuntimeError('Cache/image alignment mismatch')
            if quality:
                row['candidates'] = dict(boxes_xyxy=item['boxes'].tolist(),
                    sym_probability=item['sym'][:, :model.bbox_head.num_classes].tolist(),
                    quality_logit=item['logit'].tolist(),
                    quality_probability=item['quality'].tolist(),
                    fused_probability=item['corrected'].tolist(),
                    selected_indices=item['source'].tolist(),
                    selected_scores=item['score'].tolist())
            else:
                row['candidates'] = dict(boxes_xyxy=item['boxes'].tolist(),
                    selected_indices=item['selected'].tolist(),
                    selected_scores=item['score'].tolist())
            cache.clear()
        # Clear diagnostic tensors per image; do NOT retain GPU graphs/maps.
        for name in ('detail_debug', 'subtype_gate_stats'):
            if hasattr(model, name):
                getattr(model, name).clear()
        if quality:
            model.candidate_quality.attention_stats.clear()
        rows.append(row)
        progress.update()
        if args.max_images and len(rows) >= args.max_images:
            break
    complete = len(outputs) == len(dataset)
    metrics = None
    if complete:
        metrics = dataset.evaluate(outputs, metric='bbox', classwise=True,
            tb_official_agnostic=True, tb_class_agnostic=False, bbox_max_dets=(1, 10, 100))
        dataset.format_results(outputs, jsonfile_prefix=str(out / 'predictions'))
    config_path = out / 'resolved_config.py'
    cfg.data[args.split] = data_cfg
    cfg.dump(str(config_path))
    alpha = None
    if quality and model.selector_learned_gate:
        alpha = float((.1 + .8 * model.candidate_quality.fusion_gate_logit.sigmoid()).detach())
    export = dict(schema='tq-paper-v1', coordinate_format='native_image_xyxy',
        run=dict(tag=args.tag, config=str(Path(args.config).resolve()),
            config_sha256=sha256(args.config), resolved_config_sha256=sha256(config_path),
            checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
            checkpoint_epoch=checkpoint.get('meta', {}).get('epoch'), checkpoint_audit=audit,
            detector_sha256=detector_digest, classes=list(dataset.CLASSES),
            detector_state_sha256=detector_state_digest,
            ann_file=str(data_cfg.ann_file), ann_sha256=sha256(data_cfg.ann_file),
            split=args.split, image_count=len(rows), complete=complete,
            mode='tb', metrics=metrics, fusion_alpha=alpha,
            nms_pre=int(model.bbox_head.test_cfg.nms_pre),
            eligibility_score_thr=float(model.bbox_head.test_cfg.score_thr),
            max_per_img=int(model.bbox_head.test_cfg.max_per_img),
            selection_nms=('gaussian_soft' if quality or model.inference_nms=='tb_soft' else 'hard'),
            selection_sigma=(float(model.selector_sigma) if quality else float(model.baseline_sigma)),
            selection_topk=(128 if quality else int(model.baseline_topk)),
            quality_target=getattr(model, 'selector_quality_target', None),
            quality_score_note='sigmoid(zQ) is a threshold surrogate, NOT literal IoU',
            gt_used_in_inference=False), images=rows)
    destination = out / 'visualization.json.gz'
    write_json(destination, export)
    print('\nSaved', destination)


if __name__ == '__main__':
    main()
