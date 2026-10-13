"""GT-free official bbox export, optionally gated by real image-class probabilities."""
import argparse
import json
import math
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_external_classification(txt, image_ids, annotation):
    """Read official-example rows: line n corresponds to official image ID n.

    The TXT has no embedded IDs. We validate the declared numeric ordering,
    but cannot reconstruct/independently prove its provenance from its bytes.
    External predictions must never masquerade as our frozen classifier.
    """
    if list(image_ids) != list(range(1, 3303)):
        raise ValueError('External official-example TXT requires ordered image IDs 1..3302')
    txt = Path(txt)
    raw = txt.read_bytes()
    rows = [list(map(float, line.split()))
            for line in raw.decode('utf-8-sig').splitlines()]
    if (len(rows) != 3302 or any(len(row) != 3
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in row)
            or abs(sum(row) - 1.) > 1e-6 for row in rows)):
        raise ValueError('External TXT requires 3302 valid Healthy/Sick/TB probability rows')
    manifest = dict(
        class_order=['Healthy', 'Sick non-TB', 'TB'], image_ids=list(image_ids),
        annotation_sha256=hashlib.sha256(Path(annotation).read_bytes()).hexdigest(),
        txt_sha256=hashlib.sha256(raw).hexdigest(),
        external_classifier=True, source_txt=str(txt),
        row_order='Official-example convention: line n maps to image_id n (1..3302)',
        experiment_role='external_classifier_diagnostic',
        note='External predictions; not classification results of TQ-SymNet.')
    return rows, manifest


def main():
    import mmcv
    import numpy as np
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from TQ_SymNet.checkpoints import load_retained_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet.apis import single_gpu_test
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from TQ_SymNet.classification import CLASS_NAMES, file_sha256, state_sha256
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--json-prefix', required=True)
    parser.add_argument('--mode', choices=('tb', 'subtype'), default='tb')
    parser.add_argument('--allow-development-split', action='store_true')
    classification_group = parser.add_mutually_exclusive_group()
    classification_group.add_argument('--classification-txt', help='TXT exported by export_classification.py')
    classification_group.add_argument('--external-classification-txt',
        help='Diagnostic ONLY: official-example TXT, row n = official image ID n; no backbone match implied')
    parser.add_argument('--cls-filter', action='store_true',
                        help='Official Sym rule: run detection only where class argmax is TB')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.backbone.pretrained = None
    cfg.model.diagnostics = False
    cfg.model.inference_mode = args.mode
    cfg.data.test.test_mode = True
    annotation = Path(cfg.data.test.ann_file)
    if not args.allow_development_split and 'test' not in annotation.stem.lower():
        raise ValueError('Refusing to label a validation export as hidden-test submission')
    index = mmcv.load(str(annotation))
    image_ids = [image['id'] for image in index['images']]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError('Duplicate image IDs in the test index')
    if index.get('annotations') and not args.allow_development_split:
        raise ValueError('Hidden-test export expects an image index without GT annotations')
    if not args.allow_development_split and len(image_ids) != 3302:
        raise ValueError('Official hidden-test export expects all 3302 indexed images')
    classification_manifest = None
    external_manifest = None
    classification = None
    if args.cls_filter and not (args.classification_txt or args.external_classification_txt):
        raise ValueError('--cls-filter requires a classification TXT')
    if args.external_classification_txt:
        if not args.cls_filter or args.allow_development_split:
            raise ValueError('External official-example diagnostic requires --cls-filter and hidden-test index')
        rows, external_manifest = load_external_classification(
            args.external_classification_txt, image_ids, annotation)
        classification = np.asarray(rows, dtype=np.float64)
        print('[ONLINE-EXTERNAL-CLASSIFIER] diagnostic=True source=%s sha256=%s '
              'rowConvention=line_n_to_image_id_n; classification is NOT ours' %
              (args.external_classification_txt, external_manifest['txt_sha256']))
    if args.classification_txt:
        txt = Path(args.classification_txt)
        classification = np.loadtxt(txt, dtype=np.float64, ndmin=2)
        if (classification.shape != (len(image_ids), 3) or not np.isfinite(classification).all()
                or (classification < 0).any() or (classification > 1).any()
                or not np.allclose(classification.sum(1), 1., atol=1e-6)):
            raise ValueError('Invalid classification rows/probabilities')
        classification_sidecar = txt.with_suffix('.classification.manifest.json')
        if not classification_sidecar.exists():
            classification_sidecar = txt.with_suffix('.manifest.json')
        classification_manifest = mmcv.load(str(classification_sidecar))
        if 'class_order' not in classification_manifest:
            raise ValueError('Legacy classification manifest was overwritten by detection export; '
                             'rerun export_classification.py to restore its dedicated sidecar')
        if (classification_manifest['image_ids'] != image_ids
                or tuple(classification_manifest['class_order']) != CLASS_NAMES
                or classification_manifest['annotation_sha256'] != file_sha256(annotation)
                or classification_manifest['txt_sha256'] != file_sha256(txt)):
            raise ValueError('Classification TXT, image order, or annotation index does not match')
    import_modules_from_strings(**cfg.custom_imports)
    dataset = build_dataset(cfg.data.test)
    if list(dataset.img_ids) != image_ids:
        raise ValueError('Dataset changed official test image order/IDs')
    model = build_detector(cfg.model)
    # Full network checkpoints must contain the quality scorer, not just the detector.
    checkpoint, checkpoint_audit = load_retained_checkpoint(model, args.checkpoint)
    if (classification_manifest is not None
            and state_sha256(model.backbone) != classification_manifest['backbone_sha256']):
        raise ValueError('Detection backbone differs from the frozen classification backbone')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.quality_collect = False
    model = MMDataParallel(model.cuda(), device_ids=[0])
    selected = (np.flatnonzero(classification.argmax(1) == 2).tolist() if args.cls_filter
                else list(range(len(dataset))))
    selected_image_ids = {image_ids[position] for position in selected}
    print('[ONLINE-CLASS-FILTER] enabled=%s total=%d detectTB=%d skipNonTB=%d' %
          (args.cls_filter, len(dataset), len(selected), len(dataset)-len(selected)))
    if args.cls_filter:
        from torch.utils.data import Subset
        inference_dataset = Subset(dataset, selected)
    else:
        inference_dataset = dataset
    outputs = [[np.empty((0, 5), np.float32) for _ in dataset.CLASSES]
               for _ in range(len(dataset))]
    if selected:
        loader = build_dataloader(inference_dataset, samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
        inferred = single_gpu_test(model, loader)
        if len(inferred) != len(selected):
            raise RuntimeError('Missing detection results for TB-selected images')
        for position, result in zip(selected, inferred):
            outputs[position] = result
    if len(outputs) != len(image_ids):
        raise RuntimeError('Some test images were not evaluated')
    prefix = Path(args.json_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result_files, _ = dataset.format_results(outputs, jsonfile_prefix=str(prefix))
    detections = mmcv.load(result_files['bbox'])
    expected = set(image_ids)
    for detection in detections:
        if detection['image_id'] not in expected:
            raise ValueError('Prediction refers to an unknown image ID')
        if not all(math.isfinite(float(x)) for x in detection['bbox'] + [detection['score']]):
            raise ValueError('Nonfinite detection in submission')
        if detection['bbox'][2] < 0 or detection['bbox'][3] < 0:
            raise ValueError('Invalid COCO xywh geometry')
        if detection['category_id'] not in (1, 2):
            raise ValueError('Official submission requires Active=1 or Latent=2')
        if args.cls_filter and detection['image_id'] not in selected_image_ids:
            raise ValueError('Detection survived on a non-TB-selected image')
    classification_txt = args.classification_txt
    if external_manifest is not None:
        classification_txt = str(prefix) + '.txt'
        # Preserve the reference probability bytes; only the detector is ours.
        Path(classification_txt).write_bytes(Path(args.external_classification_txt).read_bytes())
        mmcv.dump(external_manifest, str(prefix) + '.classification.manifest.json')
    manifest = dict(mode=args.mode, images=len(image_ids), detections=len(detections),
                    annotation=str(annotation), checkpoint=args.checkpoint,
                    image_ids_in_order=image_ids, results=result_files,
                    classification_txt=classification_txt, cls_filter=args.cls_filter,
                    classification_source=('external_official_example' if external_manifest is not None
                                           else 'own_frozen_classifier' if classification_manifest is not None
                                           else 'none'),
                    experiment_role=('external_classifier_diagnostic' if external_manifest is not None
                                     else 'own_model_export'),
                    classification_txt_sha256=(external_manifest['txt_sha256']
                        if external_manifest is not None else classification_manifest.get('txt_sha256')
                        if classification_manifest is not None else None),
                    classification_counts=(np.bincount(classification.argmax(1), minlength=3).tolist()
                                           if classification is not None else None),
                    detected_image_count=len(selected), checkpoint_audit=checkpoint_audit,
                    note='GT-free predictions; JSON contains detection instances, not one row per image.')
    mmcv.dump(manifest, str(prefix) + '.detection.manifest.json')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'image_ids_in_order'}, indent=2))


if __name__ == '__main__':
    main()
