"""Export actual Healthy/Sick/TB probabilities in official test-index order."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from TQ_SymNet.classification import (CLASS_NAMES, FrozenBackboneSymClassifier,
    TBXImageDataset, file_sha256, load_frozen_backbone, state_sha256)


def main():
    from mmcv import Config, ProgressBar
    from mmcv.runner.checkpoint import _load_checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--annotation')
    parser.add_argument('--image-root')
    parser.add_argument('--backbone-checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    snapshot = _load_checkpoint(args.checkpoint, map_location='cpu')
    if tuple(snapshot['class_names']) != CLASS_NAMES:
        raise ValueError('Incompatible class order')
    cfg = Config(snapshot['config'])
    source = args.backbone_checkpoint or snapshot['backbone_checkpoint']
    if file_sha256(source) != snapshot['backbone_checkpoint_sha256']:
        raise ValueError('Backbone source checkpoint differs from classification training')
    backbone, audit = load_frozen_backbone(cfg.backbone, source)
    if state_sha256(backbone) != snapshot['backbone_state_sha256']:
        raise ValueError('Backbone tensors differ from classification training')
    model = FrozenBackboneSymClassifier(backbone, cfg.input_channels)
    model.classifier.load_state_dict(snapshot['classifier_state_dict'], strict=True)
    model.cuda().eval()
    annotation = args.annotation or cfg.data.test_annotation
    index = json.loads(Path(annotation).read_text(encoding='utf-8'))
    if index.get('annotations'):
        raise ValueError('Expected hidden-test image index without GT')
    data = TBXImageDataset(annotation, args.image_root or cfg.data.image_root, training=False, labeled=False,
                          image_size=cfg.data.image_size, mean=cfg.data.mean, std=cfg.data.std)
    if len(data) != cfg.data.expected_test_images:
        raise ValueError('Official submission expects %d images, got %d' % (cfg.data.expected_test_images, len(data)))
    loader = DataLoader(data, batch_size=args.batch_size or cfg.data.val_batch_size,
                        shuffle=False, num_workers=cfg.data.workers, pin_memory=True)
    values, positions = [], []
    progress = ProgressBar(len(data))
    with torch.no_grad():
        for images, _, ids in loader:
            values.append(model(images.cuda(non_blocking=True)).softmax(1).cpu().numpy())
            positions.extend(ids.tolist())
            for _ in ids:
                progress.update()
    if positions != list(range(len(data))):
        raise ValueError('Test-order mismatch')
    values = np.concatenate(values)
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1., atol=1e-6):
        raise ValueError('Invalid probabilities')
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, fmt='%.12g')
    manifest = dict(class_order=CLASS_NAMES, rows=len(data), annotation=str(annotation),
                    annotation_sha256=file_sha256(annotation), image_ids=data.image_ids,
                    filenames=data.filenames, classification_checkpoint=args.checkpoint,
                    classification_checkpoint_sha256=file_sha256(args.checkpoint),
                    backbone_checkpoint=str(source), backbone_sha256=snapshot['backbone_state_sha256'],
                    validation_epoch=snapshot['epoch'], validation_metrics=snapshot['metrics'],
                    checkpoint_audit=audit, txt_sha256=file_sha256(path),
                    note='Classification probabilities only; bbox filtering is configured by export_online.py.')
    path.with_suffix('.classification.manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print('Exported %d rows: Healthy Sick_nonTB TB -> %s' % (len(data), path))


if __name__ == '__main__':
    main()
