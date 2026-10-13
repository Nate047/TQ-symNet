"""Train a classification head on strictly frozen detector features."""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from TQ_SymNet.classification import (CLASS_NAMES, TBXImageDataset,
    build_frozen_classifier, classification_metrics, file_sha256, state_sha256)


def seed_worker(worker_id):
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)


def dataset(cfg, split, training):
    data = cfg.data
    result = TBXImageDataset(data[split + '_annotation'], data.image_root, training=training,
                            image_size=data.image_size, mean=data.mean, std=data.std,
                            flip_probability=data.flip_probability)
    expected = data.get('expected_' + split + '_images')
    if expected is not None and len(result) != expected:
        raise ValueError('%s split: expected %d images, got %d' % (split, expected, len(result)))
    return result


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    probabilities, labels = [], []
    debug_samples = []
    total_loss, weight_sum = 0., 0.
    for batch_index, (images, target, _) in enumerate(loader):
        if hasattr(model.classifier, 'collect_debug'):
            model.classifier.collect_debug = (batch_index % max(1, len(loader) // 8) == 0
                                              or batch_index == len(loader) - 1)
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        logits = model(images)
        batch_weight = float(criterion.weight[target].sum())
        total_loss += float(criterion(logits, target)) * batch_weight
        weight_sum += batch_weight
        probabilities.append(logits.softmax(1).cpu().numpy())
        labels.append(target.cpu().numpy())
        if getattr(model.classifier, 'last_debug', None):
            debug_samples.append(model.classifier.last_debug)
    result = classification_metrics(np.concatenate(labels), np.concatenate(probabilities))
    result['loss'] = total_loss / weight_sum
    if debug_samples:
        result['readout_debug'] = dict(
            sampled_batches=len(debug_samples),
            **{key: np.asarray([item[key] for item in debug_samples],
                               dtype=np.float64).mean(axis=0).tolist()
               for key in debug_samples[0]})
    return result


def main():
    from mmcv import Config
    from mmdet.apis import set_random_seed
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--backbone-checkpoint')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.backbone_checkpoint:
        cfg.backbone_checkpoint = args.backbone_checkpoint
    if cfg.optimizer.type != 'SGD':
        raise ValueError('This official-reference run uses SGD')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing silent CPU backbone inference')
    set_random_seed(args.seed, deterministic=args.deterministic)
    out = Path(args.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'latest.pth').exists():
        raise FileExistsError('Choose a fresh work-dir; an existing training run is present')
    cfg.dump(str(out / 'resolved_config.py'))
    logger = logging.getLogger('tq_classification')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(out / 'training.log', encoding='utf-8')):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    train, val = dataset(cfg, 'train', True), dataset(cfg, 'val', False)
    train.validate_image_files()
    val.validate_image_files()
    # Inspect the hidden-test index for overlap only; never load its GT/images
    # or use its predictions to select a training epoch.
    test_index = TBXImageDataset(cfg.data.test_annotation, cfg.data.image_root, labeled=False)
    if len(test_index) != cfg.data.expected_test_images:
        raise ValueError('Unexpected hidden-test image count: %d' % len(test_index))
    if json.loads(Path(cfg.data.test_annotation).read_text(encoding='utf-8')).get('annotations'):
        raise ValueError('Expected hidden-test image index without GT')
    indices = {'train': set(train.filenames), 'val': set(val.filenames), 'test': set(test_index.filenames)}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = indices[left] & indices[right]
        if overlap:
            raise ValueError('%s/%s filename overlap: %s' % (left, right, sorted(overlap)[:10]))
    data_audit = {
        split: dict(annotation=str(path), sha256=file_sha256(path),
                    images=len(data), sources=data.source_counts())
        for split, path, data in (
            ('train', cfg.data.train_annotation, train),
            ('val', cfg.data.val_annotation, val),
            ('test', cfg.data.test_annotation, test_index))}
    (out / 'data_audit.json').write_text(json.dumps(data_audit, indent=2), encoding='utf-8')
    logger.info('[CLS-DATA-AUDIT] filenameDisjoint=True trainOnly=True selectionSplit=val '
                'selectionMetric=%s hiddenTest=exportOnly audit=%s', cfg.selection_metric, data_audit)
    generators = [torch.Generator().manual_seed(args.seed + offset) for offset in (0, 1)]
    loaders = [DataLoader(data, batch_size=size, shuffle=shuffle, num_workers=cfg.data.workers,
                          pin_memory=True, worker_init_fn=seed_worker, generator=generator)
               for data, size, shuffle, generator in zip((train, val),
                   (cfg.data.batch_size, cfg.data.val_batch_size), (True, False), generators)]
    model, audit = build_frozen_classifier(cfg, cfg.backbone_checkpoint)
    model = model.cuda()
    frozen_hash = state_sha256(model.backbone)
    frozen_neck_hash = (state_sha256(model.neck)
                        if getattr(model, 'neck', None) is not None else None)
    source_hash = file_sha256(cfg.backbone_checkpoint)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=cfg.optimizer.lr,
                               momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.training.class_weights, device='cuda'))
    logger.info('[CLS-BUILD] head=%s backboneFrozen=True neckFrozen=%s '
                'frozenEval=True AMP=False updates=onePerBatch source=%s audit=%s',
                cfg.get('classification_head', 'sym'), frozen_neck_hash is not None,
                cfg.backbone_checkpoint, audit)
    logger.info('[CLS-DATA] train=%d counts=%s val=%d counts=%s classOrder=%s weight=%s',
                len(train), np.bincount(train.labels, minlength=3).tolist(), len(val),
                np.bincount(val.labels, minlength=3).tolist(), CLASS_NAMES, cfg.training.class_weights)
    logger.info('[CLS-FROZEN] initialBackboneSHA256=%s initialNeckSHA256=%s sourceSHA256=%s',
                frozen_hash, frozen_neck_hash, source_hash)
    start = time.perf_counter()
    steps, best = 0, float('-inf')
    total_steps = len(loaders[0]) * cfg.training.epochs
    for epoch in range(cfg.training.epochs):
        model.train()
        if hasattr(model.classifier, 'collect_debug'):
            model.classifier.collect_debug = False
        loss_sum, weight_sum, correct, seen = 0., 0., 0, 0
        for iteration, (images, target, _) in enumerate(loaders[0], 1):
            lr = cfg.optimizer.lr * cfg.training.lr_gamma ** sum(epoch >= step for step in cfg.training.lr_steps)
            if steps < cfg.training.warmup_iters:
                lr *= cfg.training.warmup_ratio + (1 - cfg.training.warmup_ratio) * steps / cfg.training.warmup_iters
            for group in optimizer.param_groups:
                group['lr'] = lr
            images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite classification loss')
            loss.backward()
            optimizer.step()
            steps += 1
            batch_weight = float(criterion.weight[target].sum())
            loss_sum += float(loss.detach()) * batch_weight
            weight_sum += batch_weight
            correct += int((logits.argmax(1) == target).sum())
            seen += len(target)
            if iteration % cfg.log_interval == 0 or iteration == len(loaders[0]):
                elapsed = time.perf_counter() - start
                logger.info('Epoch [%d][%d/%d] lr=%.6g loss=%.5f trainAcc=%.4f '
                            'updates=%d ETA=%.1fmin memory=%dMiB', epoch+1, iteration, len(loaders[0]),
                            lr, loss_sum/weight_sum, correct/seen, steps,
                            elapsed/max(steps, 1)*(total_steps-steps)/60,
                            torch.cuda.max_memory_allocated() // 1024**2)
        metrics = evaluate(model, loaders[1], 'cuda', criterion)
        after_hash = state_sha256(model.backbone)
        after_neck_hash = (state_sha256(model.neck)
                           if frozen_neck_hash is not None else None)
        if after_hash != frozen_hash or model.backbone.training or any(
                p.requires_grad or p.grad is not None for p in model.backbone.parameters()):
            raise RuntimeError('Frozen backbone contract violated')
        if frozen_neck_hash is not None and (after_neck_hash != frozen_neck_hash
                or model.neck.training or any(p.requires_grad or p.grad is not None
                                              for p in model.neck.parameters())):
            raise RuntimeError('Frozen CrossScale contract violated')
        logger.info('[CLS-FROZEN] epoch=%d unchanged=True backboneSHA256=%s neckSHA256=%s',
                    epoch+1, after_hash, after_neck_hash)
        debug = metrics.get('readout_debug')
        if debug:
            logger.info('[CLS-READOUT] epoch=%d head=%s sampledVal=%s',
                        epoch+1, cfg.get('classification_head', 'sym'), debug)
        logger.info('[CLS-VAL] epoch=%d Acc=%.5f TB_AUC=%.5f Sensitivity=%.5f Specificity=%.5f '
                    'macroAP=%.5f macroRecall=%.5f loss=%.5f confusion=%s', epoch+1,
                    metrics['accuracy'], metrics['tb_auc'], metrics['tb_sensitivity'],
                    metrics['tb_specificity'], metrics['macro_ap'], metrics['macro_recall'],
                    metrics['loss'], metrics['confusion_matrix'])
        record = dict(epoch=epoch+1, updates=steps, metrics=metrics, backbone_sha256=after_hash)
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        snapshot = dict(classifier_state_dict={k:v.detach().cpu() for k,v in model.classifier.state_dict().items()},
                        config=cfg.to_dict(), epoch=epoch+1, seed=args.seed, metrics=metrics,
                        backbone_checkpoint=str(cfg.backbone_checkpoint), backbone_checkpoint_sha256=source_hash,
                        backbone_state_sha256=frozen_hash, class_names=CLASS_NAMES,
                        neck_state_sha256=frozen_neck_hash,
                        classification_head=cfg.get('classification_head', 'sym'),
                        data_audit=data_audit, selection_metric=cfg.selection_metric,
                        optimizer=optimizer.state_dict(), updates=steps)
        torch.save(snapshot, out / 'latest.pth')
        key = metrics[cfg.selection_metric]
        if key > best:
            best = key
            torch.save(snapshot, out / 'best_cls_auc.pth')
            (out / 'best_metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            logger.info('[CLS-BEST] epoch=%d %s=%.6f', epoch+1, cfg.selection_metric, best)
    logger.info('Finished. Best %s=%.6f checkpoint=%s', cfg.selection_metric, best, out/'best_cls_auc.pth')


if __name__ == '__main__':
    main()
