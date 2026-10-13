"""Train only the official Sym image classifier; validate all three image classes."""
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
from TQ_SymNet.classification import (CLASS_NAMES, FrozenBackboneSymClassifier,
    TBXImageDataset, classification_metrics, file_sha256, load_frozen_backbone, state_sha256)


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
    total_loss, weight_sum = 0., 0.
    for images, target, _ in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        logits = model(images)
        batch_weight = float(criterion.weight[target].sum())
        total_loss += float(criterion(logits, target)) * batch_weight
        weight_sum += batch_weight
        probabilities.append(logits.softmax(1).cpu().numpy())
        labels.append(target.cpu().numpy())
    result = classification_metrics(np.concatenate(labels), np.concatenate(probabilities))
    result['loss'] = total_loss / weight_sum
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
    overlap = set(train.filenames) & set(val.filenames)
    if overlap:
        raise ValueError('Train/val filename overlap: %s' % sorted(overlap)[:10])
    generators = [torch.Generator().manual_seed(args.seed + offset) for offset in (0, 1)]
    loaders = [DataLoader(data, batch_size=size, shuffle=shuffle, num_workers=cfg.data.workers,
                          pin_memory=True, worker_init_fn=seed_worker, generator=generator)
               for data, size, shuffle, generator in zip((train, val),
                   (cfg.data.batch_size, cfg.data.val_batch_size), (True, False), generators)]
    backbone, audit = load_frozen_backbone(cfg.backbone, cfg.backbone_checkpoint)
    model = FrozenBackboneSymClassifier(backbone, cfg.input_channels).cuda()
    frozen_hash = state_sha256(model.backbone)
    source_hash = file_sha256(cfg.backbone_checkpoint)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=cfg.optimizer.lr,
                               momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.training.class_weights, device='cuda'))
    logger.info('[CLS-BUILD] officialSymConv=True input=%d hidden=512 backboneFrozen=True '
                'backboneEval=True AMP=False updates=onePerBatch source=%s audit=%s',
                cfg.input_channels, cfg.backbone_checkpoint, audit)
    logger.info('[CLS-DATA] train=%d counts=%s val=%d counts=%s classOrder=%s weight=%s',
                len(train), np.bincount(train.labels, minlength=3).tolist(), len(val),
                np.bincount(val.labels, minlength=3).tolist(), CLASS_NAMES, cfg.training.class_weights)
    logger.info('[CLS-FROZEN] initialBackboneSHA256=%s sourceSHA256=%s', frozen_hash, source_hash)
    start = time.perf_counter()
    steps, best = 0, float('-inf')
    total_steps = len(loaders[0]) * cfg.training.epochs
    for epoch in range(cfg.training.epochs):
        model.train()
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
        if after_hash != frozen_hash or model.backbone.training or any(
                p.requires_grad or p.grad is not None for p in model.backbone.parameters()):
            raise RuntimeError('Frozen backbone contract violated')
        logger.info('[CLS-FROZEN] epoch=%d unchanged=True SHA256=%s', epoch+1, after_hash)
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
