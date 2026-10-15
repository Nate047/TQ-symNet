"""Train an independent classifier with audited encoder freezing/fine-tuning."""
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
    build_frozen_classifier, classification_metrics, file_sha256, state_sha256,
    encoder_policy, classification_parameter_groups, restore_classification_snapshot)
from TQ_SymNet.classification_consistency import paired_view_logits, consistency_kl
from TQ_SymNet.classification_diagnostics import export_diagnostics, summarize_history, accuracy_key


def seed_worker(worker_id):
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)


def update_batches(loader, accumulation):
    """Group microbatches without retaining their GPU activation graphs."""
    pending = []
    for batch in loader:
        pending.append(batch)
        if len(pending) == accumulation:
            yield pending
            pending = []
    if pending:
        yield pending


def audit_encoders(model, initial_hashes):
    policy, hashes = encoder_policy(model), {}
    for name, trainable in policy.items():
        module = getattr(model, name, None)
        if module is None:
            hashes[name] = None
            continue
        hashes[name] = state_sha256(module)
        if not trainable and (hashes[name] != initial_hashes[name] or module.training or any(
                p.requires_grad or p.grad is not None for p in module.parameters())):
            raise RuntimeError('Frozen encoder contract violated: ' + name)
        if trainable and not all(p.requires_grad for p in module.parameters()):
            raise RuntimeError('Partial encoder freeze differs from configured policy: ' + name)
    return hashes


def dataset(cfg, split, training):
    data = cfg.data
    result = TBXImageDataset(data[split + '_annotation'], data.image_root, training=training,
                            image_size=data.image_size, mean=data.mean, std=data.std,
                            flip_probability=data.flip_probability, clahe=data.get('clahe'))
    expected = data.get('expected_' + split + '_images')
    if expected is not None and len(result) != expected:
        raise ValueError('%s split: expected %d images, got %d' % (split, expected, len(result)))
    return result


@torch.no_grad()
def evaluate(model, loader, device, criterion, detailed=False, branch_diagnostics=False):
    model.eval()
    probabilities, labels, positions = [], [], []
    debug_samples = []
    all_logits, gates, alternatives, examples, example_counts = [], [], {}, [], {}
    if hasattr(model.classifier, 'capture_diagnostics'):
        model.classifier.capture_diagnostics = branch_diagnostics
    total_loss, weight_sum = 0., 0.
    for batch_index, (images, target, indices) in enumerate(loader):
        if hasattr(model.classifier, 'collect_debug'):
            model.classifier.collect_debug = (batch_index % max(1, len(loader) // 8) == 0
                                              or batch_index == len(loader) - 1)
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        logits = model(images)
        batch_weight = float(criterion.weight[target].sum())
        total_loss += float(criterion(logits, target)) * batch_weight
        weight_sum += batch_weight
        probabilities.append(logits.softmax(1).cpu().numpy())
        if detailed:
            all_logits.append(logits.cpu().numpy())
        labels.append(target.cpu().numpy())
        positions.extend(indices.tolist())
        if getattr(model.classifier, 'last_debug', None):
            debug_samples.append(model.classifier.last_debug)
        if branch_diagnostics and getattr(model.classifier, 'image_diagnostics', None):
            diagnostic = model.classifier.image_diagnostics
            gates.append(diagnostic['branch_gates'])
            for name, values in diagnostic['counterfactual_logits'].items():
                alternatives.setdefault(name, []).append(values)
            prediction = logits.argmax(1).cpu().tolist()
            for j, (idx, gt, pred) in enumerate(zip(indices.tolist(), target.cpu().tolist(), prediction)):
                # Bounded, deterministic examples: up to 8 per true/predicted
                # class pair, including matched correctly classified classes.
                kind = (gt, pred)
                if example_counts.get(kind, 0) < 8:
                    examples.append(dict(image_id=loader.dataset.image_ids[idx],
                        file_name=loader.dataset.filenames[idx], gt=gt, prediction=pred,
                        maps={name: value[j] for name, value in model.classifier.attention_maps.items()}))
                    example_counts[kind] = example_counts.get(kind, 0) + 1
    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    if positions != list(range(len(loader.dataset))):
        raise ValueError('Validation output ordering mismatch')
    result = classification_metrics(labels, probabilities)
    internal = np.asarray([not name.startswith('extra/') for name in loader.dataset.filenames])
    if internal.any() and set(labels[internal].tolist()) == {0, 1, 2}:
        result['tbx11k_only'] = classification_metrics(labels[internal], probabilities[internal])
    result['loss'] = total_loss / weight_sum
    if debug_samples:
        result['readout_debug'] = dict(
            sampled_batches=len(debug_samples),
            **{key: np.asarray([item[key] for item in debug_samples],
                               dtype=np.float64).mean(axis=0).tolist()
               for key in debug_samples[0]})
    evidence = dict(probabilities=probabilities)
    if detailed:
        evidence['logits'] = np.concatenate(all_logits)
    if gates:
        evidence.update(branch_gates=np.concatenate(gates),
                        branch_names=model.classifier.branch_names,
                        counterfactual_logits={name: np.concatenate(values) for name, values in alternatives.items()},
                        attention_examples=examples)
    if hasattr(model.classifier, 'capture_diagnostics'):
        model.classifier.capture_diagnostics = False
    return result, evidence


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
    interval = int(cfg.get('diagnostic_interval', 2))
    accumulation = int(cfg.training.get('accumulation_steps', 1))
    if accumulation < 1:
        raise ValueError('accumulation_steps must be positive')
    if interval < 1 or cfg.selection_metric not in ('accuracy', 'tb_auc'):
        raise ValueError('Require positive diagnostic_interval and accuracy/tb_auc selection')
    consistency = cfg.get('consistency', None)
    consistency_weight = float(consistency.weight) if consistency is not None else 0.
    if consistency_weight < 0:
        raise ValueError('consistency.weight must be non-negative')
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
    policy = encoder_policy(model)
    initial_hashes = dict(backbone=frozen_hash, neck=frozen_neck_hash)
    optimizer = torch.optim.SGD(classification_parameter_groups(model, cfg.optimizer), lr=cfg.optimizer.lr,
                               momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.training.class_weights, device='cuda'))
    consistency_generator = (torch.Generator(device='cuda').manual_seed(args.seed + 1000003)
                             if consistency_weight else None)
    logger.info('[CLS-BUILD] head=%s backboneFrozen=%s neckFrozen=%s '
                'AMP=False microBatch=%d accumulation=%d effectiveBatch=%d '
                'consistency=%.3f source=%s audit=%s',
                cfg.get('classification_head', 'sym'), not policy['backbone'], not policy['neck'],
                cfg.data.batch_size, accumulation, cfg.data.batch_size * accumulation,
                consistency_weight, cfg.backbone_checkpoint, audit)
    logger.info('[CLS-OPTIMIZER] groups=%s', [dict(name=g['name'], lr_mult=g['lr_mult'],
                parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups])
    logger.info('[CLS-ENCODER] initial policy=%s classifierSHA256=%s', policy, state_sha256(model.classifier))
    if consistency_weight:
        logger.info('[CLS-CONSISTENCY] pixelBrightness=+/-%.2f contrast=+/-%.3f '
                    'weight=%.3f sharedDropout=True isolatedRNG=True seed=%d',
                    consistency.brightness_pixels, consistency.contrast,
                    consistency_weight, args.seed + 1000003)
    logger.info('[CLS-DATA] train=%d counts=%s val=%d counts=%s classOrder=%s weight=%s',
                len(train), np.bincount(train.labels, minlength=3).tolist(), len(val),
                np.bincount(val.labels, minlength=3).tolist(), CLASS_NAMES, cfg.training.class_weights)
    logger.info('[CLS-PRIORS] options=%s clahe=%s branches=%s',
                cfg.get('classification_options', {}), cfg.data.get('clahe'),
                getattr(model.classifier, 'branch_names', []))
    logger.info('[CLS-FROZEN] initialBackboneSHA256=%s initialNeckSHA256=%s sourceSHA256=%s',
                frozen_hash, frozen_neck_hash, source_hash)
    start = time.perf_counter()
    steps, best, best_accuracy = 0, float('-inf'), (float('-inf'),)*3
    best_auc = float('-inf')
    primary_filename = 'best_cls_accuracy.pth' if cfg.selection_metric == 'accuracy' else 'best_cls_auc.pth'
    updates_per_epoch = (len(loaders[0]) + accumulation - 1) // accumulation
    total_steps = updates_per_epoch * cfg.training.epochs
    class_weights_cpu = torch.tensor(cfg.training.class_weights)
    for epoch in range(cfg.training.epochs):
        model.train()
        if hasattr(model.classifier, 'collect_debug'):
            model.classifier.collect_debug = False
        loss_sum, ce_sum, consistency_sum = 0., 0., 0.
        weight_sum, correct, seen = 0., 0, 0
        for iteration, microbatches in enumerate(update_batches(loaders[0], accumulation), 1):
            lr = cfg.optimizer.lr * cfg.training.lr_gamma ** sum(epoch >= step for step in cfg.training.lr_steps)
            if steps < cfg.training.warmup_iters:
                lr *= cfg.training.warmup_ratio + (1 - cfg.training.warmup_ratio) * steps / cfg.training.warmup_iters
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_mult']
            optimizer.zero_grad(set_to_none=True)
            # Weighted CE mean over the WHOLE effective batch, not an average
            # of separately normalized microbatch means with different labels.
            window_weight = sum(float(class_weights_cpu[target].sum()) for _, target, _ in microbatches)
            for images, target, _ in microbatches:
                batch_weight = float(class_weights_cpu[target].sum())
                images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
                if consistency_weight:
                    logits, augmented_logits = paired_view_logits(
                        model, images, cfg.data.mean, cfg.data.std,
                        brightness_pixels=float(consistency.brightness_pixels),
                        contrast=float(consistency.contrast), generator=consistency_generator)
                    supervised_loss = criterion(logits, target)
                    consistency_loss = consistency_kl(augmented_logits, logits)
                else:
                    logits = model(images)
                    supervised_loss = criterion(logits, target)
                    consistency_loss = supervised_loss.new_zeros(())
                loss = supervised_loss + consistency_weight * consistency_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite classification loss')
                (loss * (batch_weight / window_weight)).backward()
                loss_sum += float(loss.detach()) * batch_weight
                ce_sum += float(supervised_loss.detach()) * batch_weight
                consistency_sum += float(consistency_loss.detach()) * batch_weight
                weight_sum += batch_weight
                correct += int((logits.argmax(1) == target).sum())
                seen += len(target)
            if iteration == 1 or iteration % cfg.log_interval == 0:
                grad_norms = {}
                monitored = {name:getattr(model, name, None) for name in ('classifier', 'backbone', 'neck')}
                monitored.update({name:getattr(model.classifier, name, None) for name in ('sym', 'lbp', 'gabor')})
                for name, module in monitored.items():
                    gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None] if module is not None else []
                    norm = torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt() if gradients else None
                    if norm is not None and not torch.isfinite(norm):
                        raise RuntimeError('Nonfinite gradient in ' + name)
                    grad_norms[name] = float(norm) if norm is not None else 0.
                    if policy.get(name, False) and norm is None:
                        raise RuntimeError('Trainable encoder has no gradients: ' + name)
                logger.info('[CLS-GRAD] epoch=%d update=%d norms=%s', epoch+1, iteration, grad_norms)
            optimizer.step()
            steps += 1
            if iteration % cfg.log_interval == 0 or iteration == updates_per_epoch:
                elapsed = time.perf_counter() - start
                logger.info('Epoch [%d][%d/%d] lr=%.6g loss=%.5f '
                            'lossCE=%.5f lossConsistency=%.5f trainAcc=%.4f '
                            'updates=%d ETA=%.1fmin memory=%dMiB', epoch+1, iteration, updates_per_epoch,
                            lr, loss_sum/weight_sum, ce_sum/weight_sum,
                            consistency_sum/weight_sum, correct/seen, steps,
                            elapsed/max(steps, 1)*(total_steps-steps)/60,
                            torch.cuda.max_memory_allocated() // 1024**2)
        detailed = (epoch+1) % interval == 0
        metrics, evidence = evaluate(model, loaders[1], 'cuda', criterion, detailed=detailed)
        hashes = audit_encoders(model, initial_hashes)
        after_hash, after_neck_hash = hashes['backbone'], hashes['neck']
        logger.info('[CLS-ENCODER] epoch=%d policy=%s changed=%s backboneSHA256=%s neckSHA256=%s',
                    epoch+1, policy, {k: hashes[k] != initial_hashes[k] for k in hashes},
                    after_hash, after_neck_hash)
        if not any(policy.values()):
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
        if 'tbx11k_only' in metrics:
            logger.info('[CLS-VAL-TBX11K] epoch=%d metrics=%s', epoch+1, metrics['tbx11k_only'])
        record = dict(epoch=epoch+1, updates=steps, metrics=metrics, backbone_sha256=after_hash)
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        snapshot = dict(classifier_state_dict={k:v.detach().cpu() for k,v in model.classifier.state_dict().items()},
                        config=cfg.to_dict(), epoch=epoch+1, seed=args.seed, metrics=metrics,
                        backbone_checkpoint=str(cfg.backbone_checkpoint), backbone_checkpoint_sha256=source_hash,
                        backbone_state_sha256=after_hash, class_names=CLASS_NAMES,
                        neck_state_sha256=after_neck_hash, encoder_trainability=policy,
                        initial_backbone_state_sha256=frozen_hash, initial_neck_state_sha256=frozen_neck_hash,
                        classification_head=cfg.get('classification_head', 'sym'),
                        data_audit=data_audit, selection_metric=cfg.selection_metric,
                        optimizer=optimizer.state_dict(), updates=steps)
        if any(policy.values()) or cfg.get('save_encoder', False):
            snapshot['backbone_state_dict'] = {k:v.detach().cpu() for k,v in model.backbone.state_dict().items()}
            if getattr(model, 'neck', None) is not None:
                snapshot['neck_state_dict'] = {k:v.detach().cpu() for k,v in model.neck.state_dict().items()}
        torch.save(snapshot, out / 'latest.pth')
        metadata = dict(epoch=epoch+1, seed=args.seed, config_sha256=file_sha256(out/'resolved_config.py'),
                        classifier_state_sha256=state_sha256(model.classifier),
                        backbone_state_sha256=after_hash, neck_state_sha256=after_neck_hash,
                        encoder_trainability=policy,
                        detector_checkpoint_sha256=source_hash, val_annotation_sha256=data_audit['val']['sha256'])
        if detailed:
            report = export_diagnostics(out/'diagnostics'/('epoch_%02d' % (epoch+1)), val, evidence, metadata)
            summarize_history(out/'diagnostics')
            logger.info('[CLS-DIAGNOSTICS] epoch=%d sources=%s', epoch+1, report['sources'])
        improved_accuracy = accuracy_key(metrics) > best_accuracy
        if improved_accuracy:
            best_accuracy = accuracy_key(metrics)
            torch.save(snapshot, out / 'best_cls_accuracy.pth')
        improved_auc = metrics['tb_auc'] > best_auc
        if improved_auc:
            best_auc = metrics['tb_auc']
            torch.save(snapshot, out / 'best_cls_auc.pth')
        improved = improved_accuracy if cfg.selection_metric == 'accuracy' else improved_auc
        if improved:
            best = metrics[cfg.selection_metric]
            (out / 'best_metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            logger.info('[CLS-BEST] epoch=%d %s=%.6f', epoch+1, cfg.selection_metric, best)
    # One final diagnostic pass on selected VAL-best weights, not on latest.
    from mmcv.runner.checkpoint import _load_checkpoint
    chosen = _load_checkpoint(str(out/primary_filename), map_location='cpu')
    restore_classification_snapshot(model, chosen)
    chosen_metrics, evidence = evaluate(model, loaders[1], 'cuda', criterion,
                                        detailed=True, branch_diagnostics=True)
    for name in ('accuracy', 'tb_sensitivity', 'tb_specificity'):
        if abs(chosen_metrics[name] - chosen['metrics'][name]) > 1e-8:
            raise RuntimeError('Selected checkpoint replay differs: ' + name)
    metadata.update(epoch=chosen['epoch'], classifier_state_sha256=state_sha256(model.classifier),
                    backbone_state_sha256=state_sha256(model.backbone),
                    neck_state_sha256=state_sha256(model.neck) if getattr(model, 'neck', None) is not None else None,
                    classification_checkpoint=primary_filename,
                    classification_checkpoint_sha256=file_sha256(out/primary_filename))
    report = export_diagnostics(out/'diagnostics'/'best', val, evidence, metadata)
    logger.info('[CLS-BEST-DIAGNOSTICS] epoch=%d sources=%s branchDependency=%s',
                chosen['epoch'], report['sources'], report['branch_dependency'])
    logger.info('Finished. Best %s=%.6f checkpoint=%s', cfg.selection_metric, best, out/primary_filename)


if __name__ == '__main__':
    main()
