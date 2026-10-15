"""Validation-only image diagnostics. IDs always accompanied by filenames."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

NAMES = ('Healthy', 'Sick', 'TB')


def accuracy_key(metrics):
    return tuple(float(metrics[name]) for name in
                 ('accuracy', 'tb_sensitivity', 'tb_specificity'))


def error_type(label, prediction):
    return 'correct' if label == prediction else NAMES[label] + '->' + NAMES[prediction]


def export_diagnostics(folder, dataset, evidence, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(dataset.labels)
    probabilities = evidence['probabilities']
    logits = evidence['logits']
    if probabilities.shape != (len(labels), 3) or logits.shape != probabilities.shape:
        raise ValueError('Diagnostic alignment/shape mismatch')
    if not np.isfinite(probabilities).all() or not np.isfinite(logits).all():
        raise ValueError('Nonfinite diagnostic outputs')
    predictions = probabilities.argmax(1)
    rows = []
    for i, (filename, image_id, label, prediction) in enumerate(zip(
            dataset.filenames, dataset.image_ids, labels, predictions)):
        row = dict(image_id=image_id, file_name=filename,
                   source=filename.split('/')[1] if filename.startswith('extra/') else 'TBX11K',
                   gt=NAMES[label], prediction=NAMES[prediction],
                   error_type=error_type(label, prediction),
                   tb_margin=float(logits[i, 2] - max(logits[i, :2])))
        for c, name in enumerate(NAMES):
            row['logit_' + name] = float(logits[i, c])
            row['p_' + name] = float(probabilities[i, c])
        for branch, branch_logits in evidence.get('counterfactual_logits', {}).items():
            values = branch_logits[i]
            p = np.exp(values - values.max())
            p /= p.sum()
            other_prediction = int(p.argmax())
            row['without_' + branch + '_prediction'] = NAMES[other_prediction]
            row['without_' + branch + '_pTB'] = float(p[2])
            row['without_' + branch + '_error'] = error_type(label, other_prediction)
            row[branch + '_helped'] = bool(prediction == label and other_prediction != label)
            row[branch + '_hurt'] = bool(prediction != label and other_prediction == label)
        if 'branch_gates' in evidence:
            for c, name in enumerate(NAMES):
                for b, branch in enumerate(evidence['branch_names']):
                    row['gate_' + name + '_' + branch] = float(evidence['branch_gates'][i, c, b])
        rows.append(row)
    fields = list(rows[0])
    subsets = {
        'all_predictions': rows,
        'all_errors': [r for r in rows if r['error_type'] != 'correct'],
        'tb_false_negative': [r for r in rows if r['gt'] == 'TB' and r['prediction'] != 'TB'],
        'tb_false_positive': [r for r in rows if r['gt'] != 'TB' and r['prediction'] == 'TB']}
    for name, selected in subsets.items():
        with (folder / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    with gzip.open(str(folder / 'all_predictions.json.gz'), 'wt', encoding='utf-8') as stream:
        json.dump(dict(metadata=metadata, rows=rows), stream, ensure_ascii=False)
    np.savez_compressed(folder / 'predictions.npz', logits=logits, probabilities=probabilities,
                        labels=labels, image_ids=np.asarray(dataset.image_ids),
                        filenames=np.asarray(dataset.filenames))
    summaries = {}
    for source in ['ALL'] + sorted(set(r['source'] for r in rows)):
        group = rows if source == 'ALL' else [r for r in rows if r['source'] == source]
        tp = sum(r['gt'] == 'TB' and r['prediction'] == 'TB' for r in group)
        fn = sum(r['gt'] == 'TB' and r['prediction'] != 'TB' for r in group)
        fp = sum(r['gt'] != 'TB' and r['prediction'] == 'TB' for r in group)
        tn = len(group) - tp - fn - fp
        summaries[source] = dict(images=len(group), tp=tp, fn=fn, fp=fp, tn=tn,
            accuracy=sum(r['error_type'] == 'correct' for r in group)/len(group),
            sensitivity=tp/(tp+fn) if tp+fn else None,
            specificity=tn/(tn+fp) if tn+fp else None)
    dependency = {branch: dict(helped=sum(r[branch+'_helped'] for r in rows),
                               hurt=sum(r[branch+'_hurt'] for r in rows))
                  for branch in evidence.get('counterfactual_logits', {})}
    report = dict(metadata=metadata, sources=summaries, branch_dependency=dependency,
                  note='Inference branch removal with remaining gates renormalized; not a retraining ablation.')
    (folder / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if evidence.get('attention_examples'):
        arrays, index = {}, []
        for number, sample in enumerate(evidence['attention_examples']):
            sample = dict(sample)
            maps = sample.pop('maps')
            sample['keys'] = []
            for name, value in maps.items():
                key = 'sample_%03d_%s' % (number, name)
                arrays[key] = value
                sample['keys'].append(key)
            index.append(sample)
        np.savez_compressed(folder / 'attention_examples.npz', **arrays)
        (folder / 'attention_examples.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    return report


def summarize_history(root):
    files = sorted(Path(root).glob('epoch_*/predictions.npz'))
    if not files:
        return
    history, reference = [], None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            names, labels = data['filenames'].copy(), data['labels'].copy()
            if reference is None:
                reference = (names, labels)
            elif not (np.array_equal(names, reference[0]) and np.array_equal(labels, reference[1])):
                raise ValueError('History validation index changed')
            history.append(data['probabilities'].argmax(1))
    predictions = np.stack(history)
    names, labels = reference
    with (Path(root) / 'error_history.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['file_name', 'gt', 'sampled_epochs', 'errors', 'tb_fn_count', 'tb_fp_count',
                         'prediction_flips', 'wrong_in_all_sampled_epochs'])
        for i, name in enumerate(names):
            pred = predictions[:, i]
            errors = int((pred != labels[i]).sum())
            writer.writerow([name, NAMES[labels[i]], len(files), errors,
                             int(((pred != 2) & (labels[i] == 2)).sum()),
                             int(((pred == 2) & (labels[i] != 2)).sum()),
                             int((pred[1:] != pred[:-1]).sum()), errors == len(files)])
