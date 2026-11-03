"""Explicit, provenance-preserving paper tables; no inference or model changes.

All paths in the input manifest are relative to that manifest. Values in output
are percentages. Never select an epoch on test, or pool different cohort labels.
Python >= 3.8; standard library only.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


KEYS = {
    'ca_ap': 'official_tb_bbox_mAP',
    'ca_ap50': 'official_tb_bbox_mAP_50',
    'ca_ap75': 'official_tb_bbox_mAP_75',
    'active_ap': 'bbox_ActiveTuberculosis_mAP',
    'latent_ap': 'bbox_ObsoletePulmonaryTuberculosis_mAP',
}


def read_source(root, name):
    path = (root / name).resolve()
    raw = path.read_bytes()
    return raw.decode('utf-8-sig'), {
        'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def online_metrics(text):
    classification = None
    detection = None
    for line in text.splitlines():
        if not line.lstrip().startswith('|'):
            continue
        cells = [x.strip() for x in line.split('|') if x.strip()]
        try:
            values = [float(x) for x in cells]
        except ValueError:
            continue
        if len(values) == 6:
            if classification is not None:
                raise ValueError('Multiple classification tables: supply one submission per file')
            classification = dict(zip(
                ('accuracy', 'auc', 'sensitivity', 'specificity', 'cls_ap', 'cls_ar'), values))
        elif len(values) == 9:
            if detection is not None:
                raise ValueError('Multiple detection tables: supply one submission per file')
            detection = dict(zip(
                ('ca_ap50', 'ca_ap75', 'ca_ap', 'active_ap50', 'active_ap75',
                 'active_ap', 'latent_ap50', 'latent_ap75', 'latent_ap'), values))
    if classification is None or detection is None:
        raise ValueError('Missing platform summary table; export logs are NOT online metrics')
    return dict(classification, **detection)


def validation_metrics(text):
    rows = []
    for line in text.splitlines():
        match = re.search(r'Epoch\(val\)\s*\[(\d+)\]', line)
        if not match:
            continue
        values = {}
        for short, full in KEYS.items():
            found = re.search(r'(?<![\w])' + re.escape(full) + r':\s*([0-9.eE+\-]+)', line)
            if found:
                values[short] = 100. * float(found.group(1))
        if 'ca_ap' in values:
            rows.append(dict(epoch=int(match.group(1)), **values))
    if not rows:
        raise ValueError('No Epoch(val) official_tb_bbox_mAP entries')
    if len({r['epoch'] for r in rows}) != len(rows):
        raise ValueError('Duplicate epochs: split concatenated/restarted logs before collection')
    # This is logged precision, not necessarily the runner\'s full-precision best.
    best = max(rows, key=lambda r: (r['ca_ap'], -r['epoch']))
    return best, sorted(rows, key=lambda r: r['epoch'])


def collect(manifest, root):
    results, curves, audit, warnings = [], [], [], []
    names = set()
    for spec in manifest['runs']:
        name, cohort, kind = spec['name'], spec['cohort'], spec['kind']
        if name in names:
            raise ValueError('Duplicate run name: ' + name)
        names.add(name)
        text, source = read_source(root, spec['source'])
        row = dict(name=name, cohort=cohort, kind=kind)
        if kind == 'online':
            row.update(online_metrics(text))
            if not spec.get('submission_id'):
                warnings.append(name + ': missing submission_id')
        elif kind == 'val_log':
            best, history = validation_metrics(text)
            requested = spec.get('selected_epoch')
            if requested is not None:
                best = next((r for r in history if r['epoch'] == requested), None)
                if best is None:
                    raise ValueError(name + ': selected_epoch absent from log')
            else:
                warnings.append(name + ': selected by rounded VAL AP; verify checkpoint metadata')
            row.update(best)
            row['last_logged_epoch'] = history[-1]['epoch']
            if spec.get('expected_epochs') != history[-1]['epoch']:
                warnings.append(name + ': completeness unverified; set/check expected_epochs')
            curves.extend(dict(name=name, cohort=cohort, **r) for r in history)
        elif kind in ('val_metrics', 'test_metrics'):
            doc = json.loads(text)
            metrics = doc.get('metrics', doc)
            row.update({k: float(metrics[v]) * 100 for k, v in KEYS.items() if v in metrics})
            if 'ca_ap' not in row:
                raise ValueError(name + ': missing official metric')
            row['epoch'] = doc.get('checkpoint_epoch')
            if kind == 'test_metrics' and not spec.get('selected_on_val'):
                raise ValueError(name + ': test requires selected_on_val=true')
        else:
            raise ValueError('Unsupported kind: ' + kind)
        for key in ('submission_id', 'checkpoint', 'classifier', 'notes'):
            row[key] = spec.get(key, '')
        sources = {'results': source}
        for key in ('annotation', 'classification_txt', 'config'):
            if spec.get(key):
                _, sources[key] = read_source(root, spec[key])
        if not spec.get('annotation') and kind != 'online':
            warnings.append(name + ': annotation hash unavailable; cohort equality is user-declared')
        if kind == 'online' and not spec.get('classification_txt'):
            warnings.append(name + ': identical classifier predictions not byte-verified')
        results.append(row)
        audit.append(dict(name=name, specification=spec, sources=sources))
    return results, curves, audit, warnings


def csv_write(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    source = Path(args.manifest).resolve()
    rows, curves, audit, warnings = collect(json.loads(source.read_text(encoding='utf-8-sig')), source.parent)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Separate tables by declared cohort AND evaluation kind; never merge val/test.
    groups = sorted({(r['cohort'], r['kind']) for r in rows})
    report = ['# Paper evidence inventory', '', 'All metrics are percentages. Missing values are not zero.', '']
    for index, (cohort, kind) in enumerate(groups, 1):
        subset = [r for r in rows if (r['cohort'], r['kind']) == (cohort, kind)]
        csv_write(out / ('table_%02d_%s.csv' % (index, kind)), subset)
        report += ['## %s / %s' % (cohort, kind), '', '| Run | AP | AP50 | AP75 | Active AP | Latent AP |',
                   '|---|---:|---:|---:|---:|---:|']
        for row in subset:
            report.append('| ' + row['name'] + ' | ' + ' | '.join(
                '%.2f' % row[k] if k in row else 'NA' for k in KEYS) + ' |')
        report.append('')
    csv_write(out / 'val_curves.csv', curves)
    (out / 'audit.json').write_text(json.dumps(dict(runs=audit, warnings=warnings), ensure_ascii=False, indent=2), encoding='utf-8')
    report += ['## Unresolved evidence', ''] + ['- ' + w for w in warnings]
    report += ['', '## Still required', '',
               '- Runtime / parameter cost measured on the same device and input protocol.',
               '- Paired B1 versus A0 quality/error plots using the SAME annotation set.',
               '- Recent-method comparison or explicit implementation/access limitations.',
               '- No test-GT plots or test confidence intervals without test annotations.',
               '- VAL bootstrap uncertainty does not measure training-seed variability.']
    (out / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('Wrote %d runs to %s; %d audit warnings' % (len(rows), out, len(warnings)))


if __name__ == '__main__':
    main()
