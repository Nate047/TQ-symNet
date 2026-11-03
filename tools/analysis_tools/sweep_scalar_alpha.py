"""VAL-only offline alpha sweep of selector_candidates_epoch_XX.json.

No model execution. Use the project's actual Gaussian Soft-NMS. GT is used
only for evaluation, never scoring, selecting, or modifying candidate boxes.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from TQ_SymNet.models.selection import gaussian_soft_select
from TQ_SymNet.bootstrap_ap import coco_all_area_records, ap_from_records, paired_bootstrap

VAL_SHA = 'bdec993506af15bf72010b32ec82ac279615ccc4c51ad9ca9438a7b9638d6e9d'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(rows, annotation):
    names = {im['file_name'].replace('\\', '/'): im['id'] for im in annotation['images']}
    if len(names) != len(annotation['images']):
        raise ValueError('Duplicate annotation filenames')
    prepared, seen = [], set()
    for row in rows:
        filename = row['filename'].replace('\\', '/')
        matches = [v for k, v in names.items() if filename == k or filename.endswith('/'+k)]
        if len(matches) != 1 or matches[0] in seen:
            raise ValueError('Ambiguous, duplicate or unmatched image: '+filename)
        seen.add(matches[0])
        boxes = torch.tensor(row['boxes'], dtype=torch.float32).reshape(-1, 4)
        p = torch.tensor(row['sym'], dtype=torch.float32)
        q = torch.tensor(row['residual_logit'], dtype=torch.float32)
        old = torch.tensor(row['corrected_score'], dtype=torch.float32)
        if any(v.ndim != 1 or len(v) != len(boxes) for v in (p, q, old)):
            raise ValueError('Mismatched candidate scores')
        if not all(torch.isfinite(v).all() for v in (boxes, p, q, old)):
            raise ValueError('Nonfinite candidate data')
        if (boxes[:, 2:] < boxes[:, :2]).any() or (p < 0).any() or (p > 1).any():
            raise ValueError('Invalid boxes/probabilities')
        prepared.append(dict(id=matches[0], boxes=boxes, sym_z=torch.logit(p.clamp(1e-6, 1.-1e-6)),
                             q=q, old=old, source=row.get('soft_source'),
                             soft_score=row.get('soft_score')))
    if seen != set(names.values()):
        raise ValueError('Incomplete candidate export; evaluate the entire matching VAL cohort')
    return prepared


def evaluate(prepared, gt, alpha, sigma, topk, max_dets, return_records=False):
    predictions = []
    parity = dict(images_different_source=0, max_soft_score_error=0.)
    category = sorted(gt.getCatIds())[0]  # Labels are pooled by useCats=0.
    for row in prepared:
        score = row['old'] if alpha is None else ((1.-alpha)*row['sym_z']+alpha*row['q']).sigmoid()
        ids, values = gaussian_soft_select(row['boxes'], score, topk=topk, sigma=sigma)
        if alpha is None and row['source'] is not None:
            if ids.tolist() != row['source']:
                parity['images_different_source'] += 1
            if row['soft_score'] is not None and len(values) == len(row['soft_score']):
                error = float((values-torch.tensor(row['soft_score'])).abs().max()) if len(values) else 0.
                parity['max_soft_score_error'] = max(parity['max_soft_score_error'], error)
        for index, value in zip(ids[:max_dets].tolist(), values[:max_dets].tolist()):
            x, y, x2, y2 = row['boxes'][index].tolist()
            predictions.append(dict(image_id=row['id'], category_id=category,
                                    bbox=[x, y, x2-x, y2-y], score=value))
    if not predictions:
        raise ValueError('Empty candidate export')
    with contextlib.redirect_stdout(io.StringIO()):
        result = gt.loadRes(predictions)
        ev = COCOeval(gt, result, 'bbox')
        ev.params.useCats = 0
        ev.params.catIds = [1, 2, 3]
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    records = coco_all_area_records(ev)
    curve = ap_from_records(records)
    np.testing.assert_allclose([curve.mean(), curve[0], curve[5]],
                               ev.stats[:3]*100, rtol=0., atol=1e-8)
    row = dict(alpha=alpha, AP=float(ev.stats[0]*100), AP50=float(ev.stats[1]*100),
               AP75=float(ev.stats[2]*100), predictions=len(predictions),
               ap_by_iou=curve.tolist())
    if return_records:
        return row, parity, records, predictions
    return row, parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0., .25, .5, .53571, .75, 1.])
    parser.add_argument('--sigma', type=float, default=.7)
    parser.add_argument('--topk', type=int, default=128)
    parser.add_argument('--max-dets', type=int, default=100)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--source-meta', help='JSON written by export_supplement_candidates.py')
    parser.add_argument('--expected-epoch', type=int, default=7)
    parser.add_argument('--allow-unverified-provenance', action='store_true',
                        help='Legacy exports only; report explicitly marks checkpoint provenance unverified')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--bootstrap-reference', type=float, default=.5)
    parser.add_argument('--bootstrap-comparisons', type=float, nargs='+', default=[0., 1.])
    args = parser.parse_args()
    if any(not math.isfinite(x) or not 0. <= x <= 1. for x in args.alphas):
        parser.error('Alpha must be finite and within [0,1]')
    if args.bootstrap < 0:
        parser.error('bootstrap must be nonnegative')
    if args.bootstrap and any(x not in args.alphas for x in
                             [args.bootstrap_reference]+args.bootstrap_comparisons):
        parser.error('Bootstrap alpha settings must be included in --alphas')
    if args.sigma <= 0 or not math.isfinite(args.sigma) or args.topk < args.max_dets or args.max_dets != 100:
        parser.error('Require positive sigma, topk>=100, max-dets=100')
    if args.threads < 1:
        parser.error('threads must be positive')
    if file_hash(args.annotations) != VAL_SHA:
        raise ValueError('Expected original 200-image VAL annotation hash; do not tune on TEST')
    torch.set_num_threads(args.threads)
    raw = json.loads(Path(args.candidates).read_text(encoding='utf-8-sig'))
    annotation = json.loads(Path(args.annotations).read_text(encoding='utf-8-sig'))
    prepared = prepare(raw, annotation)
    meta_path = Path(args.source_meta) if args.source_meta else Path(args.candidates).with_suffix('.meta.json')
    source_meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else None
    if source_meta is None and not args.allow_unverified_provenance:
        raise ValueError('Missing checkpoint provenance; export the selected checkpoint first, '
                         'or explicitly mark a legacy export unverified')
    if source_meta is not None:
        if source_meta['candidates_sha256'] != file_hash(args.candidates):
            raise ValueError('Candidate file does not match source metadata')
        if source_meta['annotation_sha256'] != VAL_SHA or source_meta['checkpoint_epoch'] != args.expected_epoch:
            raise ValueError('Wrong VAL cohort/checkpoint epoch; never substitute epoch8 for selected epoch7')
        if (source_meta['sigma'] != args.sigma or source_meta['topk'] != args.topk
                or source_meta['max_dets'] != args.max_dets):
            raise ValueError('Postprocessing differs from the exported run')
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.annotations)
    original, parity = evaluate(prepared, gt, None, args.sigma, args.topk, args.max_dets)
    print('Exported score replay:', original, 'CPU parity:', parity, flush=True)
    rows, caches = [], {}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for alpha in sorted(set(args.alphas)):
        row, _, cache, predictions = evaluate(prepared, gt, alpha, args.sigma,
                                              args.topk, args.max_dets, True)
        rows.append(row)
        caches[alpha] = cache
        (out/('alpha_%s.bbox.json' % str(alpha).replace('.', 'p'))).write_text(
            json.dumps(predictions), encoding='utf-8')
        print('alpha=%.6f AP/AP50/AP75=%.4f/%.4f/%.4f' %
              (alpha, row['AP'], row['AP50'], row['AP75']), flush=True)
    bootstraps = {}
    if args.bootstrap:
        for alpha in args.bootstrap_comparisons:
            bootstraps[str(alpha)] = paired_bootstrap(caches[args.bootstrap_reference],
                caches[alpha], args.bootstrap, args.bootstrap_seed)
            print('Paired bootstrap alpha=%s minus %s: %s' %
                  (alpha, args.bootstrap_reference, bootstraps[str(alpha)]), flush=True)
    report = dict(candidates=str(args.candidates), candidates_sha256=file_hash(args.candidates),
                  annotations_sha256=VAL_SHA, images=len(prepared),
                  candidate_count=sum(len(r['boxes']) for r in prepared),
                  source_epoch=Path(args.candidates).stem, sigma=args.sigma,
                  topk=args.topk, max_dets=args.max_dets, original_replay=original,
                  cpu_replay_parity=parity, results=rows, best_val=max(rows, key=lambda x:x['AP']),
                  checkpoint_provenance=source_meta, bootstrap_reference=args.bootstrap_reference,
                  paired_bootstraps=bootstraps,
                  note='Validation tuning on FIXED exported quality logits; not retraining, not a new test result. '
                       'Epoch-8 candidates do not represent the selected epoch-7 checkpoint. '
                       'Inspect replay parity before interpreting very small changes.')
    (out/'alpha_sweep.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (out/'alpha_sweep.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('Saved', out.resolve())


if __name__ == '__main__':
    main()
