"""Balance train/val annotation geometry while preserving the reserved test set.

The objective is specified entirely by annotations, never predictions or AP.
Seed 0 uses the archived v1 partition as its starting point. Only whole images
are exchanged within the same subtype stratum. Source labels are never edited.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
from pathlib import Path

import numpy as np
import generate_tb_only_splits as base

PROTOCOL = 'tbx11k-479-120-200-geometry-balanced-v2-fixed-test'
AREA_EDGES = [1024, 4096, 9216, 16384, 32768, 65536]
WIDTH_EDGES = [32, 64, 96, 128, 192, 256]
ASPECT_EDGES = [.5, .75, 1., 4./3., 2.]


def build_features(images, by_image):
    blocks, descriptors = [], []
    for category in (None, 1, 2):
        for feature, edges, bins in (
            ('area', AREA_EDGES, 7), ('width', WIDTH_EDGES, 7),
            ('height', WIDTH_EDGES, 7), ('aspect', ASPECT_EDGES, 6),
            ('position_3x3', [], 9)):
            values = np.zeros((len(images), bins), dtype=np.float64)
            for row, image in enumerate(images):
                for ann in by_image[image['id']]:
                    if category is not None and ann['category_id'] != category:
                        continue
                    x, y, w, h = ann['bbox']
                    if feature == 'position_3x3':
                        cx, cy = (x+w/2)/image['width'], (y+h/2)/image['height']
                        bucket = min(2, int(cx*3))*3 + min(2, int(cy*3))
                    else:
                        value = {'area': ann['area'], 'width': w,
                                 'height': h, 'aspect': w/h}[feature]
                        bucket = int(np.searchsorted(edges, value, side='right'))
                    values[row, bucket] += 1
            blocks.append(values)
            descriptors.append(dict(category=category, feature=feature, edges=edges, bins=bins))
    count_hist = np.zeros((len(images), 4), dtype=np.float64)
    for row, image in enumerate(images):
        count_hist[row, min(4, len(by_image[image['id']]))-1] = 1
    blocks.append(count_hist)
    descriptors.append(dict(category=None, feature='boxes_per_image', edges=[1,2,3,4], bins=4))
    return np.concatenate(blocks, axis=1), descriptors


def distribute(document, parent, seed=0):
    _, by_image = base.validate_source(document)
    reserved = {i['id'] for i in parent['test']['images']}
    images = sorted([i for i in document['images'] if i['id'] not in reserved],
        key=lambda i: (base.digest(('%s|%s|%s' % (PROTOCOL, seed, i['file_name'])).encode()), i['file_name']))
    features, descriptors = build_features(images, by_image)
    index = {i['id']: k for k, i in enumerate(images)}
    initial_val = {i['id'] for i in parent['val']['images']}
    is_val = np.array([i['id'] in initial_val for i in images])
    counts = np.array([[sum(a['category_id']==c for a in by_image[i['id']])
                       for c in (1,2)] for i in images], dtype=int)
    small = np.array([sum(a['area'] < 1024 for a in by_image[i['id']]) for i in images])
    strata = np.array(['mixed' if np.all(v > 0) else 'active_only' if v[0] else 'obsolete_only'
                       for v in counts])
    fraction = 120./599.
    expected = features.sum(axis=0)*fraction
    weights = np.zeros_like(expected)
    offset = 0
    for desc in descriptors:
        n = desc['bins']
        weights[offset:offset+n] = 1./(n*np.maximum(expected[offset:offset+n]*(1-fraction),1.))
        offset += n
    class_target = counts.sum(axis=0)*fraction
    lower, upper = np.floor(class_target).astype(int), np.ceil(class_target).astype(int)
    small_limits = (1 if small.sum() else 0, max(1, int(math.ceil(small.sum()*fraction))))

    def objective(mask):
        return float(np.sum(weights*(features[mask].sum(axis=0)-expected)**2))

    before = is_val.copy()
    trace = []
    for iteration in range(250):
        residual = features[is_val].sum(axis=0)-expected
        current_counts, current_small = counts[is_val].sum(axis=0), small[is_val].sum()
        feasible_now = (np.all(current_counts >= lower) and np.all(current_counts <= upper)
                        and small_limits[0] <= current_small <= small_limits[1])
        best = None
        for stratum in ('active_only','obsolete_only','mixed'):
            train_ids = np.flatnonzero((strata==stratum) & ~is_val)
            val_ids = np.flatnonzero((strata==stratum) & is_val)
            a, b = features[train_ids], features[val_ids]
            # Exact quadratic objective difference for exchanging one image pair.
            delta = (2*a.dot(weights*residual)+(a*a).dot(weights))[:,None]
            delta = delta + (-2*b.dot(weights*residual)+(b*b).dot(weights))[None,:]
            delta = delta - 2*(a*weights).dot(b.T)
            new_counts = current_counts + counts[train_ids,None,:]-counts[val_ids][None,:,:]
            new_small = current_small + small[train_ids,None]-small[val_ids][None,:]
            valid = ((new_counts >= lower).all(axis=-1) & (new_counts <= upper).all(axis=-1)
                     & (new_small >= small_limits[0]) & (new_small <= small_limits[1]))
            delta[~valid] = np.inf
            ij = np.unravel_index(int(np.argmin(delta)), delta.shape)
            candidate = (float(delta[ij]), int(train_ids[ij[0]]), int(val_ids[ij[1]]))
            if best is None or candidate < best:
                best = candidate
        if best is None or not math.isfinite(best[0]):
            raise RuntimeError('No feasible whole-image exchange; revise declared constraints, not AP')
        if feasible_now and best[0] >= -1e-10:
            break
        _, entering, leaving = best
        is_val[entering], is_val[leaving] = True, False
        trace.append(dict(iteration=iteration+1, entering_val=images[entering]['file_name'],
            leaving_val=images[leaving]['file_name'], objective=objective(is_val)))
    if not (np.all(counts[is_val].sum(axis=0) >= lower) and
            np.all(counts[is_val].sum(axis=0) <= upper) and
            small_limits[0] <= small[is_val].sum() <= small_limits[1]):
        raise AssertionError('Balance constraints failed')
    partition = dict(train={i['id'] for k,i in enumerate(images) if not is_val[k]},
                     val={i['id'] for k,i in enumerate(images) if is_val[k]}, test=reserved)
    for role, quota in base.QUOTAS.items():
        for phase, expected_images in zip(base.PHASES,quota):
            actual = sum(('mixed' if len({a['category_id'] for a in by_image[i]})==2 else
                         'active_only' if by_image[i][0]['category_id']==1 else 'obsolete_only')==role
                         for i in partition[phase])
            assert actual == expected_images
    distributions = []
    offset = 0
    for desc in descriptors:
        n = desc['bins']
        block = features[:,offset:offset+n]
        row = dict(desc)
        for label, mask in [('before_train',~before), ('before_val',before),
                            ('after_train',~is_val), ('after_val',is_val)]:
            totals = block[mask].sum(axis=0)
            row[label] = dict(counts=totals.astype(int).tolist(),
                             fractions=(totals/max(1,totals.sum())).tolist())
        distributions.append(row)
        offset += n
    report = dict(protocol=PROTOCOL, seed=seed, fixed_test_images=200,
        optimizer='deterministic best pair exchange within subtype strata; at most 250 exchanges',
        objective='sum of squared val-bin count deviations / (bins-per-block * max(expected*(1-val_fraction),1))',
        objective_before=objective(before), objective_after=objective(is_val),
        class_box_limits=dict(lower=lower.tolist(),upper=upper.tolist()),
        val_small_box_limits=list(small_limits), descriptors=descriptors,
        exchanges=trace, distributions=distributions,
        annotation_only=True, used_predictions_or_AP=False,
        selection_note='Protocol revision after annotation distribution audit; not an untouched original holdout.',
        patient_independence='Unknown: patient IDs unavailable; exact image duplicates audited separately.')
    return partition, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-json', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--parent-split', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_json.read_bytes()
    document = json.loads(raw)
    _, by_image = base.validate_source(document)
    parent_manifest = json.loads((args.parent_split/'manifest.json').read_text(encoding='utf-8'))
    assert base.digest(raw) == parent_manifest['source_json_sha256']
    assert parent_manifest['split_seed'] == 0
    for name, sha in parent_manifest['files'].items():
        if base.digest((args.parent_split/name).read_bytes()) != sha:
            raise ValueError('Parent split checksum mismatch: '+name)
    parent = {p:json.loads((args.parent_split/(p+'.json')).read_text(encoding='utf-8'))
              for p in ('train','val','test','train_tb','val_tb','test_tb')}
    base.verify_preserved(document,parent)
    partition, report = distribute(document,parent)
    repeated, repeated_report = distribute(document,parent)
    if partition != repeated or report != repeated_report:
        raise AssertionError('Repeated deterministic assignment differs')
    print('[TB-SPLIT-V2] repeated assignment verified; auditing image content', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        audits = list(pool.map(lambda im:base.audit_image(args.image_root,im),document['images']))
    pixels = defaultdict(list)
    for row in audits:
        pixels[row['decoded_rgb_sha256']].append(row['file_name'])
    if any(len(v)>1 for v in pixels.values()):
        raise ValueError('Exact duplicate images require grouped partitioning')
    files, generated = {}, {}
    manifest = dict(protocol=PROTOCOL,split_seed=0,source_json_sha256=base.digest(raw),
        parent_manifest_sha256=base.digest((args.parent_split/'manifest.json').read_bytes()),
        test_membership_preserved=True,files={},subsets={})
    for phase in base.PHASES:
        for merged in (False,True):
            key = phase+('_tb' if merged else '')
            doc = base.subset(document,partition[phase],0,phase,merged)
            doc['tq_split']['protocol'] = PROTOCOL
            generated[key] = doc
            value = base.encoded(doc)
            files['seed_0/'+key+'.json'] = value
            manifest['files'][key+'.json'] = base.digest(value)
        images = generated[phase]['images']
        text = ''.join(base.canonical_name(i['file_name'])+'\n' for i in images).encode()
        files['seed_0/'+phase+'_images.txt'] = text
        manifest['files'][phase+'_images.txt'] = base.digest(text)
        manifest['subsets'][phase] = dict(base.summary(generated[phase],by_image),
            file_names=[base.canonical_name(i['file_name']) for i in images],
            original_image_ids=[i['id'] for i in images])
    base.verify_preserved(document,generated)
    assert generated['test']['images'] == parent['test']['images']
    assert generated['test']['annotations'] == parent['test']['annotations']
    files['seed_0/manifest.json'] = base.encoded(manifest)
    files['balance_report.json'] = base.encoded(report)
    files['protocol.json'] = base.encoded(dict(protocol=PROTOCOL,split_seed=0,
        source_json_sha256=base.digest(raw),quotas=base.QUOTAS,test_membership_preserved=True,
        parent_manifest_sha256=manifest['parent_manifest_sha256'],
        source_images=799,source_boxes=1211,
        summaries={p:base.summary(generated[p],by_image) for p in base.PHASES},
        method=report['optimizer'],annotation_only=True,used_predictions_or_AP=False))
    files['source_image_audit.json'] = base.encoded(dict(source_json_sha256=base.digest(raw),
        exact_pixel_duplicates=[],images=sorted(audits,key=lambda i:i['file_name'])))
    for name,value in files.items():
        path = args.out_dir/name
        if path.exists() and path.read_bytes()!=value:
            raise FileExistsError('Different existing cohort; choose a new directory: '+str(path))
    for name,value in files.items():
        path = args.out_dir/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            path.write_bytes(value)
        assert path.read_bytes()==value
    assert args.source_json.read_bytes()==raw
    print(json.dumps({p:base.summary(generated[p],by_image) for p in base.PHASES},indent=2))
    print('[TB-SPLIT-V2] exchanges=%d objective=%.6f -> %.6f' %
          (len(report['exchanges']),report['objective_before'],report['objective_after']))
    print('[TB-SPLIT-V2] READY',args.out_dir.resolve())


if __name__=='__main__':
    main()
