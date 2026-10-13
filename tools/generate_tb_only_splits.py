"""Create immutable, stratified COCO partitions from the public TBX11K box pool.

The keyed SHA256 ordering is independent of model RNG and Python shuffle versions.
Existing annotations/IDs/geometries are preserved. Only train_tb/val_tb/test_tb
collapse subtype category IDs for single-class detection.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


PROTOCOL = 'tbx11k-479-120-200-stratified-sha256-v1'
PHASES = ('train', 'val', 'test')
QUOTAS = {
    'active_only': (378, 94, 158),
    'obsolete_only': (83, 21, 35),
    'mixed': (18, 5, 7),
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def canonical_name(name):
    name = name.replace('\\', '/')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Image filenames must be safe relative paths: ' + name)
    return path.as_posix()


def validate_source(document):
    images = document['images']
    annotations = document['annotations']
    if len(images) != 799 or len(annotations) != 1211:
        raise ValueError('Expected the verified 799-image, 1211-box public pool')
    image_ids = [item['id'] for item in images]
    names = [canonical_name(item['file_name']) for item in images]
    if len(set(image_ids)) != len(images) or len(set(names)) != len(images):
        raise ValueError('Duplicate source image IDs or filenames')
    if len({a['id'] for a in annotations}) != len(annotations):
        raise ValueError('Duplicate source annotation IDs')
    by_image = defaultdict(list)
    known = set(image_ids)
    for item in annotations:
        box = item['bbox']
        if (item['image_id'] not in known or item['category_id'] not in (1, 2)
                or len(box) != 4 or not all(math.isfinite(v) for v in box)
                or box[2] <= 0 or box[3] <= 0
                or item.get('iscrowd', 0) or item.get('ignore', 0)):
            raise ValueError('Unexpected annotation: %s' % item['id'])
        by_image[item['image_id']].append(item)
    groups = defaultdict(list)
    for image in images:
        labels = {a['category_id'] for a in by_image[image['id']]}
        if labels == {1}:
            group = 'active_only'
        elif labels == {2}:
            group = 'obsolete_only'
        elif labels == {1, 2}:
            group = 'mixed'
        else:
            raise ValueError('Unboxed or unknown-category image: %s' % image['file_name'])
        groups[group].append(image)
    for group, quota in QUOTAS.items():
        if len(groups[group]) != sum(quota):
            raise ValueError('Source stratum differs from verified quotas: ' + group)
    return groups, by_image


def audit_image(root, item):
    from PIL import Image
    path = root / canonical_name(item['file_name'])
    raw = path.read_bytes()
    with Image.open(path) as image:
        if image.size != (item['width'], item['height']):
            raise ValueError('Stored image dimensions differ from JSON: ' + str(path))
        rgb = image.convert('RGB')
        pixel_sha = digest(str(rgb.size).encode('ascii') + b'\0RGB\0' + rgb.tobytes())
    return dict(file_name=canonical_name(item['file_name']), file_sha256=digest(raw),
                decoded_rgb_sha256=pixel_sha, width=item['width'], height=item['height'])


def assign(groups, seed):
    partition = {phase: set() for phase in PHASES}
    for group, quota in QUOTAS.items():
        ranked = sorted(groups[group], key=lambda image: (
            digest(('%s|%d|%s|%s' % (
                PROTOCOL, seed, group, canonical_name(image['file_name']))).encode('utf-8')),
            canonical_name(image['file_name'])))
        offset = 0
        for phase, count in zip(PHASES, quota):
            partition[phase].update(i['id'] for i in ranked[offset:offset + count])
            offset += count
    if tuple(len(partition[p]) for p in PHASES) != (479, 120, 200):
        raise AssertionError('Incorrect partition size')
    if any(partition[a] & partition[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise AssertionError('Cross-partition image overlap')
    return partition


def subset(document, selected, seed, phase, merged=False):
    result = copy.deepcopy(document)
    result['images'] = sorted(
        [copy.deepcopy(image) for image in document['images'] if image['id'] in selected],
        key=lambda image: canonical_name(image['file_name']))
    result['annotations'] = [copy.deepcopy(a) for a in document['annotations']
                             if a['image_id'] in selected]
    if merged:
        result['categories'] = [dict(id=1, name='tuberculosis', supercategory='Tuberculosis')]
        for item in result['annotations']:
            item['source_category_id'] = item['category_id']
            item['category_id'] = 1
    # The released JSON stores legacy info as a list; preserve it verbatim.
    result['tq_split'] = dict(
        protocol=PROTOCOL, split_seed=seed, phase=phase,
        category_mode='single_tb' if merged else 'original_subtypes',
        provenance='Our public-pool partition; not the unpublished ConES/MIU-VL/StructuralGLIP image lists.')
    return result


def summary(document, by_image):
    roles = Counter()
    for image in document['images']:
        labels = {a['category_id'] for a in by_image[image['id']]}
        role = 'active_only' if labels == {1} else 'obsolete_only' if labels == {2} else 'mixed'
        roles[role] += 1
    annotations = document['annotations']
    sizes = Counter('small' if a['area'] < 1024 else 'medium' if a['area'] < 9216 else 'large'
                    for a in annotations)
    return dict(images=len(document['images']), boxes=len(annotations),
                image_strata=dict(roles), annotation_categories=dict(Counter(
                    str(a['category_id']) for a in annotations)), area_bins=dict(sizes))


def verify_preserved(document, generated):
    source_images = {item['id']: item for item in document['images']}
    source_annotations = {item['id']: item for item in document['annotations']}
    seen_images, seen_annotations = set(), set()
    for phase in PHASES:
        original = generated[phase]
        merged = generated[phase + '_tb']
        if original['images'] != merged['images']:
            raise AssertionError('Subtype/single-TB image lists differ')
        for item in original['images']:
            if item != source_images[item['id']] or item['id'] in seen_images:
                raise AssertionError('Image metadata changed or duplicated across splits')
            seen_images.add(item['id'])
        if len(original['annotations']) != len(merged['annotations']):
            raise AssertionError('Subtype/single-TB annotation counts differ')
        for item, single in zip(original['annotations'], merged['annotations']):
            if item != source_annotations[item['id']] or item['id'] in seen_annotations:
                raise AssertionError('Source annotation changed or duplicated across splits')
            restored = copy.deepcopy(single)
            restored['category_id'] = restored.pop('source_category_id')
            if restored != item or single['category_id'] != 1:
                raise AssertionError('Merged label changed geometry or source metadata')
            seen_annotations.add(item['id'])
    if seen_images != set(source_images) or seen_annotations != set(source_annotations):
        raise AssertionError('The three partitions do not conserve the complete source pool')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-json', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Repeated split seed')
    source_bytes = args.source_json.read_bytes()
    document = json.loads(source_bytes)
    groups, by_image = validate_source(document)
    print('[TB-SPLIT] auditing all 799 original images; source JSON remains unchanged', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        audits = list(pool.map(lambda item: audit_image(args.image_root, item), document['images']))
    duplicates = defaultdict(list)
    for item in audits:
        duplicates[item['decoded_rgb_sha256']].append(item['file_name'])
    duplicated = [names for names in duplicates.values() if len(names) > 1]
    if duplicated:
        raise ValueError('Exact duplicate images require grouped partitioning: %s' % duplicated)
    files = {}
    source_sha = digest(source_bytes)
    files['source_image_audit.json'] = encoded(dict(
        source_json_sha256=source_sha, exact_pixel_duplicates=[],
        images=sorted(audits, key=lambda item: item['file_name']),
        note='Image-content checks do not establish patient identity; patient IDs are absent.'))
    protocol = dict(protocol=PROTOCOL, seeds=args.seeds, source_json_sha256=source_sha,
        source_images=799, source_boxes=1211, quotas=QUOTAS,
        assignment='Within each image stratum, sort keyed SHA256(seed, stratum, canonical filename).',
        original_coordinates_ids_and_area_preserved=True,
        experiment_policy='Train each detector from generic pretraining on its train split; '
            'choose checkpoints on val; evaluate test only after settings are fixed.',
        annotation_modes=dict(subtype='train.json/val.json/test.json preserve Active=1 and Obsolete=2',
                              single_tb='train_tb.json/val_tb.json/test_tb.json map all boxes to TB=1'),
        seeds_summary={})
    for seed in args.seeds:
        partition = assign(groups, seed)
        generated = {}
        manifest = dict(protocol=PROTOCOL, split_seed=seed, source_json_sha256=source_sha,
                        subsets={}, files={})
        directory = 'seed_%d' % seed
        for phase in PHASES:
            for merged in (False, True):
                key = phase + ('_tb' if merged else '')
                generated[key] = subset(document, partition[phase], seed, phase, merged)
                name = directory + '/' + key + '.json'
                files[name] = encoded(generated[key])
                manifest['files'][key + '.json'] = digest(files[name])
            images = generated[phase]['images']
            list_name = directory + '/' + phase + '_images.txt'
            files[list_name] = (''.join(canonical_name(i['file_name']) + '\n' for i in images)).encode('utf-8')
            manifest['files'][phase + '_images.txt'] = digest(files[list_name])
            manifest['subsets'][phase] = dict(summary(generated[phase], by_image),
                original_image_ids=[i['id'] for i in images],
                file_names=[canonical_name(i['file_name']) for i in images])
        verify_preserved(document, generated)
        protocol['seeds_summary'][str(seed)] = {p: summary(generated[p], by_image) for p in PHASES}
        files[directory + '/manifest.json'] = encoded(manifest)
    files['protocol.json'] = encoded(protocol)
    # Check every existing artifact before writing anything. Never overwrite a
    # different cohort; rerunning the identical command is idempotent.
    for relative, value in files.items():
        path = args.out_dir / relative
        if path.exists() and path.read_bytes() != value:
            raise FileExistsError('Existing partition differs; use a new output folder: ' + str(path))
    for relative, value in files.items():
        path = args.out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(value)
        if path.read_bytes() != value:
            raise RuntimeError('Written artifact checksum mismatch: ' + str(path))
    if args.source_json.read_bytes() != source_bytes:
        raise RuntimeError('Source JSON changed during generation')
    print(json.dumps(protocol['seeds_summary'], indent=2), flush=True)
    print('[TB-SPLIT] READY:', args.out_dir.resolve(), flush=True)


if __name__ == '__main__':
    main()
