"""Dependency-free validation of the declared public TB split and checkpoint cohort."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_split_contract(manifest):
    manifest = Path(manifest).resolve()
    document = json.loads(manifest.read_text(encoding='utf-8'))
    for name, expected in document['files'].items():
        if sha256_file(manifest.parent / name) != expected:
            raise RuntimeError('Split file differs from its manifest: ' + name)
    paths, identities, boxes = {}, {}, set()
    for phase, count in zip(('train', 'val', 'test'), (479, 120, 200)):
        path = manifest.parent / (phase + '.json')
        value = json.loads(path.read_text(encoding='utf-8'))
        names = {image['file_name'] for image in value['images']}
        if len(names) != count or names != set(document['subsets'][phase]['file_names']):
            raise RuntimeError('Wrong split identities/count: ' + phase)
        if boxes & {a['id'] for a in value['annotations']}:
            raise RuntimeError('Annotations shared between partitions')
        boxes.update(a['id'] for a in value['annotations'])
        paths[phase], identities[phase] = path, names
    if any(identities[a] & identities[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise RuntimeError('Training/validation/test share image filenames')
    if len(boxes) != 1211:
        raise RuntimeError('Public-pool box conservation failed')
    contract = dict(protocol=document['protocol'], split_seed=document['split_seed'],
        source_json_sha256=document['source_json_sha256'],
        train_sha256=document['files']['train.json'],
        val_sha256=document['files']['val.json'], test_sha256=document['files']['test.json'],
        images=dict(train=479, val=120, test=200),
        manifest_sha256=sha256_file(manifest))
    return contract, paths


def verify_checkpoint_contract(saved, expected):
    if not saved:
        raise RuntimeError('Checkpoint lacks the new split contract; train the 479-image detector first')
    for key in ('protocol', 'split_seed', 'source_json_sha256', 'train_sha256', 'val_sha256', 'test_sha256'):
        if saved.get(key) != expected.get(key):
            raise RuntimeError('Checkpoint uses a different TB cohort: ' + key)


def dataset_ann_files(dataset):
    if hasattr(dataset, 'datasets'):
        return [p for child in dataset.datasets for p in dataset_ann_files(child)]
    if hasattr(dataset, 'dataset'):
        return dataset_ann_files(dataset.dataset)
    if hasattr(dataset, 'ann_file'):
        return [dataset.ann_file]
    raise RuntimeError('Cannot audit dataset annotations: ' + type(dataset).__name__)


def verify_dataset_contract(dataset, expected_sha, phase):
    for path in dataset_ann_files(dataset):
        if sha256_file(path) != expected_sha:
            raise RuntimeError('Actual %s loader uses a different annotation file: %s' % (phase, path))
