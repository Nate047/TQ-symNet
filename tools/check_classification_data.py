"""Regression checks for official image labels; no MMCV/CUDA model required."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TQ_SymNet.classification import TBXImageDataset


class ClassificationLabelsTest(unittest.TestCase):
    def test_official_external_labels_and_order(self):
        names = ['health/h0001.png', 'sick/s0001.png', 'tb/tb0001.png',
                 'extra/mc+shenzhen/train/CHNCXR_0004_0.png',
                 'extra/mc+shenzhen/val/CHNCXR_0005_1.png',
                 'extra/mc+shenzhen/train/MCUCXR_0001_0.png',
                 'extra/mc+shenzhen/val/MCUCXR_0002_1.png',
                 'extra/da+db/train/n11.png', 'extra/da+db/val/p12.png',
                 'extra/da+db/train/nx1.jpg', 'extra/da+db/val/px12.jpg']
        with tempfile.TemporaryDirectory() as folder:
            annotation = Path(folder) / 'index.json'
            annotation.write_text(json.dumps({'images': [
                dict(id=i + 1, file_name=name) for i, name in enumerate(names)]}), encoding='utf-8')
            data = TBXImageDataset(annotation, folder)
            self.assertEqual(data.labels, [0, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2])
            self.assertEqual(data.filenames, names)
            self.assertEqual(data.image_ids, list(range(1, len(names) + 1)))
            hidden = TBXImageDataset(annotation, folder, labeled=False)
            self.assertEqual(hidden.labels, [-1] * len(names))

    def test_unknown_external_label_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            annotation = Path(folder) / 'index.json'
            for name in ['extra/mc+shenzhen/train/CHNCXR_0004_2.png',
                         'extra/da+db/train/unknown.png']:
                annotation.write_text(json.dumps({'images': [dict(id=1, file_name=name)]}),
                                      encoding='utf-8')
                with self.assertRaises(ValueError):
                    TBXImageDataset(annotation, folder)

    def test_empty_image_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            annotation = Path(folder) / 'index.json'
            annotation.write_text(json.dumps({'images': [dict(id=1, file_name='health/h1.png')]}),
                                  encoding='utf-8')
            image = Path(folder) / 'health/h1.png'
            image.parent.mkdir()
            image.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'missing/empty'):
                TBXImageDataset(annotation, folder).validate_image_files()


def audit_real_splits(folder, image_root=None):
    expected = {'train': 6888, 'val': 2088, 'test': 3302}
    indices = {}
    for split, count in expected.items():
        path = folder / ('all_' + split + '.json')
        data = TBXImageDataset(path, image_root or folder, labeled=split != 'test')
        assert len(data) == count, (split, len(data), count)
        indices[split] = set(data.filenames)
        if split == 'test':
            assert not json.loads(path.read_text(encoding='utf-8')).get('annotations'), 'Hidden test has GT'
        if image_root:
            data.validate_image_files()
        print(split, 'images=', len(data), 'classCounts=',
              [data.labels.count(i) for i in range(3)],
              'external=', sum(name.startswith('extra/') for name in data.filenames))
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not indices[left] & indices[right], (left, right, 'filename overlap')
    print('PASS: official labels, index order, split sizes and filename separation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=Path)
    parser.add_argument('--image-root', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(
        ClassificationLabelsTest))
    if not result.wasSuccessful():
        sys.exit(1)
    if args.json_dir:
        audit_real_splits(args.json_dir, args.image_root)
