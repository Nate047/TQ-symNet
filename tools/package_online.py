"""Validate and zip real classification/detection outputs; never uploads anything."""
import argparse
import json
import math
import hashlib
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', required=True)
    parser.add_argument('--json', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    txt, bbox, out = map(Path, (args.txt, args.json, args.out))
    rows = [list(map(float, line.split())) for line in txt.read_text().splitlines()]
    if len(rows) != 3302 or any(len(r) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in r)
                              or abs(sum(r)-1) > 1e-6 for r in rows):
        raise ValueError('TXT must contain 3302 valid Healthy/Sick/TB probability rows')
    sidecar = txt.with_suffix('.classification.manifest.json')
    if not sidecar.exists():
        sidecar = txt.with_suffix('.manifest.json')
    manifest = json.loads(sidecar.read_text())
    if manifest.get('external_classifier'):
        print('DIAGNOSTIC PACKAGE: classification is external, not TQ-SymNet; '
              'source=%s' % manifest.get('source_txt', '<unknown>'))
    legacy_detection_manifest = 'class_order' not in manifest
    if legacy_detection_manifest:
        # Early exports overwrote the classification sidecar with the detection
        # sidecar. Its full index still allows format/order/gating validation.
        if (manifest.get('images') != 3302 or not manifest.get('cls_filter')
                or Path(manifest.get('classification_txt', '')).name != txt.name):
            raise ValueError('No usable classification/test-index manifest')
        ids = manifest['image_ids_in_order']
        print('Legacy detection manifest: classifier provenance is unavailable; '
              'using retained test index and checking TXT/bbox filtering consistency.')
    else:
        if manifest['class_order'] != ['Healthy', 'Sick non-TB', 'TB']:
            raise ValueError('Wrong class order')
        ids = manifest['image_ids']
        if ('txt_sha256' in manifest
                and hashlib.sha256(txt.read_bytes()).hexdigest() != manifest['txt_sha256']):
            raise ValueError('TXT differs from its export manifest')
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError('Invalid test index')
    predictions = json.loads(bbox.read_text())
    if not isinstance(predictions, list):
        raise ValueError('JSON must be a COCO detection-instance list')
    known = set(ids)
    tb_images = {image_id for image_id, row in zip(ids, rows)
                 if max(range(3), key=lambda j: row[j]) == 2}
    for item in predictions:
        box = item['bbox']
        if (item['image_id'] not in known or item['category_id'] not in (1, 2)
                or len(box) != 4 or not all(math.isfinite(v) for v in box+[item['score']])
                or box[2] < 0 or box[3] < 0 or not 0 <= item['score'] <= 1):
            raise ValueError('Invalid COCO detection')
        if legacy_detection_manifest and item['image_id'] not in tb_images:
            raise ValueError('BBox on an image classified as non-TB despite cls_filter=True')
    if legacy_detection_manifest:
        counts = [sum(max(range(3), key=lambda j: r[j]) == label for r in rows) for label in range(3)]
        if counts != manifest.get('classification_counts') or len(predictions) != manifest.get('detections'):
            raise ValueError('Legacy TXT/JSON no longer match their detection manifest')
    if out.exists():
        raise FileExistsError(str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(txt, 'TQSymNet.txt')
        archive.write(bbox, 'TQSymNet.json')
    print('READY: %s; %d classification rows, %d bbox instances, %d images with boxes' %
          (out, len(rows), len(predictions), len({p['image_id'] for p in predictions})))


if __name__ == '__main__':
    main()
