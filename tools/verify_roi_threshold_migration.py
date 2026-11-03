"""Read-only verification after copying the manual migration tree."""
import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='ROI_THRESHOLD_MANIFEST.json')
    args=parser.parse_args()
    manifest = (ROOT / args.manifest).resolve()
    if ROOT not in manifest.parents:
        raise SystemExit('Manifest must be inside the project')
    if not manifest.is_file():
        raise SystemExit('Copy '+args.manifest+' into the project root first')
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    for entry in doc['files']:
        target = (ROOT / entry['path']).resolve()
        if ROOT not in target.parents:
            raise SystemExit('Manifest path escapes project: ' + entry['path'])
        if not target.is_file():
            raise SystemExit('Missing migrated file: ' + entry['path'])
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise SystemExit('Migrated file hash differs: ' + entry['path'])
    print('PASS migration hashes:', len(doc['files']), 'files;', doc['experiment'])


if __name__ == '__main__':
    main()
