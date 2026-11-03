"""Verify installed supplement payload and the untouched dependency versions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'publication_supplement_manifest.json').read_text(encoding='utf-8'))
    errors = []
    for label in ('files', 'dependencies'):
        for entry in manifest[label]:
            path = (ROOT/entry['path']).resolve()
            if ROOT not in path.parents or not path.is_file():
                errors.append('Missing/unsafe '+entry['path'])
            elif hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                errors.append('Hash mismatch '+entry['path'])
    if errors:
        raise SystemExit('\n'.join(errors)+'\nDo not train on a mixed deployment; audit the differences first.')
    print('PASS publication supplement payload and dependency hashes')
    print('No data/weights were bundled; runtime/cohort checks are separate.')


if __name__ == '__main__':
    main()
