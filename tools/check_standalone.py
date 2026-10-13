"""Audit project isolation; runtime mode checks the actual import origin."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    for relative in ('mmdet/__init__.py', 'mmdet/apis/train.py',
                     'mmdet/core/hook/ema.py', 'mmdet/datasets/coco.py',
                     'mmdet/datasets/tbx11k.py', 'mmdet/datasets/pipelines/cxr_transforms.py',
                     'mmdet/models/utils/sym_attention.py',
                     'mmdet/models/utils/positional_encoding.py',
                     'mmdet/utils/deterministic_ops.py',
                     'TQ_SymNet/models/network.py', 'tools/train.py', 'tools/test.py'):
        path = ROOT / relative
        assert path.is_file(), ('missing local source', relative)
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    for relative in ('mmdet/models/__init__.py', 'mmdet/models/dense_heads/__init__.py'):
        assert 'from mmdet.models.utils.NAMLab' not in (ROOT / relative).read_text(encoding='utf-8-sig')
    for path in (ROOT / 'configs').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_base_' for t in node.targets):
                bases = ast.literal_eval(node.value)
                if isinstance(bases, str):
                    bases = [bases]
                for base in bases:
                    resolved = (path.parent / base).resolve()
                    assert ROOT in resolved.parents and resolved.is_file(), (path.name, base)
    official = json.loads((ROOT / 'OFFICIAL_SYM_TEST.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT / 'tools/test_sym_original.py').read_bytes()).hexdigest() == official['sha256']
    manifest = json.loads((ROOT / 'STANDALONE_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['unchanged_core_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, ('core drift', relative)
    if args.runtime:
        import mmdet
        source = Path(mmdet.__file__).resolve()
        assert ROOT in source.parents, ('wrong installed mmdet imported', str(source))
        from mmdet.models import build_detector
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(imports=['TQ_SymNet.models'], allow_failed_imports=False)
        print('LOCAL_MMDET', source)
    print('PASS: standalone sources, internal config bases, original Sym test, unchanged core hashes')
    if not args.runtime:
        print('NOT TESTED: full MMCV/CUDA import; run --runtime on the server.')


if __name__ == '__main__':
    main()
