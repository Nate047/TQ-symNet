"""Build an explicit small delta bundle; never bundle data, weights or .git."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGED = ['TQ_SymNet/models/light_quality.py', 'TQ_SymNet/models/network.py',
           'tools/analysis_tools/sweep_scalar_alpha.py', 'tools/benchmark_cost.py',
           'mmdet/datasets/coco.py', 'tools/export_online.py']
EXPLICIT = CHANGED + [
    'TQ_SymNet/supplement.py', 'TQ_SymNet/supplement_math.py', 'TQ_SymNet/bootstrap_ap.py',
    'TQ_SymNet/supplement_smoke.py',
    'TQ_SymNet/models/gflv2_adapter.py',
    'tools/analysis_tools/export_supplement_candidates.py',
    'tools/check_publication_supplement.py', 'tools/test_publication_supplement.py',
    'tools/test_publication_fixes.py',
    'tools/verify_publication_supplement.py', 'tools/package_publication_supplement.py',
    'configs/publication_alpha/_original_a0.py', 'RUN_PUBLICATION_SUPPLEMENT.md',
    'PUBLICATION_SUPPLEMENT_AUDIT.json']
DEPENDENCIES = [
    'TQ_SymNet/models/attention.py', 'TQ_SymNet/models/edges.py',
    'TQ_SymNet/models/detail.py', 'TQ_SymNet/models/candidates.py',
    'TQ_SymNet/models/detector.py', 'TQ_SymNet/models/head.py', 'TQ_SymNet/models/neck.py',
    'TQ_SymNet/models/__init__.py', 'TQ_SymNet/checkpoints.py',
    'TQ_SymNet/alpha_experiment.py', 'TQ_SymNet/data_contract.py',
    'mmdet/models/dense_heads/gfl_head.py', 'mmdet/models/dense_heads/vfnet_head.py',
    'mmdet/models/dense_heads/anchor_head.py', 'mmdet/models/losses/gfocal_loss.py',
    'mmdet/models/losses/varifocal_loss.py',
    'mmdet/core/evaluation/eval_hooks.py', 'mmdet/core/hook/ema.py',
    'tools/train.py', 'tools/test.py', 'tools/check_structure.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--revision', default='r2')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.revision):
        raise ValueError('Invalid revision label')
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise RuntimeError('Output already exists; use a new directory to preserve previous artifacts')
    out.mkdir(parents=True)
    paths = sorted(set(EXPLICIT + [str(x.relative_to(ROOT)).replace('\\','/')
        for folder in ('configs/publication_supplement', 'third_party/gfocalv2_reference')
        for x in (ROOT/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts]))
    def record(path):
        data = (ROOT/path).read_bytes()
        return dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest = dict(date='2026-10-26', revision=args.revision,
        type='delta_for_existing_standalone_TQ_SymNet',
        files=[record(x) for x in paths], dependencies=[record(x) for x in DEPENDENCIES],
        protocol='599/200, not 479/120/200',
        validation='Local numerical/config tests passed; full MMDet/CUDA smoke must pass on server.',
        changed_existing=CHANGED, data_and_weights_included=False)
    git = ['git', '-c', 'safe.directory='+ROOT.as_posix(), '-C', str(ROOT)]
    try:
        manifest['base_commit'] = subprocess.check_output(git+['rev-parse','HEAD'], text=True).strip()
        for path in CHANGED:
            previous = subprocess.check_output(git+['show','HEAD:'+path])
            backup = out/'rollback'/path; backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(previous)
    except (subprocess.CalledProcessError, FileNotFoundError):
        manifest['rollback_note'] = 'Git HEAD backup unavailable; back up server files before unpacking.'
    payload = out/'payload'; payload.mkdir()
    for path in paths:
        target = payload/path; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, target)
    meta = json.dumps(manifest, indent=2)
    (payload/'publication_supplement_manifest.json').write_text(meta, encoding='utf-8')
    (out/'publication_supplement_manifest.json').write_text(meta, encoding='utf-8')
    archive = out/('TQSymNet_publication_supplement_20261026_'+args.revision+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(payload).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(paths)+1
    print(json.dumps(dict(archive=str(archive), files=len(paths), bytes=archive.stat().st_size,
                         sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
