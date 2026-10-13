"""Publish the new-cohort VAL-best detector under the quality configs' stable name."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from TQ_SymNet.data_contract import load_split_contract, sha256_file, verify_checkpoint_contract
from check_structure import load_config


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def locate_checkpoint(work_dir, checkpoint=None):
    if checkpoint is not None:
        source = project_path(checkpoint)
        if not source.is_file():
            raise FileNotFoundError('Explicit checkpoint does not exist: ' + str(source))
        if not re.fullmatch(r'best_official_tb_bbox_mAP_epoch_\d+\.pth', source.name):
            raise ValueError('Supply the VAL-best best_official_tb_bbox_mAP_epoch_*.pth, not latest.pth')
        return source
    folder = project_path(work_dir)
    candidates = sorted(p for p in folder.glob('best_official_tb_bbox_mAP_epoch_*.pth')
                        if re.fullmatch(r'best_official_tb_bbox_mAP_epoch_\d+\.pth', p.name))
    if len(candidates) != 1:
        nearby = sorted(str(p) for p in (ROOT / 'work_dirs').rglob('best_official_tb_bbox_mAP_epoch_*.pth')
                        if p.parent.name == 'm2_s0')[:20]
        raise RuntimeError(
            'Expected exactly one VAL-best checkpoint in %s, found %d: %s. '
            'The training --work-dir may differ from this directory. '
            'Run tools/select_detector_checkpoint.py --work-dir /actual/M2/work-dir '
            'or --checkpoint /actual/best_official_tb_bbox_mAP_epoch_N.pth. '
            'Nearby M2 files (not automatically selected): %s' %
            (folder, len(candidates), candidates, nearby))
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--work-dir', type=Path, help='Actual M2 training directory')
    source_group.add_argument('--checkpoint', type=Path, help='Explicit VAL-best M2 checkpoint')
    parser.add_argument('--manifest', type=Path, help='Defaults to the active quality config manifest')
    args = parser.parse_args()
    config = load_config(ROOT / 'configs/a0_full.py')
    init_hook = next(h for h in config['custom_hooks'] if h['type'] == 'TQDetectorInitHook')
    target = project_path(init_hook['checkpoint'])
    manifest = project_path(args.manifest or config['split_manifest'])
    source = locate_checkpoint(args.work_dir or target.parent, args.checkpoint)
    import torch
    try:
        checkpoint = torch.load(str(source), map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(source), map_location='cpu')
    contract, _ = load_split_contract(manifest)
    verify_checkpoint_contract(checkpoint.get('meta', {}).get('tq_data_contract'), contract)
    source_hash = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != source_hash:
            raise FileExistsError('Existing selected detector differs: ' + str(target))
    else:
        shutil.copyfile(str(source), str(target))
    if sha256_file(target) != source_hash:
        raise RuntimeError('Checkpoint copy failed integrity check')
    record = dict(selection='validation official_tb_bbox_mAP', source=str(source),
                  selected=str(target), sha256=source_hash, data_contract=contract,
                  checkpoint_epoch=checkpoint.get('meta', {}).get('epoch'))
    (target.parent / 'detector_selection.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
