"""Prepare local official weights for the capacity-screening configs.

No detector weights are loaded. Requires explicit downloaded inputs.
Tiny is unwrapped into a backbone-only state_dict; Swin stays in its original
format because MMDet's convert_weights=True performs that conversion.
"""
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from TQ_SymNet.capacity_pretrained import load_official_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tiny', required=True, help='Official TransNeXt Tiny-224 .pth')
    p.add_argument('--out', default='pretrained/transnext_tiny_224_backbone.pth')
    args = p.parse_args()
    src, dst = Path(args.tiny), Path(args.out)
    if src.resolve() == dst.resolve() or dst.exists():
        raise FileExistsError('Use a new output path; input/output are never overwritten')
    # Old server PyTorch may not expose weights_only. Use only trusted official files.
    obj = load_official_checkpoint(src)
    for key in ('state_dict', 'model'):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
    state = {}
    for name, value in obj.items():
        name = name.removeprefix('module.') if hasattr(str, 'removeprefix') else (
            name[7:] if name.startswith('module.') else name)
        if name.startswith('backbone.'):
            name = name[9:]
        if not name.startswith(('head.', 'relative_pos_index', 'relative_coords_table')):
            state[name] = value
    expected_depths = [2,2,15,2]
    for stage, depth in enumerate(expected_depths, 1):
        indices = {int(k.split('.')[1]) for k in state if k.startswith('block%d.' % stage)}
        if indices != set(range(depth)):
            raise ValueError('Not official Tiny depths: stage %d = %s' % (stage, indices))
    if tuple(state['patch_embed1.proj.weight'].shape) != (72,3,7,7):
        raise ValueError('Unexpected Tiny patch embedding')
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state), str(dst))
    print('Saved', dst, 'tensors=',len(state))


if __name__ == '__main__':
    main()
