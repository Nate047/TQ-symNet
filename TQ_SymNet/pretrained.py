"""Strict generic-backbone initialization, independent of MMCV serialization rules."""
import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch


def read_pretrained_state(filename):
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError('P2T ImageNet checkpoint missing: ' + str(path.resolve()))
    try:
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch 1.10 server
        checkpoint = torch.load(str(path), map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError('Expected a state dictionary or a state_dict/model checkpoint')
    wrappers = [key for key in ('state_dict', 'model') if isinstance(checkpoint.get(key), Mapping)]
    if len(wrappers) > 1:
        raise RuntimeError('Ambiguous checkpoint containing both state_dict and model')
    state = checkpoint[wrappers[0]] if wrappers else checkpoint
    normalized = {}
    for key, value in state.items():
        name = key[7:] if key.startswith('module.') else key
        if name in normalized:
            raise RuntimeError('Duplicate normalized pretrained key: ' + name)
        normalized[name] = value
    return normalized, wrappers[0] if wrappers else 'plain'


def load_checked_pretrained(module, filename, logger=None):
    state, layout = read_pretrained_state(filename)
    expected = module.state_dict()
    missing = sorted(set(expected)-set(state))
    # ImageNet's final classifier/norm are not used by the official dense backbone.
    allowed_extra = {'head.weight', 'head.bias', 'norm.weight', 'norm.bias'}
    unexpected = sorted(set(state)-set(expected)-allowed_extra)
    shape_errors = [key for key,value in expected.items() if key in state and
                    (not isinstance(state[key],torch.Tensor) or state[key].shape != value.shape)]
    if missing or unexpected or shape_errors:
        raise RuntimeError('P2T pretrained mismatch: missing=%s unexpected=%s shape=%s. '
            'Use the ImageNet P2T-Small checkpoint, not a full TB/COCO detector or another P2T size.' %
            (missing[:12],unexpected[:12],shape_errors[:12]))
    if not expected:
        raise RuntimeError('Empty backbone state; cannot audit pretraining')
    for key in expected:
        if state[key].is_floating_point() and not torch.isfinite(state[key]).all():
            raise RuntimeError('Non-finite pretrained tensor: ' + key)
    module.load_state_dict({key:state[key] for key in expected},strict=True)
    mismatched = [key for key,value in module.state_dict().items()
                  if not torch.equal(value.detach().cpu(),state[key].to(dtype=value.dtype))]
    if mismatched:
        raise RuntimeError('P2T weights differ after loading: ' + repr(mismatched[:12]))
    digest = hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    report = dict(file=str(Path(filename).resolve()),sha256=digest.hexdigest(),layout=layout,
                  matched_keys=len(expected),total_keys=len(expected),loaded_fraction=1.0,
                  ignored_classifier_keys=sorted(set(state)-set(expected)))
    if logger is not None:
        logger.info('[TQ-PRETRAIN] P2T-Small loaded=%d/%d (100%%) layout=%s sha256=%s file=%s',
                    len(expected),len(expected),layout,report['sha256'],report['file'])
    return report
