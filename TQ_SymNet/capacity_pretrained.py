"""Strict loading for TRUSTED official capacity-screen checkpoints.

No dependency on MMCV; supports historical torch1.10 as well as newer torch.
Explicit weights_only=False is necessary for official checkpoints containing
training metadata. Never use this helper on an untrusted downloaded pickle.
"""
import inspect
from pathlib import Path
import torch


def load_official_checkpoint(path, loader=None):
    loader = torch.load if loader is None else loader
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(loader).parameters:
        kwargs['weights_only'] = False
    return loader(str(Path(path)), **kwargs)


def strict_load_backbone(net, state, allowed_new=(), regenerated=()):
    """All parameters AND persistent buffers must load, barring named exceptions.

    Old BN checkpoints may omit only the non-statistical batch counter; reset
    that counter explicitly. Running mean/variance are never optional.
    """
    target = net.state_dict()
    state = dict(state)
    mismatches = [k for k,v in state.items() if k in target and
                  tuple(v.shape) != tuple(target[k].shape)]
    missing, reset_counters, new_keys = [], [], []
    for name, value in target.items():
        if name in state:
            continue
        if name.endswith('num_batches_tracked'):
            state[name] = torch.zeros_like(value)
            reset_counters.append(name)
        elif name.startswith(allowed_new) or name in regenerated:
            new_keys.append(name)
        else:
            missing.append(name)
    invalid_bn = [k for k,v in state.items() if k in target and
        (k.endswith('running_mean') or k.endswith('running_var')) and
        (not bool(torch.isfinite(v).all()) or
         (k.endswith('running_var') and bool((v < 0).any())))]
    if mismatches or missing or invalid_bn:
        raise RuntimeError('Incomplete/invalid backbone pretraining: missing=%s '
            'mismatched=%s invalid_bn=%s' % (missing[:12],mismatches[:12],invalid_bn[:12]))
    net.load_state_dict(state, strict=False)
    params = dict(net.named_parameters())
    return dict(loaded_parameters=sum(v.numel() for k,v in params.items() if k in state),
        total_parameters=sum(v.numel() for v in params.values()),
        loaded_persistent_buffers=sum(1 for k in target if k not in params and k in state),
        reset_batch_counters=reset_counters, new_or_regenerated_keys=new_keys)
