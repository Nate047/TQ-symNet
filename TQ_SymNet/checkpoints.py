"""Explicit checkpoint contract; never silently skip active model parameters."""


_REBUILDABLE_POSITION_KEYS = frozenset(
    'backbone.%s%d' % (kind, level)
    for kind in ('relative_pos_index', 'relative_coords_table')
    for level in range(1, 5))


def _normalized_state(state):
    normalized = {}
    for name, value in state.items():
        key = name[7:] if name.startswith('module.') else name
        if key in normalized:
            raise RuntimeError('Checkpoint contains duplicate normalized key: ' + key)
        normalized[key] = value
    return normalized


def load_detector_state(model, state):
    """Load only backbone/neck/head. Quality weights are never transferred."""
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    prefixes = ('backbone.', 'neck.', 'bbox_head.')
    expected = {k: v for k, v in model.state_dict().items() if k.startswith(prefixes)}
    missing = sorted(set(expected) - set(state))
    mismatched = [k for k in expected if k in state and state[k].shape != expected[k].shape]
    if missing or mismatched:
        raise RuntimeError('Detector checkpoint mismatch: missing=%s shape=%s' % (missing, mismatched))
    if not expected:
        raise RuntimeError('No detector parameters found')
    model.load_state_dict({k: state[k] for k in expected}, strict=False)
    return dict(loaded_detector_keys=len(expected),
                discarded_quality_keys=sum(k.startswith('candidate_quality.') for k in state),
                discarded_other_keys=len(state)-len(expected))


def load_detector_checkpoint(model, filename):
    from mmcv.runner.checkpoint import _load_checkpoint
    checkpoint = _load_checkpoint(filename, map_location='cpu')
    audit = load_detector_state(model, checkpoint.get('state_dict', checkpoint))
    return checkpoint, audit


def load_retained_checkpoint(model, filename):
    # Read first, validate before mutating the model. The old load_checkpoint
    # call loaded weights before the compatibility audit raised its exception.
    from mmcv.runner.checkpoint import _load_checkpoint
    checkpoint = _load_checkpoint(filename, map_location='cpu')
    audit = load_retained_state(model, checkpoint.get('state_dict', checkpoint))
    return checkpoint, audit


def load_retained_state(model, state):
    """Load active tensors, accepting only proven redundant position caches.

    Older serializers may include TransNeXt coordinate tables / indices that
    this backbone registers as non-persistent. These tensors enter attention,
    so ignoring a DIFFERENT table is unsafe: require equality to the buffers
    already reconstructed by the current config before discarding old keys.
    Learned CPB layers/positional parameters are never covered by this rule.
    """
    import torch
    state = _normalized_state(state)
    expected_state = model.state_dict()
    keys = set(state)
    expected = set(expected_state)
    # Added after the selected MAX run; it is not read in MAX inference/training.
    allowed_missing = ({'candidate_quality.tb_lse_temperature_raw'}
                       if getattr(model, 'selector_tb_fusion', 'max') == 'max' else set())
    missing = expected - keys
    unknown_missing = missing - allowed_missing
    # Historical EMA or disabled refiner keys do not enter this model's forward.
    unexpected = keys - expected
    modules = dict(model.named_modules())
    rebuilt = set()
    cache_mismatches = []
    for name in sorted(unexpected & _REBUILDABLE_POSITION_KEYS):
        module_name, buffer_name = name.rsplit('.', 1)
        owner = modules.get(module_name)
        if (owner is None or buffer_name not in owner._buffers
                or buffer_name not in owner._non_persistent_buffers_set):
            continue  # Not a known regenerated buffer: strict audit rejects it.
        actual, saved = owner._buffers[buffer_name], state[name]
        if not isinstance(saved, torch.Tensor) or not isinstance(actual, torch.Tensor):
            cache_mismatches.append(name + ': expected tensors')
            continue
        if saved.shape != actual.shape or saved.dtype != actual.dtype:
            cache_mismatches.append('%s: checkpoint %s/%s, rebuilt %s/%s' %
                (name, tuple(saved.shape), saved.dtype, tuple(actual.shape), actual.dtype))
            continue
        a, b = saved.detach().cpu(), actual.detach().cpu()
        equal = (torch.allclose(a, b, rtol=1e-5, atol=1e-6)
                 if a.is_floating_point() else torch.equal(a, b))
        if not equal:
            cache_mismatches.append(name + ': rebuilt values differ')
        else:
            rebuilt.add(name)
    mismatched = sorted(name for name in expected & keys
                        if not isinstance(state[name], torch.Tensor)
                        or state[name].shape != expected_state[name].shape)
    unknown_unexpected = {name for name in unexpected if not name.startswith(
        ('ema_', 'local_refiner.', 'side_refiner.')) and name not in rebuilt}
    if unknown_missing or unknown_unexpected or mismatched or cache_mismatches:
        raise RuntimeError('Checkpoint is not a compatible retained model: '
            'missing=%s unexpected=%s shape=%s position_cache=%s' %
            (sorted(unknown_missing), sorted(unknown_unexpected), mismatched, cache_mismatches))
    result = model.load_state_dict({name: state[name] for name in expected & keys}, strict=False)
    if set(result.missing_keys) - allowed_missing or result.unexpected_keys:
        raise RuntimeError('Validated checkpoint failed model load: %s' % (result,))
    return dict(allowed_inactive_missing=sorted(missing),
                ignored_inactive_keys=sorted(unexpected - rebuilt),
                ignored_rebuilt_position_caches=sorted(rebuilt))
