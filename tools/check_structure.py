"""Dependency-free syntax/config/preserved-method checks. No CUDA claims."""
import ast
import hashlib
import copy
import json
import builtins
import symtable
import sys
from pathlib import Path


def merge(base, child):
    result = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict):
            value = copy.deepcopy(value)
            replace = value.pop('_delete_', False)
            result[key] = merge({} if replace else result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path):
    namespace = {}
    exec(compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec'), namespace)
    bases = namespace.pop('_base_', [])
    if isinstance(bases, str):
        bases = [bases]
    result = {}
    for base in bases:
        result = merge(result, load_config((path.parent / base).resolve()))
    return merge(result, {k: v for k, v in namespace.items()
                         if not k.startswith('__') and not callable(v)})


def check_preserved_methods(bundle, provenance=None):
    """Compare to checksum-pinned source using ONE parser, not cross-version dumps.

    Python 3.12 adds type_params fields; Python 3.8 also has Index/ExtSlice
    wrappers. A stored ast.dump hash is therefore not a portable source hash.
    The original dump fingerprints remain in PROVENANCE for historical audit.
    """
    bundle = Path(bundle).resolve()
    if provenance is None:
        provenance = json.loads((bundle / 'PROVENANCE.json').read_text(encoding='utf-8'))
    source_path = (bundle / provenance['source']).resolve()
    if bundle not in source_path.parents:
        raise AssertionError('Archived source path escapes the standalone project')
    raw = source_path.read_bytes()
    # Preserve the original byte hash, allowing only documented newline normalization.
    source_text = raw.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    raw_hash = hashlib.sha256(raw).hexdigest()
    text_hash = hashlib.sha256(source_text.encode('utf-8')).hexdigest()
    if (raw_hash != provenance['source_sha256'] and
            text_hash != provenance.get('normalized_source_sha256')):
        raise AssertionError('Archived source checksum mismatch: ' + str(source_path))
    source_tree = ast.parse(source_text, filename=str(source_path))
    owners = {'RetinaNAMGuideAttHeadV3': ('head.py', '_HeadCore'),
              'RetinaNAMGuideAttHeadV44': ('head.py', 'TQSymHead'),
              'RetinaNAMAttV330': ('detector.py', 'TQDetector')}
    trees = {}
    def method(tree, owner, name):
        matches = [node for cls in ast.walk(tree)
                   if isinstance(cls, ast.ClassDef) and cls.name == owner
                   for node in cls.body
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1:
            raise AssertionError((owner, name, 'missing or ambiguous preserved method'))
        return matches[0]
    for entry in provenance['preserved_core_methods']:
        filename, cls = owners[entry['owner']]
        path = bundle / 'TQ_SymNet/models' / filename
        if filename not in trees:
            trees[filename] = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        actual = ast.dump(method(trees[filename], cls, entry['method']), include_attributes=False)
        expected = ast.dump(method(source_tree, entry['owner'], entry['method']), include_attributes=False)
        if actual != expected:
            raise AssertionError((cls, entry['method'], 'preserved implementation changed'))
    return len(provenance['preserved_core_methods'])


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = root
    for path in bundle.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    official = json.loads((bundle / 'OFFICIAL_SYM_TEST.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((bundle / 'tools/test_sym_original.py').read_bytes()).hexdigest() == official['sha256']
    checked = check_preserved_methods(bundle)
    print('PASS: %d preserved methods against checksum-pinned archive; Python %s' %
          (checked, sys.version.split()[0]))
    for filename in ('train_detector.py', 'b0_sym_hard.py', 'b1_sym_soft.py', 'b2_sym_temperature.py',
                     'foundation_transnext_fpn.py', 'train_quality.py', 'a0_full.py',
                     'a1_no_texture.py', 'a2_no_lbp.py', 'a3_no_gabor.py',
                     'a4_no_detail.py', 'a5_no_side_query.py',
                     'a6_no_gaussian.py', 'a7_no_rank.py',
                     'a8_direct_iou.py', 'a9_high_threshold.py', 'a10_light_iou.py',
                     'a11_learned_fusion.py', 'a12_direct_iou_no_rank.py',
                     'test_val.py',
                     'test_subtype.py', 'test_online.py'):
        cfg = load_config(bundle / 'configs' / filename)
        assert cfg['model']['bbox_head']['lung_anatomy_injection'] == 'off'
        assert 'LoadLungMask' not in repr(cfg['data'])
        assert cfg['model']['bbox_head']['positional_encoding']['use_upper_prior'] is False
        assert cfg['model']['test_cfg']['max_per_img'] == 100
        assert cfg['model']['test_cfg']['nms_pre'] == 1000
        assert cfg['model']['bbox_head']['anchor_generator']['strides'] == [4, 8, 16, 32]
        if filename == 'test_online.py':
            assert cfg['data']['test']['ann_file'].endswith('/all_test.json')
            assert cfg['model']['diagnostics'] is False
            assert cfg['data']['test']['test_mode'] is True
        if filename == 'test_subtype.py':
            assert cfg['model']['inference_mode'] == 'subtype'
        assert cfg['custom_imports']['imports'] == ['TQ_SymNet.models']
    assert load_config(bundle / 'configs/b0_sym_hard.py')['model']['inference_nms'] == 'tb_hard'
    assert load_config(bundle / 'configs/b1_sym_soft.py')['model']['inference_nms'] == 'tb_soft'
    assert load_config(bundle / 'configs/b2_sym_temperature.py')['model']['baseline_temperature'] == 2.
    assert load_config(bundle / 'configs/foundation_transnext_fpn.py')['model']['neck']['type'] == 'FPN'
    ablations = {
        'a0_full.py': dict(quality_texture=True, quality_lbp=True,
            quality_gabor=True, quality_gaussian=True, selector_edges=True,
            selector_detail=True, quality_rank_weight=.10,
            selector_rank_weight=.10, selector_quality_target='coco_all_iou'),
        'a1_no_texture.py': dict(quality_texture=False, quality_lbp=False,
            quality_gabor=False),
        'a2_no_lbp.py': dict(quality_lbp=False),
        'a3_no_gabor.py': dict(quality_gabor=False),
        'a4_no_detail.py': dict(selector_detail=False),
        'a5_no_side_query.py': dict(selector_edges=False, selector_detail=False),
        'a6_no_gaussian.py': dict(quality_gaussian=False),
        'a7_no_rank.py': dict(quality_rank_weight=0., selector_rank_weight=0.),
        'a8_direct_iou.py': dict(selector_quality_target='direct_iou'),
        'a9_high_threshold.py': dict(selector_quality_target='high_75_85'),
        'a10_light_iou.py': dict(quality_arch='pooled', selector_quality_target='direct_iou',
                                selector_rank_weight=0.),
        'a11_learned_fusion.py': dict(selector_gate_trainable=True),
        'a12_direct_iou_no_rank.py': dict(selector_quality_target='direct_iou', selector_rank_weight=0.)}
    for filename, expected in ablations.items():
        cfg = load_config(bundle / 'configs' / filename)
        for key, value in expected.items():
            assert cfg['model'][key] == value, (filename, key, cfg['model'][key])
        assert cfg['load_from'] is None, filename
        assert cfg['custom_hooks'][0]['type'] == 'TQSplitContractHook', filename
        assert sum(h['type'] == 'TQDetectorInitHook' for h in cfg['custom_hooks']) == 1, filename
        assert cfg['model']['selector_fusion_init'] == .5, filename
    for path in (bundle / 'TQ_SymNet/models').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert 'NAMLab_att_v3' not in (node.module or ''), path
        symbols = symtable.symtable(path.read_text(encoding='utf-8'), str(path), 'exec')
        def walk(table):
            yield table
            for child in table.get_children():
                yield from walk(child)
        names = {s.get_name() for t in walk(symbols) for s in t.get_symbols()
                 if s.is_referenced() and s.is_global()}
        unresolved = names - {s.get_name() for s in symbols.get_symbols()} - set(dir(builtins))
        assert not unresolved, (path.name, sorted(unresolved))
    # Verify effective fusion, thresholds, schedule and pipelines, not just syntax.
    legacy_path = root / 'configs/NAM/NAM_v64_d3_score_ablation/a2_all_iou_quality_target.py'
    if not legacy_path.exists():
        print('PASS: bundle-only syntax/config/method/import checks; historical config unavailable for contract comparison.')
        return
    old = load_config(legacy_path)
    new = load_config(bundle / 'configs/train_quality.py')
    for key in ('optimizer', 'optimizer_config', 'lr_config', 'runner'):
        assert old[key] == new[key], ('training contract changed', key)
    for key, value in old['model'].items():
        if key.startswith(('quality_', 'selector_')):
            assert value == new['model'][key], ('quality contract changed', key)
    assert old['model']['test_cfg'] == new['model']['test_cfg']
    def without_masks(value):
        if isinstance(value, dict):
            return {k: without_masks(v) for k, v in value.items()}
        if isinstance(value, list):
            return [without_masks(v) for v in value
                    if not (isinstance(v, dict) and v.get('type') == 'LoadLungMask')]
        if isinstance(value, tuple):
            return tuple(v for v in value if v not in ('lung_mask', 'lung_mask_path'))
        return value
    def without_annotation_paths(value):
        if isinstance(value, dict):
            return {k: without_annotation_paths(v) for k, v in value.items() if k != 'ann_file'}
        if isinstance(value, list):
            return [without_annotation_paths(v) for v in value]
        return value
    assert without_annotation_paths(without_masks(old['data']['train'])) == without_annotation_paths(new['data']['train'])
    assert without_annotation_paths(without_masks(old['data']['test'])) == without_annotation_paths(new['data']['test'])
    # Only source annotation paths change for the declared 479/120/200 rerun.
    active = load_config(bundle / 'configs/a0_full.py')
    root = 'data/splits/tbx11k_479_120_200_balanced_v2_seed0/seed_0/'
    assert active['data']['train']['datasets'][0]['ann_file'] == root + 'train.json'
    assert active['data']['train']['datasets'][1]['dataset']['ann_file'] == root + 'train.json'
    assert active['data']['val']['ann_file'] == root + 'val.json'
    assert active['data']['test']['ann_file'] == root + 'test.json'
    print('PASS: syntax, self-contained configs, 16 preserved methods, mask-free pipeline, '
          'candidate/output budgets and no direct monolith imports.')
    print('NOT TESTED: CUDA/MMCV execution, checkpoint loading and numerical equivalence. '
          'Run compare_legacy.py in the server environment.')


if __name__ == '__main__':
    main()
