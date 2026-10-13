"""Real Torch component equivalence without importing unavailable MMCV.

Checks extracted CrossScale, full quality attention/detail and FDR forward and
gradients. Does NOT replace the full server checkpoint/dataset comparison.
"""
import ast
import copy
import importlib.util
import math
from pathlib import Path


def definitions(path, namespace, names=None):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        group = node.body if isinstance(node, ast.If) else [node]
        for entry in group:
            if isinstance(entry, (ast.ClassDef, ast.FunctionDef)) and (
                    names is None or entry.name in names):
                entry = copy.deepcopy(entry)
                entry.decorator_list = []
                exec(compile(ast.Module(body=[entry], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    root = Path(__file__).resolve().parents[1]
    package = root / 'TQ_SymNet/models'
    legacy = root / 'mmdet/models/utils'
    spec = importlib.util.spec_from_file_location('det_ops', root / 'mmdet/utils/deterministic_ops.py')
    det_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(det_ops)
    shared = dict(torch=torch, nn=nn, F=F, math=math, checkpoint=checkpoint,
                  deterministic_bilinear_interpolate=det_ops.deterministic_bilinear_interpolate)
    class Unsupported(nn.Module):
        def __init__(self, *args, **kwargs):
            raise AssertionError('A disabled branch was constructed')
    shared.update(_FreqFusion2x=Unsupported, _C5GatedC2RegressionDetail=Unsupported)
    neck_names = {'_LayerNorm2d', '_DetailPreservingDownsample', '_WindowedSpatialCrossAttn',
                  '_ChannelCrossAttnMatrix', '_CrossScaleOutputRefine', '_TBFusionCrossScale',
                  'TBFusionCrossScaleNeck', 'TQCrossScaleNeck'}
    old_neck = definitions(legacy / 'NAMLab_att_v3.py', shared.copy(), neck_names)
    new_neck = definitions(package / 'neck.py', shared.copy(), neck_names)
    def quality(directory, modern):
        scope = shared.copy()
        definitions(directory / ('attention.py' if modern else 'tb_quality_attention.py'), scope)
        definitions(directory / ('edges.py' if modern else 'tb_edge_selector.py'), scope,
                    {'edge_bias', 'EdgeQualityAttention'})
        definitions(directory / ('detail.py' if modern else 'tb_detail_selector.py'), scope,
                    {'DetailResidualBlock', 'NativeDetailEncoder', 'edge_anchor_points', 'DetailQualityAttention'})
        return scope
    old_q, new_q = quality(legacy, False), quality(package, True)
    old_dfr = definitions(legacy / 'NAMLab_att_v3.py', shared.copy(),
                          {'_distribution_refinement_expectation', '_symmetric_distribution_focal_loss'})
    new_dfr = definitions(package / 'distribution.py', shared.copy())
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    def equal(a, b):
        if isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    def gradient_equal(a, b):
        for (an, av), (bn, bv) in zip(a.named_parameters(), b.named_parameters()):
            assert an == bn
            assert (av.grad is None) == (bv.grad is None), an
            if av.grad is not None:
                equal(av.grad, bv.grad)
    for device in devices:
        torch.manual_seed(123)
        a = old_neck['TBFusionCrossScaleNeck'](in_channels=(4,8,12,16), out_channels=8).to(device)
        torch.manual_seed(123)
        b = new_neck['TQCrossScaleNeck'](in_channels=(4,8,12,16), out_channels=8).to(device)
        assert a.state_dict().keys() == b.state_dict().keys()
        for key in a.state_dict():
            assert torch.equal(a.state_dict()[key], b.state_dict()[key]), key
        features = [torch.randn(1, c, h, h, device=device) for c,h in ((4,16),(8,8),(12,4),(16,2))]
        ao, bo = a(features), b(features)
        equal(ao, bo)
        sum(x.square().mean() for x in ao).backward()
        sum(x.square().mean() for x in bo).backward()
        gradient_equal(a,b)
        torch.manual_seed(123)
        a = old_q['DetailQualityAttention'](hidden=16, heads=4, edges=True, detail=True,
            texture=True, gaussian=True, chunk=2, checkpoint_candidates=False).to(device).train()
        torch.manual_seed(123)
        b = new_q['DetailQualityAttention'](hidden=16, heads=4, edges=True, detail=True,
            texture=True, gaussian=True, chunk=2, checkpoint_candidates=False).to(device).train()
        assert a.state_dict().keys() == b.state_dict().keys()
        for key in a.state_dict():
            assert torch.equal(a.state_dict()[key], b.state_dict()[key]), key
        with torch.no_grad():
            a.high_head.weight.normal_(std=.03)
            b.load_state_dict(a.state_dict(), strict=True)
        boxes = torch.tensor([[.1,.1,.8,.9],[.5,.5,.9,.9],[.2,.25,.4,.5]],device=device)
        maps = [torch.rand(1,256,h,h,device=device) for h in (8,4,2,1)]
        gray = torch.rand(1,1,32,32,device=device)
        sym = torch.tensor([[.8,.2,0.],[.5,.4,0.],[.2,.7,0.]],device=device)
        am, bm = a.prepare(maps,gray,(32,32),(32,32)), b.prepare(maps,gray,(32,32),(32,32))
        ao, bo = a(boxes,am,sym), b(boxes,bm,sym)
        equal(ao,bo)
        ao[0].square().mean().backward(); bo[0].square().mean().backward()
        gradient_equal(a,b)
        empty = boxes[:0]
        equal(a(empty,am,sym[:0]), b(empty,bm,sym[:0]))
        a.eval(); b.eval()
        with torch.no_grad():
            b.chunk=1
            am, bm = a.prepare(maps,gray,(32,32),(32,32)), b.prepare(maps,gray,(32,32),(32,32))
            equal(a(boxes,am,sym), b(boxes,bm,sym))
        logits=torch.randn(1,9*4*17,2,2,device=device)
        equal(old_dfr['_distribution_refinement_expectation'](logits,9,16),
              new_dfr['_distribution_refinement_expectation'](logits,9,16))
        print('PASS',device,'CrossScale initialization/forward/gradient; full texture+edge+native-detail '
              'quality initialization/forward/gradient; empty candidates; chunks; FDR decode')
    print('Full MMDetection/Sym/CUDA-extension checkpoint/AP equivalence still requires compare_legacy.py.')


if __name__ == '__main__':
    main()
