"""CPU contracts for C7-C13; --require-clahe also checks installed OpenCV.

Run check_classification_runtime.py on the server for the real CUDA encoder.
"""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from TQ_SymNet.classification import FrozenPyramidClassifier, state_sha256, classification_clahe
from TQ_SymNet.classification_heads import CrossScaleClassQueryHead
from TQ_SymNet.classification_priors import PriorClassQueryHead, UniformLBP, GaborEnergy


def resolve(path):
    """Small resolver for smoke tests only; production uses MMCV.Config."""
    child = {k: v for k, v in runpy.run_path(str(path)).items() if not k.startswith('__')}
    result = {}
    for base in child.pop('_base_', []):
        merge(result, resolve(path.parent / base))
    merge(result, child)
    return result


def merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            value = dict(value)
            delete = value.pop('_delete_', False)
            if delete or key not in target:
                target[key] = {}
            merge(target[key], value)
        else:
            target[key] = value


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 1)
    def forward(self, images):
        feature = self.conv(images)
        return tuple(F.avg_pool2d(feature, 2**i) for i in (1, 2, 3, 4))


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
    def forward(self, features):
        return tuple(x*self.scale for x in features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-clahe', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(137)
    base = CrossScaleClassQueryHead(in_channels=16, embed_dim=32, scale_fusion='learned')
    base_rng = torch.get_rng_state().clone()
    raw = torch.rand(2, 1, 32, 32).expand(-1, 3, -1, -1)*255
    mean = torch.tensor([123.675, 116.28, 103.53])[None, :, None, None]
    std = torch.tensor([58.395, 57.12, 57.375])[None, :, None, None]
    image = (raw - mean) / std
    labels = torch.tensor([0, 2])
    for sym, lbp, gabor in ((False, False, False), (True, False, False),
                            (False, True, False), (False, False, True),
                            (False, True, True), (True, True, True)):
        torch.manual_seed(137)
        head = PriorClassQueryHead(in_channels=16, embed_dim=32, scale_fusion='learned',
                                  use_sym=sym, use_lbp=lbp, use_gabor=gabor, texture_grid=8)
        assert torch.equal(base_rng, torch.get_rng_state()), 'Optional initialization changes RNG'
        for name, value in base.state_dict().items():
            assert torch.equal(value, head.state_dict()[name]), name
        model = FrozenPyramidClassifier(Backbone(), Neck(), head).train()
        hashes = [state_sha256(model.backbone), state_sha256(model.neck)]
        optimizer = torch.optim.SGD(head.parameters(), lr=.01)
        head.collect_debug = True
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(image)
            assert output.shape == (2, 3) and torch.isfinite(output).all()
            F.cross_entropy(output, labels).backward()
            for name, parameter in head.named_parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all(), name
            optimizer.step()
        for prefix in (['sym.'] if sym else []) + (['lbp.'] if lbp else []) + (['gabor.'] if gabor else []):
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for n, p in head.named_parameters() if n.startswith(prefix)), prefix
        if gabor:
            assert head.gabor.theta.grad.abs().sum() > 0
            assert head.gabor.wave_raw.grad.abs().sum() > 0
        if lbp:
            assert head.lbp.bin_logits.grad.abs().sum() > 0
        for module, before in zip((model.backbone, model.neck), hashes):
            assert not module.training and state_sha256(module) == before
            assert all(p.grad is None and not p.requires_grad for p in module.parameters())
        model.eval()
        with torch.no_grad():
            output = model(image[:1])
            clone = PriorClassQueryHead(in_channels=16, embed_dim=32, scale_fusion='learned',
                                       use_sym=sym, use_lbp=lbp, use_gabor=gabor, texture_grid=8).eval()
            clone.load_state_dict(head.state_dict(), strict=True)
            pyramid = model.neck(model.backbone(image[:1]))
            torch.testing.assert_close(output, clone(pyramid, images=image[:1]))
        gate = torch.tensor(head.last_debug['prior_gate_by_class'])
        torch.testing.assert_close(gate.sum(-1), torch.ones(3))
        print('PASS: sym/lbp/gabor=', sym, lbp, gabor, 'frozen encoders, live gradients, strict reload')
    lbp = UniformLBP(8)
    assert len(lbp.mapping.unique()) == 59
    assert int((lbp.mapping == 58).sum()) == 198
    kernels = GaborEnergy(8).kernels()
    torch.testing.assert_close(kernels.mean((-2, -1)), torch.zeros(16, 1), atol=1e-7, rtol=0)
    torch.testing.assert_close(kernels.square().sum((-2, -1)), torch.ones(16, 1))
    for path in sorted((ROOT/'configs/classification').glob('c*.py')):
        cfg = resolve(path)
        assert cfg['data']['expected_train_images'] == 6888
        if cfg.get('classification_head') == 'prior_query':
            PriorClassQueryHead(**cfg['classification_options'])
    print('PASS: config inheritance, 59-bin mapping, normalized DC-free quadrature kernels')
    try:
        import cv2  # noqa: F401
    except ImportError:
        if args.require_clahe:
            raise
        print('SKIP: OpenCV unavailable; run --require-clahe on server')
    else:
        array = np.random.RandomState(1).randint(0, 256, (64, 64), dtype=np.uint8)
        array = np.repeat(array[:, :, None], 3, axis=2)
        a = classification_clahe(array, dict(clip_limit=2., tile_grid_size=(8, 8)))
        b = classification_clahe(array, dict(clip_limit=2., tile_grid_size=(8, 8)))
        assert np.array_equal(a, b) and np.array_equal(a[:, :, 0], a[:, :, 2])
        assert a.dtype == np.uint8 and a.shape == array.shape
        print('PASS: deterministic grayscale CLAHE')


if __name__ == '__main__':
    main()
