"""CPU smoke check for frozen encoders and the new classification read-outs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from TQ_SymNet.classification import FrozenPyramidClassifier, state_sha256
from TQ_SymNet.classification_heads import CrossScaleClassQueryHead, CrossScaleHybridPoolHead


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, image):
        p2 = self.projection(image)
        return tuple(F.avg_pool2d(p2, 2 ** level) for level in range(4))


class DummyNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4))

    def forward(self, features):
        return tuple(feature * self.scale[level] for level, feature in enumerate(features))


def check_query_ablation_contract():
    torch.manual_seed(17)
    fixed = CrossScaleClassQueryHead(in_channels=16, embed_dim=32, scale_fusion='fixed')
    fixed_rng = torch.get_rng_state().clone()
    torch.manual_seed(17)
    learned = CrossScaleClassQueryHead(in_channels=16, embed_dim=32, scale_fusion='learned')
    assert torch.equal(fixed_rng, torch.get_rng_state()), 'Optional gate changes global RNG'
    fixed_state, learned_state = fixed.state_dict(), learned.state_dict()
    for key, value in fixed_state.items():
        assert torch.equal(value, learned_state[key]), ('shared initialization differs', key)
    pyramid = [torch.randn(2, 16, size, size) for size in (16, 8, 4, 2)]
    fixed.eval()
    learned.eval()
    with torch.no_grad():
        torch.testing.assert_close(fixed(pyramid), learned(pyramid))
    # Normalization at one spatial position must not depend on other positions.
    changed = pyramid[1].clone()
    changed[:, :, -1, -1] += 100.
    with torch.no_grad():
        torch.testing.assert_close(fixed.projections[0](pyramid[1])[:, :, 0, 0],
                                   fixed.projections[0](changed)[:, :, 0, 0])
    # The zero final gate layer deliberately blocks upstream gate gradients at
    # initialization. After its first update, level identity must affect output.
    optimizer = torch.optim.SGD(learned.parameters(), lr=0.1)
    target = torch.tensor([0, 2])
    F.cross_entropy(learned(pyramid), target).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(learned(pyramid), target).backward()
    gradient = learned.level_embeddings.grad
    assert gradient is not None and gradient.abs().max() > 1e-8, 'Dead scale embeddings'
    with torch.no_grad():
        before = learned(pyramid)
        learned.level_embeddings.add_(torch.randn_like(learned.level_embeddings))
        after = learned(pyramid)
    assert (before - after).abs().max() > 1e-7, 'Scale identity does not affect logits'
    print('PASS: C2/C3 shared initialization/RNG/output, per-position norm, live scale embeddings')


def main():
    check_query_ablation_contract()
    torch.manual_seed(1)
    image = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 2])
    variants = [('hybrid_pool', CrossScaleHybridPoolHead(in_channels=16, hidden=32)),
                ('query_fixed', CrossScaleClassQueryHead(
                    in_channels=16, embed_dim=32, heads=4, scale_fusion='fixed')),
                ('query_learned', CrossScaleClassQueryHead(
                    in_channels=16, embed_dim=32, heads=4, scale_fusion='learned'))]
    for name, head in variants:
        model = FrozenPyramidClassifier(DummyBackbone(), DummyNeck(), head).train()
        backbone_hash, neck_hash = state_sha256(model.backbone), state_sha256(model.neck)
        logits = model(image)
        assert logits.shape == (2, 3) and torch.isfinite(logits).all(), name
        F.cross_entropy(logits, labels).backward()
        assert not model.backbone.training and not model.neck.training, name
        assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters()), name
        assert all(p.grad is None and not p.requires_grad for p in model.neck.parameters()), name
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in head.parameters()), name
        assert state_sha256(model.backbone) == backbone_hash, name
        assert state_sha256(model.neck) == neck_hash, name
        model.eval()
        head.collect_debug = True
        with torch.no_grad():
            assert model(image[:1]).shape == (1, 3), name
        if name.startswith('query'):
            gates = torch.tensor(head.last_debug['scale_gate_by_class'])
            torch.testing.assert_close(gates.sum(-1), torch.ones(3))
            assert len(head.last_debug['attention_entropy_by_level']) == 3
            if name == 'query_learned':
                torch.testing.assert_close(gates, torch.full((3, 3), 1. / 3.))
        print('PASS:', name, '3-class output, trainable head, frozen features, one-image inference')


if __name__ == '__main__':
    main()
