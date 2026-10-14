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
from TQ_SymNet.classification_consistency import (
    brightness_contrast_view, consistency_kl, paired_view_logits)


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


def check_temperature_contract():
    torch.manual_seed(29)
    baseline = CrossScaleClassQueryHead(in_channels=16, embed_dim=32,
                                       scale_fusion='learned', dropout=0.)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(29)
    adaptive = CrossScaleClassQueryHead(in_channels=16, embed_dim=32,
                                       scale_fusion='learned', dropout=0.,
                                       attention_temperature=True)
    assert torch.equal(baseline_rng, torch.get_rng_state()), 'Temperature changes shared RNG'
    for name, parameter in baseline.state_dict().items():
        torch.testing.assert_close(parameter, adaptive.state_dict()[name], rtol=0., atol=0.)
    images = [torch.randn(2, 16, side, side) for side in (16, 8, 4, 2)]
    baseline.eval()
    adaptive.eval()
    with torch.no_grad():
        torch.testing.assert_close(adaptive(images), baseline(images), rtol=1e-5, atol=1e-6)
    adaptive.collect_debug = True
    with torch.no_grad():
        adaptive(images)
    torch.testing.assert_close(torch.tensor(adaptive.last_debug['attention_temperature_by_class']),
                               torch.ones(3))
    adaptive.zero_grad(set_to_none=True)
    F.cross_entropy(adaptive(images), torch.tensor([0, 2])).backward()
    assert adaptive.temperature_head[-1].weight.grad is not None
    assert adaptive.temperature_head[-1].weight.grad.abs().max() > 0
    with torch.no_grad():
        before = adaptive(images)
        adaptive.temperature_head[-1].bias.fill_(0.5)
        after = adaptive(images)
    assert (before - after).abs().max() > 1e-7, 'Temperature does not change query features'
    print('PASS: C3/C4 common state/RNG/logits, live bounded attention temperature')


def check_consistency_contract():
    images = torch.randn(3, 3, 8, 8)
    mean, std = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
    torch.manual_seed(5)
    altered = brightness_contrast_view(images, mean, std, 5., .05)
    torch.manual_seed(5)
    repeated = brightness_contrast_view(images, mean, std, 5., .05)
    assert torch.equal(altered, repeated)
    assert altered.shape == images.shape and not torch.equal(altered, images)
    low = -torch.tensor(mean) / torch.tensor(std)
    high = (255. - torch.tensor(mean)) / torch.tensor(std)
    assert torch.all(altered >= low[None, :, None, None] - 1e-6)
    assert torch.all(altered <= high[None, :, None, None] + 1e-6)
    student = torch.randn(3, 3, requires_grad=True)
    teacher = torch.randn(3, 3, requires_grad=True)
    loss = consistency_kl(student, teacher)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None, 'Reference logits must be stop-gradient'
    # Raw grayscale must remain grayscale despite channel-specific normalization.
    raw_gray = torch.rand(3, 1, 8, 8).expand(-1, 3, -1, -1) * 255.
    mean_tensor = torch.tensor(mean)[None, :, None, None]
    std_tensor = torch.tensor(std)[None, :, None, None]
    normalized = (raw_gray - mean_tensor) / std_tensor
    changed_gray = brightness_contrast_view(normalized, mean, std) * std_tensor + mean_tensor
    torch.testing.assert_close(changed_gray[:, 0], changed_gray[:, 1], atol=3e-5, rtol=1e-6)
    torch.testing.assert_close(changed_gray[:, 0], changed_gray[:, 2], atol=3e-5, rtol=1e-6)
    # A zero image perturbation gives zero disagreement even with dropout, and
    # the second view must leave the next training iteration's RNG untouched.
    probe = nn.Sequential(nn.Flatten(), nn.Linear(192, 16), nn.ReLU(),
                          nn.Dropout(0.5), nn.Linear(16, 3)).train()
    torch.manual_seed(71)
    reference = probe(normalized)
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    original, changed = paired_view_logits(
        probe, normalized, mean, std, 0., 0., torch.Generator().manual_seed(1000074))
    torch.testing.assert_close(original, reference, atol=0., rtol=0.)
    torch.testing.assert_close(original, changed, atol=0., rtol=0.)
    assert torch.equal(reference_rng, torch.get_rng_state()), 'Second view shifts dropout RNG'
    print('PASS: grayscale intensity perturbation, shared dropout, isolated RNG, student-only KL')


def main():
    check_query_ablation_contract()
    check_temperature_contract()
    check_consistency_contract()
    torch.manual_seed(1)
    image = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 2])
    variants = [('hybrid_pool', CrossScaleHybridPoolHead(in_channels=16, hidden=32)),
                ('query_fixed', CrossScaleClassQueryHead(
                    in_channels=16, embed_dim=32, heads=4, scale_fusion='fixed')),
                ('query_learned', CrossScaleClassQueryHead(
                    in_channels=16, embed_dim=32, heads=4, scale_fusion='learned')),
                ('query_temperature', CrossScaleClassQueryHead(
                    in_channels=16, embed_dim=32, heads=4, scale_fusion='learned',
                    attention_temperature=True))]
    for name, head in variants:
        model = FrozenPyramidClassifier(DummyBackbone(), DummyNeck(), head).train()
        backbone_hash, neck_hash = state_sha256(model.backbone), state_sha256(model.neck)
        if name == 'query_temperature':
            logits, changed_logits = paired_view_logits(
                model, image, (123.675, 116.28, 103.53), (58.395, 57.12, 57.375),
                generator=torch.Generator().manual_seed(1000004))
            extra_loss = .1 * consistency_kl(changed_logits, logits)
        else:
            logits = model(image)
            extra_loss = logits.new_zeros(())
        assert logits.shape == (2, 3) and torch.isfinite(logits).all(), name
        (F.cross_entropy(logits, labels) + extra_loss).backward()
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
