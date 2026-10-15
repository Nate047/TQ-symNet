"""CPU integration checks for freeze policies, accumulation and encoder snapshots."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from TQ_SymNet.classification import (TunablePyramidClassifier, state_sha256,
    classification_parameter_groups, restore_classification_snapshot, encoder_policy)
from train_classification import update_batches, audit_encoders
from check_classification_priors import resolve


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
    def forward(self, images):
        x = self.conv(images)
        return tuple(F.avg_pool2d(x, 2**level) for level in range(4))


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 1)
    def forward(self, pyramid):
        return tuple(self.conv(x) for x in pyramid)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 3)
    def forward(self, pyramid):
        return self.linear(torch.cat([x.mean((-2, -1)) for x in pyramid], 1))


def snapshot(model):
    return copy.deepcopy(dict(classifier_state_dict=model.classifier.state_dict(),
        backbone_state_dict=model.backbone.state_dict(), neck_state_dict=model.neck.state_dict(),
        backbone_state_sha256=state_sha256(model.backbone), neck_state_sha256=state_sha256(model.neck),
        encoder_trainability=encoder_policy(model)))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(119)
    images = torch.randn(8, 3, 16, 16, dtype=torch.float64)
    labels = torch.tensor([2, 2, 0, 1, 0, 2, 1, 0])
    weights = torch.tensor([1., 1., 4.], dtype=torch.float64)
    options = dict(lr=.01, backbone_lr_mult=.01, neck_lr_mult=.1)
    for train_backbone, train_neck in ((False, False), (False, True), (True, False), (True, True)):
        torch.manual_seed(31)
        model = TunablePyramidClassifier(Backbone(), Neck(), Head(), train_backbone, train_neck).double().train()
        initial = {name:state_sha256(getattr(model, name)) for name in ('backbone', 'neck')}
        full = copy.deepcopy(model)
        F.cross_entropy(full(images), labels, weight=weights).backward()
        batches = [(images[i:i+2], labels[i:i+2], None) for i in range(0, 8, 2)]
        windows = list(update_batches(batches, 4))
        assert len(windows) == 1 and len(windows[0]) == 4
        total = float(weights[labels].sum())
        for x, y, _ in windows[0]:
            (F.cross_entropy(model(x), y, weight=weights) * float(weights[y].sum())/total).backward()
        for (name, parameter), (_, reference) in zip(model.named_parameters(), full.named_parameters()):
            if reference.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-7, rtol=1e-5)
        for name, trainable in encoder_policy(model).items():
            module = getattr(model, name)
            assert module.training == trainable
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()) == trainable
        groups = classification_parameter_groups(model, options)
        assert len(groups) == 1+train_backbone+train_neck
        assert sum(len(g['params']) for g in groups) == len({id(p) for p in model.parameters() if p.requires_grad})
        optimizer = torch.optim.SGD(groups, lr=.01, momentum=.9)
        optimizer.step()
        model.eval()
        hashes = audit_encoders(model, initial)
        for name, trainable in encoder_policy(model).items():
            assert (hashes[name] != initial[name]) == trainable, name
        chosen = snapshot(model)
        with torch.no_grad():
            expected = model(images).clone()
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(.02)
        restore_classification_snapshot(model, chosen)
        with torch.no_grad():
            torch.testing.assert_close(model(images), expected, rtol=0., atol=0.)
        print('PASS: backbone/neck=', train_backbone, train_neck,
              'gradient through frozen neck, weighted effective-batch parity, encoder restore')
    for index in range(14, 22):
        path = next((ROOT/'configs/classification').glob('c%d_*.py' % index))
        cfg = resolve(path)
        assert cfg['classification_options']['sym_mode'] == 'original'
        assert cfg['classification_options']['use_lbp'] and cfg['classification_options']['use_gabor']
        assert cfg['data']['batch_size'] * cfg['training']['accumulation_steps'] == 8
        assert cfg['selection_metric'] == 'accuracy' and cfg['diagnostic_interval'] == 2
        assert cfg['training']['epochs'] == 12
        assert bool(cfg['data'].get('clahe')) == (index % 2 == 1)
        expected_policy = ((False, False), (False, True), (True, False), (True, True))[(index-14)//2]
        assert tuple(cfg['encoder_training'][k] for k in ('train_backbone','train_neck')) == expected_policy
    print('PASS: C14-C21 original-Sym flags, matched batch/update budgets and 2x2x2 design')


if __name__ == '__main__':
    main()
