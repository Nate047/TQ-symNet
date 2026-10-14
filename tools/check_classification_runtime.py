"""On-server one-image build and backward check for a classification config."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from TQ_SymNet.classification import build_frozen_classifier, state_sha256
from TQ_SymNet.classification_consistency import paired_view_logits, consistency_kl


def main():
    from mmcv import Config

    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    source = args.checkpoint or cfg.backbone_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError('Run this full detector build check on the CUDA server')
    model, audit = build_frozen_classifier(cfg, source)
    model.cuda().train()
    before_backbone = state_sha256(model.backbone)
    before_neck = (state_sha256(model.neck)
                   if getattr(model, 'neck', None) is not None else None)
    x = torch.zeros((1, 3, cfg.data.image_size, cfg.data.image_size), device='cuda')
    consistency = cfg.get('consistency', None)
    if consistency is not None and consistency.weight > 0:
        generator = torch.Generator(device='cuda').manual_seed(1000003)
        logits, changed_logits = paired_view_logits(
            model, x, cfg.data.mean, cfg.data.std,
            consistency.brightness_pixels, consistency.contrast, generator)
        extra_loss = consistency.weight * consistency_kl(changed_logits, logits)
    else:
        logits = model(x)
        extra_loss = logits.new_zeros(())
    if logits.shape != (1, 3) or not torch.isfinite(logits).all():
        raise RuntimeError('Classifier must return one finite three-class logit vector')
    (F.cross_entropy(logits, torch.tensor([2], device='cuda')) + extra_loss).backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.classifier.parameters()):
        raise RuntimeError('Classification head received no finite gradient')
    for name in ('backbone', 'neck'):
        module = getattr(model, name, None)
        if module is not None and (module.training or any(
                p.requires_grad or p.grad is not None for p in module.parameters())):
            raise RuntimeError('Frozen %s received a training update' % name)
    if state_sha256(model.backbone) != before_backbone or (
            before_neck is not None and state_sha256(model.neck) != before_neck):
        raise RuntimeError('Frozen detector state changed')
    print('PASS: head=%s source=%s logits=%s CUDA_MiB=%d audit=%s' % (
        cfg.get('classification_head', 'sym'), source,
        logits.detach().cpu().tolist(), torch.cuda.max_memory_allocated() // 1048576, audit))


if __name__ == '__main__':
    main()
