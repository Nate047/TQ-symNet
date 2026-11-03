"""Real optimization checks for smoke tests only; never used in training."""
import hashlib

import torch
from torch.nn import functional as F


def state_digest(model, include_quality=False):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if include_quality or not name.startswith('candidate_quality.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checked_optimizer_update(model, optimizer, loss):
    """Reject a vacuous backward, then verify an actual parameter update."""
    if not torch.is_tensor(loss) or not torch.isfinite(loss).all() or loss.numel() != 1:
        raise RuntimeError('Smoke loss must be a finite scalar')
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError('Nonfinite smoke gradient')
    nonzero = sum(bool((g != 0).any()) for g in gradients)
    if not nonzero:
        raise RuntimeError('Smoke backward has no nonzero gradients')
    optimizer.step()
    changed = sum(not torch.equal(before[n], p.detach())
                  for n, p in model.named_parameters() if n in before)
    if not changed:
        raise RuntimeError('Smoke optimizer did not change any trainable parameter')
    return dict(nonzero_gradient_parameters=nonzero, changed_parameters=changed)


def quality_update_smoke(core, device):
    """Train the zero-initialized ROI head on nonzero targets for two steps.

    Second step checks gradient reaches the feature projection, not only the
    initially zero output layer. All updates occur on the temporary smoke model.
    """
    q = core.candidate_quality.to(device).train()
    if any(p.requires_grad for n, p in core.named_parameters()
           if not n.startswith('candidate_quality.')):
        raise RuntimeError('Smoke detector must be frozen')
    original = state_digest(core)
    maps = [torch.rand(1,256,s,s,device=device) for s in (16,8,4,2)]
    boxes = torch.tensor([[.1,.1,.8,.9],[.2,.3,.6,.7]],device=device)
    target = boxes.new_tensor([.85,.20])
    optimizer = torch.optim.AdamW([p for p in q.parameters() if p.requires_grad],
                                   lr=1e-4, weight_decay=0.)
    first_loss = None
    changes = []
    for _ in range(2):
        memory = q.prepare(maps,None,(64,64),(64,64))
        pred = q(boxes,memory,boxes.new_zeros((2,3)))[0]
        if pred.shape != (2,2):
            raise RuntimeError('Invalid ROI smoke output shape')
        loss = F.binary_cross_entropy_with_logits(pred[:,0],target)
        if first_loss is None:
            first_loss = float(loss.detach())
        changes.append(checked_optimizer_update(q,optimizer,loss))
    projection_grad = q.visual_projection[0].weight.grad
    if projection_grad is None or not bool((projection_grad != 0).any()):
        raise RuntimeError('No supervised gradient reached the ROI visual projection')
    if state_digest(core) != original:
        raise RuntimeError('Smoke update changed frozen detector state')
    empty = q(boxes[:0],q.prepare(maps,None,(64,64),(64,64)),boxes.new_zeros((0,3)))[0]
    if empty.shape != (0,2):
        raise RuntimeError('Empty ROI output shape changed')
    return dict(first_loss=first_loss, detector_unchanged=True,
                nonzero_gradient_parameters=changes[-1]['nonzero_gradient_parameters'],
                changed_quality_parameters=sum(x['changed_parameters'] for x in changes))
