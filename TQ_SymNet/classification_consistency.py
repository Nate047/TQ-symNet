"""Small, label-preserving intensity perturbations for CXR classification."""
import torch
from torch.nn import functional as F


def brightness_contrast_view(images, mean, std, brightness_pixels=5., contrast=0.05,
                             generator=None):
    if brightness_pixels < 0 or contrast < 0 or contrast >= 1:
        raise ValueError('Invalid brightness/contrast range')
    if brightness_pixels == 0 and contrast == 0:
        return images
    mean = images.new_tensor(mean).view(1, 3, 1, 1)
    std = images.new_tensor(std).view(1, 3, 1, 1)
    batch = images.shape[0]
    scale = 1. + (2. * torch.rand(batch, 1, 1, 1, device=images.device,
                                 generator=generator) - 1.) * contrast
    shift = (2. * torch.rand(batch, 1, 1, 1, device=images.device,
                             generator=generator) - 1.) * brightness_pixels
    # Use a shared image-intensity centre, not ImageNet's different RGB means.
    # A grayscale radiograph must stay grayscale after denormalization.
    raw = images * std + mean
    centre = raw.mean(dim=(1, 2, 3), keepdim=True)
    changed = ((raw - centre) * scale + centre + shift).clamp(0., 255.)
    return (changed - mean) / std


def paired_view_logits(model, images, mean, std, brightness_pixels=5., contrast=0.05,
                       generator=None):
    """Reuse one stochastic-head mask and preserve the next CE pass's RNG.

    Only the image perturbation differs between the two train-time forwards.
    A separate, seed-derived generator controls the intensity perturbation.
    The second forward must not advance the original CUDA/CPU dropout stream.
    """
    if generator is None:
        raise ValueError('Consistency views require a dedicated seed-derived generator')
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(images.device) if images.is_cuda else None
    original_logits = model(images)
    devices = [images.device.index] if images.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        changed = brightness_contrast_view(images, mean, std,
                                           brightness_pixels, contrast, generator)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, images.device)
        changed_logits = model(changed)
    return original_logits, changed_logits


def consistency_kl(student_logits, teacher_logits):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError('Expected paired (batch, classes) logits')
    return F.kl_div(F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits.detach(), dim=1),
                    reduction='batchmean')
