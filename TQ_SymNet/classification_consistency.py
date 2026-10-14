"""Small, label-preserving intensity perturbations for CXR classification."""
import torch
from torch.nn import functional as F


def brightness_contrast_view(images, mean, std, brightness_pixels=5., contrast=0.05):
    if brightness_pixels < 0 or contrast < 0 or contrast >= 1:
        raise ValueError('Invalid brightness/contrast range')
    mean = images.new_tensor(mean).view(1, 3, 1, 1)
    std = images.new_tensor(std).view(1, 3, 1, 1)
    batch = images.shape[0]
    scale = 1. + (2. * torch.rand(batch, 1, 1, 1, device=images.device) - 1.) * contrast
    shift = (2. * torch.rand(batch, 1, 1, 1, device=images.device) - 1.) * brightness_pixels
    # Inputs are (RGB - mean) / std. Scale about the channel mean, then add
    # a shared grayscale brightness shift in raw 0..255 pixel units.
    changed = images * scale + shift / std
    return torch.max(torch.min(changed, (255. - mean) / std), -mean / std)


def consistency_kl(student_logits, teacher_logits):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError('Expected paired (batch, classes) logits')
    return F.kl_div(F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits.detach(), dim=1),
                    reduction='batchmean')
