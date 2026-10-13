"""Opt-in deterministic equivalents for legacy PyTorch CUDA operations."""

import torch
import torch.nn.functional as F


def _linear_resize_matrix(in_size, out_size, device, dtype,
                          align_corners=False):
    output = torch.arange(out_size, device=device, dtype=torch.float64)
    if align_corners:
        source = (output * (in_size - 1) / (out_size - 1)
                  if out_size > 1 else output.new_zeros(1))
    else:
        source = (output + 0.5) * in_size / out_size - 0.5
    source = source.clamp(0, in_size - 1)
    lower = source.floor().long()
    upper = (lower + 1).clamp(max=in_size - 1)
    upper_weight = source - lower.to(source.dtype)
    lower_weight = 1.0 - upper_weight
    columns = torch.arange(in_size, device=device)[None, :]
    matrix = ((columns == lower[:, None]) * lower_weight[:, None]
              + (columns == upper[:, None]) * upper_weight[:, None])
    return matrix.to(dtype=dtype)


class _DeterministicBilinear2d(torch.autograd.Function):
    """Native bilinear forward with a deterministic matrix-form backward."""

    @staticmethod
    def forward(ctx, input_tensor, out_h, out_w, align_corners):
        ctx.input_shape = tuple(input_tensor.shape)
        ctx.output_size = (int(out_h), int(out_w))
        ctx.align_corners = bool(align_corners)
        return F.interpolate(
            input_tensor, size=ctx.output_size, mode='bilinear',
            align_corners=ctx.align_corners)

    @staticmethod
    def backward(ctx, grad_output):
        in_h, in_w = ctx.input_shape[-2:]
        out_h, out_w = ctx.output_size
        height_matrix = _linear_resize_matrix(
            in_h, out_h, grad_output.device, grad_output.dtype,
            ctx.align_corners)
        width_matrix = _linear_resize_matrix(
            in_w, out_w, grad_output.device, grad_output.dtype,
            ctx.align_corners)
        flat_grad = grad_output.reshape(-1, out_h, out_w)
        grad_input = torch.matmul(height_matrix.transpose(0, 1), flat_grad)
        grad_input = torch.matmul(grad_input, width_matrix)
        return grad_input.reshape(ctx.input_shape), None, None, None


def deterministic_bilinear_interpolate(input_tensor, size,
                                       align_corners=False):
    """Use an exact deterministic backward only under strict CUDA mode."""
    if isinstance(size, int):
        out_h = out_w = size
    else:
        out_h, out_w = int(size[0]), int(size[1])
    if (torch.are_deterministic_algorithms_enabled()
            and input_tensor.is_cuda and input_tensor.requires_grad):
        return _DeterministicBilinear2d.apply(
            input_tensor, out_h, out_w, align_corners)
    return F.interpolate(
        input_tensor, size=(out_h, out_w), mode='bilinear',
        align_corners=align_corners)
