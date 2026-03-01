"""
Low-level weight quantization and dequantization operations.

Includes group-wise absmax computation, fake quantize-dequantize for
calibration, integer quantization, and FP8 per-tensor quantization.
"""

import torch


def compute_groupwise_abmax(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    Compute the per-group absolute maximum of `weight`.

    The last dimension is partitioned into groups of `group_size`, and the
    absolute maximum within each group is returned.

    Args:
        weight: Tensor of shape (..., K) where K is divisible by group_size.
        group_size: Number of elements per quantization group.

    Returns:
        Tensor of shape (..., K // group_size).
    """
    grouped = weight.view(-1, weight.shape[-1] // group_size, group_size)
    scale, _ = grouped.abs().max(dim=-1, keepdim=True)
    return scale.squeeze(-1)


def fake_quant_dequant(
    x: torch.Tensor, bits: int = 4, group_size: int = 128
) -> torch.Tensor:
    """
    Simulate quantization followed by dequantization (straight-through estimator).

    This rounds each value to the nearest quantization level and reconstructs
    the approximate original value, introducing quantization noise.

    Args:
        x: Input tensor.
        bits: Quantization bit-width.
        group_size: Group size for scale computation.

    Returns:
        Tensor of the same shape as `x` with simulated quantization noise.
    """
    bnt = (1 << (bits - 1)) - 1
    quant_scale = compute_groupwise_abmax(x, group_size=group_size)
    quant_scale = torch.repeat_interleave(quant_scale, group_size, dim=-1)
    quant_scale = quant_scale.reshape(x.shape)

    # Expand dimensions if needed
    for _ in range(len(x.shape) - len(quant_scale.shape)):
        quant_scale = quant_scale.unsqueeze(-1)

    quant_value = torch.clamp(torch.round(x / quant_scale * bnt), -bnt - 1, bnt)
    return quant_value / bnt * quant_scale


def quantize_weight_int(
    x: torch.Tensor, scales: torch.Tensor, bits: int = 8
) -> tuple:
    """
    Quantize weights to signed integer representation using given scales.

    Args:
        x: Weight tensor (modified in-place).
        scales: Per-group scale factors.
        bits: Quantization bit-width.

    Returns:
        Tuple of (quantized_weight, adjusted_scales).
    """
    if scales.ndim == 2:
        scales = torch.repeat_interleave(scales, x.shape[1] // scales.shape[1], dim=-1)
    bnt = (1 << (bits - 1)) - 1

    while scales.ndim < x.ndim:
        scales = scales.unsqueeze(-1)
    scales.div_(bnt)
    x.div_(scales).round_().clamp_(-bnt - 1, bnt)
    return x, scales


def quantize_weight_per_tensor_fp8(
    tensor: torch.Tensor, scale: torch.Tensor
) -> tuple:
    """
    Quantize a weight tensor to FP8 (E4M3FN) format using a per-tensor scale.

    Args:
        tensor: Weight tensor in higher precision.
        scale: Per-tensor scale factor.

    Returns:
        Tuple of (fp8_weight, float32_scale).
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()
    return qweight, scale
