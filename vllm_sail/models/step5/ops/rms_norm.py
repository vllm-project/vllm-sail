# SPDX-License-Identifier: Apache-2.0
"""Opt-in Triton RMS/AddRMS for Step5; keeps the out-of-place residual/out contract."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def supports_fused_rms_norm(x, weight, out_dtype=None) -> bool:
    """Metadata only; reject unsupported devices, dtypes, ranks and huge reductions."""
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if x.is_neg() or weight.is_neg():
        return False
    if x.layout != torch.strided or weight.layout != torch.strided:
        return False
    if not x.is_cuda or weight.device != x.device:
        return False
    if x.dtype not in _DTYPES or weight.dtype not in _DTYPES:
        return False
    if (out_dtype or x.dtype) not in _DTYPES:
        return False
    if not 1 <= x.ndim <= 8 or not 1 <= x.shape[-1] <= 8192:
        return False
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        return False
    if any(
        step <= 0 for size, step in zip(x.shape, x.stride(), strict=True) if size > 1
    ):
        return False
    if weight.stride(0) <= 0 or math.prod(x.shape[:-1]) >= 2**31:
        return False
    # The hand-written kernel has no autograd; training calls keep the eager graph.
    return not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))


def supports_fused_add_rms_norm(x, residual, weight, out_dtype=None) -> bool:
    """Keep the original path for broadcast/training; metadata only, no add first."""
    if not supports_fused_rms_norm(x, weight, out_dtype):
        return False
    if not isinstance(residual, torch.Tensor) or residual.layout != torch.strided:
        return False
    if residual.is_neg():
        return False
    if residual.device != x.device or residual.dtype not in _DTYPES:
        return False
    if residual.shape != x.shape:
        return False
    if any(
        step <= 0
        for size, step in zip(residual.shape, residual.stride(), strict=True)
        if size > 1
    ):
        return False
    return not (torch.is_grad_enabled() and residual.requires_grad)


@triton.jit
def _fused_step5_rms_kernel(
    X,
    W,
    Y,
    R,
    R_OUT,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    W_STRIDE: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    R_STRIDES: tl.constexpr,
):
    row = tl.program_id(0)
    remaining = row.to(tl.int64)
    offset = tl.full((), 0, tl.int64)
    # Restore the leading dims one by one; only the last dim is RMS-reduced.
    for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
        offset += (remaining % SHAPE[axis]) * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    col = tl.arange(0, BLOCK)
    value = tl.load(X + offset + col * STRIDES[-1], col < H, 0).to(tl.float32)
    if HAS_RESIDUAL:
        remaining_r = row.to(tl.int64)
        offset_r = tl.full((), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
            offset_r += (remaining_r % SHAPE[axis]) * R_STRIDES[axis]
            remaining_r = remaining_r // SHAPE[axis]
        residual = tl.load(R + offset_r + col * R_STRIDES[-1], col < H, 0).to(
            tl.float32
        )
        # Like torch.add: round to the promoted sum dtype first, then do the FP32 RMS.
        # Mixed BF16/FP32 loses no precision; two low-precision inputs keep the
        # intermediate rounding.
        summed = (value + residual).to(R_OUT.dtype.element_ty)
        tl.store(R_OUT + row.to(tl.int64) * H + col, summed, col < H)
        value = summed.to(tl.float32)
    variance = tl.sum(value * value, axis=0) / H
    normalized = value * tl.rsqrt(variance + EPS)
    scale = tl.load(W + col * W_STRIDE, col < H, 0).to(tl.float32)
    if ZERO_CENTERED:
        scale = scale + 1.0
    result = normalized * scale
    tl.store(Y + row.to(tl.int64) * H + col, result, col < H)


def fused_rms_norm(x, weight, variance_epsilon, zero_centered=False, out_dtype=None):
    """Explicit candidate without a hidden eager fallback.

    An empty batch does not launch and the output keeps its full shape.
    """
    if not supports_fused_rms_norm(x, weight, out_dtype):
        raise ValueError(
            "fused RMS kernel does not support this input/weight/output metadata"
        )
    if not math.isfinite(variance_epsilon) or variance_epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    output = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    rows = math.prod(x.shape[:-1])
    if rows:
        hidden = x.shape[-1]
        block = triton.next_power_of_2(hidden)
        _fused_step5_rms_kernel[(rows,)](
            x,
            weight,
            output,
            x,
            output,
            variance_epsilon,
            bool(zero_centered),
            tuple(x.shape),
            tuple(x.stride()),
            weight.stride(0),
            hidden,
            block,
            False,
            (),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def fused_add_rms_norm(
    x, residual, weight, variance_epsilon, zero_centered=False, out_dtype=None
):
    """One launch returns (norm, new residual); inputs and errors are left alone."""
    if not supports_fused_add_rms_norm(x, residual, weight, out_dtype):
        raise ValueError(
            "fused AddRMS does not support this input/residual/weight/output metadata"
        )
    if not math.isfinite(variance_epsilon) or variance_epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    output = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    residual_out = torch.empty(
        x.shape, dtype=torch.promote_types(x.dtype, residual.dtype), device=x.device
    )
    rows = math.prod(x.shape[:-1])
    if rows:
        hidden = x.shape[-1]
        block = triton.next_power_of_2(hidden)
        _fused_step5_rms_kernel[(rows,)](
            x,
            weight,
            output,
            residual,
            residual_out,
            variance_epsilon,
            bool(zero_centered),
            tuple(x.shape),
            tuple(x.stride()),
            weight.stride(0),
            hidden,
            block,
            True,
            tuple(residual.stride()),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, residual_out


def supports_three_input_rms_norm(x, shared, residual, weight, out_dtype=None):
    return (
        supports_fused_add_rms_norm(x, residual, weight, out_dtype)
        and isinstance(shared, torch.Tensor)
        and not shared.is_neg()
        and x.ndim == 2
        and x.shape[-1] == 4096
        and x.dtype == shared.dtype == torch.bfloat16
        and residual.dtype in (torch.bfloat16, torch.float32)
        and weight.dtype == torch.float32
        and shared.shape == x.shape
        and shared.device == x.device
        and all(t.is_contiguous() for t in (x, shared, residual, weight))
        and not (torch.is_grad_enabled() and shared.requires_grad)
    )


@triton.jit
def _three_input_rms_kernel(
    X,
    S,
    R,
    W,
    Y,
    R_OUT,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    RES_FP32: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    offset = row * H + col
    x = tl.load(X + offset, col < H, other=0).to(tl.float32)
    shared = tl.load(S + offset, col < H, other=0).to(tl.float32)
    residual = tl.load(R + offset, col < H, other=0).to(tl.float32)
    if RES_FP32:
        # Match add_fp32(moe, shared) then the FP32 AddRMS sum: no intermediate
        # rounding, same conversion and add order as the unfused path.
        combined = (x + shared) + residual
        tl.store(R_OUT + offset, combined, col < H)
        value = combined
    else:
        # Match the two BF16 additions before the FP32 RMS reduction.
        summed = (x + shared).to(tl.bfloat16).to(tl.float32)
        combined = (summed + residual).to(tl.bfloat16)
        tl.store(R_OUT + offset, combined, col < H)
        value = combined.to(tl.float32)
    variance = tl.sum(value * value, axis=0) / H
    normalized = value * tl.rsqrt(variance + EPS)
    scale = tl.load(W + col, col < H, other=0).to(tl.float32)
    if ZERO_CENTERED:
        scale = scale + 1.0
    tl.store(Y + offset, normalized * scale, col < H)


def three_input_rms_norm(
    x, shared, residual, weight, variance_epsilon, zero_centered=False, out_dtype=None
):
    output = torch.empty_like(x, dtype=out_dtype or x.dtype)
    residual_out = torch.empty_like(residual)
    if x.shape[0]:
        _three_input_rms_kernel[(x.shape[0],)](
            x,
            shared,
            residual,
            weight,
            output,
            residual_out,
            variance_epsilon,
            zero_centered,
            x.shape[1],
            triton.next_power_of_2(x.shape[1]),
            residual.dtype == torch.float32,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, residual_out
