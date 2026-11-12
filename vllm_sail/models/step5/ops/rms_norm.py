# SPDX-License-Identifier: Apache-2.0
"""Step5 的 opt-in Triton RMS/AddRMS；保持非原地 residual/out 契约。"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def supports_fused_rms_norm(x, weight, out_dtype=None) -> bool:
    """仅查询元数据；拒绝未支持的设备、dtype、rank 和过大的归约。"""
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
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
    if any(step <= 0 for size, step in zip(x.shape, x.stride()) if size > 1):
        return False
    if weight.stride(0) <= 0 or math.prod(x.shape[:-1]) >= 2**31:
        return False
    # 手写核心没有 autograd；训练调用保留原 eager 图。
    return not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))


def supports_fused_add_rms_norm(x, residual, weight, out_dtype=None) -> bool:
    """不支持 broadcast/训练时保留原路径；仅检查元数据，不先执行加法。"""
    if not supports_fused_rms_norm(x, weight, out_dtype):
        return False
    if not isinstance(residual, torch.Tensor) or residual.layout != torch.strided:
        return False
    if residual.device != x.device or residual.dtype not in _DTYPES:
        return False
    if residual.shape != x.shape:
        return False
    if any(step <= 0 for size, step in zip(residual.shape, residual.stride()) if size > 1):
        return False
    return not (torch.is_grad_enabled() and residual.requires_grad)


@triton.jit
def _fused_step5_rms_kernel(
    X, W, Y, R, R_OUT, EPS: tl.constexpr, ZERO_CENTERED: tl.constexpr,
    SHAPE: tl.constexpr, STRIDES: tl.constexpr, W_STRIDE: tl.constexpr,
    H: tl.constexpr, BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr, R_STRIDES: tl.constexpr,
):
    row = tl.program_id(0)
    remaining = row.to(tl.int64)
    offset = tl.full((), 0, tl.int64)
    # 将 leading 维度逐个还原，仅最后一维参与 RMS 归约。
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
        residual = tl.load(R + offset_r + col * R_STRIDES[-1], col < H, 0).to(tl.float32)
        # 与 torch.add 一致：先舍入到提升后的 sum dtype，再做 FP32 RMS。
        # 混合 BF16/FP32 不降精度；两个低精度输入保留中间舍入。
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
    """显式候选不隐藏 eager 回退；空 batch 不 launch，输出保持完整 shape。"""
    if not supports_fused_rms_norm(x, weight, out_dtype):
        raise ValueError("融合 RMS 核心不支持该 input/weight/output 元数据")
    if not math.isfinite(variance_epsilon) or variance_epsilon <= 0:
        raise ValueError("epsilon 必须有限且大于零")
    output = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    rows = math.prod(x.shape[:-1])
    if rows:
        hidden = x.shape[-1]
        block = triton.next_power_of_2(hidden)
        _fused_step5_rms_kernel[(rows,)](
            x, weight, output, x, output, variance_epsilon, bool(zero_centered),
            tuple(x.shape), tuple(x.stride()), weight.stride(0), hidden, block,
            False, (), num_warps=4, enable_fp_fusion=False,
        )
    return output


def fused_add_rms_norm(x, residual, weight, variance_epsilon,
                       zero_centered=False, out_dtype=None):
    """单次 launch 返回 (Norm, 新 residual)，不改写输入或隐藏执行错误。"""
    if not supports_fused_add_rms_norm(x, residual, weight, out_dtype):
        raise ValueError("融合 AddRMS 不支持该 input/residual/weight/output 元数据")
    if not math.isfinite(variance_epsilon) or variance_epsilon <= 0:
        raise ValueError("epsilon 必须有限且大于零")
    output = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    residual_out = torch.empty(x.shape, dtype=torch.promote_types(x.dtype, residual.dtype),
                               device=x.device)
    rows = math.prod(x.shape[:-1])
    if rows:
        hidden = x.shape[-1]
        block = triton.next_power_of_2(hidden)
        _fused_step5_rms_kernel[(rows,)](
            x, weight, output, residual, residual_out, variance_epsilon, bool(zero_centered),
            tuple(x.shape), tuple(x.stride()), weight.stride(0), hidden, block,
            True, tuple(residual.stride()), num_warps=4, enable_fp_fusion=False,
        )
    return output, residual_out
