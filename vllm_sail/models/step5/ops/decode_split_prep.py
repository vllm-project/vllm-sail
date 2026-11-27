# SPDX-License-Identifier: Apache-2.0
"""Split Step5 decode preparation into the two PPU Triton kernels.

At decode verification sizes (M <= 128) the separate PPU kernels replace the
upstream ``_joint_prepare``: the main QK norm/RoPE/KV scatter uses the
row-parallel ``ppu_step5_qknorm_rope_cache`` op and the indexer uses the
slot-first single-warp kernel, which also performs the CSA ring/summary update
(``UPDATE_CSA``). Larger shapes, FP8 KV or FP8 query preparation keep the joint
path.

``maybe_split`` is consulted by the indexer-fp16 launcher patch (the single
patch site on ``qkv_indexer_norm_rope``); it returns ``None`` when the joint
path must run.
"""

import torch
from vllm.models.step5.nvidia.ops import indexer_norm_rope as _indexer_mod
from vllm.models.step5.nvidia.ops.summary_update import summary_layout
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.models.step5.ops import soft_fp8


def _split_supported(
    query,
    key,
    weights,
    q_weight,
    k_weight,
    k_bias,
    cos_sin_cache,
    positions,
    zero_centered,
    main_inputs,
    decode_cache,
    decode_ring,
    decode_z,
    decode_slots,
    q_dtype,
):
    from vllm_sail.patch.performance.step5_qknorm_rope_cache import (
        _supported as _main_supported,
    )

    (
        main_qkv,
        main_qw,
        main_kw,
        main_rope,
        main_slots,
        main_kc,
        main_vc,
        main_qh,
        main_kh,
        main_rotary,
        _main_eps,
        _main_bias,
        _k_scale,
        _v_scale,
    ) = main_inputs
    if not (
        envs.VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV
        and current_platform.is_ppu()
        and current_platform.has_device_capability(89)
        and hasattr(torch.ops.vllm, "ppu_step5_qknorm_rope_cache")
    ):
        return False
    tokens = query.shape[0]
    if (
        not 0 < tokens <= 128
        or q_dtype not in (None, main_qkv.dtype)
        or main_kh != 1
        or query.ndim != 2
        or query.shape[1] % 256
        or query.shape[1] // 256 not in (4, 8)
        or cos_sin_cache.shape[1] not in (32, 64)
        or float(zero_centered) != 1.0
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or weights.dtype != torch.float32
        or main_kc.dtype in (torch.uint8, torch.float8_e4m3fn)
        or query.stride(1) != 1
        or key.stride(1) != 1
        or weights.stride(1) != 1
        or cos_sin_cache.stride(1) != 1
        or positions.dtype != torch.int64
    ):
        return False
    for t in (q_weight, k_weight, k_bias):
        if t.shape != (256,) or not t.is_contiguous() or t.device != query.device:
            return False
    if not _main_supported(
        main_qkv,
        main_qw,
        main_kw,
        main_rope,
        positions,
        main_slots,
        main_kc,
        main_vc,
        main_qh,
        main_kh,
        192,
        main_rotary,
        "auto",
    ):
        return False
    if decode_cache is not None:
        if (
            decode_ring is None
            or decode_z is None
            or decode_slots is None
            or decode_slots.numel() > tokens
            or decode_slots.dtype != torch.int64
            or decode_z.stride(1) != 1
            or not decode_ring.rows.is_contiguous()
        ):
            return False
    elif decode_z is not None or decode_slots is not None or decode_ring is not None:
        return False
    return True


def _split_prepare(
    query,
    key,
    weights,
    q_weight,
    k_weight,
    k_bias,
    cos_sin_cache,
    positions,
    eps,
    zero_centered,
    native_rope,
    main_inputs,
    decode_cache,
    decode_ring,
    decode_z,
    decode_slots,
):
    (
        main_qkv,
        main_qw,
        main_kw,
        main_rope,
        main_slots,
        main_kc,
        main_vc,
        main_qh,
        main_kh,
        main_rotary,
        main_eps,
        main_bias,
        _k_scale,
        _v_scale,
    ) = main_inputs
    tokens = query.shape[0]
    heads = query.shape[1] // 256
    padded_heads = max(heads, 8)
    q = torch.empty(
        (1, tokens, padded_heads, 256),
        device=query.device,
        dtype=torch.float16 if soft_fp8.indexer_fp16() else torch.float8_e4m3fn,
    )
    k = torch.empty((tokens, 1, 256), device=key.device, dtype=key.dtype)
    w = torch.empty(
        (1, tokens, padded_heads), device=weights.device, dtype=torch.float32
    )
    main_q, main_k = torch.ops.vllm.ppu_step5_qknorm_rope_cache(
        main_qkv,
        main_qw,
        main_kw,
        main_rope,
        positions,
        main_slots,
        main_kc,
        main_vc,
        main_qh,
        main_kh,
        main_rotary,
        main_eps,
        main_bias,
    )
    main_v = main_qkv[:, (main_qh + main_kh) * 192 :]
    _indexer_mod._indexer_norm_rope[(tokens, padded_heads + 1)](
        query,
        key,
        weights,
        q_weight,
        k_weight,
        k_bias,
        cos_sin_cache,
        positions,
        q,
        k,
        w,
        decode_z,
        decode_slots,
        decode_cache.mean if decode_cache is not None else None,
        decode_ring.rows if decode_ring is not None else None,
        decode_ring.metadata().base if decode_ring is not None else None,
        decode_cache is not None,
        decode_slots.numel() if decode_slots is not None else 0,
        decode_z.stride(0) if decode_z is not None else 0,
        decode_slots.stride(0) if decode_slots is not None else 0,
        decode_ring.capacity if decode_ring is not None else 8,
        *(summary_layout(decode_cache.mean) if decode_cache is not None else (0, 8)),
        query.stride(0),
        key.stride(0),
        weights.stride(0),
        heads,
        padded_heads,
        1,
        tokens,
        cos_sin_cache.shape[1],
        cos_sin_cache.stride(0),
        eps,
        float(zero_centered),
        native_rope,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q.transpose(0, 1), k, w.transpose(0, 1), main_q, main_k, main_v


def maybe_split(
    query,
    key,
    weights,
    q_weight,
    k_weight,
    k_bias,
    cos_sin_cache,
    positions,
    eps,
    zero_centered,
    native_rope=False,
    *,
    main_inputs,
    decode_cache=None,
    decode_ring=None,
    decode_z=None,
    decode_slots=None,
    q_dtype=None,
    q_scale=1.0,
):
    if _split_supported(
        query,
        key,
        weights,
        q_weight,
        k_weight,
        k_bias,
        cos_sin_cache,
        positions,
        zero_centered,
        main_inputs,
        decode_cache,
        decode_ring,
        decode_z,
        decode_slots,
        q_dtype,
    ):
        return _split_prepare(
            query,
            key,
            weights,
            q_weight,
            k_weight,
            k_bias,
            cos_sin_cache,
            positions,
            eps,
            zero_centered,
            native_rope,
            main_inputs,
            decode_cache,
            decode_ring,
            decode_z,
            decode_slots,
        )
    return None
