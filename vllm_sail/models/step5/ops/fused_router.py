# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 BF16 gate GEMM and expert selection."""

import torch
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    _get_padding_mask,
    fused_topk_bias,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# Configurations from the v0.23 M890 Step5 router sweep.
_CONFIGS = (
    (16, (16, 32, 64, 7, 1, 4, 4)),
    (48, (32, 32, 64, 7, 1, 4, 4)),
    (64, (64, 32, 64, 7, 1, 4, 4)),
    (128, (128, 64, 64, 6, 1, 8, 4)),
    (256, (64, 128, 32, 6, 8, 8, 4)),
    (320, (64, 128, 32, 5, 1, 8, 4)),
    (384, (64, 128, 32, 6, 1, 8, 3)),
    (512, (64, 128, 32, 3, 8, 8, 6)),
    (640, (128, 128, 32, 5, 2, 8, 3)),
    (1024, (128, 128, 32, 3, 8, 8, 4)),
    (1280, (64, 128, 64, 1, 1, 8, 4)),
    (1536, (64, 128, 64, 1, 16, 8, 4)),
    (1664, (64, 128, 64, 1, 16, 8, 4)),
    (1920, (64, 128, 32, 1, 16, 8, 4)),
    (2048, (64, 128, 32, 1, 2, 8, 6)),
    (2304, (64, 128, 64, 1, 2, 8, 4)),
    (2432, (64, 128, 32, 1, 1, 8, 4)),
    (4608, (128, 128, 32, 1, 2, 8, 5)),
    (7168, (64, 128, 64, 1, 1, 8, 4)),
    (8192, (64, 128, 32, 1, 1, 8, 5)),
    (9216, (128, 128, 32, 1, 2, 8, 5)),
    (12288, (64, 128, 64, 1, 1, 8, 4)),
)


@triton.jit
def _router_gemm_kernel(
    a_ptr,  # [M, K] bf16, contiguous
    w_ptr,  # [N, K] bf16, contiguous -- the weight as the checkpoint stores it
    out_ptr,  # [M, N] fp32 or bf16, contiguous
    part_ptr,  # [SPLIT_K, M, N] fp32, contiguous; unused when SPLIT_K == 1
    M,
    N,
    K,
    NB,  # K // BLOCK_K -- the number of BLOCK_K steps to cover all of K
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M > 1:
        num_in_group = GROUP_M * num_pid_n
        group_id = pid // num_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_in_group) % group_size_m)
        pid_n = (pid % num_in_group) // group_size_m
    else:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

    # Uneven split-K: CTA ``pid_s`` owns ``nb_s`` consecutive BLOCK_K steps
    # starting at step ``st_s``.  SPLIT_K <= NB is guaranteed by _legal_cfg.
    base = NB // SPLIT_K
    rem = NB % SPLIT_K
    nb_s = base + (pid_s < rem).to(tl.int32)
    st_s = pid_s * base + tl.minimum(pid_s, rem)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = st_s * BLOCK_K + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
    m_mask = offs_m < M
    n_mask = offs_n < N
    store_mask = m_mask[:, None] & n_mask[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(nb_s):
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        if EVEN_N:
            w = tl.load(w_ptrs)
        else:
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        acc = tl.dot(a, tl.trans(w), acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    out_offs = offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        if OUT_BF16:
            if EVEN_M and EVEN_N:
                tl.store(out_ptr + out_offs, acc.to(tl.bfloat16))
            else:
                tl.store(out_ptr + out_offs, acc.to(tl.bfloat16), mask=store_mask)
        else:
            if EVEN_M and EVEN_N:
                tl.store(out_ptr + out_offs, acc)
            else:
                tl.store(out_ptr + out_offs, acc, mask=store_mask)
    else:
        part_offs = pid_s * (M * stride_om) + out_offs
        if EVEN_M and EVEN_N:
            tl.store(part_ptr + part_offs, acc)
        else:
            tl.store(part_ptr + part_offs, acc, mask=store_mask)


@triton.jit
def _reduce_router_kernel(
    partials,
    bias_ptr,
    weights_ptr,
    indices_ptr,
    padding_ptr,
    stride_row,
    stride_split,
    SPLIT_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    SCALING: tl.constexpr,
    IDX64: tl.constexpr,
    PARALLEL_TOPK: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    source_rows_ptr=None,
):
    row = tl.program_id(0)
    experts = tl.arange(0, 512)
    mask = experts < 352
    ptr = partials + row * stride_row + experts
    logits = tl.load(ptr, mask=mask, other=0.0)
    # Match the original GEMM reducer's accumulation order exactly.
    for split in tl.static_range(1, SPLIT_K):
        logits += tl.load(ptr + split * stride_split, mask=mask, other=0.0)
    # Match the independent sigmoid TopK path for individual NaN logits.
    probs = tl.where(logits == logits, tl.sigmoid(logits), 0.0)
    bias = tl.load(bias_ptr + experts, mask=mask, other=0.0).to(tl.float32)
    scores = tl.where(mask, probs + bias, -float("inf"))
    ks = tl.arange(0, 8)
    if PARALLEL_TOPK:
        # An ordered FP32 bit pattern plus the inverted expert ID gives
        # an exact comparison key, including lower-ID-first score ties.
        scores = tl.where(scores == 0, 0.0, scores)
        bits = scores.to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(
            tl.uint32
        )
        keys = (ordered.to(tl.uint64) << 32) | (511 - experts.to(tl.uint32)).to(
            tl.uint64
        )
        # Padding never competes with a real expert, even for -inf scores.
        keys = tl.where(mask, keys, 0)
        selected = tl.topk(keys, 8)
        indices = (511 - selected.to(tl.uint32)).to(tl.int32)
        ordered = (selected >> 32).to(tl.uint32)
        bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered).to(
            tl.uint32
        )
        selected_scores = bits.to(tl.float32, bitcast=True)
        weights = selected_scores - tl.load(bias_ptr + indices).to(tl.float32)
        total = tl.sum(weights, axis=0)
    else:
        weights = tl.zeros((8,), tl.float32)
        indices = tl.zeros((8,), tl.int32)
        total = 0.0
        for k in tl.static_range(8):
            score, index = tl.max(scores, axis=0, return_indices=True)
            if source_rows_ptr is None:
                weight = score - tl.load(bias_ptr + index).to(tl.float32)
                weights = tl.where(ks == k, weight, weights)
                total += weight
            indices = tl.where(ks == k, index, indices)
            scores = tl.where(experts == index, -float("inf"), scores)
    if source_rows_ptr is not None:
        weights = tl.gather(probs, indices, 0)
        total = tl.sum(weights, axis=0)
        tl.store(source_rows_ptr + row * 8 + ks, ks * tl.num_programs(0) + row)
    if RENORMALIZE:
        if PARALLEL_TOPK:
            weights *= tl.div_rn(SCALING, total + 1e-20)
        else:
            weights = weights / (total + 1e-20)
            weights *= SCALING
    else:
        weights *= SCALING
    if HAS_PADDING:
        is_pad = tl.load(padding_ptr + row)
        indices = tl.where(is_pad, -1, indices)
        weights = tl.where(is_pad, 0.0, weights)
    if IDX64:
        indices = indices.to(tl.int64)
    tl.store(weights_ptr + row * 8 + ks, weights)
    tl.store(indices_ptr + row * 8 + ks, indices)


def fused_router(
    x: torch.Tensor,
    weight: torch.Tensor,
    bf16_weight: torch.Tensor,
    bias: torch.Tensor,
    renormalize: bool,
    scaling: float,
    padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.shape[0]
    if rows == 0:
        return _fake(x, weight, bf16_weight, bias, renormalize, scaling, padding)
    bm, bn, bk, splits, gm, warps, stages = next(
        (cfg for limit, cfg in _CONFIGS if rows <= limit), _CONFIGS[-1][1]
    )
    x = x.contiguous()
    result = x.new_empty((splits, rows, 352), dtype=torch.float32)
    _router_gemm_kernel[(triton.cdiv(rows, bm) * triton.cdiv(352, bn), splits)](
        x,
        bf16_weight,
        result,
        result,
        rows,
        352,
        4096,
        4096 // bk,
        352,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        SPLIT_K=splits,
        GROUP_M=gm,
        EVEN_M=rows % bm == 0,
        EVEN_N=352 % bn == 0,
        OUT_BF16=False,
        num_warps=warps,
        num_stages=stages,
    )
    if splits == 1 and padding is None:
        # The GEMM has already produced logits. Keep SAIL's native biased TopK.
        weights, indices = fused_topk_bias(
            x,
            result[0],
            "sigmoid",
            bias,
            8,
            renormalize,
            torch.int64,
            routed_scaling_factor=scaling,
        )
        # The upstream TopK kernel sets padding ids to -1 but keeps their
        # weights; zero them so MegaMoE can skip re-masking padding rows.
        pad = _get_padding_mask(rows)
        if pad is not None:
            weights.masked_fill_(pad[:, None], 0.0)
        return weights, indices
    weights, indices = _fake(
        x, weight, bf16_weight, bias, renormalize, scaling, padding
    )
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    topk_warps = 8 if rows <= max(32, sms) else 4 if rows <= 4 * sms else 1
    # An explicit mask is the rank-local sequence-parallel mask; the fused TopK
    # below honors it directly. Without one, read the global forward-context
    # prefix, which is only correct for the unsharded (non-SP) token set.
    pad = padding if padding is not None else _get_padding_mask(rows)
    if pad is not None:
        pad = pad.contiguous()
    _reduce_router_kernel[(rows,)](
        result,
        bias,
        weights,
        indices,
        pad,
        352,
        rows * 352,
        SPLIT_K=splits,
        RENORMALIZE=renormalize,
        SCALING=scaling,
        IDX64=True,
        PARALLEL_TOPK=rows <= 32,
        HAS_PADDING=pad is not None,
        num_warps=topk_warps,
        num_stages=2,
    )
    return weights, indices


def _fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bf16_weight: torch.Tensor,
    bias: torch.Tensor,
    renormalize: bool,
    scaling: float,
    padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x.new_empty((x.shape[0], 8), dtype=torch.float32),
        x.new_empty((x.shape[0], 8), dtype=torch.int64),
    )


def register_ops() -> None:
    direct_register_custom_op(
        op_name="ppu_step5_fused_router",
        op_func=fused_router,
        mutates_args=[],
        fake_impl=_fake,
    )
