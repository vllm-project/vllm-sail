# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _mask_routes(
    IDS,
    WEIGHTS,
    PADDING,
    OUT_IDS,
    OUT_WEIGHTS,
    N,
    TOP_K: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    padding = tl.load(PADDING + offsets // TOP_K * MASK_STRIDE, valid, other=False)
    ids = tl.load(IDS + offsets, valid, other=0)
    # Preserve NaN payloads and signed zero on unmasked routes.
    weight_bits = tl.load(WEIGHTS + offsets, valid, other=0).to(tl.uint32, bitcast=True)
    tl.store(OUT_IDS + offsets, tl.where(padding, -1, ids), valid)
    result_bits = tl.where(padding, 0, weight_bits).to(tl.uint32)
    tl.store(OUT_WEIGHTS + offsets, result_bits.to(tl.float32, bitcast=True), valid)


def mask_routes(ids, weights, padding, *, block=256, num_warps=4):
    out_ids = torch.empty_like(ids)
    out_weights = torch.empty_like(weights)
    if ids.numel():
        _mask_routes[(triton.cdiv(ids.numel(), block),)](
            ids,
            weights,
            padding,
            out_ids,
            out_weights,
            ids.numel(),
            ids.shape[1],
            padding.stride(0),
            block,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return out_ids, out_weights
