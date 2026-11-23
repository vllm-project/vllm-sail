# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _exact_bf16_to_fp32(X, Y, N, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offset, offset < N, other=0)
    # Numeric conversions may canonicalize NaN payloads.
    bits = value.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
    tl.store(Y + offset, bits.to(tl.float32, bitcast=True), offset < N)


def exact_bf16_to_fp32(x):
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.is_neg()
    ):
        raise ValueError("Exact cast requires resolved contiguous device BF16 input")
    output = torch.empty_like(x, dtype=torch.float32)
    if x.numel():
        _exact_bf16_to_fp32[(triton.cdiv(x.numel(), 256),)](
            x, output, x.numel(), 256, num_warps=4, enable_fp_fusion=False
        )
    return output
