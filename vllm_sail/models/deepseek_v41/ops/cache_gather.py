# SPDX-License-Identifier: Apache-2.0
"""Software gather for the interleaved V4.1 KV records on Zhenwu 810E.

Upstream's V4.1 gather kernels bitcast uint8 to ``tl.float8e4nv``, which Triton
refuses to compile on SM80 (PPU-ZW810E, capability 8.0). This module mirrors
their address arithmetic and quantization semantics with the software decoders
already used by the V4 planar gather, so 810E can read both V4.1 records.
ZW-M890P (capability 8.9) keeps the upstream kernels.

Record contract, from vllm/models/deepseek_v41/common/ops/cache_utils.py:39-56
and .../common/ops/fused_compress_quant_cache.py:305-315:

* 584B V4 -- **planar**: a page holds ``[B x 576 value bytes][B x 8 scale
  bytes]``; 448 fp8 e4m3 NoPE + 64 bf16 RoPE, 7 UE8M0 scales of 64 dims plus a
  pad byte. Owned by ``vllm_sail.models.deepseek_v4.ops.cache``.
* 528B V4.1 MXFP8 -- **interleaved**: token t is ``[512 fp8 e4m3 | 16 UE8M0
  scales of 32 dims]``, so the token stride is 528 and the scales sit at +512.
* 288B V4.1 NVFP4 -- **interleaved**: token t is ``[256 e2m1 pairs | 32 fp8
  e4m3 scales of 16 dims]``, even element in the low nibble, token stride 288
  and scales at +256. Compressed cache only.

The two layouts are not interchangeable: reading an interleaved record with the
planar formula shifts every token's values by 16 bytes and reads another
token's value bytes as UE8M0 exponents, which silently produces inf/NaN instead
of raising. ``record_layout``/``token_offsets`` below are the single source of
truth for the address arithmetic and are read by the offline layout UT.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_sail.models.deepseek_v4.ops.cache import _decode_e4m3fn


def record_layout(record_bytes):
    """Return the byte layout of one paged KV record.

    Pure Python on purpose: the offline UT extracts this body without torch or
    vLLM. ``token_stride`` is the per-token byte step of the value region and
    ``scale_offset`` the distance from a token's values to its own scales;
    ``scale_offset`` is None for the planar V4 record, whose scales live in a
    separate region after the whole value plane.
    """
    if record_bytes == 584:
        return {
            "planar": True,
            "token_stride": 576,
            "scale_offset": None,
            "scale_bytes": 8,
            "head_dim": 512,
            "quant_block": 64,
            "num_scales": 7,
        }
    if record_bytes == 528:
        return {
            "planar": False,
            "token_stride": 528,
            "scale_offset": 512,
            "scale_bytes": 16,
            "head_dim": 512,
            "quant_block": 32,
            "num_scales": 16,
        }
    if record_bytes == 288:
        return {
            "planar": False,
            "token_stride": 288,
            "scale_offset": 256,
            "scale_bytes": 32,
            "head_dim": 512,
            "quant_block": 16,
            "num_scales": 32,
        }
    raise ValueError(f"unsupported paged KV record width {record_bytes}")


def token_offsets(layout, block_size, t):
    """Byte offsets of token ``t``'s value and scale regions inside one page.

    Pure Python; mirrors the pointer arithmetic the kernels below perform.
    """
    stride = layout["token_stride"]
    if layout["planar"]:
        return (t * stride, block_size * stride + t * layout["scale_bytes"])
    base = t * stride
    return (base, base + layout["scale_offset"])


@triton.jit
def _decode_e2m1(nibble):
    """Decode a 4-bit e2m1 value (1 sign, 2 exp bias 1, 1 mant) to fp32.

    Pure integer/float ops, so it compiles on SM80 where neither
    ``tl.float8e4nv`` nor a float4 dtype is available. The eight magnitudes are
    0, 0.5, 1, 1.5, 2, 3, 4, 6.
    """
    sign = (nibble >> 3) & 1
    exponent = (nibble >> 1) & 3
    mant = nibble & 1
    normal = (1.0 + mant.to(tl.float32) * 0.5) * tl.exp2((exponent - 1).to(tl.float32))
    subnormal = mant.to(tl.float32) * 0.5
    magnitude = tl.where(exponent != 0, normal, subnormal)
    return tl.where(sign != 0, -magnitude, magnitude)


@triton.jit
def _gather_mxfp8_sm80_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    block_stride: tl.constexpr,
    token_stride: tl.constexpr,
    scale_offset: tl.constexpr,
    head_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
):
    """528B V4.1 MXFP8 record: interleaved values then per-token UE8M0 scales.

    Same launch shape and dequantization semantics as upstream's
    ``_dequantize_and_gather_k_mxfp8_kernel``, with the hardware fp8 bitcast
    replaced by ``_decode_e4m3fn``.
    """
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    d = tl.arange(0, head_dim)
    s = tl.arange(0, scale_dim)
    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size
        physical_block_idx = tl.load(
            block_table_ptr + batch_idx * max_blocks_per_seq + block_in_seq
        )
        page = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        token_ptr = page + pos_in_block * token_stride

        x = _decode_e4m3fn(tl.load(token_ptr + d))
        encoded = tl.load(token_ptr + scale_offset + s)
        scale = tl.exp2(encoded.to(tl.float32) - 127.0)
        tiles = tl.reshape(x, (scale_dim, quant_block))
        dequant = tl.reshape(tiles * tl.reshape(scale, (scale_dim, 1)), (head_dim,))

        out_row = (
            out_ptr
            + batch_idx.to(tl.int64) * out_stride0
            + (offset + i).to(tl.int64) * out_stride1
        )
        tl.store(out_row + d, dequant.to(tl.bfloat16))


@triton.jit
def _gather_nvfp4_sm80_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    block_stride: tl.constexpr,
    token_stride: tl.constexpr,
    scale_offset: tl.constexpr,
    head_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
):
    """288B V4.1 NVFP4 record: interleaved e2m1 pairs then per-token e4m3 scales.

    Upstream places each nibble as fp16 bits and compensates with ``* 16384``;
    decoding e2m1 to its true magnitude instead removes both that factor and the
    int16->fp16 bitcast, and ``tl.interleave`` replaces ``tl.join``. The e4m3
    scale is decoded in software, so no ``tl.float8e4nv`` reference remains.
    """
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    packed_dim: tl.constexpr = head_dim // 2
    d = tl.arange(0, head_dim)
    b = tl.arange(0, packed_dim)
    s = tl.arange(0, scale_dim)
    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size
        physical_block_idx = tl.load(
            block_table_ptr + batch_idx * max_blocks_per_seq + block_in_seq
        )
        page = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        token_ptr = page + pos_in_block * token_stride

        packed = tl.load(token_ptr + b).to(tl.int32)
        even = _decode_e2m1(packed & 0xF)
        odd = _decode_e2m1((packed >> 4) & 0xF)
        values = tl.reshape(tl.interleave(even, odd), (head_dim,))
        scale = _decode_e4m3fn(tl.load(token_ptr + scale_offset + s))
        tiles = tl.reshape(values, (scale_dim, quant_block))
        dequant = tl.reshape(tiles * tl.reshape(scale, (scale_dim, 1)), (head_dim,))

        out_row = (
            out_ptr
            + batch_idx.to(tl.int64) * out_stride0
            + (offset + i).to(tl.int64) * out_stride1
        )
        tl.store(out_row + d, dequant.to(tl.bfloat16))


_NUM_WORKERS = 128


def _launch(
    kernel,
    record_bytes,
    out,
    k_cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset,
):
    layout = record_layout(record_bytes)
    if layout["planar"]:
        raise ValueError("the planar V4 record belongs to the V4 software gather")
    if k_cache.shape[-1] != record_bytes:
        raise ValueError(
            f"expected a {record_bytes}-byte record, got {k_cache.shape[-1]}"
        )
    kernel[(seq_lens.shape[0], _NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        block_stride=k_cache.stride(0),
        token_stride=layout["token_stride"],
        scale_offset=layout["scale_offset"],
        head_dim=layout["head_dim"],
        scale_dim=layout["num_scales"],
        quant_block=layout["quant_block"],
    )


def _gather_mxfp8_sm80_op(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    _launch(
        _gather_mxfp8_sm80_kernel,
        528,
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
    )


def _gather_nvfp4_sm80_op(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    _launch(
        _gather_nvfp4_sm80_kernel,
        288,
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
    )


def _gather_mxfp8_sm80_op_fake(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    return None


def _gather_nvfp4_sm80_op_fake(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    return None


def register_ops() -> None:
    """Register before tracing; module import alone does not mutate torch.ops."""
    global _registered
    if _registered:
        return
    direct_register_custom_op(
        op_name="ppu_deepseek_v41_mxfp8_dequant_gather",
        op_func=_gather_mxfp8_sm80_op,
        mutates_args=["out"],
        fake_impl=_gather_mxfp8_sm80_op_fake,
    )
    direct_register_custom_op(
        op_name="ppu_deepseek_v41_nvfp4_dequant_gather",
        op_func=_gather_nvfp4_sm80_op,
        mutates_args=["out"],
        fake_impl=_gather_nvfp4_sm80_op_fake,
    )
    _registered = True


_registered = False
