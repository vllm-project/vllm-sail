# SPDX-License-Identifier: Apache-2.0
"""SAIL sparse logits behind the vLLM v0.33 DeepGEMM interface.

Candidate expansion, per-step schedule reuse, BF16 top-k and request-local
remapping stay upstream. Only the external SAIL API and its layouts live here.
"""

import functools
import importlib

import torch
from vllm.platforms import current_platform

from vllm_sail.utils.deep_gemm import is_deep_gemm_supported
from vllm_sail.utils.deep_select import is_deep_select_available

_APIS = (
    "get_sparse_mqa_logits_metadata",
    "get_paged_sparse_mqa_logits_metadata",
    "fp8_fp4_sparse_mqa_logits",
    "fp8_fp4_paged_sparse_mqa_logits",
)


@functools.cache
def _backend():
    try:
        return importlib.import_module("deep_gemm")
    except ModuleNotFoundError as exc:
        if exc.name == "deep_gemm":
            return None
        raise


def has_deep_gemm_sparse_mqa() -> bool:
    return (
        current_platform.is_ppu()
        and current_platform.is_device_capability(89)
        and is_deep_gemm_supported()
        and all(callable(getattr(_backend(), name, None)) for name in _APIS)
    )


def check_sparse_indexer_support(vllm_config) -> None:
    """Fail at model/builder construction, before any sparse kernel launch."""
    if not has_deep_gemm_sparse_mqa():
        raise ValueError(
            "indexer_sparse_logits requires M890 and all four SAIL DeepGEMM "
            "sparse logits APIs (VLLM_USE_DEEP_GEMM must be enabled)"
        )
    if not is_deep_select_available():
        raise ValueError(
            "PPU indexer_sparse_logits requires the enabled SAIL DeepSelect wheel"
        )
    hf = vllm_config.model_config.hf_text_config
    if hf.model_type not in ("deepseek_v41", "deepseek_v41_text"):
        raise ValueError(
            "PPU indexer_sparse_logits is supported only for DeepSeek V4.1"
        )
    if (hf.index_n_heads, hf.index_head_dim) != (32, 128):
        raise ValueError(
            "SAIL sparse logits require index_n_heads=32 and index_head_dim=128"
        )
    block = getattr(hf, "candidate_block_size", 0)
    count = getattr(hf, "candidate_topk_blocks", 0)
    topk = getattr(hf, "index_topk", 0)
    if block not in (8, 16, 32) or count <= 0 or count & (count - 1):
        raise ValueError(
            "SAIL sparse logits require candidate blocks of 8/16/32 and a power-of-two count"
        )
    num_sparse = count * (2 if block == 32 else 1)
    if num_sparse % 4 or num_sparse > 4096:
        raise ValueError(
            "SAIL sparse logits require 4..4096 sparse blocks, divisible by 4"
        )
    if not 0 < topk <= min(4096, block * count):
        raise ValueError(
            "Sparse indexer top-k must be positive, <=4096 and <=candidate width"
        )


def get_sparse_mqa_logits_metadata(
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    num_kv_tokens: int,
    sparse_kv_block_indices: torch.Tensor,
    qk_dtype: torch.dtype,
    sparse_block_kv: int,
) -> torch.Tensor:
    return _backend().get_sparse_mqa_logits_metadata(
        cu_seqlen_ks.contiguous(),
        cu_seqlen_ke.contiguous(),
        num_kv_tokens,
        sparse_kv_block_indices.contiguous(),
        qk_dtype,
        sparse_block_kv,
        use_unaligned_ks=True,
    )


def get_paged_sparse_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    indices: torch.Tensor,
    page_kv: int,
    sparse_kv_block_indices: torch.Tensor,
    qk_dtype: torch.dtype,
    sparse_block_kv: int,
) -> torch.Tensor:
    return _backend().get_paged_sparse_mqa_logits_metadata(
        context_lens.contiguous(),
        block_table.contiguous(),
        indices.contiguous(),
        page_kv,
        sparse_kv_block_indices.contiguous(),
        qk_dtype,
        sparse_block_kv,
    )


def _query(q, weights, *, paged):
    values, scales = q
    expected = (1, 32, 64) if paged else (32, 64)
    if tuple(values.shape[1:]) != expected or values.dtype != torch.int8:
        raise ValueError(
            "SAIL sparse Q requires packed MXFP4 H32/D128 and paged next_n=1"
        )
    if scales.dtype != torch.int32 or scales.shape != values.shape[:-1]:
        raise ValueError("SAIL sparse Q requires one packed int32 scale per head")
    if weights.dtype != torch.bfloat16 or weights.shape != (values.shape[0], 32):
        raise ValueError(
            "SAIL sparse logits require producer-owned BF16 [rows,32] weights"
        )
    return (values.contiguous(), scales.contiguous()), weights.contiguous()


def fp8_fp4_sparse_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    metadata: torch.Tensor,
    num_max_sparse_blocks: int,
    sparse_block_kv: int,
) -> torch.Tensor:
    q, weights = _query(q, weights, paged=False)
    return _backend().fp8_fp4_sparse_mqa_logits(
        q,
        (kv[0].contiguous(), kv[1].contiguous()),
        weights,
        metadata,
        num_max_sparse_blocks,
        sparse_block_kv,
        use_unaligned_ks=True,
    )


def fp8_fp4_paged_sparse_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    metadata: torch.Tensor,
    num_max_sparse_blocks: int,
    sparse_block_kv: int,
) -> torch.Tensor:
    q, weights = _query(q, weights, paged=True)
    if kv_cache.dtype != torch.uint8 or kv_cache.stride(0) % 512:
        raise ValueError(
            "SAIL sparse cache requires uint8 pages with a 512-byte-aligned stride"
        )
    # Preserve padded/interleaved cache pages; making this contiguous changes
    # their physical stride and can break the 512-byte SAIL contract.
    return _backend().fp8_fp4_paged_sparse_mqa_logits(
        q, kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv
    )
