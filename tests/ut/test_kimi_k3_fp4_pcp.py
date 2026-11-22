# SPDX-License-Identifier: Apache-2.0
"""Device tests for Kimi K3.1's gathered unfused MXFP4 indexer writer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

from vllm.platforms import current_platform

_capability = current_platform.get_device_capability()
pytestmark = [
    pytest.mark.requires_runtime,
    pytest.mark.skipif(
        _capability is None or _capability.to_int() < 89,
        reason="the MXFP4 indexer writer requires SM89+",
    ),
]

@pytest.mark.parametrize("gap", [0, 36])
def test_mxfp4_unfused_writer_matches_fused_and_preserves_padding(gap):
    """Normalized keys produce the fused writer's bytes without a top-k clear."""
    pytest.importorskip(
        "vllm_sail.patch.enhancement.models.kimi_k3_pcp",
        reason="requires the Kimi PCP patch",
    )
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
        mxfp4_k_quant_and_cache,
    )

    rows, blocks, block_size, row_bytes = 5, 2, 64, 68
    stride = (block_size + gap) * row_bytes

    def make_cache():
        raw = torch.full(
            (blocks * stride,), 0xA5, dtype=torch.uint8, device="cuda"
        )
        return torch.as_strided(
            raw, (blocks, block_size, row_bytes), (stride, row_bytes, 1)
        )

    slots = torch.tensor([0, 63, -1, 64, 67], device="cuda", dtype=torch.int64)
    raw_k = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    # A zero LayerNorm weight makes the fused writer emit norm_bias exactly;
    # this supplies an arbitrary normalized row without a second norm kernel.
    norm_w = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    norm_b = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    normalized_k = norm_b.unsqueeze(0).repeat(rows, 1)
    fused_cache = make_cache()
    unfused_cache = make_cache()
    fused_topk = torch.full((rows, 16), 7, device="cuda", dtype=torch.int32)
    unfused_topk = fused_topk.clone()

    mxfp4_k_norm_quant_insert(
        raw_k, norm_w, norm_b, 1e-6, slots, fused_cache, fused_topk, rows
    )
    mxfp4_k_quant_and_cache(normalized_k, slots, unfused_cache)

    assert torch.equal(unfused_cache, fused_cache)
    assert torch.equal(unfused_topk, torch.full_like(unfused_topk, 7))

    mxfp4_k_quant_and_cache(
        normalized_k[:0], slots[:0], unfused_cache
    )  # Empty PCP chunks are valid no-ops.
    with pytest.raises(ValueError, match="align by row"):
        mxfp4_k_quant_and_cache(normalized_k, slots[:-1], unfused_cache)
