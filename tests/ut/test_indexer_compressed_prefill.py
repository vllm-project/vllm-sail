# SPDX-License-Identifier: Apache-2.0
"""Compressed DSA logits retain request-local masks and top-k coordinates."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

INDEXER = "vllm_sail/attention/ops/mla_sparse.py"


@pytest.mark.parametrize("policy", ["mxfp4", "int8", "fp8"])
@pytest.mark.parametrize("model_bound,capacity", [(7, 7), (300, 205)])
def test_prefill_passes_window_bound_and_masks_relative_columns(
    policy, model_bound, capacity
):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("optional torch lacks FP8 dtype")
    fp4 = policy == "mxfp4"
    q_dtype = (
        torch.uint8
        if fp4
        else (torch.int8 if policy == "int8" else torch.float8_e4m3fn)
    )
    chunk = NS(
        total_seq_lens=205,
        skip_kv_gather=True,
        token_start=0,
        token_end=2,
        cu_seqlen_ks=torch.tensor([0, 100], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([3, 105], dtype=torch.int32),
    )
    metadata = NS(
        slot_mapping=torch.arange(2),
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
        prefill=NS(chunks=[chunk]),
    )
    weights = torch.ones((2, 32), dtype=torch.bfloat16 if fp4 else torch.float32)
    calls = []
    masks = []

    def logits_api(q, kv, passed_weights, starts, ends, **kwargs):
        calls.append(kwargs)
        assert passed_weights.data_ptr() == weights.data_ptr()
        assert starts.tolist() == [0, 100] and ends.tolist() == [3, 105]
        logits = torch.arange(capacity).expand(2, -1).clone()
        return logits.to(torch.bfloat16 if fp4 else torch.float32)

    def topk(logits, starts, ends, out, *args):
        assert logits.shape == (2, capacity)
        assert starts.tolist() == [0, 0] and ends.tolist() == [3, 5]
        masked = logits.masked_fill(
            torch.arange(capacity)[None, :] >= ends[:, None], -float("inf")
        )
        out.copy_(masked.topk(2, dim=-1).indices)

    def filter_candidates(logits, starts, ends, *args):
        masks.append((starts.tolist(), ends.tolist()))

    class Torch:
        ops = NS(_C=NS(top_k_per_row_prefill=topk))

        def __getattr__(self, name):
            return getattr(torch, name)

    workspace = NS(
        get_simultaneous=lambda *specs: [
            torch.empty(shape, dtype=dtype) for shape, dtype in specs
        ]
    )
    namespace = {
        "torch": Torch(),
        "_resolve_layer_name": lambda name: name,
        "get_forward_context": lambda: NS(attn_metadata={"cache": metadata}),
        "DeepseekV32IndexerMetadata": NS,
        "current_workspace_manager": lambda: workspace,
        "is_deep_gemm_supported": lambda: True,
        "fp8_fp4_mqa_logits": logits_api,
        "int8_mqa_logits": logits_api,
        "fp8_mqa_logits": logits_api,
        "sail_ops": NS(top_k_per_row_prefill_bf16=topk),
        "_filter_candidates": filter_candidates,
        "_use_deep_select": lambda *args: False,
        "warn_deep_select_fallback": lambda: None,
    }
    namespace["_gather_workspace_shapes"] = function(
        INDEXER, "_gather_workspace_shapes", {"torch": torch, "MXFP4_BLOCK_SIZE": 32}
    )
    fn = function(INDEXER, "ppu_sparse_attn_indexer", namespace)
    out = torch.full((2, 2), -1, dtype=torch.int32)
    result = fn(
        torch.empty((2, 1)),
        "cache",
        None,
        torch.empty((2, 32, 64 if fp4 else 128), dtype=q_dtype),
        torch.empty((2, 32), dtype=torch.int32) if fp4 else None,
        None,
        weights,
        128,
        "ue8m0",
        2,
        128,
        model_bound,
        205,
        out,
        True,
        use_fp4_cache=fp4,
        candidate_blocks=torch.empty((2, 1)),
        candidate_block_size=64,
    )
    assert result is out
    assert calls[0]["max_seqlen_k"] == capacity
    assert masks == [([0, 0], [3, 5])]
    assert out.tolist() == [[2, 1], [4, 3]]
