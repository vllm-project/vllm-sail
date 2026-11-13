# SPDX-License-Identifier: Apache-2.0
"""SAIL DeepGEMM indexer API regression coverage."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

DEEP_GEMM = "vllm_sail/utils/deep_gemm.py"
INDEXER = "vllm_sail/attention/ops/mla_sparse.py"


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)

    def to(self, dtype):
        pytest.fail("weights must be forwarded without casting")

    def contiguous(self):
        pytest.fail("weights must retain their original row layout")


def _api_namespace(symbol):
    calls = []

    def api(*args, **kwargs):
        calls.append((args, kwargs))
        return "logits"

    return {
        "torch": NS(
            bfloat16="bf16",
            float32="fp32",
            int64="int64",
            empty=lambda shape, dtype, device: Tensor(shape, dtype),
        ),
        "_lazy_init": lambda: None,
        "_missing": lambda: pytest.fail("unexpected missing API"),
        symbol: api,
    }, calls


def test_weighted_mxfp4_uses_relative_column_api():
    ns, calls = _api_namespace("_fp8_fp4_mqa_logits_impl")
    fn = function(DEEP_GEMM, "fp8_fp4_mqa_logits", ns)
    q, kv = (Tensor((2, 32, 64)), object()), (Tensor((257, 64)), object())
    weights, starts, ends = Tensor((2, 32)), object(), object()
    assert fn(q, kv, weights, starts, ends, False, max_seqlen_k=129) == "logits"
    assert calls == [((q, kv, weights, starts, ends), {"max_seqlen_k": 129})]
    assert calls[0][0][2] is weights
    assert weights.dtype == "bf16"
    with pytest.raises(ValueError, match="BF16"):
        fn(q, kv, weights, starts, ends, False, logits_dtype="fp32", max_seqlen_k=129)
    assert len(calls) == 1


def test_paged_weighted_mxfp4_forwards_the_same_request_ids():
    ns, calls = _api_namespace("_fp8_fp4_paged_mqa_logits_impl")
    fn = function(DEEP_GEMM, "fp8_fp4_paged_mqa_logits", ns)
    ids, contexts, tables, schedule = object(), object(), object(), object()
    q, cache, weights = object(), object(), Tensor((6, 32))
    assert (
        fn(q, cache, weights, contexts, tables, schedule, 8192, False, indices=ids)
        == "logits"
    )
    assert calls == [
        ((q, cache, weights, contexts, tables, schedule, 8192), {"indices": ids})
    ]
    assert calls[0][0][2] is weights


@pytest.mark.parametrize("paged", [False, True])
def test_weighted_mxfp4_rejects_fp32_instead_of_silently_casting(paged):
    name = "fp8_fp4_paged_mqa_logits" if paged else "fp8_fp4_mqa_logits"
    ns, calls = _api_namespace("_" + name + "_impl")
    fn = function(DEEP_GEMM, name, ns)
    weights = Tensor((2, 32), "fp32")
    with pytest.raises(TypeError, match="BF16 weights"):
        if paged:
            fn(None, None, weights, None, None, None, 129, False, indices=None)
        else:
            fn(None, None, weights, None, None, False, max_seqlen_k=129)
    assert not calls


@pytest.mark.parametrize("name", ["fp8_mqa_logits", "int8_mqa_logits"])
@pytest.mark.parametrize("capacity", [0, 129])
def test_legacy_nonpaged_forwards_compressed_capacity_and_original_weights(
    name, capacity
):
    ns, calls = _api_namespace("_" + name + "_impl")
    fn = function(DEEP_GEMM, name, ns)
    weights = Tensor((2, 32), "fp32")
    assert fn(None, None, weights, None, None, False, max_seqlen_k=capacity) == "logits"
    assert calls[0][0][2] is weights
    assert calls[0][1] == {"clean_logits": False, "max_seqlen_k": capacity}


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_int8_preserves_callers_weights_dtype(dtype):
    ns, calls = _api_namespace("_int8_mqa_logits_impl")
    weights = Tensor((2, 32), dtype)
    fn = function(DEEP_GEMM, "int8_mqa_logits", ns)
    fn(None, None, weights, None, None, False, max_seqlen_k=129)
    assert calls[0][0][2] is weights
    assert weights.dtype == dtype


@pytest.mark.parametrize("capacity", [0, -1])
def test_mxfp4_rejects_nonpositive_window_capacity(capacity):
    ns, calls = _api_namespace("_fp8_fp4_mqa_logits_impl")
    fn = function(DEEP_GEMM, "fp8_fp4_mqa_logits", ns)
    with pytest.raises(ValueError, match="positive max_seqlen_k"):
        fn(None, None, Tensor((2, 32)), None, None, False, max_seqlen_k=capacity)
    assert not calls


@pytest.mark.parametrize(
    "contexts,lengths,expected",
    [
        ([[9, 10], [19, 20]], [2, 1], [[9], [10], [20], [0]]),
        # C2 contexts must be shifted after compression, without a token offset.
        ([[4, 5], [9, 10]], [2, 1], [[4], [5], [10], [0]]),
    ],
)
def test_ragged_verify_keeps_request_contexts(contexts, lengths, expected):
    torch = pytest.importorskip("torch")
    fn = function(INDEXER, "_paged_decode_rows", {"torch": torch})
    lens, tables, ids = fn(
        torch.tensor(contexts, dtype=torch.int32),
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        2,
        None,
    )
    assert lens.tolist() == expected
    assert tables.tolist() == [[11, 12], [11, 12], [21, 22], [21, 22]]
    assert ids.tolist() == [0, 0, 1, -4]


def test_upstream_flattened_mtp_recovers_true_request_ownership():
    torch = pytest.importorskip("torch")
    fn = function(INDEXER, "_paged_decode_rows", {"torch": torch})
    _, _, ids = fn(
        torch.tensor([[8], [9], [10], [7], [9], [10]], dtype=torch.int32),
        torch.zeros((6, 2), dtype=torch.int32),
        torch.ones(6, dtype=torch.int32),
        1,
        None,
        torch.tensor([3, 1, 2], dtype=torch.int32),
    )
    assert ids.tolist() == [0, 0, 0, 1, 2, 2]


def test_varlen_device_rows_preserve_explicit_request_ids_and_padding():
    torch = pytest.importorskip("torch")
    fn = function(INDEXER, "_paged_decode_rows", {"torch": torch})
    contexts = torch.tensor([[8], [9], [10], [7], [9], [10], [0]], dtype=torch.int32)
    tables = torch.arange(14, dtype=torch.int32).reshape(7, 2)
    request_ids = torch.tensor([0, 0, 0, 1, 2, 2, 3], dtype=torch.int32)
    lens, expanded_tables, ids = fn(
        contexts,
        tables,
        torch.ones(7, dtype=torch.int32),
        1,
        request_ids,
        torch.tensor([3, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(lens, contexts)
    assert torch.equal(expanded_tables, tables)
    assert torch.equal(ids, request_ids)
