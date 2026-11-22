# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 PCP regression tests without torch, vLLM, or a device."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace as NS

import pytest

from tests.support.source import (
    PACKAGE_ROOT,
    assert_accepts_upstream_keywords,
    definition,
    function,
)

PATCH = "vllm_sail/patch/enhancement/models/kimi_k3_pcp.py"


class HostTensor:
    """Small one-dimensional tensor double supporting the PCP index algebra."""

    def __init__(
        self,
        values,
        *,
        dtype=None,
        device="cpu",
        _storage=None,
        _indices=None,
    ):
        self._storage = list(values) if _storage is None else _storage
        self._indices = (
            list(range(len(self._storage))) if _indices is None else list(_indices)
        )
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        return (len(self._indices),)

    def size(self, dim):
        assert dim == 0
        return len(self._indices)

    def contiguous(self):
        return self

    def new_full(self, shape, value):
        return HostTensor(
            [value] * shape[0], dtype=self.dtype, device=self.device
        )

    def tolist(self):
        return [self._storage[index] for index in self._indices]

    def clone(self):
        return HostTensor(self.tolist(), dtype=self.dtype, device=self.device)

    def cpu(self):
        return self

    def to(self, dtype=None, *, device=None):
        return HostTensor(
            self.tolist(),
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
        )

    def diff(self):
        values = self.tolist()
        return HostTensor(
            [
                right - left
                for left, right in zip(values, values[1:], strict=False)
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def flatten(self):
        return self

    def add_(self, value):
        for index in self._indices:
            self._storage[index] += value
        return self

    def fill_(self, value):
        for index in self._indices:
            self._storage[index] = value
        return self

    def sum(self):
        return HostTensor([sum(self.tolist())], dtype=self.dtype)

    def item(self):
        values = self.tolist()
        assert len(values) == 1
        return values[0]

    def _selected(self, index):
        if isinstance(index, slice):
            return self._indices[index]
        if isinstance(index, int):
            return [self._indices[index]]
        indices = index.tolist() if isinstance(index, HostTensor) else list(index)
        if indices and all(isinstance(value, bool) for value in indices):
            assert len(indices) == len(self._indices)
            return [
                storage_index
                for storage_index, selected in zip(
                    self._indices, indices, strict=True
                )
                if selected
            ]
        return [self._indices[value] for value in indices]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._storage[self._indices[index]]
        return HostTensor(
            [],
            dtype=self.dtype,
            device=self.device,
            _storage=self._storage,
            _indices=self._selected(index),
        )

    def __setitem__(self, index, value):
        selected = self._selected(index)
        values = value.tolist() if isinstance(value, HostTensor) else value
        if isinstance(values, list):
            assert len(selected) == len(values)
        else:
            values = [values] * len(selected)
        for storage_index, item in zip(selected, values, strict=True):
            self._storage[storage_index] = item

    def _binary(self, other, operation, *, dtype=None):
        left = self.tolist()
        right = other.tolist() if isinstance(other, HostTensor) else other
        if isinstance(right, list):
            assert len(left) == len(right)
        else:
            right = [right] * len(left)
        return HostTensor(
            [operation(a, b) for a, b in zip(left, right, strict=True)],
            dtype=dtype,
            device=self.device,
        )

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b, dtype=self.dtype)

    def __ne__(self, other):
        return self._binary(other, lambda a, b: a != b, dtype="bool")

    def __ge__(self, other):
        return self._binary(other, lambda a, b: a >= b, dtype="bool")

    def __gt__(self, other):
        return self._binary(other, lambda a, b: a > b, dtype="bool")

    def __or__(self, other):
        return self._binary(other, lambda a, b: a or b, dtype="bool")

    def __iand__(self, other):
        result = self._binary(other, lambda a, b: a and b, dtype="bool")
        self[:] = result
        return self


def _tensor(values, *, dtype=None, device=None):
    if isinstance(values, int):
        values = [0] * values
    return HostTensor(
        values,
        dtype=dtype,
        device="cpu" if device is None else device,
    )


def _cumsum(tensor, *, dim, out=None):
    assert dim == 0
    total = 0
    values = []
    for value in tensor.tolist():
        total += value
        values.append(total)
    result = HostTensor(values, dtype=tensor.dtype, device=tensor.device)
    if out is not None:
        out[:] = result
        return out
    return result


def _repeat_interleave(tensor, repeats):
    return HostTensor(
        [
            value
            for value, count in zip(
                tensor.tolist(), repeats.tolist(), strict=True
            )
            for _ in range(count)
        ],
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _cat(tensors):
    return HostTensor(
        [value for tensor in tensors for value in tensor.tolist()],
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )


def _argsort(tensor, *, stable=False, descending=False):
    values = tensor.tolist()
    return HostTensor(
        sorted(range(len(values)), key=values.__getitem__, reverse=descending),
        dtype="int64",
        device=tensor.device,
    )


FAKE_TORCH = NS(
    Tensor=HostTensor,
    int32="int32",
    int64="int64",
    bool="bool",
    tensor=_tensor,
    zeros=_tensor,
    repeat_interleave=_repeat_interleave,
    cat=_cat,
    nonzero=lambda tensor: HostTensor(
        [index for index, value in enumerate(tensor.tolist()) if value]
    ),
    cumsum=_cumsum,
    argsort=_argsort,
    full=lambda shape, value, **kwargs: HostTensor(
        [value] * shape[0], dtype=kwargs.get("dtype"), device=kwargs.get("device")
    ),
    ones_like=lambda tensor, **kwargs: HostTensor(
        [True] * tensor.shape[0],
        dtype=kwargs.get("dtype"),
        device=tensor.device,
    ),
    bincount=lambda tensor, minlength: HostTensor(
        [
            tensor.tolist().count(index)
            for index in range(max(minlength, max(tensor.tolist(), default=-1) + 1))
        ]
    ),
)


def _load(name, **namespace):
    return function(PATCH, name, namespace)


def test_pcp_split_prefill_requests_uses_adjacent_run_boundaries():
    split = _load("_pcp_split_prefill_requests", torch=FAKE_TORCH)
    context_lens = HostTensor([0, 0], dtype="int32")
    query_start_loc = HostTensor([0, 5, 7], dtype="int32")
    common = NS(positions=HostTensor([20, 0, 1, 2, 7, 8, 0, 1]))

    split_context, split_qsl, expand_idx = split(
        NS(), context_lens, query_start_loc, common, num_decode_tokens=1
    )

    assert split_context.tolist() == [0, 7, 0]
    assert split_qsl.tolist() == [0, 3, 5, 7]
    assert split_qsl.diff().tolist() == [3, 2, 2]
    assert expand_idx.tolist() == [0, 0, 1]


def test_pcp_split_view_device_qsl_and_query_lens():
    split = _load("_pcp_split_prefill_requests", torch=FAKE_TORCH)
    original_qsl = HostTensor([0, 5, 7], dtype="int32")
    common = NS(positions=HostTensor([20, 0, 1, 2, 7, 8, 0, 1]))
    _, split_qsl, _ = split(
        NS(),
        HostTensor([0, 0], dtype="int32"),
        original_qsl,
        common,
        num_decode_tokens=1,
    )

    device_qsl = split_qsl.to(device="device-double")
    query_lens = split_qsl.diff()

    assert device_qsl.tolist() == [0, 3, 5, 7]
    assert device_qsl.device == "device-double"
    assert device_qsl.tolist() != original_qsl.tolist()
    assert query_lens.tolist() == [3, 2, 2]
    assert query_lens.sum().item() == 7


def test_pcp_global_batch_reconstruction_deduplicates_and_restores_rows():
    reconstruct = _load("_reconstruct_pcp_global_batch", torch=FAKE_TORCH)
    positions = HostTensor([20, 0, 1, 6, 7, 0, 20, 2, 3, 4, 5, 0])
    state_slots = HostTensor([4, 7, 7, 7, 7, -1, 4, 7, 7, 7, 7, -1])

    global_idx, inverse, global_qsl, has_initial, request_state = reconstruct(
        positions, state_slots
    )

    assert positions[global_idx].tolist() == [20, 0, 1, 2, 3, 4, 5, 6, 7]
    assert state_slots[global_idx].tolist() == [4, 7, 7, 7, 7, 7, 7, 7, 7]
    assert inverse.tolist() == [0, 1, 2, 7, 8, -1, 0, 3, 4, 5, 6, -1]
    assert global_qsl.tolist() == [0, 1, 9]
    assert global_qsl.dtype == "int32"
    assert has_initial.tolist() == [True, False]
    assert request_state.tolist() == [4, 7]


@dataclass
class CommonMetadata:
    slot_mapping: HostTensor
    num_actual_tokens: int
    query_start_loc: HostTensor
    seq_lens: HostTensor
    block_table_tensor: HostTensor


class IndexerBase:
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        assert common_prefix_len == 0
        assert fast_build is False
        return common_attn_metadata


class KimiK3IndexerMetadataBuilder(IndexerBase):
    pass


def _dsa_builder():
    builder = object.__new__(KimiK3IndexerMetadataBuilder)
    builder.use_pcp = True
    builder.pcp_rank = 1
    builder.dcp_world_size = 1
    builder.cp_kv_cache_interleave_size = 1
    builder.kv_cache_spec = NS(block_size=64)
    builder.slot_mapping_buffer = HostTensor([0] * 8, dtype="int64")
    return builder


def test_kimi_k3_pcp_indexer_gather_uses_padded_slot_mapping(monkeypatch):
    gathered = HostTensor([10, 11, -1, -1, 101, 102, -1, -1], dtype="int64")

    def compress(num_tokens, slot_mapping, *args, out, **kwargs):
        assert num_tokens == 2
        assert slot_mapping.tolist() == [55, 66, 77, 88]
        out.fill_(-1)
        out[:num_tokens] = HostTensor([101, 102])
        return out[:num_tokens]

    class PCPGroup:
        world_size = 2
        calls = 0

        def all_gather(self, tensor, dim=0):
            self.calls += 1
            assert dim == 0
            assert tensor.tolist() == [101, 102, -1, -1]
            return gathered

    group = PCPGroup()
    parallel_state = ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_pcp_group = lambda: group
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)
    build = _load(
        "_dsa_build",
        get_compressed_slot_mapping=compress,
        KimiK3IndexerMetadataBuilder=KimiK3IndexerMetadataBuilder,
        replace=replace,
    )
    common = CommonMetadata(
        slot_mapping=HostTensor([11, 22, 33, 44, 55, 66, 77, 88]),
        num_actual_tokens=2,
        query_start_loc=HostTensor([0, 2], dtype="int32"),
        seq_lens=HostTensor([2], dtype="int32"),
        block_table_tensor=HostTensor([0], dtype="int32"),
    )

    metadata = build(_dsa_builder(), 0, common)

    assert group.calls == 1
    assert metadata.slot_mapping is gathered


def test_kimi_k3_indexer_collapsed_mapping_does_not_gather():
    def compress(num_tokens, slot_mapping, *args, out, **kwargs):
        out.fill_(-1)
        out[:num_tokens] = slot_mapping
        return out[:num_tokens]

    build = _load(
        "_dsa_build",
        get_pcp_group=lambda: (_ for _ in ()).throw(
            AssertionError("collapsed PCP mapping must not all-gather")
        ),
        get_compressed_slot_mapping=compress,
        KimiK3IndexerMetadataBuilder=KimiK3IndexerMetadataBuilder,
        replace=replace,
    )
    common = CommonMetadata(
        slot_mapping=HostTensor([55, 66]),
        num_actual_tokens=2,
        query_start_loc=HostTensor([0, 2]),
        seq_lens=HostTensor([2]),
        block_table_tensor=HostTensor([0]),
    )

    metadata = build(_dsa_builder(), 0, common)

    assert metadata.slot_mapping.tolist() == [55, 66]


def test_pcp_manager_uses_global_prefill_and_publishes_registry():
    has_global_prefill = _load("has_global_prefill")
    assert has_global_prefill(NS(_global_batch=None)) is False
    assert has_global_prefill(NS(_global_batch=NS(has_prefill=False))) is False
    assert has_global_prefill(NS(_global_batch=NS(has_prefill=True))) is True

    manager = object()
    provider = NS(_current_pcp_manager=None)
    build_manager = _load(
        "maybe_build_pcp_manager",
        _upstream_maybe_build_pcp_manager=lambda *args: manager,
        _pcp_manager=provider,
    )
    get_manager = _load("get_current_pcp_manager", _pcp_manager=provider)

    assert build_manager("config") is manager
    assert get_manager() is manager


def test_kda_forward_pcp_uses_non_spec_query_metadata(monkeypatch):
    utils = ModuleType("vllm.v1.attention.backends.utils")
    utils.compute_causal_conv1d_metadata = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    class Metadata:
        pass

    metadata = Metadata()
    metadata.non_spec_query_start_loc = HostTensor([0, 2, 3], dtype="int32")
    metadata.non_spec_state_indices_tensor = HostTensor([4, 7], dtype="int64")
    attn_metadata = {"layer": metadata}

    class StopAfterStateGather(Exception):
        pass

    class PCPGroup:
        rank_in_group = 0

        def __init__(self):
            self.inputs = []

        def all_gather(self, tensor, dim=0):
            assert dim == 0
            self.inputs.append(tensor.tolist())
            if len(self.inputs) == 3:
                raise StopAfterStateGather
            return tensor

    group = PCPGroup()
    forward = _load(
        "_forward_pcp",
        torch=FAKE_TORCH,
        get_current_pcp_manager=lambda: NS(has_global_prefill=lambda: True),
        get_forward_context=lambda: NS(attn_metadata=attn_metadata),
        get_pcp_group=lambda: group,
        _kda=NS(KimiK3KDAMetadata=Metadata),
    )

    with pytest.raises(StopAfterStateGather):
        forward(
            NS(prefix="layer"),
            HostTensor([10, 20, 30, 0]),
            HostTensor([0, 1, 7, 0]),
        )

    assert group.inputs == [
        [10, 20, 30, 0],
        [0, 1, 7, 0],
        [4, 4, 7, -1],
    ]


def test_kda_forward_delegates_decode_only_and_uses_pcp_result():
    calls = []

    def upstream(self, hidden_states, positions):
        calls.append((hidden_states, positions))
        return "local"

    forward = _load("_kda_forward", _upstream_kda_forward=upstream)
    layer = NS(use_pcp=False)
    assert forward(layer, "hidden", "positions") == "local"

    layer.use_pcp = True
    layer._forward_pcp = lambda hidden, positions: None
    assert forward(layer, "hidden", "positions") == "local"

    layer._forward_pcp = lambda hidden, positions: "pcp"
    assert forward(layer, "hidden", "positions") == "pcp"
    assert calls == [("hidden", "positions"), ("hidden", "positions")]


def test_attention_capture_decorator_wraps_upstream_bound_body():
    source = (PACKAGE_ROOT / PATCH).read_text()
    assert ")(_mla.eager_break_during_capture(bind_body(_attention_body, _mla)))" in (
        source
    )
    assert "bind_body(_mla.eager_break_during_capture(_attention_body), _mla)" not in (
        source
    )


def test_kda_backend_declares_pcp_support():
    backend_supports_pcp = _load("supports_pcp")

    assert backend_supports_pcp(object()) is True


def test_fp4_guard_accepts_pcp_and_preserves_the_norm_contract():
    guard = _load(
        "_kimi_k3_indexer_uses_fp4",
        dsa_indexer_uses_fp4=lambda config: True,
    )
    dsa_config = NS(index_k_norm_type="layernorm", index_rope_dim=0)
    config = NS(
        model_config=NS(hf_text_config=NS(dsa_config=dsa_config)),
        parallel_config=NS(prefill_context_parallel_size=2),
    )
    assert guard(config) is True

    dsa_config.index_k_norm_type = "rmsnorm"
    with pytest.raises(ValueError, match="index_k_norm_type='layernorm'"):
        guard(config)


def test_fp4_pcp_guards_only_enable_the_normalized_key_writer_path():
    source = (PACKAGE_ROOT / PATCH).read_text()
    path = PACKAGE_ROOT / PATCH
    fp4_guard = ast.get_source_segment(
        source, definition(path, "_kimi_k3_indexer_uses_fp4")
    )
    indexer_init = ast.get_source_segment(source, definition(path, "_dsa_indexer_init"))
    score = ast.get_source_segment(source, definition(path, "_indexer_score_body"))
    fused = ast.get_source_segment(source, definition(path, "_mxfp4_k_insert"))
    unfused = ast.get_source_segment(source, definition(path, "_mxfp4_k_quant_insert"))
    writer = ast.get_source_segment(source, definition(path, "mxfp4_k_quant_and_cache"))
    assert "not supported with prefill context parallelism" not in fp4_guard
    assert "pcp_unfused_fp4" in indexer_init
    assert 'index_k_norm_type == "layernorm"' in indexer_init
    assert "index_rope_dim == 0" in indexer_init
    assert (
        "weight_scale = self.index_softmax_scale * self.indexer_weight_scale"
        in score
    )
    assert "q_scale: torch.Tensor | None = None" in score
    assert score.count(
        "index_q, index_weights, weight_scale, self.padded_index_num_heads"
    ) == 2
    for stale_attribute in (
        "self.indexer_scale",
        "self.k_cache_block_size",
        "self.n_head",
        "self.rotary_emb",
        "self.cos_sin_cache",
        "self.is_neox_style",
    ):
        assert stale_attribute not in score
    assert "skip_topk_buffer_clear=self.use_fused_k_insert" in score
    assert "_mxfp4_k_quant_store" in fused
    assert "_mxfp4_k_quant_store" in unfused
    assert "if slot >= 0" in unfused
    assert "topk" not in writer


@pytest.mark.requires_runtime
def test_fp4_pcp_gathers_normalized_keys_before_mxfp4_store(monkeypatch):
    torch = pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    from vllm_sail.patch.enhancement.models import kimi_k3_pcp as patch_module

    local_k = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
    expanded_slots = torch.tensor([8, -1, 10, 20, 21, 22])
    metadata = NS(slot_mapping=expanded_slots, num_decode_tokens=1)
    cache = NS(prefix="layer.indexer", kv_cache=object())
    indexer = NS(
        use_fp4_cache=True,
        use_pcp=True,
        use_fused_k_insert=False,
        k_cache=cache,
        _parallel_config=NS(pcp_shard_decode_requests=True),
    )
    calls = {}

    def gather(k, slots, num_decode_tokens, use_pcp, pcp_shard_decode_requests):
        calls["gather"] = (
            k,
            slots,
            num_decode_tokens,
            use_pcp,
            pcp_shard_decode_requests,
        )
        return torch.cat((k, k + 100)), slots

    def store(k, slots, kv_cache):
        calls["store"] = (k, slots, kv_cache)

    monkeypatch.setattr(
        patch_module._dsa,
        "get_forward_context",
        lambda: NS(attn_metadata={"layer.indexer": metadata}),
    )
    # LayerName wraps custom-op arguments on newer torch versions, whereas the
    # metadata mapping remains keyed by the original plain string.
    monkeypatch.setattr(
        patch_module._dsa, "_encode_layer_name", lambda name: object()
    )
    monkeypatch.setattr(
        patch_module, "get_pcp_group", lambda: NS(world_size=2)
    )
    monkeypatch.setattr(patch_module, "maybe_gather_indexer_k", gather)
    monkeypatch.setattr(patch_module._mxfp4, "mxfp4_k_quant_and_cache", store)

    remaining_k, skip_insert = patch_module._pcp_mxfp4_k_cache_insert(indexer, local_k)

    assert remaining_k is None
    assert skip_insert is True
    gathered_k, gathered_slots, kv_cache = calls["store"]
    assert torch.equal(gathered_k[:3], local_k)
    assert torch.equal(gathered_k[3:], local_k + 100)
    assert gathered_slots is expanded_slots
    assert kv_cache is cache.kv_cache
    assert calls["gather"][2:] == (1, True, True)


@pytest.mark.requires_runtime
def test_fp4_pcp_dummy_run_does_not_gather_or_write(monkeypatch):
    torch = pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    from vllm_sail.patch.enhancement.models import kimi_k3_pcp as patch_module

    local_k = torch.empty(2, 4, dtype=torch.bfloat16)
    indexer = NS(use_fp4_cache=True, use_pcp=True, use_fused_k_insert=False)
    monkeypatch.setattr(
        patch_module._dsa,
        "get_forward_context",
        lambda: NS(attn_metadata=None),
    )
    monkeypatch.setattr(
        patch_module,
        "maybe_gather_indexer_k",
        lambda *args, **kwargs: pytest.fail("dummy run must not gather keys"),
    )

    remaining_k, skip_insert = patch_module._pcp_mxfp4_k_cache_insert(indexer, local_k)

    assert remaining_k is local_k
    assert skip_insert is False


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local_name,upstream,upstream_name",
    [
        (
            "_mla_build_body",
            "model_executor/layers/attention/mla_attention",
            "MLACommonMetadataBuilder.build",
        ),
        (
            "_mla_init",
            "models/kimi_k3/nvidia/mla",
            "MultiHeadLatentAttention.__init__",
        ),
        (
            "_attention_body",
            "models/kimi_k3/nvidia/mla",
            "MultiHeadLatentAttention._attention",
        ),
        (
            "_forward_prefill_fused_body",
            "models/kimi_k3/nvidia/mla",
            "MultiHeadLatentAttention._forward_prefill_fused",
        ),
        (
            "_kimi_k3_indexer_uses_fp4",
            "models/kimi_k3/common/dsa_cache",
            "kimi_k3_indexer_uses_fp4",
        ),
        (
            "_dsa_indexer_init",
            "models/kimi_k3/nvidia/dsa",
            "KimiK3DSAIndexer.__init__",
        ),
        (
            "_indexer_score_body",
            "models/kimi_k3/nvidia/dsa",
            "KimiK3DSAIndexer.score",
        ),
        (
            "_dsa_init",
            "models/kimi_k3/common/dsa_cache",
            "KimiK3IndexerMetadataBuilder.__init__",
        ),
        (
            "_dsa_build",
            "models/kimi_k3/common/dsa_cache",
            "KimiK3IndexerMetadataBuilder.build",
        ),
        (
            "_kda_init",
            "models/kimi_k3/nvidia/kda",
            "KimiK3DeltaAttention.__init__",
        ),
        (
            "_kda_forward",
            "models/kimi_k3/nvidia/kda",
            "KimiK3DeltaAttention.forward",
        ),
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        "models/kimi_k3_pcp",
        local_name,
        upstream_source_root,
        upstream,
        upstream_name,
    )
