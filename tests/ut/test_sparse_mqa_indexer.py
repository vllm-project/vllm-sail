# SPDX-License-Identifier: Apache-2.0
"""Sparse-indexer integration contracts, runnable without torch or vLLM."""

import inspect
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/attention/sparse_mqa_indexer.py"
ADAPTER = "vllm_sail/attention/ops/sparse_mqa.py"
KERNEL = "vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits"
LAYER = "vllm.model_executor.layers.sparse_mqa_indexer"
BUILDER = "vllm.v1.attention.backends.mla.sparse_indexer"


class Tensor:
    def __init__(self, shape, dtype="int32", *, contiguous=False, page_stride=512):
        self.shape, self.dtype = shape, dtype
        self.packed, self.page_stride = contiguous, page_stride

    def contiguous(self):
        return self if self.packed else Tensor(self.shape, self.dtype, contiguous=True)

    def stride(self, dim):
        return 1 if dim == -1 else self.page_stride

    def is_contiguous(self):
        return self.packed


def config():
    return NS(
        model_config=NS(
            hf_text_config=NS(
                model_type="deepseek_v41_text",
                index_n_heads=32,
                index_head_dim=128,
                candidate_block_size=16,
                candidate_topk_blocks=32,
                index_topk=32,
            )
        ),
        kernel_config=NS(sparse_indexer_topk_backend="auto"),
    )


@pytest.mark.parametrize("missing", ["deep_gemm", "sail_runtime"])
def test_sparse_backend_distinguishes_missing_wheel_from_broken_dependency(missing):
    def import_module(name):
        raise ModuleNotFoundError(name=missing)

    load = function(ADAPTER, "_backend", {"importlib": NS(import_module=import_module)})
    if missing == "deep_gemm":
        assert load() is None
    else:
        with pytest.raises(ModuleNotFoundError):
            load()


@pytest.fixture
def adapter(modules, monkeypatch):
    state = NS(ppu=True, capability=89, gemm=True, select=True, calls=[])
    state.platform = NS(
        is_ppu=lambda: state.ppu,
        is_device_capability=lambda cap: state.capability == cap,
        is_cuda=lambda: True,
        is_device_capability_family=lambda cap: state.capability == cap,
    )
    state.torch = modules(
        "torch",
        Tensor=Tensor,
        dtype=object,
        bfloat16="bf16",
        float32="fp32",
        int32="int32",
        int8="int8",
        uint8="uint8",
        zeros=lambda shape, **kw: Tensor(shape, kw["dtype"]),
    )
    modules("vllm.platforms", current_platform=state.platform)
    modules("vllm_sail.utils.deep_gemm", is_deep_gemm_supported=lambda: state.gemm)
    modules(
        "vllm_sail.utils.deep_select", is_deep_select_available=lambda: state.select
    )
    state.mod = load_patch(ADAPTER)

    def entry(name):
        def call(*args, **kwargs):
            state.calls.append((name, args, kwargs))
            return name

        return call

    state.backend = NS(**{name: entry(name) for name in state.mod._APIS})
    monkeypatch.setattr(state.mod, "_backend", lambda: state.backend)
    return state


@pytest.mark.parametrize(
    "missing", [None, "metadata", "compute", "device", "gemm", "select"]
)
def test_sparse_support_requires_the_complete_sail_stack(adapter, missing):
    state = adapter
    if missing == "metadata":
        state.backend.get_paged_sparse_mqa_logits_metadata = None
    elif missing == "compute":
        state.backend.fp8_fp4_sparse_mqa_logits = None
    elif missing == "device":
        state.capability = 80
    elif missing == "gemm":
        state.gemm = False
    elif missing == "select":
        state.select = False
    if missing is None:
        state.mod.check_sparse_indexer_support(config())
    else:
        with pytest.raises(ValueError, match="requires"):
            state.mod.check_sparse_indexer_support(config())


@pytest.mark.parametrize(
    "attr,value",
    [
        ("model_type", "deepseek_v4"),
        ("index_n_heads", 64),
        ("index_head_dim", 64),
        ("candidate_block_size", 64),
        ("candidate_topk_blocks", 3),
        ("candidate_topk_blocks", 2),
        ("candidate_topk_blocks", 8192),
        ("index_topk", 0),
        ("index_topk", 513),
    ],
)
def test_sparse_support_rejects_unsupported_geometry(adapter, attr, value):
    cfg = config()
    setattr(cfg.model_config.hf_text_config, attr, value)
    with pytest.raises(ValueError):
        adapter.mod.check_sparse_indexer_support(cfg)


def test_prefill_uses_the_same_unaligned_variant_for_schedule_and_compute(adapter):
    m = adapter.mod
    ks, ke, si = Tensor((3,)), Tensor((3,)), Tensor((3, 32))
    schedule = m.get_sparse_mqa_logits_metadata(ks, ke, 103, si, "int8", 16)
    q = (Tensor((3, 32, 64), "int8"), Tensor((3, 32)))
    kv = (Tensor((103, 64), "int8"), Tensor((103,)))
    weights = Tensor((3, 32), "bf16")
    m.fp8_fp4_sparse_mqa_logits(q, kv, weights, schedule, 32, 16)
    meta, compute = adapter.calls
    assert meta[2] == compute[2] == {"use_unaligned_ks": True}
    assert all(t.packed for t in (meta[1][0], meta[1][1], meta[1][3]))
    assert meta[1][2] == 103
    assert compute[1][3:] == (schedule, 32, 16)
    assert all(t.packed for pair in compute[1][:2] for t in pair)
    assert compute[1][2].packed and compute[1][2].dtype == "bf16"


def test_paged_contract_preserves_request_map_and_physical_cache(adapter):
    m = adapter.mod
    contexts, table, ids, si = (
        Tensor((3,)),
        Tensor((3, 7)),
        Tensor((3,)),
        Tensor((3, 32)),
    )
    schedule = m.get_paged_sparse_mqa_logits_metadata(
        contexts, table, ids, 64, si, "int8", 16
    )
    q = (Tensor((3, 1, 32, 64), "int8"), Tensor((3, 1, 32)))
    cache = Tensor((20, 64, 1, 68), "uint8", page_stride=4608)
    m.fp8_fp4_paged_sparse_mqa_logits(
        q, cache, Tensor((3, 32), "bf16"), schedule, 32, 16
    )
    meta, compute = adapter.calls
    assert meta[1][2].shape == ids.shape
    assert all(meta[1][i].packed for i in (0, 1, 2, 4))
    assert compute[1][1] is cache
    assert compute[1][3:] == (schedule, 32, 16)
    assert meta[2] == compute[2] == {}
    cache.page_stride = 256
    with pytest.raises(ValueError, match="512-byte"):
        m.fp8_fp4_paged_sparse_mqa_logits(
            q, cache, Tensor((3, 32), "bf16"), schedule, 32, 16
        )


@pytest.mark.parametrize(
    "shape,dtype",
    [((3, 2, 32, 64), "int8"), ((3, 1, 64, 64), "int8"), ((3, 1, 32, 64), "fp8")],
)
def test_paged_query_rejects_mtp_axis_or_wrong_format(adapter, shape, dtype):
    with pytest.raises(ValueError, match="packed MXFP4"):
        adapter.mod._query(
            (Tensor(shape, dtype), Tensor(shape[:-1])),
            Tensor((3, 32), "bf16"),
            paged=True,
        )


def test_sparse_weights_are_never_silently_rounded(adapter):
    with pytest.raises(ValueError, match="producer-owned BF16"):
        adapter.mod._query(
            (Tensor((3, 32, 64), "int8"), Tensor((3, 32))),
            Tensor((3, 32), "fp32"),
            paged=False,
        )


@pytest.fixture
def installed(adapter, modules, patch_utils_module):
    s = adapter
    s.config, s.cp, s.varlen, s.fp4 = config(), False, True, True
    modules("vllm.config", get_current_vllm_config=lambda: s.config)
    modules("vllm_sail.attention.ops", sparse_mqa=s.mod)
    modules("vllm_sail.attention.ops.sparse_mqa", **vars(s.mod))

    class Module:
        def __init__(self):
            self.initialized = True

    class Layer(Module):
        weights_dtype = "bf16"

        def __init__(self, *a, **kw):
            raise AssertionError("unpatched CUDA constructor")

        def forward(self, *a):
            return "upstream-forward"

    class DenseBuilder:
        def __init__(self, *a, **kw):
            self.dcp_world_size, self.use_pcp = 1, s.cp
            self.supports_varlen = s.varlen
            self.arange_buffer = Tensor((64,))

    class Builder(DenseBuilder):
        def __init__(self, *a, **kw):
            raise AssertionError("unpatched CUDA builder")

        def build(self, *a):
            return "upstream-build"

    common = dict(
        current_platform=s.platform,
        has_deep_select=lambda: s.select,
        has_deep_gemm_sparse_mqa=lambda: "cuda-capability",
    )
    layer_module = modules(LAYER, SparseMQAIndexer=Layer, **common)
    builder_module = modules(
        BUILDER,
        DeepseekV41SparseIndexerMetadataBuilder=Builder,
        torch=s.torch,
        dsa_indexer_uses_fp4=lambda config: s.fp4,
        pick_sparse_block_kv=lambda block: 16,
        check_deep_select_layout=lambda cols, topk: None,
        logger=NS(info_once=lambda *a: None),
        **common,
    )
    kernel_module = modules(
        KERNEL, **{name: (lambda *a, **kw: "cuda") for name in s.mod._APIS}
    )
    originals = {name: getattr(kernel_module, name) for name in s.mod._APIS}
    originals["layer"] = Layer.__init__
    originals["builder"] = Builder.__init__
    originals["capability"] = common["has_deep_gemm_sparse_mqa"]
    forward, build = Layer.forward, Builder.build
    s.patch = load_patch(PATCH)
    assert Layer.forward is forward and Builder.build is build
    assert Layer.__init__.__globals__ is vars(layer_module)
    assert Builder.__init__.__globals__ is vars(builder_module)
    for target, fn, original in (
        (LAYER + ".SparseMQAIndexer.__init__", Layer.__init__, originals["layer"]),
        (
            BUILDER + ".DeepseekV41SparseIndexerMetadataBuilder.__init__",
            Builder.__init__,
            originals["builder"],
        ),
        *(
            (KERNEL + "." + name, getattr(kernel_module, name), originals[name])
            for name in s.mod._APIS
        ),
        (
            LAYER + ".has_deep_gemm_sparse_mqa",
            layer_module.has_deep_gemm_sparse_mqa,
            originals["capability"],
        ),
        (
            BUILDER + ".has_deep_gemm_sparse_mqa",
            builder_module.has_deep_gemm_sparse_mqa,
            originals["capability"],
        ),
    ):
        assert patch_utils_module.original_of(fn, target) is original
        module_name = next(
            m for m in (LAYER, BUILDER, KERNEL) if target.startswith(m + ".")
        )
        with pytest.raises(RuntimeError, match="already"):
            patch_utils_module.patch(
                module_name,
                target[len(module_name) + 1 :],
                reason="Reject duplicate sparse-indexer installation.",
                affected_versions=">=0.33.0,<0.34.0",
                remove_when="The platform exposes a sparse-indexer provider hook.",
            )(fn)
    with pytest.raises(RuntimeError, match="already"):
        s.patch._install()
    s.Layer, s.Builder, s.kernel = Layer, Builder, kernel_module
    return s


def test_sparse_constructor_and_builder_keep_upstream_state(installed):
    s = installed
    cache, output, candidates = object(), object(), object()
    layer = s.Layer(cache, 32, 128, 4096, output, candidates, 16)
    assert layer.initialized and layer.weights_dtype == "bf16"
    assert layer.k_cache is cache and layer.topk_indices_buffer is output
    assert layer.candidate_blocks is candidates
    builder = s.Builder(NS(block_size=64), ["indexer"], s.config, "ppu")
    assert builder.sparse_block_kv == 16 and builder.num_sparse_blocks == 32
    assert builder.min_split_seq_len == 256  # BF16 bytes expressed as FP32 columns.
    assert builder.sparse_indices_buffer.shape == (64, 32)
    assert builder.col_indices_buffer.shape == (64, 32)
    s.config.model_config.hf_text_config.candidate_topk_blocks = 16
    narrow = s.Builder(NS(block_size=64), ["indexer"], s.config, "ppu")
    assert narrow.num_sparse_blocks == 16
    assert narrow.min_split_seq_len == 256  # Physical stride is still 1024B.
    s.ppu = False
    assert s.kernel.fp8_fp4_sparse_mqa_logits(None) == "cuda"
    with pytest.raises(ValueError, match="SM100"):
        s.Layer(cache, 32, 128, 4096, output, candidates, 16)
    s.capability = 100
    assert s.Layer(cache, 32, 128, 4096, output, candidates, 16).initialized
    cuda = s.Builder(NS(block_size=64), ["indexer"], s.config, "cuda")
    assert cuda.min_split_seq_len == 128  # Keep upstream's CUDA budget.


@pytest.mark.parametrize("restriction", ["fp4", "cp", "varlen", "backend"])
def test_sparse_builder_preserves_upstream_restrictions(installed, restriction):
    s = installed
    if restriction == "backend":
        s.config.kernel_config.sparse_indexer_topk_backend = "torch"
    else:
        setattr(s, restriction, restriction == "cp")
    with pytest.raises((ValueError, NotImplementedError)):
        s.Builder(NS(block_size=64), [], s.config, "ppu")


@pytest.mark.upstream_source
def test_sparse_api_signatures_match_the_v033_pin(upstream_source_root, adapter):
    for name in adapter.mod._APIS:
        upstream = function(upstream_source_root / "vllm/utils/deep_gemm.py", name, {})
        assert (
            inspect.signature(getattr(adapter.mod, name)).parameters.keys()
            == inspect.signature(upstream).parameters.keys()
        )


@pytest.mark.upstream_source
@pytest.mark.parametrize("paged", [False, True])
def test_upstream_sparse_pipeline_reuses_schedule_within_a_step(
    upstream_source_root, adapter, paged
):
    """Exercise upstream orchestration with the real SAIL argument adapters."""
    s = adapter
    calls = []
    buffers = dict(
        sparse_indices=Tensor((3, 32)), end=Tensor((3,)), col_indices=Tensor((3, 32))
    )
    namespace = {
        "torch": s.torch,
        "candidate_blocks_to_sparse_indices": lambda *a, **kw: calls.append(
            ("expand", a, kw)
        ),
        "sparse_topk_remap": lambda *a, **kw: calls.append(("topk", a, kw)),
        **{name: getattr(s.mod, name) for name in s.mod._APIS},
    }
    path = (
        upstream_source_root
        / "vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py"
    )
    if paged:
        fn = function(path, "sparse_mqa_logits_paged_decode", namespace)
        args = (
            Tensor((3, 1, 32, 64), "int8"),
            Tensor((3, 1, 32)),
            Tensor((20, 64, 1, 68), "uint8", page_stride=4608),
            Tensor((3, 32), "bf16"),
            Tensor((3,), contiguous=True),
            Tensor((3, 7)),
            Tensor((3,)),
            Tensor((3, 32)),
            16,
            16,
            32,
            object(),
        )
        buffers["row_ks"] = Tensor((3,))
    else:
        fn = function(path, "sparse_mqa_logits_prefill_chunk", namespace)
        args = (
            Tensor((3, 32, 64), "int8"),
            Tensor((3, 32)),
            Tensor((103, 64), "int8"),
            Tensor((103,)),
            Tensor((3, 32), "bf16"),
            Tensor((3,)),
            Tensor((3,)),
            Tensor((3, 32)),
            16,
            16,
            32,
            object(),
        )
    metadata = fn(*args, **buffers)
    assert fn(*args, **buffers, kernel_metadata=metadata) == metadata
    assert [name for name, *_ in calls] == ["expand", "topk", "topk"]
    assert calls[0][2]["out"] == (buffers["sparse_indices"], buffers["end"])
    assert calls[1][1][1] is buffers["sparse_indices"]
    assert calls[1][1][2] is buffers["end"]
    assert calls[1][1][-1] is args[-1]  # Caller-owned output survives.
    assert calls[1][2]["col_indices"] is buffers["col_indices"]
    assert len(s.calls) == 3  # one schedule, two scoring layers
    fn(*args, **buffers)  # New step starts without a cached schedule.
    assert len(s.calls) == 5
    assert [name for name, *_ in calls].count("expand") == 2


def test_paged_adapter_keeps_ragged_request_ids_and_weights_exact():
    torch = pytest.importorskip("torch")
    calls = []
    backend = NS(
        get_paged_sparse_mqa_logits_metadata=lambda *a: calls.append(a) or "schedule",
        fp8_fp4_paged_sparse_mqa_logits=lambda *a: calls.append(a) or "logits",
    )
    namespace = {"torch": torch, "_backend": lambda: backend}
    function(ADAPTER, "_query", namespace)
    metadata = function(ADAPTER, "get_paged_sparse_mqa_logits_metadata", namespace)
    compute = function(ADAPTER, "fp8_fp4_paged_sparse_mqa_logits", namespace)
    ids = torch.tensor([0, 0, 1, 3], dtype=torch.int32)
    contexts = torch.tensor([11, 12, 21, 0], dtype=torch.int32)
    table = torch.arange(32, dtype=torch.int32).reshape(4, 8)[:, :3]
    sparse = torch.zeros((4, 32), dtype=torch.int32)
    schedule = metadata(contexts, table, ids, 64, sparse, torch.int8, 16)
    assert calls[0][2] is ids
    assert torch.equal(calls[0][1], table) and calls[0][1].is_contiguous()
    q = torch.zeros((4, 1, 32, 64), dtype=torch.int8)
    qs = torch.zeros((4, 1, 32), dtype=torch.int32)
    weights = torch.arange(256, dtype=torch.bfloat16).reshape(4, 64)[:, ::2]
    cache = torch.empty_strided((2, 64, 1, 68), (4608, 68, 68, 1), dtype=torch.uint8)
    compute((q, qs), cache, weights, schedule, 32, 16)
    assert calls[1][1] is cache
    assert torch.equal(calls[1][2], weights)
    assert calls[1][2].dtype == torch.bfloat16 and calls[1][2].is_contiguous()
