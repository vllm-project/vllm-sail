# SPDX-License-Identifier: Apache-2.0
"""FlashMLA patches cover upstream consumers that import operations by value."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.support.modules import load_patch
from tests.support.source import (
    ROOT,
    assert_accepts_upstream_keywords,
    definition,
    function,
)


@pytest.mark.parametrize(
    "patch_module,upstream_module",
    (
        ("flashmla_backend", "vllm.v1.attention.backends.mla.flashmla"),
        ("flashmla_ops", "vllm.v1.attention.ops.flashmla"),
    ),
)
def test_flashmla_expected_patch_targets_match_declarations(
    patch_module, upstream_module
):
    """Catch stale runtime-test expectations even without torch or vLLM."""
    path = ROOT / "tests/ut/test_attention_patches.py"
    assignment = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_ATTENTION_TARGETS"
    )
    expected = {
        attr
        for module, attr in ast.literal_eval(assignment.value)
        if module == upstream_module
    }
    path = ROOT / f"vllm_sail/patch/enhancement/attention/{patch_module}.py"
    declared = {
        ast.literal_eval(decorator.args[1])
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "patch"
    }
    assert expected == declared


@pytest.mark.upstream_source
def test_flashmla_consumer_inventory_against_source_without_vllm_imports(
    upstream_source_root,
):
    """Keep preloaded consumer coverage checkable on the dependency-free runner."""
    path = ROOT / "vllm_sail/patch/enhancement/attention/flashmla_ops.py"
    assignment = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FLASHMLA_ALIAS_CONSUMERS"
    )
    expected = {
        name: set(consumers)
        for name, consumers in ast.literal_eval(assignment.value).items()
    }
    discovered = {name: set() for name in expected}
    source = upstream_source_root / "vllm"
    for path in source.rglob("*.py"):
        consumer = ".".join(("vllm", *path.relative_to(source).with_suffix("").parts))
        for node in ast.parse(path.read_text()).body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "vllm.v1.attention.ops.flashmla"
            ):
                for alias in node.names:
                    if alias.name in discovered:
                        discovered[alias.name].add(consumer)
    assert discovered == expected


_SPARSE_PATCH = "vllm_sail/patch/enhancement/attention/flashmla_sparse_backend.py"


def test_sparse_backend_import_registers_fp8_patch(modules, patch_utils_module):
    """Exercise decorators too: extracting function bodies misses startup errors."""

    class Backend:
        @classmethod
        def supports_compute_capability(cls, capability):
            return True

        @classmethod
        def supports_combination(cls, *args):
            return None

    class Builder:
        def __init__(self, *args):
            pass

    class Impl:
        def _fp8_flash_mla_kernel(self, *args, **kwargs):
            pass

    original = Impl._fp8_flash_mla_kernel
    modules("torch")
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    modules("vllm.platforms.interface", DeviceCapability=SimpleNamespace)
    modules("vllm.utils.platform_utils", num_compute_units=lambda index: 1)
    backend = modules(
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        FlashMLASparseBackend=Backend,
        FlashMLASparseMetadataBuilder=Builder,
        FlashMLASparseImpl=Impl,
    )
    plugin = load_patch(_SPARSE_PATCH)
    target = f"{backend.__name__}.FlashMLASparseImpl._fp8_flash_mla_kernel"
    replacement = Impl._fp8_flash_mla_kernel
    assert replacement is plugin.fp8_flash_mla_kernel
    assert patch_utils_module.original_of(replacement, target) is original
    assert target in {record.target for record in patch_utils_module.PATCH_REGISTRY}


@pytest.mark.parametrize("heads", [1, 8, 16, 32, 48, 64, 96, 128])
@pytest.mark.parametrize("query_len", [1, 4, 8])
def test_ppu_sparse_fp8_decode_forwards_native_query_without_padding(heads, query_len):
    # Deliberately expose no allocation or slicing methods: either operation
    # would reintroduce the GPU head-padding contract on the PPU path.
    query = SimpleNamespace(shape=(2, query_len, heads, 576))
    output = SimpleNamespace(shape=(2, query_len, heads, 512))
    lse = SimpleNamespace(shape=(2, heads, query_len))
    result = (output, lse)
    provider = Mock(return_value=result)
    upstream = Mock(side_effect=AssertionError("PPU must bypass GPU head padding"))
    uint8 = object()
    cache_view = object()
    cache = Mock()
    cache.view.return_value.unsqueeze.return_value = cache_view
    metadata = SimpleNamespace(
        dummy_block_table=object(), cache_lens=object(), scheduler_metadata=object()
    )
    indices = object()
    # A stale/local envelope must not replace the runtime (possibly
    # DCP-gathered) query head count. No per-layer state is mutated in forward.
    impl = SimpleNamespace(softmax_scale=0.125, fp8_decode_padded_heads=64)
    state = vars(impl).copy()
    forward = function(
        _SPARSE_PATCH,
        "fp8_flash_mla_kernel",
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: True),
            "torch": SimpleNamespace(uint8=uint8),
            "_flashmla_sparse_backend": SimpleNamespace(
                flash_mla_with_kvcache=provider
            ),
            "_upstream_fp8_flash_mla_kernel": upstream,
        },
    )
    assert forward(impl, query, cache, indices, metadata) is result
    assert vars(impl) == state
    cache.view.assert_called_once_with(uint8)
    cache.view.return_value.unsqueeze.assert_called_once_with(-2)
    provider.assert_called_once_with(
        q=query,
        k_cache=cache_view,
        block_table=metadata.dummy_block_table,
        head_dim_v=512,
        cache_seqlens=metadata.cache_lens,
        tile_scheduler_metadata=metadata.scheduler_metadata,
        is_fp8_kvcache=True,
        indices=indices,
        softmax_scale=0.125,
        topk_length=None,
    )
    upstream.assert_not_called()


@pytest.mark.parametrize("has_topk_length", [False, True])
def test_non_ppu_sparse_fp8_decode_delegates_unchanged(has_topk_length):
    result = object()
    upstream = Mock(return_value=result)
    forward = function(
        _SPARSE_PATCH,
        "fp8_flash_mla_kernel",
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: False),
            "_upstream_fp8_flash_mla_kernel": upstream,
        },
    )
    args = tuple(object() for _ in range(5))
    lengths = object() if has_topk_length else None
    assert forward(*args, topk_length=lengths) is result
    upstream.assert_called_once_with(*args, topk_length=lengths)


@pytest.mark.upstream_source
def test_sparse_fp8_decode_patch_preserves_upstream_call_contract(upstream_source_root):
    assert_accepts_upstream_keywords(
        "attention/flashmla_sparse_backend",
        "fp8_flash_mla_kernel",
        upstream_source_root,
        "v1/attention/backends/mla/flashmla_sparse",
        "FlashMLASparseImpl._fp8_flash_mla_kernel",
    )
    upstream = definition(
        upstream_source_root / "vllm/v1/attention/backends/mla/flashmla_sparse.py",
        "FlashMLASparseImpl._fp8_flash_mla_kernel",
    )
    replacement = definition(ROOT / _SPARSE_PATCH, "fp8_flash_mla_kernel")
    original_call = next(
        node
        for node in ast.walk(upstream)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "flash_mla_with_kvcache"
    )
    replacement_call = next(
        node
        for node in ast.walk(replacement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "flash_mla_with_kvcache"
    )
    # Track all kernel arguments, not just the wrapper signature: an upstream
    # cache format, scale or metadata contract change needs an explicit review.
    assert not original_call.args and not replacement_call.args
    # PPU resolves both providers on the backend module so lazy op rebinding
    # remains effective. Ignore only that namespace qualification.
    assert ast.unparse(replacement_call).replace(
        "_flashmla_sparse_backend.", ""
    ) == ast.unparse(original_call)


def test_ppu_sparse_fp8_decode_refreshes_length_dependent_schedule():
    schedules = [object(), object()]
    get_metadata = Mock(side_effect=[(schedule, None) for schedule in schedules])
    provider = Mock(return_value=(object(), object()))
    forward = function(
        _SPARSE_PATCH,
        "fp8_flash_mla_kernel",
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: True),
            "torch": SimpleNamespace(uint8=object()),
            "_flashmla_sparse_backend": SimpleNamespace(
                flash_mla_with_kvcache=provider, get_mla_metadata=get_metadata
            ),
        },
    )
    shared_schedule = object()
    metadata = SimpleNamespace(
        dummy_block_table=object(),
        cache_lens=object(),
        scheduler_metadata=shared_schedule,
    )
    impl = SimpleNamespace(softmax_scale=0.125)
    query, cache, indices = object(), Mock(), object()
    for schedule in schedules:
        lengths = object()
        forward(impl, query, cache, indices, metadata, topk_length=lengths)
        kwargs = provider.call_args.kwargs
        assert kwargs["q"] is query
        assert kwargs["topk_length"] is lengths
        assert kwargs["tile_scheduler_metadata"] is schedule
    assert get_metadata.call_count == 2
    assert provider.call_count == 2
    assert metadata.scheduler_metadata is shared_schedule
