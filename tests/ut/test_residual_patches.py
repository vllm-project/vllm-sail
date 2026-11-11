# SPDX-License-Identifier: Apache-2.0
"""Tests for the residual fork-M-file patches (``patch/enhancement/residual/``).

Two tiers, both runnable on a bare CPU runner with no vLLM:

* **Bare-CPU import + metadata shape.** The residual leaf modules define their
  replacements without importing vLLM at module scope, so they import cleanly
  here. Their mandatory ``reason`` / ``affected_versions`` / ``remove_when``
  metadata is asserted twice: from the ``METADATA`` tuples the modules expose,
  and structurally, by AST-scanning every ``patch`` / ``patch_value`` /
  ``PatchRecord`` call for the three keyword arguments.
* **Behaviour against stubs.** Each ``install()`` is exercised against a
  throwaway stand-in for its upstream module, asserting the patch lands (with
  the ``__vllm_sail_patch__`` marker), behaves as the fork intended, and is
  idempotent.

The residual *package* (``residual/__init__``) installs on import and therefore
needs vLLM; it is covered by the WIRING PENDING note, not imported here.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pytest

from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY

RESIDUAL_DIR = (
    Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement" / "residual"
)
MODULE_NAMES = (
    "all2all_deepep_shrink",
    "compilation_attention_ops",
    "gpu_worker_max_split",
    "kernel_moe_backend_literal",
    "minimax_fused_ar_rms_qk",
)
REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")

#: Loaded once per session: importing through the ``vllm_sail.patch.enhancement``
#: package chain would execute that package's vLLM-dependent ``__init__``, so the
#: leaf modules are loaded by file path instead (same mechanism as the
#: ``patch_utils_module`` conftest fixture). Caching keeps module-level state
#: (the idempotency flags) process-like across tests.
_LOADED: dict[str, types.ModuleType] = {}


def _module(name: str) -> types.ModuleType:
    if name not in _LOADED:
        path = RESIDUAL_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(
            f"_residual_leaf_test_{name}", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _LOADED[name] = module
    return _LOADED[name]


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """PATCH_REGISTRY is process-global; drop records these tests add."""
    before = len(PATCH_REGISTRY)
    yield
    del PATCH_REGISTRY[before:]


class _Platform:
    """Settable current_platform stand-in for the delegating patches."""

    ppu = True

    @classmethod
    def is_ppu(cls) -> bool:
        return cls.ppu


def _register_chain(
    monkeypatch: pytest.MonkeyPatch, leaf_name: str, leaf_module: types.ModuleType
) -> None:
    """Register ``leaf_name`` and every parent package in sys.modules."""
    parts = leaf_name.split(".")
    chain = [".".join(parts[: i + 1]) for i in range(len(parts))]
    modules: dict[str, types.ModuleType] = {}
    for name in chain[:-1]:
        existing = sys.modules.get(name)
        module = (
            existing
            if isinstance(existing, types.ModuleType)
            else types.ModuleType(name)
        )
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules[leaf_name] = leaf_module
    monkeypatch.setitem(sys.modules, leaf_name, leaf_module)
    for parent_name, child_name in zip(chain, chain[1:], strict=False):
        setattr(
            modules[parent_name],
            child_name.rpartition(".")[2],
            modules[child_name],
        )


def _stub_platform(monkeypatch: pytest.MonkeyPatch) -> type[_Platform]:
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = _Platform  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.platforms", platforms)
    return _Platform


def _stub_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.logger", logger_module)


# ---------------------------------------------------------------------------
# Bare-CPU import and metadata shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_leaf_module_imports_on_bare_cpu(name: str) -> None:
    """Residual leaf modules must not need vLLM (or side effects) to import."""
    if "vllm" in sys.modules:
        pytest.skip("bare-CPU contract test; real vLLM is loaded (mode B)")
    module = _module(name)
    assert callable(module.install)
    assert module.__doc__


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_metadata_shape(name: str) -> None:
    module = _module(name)
    metadata = module.METADATA
    assert metadata, f"{name} declares no patch metadata"
    for target, reason, affected, remove_when in metadata:
        assert target.startswith("vllm."), target
        for value in (reason, affected, remove_when):
            assert isinstance(value, str) and value.strip(), target
        assert ">=" in affected and "<" in affected, target
        assert remove_when.strip().lower() != "todo", target


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_no_top_level_vllm_import(name: str) -> None:
    """Bare-CPU importability is structural: no vllm import at module scope."""
    path = Path(_module(name).__file__)
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for imported in names:
            assert imported != "vllm" and not imported.startswith("vllm."), (
                f"{name} imports {imported!r} at module level; residual leaf "
                "modules must stay importable without vLLM."
            )


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_every_patch_call_carries_metadata(name: str) -> None:
    path = Path(_module(name).__file__)
    tree = ast.parse(path.read_text())
    watched = {"patch", "patch_value", "PatchRecord"}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _callee_name(node) in watched
    ]
    assert calls, f"{name} installs no patches"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        missing = set(REQUIRED_METADATA) - keywords
        assert not missing, f"{name}: patch call at line {call.lineno} lacks {missing}"


# ---------------------------------------------------------------------------
# Behaviour against stubs
# ---------------------------------------------------------------------------


def test_all2all_drops_enable_shrink_on_ppu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("all2all_deepep_shrink")
    platform = _stub_platform(monkeypatch)

    captured: dict[str, Any] = {}

    class DeepEPLLAll2AllManager:
        def _make_all2all_kwargs(self, num_ep_ranks: int) -> dict[str, Any]:
            captured["num_ep_ranks"] = num_ep_ranks
            return {
                "group": "stub",
                "enable_shrink": True,
                "explicitly_destroy": True,
            }

    all2all = types.ModuleType("vllm.distributed.device_communicators.all2all")
    all2all.DeepEPLLAll2AllManager = DeepEPLLAll2AllManager
    _register_chain(
        monkeypatch, "vllm.distributed.device_communicators.all2all", all2all
    )

    module.install()
    manager = DeepEPLLAll2AllManager()

    platform.ppu = True
    kwargs = manager._make_all2all_kwargs(8)
    assert "enable_shrink" not in kwargs
    assert kwargs["explicitly_destroy"] is True
    assert captured["num_ep_ranks"] == 8

    platform.ppu = False
    assert "enable_shrink" in manager._make_all2all_kwargs(8)

    replaced = inspect.getattr_static(DeepEPLLAll2AllManager, "_make_all2all_kwargs")
    assert getattr(replaced, PATCH_MARKER, None) is not None
    module.install()  # idempotent: the guard, not the double-patch error


def test_gpu_worker_raises_allocator_floor_on_ppu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("gpu_worker_max_split")
    platform = _stub_platform(monkeypatch)

    seen: list[int] = []

    class Worker:
        @contextlib.contextmanager
        def _scoped_allocator_max_split(self, max_split_size_mb: int):
            seen.append(max_split_size_mb)
            yield

    gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker.Worker = Worker  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.v1.worker.gpu_worker", gpu_worker)

    module.install()
    worker = Worker()

    platform.ppu = True
    with worker._scoped_allocator_max_split(20):
        pass
    assert seen == [module.PPU_MIN_MAX_SPLIT_MB]
    with worker._scoped_allocator_max_split(64):
        pass
    assert seen[-1] == 64

    platform.ppu = False
    with worker._scoped_allocator_max_split(20):
        pass
    assert seen[-1] == 20

    replaced = inspect.getattr_static(Worker, "_scoped_allocator_max_split")
    assert getattr(replaced, PATCH_MARKER, None) is not None


def test_compilation_registers_ppu_indexer_splitting_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("compilation_attention_ops")

    class CompilationConfig:
        _attention_ops = [
            "vllm::unified_attention_with_output",
            "vllm::sparse_attn_indexer",
        ]

        def set_splitting_ops_for_v1(self, all2all_backend, data_parallel_size=1):
            self.splitting_ops = list(type(self)._attention_ops)

    compilation = types.ModuleType("vllm.config.compilation")
    compilation.CompilationConfig = CompilationConfig  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.config.compilation", compilation)

    module.install()
    config = CompilationConfig()
    config.set_splitting_ops_for_v1("deepep_low_latency")
    assert "vllm::ppu_sparse_attn_indexer" in config.splitting_ops
    assert "vllm::sparse_attn_indexer" in config.splitting_ops

    # Idempotent: a second config resolution must not duplicate the entry.
    config.set_splitting_ops_for_v1("deepep_low_latency")
    assert CompilationConfig._attention_ops.count("vllm::ppu_sparse_attn_indexer") == 1


def test_kernel_literal_widening(monkeypatch: pytest.MonkeyPatch) -> None:
    import pydantic
    from pydantic.dataclasses import dataclass

    module = _module("kernel_moe_backend_literal")

    MoEBackend = Literal["auto", "triton", "deep_gemm"]

    @dataclass(config=pydantic.ConfigDict(extra="forbid"))
    class KernelConfig:
        moe_backend: MoEBackend = "auto"

    kernel = types.ModuleType("vllm.config.kernel")
    kernel.MoEBackend = MoEBackend  # type: ignore[attr-defined]
    kernel.KernelConfig = KernelConfig  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.config.kernel", kernel)
    _stub_logger(monkeypatch)

    with pytest.raises(pydantic.ValidationError):
        KernelConfig(moe_backend="ppu_deep_gemm")

    module.install()

    assert KernelConfig(moe_backend="ppu_deep_gemm").moe_backend == "ppu_deep_gemm"
    assert KernelConfig(moe_backend="ppu_acext").moe_backend == "ppu_acext"
    assert KernelConfig(moe_backend="auto").moe_backend == "auto"
    with pytest.raises(pydantic.ValidationError):
        KernelConfig(moe_backend="bogus")
    from typing import get_args

    assert set(module.PPU_MOE_BACKENDS) <= set(get_args(kernel.MoEBackend))
    assert any(
        record.target == "vllm.config.kernel.MoEBackend" for record in PATCH_REGISTRY
    )

    before = len(PATCH_REGISTRY)
    module.install()  # idempotent, no duplicate record
    assert len(PATCH_REGISTRY) == before


def test_minimax_fused_kernel_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module("minimax_fused_ar_rms_qk")

    rms_norm_tp = types.ModuleType(
        "vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp"
    )
    sentinel = object()
    rms_norm_tp._MINIMAX_FUSED_AR_RMS_QK = sentinel  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp",
        rms_norm_tp,
    )
    _stub_logger(monkeypatch)

    module.install()
    assert rms_norm_tp._MINIMAX_FUSED_AR_RMS_QK is None
    assert any(
        record.target.endswith("rms_norm_tp._MINIMAX_FUSED_AR_RMS_QK")
        for record in PATCH_REGISTRY
    )

    before = len(PATCH_REGISTRY)
    module.install()  # idempotent, no duplicate record
    assert len(PATCH_REGISTRY) == before
