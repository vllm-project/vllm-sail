# SPDX-License-Identifier: Apache-2.0
"""Regression tests for vLLM MoE oracle aliases captured before registration."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest


def test_register_rebinds_preloaded_consumers_once_without_loading_others(
    monkeypatch, patch_utils_module
):
    from vllm_sail.registry import moe_backends

    monkeypatch.setattr(moe_backends, "_registered", False)
    monkeypatch.setitem(sys.modules, "vllm_sail.patch.utils", patch_utils_module)
    backend_names = ("unquantized", "fp8", "int8", "mxfp4", "int_wna16")
    imports = []
    replacements = []
    expected_targets = []

    def function():
        return lambda: None

    for backend, aliases in moe_backends._ORACLE_ALIAS_CONSUMERS.items():
        oracle_name = f"vllm.model_executor.layers.fused_moe.oracle.{backend}"
        oracle = ModuleType(oracle_name)
        monkeypatch.setitem(sys.modules, oracle_name, oracle)
        for alias, consumers in aliases.items():
            original, replacement = function(), function()
            setattr(
                replacement,
                patch_utils_module.PATCH_MARKER,
                {f"{oracle_name}.{alias}": original},
            )
            setattr(oracle, alias, replacement)
            for name in consumers:
                if name not in sys.modules:
                    monkeypatch.setitem(sys.modules, name, ModuleType(name))
                consumer = sys.modules[name]
                monkeypatch.setattr(consumer, alias, original, raising=False)
                replacements.append((consumer, alias, replacement))
                expected_targets.append(f"{name}.{alias}")

    original_import = importlib.import_module

    def import_module(name, package=None):
        if name.startswith(f"{moe_backends.__name__}."):
            imports.append(name.rsplit(".", 1)[1])
            return ModuleType(name)
        if name.startswith("vllm."):
            # Every backend must load before any consumer can be rebound.
            assert tuple(imports) == backend_names
            return sys.modules[name]
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    moe_backends.register()
    moe_backends.register()

    assert tuple(imports) == backend_names
    assert [
        record.target for record in patch_utils_module.PATCH_REGISTRY
    ] == expected_targets
    for consumer, alias, replacement in replacements:
        assert getattr(consumer, alias) is replacement


def _module_nodes(tree):
    """Only imports executed in module scope create consumer aliases."""
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


@pytest.mark.parametrize(
    ("consumer_name", "alias_name"),
    [
        (
            "vllm.model_executor.layers.quantization.mxfp4",
            "backend_to_kernel_cls",
        ),
        (
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4",
            "convert_weight_to_mxfp4_moe_kernel_format",
        ),
    ],
)
def test_new_mxfp4_consumers_call_replacement_after_registration(
    monkeypatch, patch_utils_module, consumer_name, alias_name
):
    from vllm_sail.registry import moe_backends

    aliases = moe_backends._ORACLE_ALIAS_CONSUMERS["mxfp4"]
    monkeypatch.setattr(moe_backends, "_ORACLE_ALIAS_CONSUMERS", {"mxfp4": aliases})
    monkeypatch.setitem(sys.modules, "vllm_sail.patch.utils", patch_utils_module)
    for consumers in aliases.values():
        for name in consumers:
            monkeypatch.delitem(sys.modules, name, raising=False)

    oracle_name = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
    oracle = ModuleType(oracle_name)
    consumer = ModuleType(consumer_name)
    for name in aliases:

        def original():
            return "upstream"

        def replacement():
            return "ppu"

        setattr(
            replacement,
            patch_utils_module.PATCH_MARKER,
            {f"{oracle_name}.{name}": original},
        )
        setattr(oracle, name, replacement)
        setattr(consumer, name, original)
    monkeypatch.setitem(sys.modules, oracle_name, oracle)
    monkeypatch.setitem(sys.modules, consumer_name, consumer)

    moe_backends._rebind_loaded_oracle_aliases({"mxfp4": ModuleType("backend")})

    assert getattr(consumer, alias_name)() == "ppu"
    assert f"{consumer_name}.{alias_name}" in {
        record.target for record in patch_utils_module.PATCH_REGISTRY
    }


@pytest.mark.upstream_source
def test_consumer_inventory_matches_vllm_source(upstream_source_root: Path) -> None:
    """Discover oracle imports independently so a new consumer cannot hide."""
    from vllm_sail.registry.moe_backends import _ORACLE_ALIAS_CONSUMERS

    vllm_root = upstream_source_root / "vllm"
    source_root = vllm_root / "model_executor"
    expected: dict[tuple[str, str], set[str]] = {}
    for backend_name, aliases in _ORACLE_ALIAS_CONSUMERS.items():
        oracle_name = f"vllm.model_executor.layers.fused_moe.oracle.{backend_name}"
        for alias_name, consumer_names in aliases.items():
            expected[(oracle_name, alias_name)] = set(consumer_names)

    discovered = {key: set() for key in expected}
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        consumer_name = ".".join(
            ("vllm", *path.relative_to(vllm_root).with_suffix("").parts)
        )
        for node in _module_nodes(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            for imported in node.names:
                key = (node.module, imported.name)
                if key in discovered:
                    discovered[key].add(consumer_name)

    assert discovered == expected


def test_register_updates_real_vllm_moe_consumer_aliases() -> None:
    """Consumers imported first must use every oracle function patched by PPU."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    script = textwrap.dedent(
        """
        import ast
        import importlib
        from pathlib import Path

        import vllm

        def _module_nodes(tree):
            # Only imports in module scope create consumer aliases.
            pending = list(tree.body)
            while pending:
                node = pending.pop()
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                yield node
                pending.extend(ast.iter_child_nodes(node))


        from vllm_sail.registry.moe_backends import _ORACLE_ALIAS_CONSUMERS

        aliases = []
        for backend_name, backend_aliases in _ORACLE_ALIAS_CONSUMERS.items():
            oracle_name = (
                "vllm.model_executor.layers.fused_moe.oracle."
                f"{backend_name}"
            )
            for alias_name, consumer_names in backend_aliases.items():
                aliases.extend(
                    (consumer_name, oracle_name, alias_name)
                    for consumer_name in consumer_names
                )

        consumers = {
            module_name: importlib.import_module(module_name)
            for module_name, _, _ in aliases
        }
        originals = {
            (module_name, alias_name): getattr(consumers[module_name], alias_name)
            for module_name, _, alias_name in aliases
        }

        import vllm_sail.patch
        from vllm_sail.registry import moe_backends

        vllm_sail.patch.install()
        moe_backends.register()

        # Independently derive the patched oracle surface from the live patch
        # registry. This catches a newly patched function whose by-value vLLM
        # imports were omitted from _ORACLE_ALIAS_CONSUMERS.
        from vllm_sail.patch.utils import PATCH_REGISTRY

        oracle_prefix = "vllm.model_executor.layers.fused_moe.oracle."
        patched_aliases = set()
        for record in PATCH_REGISTRY:
            module_name, _, alias_name = record.target.rpartition(".")
            if module_name.startswith(oracle_prefix):
                patched_aliases.add((module_name, alias_name))

        vllm_root = Path(vllm.__file__).resolve().parent
        discovered = {}
        for path in (vllm_root / "model_executor").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            consumer_name = ".".join(
                ("vllm", *path.relative_to(vllm_root).with_suffix("").parts)
            )
            for node in _module_nodes(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                for imported in node.names:
                    key = (node.module, imported.name)
                    if key in patched_aliases:
                        discovered.setdefault(key, set()).add(consumer_name)

        expected = {}
        for backend_name, backend_aliases in _ORACLE_ALIAS_CONSUMERS.items():
            oracle_name = (
                "vllm.model_executor.layers.fused_moe.oracle."
                f"{backend_name}"
            )
            for alias_name, consumer_names in backend_aliases.items():
                expected[(oracle_name, alias_name)] = set(consumer_names)
        # Keep patched aliases with no by-value consumers in the inventory.
        # The source scan only creates entries when it finds an import.
        for key in expected:
            discovered.setdefault(key, set())
        assert discovered == expected, (discovered, expected)

        for module_name, oracle_name, alias_name in aliases:
            consumer_alias = getattr(consumers[module_name], alias_name)
            oracle_alias = getattr(importlib.import_module(oracle_name), alias_name)
            assert consumer_alias is oracle_alias, (module_name, alias_name)
            assert consumer_alias is not originals[(module_name, alias_name)], (
                module_name,
                alias_name,
            )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
