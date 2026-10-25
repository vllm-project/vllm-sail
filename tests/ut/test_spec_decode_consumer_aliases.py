# SPDX-License-Identifier: Apache-2.0
"""Keep speculative sampling consumers on the platform-specific dispatcher."""

from __future__ import annotations

import ast
import functools
import importlib.util
from types import ModuleType, SimpleNamespace

import pytest

from tests.support.source import ROOT, function
from vllm_sail.patch.bodies import bind_body

PROVIDER = "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils"
CONSUMERS = (
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
    "vllm.v1.watermarking.spec_decode",
)
RUNTIME = "vllm_sail/patch/enhancement/runtime.py"


def _declared_consumers():
    tree = ast.parse((ROOT / RUNTIME).read_text())
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_CONSUMERS"
            for target in node.targets
        )
    ]
    assert len(declarations) == 1
    return ast.literal_eval(declarations[0].value)


@pytest.fixture
def installer():
    provider = ModuleType(PROVIDER)
    modules = {PROVIDER: provider}
    patches = []
    platform = SimpleNamespace(ppu=True)

    def original(*args, **kwargs):
        return "upstream", args, kwargs

    def replacement(*args, **kwargs):
        return "ppu", args, kwargs

    provider.rejection_sample = original

    def patch(module_name, attribute, **metadata):
        def apply(dispatch):
            setattr(modules[module_name], attribute, dispatch)
            patches.append((module_name, attribute))
            return dispatch

        return apply

    install = function(
        RUNTIME,
        "_install",
        dict(
            bind_body=bind_body,
            functools=functools,
            current_platform=SimpleNamespace(is_ppu=lambda: platform.ppu),
            patch=patch,
            sys=SimpleNamespace(modules=modules),
            _CONSUMERS=_declared_consumers(),
        ),
    )
    return SimpleNamespace(
        provider=provider,
        modules=modules,
        patches=patches,
        platform=platform,
        original=original,
        replacement=replacement,
        install=lambda: install(provider, provider, "rejection_sample", replacement),
    )


@pytest.mark.parametrize("consumer_name", CONSUMERS)
@pytest.mark.parametrize("is_ppu", [True, False])
def test_preloaded_consumer_uses_the_same_platform_dispatcher(
    installer, consumer_name, is_ppu
):
    consumer = ModuleType(consumer_name)
    consumer.rejection_sample = installer.original
    installer.modules[consumer_name] = consumer
    installer.platform.ppu = is_ppu
    installer.install()

    assert consumer.rejection_sample is installer.provider.rejection_sample
    token, contexts = object(), object()
    assert consumer.rejection_sample(token, contexts=contexts) == (
        "ppu" if is_ppu else "upstream",
        (token,),
        {"contexts": contexts},
    )


@pytest.mark.parametrize("consumer_name", CONSUMERS)
def test_consumer_with_a_different_binding_is_preserved(installer, consumer_name):
    consumer = ModuleType(consumer_name)

    def unrelated():
        return "unrelated"

    consumer.rejection_sample = unrelated
    installer.modules[consumer_name] = consumer
    installer.install()

    assert consumer.rejection_sample is unrelated
    assert installer.patches == [(PROVIDER, "rejection_sample")]


def test_cold_install_does_not_load_consumers(installer):
    installer.install()

    assert set(installer.modules) == {PROVIDER}
    assert installer.provider.rejection_sample() == ("ppu", (), {})
    assert installer.patches == [(PROVIDER, "rejection_sample")]


@pytest.mark.upstream_source
def test_consumer_inventory_matches_all_module_scope_imports(upstream_source_root):
    discovered = set()
    for path in (upstream_source_root / "vllm").rglob("*.py"):
        relative = path.relative_to(upstream_source_root).with_suffix("")
        parts = list(relative.parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module_name = ".".join(parts)
        package = module_name if is_package else module_name.rpartition(".")[0]
        pending = list(ast.parse(path.read_text()).body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if isinstance(node, ast.ImportFrom):
                imported = node.module or ""
                if node.level:
                    imported = importlib.util.resolve_name(
                        "." * node.level + imported, package
                    )
                if imported == PROVIDER:
                    for alias in node.names:
                        if alias.name in {"rejection_sample", "*"}:
                            discovered.add(
                                (module_name, alias.asname or "rejection_sample")
                            )
            pending.extend(ast.iter_child_nodes(node))

    assert discovered == {(name, "rejection_sample") for name in _declared_consumers()}


@pytest.mark.upstream_source
@pytest.mark.parametrize("is_ppu", [True, False])
def test_real_watermark_wrapper_preserves_sampling_inputs(
    installer, upstream_source_root, is_ppu
):
    consumer = ModuleType("vllm.v1.watermarking.spec_decode")
    consumer.rejection_sample = installer.original
    installer.modules[consumer.__name__] = consumer
    installer.platform.ppu = is_ppu
    installer.install()
    watermarker = SimpleNamespace(context_width=3)

    def resolve_key(actual):
        assert actual is watermarker
        return 123

    wrapper = function(
        upstream_source_root / "vllm/v1/watermarking/spec_decode.py",
        "watermarked_rejection_sample",
        dict(
            rejection_sample=consumer.rejection_sample,
            _resolve_watermark_key=resolve_key,
        ),
    )
    args = [SimpleNamespace(shape=(2, 32)), *(object() for _ in range(9)), 7]
    contexts = SimpleNamespace(shape=(2, 3))
    watermarking, skip = object(), object()
    result = wrapper(
        *args,
        contexts,
        watermarking,
        watermarker,
        watermarking_skip_mask=skip,
        use_fp64=True,
    )

    assert result == (
        "ppu" if is_ppu else "upstream",
        tuple(args),
        dict(
            use_fp64=True,
            contexts=contexts,
            watermarking=watermarking,
            watermarking_skip_mask=skip,
            watermark_key=123,
        ),
    )
