# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for copied MXFP4 candidate validation and fallback."""

import ast
import sys
from enum import Enum
from types import SimpleNamespace as NS

import pytest

from tests.support.source import definition, function
from vllm_sail.registry.moe_backends._extend import extend_enum

PATCH = "vllm_sail/registry/moe_backends/mxfp4.py"
Backend = Enum(
    "Backend",
    {
        name: name
        for name in (
            "MARLIN",
            "BATCHED_MARLIN",
            "EMULATION",
            "XPU",
            "PPU_DEEPGEMM_MXFP4",
            "BATCHED_PPU_DEEPGEMM_MXFP4",
            "PPU_DEEPGEMM_MXFP4_BF16",
            "BATCHED_PPU_DEEPGEMM_MXFP4_BF16",
            "PPU_DEEPGEMM_MXFP4_BF16_MMA",
            "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
        )
    },
)


def selection(
    candidates,
    *,
    requested="fp8",
    explicit=False,
    batched=False,
    supports=lambda *_: True,
    upstream=None,
    lora=False,
):
    calls, warnings = [], []
    native_keys = {b: "mxfp4" if b.name.endswith("MXFP4") else None for b in Backend}
    native_keys[Backend.EMULATION] = None
    standard, batch = object(), object()

    def kernel(backend):
        class Experts:
            @staticmethod
            def is_supported_config(cls, config, weight, activation, activation_format):
                calls.append((backend, weight, activation, activation_format))
                return supports(backend, activation), "unsupported candidate fixture"

        return Experts

    def explicit_return(backend, config, weight, activation, activation_format):
        cls = kernel(backend)
        supported, reason = cls.is_supported_config(
            cls, config, weight, activation, activation_format
        )
        if not supported:
            raise ValueError(reason)
        return backend, cls

    def warn(backend, activation):
        if (
            activation is not None
            and backend != Backend.EMULATION
            and native_keys[backend] is None
        ):
            warnings.append(backend)

    namespace = {
        "Mxfp4MoeBackend": Backend,
        "mk": NS(FusedMoEActivationFormat=NS(Standard=standard, BatchedExperts=batch)),
        "kMxfp4Static": "weights",
        "_resolve_activation_key": lambda key: key,
        "_get_requested_backends": lambda *_: list(candidates),
        "_get_priority_backends_for_gpt_oss": lambda: list(candidates),
        "_filter_by_activation": lambda backends, _: backends,
        "_backend_activation_key": native_keys.__getitem__,
        "_requires_qwen38_tep8_emulation": lambda *_: False,
        "backend_to_kernel_cls": lambda backend: [kernel(backend)],
        "_return_or_raise": explicit_return,
        "_warn_on_bf16_fallback": warn,
        "_make_log_backend": lambda backend: backend.value,
        "_make_log_unsupported": lambda backend, reason: reason,
        "logger": NS(
            info_once=lambda *a, **kw: None,
            warning_once=lambda *a: warnings.append(Backend(a[-1])),
            debug_once=lambda *a: None,
        ),
        "current_platform": NS(is_xpu=lambda: False),
    }
    if upstream is not None:
        namespace.update(
            kFp8Dynamic128Sym="fp8-block",
            kMxfp8Dynamic="mxfp8",
            kFp8StaticTensorSym="fp8",
            kMxfp4Dynamic="mxfp4",
            oracle=NS(kMxfp4Dynamic="mxfp4"),
            map_mxfp4_backend=lambda _: list(candidates),
        )
        namespace["_upstream_backend_activation_key"] = function(
            upstream, "_backend_activation_key", namespace
        )
        function(PATCH, "_backend_activation_key", namespace)
        constants = [
            node
            for node in ast.parse(upstream.read_text()).body
            if isinstance(node, ast.Assign)
            and any(
                getattr(target, "id", None) == "TRTLLM_BACKENDS"
                for target in node.targets
            )
        ]
        assert len(constants) == 1
        exec(
            compile(ast.Module(body=constants, type_ignores=[]), str(upstream), "exec"),
            namespace,
        )
        for helper in (
            "_filter_by_activation",
            "_get_requested_backends",
            "_warn_on_bf16_fallback",
            "_requested_kernel_cls",
            "_return_or_raise",
        ):
            if helper != "_requested_kernel_cls" or any(
                isinstance(node, ast.FunctionDef) and node.name == helper
                for node in ast.parse(upstream.read_text()).body
            ):
                function(upstream, helper, namespace)
    config = NS(
        moe_backend="explicit" if explicit else "auto",
        moe_parallel_config=NS(use_batched_activation_format=batched),
        is_lora_enabled=lora,
    )
    result = function(PATCH, "select_mxfp4_moe_backend", namespace)(
        config, activation_key=requested
    )
    return result, calls, warnings, batch if batched else standard


def test_automatic_bf16_candidate_uses_own_activation_key():
    (backend, _), calls, warnings, format = selection(
        [Backend.MARLIN], supports=lambda _, key: key is None
    )
    assert backend is Backend.MARLIN
    assert calls == [(Backend.MARLIN, "weights", None, format)]
    assert warnings == [Backend.MARLIN]


@pytest.mark.parametrize("explicit", [False, True])
def test_successful_bf16_fallback_warns(explicit):
    (backend, _), _, warnings, _ = selection([Backend.MARLIN], explicit=explicit)
    assert backend is Backend.MARLIN
    assert warnings == [Backend.MARLIN]


def test_emulation_uses_checkpoint_activation_without_bf16_warning():
    (backend, _), calls, warnings, format = selection([Backend.EMULATION])
    assert backend is Backend.EMULATION
    assert calls == [(backend, "weights", "fp8", format)]
    assert not warnings


@pytest.mark.upstream_source
@pytest.mark.parametrize("lora", [False, True])
def test_explicit_selection_retains_the_ppu_candidate_with_lora(
    real_oracle_source, lora
):
    (backend, _), calls, warnings, format = selection(
        [Backend.PPU_DEEPGEMM_MXFP4_BF16],
        explicit=True,
        requested=None,
        upstream=real_oracle_source,
        lora=lora,
    )
    assert backend is Backend.PPU_DEEPGEMM_MXFP4_BF16
    assert calls == [(backend, "weights", None, format)]
    assert not warnings


@pytest.mark.parametrize(
    "name",
    [
        "PPU_DEEPGEMM_MXFP4",
        "PPU_DEEPGEMM_MXFP4_BF16",
        "PPU_DEEPGEMM_MXFP4_BF16_MMA",
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_explicit_ppu_variant_preserves_backend_and_activation(name, batched):
    (backend, _), calls, warnings, format = selection(
        [Backend[name]], explicit=True, batched=batched, requested=None
    )
    assert backend is Backend[("BATCHED_" if batched else "") + name]
    assert calls == [
        (backend, "weights", "mxfp4" if name.endswith("MXFP4") else None, format)
    ]
    assert not warnings


@pytest.mark.parametrize("explicit", [False, True])
def test_failed_candidate_does_not_warn_before_successful_fallback(explicit):
    (backend, _), calls, warnings, _ = selection(
        [Backend.MARLIN, Backend.EMULATION],
        explicit=explicit,
        supports=lambda candidate, _: candidate is Backend.EMULATION,
    )
    assert backend is Backend.EMULATION
    assert [c[0] for c in calls] == [Backend.MARLIN, Backend.EMULATION]
    assert calls[-1][2] == "fp8"
    assert not warnings


@pytest.fixture
def real_oracle_source(monkeypatch, upstream_source_root):
    path = upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
    node = definition(path, "Mxfp4MoeBackend")
    namespace = {"Enum": Enum}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    enum = namespace["Mxfp4MoeBackend"]
    for member in Backend:
        if not hasattr(enum, member.name):
            extend_enum(enum, member.name, member.value)
    monkeypatch.setattr(sys.modules[__name__], "Backend", enum)
    return path


@pytest.mark.upstream_source
@pytest.mark.parametrize("explicit", [False, True])
def test_real_helpers_reject_first_candidate_then_warn_for_bf16_fallback(
    real_oracle_source, explicit
):
    (backend, _), calls, warnings, format = selection(
        [Backend.PPU_DEEPGEMM_MXFP4, Backend.MARLIN],
        requested="mxfp4",
        explicit=explicit,
        upstream=real_oracle_source,
        supports=lambda candidate, key: candidate is Backend.MARLIN and key is None,
    )
    assert backend is Backend.MARLIN
    assert calls == [
        (Backend.PPU_DEEPGEMM_MXFP4, "weights", "mxfp4", format),
        (Backend.MARLIN, "weights", None, format),
    ]
    assert warnings == [Backend.MARLIN]


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "name",
    [
        "PPU_DEEPGEMM_MXFP4",
        "PPU_DEEPGEMM_MXFP4_BF16",
        "PPU_DEEPGEMM_MXFP4_BF16_MMA",
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_real_helpers_preserve_ppu_variant_activation(
    real_oracle_source, name, batched
):
    (backend, _), calls, warnings, format = selection(
        [Backend[name]],
        explicit=True,
        batched=batched,
        requested=None,
        upstream=real_oracle_source,
    )
    assert backend is Backend[("BATCHED_" if batched else "") + name]
    assert calls == [
        (backend, "weights", "mxfp4" if name.endswith("MXFP4") else None, format)
    ]
    assert not warnings


@pytest.mark.upstream_source
@pytest.mark.parametrize("explicit", [False, True])
def test_real_helpers_preserve_emulation_checkpoint_key(real_oracle_source, explicit):
    (backend, _), calls, warnings, format = selection(
        [Backend.EMULATION], explicit=explicit, upstream=real_oracle_source
    )
    assert backend is Backend.EMULATION
    assert calls == [(backend, "weights", "fp8", format)]
    assert not warnings
