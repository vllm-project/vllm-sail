# SPDX-License-Identifier: Apache-2.0
"""u258 P2: layout-driven MoE backend decision matrix, torch-free.

Locks:
 1. decision matrix cell-by-cell (CLI > env > quant_method-derived),
    incompatible requests raise at construction time (before weight load);
 2. CLI --moe-backend ppu_acext and env VLLM_SAIL_MOE_BACKEND=acext hit the
    SAME raise site with the SAME three-element message (quant_method,
    requested backend, fix suggestion);
 3. D1 single-point constant: shipped "deepgemm" (group-owner 2026-11-25),
    "raise" alternative and None pending state both exercised;
 4. D2 single-point constant: True allows mixed explicit opt-in, False raises;
 5. decision log line format + scope="process" (per-worker, not rank0-only);
 6. the dead env spelling branch in _get_priority_backends is gone.
"""

from __future__ import annotations

import ast
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.support.source import ROOT, definition, function

REGISTRY = "vllm_sail/registry/moe_backends/w4a8_int8.py"
MIXED_W4 = "vllm_sail/model_executor/layers/quantization/mixed_precision_w4.py"

NATIVE = "native"
R1 = "R1"  # plain layout + acext
R2 = "R2"  # t_order + acext on DeepEP LL (or D1=raise)
R3 = "R3"  # deepgemm request without batched format
OTHER = "OTHER"  # backend targeting a different quant format
PENDING = "PENDING"  # D1 undecided


class _FakeExperts:
    @staticmethod
    def is_supported_config(*args):
        return True, None


def _decision_ns(env=None, d1="deepgemm", d2=True):
    ns = {
        "ppu_envs": SimpleNamespace(VLLM_SAIL_MOE_BACKEND=env),
        "LAYOUT_QUANT_METHOD": {
            "deepgemm_plain": "w4a8_int8",
            "acext_t_order": "mixed_precision_w4",
        },
        "D1_MIXED_DEEPEP_LL_AUTO": d1,
        "D2_ALLOW_MIXED_EXPLICIT_DEEPGEMM": d2,
        "W4A8Int8MoeBackend": SimpleNamespace(
            BATCHED_PPU_DEEPGEMM=SimpleNamespace(value="BATCHED_PPU_DEEPGEMM")
        ),
        "mk": SimpleNamespace(
            FusedMoEActivationFormat=SimpleNamespace(BatchedExperts="BATCHED")
        ),
        "logger": SimpleNamespace(info_once=lambda *a, **k: None),
        "backend_to_kernel_cls": lambda backend: [_FakeExperts],
    }
    for name in (
        "_resolve_requested",
        "_canonical_backend",
        "_incompatible_error",
        "_native_backend_or_raise",
        "_select_by_layout",
    ):
        function(REGISTRY, name, ns)
    # T1: exercise the REAL compat fallback helper, not a stub.
    function("vllm_sail/compat.py", "fused_moe_use_batched", ns)
    return ns


def _config(cli="auto", batched=True, shape="v0.34"):
    # T1: stub shape must match the real upstream config shape.
    # v0.34: use_batched_activation_format lives on FusedMoEConfig itself;
    # v0.33: it lives on FusedMoEParallelConfig (moe_parallel_config).
    if shape == "v0.34":
        parallel = SimpleNamespace()
        assert not hasattr(parallel, "use_batched_activation_format")
        return SimpleNamespace(
            moe_backend=cli,
            use_batched_activation_format=batched,
            moe_parallel_config=parallel,
        )
    assert shape == "v0.33"
    cfg = SimpleNamespace(
        moe_backend=cli,
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=batched),
    )
    assert not hasattr(cfg, "use_batched_activation_format")
    return cfg


def _run(ns, cli="auto", batched=True, layout="deepgemm_plain", shape="v0.34"):
    return ns["_select_by_layout"](_config(cli, batched, shape), "W4", "A8", layout)


# ---------------------------------------------------------------------------
# 1. decision matrix, cell by cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout,cli,env,batched,d1,d2,expect",
    [
        # plain (w4a8_int8) + batched: auto/env/CLI deepgemm -> native
        ("deepgemm_plain", "auto", None, True, "deepgemm", True, NATIVE),
        ("deepgemm_plain", "auto", "deepgemm", True, "deepgemm", True, NATIVE),
        ("deepgemm_plain", "ppu_deep_gemm_w4a8_int8", None, True, "deepgemm", True, NATIVE),
        # plain + acext via either knob -> R1
        ("deepgemm_plain", "auto", "acext", True, "deepgemm", True, R1),
        ("deepgemm_plain", "ppu_acext", None, True, "deepgemm", True, R1),
        # plain + other-format backends -> OTHER
        ("deepgemm_plain", "auto", "triton", True, "deepgemm", True, OTHER),
        ("deepgemm_plain", "ppu_deep_gemm", None, True, "deepgemm", True, OTHER),
        # plain without batched format -> R3 for deepgemm/auto, R1 for acext
        ("deepgemm_plain", "auto", None, False, "deepgemm", True, R3),
        ("deepgemm_plain", "ppu_deep_gemm_w4a8_int8", None, False, "deepgemm", True, R3),
        ("deepgemm_plain", "auto", "acext", False, "deepgemm", True, R1),
        # t_order (mixed) + batched: D1 shipped default keeps PR deepgemm
        ("acext_t_order", "auto", None, True, "deepgemm", True, NATIVE),
        ("acext_t_order", "auto", None, True, "raise", True, R2),
        ("acext_t_order", "auto", None, True, None, True, PENDING),
        # t_order + acext via either knob -> R2
        ("acext_t_order", "auto", "acext", True, "deepgemm", True, R2),
        ("acext_t_order", "ppu_acext", None, True, "deepgemm", True, R2),
        # t_order + explicit deepgemm -> D2 gate
        ("acext_t_order", "ppu_deep_gemm_w4a8_int8", None, True, "deepgemm", True, NATIVE),
        ("acext_t_order", "ppu_deep_gemm_w4a8_int8", None, True, "deepgemm", False, OTHER),
        ("acext_t_order", "auto", "deepgemm", True, "deepgemm", True, NATIVE),
        # CLI wins over env
        ("deepgemm_plain", "ppu_deep_gemm_w4a8_int8", "acext", True, "deepgemm", True, NATIVE),
        ("deepgemm_plain", "ppu_acext", "deepgemm", True, "deepgemm", True, R1),
    ],
)
def test_decision_matrix(layout, cli, env, batched, d1, d2, expect):
    ns = _decision_ns(env=env, d1=d1, d2=d2)
    if expect == NATIVE:
        backend, cls = _run(ns, cli, batched, layout)
        assert backend.value == "BATCHED_PPU_DEEPGEMM"
        assert cls is _FakeExperts
        return
    if expect == PENDING:
        with pytest.raises(RuntimeError, match="D1 policy pending"):
            _run(ns, cli, batched, layout)
        return
    with pytest.raises(ValueError) as exc:
        _run(ns, cli, batched, layout)
    msg = str(exc.value)
    quant = "w4a8_int8" if layout == "deepgemm_plain" else "mixed_precision_w4"
    assert f"quant_method='{quant}'" in msg
    assert "Fix:" in msg
    if expect == R1:
        assert "'acext'" in msg
    elif expect == R2:
        assert "'acext'" in msg
        assert "DeepEP low-latency" in msg
    elif expect == R3:
        assert "'deepgemm'" in msg
        assert "batched" in msg


def test_t_order_non_batched_is_defensive_assert():
    ns = _decision_ns()
    with pytest.raises(AssertionError, match="legacy acext path"):
        _run(ns, "auto", False, "acext_t_order")


# ---------------------------------------------------------------------------
# 2. CLI and env hit the SAME raise with the SAME message
# ---------------------------------------------------------------------------


def test_cli_ppu_acext_and_env_acext_identical_raise():
    ns_env = _decision_ns(env="acext")
    ns_cli = _decision_ns(env=None)
    with pytest.raises(ValueError) as e_env:
        _run(ns_env, "auto", True, "deepgemm_plain")
    with pytest.raises(ValueError) as e_cli:
        _run(ns_cli, "ppu_acext", True, "deepgemm_plain")
    assert type(e_env.value) is type(e_cli.value) is ValueError
    assert str(e_env.value) == str(e_cli.value)
    for element in ("quant_method='w4a8_int8'", "'acext'", "Fix:"):
        assert element in str(e_cli.value)


def test_cli_and_env_acext_identical_on_t_order():
    ns_env = _decision_ns(env="acext")
    ns_cli = _decision_ns(env=None)
    with pytest.raises(ValueError) as e_env:
        _run(ns_env, "auto", True, "acext_t_order")
    with pytest.raises(ValueError) as e_cli:
        _run(ns_cli, "ppu_acext", True, "acext_t_order")
    assert str(e_env.value) == str(e_cli.value)


# ---------------------------------------------------------------------------
# 3/4. shipped constants and single-point flippability
# ---------------------------------------------------------------------------


def _module_assign(name):
    node = ast.parse((ROOT / REGISTRY).read_text())
    for part in node.body:
        targets = (
            part.targets if isinstance(part, ast.Assign)
            else [part.target] if isinstance(part, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(part.value)
    raise AssertionError(f"{name} not found")


def test_shipped_d1_is_deepgemm_per_group_owner_ruling():
    assert _module_assign("D1_MIXED_DEEPEP_LL_AUTO") == "deepgemm"


def test_shipped_d2_allows_mixed_explicit_optin():
    assert _module_assign("D2_ALLOW_MIXED_EXPLICIT_DEEPGEMM") is True


def test_d1_raise_variant_still_present():
    # The alternative must stay one constant-flip away, not deleted.
    src = (ROOT / REGISTRY).read_text()
    assert 'D1_MIXED_DEEPEP_LL_AUTO == "raise"' in src
    assert '"deepgemm"' in src and '"raise"' in src


# ---------------------------------------------------------------------------
# 5. decision log line: format, scope, rendered sample
# ---------------------------------------------------------------------------


def test_decision_log_format_and_scope():
    node = definition(ROOT / REGISTRY, "make_w4a8_int8_moe_kernel")
    calls = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "info_once"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and "[vllm_sail] w4a8 MoE decision:" in str(n.args[0].value)
    ]
    assert len(calls) == 1
    fmt = calls[0].args[0].value
    for field in (
        "quant_method=%s", "layout=%s", "requested=%s", "selected=%s",
        "experts=%s", "prepare=%s", "pid=%s",
    ):
        assert field in fmt
    scope = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
    assert scope.get("scope") == "process", (
        "decision line must print per worker process; default 'local' scope "
        "would print only on node-local rank 0"
    )


def test_decision_log_rendered_sample(monkeypatch):
    a2a_name = "vllm.model_executor.layers.fused_moe.all2all_utils"

    class DeepEPLLPrepareAndFinalize:
        activation_format = "BATCHED"

        def max_num_tokens_per_rank(self):
            return 128

        def num_dispatchers(self):
            return 2

    pf = DeepEPLLPrepareAndFinalize()
    a2a = ModuleType(a2a_name)
    a2a.maybe_make_prepare_finalize = lambda **kw: pf
    monkeypatch.setitem(sys.modules, a2a_name, a2a)
    monkeypatch.setattr(os, "getpid", lambda: 4242)

    logged = []

    _Experts = type(
        "PPUDeepGemmExpertsW4A8Int8",
        (),
        {
            "__init__": lambda self, **kwargs: None,
            "is_monolithic": staticmethod(lambda: False),
        },
    )

    ns = {
        "mk": SimpleNamespace(
            FusedMoEActivationFormat=SimpleNamespace(BatchedExperts="BATCHED"),
            FusedMoEKernel=lambda p, e: (p, e),
        ),
        "logger": SimpleNamespace(
            info_once=lambda msg, *a, **k: logged.append((msg, a, k))
        ),
        "ppu_envs": SimpleNamespace(VLLM_SAIL_MOE_BACKEND=None),
        "LAYOUT_QUANT_METHOD": {"deepgemm_plain": "w4a8_int8"},
        "_resolve_requested": function(REGISTRY, "_resolve_requested", {
            "ppu_envs": SimpleNamespace(VLLM_SAIL_MOE_BACKEND=None)
        }),
    }
    make = function(REGISTRY, "make_w4a8_int8_moe_kernel", ns)
    make(
        moe_quant_config="QC",
        moe_config=_config("auto", True),
        experts_cls=_Experts,
        routing_tables=None,
        weight_layout="deepgemm_plain",
    )
    decision = [(m, a) for m, a, k in logged
                if m.startswith("[vllm_sail] w4a8 MoE decision:")]
    assert len(decision) == 1
    rendered = decision[0][0] % decision[0][1]
    assert rendered == (
        "[vllm_sail] w4a8 MoE decision: quant_method=w4a8_int8 "
        "layout=deepgemm_plain requested=auto selected=BATCHED_PPU_DEEPGEMM "
        "experts=PPUDeepGemmExpertsW4A8Int8 "
        "prepare=DeepEPLLPrepareAndFinalize pid=4242"
    )


# ---------------------------------------------------------------------------
# 6. wiring locks
# ---------------------------------------------------------------------------


def test_select_signature_has_layout_kwarg_default_none():
    node = definition(ROOT / REGISTRY, "select_w4a8_int8_moe_backend")
    kwonly = {
        a.arg: ast.literal_eval(d) if d is not None else None
        for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
    }
    assert kwonly.get("weight_layout") is None


def test_mixed_w4_init_passes_weight_layout_to_select():
    node = definition(ROOT / MIXED_W4, "W4AInt8MoEMethod.__init__")
    calls = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "select_w4a8_int8_moe_backend"
    ]
    assert len(calls) == 1
    kws = {kw.arg for kw in calls[0].keywords}
    assert "weight_layout" in kws


def test_dead_env_spelling_branch_removed():
    node = definition(ROOT / REGISTRY, "_get_priority_backends")
    src = ast.unparse(node)
    assert "requested == 'ppu_deep_gemm_w4a8_int8'" not in src
    assert "requested == 'deepgemm'" in src


# ---------------------------------------------------------------------------
# 7. T1/B1: compat fallback reads BOTH upstream homes of the batched flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["v0.34", "v0.33"])
@pytest.mark.parametrize("batched", [True, False])
def test_decision_same_verdict_on_both_config_shapes(shape, batched):
    """v0.34 top-level and v0.33 parallel-config homes must decide identically."""
    ns = _decision_ns()
    if batched:
        backend, cls = _run(ns, "auto", True, "deepgemm_plain", shape)
        assert backend.value == "BATCHED_PPU_DEEPGEMM"
        assert cls is _FakeExperts
    else:
        with pytest.raises(ValueError, match="batched"):
            _run(ns, "auto", False, "deepgemm_plain", shape)


def _compat_helper():
    ns = {}
    function("vllm_sail/compat.py", "fused_moe_use_batched", ns)
    return ns["fused_moe_use_batched"]


@pytest.mark.parametrize("shape", ["v0.34", "v0.33"])
@pytest.mark.parametrize("value", [True, False])
def test_compat_helper_preserves_false_on_both_homes(shape, value):
    # False is a legal value: the helper must return it, not fall through.
    assert _compat_helper()(_config("auto", value, shape)) is value


def test_compat_helper_new_home_wins_over_old():
    cfg = SimpleNamespace(
        use_batched_activation_format=False,
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=True),
    )
    assert _compat_helper()(cfg) is False


def test_compat_helper_missing_everywhere_raises():
    cfg = SimpleNamespace(moe_parallel_config=SimpleNamespace())
    with pytest.raises(AttributeError, match="use_batched_activation_format"):
        _compat_helper()(cfg)
    with pytest.raises(AttributeError, match="use_batched_activation_format"):
        _compat_helper()(SimpleNamespace())
