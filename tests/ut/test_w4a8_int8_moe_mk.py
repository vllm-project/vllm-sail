# SPDX-License-Identifier: Apache-2.0
"""W4A8-INT8 (MixedPrecisionW4) modular-kernel wiring, without torch or a device.

Locks the four behaviours the review asked for:
  1. the legacy branch still returns None and keeps the 16-argument acext call;
  2. MK assembly happens only under DeepEP low-latency + batched activations;
  3. the acext Q_type comes from explicit constants, never from quant_dtype flags;
  4. weight parameter names match the names the modular kernel reads.

Device-dependent numeric checks are marked ``ppu`` and skipped by
``tests/conftest.py`` on CPU; a skip is not evidence of a pass.
"""

from __future__ import annotations

import ast
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.support.source import ROOT, definition, function

MIXED_W4 = "vllm_sail/model_executor/layers/quantization/mixed_precision_w4.py"
EXPERTS = "vllm_sail/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
REGISTRY = "vllm_sail/registry/moe_backends/w4a8_int8.py"
CONSUMERS = "vllm_sail/registry/moe_backends/__init__.py"
LITERAL = "vllm_sail/patch/enhancement/residual/kernel_moe_backend_literal.py"

NEW_BACKEND = "ppu_deep_gemm_w4a8_int8"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parallel_config(deepep_ll: bool, batched: bool) -> SimpleNamespace:
    return SimpleNamespace(
        use_deepep_ll_kernels=deepep_ll,
        use_batched_activation_format=batched,
    )


def _mk_enabled(deepep_ll: bool, batched: bool) -> bool:
    method = function(
        MIXED_W4,
        "W4AInt8MoEMethod._mk_enabled",
        {},
    )
    return method(SimpleNamespace(moe=SimpleNamespace(moe_parallel_config=_parallel_config(deepep_ll, batched))))


def _install_acext_stubs(monkeypatch, calls):
    """Provide the two lazy imports inside apply() without torch or acext."""

    def fusedmoe_wrapper(*args):
        calls.append(args)

    def get_enum_from_booleans(**kwargs):
        calls.append(("Q_type", kwargs))
        return "Q_TYPE_SENTINEL"

    acext = ModuleType("acext")
    acext.fusedmoe_wrapper = fusedmoe_wrapper
    acext.get_enum_from_booleans = get_enum_from_booleans
    acext.pad_to_multiple_of_16 = lambda num_tokens, topk: num_tokens * topk
    monkeypatch.setitem(sys.modules, "acext", acext)

    utils_acext = ModuleType("vllm_sail.utils.acext")
    utils_acext.load_acext = lambda: None
    monkeypatch.setitem(sys.modules, "vllm_sail.utils.acext", utils_acext)


def _legacy_self(calls, moe_kernel=None):
    return SimpleNamespace(
        moe_kernel=moe_kernel,
        ep_rank=3,
        ep_size=8,
        quant_config=SimpleNamespace(pack_factor=2),
    )


def _legacy_layer():
    return SimpleNamespace(
        activation="SILU",
        w13_weight="W13",
        w2_weight="W2",
        w13_weight_scale="S13",
        w2_weight_scale="S2",
    )


def _run_legacy_apply(monkeypatch, calls):
    _install_acext_stubs(monkeypatch, calls)
    apply = function(
        MIXED_W4,
        "W4AInt8MoEMethod.apply",
        {
            "torch": SimpleNamespace(empty_like=lambda x: "OUTPUT"),
            "MoEActivation": SimpleNamespace(SILU="SILU"),
        },
    )
    x = SimpleNamespace(shape=(4, 8))
    topk_ids = SimpleNamespace(shape=(4, 2))
    return apply(
        _legacy_self(calls),
        _legacy_layer(),
        x,
        "TOPK_WEIGHTS",
        topk_ids,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# 2. MK eligibility is DeepEP-LL + batched only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deepep_ll,batched,expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_mk_enabled_requires_deepep_ll_and_batched(deepep_ll, batched, expected):
    assert _mk_enabled(deepep_ll, batched) is expected


def test_legacy_quant_config_returns_none():
    get_config = function(MIXED_W4, "W4AInt8MoEMethod.get_fused_moe_quant_config", {})
    assert get_config(SimpleNamespace(_mk_enabled=lambda: False), object()) is None


def test_mk_quant_config_keeps_activations_unquantized_on_the_wire():
    """PER_TOKEN + dtype None keeps block_shape None so DeepEP dispatches BF16.

    A PER_CHANNEL activation shape would make block_shape [-1, 1] and an int8
    activation dtype would flip sail's DeepEP int8-dispatch gate; both would
    hand acext a pre-quantized buffer it does not consume.
    """
    recorded = {}

    class Desc:
        def __init__(self, dtype=None, shape=None, scale=None):
            recorded.setdefault("descs", []).append((dtype, shape, scale))

    class Config:
        def __init__(self, **kwargs):
            recorded["config"] = kwargs

    get_config = function(
        MIXED_W4,
        "W4AInt8MoEMethod.get_fused_moe_quant_config",
        {
            "FusedMoEQuantConfig": Config,
            "FusedMoEQuantDesc": Desc,
            "GroupShape": SimpleNamespace(PER_TOKEN="PER_TOKEN", PER_CHANNEL="PER_CHANNEL"),
        },
    )
    layer = SimpleNamespace(w13_weight_scale="S13", w2_weight_scale="S2")
    get_config(SimpleNamespace(_mk_enabled=lambda: True), layer)

    descs = recorded["descs"]
    assert descs[0] == (None, "PER_TOKEN", None)
    assert descs[1] == (None, "PER_TOKEN", None)
    assert descs[2] == (None, "PER_CHANNEL", "S13")
    assert descs[3] == (None, "PER_CHANNEL", "S2")


# ---------------------------------------------------------------------------
# 1. legacy acext call contract is unchanged
# ---------------------------------------------------------------------------


def test_legacy_apply_keeps_sixteen_positional_arguments(monkeypatch):
    calls: list = []
    result = _run_legacy_apply(monkeypatch, calls)
    assert result == "OUTPUT"

    wrapper_calls = [c for c in calls if c and c[0] != "Q_type"]
    assert len(wrapper_calls) == 1
    args = wrapper_calls[0]
    assert len(args) == 16, f"legacy acext contract changed: {len(args)} args"
    # x, w1, w2, topk_weights, topk_ids, output, expanded_size,
    # w1_scale, w2_scale, None, None, None, None, ep_rank, ep_size, Q_type
    assert args[1] == "W13"
    assert args[2] == "W2"
    assert args[3] == "TOPK_WEIGHTS"
    assert args[5] == "OUTPUT"
    assert args[6] == 8  # pad_to_multiple_of_16(num_tokens=4, topk=2)
    assert args[7] == "S13"
    assert args[8] == "S2"
    assert args[9:13] == (None, None, None, None)
    assert args[13] == 3  # ep_rank passed through
    assert args[14] == 8  # ep_size passed through
    assert args[15] == "Q_TYPE_SENTINEL"


def test_legacy_qtype_selects_w4a8(monkeypatch):
    calls: list = []
    _run_legacy_apply(monkeypatch, calls)
    qtype_calls = [c for c in calls if c and c[0] == "Q_type"]
    assert len(qtype_calls) == 1
    kwargs = qtype_calls[0][1]
    assert kwargs["use_int8_w4a8"] is True
    for flag in (
        "use_fp8_w8a8",
        "use_int8_w8a8",
        "use_int8_w8a16",
        "use_int4_w4a16",
        "use_fp8_w8a16",
    ):
        assert kwargs[flag] is False


# ---------------------------------------------------------------------------
# 4. weight parameter names match what the kernel reads
# ---------------------------------------------------------------------------


def test_mk_apply_reads_original_weight_parameter_names(monkeypatch):
    recorded = {}

    class Kernel:
        def apply(self, *args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return "MK_OUTPUT"

    apply = function(MIXED_W4, "W4AInt8MoEMethod.apply", {})
    layer = SimpleNamespace(
        activation="SILU",
        w13_weight="W13",
        w2_weight="W2",
        global_num_experts=64,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    result = apply(
        SimpleNamespace(moe_kernel=Kernel()),
        layer,
        "X",
        "TW",
        "TI",
        None,
        None,
    )
    assert result == "MK_OUTPUT"
    assert recorded["args"][1] == "W13"
    assert recorded["args"][2] == "W2"


def test_process_weights_replaces_original_names_only():
    """No w13_weight_packed-style alias: the kernel would read stale tensors."""
    node = definition(ROOT / MIXED_W4, "W4AInt8MoEMethod.process_weights_after_loading")
    replaced = [
        call.args[1].value
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "replace_parameter"
        and len(call.args) > 1
        and isinstance(call.args[1], ast.Constant)
    ]
    assert replaced == ["w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"]
    assert not any("packed" in name for name in replaced)


def test_mk_conversion_unpacks_six_values():
    node = definition(ROOT / MIXED_W4, "W4AInt8MoEMethod.process_weights_after_loading")
    targets = [
        target
        for target in ast.walk(node)
        if isinstance(target, ast.Tuple) and len(target.elts) == 6
    ]
    assert targets, "conversion result must be unpacked as upstream's 6-tuple"


# ---------------------------------------------------------------------------
# 3. Q_type is an explicit constant in the MK experts too
# ---------------------------------------------------------------------------


def test_mk_experts_qtype_is_literal_constants():
    node = definition(ROOT / EXPERTS, "PPUDeepGemmExpertsW4A8Int8.apply")
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "get_enum_from_booleans"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert keywords, "Q_type must be built from explicit keyword constants"
    for name, value in keywords.items():
        assert isinstance(value, ast.Constant), (
            f"{name} must be a literal, not derived from quant_config"
        )
    assert keywords["use_int8_w4a8"].value is True
    assert keywords["use_int8_w8a8"].value is False
    assert not calls[0].args, "no positional flags: they would be order-fragile"


def test_mk_experts_never_reads_quant_config_flags_for_qtype():
    node = definition(ROOT / EXPERTS, "PPUDeepGemmExpertsW4A8Int8.apply")
    forbidden = {
        "use_int8_w8a8",
        "use_fp8_w8a8",
        "use_int8_w8a16",
        "use_int4_w4a16",
        "use_fp8_w8a16",
        "quant_dtype",
    }
    used = {
        attr.attr
        for attr in ast.walk(node)
        if isinstance(attr, ast.Attribute) and attr.attr in forbidden
    }
    assert not used, f"Q_type must not be derived from {used}"


def test_mk_experts_declares_batched_format_and_delegate():
    fmt = function(EXPERTS, "PPUDeepGemmExpertsW4A8Int8.activation_format", {"mk": SimpleNamespace(FusedMoEActivationFormat=SimpleNamespace(BatchedExperts="BATCHED"))})
    assert fmt() == "BATCHED"

    node = definition(ROOT / EXPERTS, "PPUDeepGemmExpertsW4A8Int8.finalize_weight_and_reduce_impl")
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert len(returns) == 1
    assert isinstance(returns[0].value, ast.Call)
    assert returns[0].value.func.id == "TopKWeightAndReduceDelegate"


def test_mk_experts_supports_only_w4a8_int4_scheme():
    supports = function(
        EXPERTS,
        "PPUDeepGemmExpertsW4A8Int8._supports_quant_scheme",
        {
            "kInt4W4A8StaticChannelSym": "W4",
            "kInt8DynamicTokenSym": "A8",
        },
    )
    assert supports("W4", "A8") is True
    assert supports("W8", "A8") is False
    assert supports(None, None) is False


def test_mk_experts_apply_has_no_host_sync():
    """expert_num_tokens_cpu is None on this path: .item() would break graphs."""
    node = definition(ROOT / EXPERTS, "PPUDeepGemmExpertsW4A8Int8.apply")
    synced = [
        attr.attr
        for attr in ast.walk(node)
        if isinstance(attr, ast.Attribute) and attr.attr in {"item", "tolist", "cpu"}
    ]
    assert not synced, f"host synchronisation found: {synced}"


# ---------------------------------------------------------------------------
# layout reference plan (pure Python, no torch)
# ---------------------------------------------------------------------------


def test_batched_row_plan_is_expert_major():
    plan = function(EXPERTS, "_w4a8_batched_row_plan", {})
    ids, valid, num_valid = plan([2, 0, 3], 4)
    assert ids == [0] * 4 + [1] * 4 + [2] * 4
    assert valid == [True, True, False, False] + [False] * 4 + [True, True, True, False]
    assert num_valid == 5
    assert len(ids) == 3 * 4


def test_batched_row_plan_empty_and_full():
    plan = function(EXPERTS, "_w4a8_batched_row_plan", {})
    assert plan([], 4) == ([], [], 0)
    ids, valid, num_valid = plan([4, 4], 4)
    assert num_valid == 8
    assert all(valid)
    assert ids == [0] * 4 + [1] * 4


@pytest.mark.parametrize(
    "value,expected", [(0, 0), (1, 16), (16, 16), (17, 32), (32, 32), (33, 48)]
)
def test_pad_to_multiple_of_16(value, expected):
    pad = function(EXPERTS, "_w4a8_pad_to_multiple_of_16", {})
    assert pad(value) == expected


def test_moe_problem_size_uses_batched_token_axis():
    """a1 is [E, T, K]: M must be T, not E (the draft took a1.size(0))."""
    node = definition(ROOT / EXPERTS, "PPUDeepGemmExpertsW4A8Int8.moe_problem_size")
    source = ast.unparse(node)
    assert "a1.size(1)" in source
    assert "a1.dim() == 3" in source


# ---------------------------------------------------------------------------
# P1-1 registry: mapping, selection, conversion, kernel factory
# ---------------------------------------------------------------------------


def test_map_backend_recognizes_new_name_only():
    mapping = function(
        REGISTRY,
        "map_w4a8_int8_backend",
        {
            "W4A8Int8MoeBackend": SimpleNamespace(BATCHED_PPU_DEEPGEMM="BATCHED"),
            "_upstream_map_w4a8_int8_backend": lambda b: f"upstream:{b}",
        },
    )
    assert mapping(NEW_BACKEND) == "BATCHED"
    assert mapping("ppu_deep_gemm_w4a8") == "upstream:ppu_deep_gemm_w4a8"
    assert mapping("cpu") == "upstream:cpu"


def test_backend_to_kernel_cls_resolves_ppu_expert():
    node = definition(ROOT / REGISTRY, "backend_to_kernel_cls")
    source = ast.unparse(node)
    assert "PPUDeepGemmExpertsW4A8Int8" in source
    assert "BATCHED_PPU_DEEPGEMM" in source


def test_select_rejects_unsupported_without_nameerror():
    """`reason` must be initialised: an empty candidate list caused NameError."""
    node = definition(ROOT / REGISTRY, "select_w4a8_int8_moe_backend")
    assigns = [
        n for n in ast.walk(node)
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "reason"
    ]
    assert assigns, "reason must be declared before the candidate loop"


def test_convert_passthrough_preserves_six_tuple(monkeypatch):
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(is_ppu=lambda: True)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    convert = function(
        REGISTRY,
        "convert_to_w4a8_int8_moe_format",
        {
            "torch": SimpleNamespace(empty=lambda n: f"EMPTY{n}"),
            "_upstream_convert_to_w4a8_int8_moe_format": lambda *a, **k: "UPSTREAM",
        },
    )
    result = convert("W13", "W2", "S13", "S2", -1)
    assert isinstance(result, tuple)
    assert len(result) == 6
    assert result[:4] == ("W13", "W2", "S13", "S2")


def test_convert_delegates_upstream_off_ppu(monkeypatch):
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(is_ppu=lambda: False)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    convert = function(
        REGISTRY,
        "convert_to_w4a8_int8_moe_format",
        {
            "torch": SimpleNamespace(empty=lambda n: f"EMPTY{n}"),
            "_upstream_convert_to_w4a8_int8_moe_format": lambda *a, **k: "UPSTREAM",
        },
    )
    assert convert("W13", "W2", "S13", "S2", -1) == "UPSTREAM"


def test_make_kernel_passes_batched_extra_args():
    node = definition(ROOT / REGISTRY, "make_w4a8_int8_moe_kernel")
    source = ast.unparse(node)
    assert "max_num_tokens" in source
    assert "num_dispatchers" in source
    assert "BatchedExperts" in source
    # Must not reuse upstream's monolithic-only restriction.
    assert "only supports monolithic" not in source


def test_make_kernel_resolves_factory_lazily():
    node = definition(ROOT / REGISTRY, "make_w4a8_int8_moe_kernel")
    imports = [
        n for n in ast.walk(node)
        if isinstance(n, ast.ImportFrom) and n.module == "vllm.model_executor.layers.fused_moe.all2all_utils"
    ]
    assert imports, "factory must be imported inside the function, not captured"


# ---------------------------------------------------------------------------
# P1-2 CLI literal / P1-3 alias registration
# ---------------------------------------------------------------------------


def _literal_tuple() -> tuple[str, ...]:
    node = ast.parse((ROOT / LITERAL).read_text())
    for part in node.body:
        if isinstance(part, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PPU_MOE_BACKENDS" for t in part.targets
        ):
            return tuple(ast.literal_eval(part.value))
    raise AssertionError("PPU_MOE_BACKENDS not found")


def test_cli_literal_accepts_new_backend():
    names = _literal_tuple()
    assert NEW_BACKEND in names
    # Must not reuse the MXFP4 block-format name.
    assert "ppu_deep_gemm_w4a8" in names
    assert names.count(NEW_BACKEND) == 1


def test_cli_literal_keeps_existing_names():
    names = _literal_tuple()
    for existing in ("ppu_deep_gemm", "ppu_acext", "ppu_deep_gemm_w4a16", "ppu_deep_gemm_w4a8"):
        assert existing in names


def _consumers() -> dict:
    node = ast.parse((ROOT / CONSUMERS).read_text())
    for part in node.body:
        if isinstance(part, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_ORACLE_ALIAS_CONSUMERS" for t in part.targets
        ):
            return ast.literal_eval(part.value)
    raise AssertionError("_ORACLE_ALIAS_CONSUMERS not found")


def test_alias_consumers_register_w4a8_int8():
    table = _consumers()
    assert "w4a8_int8" in table
    entry = table["w4a8_int8"]
    for alias in (
        "convert_to_w4a8_int8_moe_format",
        "make_w4a8_int8_moe_kernel",
        "select_w4a8_int8_moe_backend",
    ):
        assert alias in entry
        assert entry[alias], f"{alias} must list the CT consumer"


def test_alias_consumers_use_real_module_paths():
    """The draft pointed at an upstream path that does not exist."""
    entry = _consumers()["w4a8_int8"]
    for consumers in entry.values():
        for path in consumers:
            assert path.startswith("vllm.model_executor.layers.quantization.compressed_tensors")
            assert "mixed_precision_w4" not in path


def test_alias_consumers_keep_other_backends():
    table = _consumers()
    for existing in ("unquantized", "fp8", "int8", "mxfp4", "int_wna16"):
        assert existing in table


def test_every_registered_backend_module_exists():
    for name in _consumers():
        assert (ROOT / "vllm_sail/registry/moe_backends" / f"{name}.py").is_file(), (
            f"register() imports a module for every key; {name}.py is missing"
        )


# ---------------------------------------------------------------------------
# PPU-only: declared, never claimed as passing on CPU
# ---------------------------------------------------------------------------


@pytest.mark.ppu
def test_acext_w4a8_batched_numeric_matches_dense_reference():
    """Device oracle: batched reshape + synthetic top-k=1 must match dense MoE.

    Requires acext, a real PPU and the DeepEP LL dispatch layout. Skipped on CPU.
    """
    pytest.skip("requires PPU + acext + DeepEP LL; run on na131t-swu110")


@pytest.mark.ppu
def test_deepep_ll_combine_applies_weights_exactly_once():
    """Device oracle: ones fed to acext + Delegate combine must equal legacy."""
    pytest.skip("requires PPU + DeepEP LL; run on na131t-swu110")