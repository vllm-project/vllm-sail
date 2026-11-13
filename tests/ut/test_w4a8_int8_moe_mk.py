# SPDX-License-Identifier: Apache-2.0
"""W4A8-INT8 (MixedPrecisionW4) modular-kernel wiring, without torch or a device.

Default backend is DeepGemm native masked INT4xINT8 (PPUDeepGemmExpertsW4A8Int8);
the original acext shell is retained as PPUAcextW4A8Int8Fallback (回退用, not
default). Locks:
  1. the legacy non-MK branch still returns None and keeps the 16-arg acext call;
  2. MK assembly happens only under DeepEP low-latency + batched activations;
  3. the fallback acext Q_type comes from explicit constants, never quant flags;
  4. weight parameter names match the names the modular kernel reads;
  5. (B2/D6) portable INT4 pack round-trip, construction shape asserts, native
     masked flow, allow-defer guard, fallback retention, per-token INT8 quant.

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
# 3. Q_type is an explicit constant in the acext FALLBACK experts
# ---------------------------------------------------------------------------


def test_fallback_experts_qtype_is_literal_constants():
    node = definition(ROOT / EXPERTS, "PPUAcextW4A8Int8Fallback.apply")
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


def test_fallback_experts_never_reads_quant_config_flags_for_qtype():
    node = definition(ROOT / EXPERTS, "PPUAcextW4A8Int8Fallback.apply")
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


# ---------------------------------------------------------------------------
# B2 / D6: native masked INT4xINT8, portable pack, allow-defer, fallback
# ---------------------------------------------------------------------------

DEEPEP_LL = "vllm_sail/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py"
NATIVE = "PPUDeepGemmExpertsW4A8Int8"
FALLBACK = "PPUAcextW4A8Int8Fallback"


def _pack():
    return function(EXPERTS, "_w4a8_pack_int4_reference", {})


def _unpack():
    return function(EXPERTS, "_w4a8_unpack_int4_reference", {})


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 123])
def test_pack_int4_roundtrip_random_groups(seed):
    """int8[-8,7] -> portable uint8 -> inverse == original (DeepGemm convention)."""
    import random

    rng = random.Random(seed)
    pack, unpack = _pack(), _unpack()
    for _ in range(8):
        n = rng.choice([2, 4, 16, 32, 64])
        values = [rng.randint(-8, 7) for _ in range(n)]
        packed = pack(values)
        assert len(packed) == n // 2
        assert all(0 <= b <= 255 for b in packed)
        assert unpack(packed) == values


def test_pack_int4_boundaries_minus8_and_7():
    pack, unpack = _pack(), _unpack()
    for v in (-8, -1, 0, 7):
        assert unpack(pack([v, v])) == [v, v]
    # signed two's-complement nibbles: -8 -> 0x8, -1 -> 0xF, 7 -> 0x7
    assert pack([-8, -8]) == [0x88]
    assert pack([-1, -1]) == [0xFF]
    assert pack([7, 7]) == [0x77]
    assert pack([0, 0]) == [0x00]


def test_pack_int4_even_low_odd_high():
    """Even-K element occupies the low nibble, odd-K the high nibble."""
    pack = _pack()
    # values[0]=1 (even, low), values[1]=2 (odd, high) -> 0x21
    assert pack([1, 2]) == [0x21]
    assert pack([3, -8]) == [0x83]  # low=3, high=-8&0xF=8
    assert _unpack()([0x21]) == [1, 2]


def test_pack_int4_rejects_out_of_range():
    pack = _pack()
    for bad in ([8, 0], [0, -9], [16, 1]):
        with pytest.raises(AssertionError):
            pack(bad)


def test_pack_int4_rejects_odd_length():
    with pytest.raises(AssertionError):
        _pack()([1, 2, 3])


def test_model_dims_divisibility_qwen38_flash_next():
    """Actual config.json dims satisfy gemm.hpp:604 (k%32) / :607 (n%8).

    hidden=2560, moe_intermediate=640. gemm1 K=hidden N=2*inter;
    gemm2 K=inter N=hidden. EP8 -> 64 local routed experts (group count).
    """
    hidden, inter = 2560, 640
    assert hidden % 32 == 0 and (2 * inter) % 8 == 0  # gemm1
    assert inter % 32 == 0 and hidden % 8 == 0  # gemm2
    assert 512 % 8 == 0 and 512 // 8 == 64  # global routed -> local under EP8


@pytest.mark.parametrize("hidden,inter", [(2561, 640), (2560, 641), (2560, 636)])
def test_bad_dims_violate_constraints(hidden, inter):
    """If real dims ever fail this, stop and report — never pad to fit (D6)."""
    ok = hidden % 32 == 0 and inter % 32 == 0 and (2 * inter) % 8 == 0 and hidden % 8 == 0
    assert not ok


def test_native_init_asserts_shape_constraints():
    node = definition(ROOT / EXPERTS, f"{NATIVE}.__init__")
    src = ast.unparse(node)
    assert "hidden % 32 == 0" in src
    assert "inter % 32 == 0" in src
    # ast.unparse drops redundant parens: (2 * inter) % 8 -> 2 * inter % 8
    assert "inter % 8 == 0" in src
    assert "hidden % 8 == 0" in src
    assert "gemm.hpp:604" in src and "gemm.hpp:607" in src


def test_fallback_class_retained_and_not_default():
    tree = ast.parse((ROOT / EXPERTS).read_text())
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert FALLBACK in classes, "acext fallback must be retained, not deleted"
    assert NATIVE in classes
    # fallback docstring marks it 回退用
    fb = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == FALLBACK)
    assert "回退用" in ast.get_docstring(fb)
    # default registry resolves the native class, not the fallback
    reg = definition(ROOT / REGISTRY, "backend_to_kernel_cls")
    rsrc = ast.unparse(reg)
    assert NATIVE in rsrc
    assert FALLBACK not in rsrc


def _defer_guard_node(method: str):
    node = definition(ROOT / DEEPEP_LL, method)
    guards = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.If)
        and any(isinstance(s, ast.Raise) for s in n.body)
    ]
    assert guards, f"{method} must keep a defer rejection guard"
    return guards[0].test


def _interp_bool(node, ns):
    """Allowlisted interpreter for the guard's boolean expression (no eval).

    Handles only BoolOp(and/or), Name, Attribute and Constant — the exact
    expression language of `defer_input_quant and (self.use_*_dispatch or ...)`.
    """
    if isinstance(node, ast.BoolOp):
        vals = [_interp_bool(v, ns) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Name):
        return ns[node.id]
    if isinstance(node, ast.Attribute):
        return getattr(_interp_bool(node.value, ns), node.attr)
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected guard expression node: {ast.dump(node)}")


@pytest.mark.parametrize("method", ["DeepEPLLPrepareAndFinalize.prepare_async", "DeepEPLLPrepareAndFinalize.prepare"])
@pytest.mark.parametrize(
    "defer,fp8,int8,mxfp4,should_raise",
    [
        (True, True, False, False, True),
        (True, False, True, False, True),
        (True, False, False, True, True),
        (True, False, False, False, False),  # raw bf16 dispatch: accept
        (False, True, True, True, False),
    ],
)
def test_allow_defer_guard_only_rejects_quantized_dispatch(method, defer, fp8, int8, mxfp4, should_raise):
    guard = _defer_guard_node(method)
    ns = {
        "defer_input_quant": defer,
        "self": SimpleNamespace(
            use_fp8_dispatch=fp8, use_int8_dispatch=int8, use_mxfp4_dispatch=mxfp4
        ),
    }
    assert bool(_interp_bool(guard, ns)) is should_raise


def test_prepare_forwards_defer_to_async():
    node = definition(ROOT / DEEPEP_LL, "DeepEPLLPrepareAndFinalize.prepare")
    calls = [
        c
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "prepare_async"
    ]
    assert calls
    assert any(kw.arg == "defer_input_quant" for kw in calls[0].keywords)


def test_native_apply_uses_masked_int4_flow():
    node = definition(ROOT / EXPERTS, f"{NATIVE}.apply")
    called = [
        c.func.id
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    ]
    assert called.count("int8_int4_m_grouped_gemm_nt_masked") == 2  # gemm1 + gemm2
    assert "per_token_quant_int8" in called  # internal gemm1-input quant
    assert "int8_persistent_masked_m_silu_mul_quant" in called  # gemm2 pre-step
    # no acext wrapper in the native path
    assert "fusedmoe_wrapper" not in called


def test_native_expects_unquantized_inputs_true():
    node = definition(ROOT / EXPERTS, f"{NATIVE}.expects_unquantized_inputs")
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert len(returns) == 1 and returns[0].value.value is True


def test_native_process_weights_uint8_view_and_preprocess():
    node = definition(ROOT / EXPERTS, f"{NATIVE}.process_weights_after_loading")
    src = ast.unparse(node)
    assert "view(torch.uint8)" in src
    assert "preprocess_a8w4_weight" in src
    assert "replace_parameter" in src
    assert "torch.int8" in src  # asserts checkpoint dtype
    # B2-fix: T^-1 must run BEFORE preprocess (checkpoint is already T-order).
    assert "_w4a8_undo_checkpoint_T" in src
    assert src.index("_w4a8_undo_checkpoint_T") < src.index("preprocess_a8w4_weight")
    assert ".contiguous()" in src  # preprocess requires contiguous uint8


def test_native_process_weights_no_scale_replacement():
    """Scales stay captured by quant_config; replacing them would go stale."""
    node = definition(ROOT / EXPERTS, f"{NATIVE}.process_weights_after_loading")
    # replace_parameter(layer, name, ...) uses the loop variable; lock the
    # loop's iterand tuple to exactly the two weight names (no scale).
    loops = [n for n in ast.walk(node) if isinstance(n, ast.For)]
    assert len(loops) == 1
    iterated = ast.literal_eval(loops[0].iter)
    assert iterated == ("w13_weight", "w2_weight")
    assert not any("scale" in n for n in iterated)
    # the replace_parameter call passes the loop var, not a scale constant
    calls = [
        c for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id == "replace_parameter"
    ]
    assert len(calls) == 1
    assert isinstance(calls[0].args[1], ast.Name) and calls[0].args[1].id == "name"


# ---------------------------------------------------------------------------
# B2-fix: checkpoint T-order -> T^-1 -> portable -> preprocess (A1b cos 0.9998)
# CPU-runnable, no torch/numpy/device. Pure-Python T / T^-1 references + AST.
# ---------------------------------------------------------------------------


def _T():
    return function(EXPERTS, "_w4a8_checkpoint_T_reference", {})


def _Tinv():
    return function(EXPERTS, "_w4a8_undo_checkpoint_T_reference", {})


# Tinv.first32 (nibble-level gather) from the A1b report, asserted value-by-value.
_TINV_FIRST32 = [
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31,
]
# Byte-level gather table (16-byte K32 group) = first32 folded by pairs / 2.
_TINV_BYTES = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]


def test_Tinv_first32_table_matches_report():
    """req2: Tinv.first32 asserted value-by-value; byte fold == _TINV_BYTES."""
    folded = [_TINV_FIRST32[2 * i] // 2 for i in range(16)]
    assert folded == _TINV_BYTES
    # each byte's two nibbles are consecutive (low, high)
    for i in range(16):
        assert _TINV_FIRST32[2 * i + 1] == _TINV_FIRST32[2 * i] + 1
    # the device T^-1 gather applied to range(16) reproduces the byte table
    assert _Tinv()(list(range(16))) == _TINV_BYTES


def test_T_is_bijection_not_self_inverse():
    """T is a permutation of 16 bytes; T has order 3, T^-1 != T (A1b)."""
    T, Tinv = _T(), _Tinv()
    fwd = T(list(range(16)))
    assert sorted(fwd) == list(range(16))  # bijection
    inv = Tinv(list(range(16)))
    assert sorted(inv) == list(range(16))
    assert fwd != inv, "T must NOT be self-inverse"
    assert T(fwd) != list(range(16)), "T∘T != I (order 3, not 2)"
    assert T(T(fwd)) == list(range(16)), "T^3 == I"


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 123])
def test_Tinv_composes_to_identity(seed):
    """req2: T^-1∘T == I and T∘T^-1 == I on random whole-group byte arrays."""
    import random

    rng = random.Random(seed)
    T, Tinv = _T(), _Tinv()
    for groups in (1, 2, 20, 80):  # 20=K640/32, 80=K2560/32
        n = groups * 16
        x = [rng.randint(0, 255) for _ in range(n)]
        assert Tinv(T(x)) == x
        assert T(Tinv(x)) == x


def test_checkpoint_T_then_undo_then_preprocessT_equals_plain_preprocessT():
    """req1: checkpoint(T-order) -> T^-1 -> preprocess-T == plain -> preprocess-T.

    preprocess = XOR∘R∘T (common_preprocess.hpp:262/277-282/284). R and XOR are
    downstream bijections, so byte-for-byte equality of the whole preprocess
    reduces to equality of its leading T step's INPUT. Locking
    preprocess_T(T^-1(checkpoint)) == preprocess_T(portable) with
    checkpoint = T(portable) proves T^-1 cancels preprocess's internal T.
    The full device preprocess (R+XOR) equality is the @ppu oracle below.
    """
    import random

    T, Tinv = _T(), _Tinv()
    rng = random.Random(2026)
    for groups in (20, 80):
        portable = [rng.randint(0, 255) for _ in range(groups * 16)]
        checkpoint = T(portable)  # checkpoint落盘已是 T 序
        # path A: checkpoint -> T^-1 -> preprocess's T step
        a = T(Tinv(checkpoint))
        # path B: portable plain -> preprocess's T step
        b = T(portable)
        assert a == b, "T^-1 must cancel preprocess internal T byte-for-byte"
        assert Tinv(checkpoint) == portable


@pytest.mark.parametrize("k_half,groups", [(1280, 80), (320, 20)])
def test_Tinv_covers_w13_and_w2_shapes(k_half, groups):
    """req3: w13 K=2560 (K/2=1280, 80 groups) and w2 K=640 (K/2=320, 20 groups)."""
    import random

    assert k_half % 16 == 0 and k_half // 16 == groups
    T, Tinv = _T(), _Tinv()
    rng = random.Random(k_half)
    x = [rng.randint(0, 255) for _ in range(k_half)]
    assert Tinv(T(x)) == x
    assert len(Tinv(x)) == k_half


def test_Tinv_rejects_non_multiple_of_16():
    for bad in (1, 15, 17, 31):
        with pytest.raises(AssertionError):
            _Tinv()([0] * bad)
        with pytest.raises(AssertionError):
            _T()([0] * bad)


def test_device_undo_T_uses_transpose_recipe():
    """AST lock: device T^-1 = reshape [...,K/32,4,2,2] -> transpose(-3,-2)."""
    node = definition(ROOT / EXPERTS, "_w4a8_undo_checkpoint_T")
    src = ast.unparse(node)
    assert "reshape" in src and "transpose(-3, -2)" in src
    assert "4, 2, 2" in src
    assert "flatten(-3)" in src and "contiguous()" in src
    assert "k_half % 16 == 0" in src


def test_legacy_and_fallback_have_no_byte_transform():
    """R5: legacy + acext fallback must stay byte-level unchanged (no T^-1)."""
    legacy = definition(ROOT / MIXED_W4, "W4AInt8MoEMethod.process_weights_after_loading")
    first = next(n for n in legacy.body if isinstance(n, ast.If))
    legacy_branch = ast.unparse(first)
    assert "_w4a8_undo_checkpoint_T" not in legacy_branch
    assert "preprocess_a8w4_weight" not in legacy_branch
    fb = definition(ROOT / EXPERTS, f"{FALLBACK}.process_weights_after_loading")
    fbsrc = ast.unparse(fb)
    assert "_w4a8_undo_checkpoint_T" not in fbsrc
    assert "preprocess_a8w4_weight" not in fbsrc


def test_undo_T_only_in_native_mk_path():
    """The T^-1 transform lives only in the native MK experts, gated by MK."""
    tree = ast.parse((ROOT / EXPERTS).read_text())
    callers = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
        for m in n.body
        if isinstance(m, ast.FunctionDef)
        and "_w4a8_undo_checkpoint_T(" in ast.unparse(m)
    ]
    assert callers == [NATIVE], f"T^-1 must only be called by {NATIVE}, got {callers}"


@pytest.mark.ppu
def test_device_undo_T_matches_reference_and_preprocess_cancels():
    """Device oracle: torch T^-1 == pure-Python reference; full preprocess
    (T+R+XOR) on T^-1(checkpoint) == preprocess(portable), byte-for-byte.
    Requires PPU 810E + DeepGemm. Skipped on CPU; a skip is not a pass."""
    pytest.skip("requires PPU 810E + DeepGemm preprocess_a8w4_weight; run on card")


def test_real_checkpoint_slice_shape_integrity():
    """req4: real checkpoint slice (single layer/expert) shape/byte integrity.

    Reads a fixture copied from /ppusw READ-ONLY into our writable dir; asserts
    the uint8 last dim is divisible by 16 (whole K32 groups, no remainder).
    Shape/byte integrity ONLY — numeric correctness is the challenger's on-card
    A1b/probe job. Skips when the fixture is absent (NOT claimed as a pass);
    populate W4A8_CKPT_FIXTURE with the copied slice path to enable.
    """
    import os

    path = os.environ.get("W4A8_CKPT_FIXTURE")
    if not path or not os.path.isfile(path):
        pytest.skip(
            "real checkpoint fixture not populated; copy a single-layer/expert "
            "slice from /ppusw (read-only) to a writable dir and set "
            "W4A8_CKPT_FIXTURE. Shape/integrity only, no numeric claim."
        )
    import json

    meta = json.loads(open(path).read())
    for name in ("w13_weight", "w2_weight"):
        shape = meta[name]["shape"]
        k_half = shape[-1]
        assert k_half % 16 == 0, f"{name} K/2={k_half} not divisible by 16"
        assert meta[name]["dtype"] in ("int8", "uint8")


# ---------------------------------------------------------------------------
# gate_first concat order, locked on A1c real-checkpoint evidence (challenger):
#   layer0  gate_first cos +0.999841 norm 0.9995 g_l2 0.0178  -> pass
#   layer20 gate_first cos +0.999827 norm 1.0002 g_l2 0.0186  -> pass
#   up_first cos 0.9963 / 0.9967, g_l2 0.086 / 0.082          -> rejected
# w13 is [E, 2I, K/2] with gate in the FIRST I rows, up in the second I; the
# silu_mul epilogue applies SiLU to the first group_size columns. CPU-only.
# ---------------------------------------------------------------------------


def _silu(x):
    import math

    return x / (1.0 + math.exp(-x))


def _silu_mul_gate_first(row, group_size):
    """SiLU(first half) * second half — the confirmed checkpoint convention."""
    return [_silu(row[i]) * row[group_size + i] for i in range(group_size)]


def _silu_mul_up_first(row, group_size):
    """The rejected alternative: SiLU(second half) * first half."""
    return [_silu(row[group_size + i]) * row[i] for i in range(group_size)]


def test_gate_first_and_up_first_are_discriminating():
    """The two orders must differ, so the gate_first lock is not vacuous."""
    row = [1.0, 2.0, 3.0, 4.0]
    g, u = _silu_mul_gate_first(row, 2), _silu_mul_up_first(row, 2)
    assert g != u
    # gate_first: SiLU(1)*3, SiLU(2)*4
    assert abs(g[0] - _silu(1.0) * 3.0) < 1e-12
    assert abs(g[1] - _silu(2.0) * 4.0) < 1e-12
    # up_first would pair SiLU(3)*1, SiLU(4)*2 — wrong per A1c
    assert abs(u[0] - _silu(3.0) * 1.0) < 1e-12


def test_gate_first_matches_a1c_evidence_over_up_first():
    """A1c: gate_first cos ~0.9998 vs up_first ~0.9963 on real weights.

    Reproduces the ORDERING of the two candidates' agreement with a gate_first
    ground truth on synthetic rows: gate_first is exact (cos 1), up_first is
    not. The real-weight cos/g_l2 numbers are the challenger's on-card oracle.
    """
    import math
    import random

    rng = random.Random(31)
    I = 8
    rows = [[rng.uniform(-3, 3) for _ in range(2 * I)] for _ in range(16)]

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    gate_cos, up_cos = [], []
    for row in rows:
        truth = _silu_mul_gate_first(row, I)
        gate_cos.append(cos(truth, _silu_mul_gate_first(row, I)))
        up_cos.append(cos(truth, _silu_mul_up_first(row, I)))
    assert all(abs(c - 1.0) < 1e-12 for c in gate_cos)
    assert max(up_cos) < 0.999, "up_first must be clearly worse than gate_first"


def test_native_silu_mul_group_size_is_half_width():
    """group_size = N//2 pins SiLU to the first half == gate (gate_first)."""
    node = definition(ROOT / EXPERTS, f"{NATIVE}.apply")
    src = ast.unparse(node)
    assert "int8_persistent_masked_m_silu_mul_quant" in src
    assert "group_size=workspace1.shape[-1] // 2" in src


def test_native_path_never_swaps_gate_up_halves():
    """No half reordering anywhere in the native weight/apply path."""
    forbidden = {"flip", "fliplr", "flipud", "roll", "reverse", "argsort"}
    for method in ("apply", "process_weights_after_loading"):
        node = definition(ROOT / EXPERTS, f"{NATIVE}.{method}")
        used = {
            a.attr
            for a in ast.walk(node)
            if isinstance(a, ast.Attribute) and a.attr in forbidden
        }
        assert not used, f"{method} reorders gate/up halves: {used}"


def test_w13_is_fused_gate_up_with_n_twice_intermediate():
    """create_weights builds w13 [E, 2I, K/2] (gate+up fused, gate first)."""
    node = definition(ROOT / MIXED_W4, "W4AInt8MoEMethod.create_weights")
    src = ast.unparse(node)
    assert "2 * intermediate_size_per_partition" in src
    assert "hidden_size // self.quant_config.pack_factor" in src
    # w2 stays [E, hidden, I/2] — not fused, no factor 2 on hidden
    assert "intermediate_size_per_partition // self.quant_config.pack_factor" in src


def test_tinv_does_not_touch_gate_up_row_order():
    """T^-1 permutes only the K axis (bytes within K32 groups), never N rows.

    gate_first is an N-axis (row) convention; T/T^-1 are K-axis byte
    permutations, so the two fixes are orthogonal and must not interfere.
    """
    node = definition(ROOT / EXPERTS, "_w4a8_undo_checkpoint_T")
    src = ast.unparse(node)
    # reshape/transpose act on trailing dims only; lead dims (E, N) preserved
    assert "*lead" in src
    assert "transpose(-3, -2)" in src
    T, Tinv = _T(), _Tinv()
    # a full w13-like row set: N rows each K/2 bytes; row order unchanged
    import random

    rng = random.Random(5)
    N, k_half = 6, 32
    rows = [[rng.randint(0, 255) for _ in range(k_half)] for _ in range(N)]
    flat = [b for row in rows for b in row]
    out = Tinv(flat)
    recovered = [out[i * k_half:(i + 1) * k_half] for i in range(N)]
    # each row is independently permuted; row identity/order preserved
    assert len(recovered) == N
    for i in range(N):
        assert sorted(recovered[i]) == sorted(rows[i])
        assert T(recovered[i]) == rows[i]


def test_native_moe_problem_size_from_weight_shard():
    node = definition(ROOT / EXPERTS, f"{NATIVE}.moe_problem_size")
    src = ast.unparse(node)
    assert "E, N, k_half = w1.size()" in src  # group count = local routed shard
    assert "K = k_half * 2" in src
    assert "a1.size(1)" in src and "a1.dim() == 3" in src


def test_native_device_gate_is_810e():
    node = definition(ROOT / EXPERTS, f"{NATIVE}._supports_current_device")
    src = ast.unparse(node)
    assert "is_deep_gemm_supported()" in src
    assert "is_device_capability((8, 0))" in src


def test_native_apply_has_no_host_sync():
    node = definition(ROOT / EXPERTS, f"{NATIVE}.apply")
    synced = [
        a.attr
        for a in ast.walk(node)
        if isinstance(a, ast.Attribute) and a.attr in {"item", "tolist", "cpu"}
    ]
    assert not synced, f"host synchronisation found: {synced}"


def test_legacy_relabel_stays_in_non_mk_branch():
    """R5: the legacy acext view relabel must remain under `not _mk_enabled()`."""
    node = definition(ROOT / MIXED_W4, "W4AInt8MoEMethod.process_weights_after_loading")
    # body[0] may be the docstring; the first If is the `not _mk_enabled()` gate
    first = next(n for n in node.body if isinstance(n, ast.If))
    assert "not self._mk_enabled()" in ast.unparse(first.test)
    branch_src = ast.unparse(first)
    assert "pack_factor" in branch_src and "return" in branch_src
    # the MK path (after the branch) must NOT relabel; it passes through
    idx = node.body.index(first)
    rest_src = ast.unparse(ast.Module(body=node.body[idx + 1:], type_ignores=[]))
    assert "convert_to_w4a8_int8_moe_format" in rest_src
    assert "pack_factor" not in rest_src


def test_per_token_int8_scale_shape_reference():
    """Contract of per_token_quant_int8 on a batched [E,T,K] buffer.

    Upstream vllm int8_utils.per_token_quant_int8 reshapes scales to
    (*shape[:-1], 1) == [E,T,1] and uses scale=amax/127 (matches DeepGemm
    per_token_cast_to_int8 and the masked lhs_scales [num_groups,m,1]).
    Pure-Python reference; the triton kernel itself is a PPU/device oracle.
    """
    E, T, K = 4, 7, 16
    shape = (E, T, K)
    scale_shape = (*shape[:-1], 1)
    assert scale_shape == (E, T, 1)
    # amax/127 dequant convention: x ~= x_q * scale
    row = [0.0] * K
    row[3] = 2.54  # amax
    amax = max(abs(v) for v in row) or 1e-10
    scale = amax / 127
    xq = round(2.54 * (127 / amax))
    assert xq == 127
    assert abs(xq * scale - 2.54) < 1e-6


@pytest.mark.ppu
def test_native_masked_int4_numeric_matches_dense_reference():
    """Device oracle: native masked INT4xINT8 must match a dense dequant GEMM.

    Requires PPU 810E + DeepGemm + the checkpoint pack convention. Skipped on
    CPU; a skip is not evidence of a pass. Also verifies the checkpoint nibble
    convention equals DeepGemm pack_standard_int4_weight (B3 must confirm).
    """
    pytest.skip("requires PPU 810E + DeepGemm native masked; run on na131t-swu110")


# ---------------------------------------------------------------------------
# R5 regression guard: legacy must NOT create MK-only attributes
# (upstream FusedMoEMethodBase.is_monolithic uses hasattr(self, "experts_cls"))
# ---------------------------------------------------------------------------


def _is_monolithic_contract(obj):
    """Replica of upstream fused_moe_method_base.py:135-141 is_monolithic.

    Kept here so the legacy contract is testable without importing vLLM. The
    real source is AST-checked separately to ensure the contract still holds.
    """
    if obj.moe_kernel is None:
        if hasattr(obj, "experts_cls"):
            return obj.experts_cls.is_monolithic()
        return False
    return obj.moe_kernel.is_monolithic


def _run_init(mk_enabled: bool):
    init = function(
        MIXED_W4,
        "W4AInt8MoEMethod.__init__",
        {"super": lambda: SimpleNamespace(__init__=lambda moe: None)},
    )
    moe = SimpleNamespace(
        ep_size=8, ep_rank=0,
        moe_parallel_config=_parallel_config(mk_enabled, mk_enabled),
    )
    obj = SimpleNamespace(moe=moe, _mk_enabled=lambda: mk_enabled, moe_kernel=None)
    init(obj, SimpleNamespace(), moe)
    return obj


def test_legacy_init_does_not_create_experts_cls_or_backend():
    obj = _run_init(mk_enabled=False)
    assert not hasattr(obj, "experts_cls"), (
        "legacy must not create experts_cls: upstream is_monolithic does "
        "hasattr(self,'experts_cls') and would call None.is_monolithic()"
    )
    assert not hasattr(obj, "backend")


def test_legacy_is_monolithic_returns_false_without_raising():
    obj = _run_init(mk_enabled=False)
    assert _is_monolithic_contract(obj) is False


def test_experts_cls_none_would_crash_is_monolithic():
    """Demonstrates the regression the fix prevents (None.is_monolithic())."""
    obj = SimpleNamespace(moe_kernel=None, experts_cls=None)
    with pytest.raises(AttributeError):
        _is_monolithic_contract(obj)


def _register_chain(monkeypatch, dotted: str):
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))


def test_mk_init_creates_experts_cls(monkeypatch):
    oracle_pkg = "vllm.model_executor.layers.fused_moe.oracle"
    _register_chain(monkeypatch, oracle_pkg)
    oracle = ModuleType(f"{oracle_pkg}.w4a8_int8")
    oracle.select_w4a8_int8_moe_backend = lambda **kw: ("BATCHED", "EXPERTS_CLS")
    monkeypatch.setitem(sys.modules, f"{oracle_pkg}.w4a8_int8", oracle)
    setattr(sys.modules[oracle_pkg], "w4a8_int8", oracle)

    qu_name = "vllm.model_executor.layers.quantization.utils.quant_utils"
    _register_chain(monkeypatch, qu_name)
    qu = sys.modules[qu_name]
    qu.kInt4W4A8StaticChannelSym = "W4"
    qu.kInt8DynamicTokenSym = "A8"

    obj = _run_init(mk_enabled=True)
    assert obj.experts_cls == "EXPERTS_CLS"
    assert obj.backend == "BATCHED"


def test_init_assigns_mk_attrs_only_inside_mk_branch():
    node = definition(ROOT / MIXED_W4, "W4AInt8MoEMethod.__init__")
    # direct children of __init__ must not assign self.experts_cls / self.backend
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                    assert t.attr not in {"experts_cls", "backend"}, (
                        f"self.{t.attr} assigned unconditionally in __init__"
                    )
    # the assignments must live under an `if self._mk_enabled():` block
    mk_ifs = [
        n for n in node.body
        if isinstance(n, ast.If) and "_mk_enabled" in ast.unparse(n.test)
    ]
    assert mk_ifs, "__init__ must gate MK attrs behind if self._mk_enabled()"
    inner = ast.unparse(mk_ifs[0])
    assert "self.experts_cls" in inner and "self.backend" in inner


def test_upstream_is_monolithic_still_uses_hasattr():
    """Guard the upstream contract this fix relies on (skip if no vLLM source)."""
    import os

    root = os.environ.get("VLLM_SOURCE_ROOT")
    if not root:
        pytest.skip("set VLLM_SOURCE_ROOT to check upstream is_monolithic contract")
    from pathlib import Path

    p = Path(root) / "vllm/model_executor/layers/fused_moe/fused_moe_method_base.py"
    if not p.is_file():
        pytest.skip(f"upstream source not found at {p}")
    node = definition(p, "FusedMoEMethodBase.is_monolithic")
    src = ast.unparse(node)
    assert "hasattr(self, 'experts_cls')" in src or 'hasattr(self, "experts_cls")' in src