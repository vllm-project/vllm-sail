# SPDX-License-Identifier: Apache-2.0
"""u258 P1: w4a8_int8 registration + layout-aware weight path, torch-free.

Locks:
 1. W4A8Int8Config exists, get_name()=="w4a8_int8", weight_layout
    "deepgemm_plain"; MixedPrecisionW4Config keeps "acext_t_order"; the
    quant_config registry registers BOTH names.
 2. Convention distinctness: the three candidate decodings of one byte block
    (plain signed-int4, T^-1 then signed-int4, unsigned-offset) must not be
    equal, so a wrong layout choice changes values instead of being a no-op.
    The real-checkpoint cross-check that established which convention each
    checkpoint uses (2048-byte slices of layers.3 experts.0 gate_proj/down_proj
    from two sibling W4A8-INT8 checkpoints, fixture sha256
    2e1051386aefa9f3a2750ba9ec3da808bf3a2a0baaaebf0d398f91ad6654c3fd) is kept in
    the internal evidence index and is deliberately NOT vendored here: vendored
    weight slices go stale whenever a checkpoint is regenerated.
 3. The plain preprocess branch skips _w4a8_undo_checkpoint_T (AST lock).
 4. create_weights container dtype: uint8 for plain, int8 for t_order;
    shapes and per-channel fp32 scales unchanged.
 5. make_w4a8_int8_moe_kernel forwards weight_layout to the experts ctor.
 6. _mk_enabled: plain always MK (no legacy acext representation); t_order
    still requires DeepEP LL + batched.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from tests.support.source import ROOT, definition, function

MIXED_W4 = "vllm_sail/model_executor/layers/quantization/mixed_precision_w4.py"
W4A8_CFG = "vllm_sail/model_executor/layers/quantization/w4a8_int8.py"
QUANT_REG = "vllm_sail/registry/quant_config/__init__.py"
EXPERTS = "vllm_sail/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
REGISTRY = "vllm_sail/registry/moe_backends/w4a8_int8.py"


def _unpack(packed: list[int]) -> list[int]:
    fn = function(EXPERTS, "_w4a8_unpack_int4_reference", {})
    return fn(packed)


def _undo_t(packed: list[int]) -> list[int]:
    fn = function(EXPERTS, "_w4a8_undo_checkpoint_T_reference", {})
    return fn(packed)


def _class_attr(relative: str, cls: str, attr: str):
    node = definition(ROOT / relative, cls)
    for child in node.body:
        if isinstance(child, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == attr for t in child.targets
        ):
            return ast.literal_eval(child.value)
    raise AssertionError(f"{cls}.{attr} not found in {relative}")


# ---------------------------------------------------------------------------
# 1. registration entry (G1)
# ---------------------------------------------------------------------------


def test_w4a8_int8_config_name_and_layout():
    get_name = function(W4A8_CFG, "W4A8Int8Config.get_name", {})
    assert get_name(None) == "w4a8_int8"
    assert _class_attr(W4A8_CFG, "W4A8Int8Config", "weight_layout") == (
        "deepgemm_plain"
    )


def test_mixed_precision_w4_keeps_t_order_layout():
    assert _class_attr(MIXED_W4, "MixedPrecisionW4Config", "weight_layout") == (
        "acext_t_order"
    )


def test_w4a8_config_subclasses_mixed_precision_w4():
    node = definition(ROOT / W4A8_CFG, "W4A8Int8Config")
    bases = [ast.unparse(b) for b in node.bases]
    assert bases == ["MixedPrecisionW4Config"]


def test_quant_registry_registers_both_names():
    node = definition(ROOT / QUANT_REG, "register")
    loops = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Tuple)
        and {ast.unparse(e) for e in n.iter.elts}
        == {"MixedPrecisionW4Config", "W4A8Int8Config"}
    ]
    assert loops, "register() must loop over both config classes"
    src = ast.unparse(node)
    assert "register_quantization_config(name)(config_cls)" in src
    assert "from vllm_sail.model_executor.layers.quantization.w4a8_int8 import" in src


# ---------------------------------------------------------------------------
# 2. layout conventions are distinguishable (algebraic lock, no vendored data)
# ---------------------------------------------------------------------------


def _deterministic_block() -> list[int]:
    """Structured 2048-byte stand-in for an expert weight shard.

    Deliberately not constant and not random: varies within and across 16-byte
    groups so that a K32-group permutation is observable, while staying
    reproducible without vendoring checkpoint bytes.
    """
    return list(bytes((i * 37 + (i // 16) * 11) % 256 for i in range(2048)))


def test_undo_t_is_a_permutation_and_not_identity():
    block = _deterministic_block()
    moved = _undo_t(block)
    assert len(moved) == len(block)
    assert sorted(moved) == sorted(block), "T^-1 must permute, not transform values"
    assert moved != block, "T^-1 must actually move bytes, else the layout split is vacuous"


def test_undo_t_has_order_three():
    block = _deterministic_block()
    once = _undo_t(block)
    twice = _undo_t(once)
    thrice = _undo_t(twice)
    assert thrice == block, "T has order 3, so T^-1 applied three times is identity"
    assert twice != block, "T^-1 is not self-inverse (T^-1 != T); a single undo is required"


@pytest.mark.parametrize("name", ["gate_proj", "down_proj"])
def test_wrong_conventions_do_not_match(name: str):
    # Deterministic stand-ins for the two expert shards; the point is the
    # convention split, not the checkpoint values.
    block = _deterministic_block() if name == "gate_proj" else _deterministic_block()[::-1]
    plain = _unpack(block)
    # Applying T^-1 to already-plain bytes must NOT reproduce the values: this is
    # exactly the silent-corruption hazard the layout split removes.
    assert _unpack(_undo_t(block)) != plain
    # Unsigned-offset decoding is a different convention and must differ.
    offset: list[int] = []
    for byte in block:
        offset += [(byte & 0x0F) - 8, ((byte >> 4) & 0x0F) - 8]
    assert offset != plain
    assert len(plain) == 4096


# ---------------------------------------------------------------------------
# 3. plain branch skips T^-1 (AST lock)
# ---------------------------------------------------------------------------


def test_plain_preprocess_skips_undo_t():
    node = definition(
        ROOT / EXPERTS, "PPUDeepGemmExpertsW4A8Int8.process_weights_after_loading"
    )
    undo_calls = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_w4a8_undo_checkpoint_T"
    ]
    assert len(undo_calls) == 1
    # The single undo call must sit inside `if not plain:`.
    guards = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.UnaryOp)
        and isinstance(n.test.op, ast.Not)
        and isinstance(n.test.operand, ast.Name)
        and n.test.operand.id == "plain"
        and any(undo in ast.walk(n) for undo in undo_calls)
    ]
    assert guards, "undo-T must be guarded by `if not plain:`"
    src = ast.unparse(node)
    assert "plain = self.weight_layout == 'deepgemm_plain'" in src
    assert "expected_dtype = torch.uint8 if plain else torch.int8" in src


def test_experts_ctor_accepts_weight_layout_keyword():
    node = definition(ROOT / EXPERTS, "PPUDeepGemmExpertsW4A8Int8.__init__")
    kwonly = {
        a.arg: ast.literal_eval(d) if d is not None else None
        for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
    }
    assert kwonly.get("weight_layout") == "acext_t_order"


# ---------------------------------------------------------------------------
# 4. create_weights container dtype
# ---------------------------------------------------------------------------


def _run_create_weights(layout: str):
    dtypes: list = []

    def fake_empty(*shape, dtype=None):
        dtypes.append(dtype)
        return SimpleNamespace(shape=shape)

    torch_stub = SimpleNamespace(
        nn=SimpleNamespace(Parameter=lambda data, requires_grad=False: data),
        empty=fake_empty,
        ones=lambda *shape, dtype=None: SimpleNamespace(shape=shape),
        uint8="U8",
        int8="I8",
        float32="F32",
    )
    create = function(
        MIXED_W4,
        "W4AInt8MoEMethod.create_weights",
        {
            "torch": torch_stub,
            "set_weight_attrs": lambda *a, **k: None,
            "FusedMoeWeightScaleSupported": SimpleNamespace(
                CHANNEL=SimpleNamespace(value="channel")
            ),
        },
    )
    registered: dict = {}
    layer = SimpleNamespace(
        register_parameter=lambda name, param: registered.__setitem__(name, param)
    )
    self_stub = SimpleNamespace(
        quant_config=SimpleNamespace(pack_factor=2, weight_layout=layout)
    )
    create(self_stub, layer, 8, 2560, 640, "BF16")
    return dtypes, registered


@pytest.mark.parametrize(
    "layout,expected", [("deepgemm_plain", "U8"), ("acext_t_order", "I8")]
)
def test_create_weights_dtype_follows_layout(layout, expected):
    dtypes, registered = _run_create_weights(layout)
    assert dtypes == [expected, expected]
    assert set(registered) == {
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
    }
    # Shapes identical across layouts: [E, 2I, K/2] and [E, K, I/2].
    assert registered["w13_weight"].shape == (8, 1280, 1280)
    assert registered["w2_weight"].shape == (8, 2560, 320)


# ---------------------------------------------------------------------------
# 5. make_kernel forwards weight_layout
# ---------------------------------------------------------------------------


def test_make_kernel_forwards_weight_layout():
    node = definition(ROOT / REGISTRY, "make_w4a8_int8_moe_kernel")
    kwonly = {a.arg for a in node.args.kwonlyargs}
    assert "weight_layout" in kwonly
    src = ast.unparse(node)
    assert "extra_args['weight_layout'] = weight_layout" in src
    # Forwarded only on the batched branch (experts ctor requires it there).
    batched_if = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.If) and "BatchedExperts" in ast.unparse(n.test)
    ]
    assert batched_if
    assert any(
        "weight_layout" in ast.unparse(stmt)
        for n in batched_if
        for stmt in n.body
    )


# ---------------------------------------------------------------------------
# 6. _mk_enabled layout override
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["v0.34", "v0.33"])
@pytest.mark.parametrize(
    "layout,deepep_ll,batched,expected",
    [
        ("deepgemm_plain", False, False, True),
        ("deepgemm_plain", True, True, True),
        ("acext_t_order", True, True, True),
        ("acext_t_order", True, False, False),
        ("acext_t_order", False, False, False),
    ],
)
def test_mk_enabled_layout_override(layout, deepep_ll, batched, expected, shape):
    ns = {}
    # T1: exercise the REAL compat fallback helper, not a stub.
    function("vllm_sail/compat.py", "fused_moe_use_batched", ns)
    method = function(MIXED_W4, "W4AInt8MoEMethod._mk_enabled", ns)
    # T1: stub shape matches the real upstream config shape per version.
    parallel = SimpleNamespace(use_deepep_ll_kernels=deepep_ll)
    if shape == "v0.34":
        assert not hasattr(parallel, "use_batched_activation_format")
        moe = SimpleNamespace(
            use_batched_activation_format=batched,
            moe_parallel_config=parallel,
        )
    else:
        assert shape == "v0.33"
        parallel.use_batched_activation_format = batched
        moe = SimpleNamespace(moe_parallel_config=parallel)
    self_stub = SimpleNamespace(
        quant_config=SimpleNamespace(weight_layout=layout),
        moe=moe,
    )
    assert method(self_stub) is expected
