# SPDX-License-Identifier: Apache-2.0
"""The PPU SWA decode patch relaxes the capability-103 and FA-version gates."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_sail import envs

PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_sail/patch/performance/step5_swa_decode.py"
)


def _load_init():
    function = next(
        node
        for node in ast.walk(ast.parse(PATH.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "init"
    )
    function.decorator_list = []
    bf16 = object()
    calls = []

    def original_init(self, *args, **kwargs):
        calls.append((args, kwargs))

    namespace = {
        "torch": SimpleNamespace(bfloat16=bf16),
        "original_init": original_init,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(PATH), "exec"),
        namespace,
    )
    return namespace["init"], bf16, calls


def _layer(
    bf16, *, fused=False, dsa=False, gated=True, fa_version=3, head_size=192, dtype=None
):
    return SimpleNamespace(
        fuse_swa_gate=fused,
        use_dsa_backend=dsa,
        use_head_wise_attn_gate=gated,
        params_dtype=bf16 if dtype is None else dtype,
        attn=SimpleNamespace(
            impl=SimpleNamespace(
                head_size=head_size, vllm_flash_attn_version=fa_version
            )
        ),
    )


@pytest.mark.parametrize("fa_version", [2, 3])
def test_eligible_swa_layer_enables_fused_gate(fa_version):
    init, bf16, calls = _load_init()
    layer = _layer(bf16, fa_version=fa_version)
    init(layer, "arg", kw=1)
    assert layer.fuse_swa_gate is True
    assert calls == [(("arg",), {"kw": 1})]


def test_already_fused_layer_is_untouched():
    init, bf16, calls = _load_init()
    layer = _layer(bf16, fused=True)
    init(layer)
    assert layer.fuse_swa_gate is True
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dsa": True},
        {"gated": False},
        {"fa_version": 4},
        {"fa_version": None},
        {"head_size": 128},
        {"dtype": object()},
    ],
)
def test_ineligible_layers_keep_upstream_decision(kwargs):
    init, bf16, calls = _load_init()
    layer = _layer(bf16, **kwargs)
    init(layer)
    assert layer.fuse_swa_gate is False
    assert len(calls) == 1


def test_swa_decode_env_defaults_on():
    assert envs.VLLM_SAIL_STEP5_FUSED_SWA_DECODE is True


def _load_gate():
    function = next(
        node
        for node in ast.walk(ast.parse(PATH.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "headwise_sigmoid_gate"
    )
    function.decorator_list = []
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    namespace = {
        "attention_gate": SimpleNamespace(_headwise_sigmoid_gate=Kernel()),
        "triton": SimpleNamespace(cdiv=lambda a, b: -(-a // b)),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(PATH), "exec"),
        namespace,
    )
    return namespace["headwise_sigmoid_gate"], launches


def _tensor(rows, width, stride=None):
    s = stride if stride is not None else (width, 1)
    return SimpleNamespace(
        shape=(rows, width),
        stride=lambda *a: s[a[0]] if a else s,
        numel=lambda: rows,
    )


def test_fallback_gate_writes_in_place_and_returns_input():
    gate_fn, launches = _load_gate()
    attention = _tensor(8, 192)
    gate = _tensor(8, 8, stride=(8, 1))
    result = gate_fn(attention, gate)
    assert result is attention
    ((grid, args, kwargs),) = launches
    assert grid == (8, 1)
    assert args[0] is attention and args[2] is attention
    assert args[1] is gate
    assert args[3] is None and args[4] == 0
    assert args[5] is False and args[6] is False and args[7] == 0
    assert args[8] == 192 and args[9] == 24
    assert args[10:12] == (192, 1) and args[12:14] == (8, 1)
    assert args[14] == 1024
    assert kwargs == {"enable_fp_fusion": False}


def test_fallback_gate_preserves_active_mask_semantics():
    gate_fn, launches = _load_gate()
    attention = _tensor(4, 384)
    gate = _tensor(4, 8, stride=(8, 1))
    active = _tensor(4, 1, stride=(1, 1))
    gate_fn(attention, gate, active)
    ((_, args, _),) = launches
    assert args[3] is active and args[4] == 1
    assert args[5] is True and args[6] is False and args[7] == 4


def test_fallback_gate_preserves_slot_mask_semantics():
    gate_fn, launches = _load_gate()
    attention = _tensor(4, 384)
    gate = _tensor(4, 8, stride=(8, 1))
    slots = _tensor(4, 1, stride=(1, 1))
    gate_fn(attention, gate, slot_mapping=slots)
    ((_, args, _),) = launches
    assert args[3] is slots and args[4] == 1
    assert args[5] is True and args[6] is True and args[7] == 4


def test_fallback_gate_rejects_both_masks():
    gate_fn, launches = _load_gate()
    attention = _tensor(4, 192)
    gate = _tensor(4, 8, stride=(8, 1))
    mask = _tensor(4, 1, stride=(1, 1))
    with pytest.raises(ValueError, match="either active flags or decode slots"):
        gate_fn(attention, gate, mask, slot_mapping=mask)
    assert launches == []
