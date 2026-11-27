# SPDX-License-Identifier: Apache-2.0
"""The decode split-prep gate keeps the joint path outside its validated shapes."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_sail import envs

PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_sail/models/step5/ops/decode_split_prep.py"
)


def _load_gate(*, registered=True, env_on=True, ppu=True, cap89=True):
    function = next(
        node
        for node in ast.walk(ast.parse(PATH.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_split_supported"
    )
    function.decorator_list = []
    # The lazy _main_supported import is replaced by the namespace stub.
    function.body = [
        node
        for node in function.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and "step5_qknorm_rope_cache" in node.module
        )
    ]
    dtypes = {name: object() for name in ("bf16", "f32", "i64", "u8", "fp8")}
    vllm_ops = SimpleNamespace()
    if registered:
        vllm_ops.ppu_step5_qknorm_rope_cache = object()
    namespace = {
        "torch": SimpleNamespace(
            bfloat16=dtypes["bf16"],
            float32=dtypes["f32"],
            int64=dtypes["i64"],
            uint8=dtypes["u8"],
            float8_e4m3fn=dtypes["fp8"],
            ops=SimpleNamespace(vllm=vllm_ops),
        ),
        "envs": SimpleNamespace(VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV=env_on),
        "current_platform": SimpleNamespace(
            is_ppu=lambda: ppu,
            has_device_capability=lambda capability: cap89 and capability == 89,
        ),
        "_main_supported": lambda *args: True,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(PATH), "exec"),
        namespace,
    )
    return namespace["_split_supported"], dtypes


def _tensor(dtype, shape, stride=None, device="cuda:0"):
    if stride is None:
        stride = tuple(
            1 if i == len(shape) - 1 else shape[i + 1] for i in range(len(shape))
        )
    return SimpleNamespace(
        dtype=dtype,
        shape=shape,
        ndim=len(shape),
        stride=lambda dim: stride[dim],
        device=device,
        is_contiguous=lambda: True,
        numel=lambda: shape[0] if shape else 1,
    )


def _args(dtypes, *, tokens=64, heads=4, rotary=32, cache_dtype=None, q_dtype=None):
    bf16, f32, i64 = dtypes["bf16"], dtypes["f32"], dtypes["i64"]
    cache_dtype = bf16 if cache_dtype is None else cache_dtype
    main_inputs = (
        _tensor(bf16, (tokens, 18 * 192)),
        _tensor(bf16, (192,)),
        _tensor(bf16, (192,)),
        _tensor(bf16, (4096, 64)),
        _tensor(i64, (tokens,)),
        _tensor(cache_dtype, (4, 64, 1, 192)),
        _tensor(cache_dtype, (4, 64, 1, 192)),
        16,
        1,
        64,
        1e-6,
        1.0,
        None,
        None,
    )
    ring = SimpleNamespace(
        rows=_tensor(bf16, (256, 512)),
        capacity=256,
        metadata=lambda: SimpleNamespace(base=_tensor(i64, (tokens,))),
    )
    cache = SimpleNamespace(mean=_tensor(dtypes["u8"], (1, 32, 256)))
    return (
        _tensor(bf16, (tokens, heads * 256)),
        _tensor(bf16, (tokens, 256)),
        _tensor(f32, (tokens, heads)),
        _tensor(bf16, (256,)),
        _tensor(bf16, (256,)),
        _tensor(bf16, (256,)),
        _tensor(bf16, (4096, rotary)),
        _tensor(i64, (tokens,)),
        True,
        main_inputs,
        cache,
        ring,
        _tensor(bf16, (tokens, 256)),
        _tensor(i64, (tokens,)),
        q_dtype,
    )


def test_eligible_decode_shape_splits():
    gate, dtypes = _load_gate()
    assert gate(*_args(dtypes))


@pytest.mark.parametrize(
    "overrides",
    [
        {"tokens": 0},
        {"tokens": 256},
        {"heads": 2},
        {"heads": 16},
        {"rotary": 128},
        {"cache_dtype": "fp8"},
        {"q_dtype": "fp8"},
    ],
)
def test_out_of_scope_shapes_keep_joint(overrides):
    gate, dtypes = _load_gate()
    if overrides.get("cache_dtype") == "fp8":
        overrides["cache_dtype"] = dtypes["fp8"]
    if overrides.get("q_dtype") == "fp8":
        overrides["q_dtype"] = dtypes["fp8"]
    assert not gate(*_args(dtypes, **overrides))


@pytest.mark.parametrize(
    "kwargs",
    [{"env_on": False}, {"ppu": False}, {"cap89": False}, {"registered": False}],
)
def test_disabled_environment_keeps_joint(kwargs):
    gate, dtypes = _load_gate(**kwargs)
    assert not gate(*_args(dtypes))


def test_incomplete_csa_inputs_keep_joint():
    gate, dtypes = _load_gate()
    args = list(_args(dtypes))
    args[12] = None  # decode_z missing while the summary cache is present
    assert not gate(*args)


def test_default_matches_library_contract():
    assert envs.VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV is True
    assert envs.VLLM_PPU_STEP5_QKNORM_ROPE_CACHEKV is True
