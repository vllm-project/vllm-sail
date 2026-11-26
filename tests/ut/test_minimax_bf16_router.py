from types import SimpleNamespace

import pytest

from tests.support.source import function


@pytest.mark.parametrize(
    "case",
    [
        "default",
        "other_platform",
        "fp16",
        "invariant",
        "forced_fp32",
    ],
)
def test_minimax_gate_creation(case):
    ctor = function(
        "vllm_sail/patch/enhancement/models/minimax_m3.py",
        "make_gate_linear",
        {
            "current_platform": SimpleNamespace(
                is_ppu=lambda: case != "other_platform",
            ),
            "_model": SimpleNamespace(
                envs=SimpleNamespace(VLLM_BATCH_INVARIANT=case == "invariant"),
                torch=SimpleNamespace(
                    float32="fp32",
                    bfloat16="bf16",
                    get_default_dtype=lambda: "fp16" if case == "fp16" else "bf16",
                ),
            ),
            "_original_gate_linear": lambda *args, **kwargs: (args, kwargs),
        },
    )
    params = dict(bias=False, params_dtype="fp32", out_dtype="fp32", prefix="mlp.gate")
    if case == "forced_fp32":
        params["force_fp32_compute"] = True
    args, actual = ctor(6144, 128, **params)
    assert args == (6144, 128)
    assert actual == {**params, "params_dtype": "bf16" if case == "default" else "fp32"}
