# SPDX-License-Identifier: Apache-2.0
"""Preserve the activation-scale layout selected by the FP8 backend."""

from enum import Enum
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/fused_moe_ppu.py"


def platform(monkeypatch, ppu):
    import sys

    module = ModuleType("vllm.platforms")
    module.current_platform = NS(is_ppu=lambda: ppu, fp8_dtype=lambda: "fp8")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.current_platform


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_quantizer_forwards_new_layout_keyword(monkeypatch, ppu, transpose):
    platform(monkeypatch, ppu)
    expected, tensor, scale = object(), object(), object()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    quantize = function(
        PATCH,
        "install._moe_kernel_quantize_input",
        {
            "_upstream_quantize_input": original,
        },
    )
    assert (
        quantize(tensor, scale, "fp8", False, [128, 128], is_scale_transposed=transpose)
        is expected
    )
    assert calls == [
        (
            (tensor, scale, "fp8", False),
            dict(block_shape=[128, 128], is_scale_transposed=transpose),
        )
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize(
    "backend,block,transpose",
    [
        ("FLASHINFER_TRTLLM", [128, 128], True),
        ("FLASHINFER_TRTLLM", [64, 128], False),
        ("TRITON", [128, 128], False),
        ("FLASHINFER_CUTLASS", [128, 128], False),
    ],
)
def test_actual_release_config_uses_the_provider_default_scale_layout(
    monkeypatch, upstream_source_root, ppu, backend, block, transpose
):
    device = platform(monkeypatch, ppu)
    oracle_path = (
        upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/fp8.py"
    )
    utils_path = upstream_source_root / "vllm/model_executor/layers/fused_moe/utils.py"
    backend_type = function(oracle_path, "Fp8MoeBackend", {"Enum": Enum})
    original_config = function(
        oracle_path,
        "make_fp8_moe_quant_config",
        {
            "Fp8MoeBackend": backend_type,
            "fp8_w8a8_moe_quant_config": lambda **kwargs: NS(**kwargs),
        },
    )
    make_config = function(
        "vllm_sail/registry/moe_backends/fp8.py",
        "make_fp8_moe_quant_config",
        {"_upstream_quant_config": original_config},
    )
    alpha, beta = object(), object()
    config = make_config(
        getattr(backend_type, backend),
        object(),
        object(),
        None,
        None,
        block_shape=block,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
    )
    assert not hasattr(config, "is_scale_transposed")
    assert config.gemm1_alpha is alpha and config.gemm1_beta is beta
    calls = []
    values = NS(size=lambda axis: 512)
    scales = NS(size=lambda axis: 4)

    def group_quant(tensor, size, **kwargs):
        calls.append((tensor, size, kwargs))
        return values, scales

    namespace = {
        "current_platform": device,
        "per_token_group_quant_fp8": group_quant,
        "cdiv": lambda a, b: (a + b - 1) // b,
    }
    function(utils_path, "_fp8_quantize", namespace)
    upstream_quantizer = function(utils_path, "moe_kernel_quantize_input", namespace)
    quantize = function(
        PATCH,
        "install._moe_kernel_quantize_input",
        {"_upstream_quantize_input": upstream_quantizer},
    )
    tensor = object()
    assert quantize(
        tensor,
        None,
        "fp8",
        False,
        block,
    ) == (values, scales)
    assert calls == [(tensor, 128, {})]
    # The real release provider rejects this later-main keyword. The wrapper
    # still forwards arbitrary keywords, so unsupported input must fail here.
    with pytest.raises(TypeError, match="is_scale_transposed"):
        quantize(tensor, None, "fp8", False, block, is_scale_transposed=transpose)
