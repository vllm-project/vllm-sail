# SPDX-License-Identifier: Apache-2.0
"""Capture expert and channel strides without running a Triton kernel."""

import math
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules
PATCH = "vllm_sail/patch/enhancement/triton_moe.py"
UPSTREAM = "vllm/model_executor/layers/fused_moe/fused_moe.py"
SCALES = [
    (None, None, (0, 0, 0)),
    ((32,), (5,), (5, 0, 0)),
    ((32, 1), (7, 3), (7, 0, 0)),
    ((32, 8), (37, 2), (37, 0, 2)),
    ((32, 8, 2), (131, 9, 2), (131, 2, 9)),
]


class Tensor:
    device = "host-double"

    def __init__(self, shape, strides=None):
        self.shape, self.ndim = shape, len(shape)
        self.strides = strides

    def size(self, axis):
        return self.shape[axis]

    def stride(self, axis):
        if self.strides is not None:
            return self.strides[axis]
        return math.prod(self.shape[axis % self.ndim + 1 :])


def check_launch(path, modules, shape, strides, expected, capability, copied):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            assert grid({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 8}) == (4,)
            return lambda *args, **kwargs: calls.append((args, kwargs))

    kernel = Kernel()
    modules(
        "vllm_sail.model_executor.layers.fused_moe.triton_kernels",
        fused_moe_kernel=kernel,
    )
    ppu = capability != 90
    namespace = {
        "fused_moe_kernel": kernel,
        "warn_if_moe_use_td_ineffective": lambda *a, **kw: None,
        "resolve_moe_use_td": lambda: False,
        "triton": NS(cdiv=lambda n, d: (n + d - 1) // d),
        "current_platform": NS(
            is_ppu=lambda: ppu,
            has_device_capability=lambda cap: capability >= cap[0] * 10 + cap[1],
        ),
    }
    scale = Tensor(shape, strides) if shape is not None else None
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 4}
    kwargs = {"use_valu": True} if copied else {}
    function(path, "invoke_fused_moe_triton_kernel", namespace)(
        Tensor((2, 4)),
        Tensor((32, 8, 4)),
        Tensor((2, 2, 8)),
        None,
        scale,
        None,
        None,
        Tensor((4,)),
        Tensor((1,)),
        False,
        2,
        config,
        compute_type="bf16",
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=scale is not None,
        use_int4_w4a16=False,
        per_channel_quant=True,
        **kwargs,
    )
    assert len(calls) == 1
    args, launch = calls[0]
    assert args[5] is scale and args[23:26] == expected
    assert launch["naive_block_assignment"] is True
    assert config == {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 4}
    if copied:
        assert launch["use_valu"] is True
        assert launch["even_Ks"] is (ppu and capability >= 89)


@pytest.mark.parametrize("shape,strides,expected", SCALES)
@pytest.mark.parametrize("capability", [80, 89, 90])
def test_weight_scale_strides_reach_the_kernel(
    modules, shape, strides, expected, capability
):
    check_launch(PATCH, modules, shape, strides, expected, capability, True)


@pytest.mark.upstream_source
@pytest.mark.parametrize("shape,strides,expected", SCALES)
@pytest.mark.parametrize("capability", [80, 89, 90])
@pytest.mark.parametrize("copied", [False, True])
def test_actual_weight_scale_launch_contract(
    upstream_source_root, modules, shape, strides, expected, capability, copied
):
    path = PATCH if copied else upstream_source_root / UPSTREAM
    check_launch(path, modules, shape, strides, expected, capability, copied)
