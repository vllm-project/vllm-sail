# SPDX-License-Identifier: Apache-2.0
"""Numerical checks for the V4.1 MXFP8 fallback; run when PPU is available."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("dequant_at_load", [False, True])
def test_mxfp8_32x32_scale_loading_and_dense_output(monkeypatch, dequant_at_load):
    import torch
    import vllm.envs as envs
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.modelopt import CkptCtx, KMxfp8Static

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.model_executor.kernels.linear.mxfp8 import (
        PPUEmulationMxfp8LinearKernel,
    )

    monkeypatch.setattr(envs, "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", dequant_at_load)
    layer = torch.nn.Module()
    weight = torch.ones(64, 64, device="cuda", dtype=torch.float8_e4m3fn)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.empty(64, 2, device="cuda", dtype=torch.uint8), requires_grad=False
    )
    # Distinct row and column blocks catch accidental 32 -> 128 reinterpretation.
    checkpoint_scales = torch.tensor([[127, 128], [126, 129]], dtype=torch.uint8)
    loader = KMxfp8Static.get_scale_weight_loader(
        lambda param, loaded: param.copy_(loaded), CkptCtx(scale_block_size=(32, 32))
    )
    loader(layer.weight_scale, checkpoint_scales.to("cuda"))
    kernel = PPUEmulationMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    x = torch.ones(3, 64, device="cuda", dtype=torch.bfloat16)
    bias = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    actual = kernel.apply_weights(layer, x, bias)
    expected = torch.tensor([97.0] * 32 + [145.0] * 32, dtype=torch.bfloat16)
    torch.testing.assert_close(actual.cpu(), expected.expand(3, 64), rtol=0, atol=0)
