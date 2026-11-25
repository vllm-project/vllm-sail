# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual prepare/quantizer boundary used by PPU MXFP4 experts."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float32"])
def test_mxfp4_prepare_uses_ppu_quantizer(dtype_name):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.fused_moe.config import ocp_mx_moe_quant_config
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        PPUDeepGemmExpertsMXFP4,
    )

    quant = ocp_mx_moe_quant_config(
        quant_dtype="mxfp4",
        weight_dtype="mxfp4",
        w1_scale=torch.ones(1, device="cuda"),
        w2_scale=torch.ones(1, device="cuda"),
        block_shape=[1, 16],
    )
    # Constant rows have exact FP4 encodings: +/-1 at scale 2^-2 and
    # +2 at scale 2^-1 all have magnitude 4 (E2M1 code 6).
    x = torch.tensor([1, -1, 2], device="cuda", dtype=getattr(torch, dtype_name))
    x = x[:, None].expand(3, 128).contiguous()
    expert = PPUDeepGemmExpertsMXFP4.__new__(PPUDeepGemmExpertsMXFP4)
    packed, scales, *_ = MoEPrepareAndFinalizeNoDPEPModular().prepare(
        x,
        torch.ones(3, 1, device="cuda"),
        torch.zeros(3, 1, device="cuda", dtype=torch.int32),
        1,
        None,
        False,
        quant,
        defer_input_quant=expert.expects_unquantized_inputs,
    )
    torch.cuda.synchronize()
    assert packed.dtype == torch.uint8 and packed.shape == (3, 64)
    assert scales.dtype == torch.uint16 and scales.shape == (3, 2)
    torch.testing.assert_close(
        packed.cpu(),
        torch.tensor([0x66, 0xEE, 0x66], dtype=torch.uint8)[:, None].expand(3, 64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scales.cpu().to(torch.int32),
        torch.tensor([0x7D7D, 0x7D7D, 0x7E7E], dtype=torch.int32)[:, None].expand(3, 2),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16", "float16"])
def test_downcast_to_mxfp4_rounding_matches_across_scale_formats(dtype_name):
    """Issue #399: UINT8_ROW_MAJOR and UINT16_COL_MAJOR must match on midpoints."""
    import torch

    from vllm_sail.model_executor.layers.quantization.utils.mxfp4_utils import (
        ScaleFormat,
        downcast_to_mxfp4,
    )

    dtype = getattr(torch, dtype_name)
    values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
    ref_pos = [0, 2, 2, 4, 4, 6, 6, 7]
    ref_neg = [code | 0x8 for code in ref_pos]

    row = values * 4 + [-v for v in values] * 4
    expected_codes = ref_pos * 4 + ref_neg * 4
    x = torch.tensor(row, dtype=dtype, device="cuda").reshape(1, 64)

    q_u8, s_u8 = downcast_to_mxfp4(x, axis=1, scale_format=ScaleFormat.UINT8_ROW_MAJOR)
    q_u16, s_u16 = downcast_to_mxfp4(
        x, axis=1, scale_format=ScaleFormat.UINT16_COL_MAJOR
    )
    torch.cuda.synchronize()

    assert torch.equal(q_u8.cpu(), q_u16.cpu())
    assert s_u8.cpu().reshape(-1).tolist() == [127, 127]
    assert s_u16.cpu().contiguous().view(torch.uint8).reshape(-1).tolist() == [127, 127]

    packed = q_u8.cpu().reshape(-1).tolist()
    codes = [code for byte in packed for code in (byte & 0x0F, byte >> 4)]
    assert codes == expected_codes
