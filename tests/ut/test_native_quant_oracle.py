# SPDX-License-Identifier: Apache-2.0
"""Guard the device suite's FP8 comparator against saturation false positives."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.requires_runtime
def test_fp8_oracle_rejects_nan_next_to_finite_max(sign):
    torch = pytest.importorskip("torch")
    suite = Path(__file__).parents[1] / "e2e/tier_b/test_norm_quant.py"
    check = runpy.run_path(str(suite))["_check_quant"]
    reference = torch.tensor([sign * 448], dtype=torch.float8_e4m3fn)
    adjacent = torch.tensor([sign * 416], dtype=reference.dtype)
    check(torch, reference, reference)
    check(torch, adjacent, reference)
    nan = torch.tensor([0xFF if sign < 0 else 0x7F], dtype=torch.uint8).view(
        reference.dtype
    )
    with pytest.raises(AssertionError):
        check(torch, nan, reference)
