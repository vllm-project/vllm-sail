# SPDX-License-Identifier: Apache-2.0
"""Bitwise-equality gate for the v1.2 F2 fused-dequant tiers.

Numerical gate from decode_regression_rootcause_fixplan_20261111 §三.7: the
``triton`` and ``torch`` dequant tiers must be *bitwise* identical to the
pre-v1.2 ``legacy`` ATen chain on random inputs (e4m3 x 2^ue8m0 is exact in
f32/bf16, so any deviation is a layout/byte-order bug, not rounding).

Environment: needs torch + vllm importable; the ``triton`` tier additionally
needs triton and an accelerator. The offline CPU CI skips these tests. The
triton tier asserts that the dispatcher did NOT silently degrade to the
broadcast fallback — a platform whose Triton cannot compile the fp8e4m3 load
(fixplan §四.4, needs-measurement) fails here loudly instead of passing via
the fallback.

Exponent byte 255 (e8m0 NaN) is excluded: NaN payloads are not required to
match bitwise. Byte 0 (2^-127, f32 denormal) IS included — the Triton tier
special-cases it to match torch's e8m0 conversion.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_sail.models.deepseek_v41 import flash_mla_mega_attn as mod

_HAS_DEVICE = torch.cuda.is_available()
_DEVICE = "cuda" if _HAS_DEVICE else "cpu"

# TP8 DSV4.1-Flash runtime geometry (server-log pinned): T=256, G=8, D=4096.
# The UT uses a shrunken T to stay fast; G/D keep the real contract.
_T, _G, _D = 64, 8, 4096


def _make_inputs(
    t: int,
    g: int,
    d: int,
    scale_word_width: int | None = None,
    data_group_width: int | None = None,
    seed: int = 0,
):
    """Random (data fp8e4m3 [t,g,d], scale int32 [t,g,d//128]) pair.

    ``scale_word_width`` / ``data_group_width`` build the tensors as narrow
    views of wider buffers so the dequant tiers see the non-contiguous
    strides the F1 zero-copy stash hands over (out_sf is a transposed
    narrow view of allocate_scale_factor's buffer; data[:, :groups] slices
    the group dim when n_local_groups < n_wv_group).
    """
    gen = torch.Generator().manual_seed(seed)
    n_words = d // 128
    w = scale_word_width or n_words
    raw_scale = torch.randint(
        0, 255, (t, g, w * 4), generator=gen, dtype=torch.uint8
    ).to(torch.uint8)
    packed_full = raw_scale.view(torch.int32)  # [t, g, w], little-endian words
    scale = packed_full[:, :, :n_words]
    data_full = (
        (torch.randn(t, data_group_width or g, d, generator=gen) * 0.5)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    data = data_full[:, :g]
    return data.to(_DEVICE), scale.to(_DEVICE)


def _run_tier(monkeypatch, mode, data, scale):
    monkeypatch.setenv("VLLM_SAIL_FUSED_DEQUANT_MODE", mode)
    mod._DEQUANT_TRITON_BROKEN = False
    out = mod._dequant_fused_output(data, scale)
    degraded = mod._DEQUANT_TRITON_BROKEN
    return out, degraded


@pytest.mark.parametrize("mode", ["torch", "triton"])
def test_dequant_tiers_bitwise_equal_legacy(mode, monkeypatch):
    if mode == "triton":
        pytest.importorskip("triton")
        if not _HAS_DEVICE:
            pytest.skip("triton tier needs an accelerator")
    data, scale = _make_inputs(_T, _G, _D, seed=1)
    legacy = _run_tier(monkeypatch, "legacy", data, scale)[0]
    out, degraded = _run_tier(monkeypatch, mode, data, scale)
    if mode == "triton":
        assert not degraded, (
            "dispatcher degraded to the broadcast fallback: the Triton "
            "kernel failed on this platform (fixplan §四.4) — this gate "
            "requires the real Triton tier"
        )
    assert out.shape == legacy.shape and out.dtype == torch.bfloat16
    assert torch.equal(out.view(torch.uint16), legacy.view(torch.uint16)), (
        f"mode={mode} is not bitwise identical to the legacy chain"
    )


@pytest.mark.parametrize("mode", ["torch", "triton"])
def test_dequant_tiers_bitwise_equal_legacy_strided(mode, monkeypatch):
    """Non-contiguous data/scale views (the F1 stash reality) stay exact."""
    if mode == "triton":
        pytest.importorskip("triton")
        if not _HAS_DEVICE:
            pytest.skip("triton tier needs an accelerator")
    data, scale = _make_inputs(
        _T, _G, _D, scale_word_width=_D // 128 + 8, data_group_width=_G + 2, seed=2
    )
    assert not scale.is_contiguous() or not data.is_contiguous()
    legacy = _run_tier(monkeypatch, "legacy", data, scale)[0]
    out, degraded = _run_tier(monkeypatch, mode, data, scale)
    if mode == "triton":
        assert not degraded
    assert torch.equal(out.view(torch.uint16), legacy.view(torch.uint16))


def test_runtime_geometry_t256_bitwise(monkeypatch):
    """The pinned TP8 runtime geometry (T=256, G=8, D=4096) on device."""
    pytest.importorskip("triton")
    if not _HAS_DEVICE:
        pytest.skip("triton tier needs an accelerator")
    data, scale = _make_inputs(256, _G, _D, seed=3)
    legacy = _run_tier(monkeypatch, "legacy", data, scale)[0]
    out, degraded = _run_tier(monkeypatch, "triton", data, scale)
    assert not degraded
    assert torch.equal(out.view(torch.uint16), legacy.view(torch.uint16))
