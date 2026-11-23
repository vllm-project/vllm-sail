# SPDX-License-Identifier: Apache-2.0
"""Tests for the sparse MLA prefill head-padding narrowing patch.

The patch narrows ``FlashMLASparseImpl.prefill_padding`` from the flat 64 to
PPU's kernel head tile (16/32/64); on non-PPU platforms it must stay a pure
pass-through of the upstream value.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

import vllm_sail  # noqa: E402
import vllm_sail.patch as patch_pkg  # noqa: E402
from vllm_sail.patch.enhancement.attention import (  # noqa: E402
    sparse_mla_prefill_padding as plugin,
)
from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY  # noqa: E402

_BACKEND = importlib.import_module("vllm.v1.attention.backends.mla.flashmla_sparse")
_TARGET = f"{plugin._MODULE}.FlashMLASparseImpl.prefill_padding"


class _PlatformDouble:
    def __init__(self, ppu: bool) -> None:
        self._ppu = ppu

    def is_ppu(self) -> bool:
        return self._ppu


@pytest.fixture(scope="module")
def installed():
    vllm_sail.register()
    patch_pkg.install()
    return PATCH_REGISTRY


@pytest.fixture()
def platform(monkeypatch: pytest.MonkeyPatch):
    def _set(ppu: bool) -> _PlatformDouble:
        double = _PlatformDouble(ppu)
        monkeypatch.setattr(plugin, "current_platform", double)
        return double

    return _set


def _new_impl(num_heads: int):
    """A bare impl instance: the descriptor needs no real __init__."""
    impl = object.__new__(_BACKEND.FlashMLASparseImpl)
    impl.num_heads = num_heads
    return impl


# ---------------------------------------------------------------------------
# Patch wiring
# ---------------------------------------------------------------------------
def test_target_is_registered(installed) -> None:
    registered = {record.target for record in installed}
    assert _TARGET in registered


def test_marker_lands_on_the_descriptor(installed) -> None:
    descriptor = getattr(plugin, "_get_prefill_padding", None)
    assert descriptor is not None
    markers = getattr(descriptor, PATCH_MARKER, None)
    assert isinstance(markers, dict) and _TARGET in markers


def test_descriptor_is_installed_on_the_impl_class(installed) -> None:
    descriptor = vars(_BACKEND.FlashMLASparseImpl)["prefill_padding"]
    assert isinstance(descriptor, property)
    assert descriptor.fset is not None, "the intrinsic assignment must be intercepted"


# ---------------------------------------------------------------------------
# PPU: padding follows the kernel head tile
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("num_heads", "expected"),
    [(1, 16), (8, 16), (16, 16), (17, 32), (32, 32), (33, 64), (64, 64), (128, 64)],
)
def test_on_ppu_padding_follows_the_kernel_head_tile(
    installed, platform, num_heads: int, expected: int
) -> None:
    platform(ppu=True)
    impl = _new_impl(num_heads)
    impl.prefill_padding = 64
    assert impl.prefill_padding == expected


def test_on_ppu_wide_heads_pass_the_value_through(installed, platform) -> None:
    platform(ppu=True)
    impl = _new_impl(64)
    impl.prefill_padding = 32
    assert impl.prefill_padding == 32


def test_unset_padding_reads_as_upstream_default(installed, platform) -> None:
    platform(ppu=True)
    assert _new_impl(16).prefill_padding == 64


# ---------------------------------------------------------------------------
# Non-PPU: pure delegation to upstream
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_heads", [1, 16, 32, 64])
def test_off_ppu_keeps_the_upstream_padding(
    installed, platform, num_heads: int
) -> None:
    platform(ppu=False)
    impl = _new_impl(num_heads)
    impl.prefill_padding = 64
    assert impl.prefill_padding == 64


def test_non_positive_head_count_falls_back(installed, platform) -> None:
    platform(ppu=True)
    impl = _new_impl(0)
    impl.prefill_padding = 64
    assert impl.prefill_padding == 64
