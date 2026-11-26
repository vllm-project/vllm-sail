# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the packed kernel-block split patch.

Upstream (e83d82a) maps self-addressed specs (``uses_slot_mapping`` False)
natively, so the patch delegates those straight to upstream and only force-
splits a slot-mapping-written token cache sharing the packed indexer's forced
kernel block size. These tests pin the delegation paths (self-addressed, non-
PPU, no split needed, unaddressable layout) and the force-split arithmetic
(``kernel_block_stride`` / ``num_pages`` / ``as_strided``, with the padded-page
``page_stride`` formula aligned with upstream).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_runtime

torch = pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

from vllm.platforms import current_platform  # noqa: E402

import vllm_sail  # noqa: E402,F401  (installs the plugin hooks)
from vllm_sail.patch.enhancement.attention import (  # noqa: E402
    packed_token_cache_split as plugin,
)

_HEAD_DIM = 8


class _FakeSpec:
    def __init__(
        self,
        kernel_rows: int,
        block_size: int,
        tokens_per_state: int,
        uses_slot_mapping: bool = True,
    ):
        self._kernel_rows = kernel_rows
        self.block_size = block_size
        self.tokens_per_state = tokens_per_state
        self.uses_slot_mapping = uses_slot_mapping
        self.num_heads = 1
        # One bf16 head_dim element row, so page_size_bytes equals
        # num_states * row_bytes and the padded-page formula reduces to the
        # unpadded page stride (matching the continuous fake layout).
        self.state_content_size_bytes = _HEAD_DIM * 2
        self.page_size_bytes = block_size * self.state_content_size_bytes

    @property
    def num_states(self) -> int:
        return self.get_num_kernel_states(self.block_size)

    def get_num_kernel_states(self, kernel_block_size: int) -> int:
        # Upstream semantics: kernel_block_size // tokens_per_state. The
        # padded-page page_stride formula calls this with the manager block
        # size too (via num_states), so it must not assume 64.
        return kernel_block_size // self.tokens_per_state


class _FakeBackend:
    def get_name(self) -> str:
        return "FakeKimiIndexer"


class _FakeGroup:
    def __init__(self, spec, kernel_block_size=64, layer_names=()):
        self.kv_cache_spec = spec
        self.backend = _FakeBackend()
        self.kernel_block_size = kernel_block_size
        self.kernel_block_stride = 0
        # Layers whose slot mappings the force-split would rewrite; empty in
        # these tests (the arithmetic does not depend on it).
        self.layer_names = list(layer_names)


def _stub_upstream_marker(monkeypatch: pytest.MonkeyPatch) -> object:
    """Replace the captured upstream with a marker-returning stand-in.

    The delegation tests must not depend on how the host vLLM's guard is
    written; they only pin that the fallback hands the group to upstream.
    """
    marker = object()

    def _guard(self, kv_cache):
        return marker

    monkeypatch.setattr(plugin, "_upstream_map_kv_cache_to_kernel_blocks", _guard)
    return marker


def _stub_upstream_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the captured upstream with a version-independent guard that
    rejects the split, like an unaddressable cache does on a real build."""

    def _guard(self, kv_cache):
        raise ValueError(
            "FakeKimiIndexer manager block cannot be split into 64-token kernel blocks"
        )

    monkeypatch.setattr(plugin, "_upstream_map_kv_cache_to_kernel_blocks", _guard)


def _packed_cache(
    manager_blocks: int,
    kernel_blocks_per_manager_block: int,
    kernel_rows: int,
    head_dim: int = _HEAD_DIM,
) -> torch.Tensor:
    """A [B, 1, k*kernel_rows, D] cache whose manager-block stride is a whole
    number of kernel pages, like the real packed layout (which additionally
    leaves other layers' pages between blocks; the arithmetic only needs the
    divisibility the split checks)."""
    k = kernel_blocks_per_manager_block
    return torch.empty(
        manager_blocks, 1, k * kernel_rows, head_dim, dtype=torch.bfloat16
    )


@pytest.fixture
def ppu_platform(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: True)
    return None


def test_self_addressed_spec_delegates_to_upstream(ppu_platform, monkeypatch) -> None:
    marker = _stub_upstream_marker(monkeypatch)
    spec = _FakeSpec(
        kernel_rows=64, block_size=256, tokens_per_state=1, uses_slot_mapping=False
    )
    group = _FakeGroup(spec)
    cache = _packed_cache(3, 4, 64)
    assert plugin.map_kv_cache_to_kernel_blocks(group, cache) is marker


def test_non_ppu_keeps_upstreams_guard(monkeypatch) -> None:
    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: False)
    _stub_upstream_guard(monkeypatch)
    spec = _FakeSpec(kernel_rows=64, block_size=256, tokens_per_state=1)
    group = _FakeGroup(spec)
    cache = _packed_cache(3, 4, 64)
    with pytest.raises(ValueError, match="cannot be split into"):
        plugin.map_kv_cache_to_kernel_blocks(group, cache)


def test_split_view_matches_the_arithmetic(ppu_platform) -> None:
    spec = _FakeSpec(kernel_rows=64, block_size=256, tokens_per_state=1)
    group = _FakeGroup(spec)
    cache = _packed_cache(
        manager_blocks=3, kernel_blocks_per_manager_block=4, kernel_rows=64
    )
    view = plugin.map_kv_cache_to_kernel_blocks(group, cache)
    # Continuous cache: page_stride=512, kernel_block_stride=4, num_pages=12.
    assert view.shape == (12, 1, 64, 8)
    assert view.stride() == (512, 2048, 8, 1)
    assert group.kernel_block_stride == 4
    # Kernel block j of block b is page b*kernel_block_stride + j: dim0 of the
    # view is the flat page index, so page 2*4+3=11 holds block (b=2, j=3) at
    # 11*512 elements, plus the in-page (row, col) offset. data_ptr is bytes.
    assert view.data_ptr() == cache.data_ptr()
    assert view[11, 0, 7, 3].data_ptr() == cache.data_ptr() + (11 * 512 + 7 * 8 + 3) * 2


def test_short_cache_delegates_when_no_split_is_needed(
    ppu_platform, monkeypatch
) -> None:
    marker = _stub_upstream_marker(monkeypatch)
    spec = _FakeSpec(kernel_rows=64, block_size=64, tokens_per_state=1)
    group = _FakeGroup(spec)
    cache = torch.empty(4, 1, 64, 8, dtype=torch.bfloat16)
    assert plugin.map_kv_cache_to_kernel_blocks(group, cache) is marker


def test_unsplittable_cache_keeps_upstreams_error(ppu_platform, monkeypatch) -> None:
    _stub_upstream_guard(monkeypatch)
    spec = _FakeSpec(kernel_rows=64, block_size=256, tokens_per_state=1)
    group = _FakeGroup(spec)
    # stride(0) is not a multiple of the page stride: the layout is not the
    # addressable one, so the force-split must refuse instead of
    # mis-addressing, and the caller keeps upstream's error.
    cache = torch.empty(3, 2, 256, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cannot be split into"):
        plugin.map_kv_cache_to_kernel_blocks(group, cache)
