# SPDX-License-Identifier: Apache-2.0
"""PPU patch: narrow the sparse MLA prefill head padding to the kernel tile.

Upstream's ``FlashMLASparseImpl`` hard-codes ``self.prefill_padding = 64``,
the widest query-head tile of the NVIDIA cute prefill kernel, and pads the
prefill query heads up to it before calling ``flash_mla_sparse_fwd``. PPU's
BF16 sparse prefill kernel instead picks its query-head tile from the head
count itself (kBlockM = 16/32/64, see FlashMLA-for-sail
``run_sparse_prefill_fwd_dispatch`` / ``SEQLENG_SWITCH_ALIGN``), so with 16
or 32 heads per rank the padded rows are pure waste: the workspace is
allocated wider and the kernel computes and stores head rows that are sliced
off right after the call.

This patch turns ``FlashMLASparseImpl.prefill_padding`` into a data
descriptor that narrows the value to PPU's actual head tile while the
backend is constructed. It has to be a descriptor rather than a late
assignment: ``__init__`` sizes the prefill workspace (query/out tensors)
from ``self.prefill_padding``, so a post-init change would leave the
workspace wider than the kernel call and trip the shape guards.

Non-PPU platforms keep the upstream flat 64 unchanged.
"""

from __future__ import annotations

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import flashmla_sparse as _flashmla_sparse_backend

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.34.0,<0.35.0"
_MODULE = "vllm.v1.attention.backends.mla.flashmla_sparse"

#: Upstream pads the prefill query heads to 64 rows (``self.prefill_padding
#: = 64``), the cute kernel's widest query-head tile.
_UPSTREAM_PREFILL_PADDING = 64


def _prefill_head_align(num_heads: int, upstream_align: int) -> int:
    """PPU prefill head tile for ``num_heads`` (upstream's flat 64 when off)."""
    if not current_platform.is_ppu() or num_heads <= 0:
        return upstream_align
    if num_heads <= 16:
        return 16
    if num_heads <= 32:
        return 32
    return upstream_align


def _get_prefill_padding(self) -> int:
    return self.__dict__.get("_sail_prefill_padding", _UPSTREAM_PREFILL_PADDING)


def _set_prefill_padding(self, value: int) -> None:
    self.__dict__["_sail_prefill_padding"] = _prefill_head_align(
        getattr(self, "num_heads", 0), value
    )


patch(
    _MODULE,
    "FlashMLASparseImpl.prefill_padding",
    reason=(
        "Narrow the prefill head padding from the flat 64 to PPU's actual "
        "query head tile (kBlockM = 16/32/64). Upstream's 64 is the cute "
        "kernel's widest tile, so with 16 or 32 heads the padded rows are "
        "pure waste; PPU's kernel tiles by head count and handles the "
        "remainder in its epilogue. A data descriptor is required because "
        "the value sizes the prefill workspace inside __init__."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream derives the prefill head padding from the actual head "
        "count instead of hard-coding 64, or PPU's kernel stops tiling by "
        "head count."
    ),
    # Upstream only assigns ``self.prefill_padding`` inside ``__init__``; the
    # name does not exist in the class namespace, so this is an additive
    # descriptor install rather than a replacement of a class attribute.
    allow_missing=True,
)(
    property(
        _get_prefill_padding,
        _set_prefill_padding,
        doc=(
            "Prefill query-head padding: upstream's flat 64, narrowed to "
            "PPU's kernel head tile (16/32/64) on PPU."
        ),
    )
)
