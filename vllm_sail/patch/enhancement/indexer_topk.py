# SPDX-License-Identifier: Apache-2.0
"""Run ``sparse_indexer_topk_backend='deep_select'`` on PPU.

Upstream's DeepSelect backend uses ``vllm._deepselect_C`` and admits only the
SM100 family, so an explicit ``deep_select`` raises on PPU. PPU has its own
DeepSelect port (the ``deep_select`` wheel, SM 8.9, BF16 only). On PPU the
``deep_select`` backend runs that port whenever its constraints hold
(``vllm_sail.utils.deep_select``) and otherwise falls back to ``per_row``
(``topKPerRowDecode``, on an exact FP32 upcast) instead of raising. ``auto``
keeps upstream's chain, where DeepSelect is opt-in only; other platforms are
unchanged.
"""

from __future__ import annotations

from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch
from vllm_sail.utils import deep_select

_upstream_forward = SparseIndexerTopk.forward


@patch(
    "vllm.model_executor.layers.indexer_topk",
    "SparseIndexerTopk.forward",
    reason=(
        "Upstream's deep_select backend requires vllm._deepselect_C on the "
        "SM100 family and raises elsewhere; PPU runs the deep_select wheel "
        "(SM 8.9, BF16) and falls back to per_row topKPerRowDecode."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when=(
        "SparseIndexerTopk dispatches DeepSelect through platform "
        "capabilities that PPU can provide."
    ),
)
def _forward(
    self, logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len
):
    if self._backend != "deep_select" or not current_platform.is_ppu():
        return _upstream_forward(
            self, logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len
        )
    backend = deep_select.resolve_backend(
        logits, topk_tokens, self._backend, topk_indices
    )
    if backend == "deep_select":
        row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
        deep_select.topk(logits, topk_tokens, end=row_ends, output_idx=topk_indices)
        return None
    # topKPerRowDecode reads FP32 only; the BF16/FP16 upcast is exact.
    return self._per_row(logits.float(), seq_lens, next_n, topk_indices, topk_tokens)
