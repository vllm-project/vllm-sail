# SPDX-License-Identifier: Apache-2.0
"""Layout helpers shared by the SAIL MSA indexer and sparse attend.

The SAIL kernels consume a *compacted* page table (``sum_pages`` entries, one
row per request laid out back to back), while vLLM's block table is a dense
``[requests, max_blocks]`` matrix. They also require per-row ``-1`` padding in
the selected block ids; ``sparse_topk_select`` only takes a scalar
``num_valid_pages``, so the per-token causal clamp is applied here.
"""

from __future__ import annotations

import torch

PAGE_SIZE = 128

_INT32_MAX = torch.iinfo(torch.int32).max


def plan_device(t: torch.Tensor) -> int:
    """Device index to pass to ``fmha_sail_plan``.

    The planner moves host lens to ``torch.device("cuda", device)`` only when
    ``device`` is an int; handed a ``torch.device`` it falls back to ``cuda:0``
    and the kernel then dereferences another GPU's pointers (illegal access as
    soon as the worker is not on GPU 0).
    """
    index = t.device.index
    return index if index is not None else torch.cuda.current_device()


def query_lens(query_start_loc: torch.Tensor, num_reqs: int) -> torch.Tensor:
    """Per-request query lengths ``[num_reqs]`` int32 from the cumulative map."""
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    return qsl[1:] - qsl[:-1]


def flat_page_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
) -> torch.Tensor:
    """Compact ``[requests, max_blocks]`` into the ``sum_pages`` SAIL form.

    ``block_table`` rows are cut to each request's causal page count and
    concatenated in request order, which is exactly the layout the planner
    derives from ``kv_segment_lens``.
    """
    table = block_table[:num_reqs]
    pages = (seq_lens[:num_reqs].to(torch.int64) + PAGE_SIZE - 1) // PAGE_SIZE
    cols = torch.arange(table.shape[1], device=table.device)
    valid = cols.unsqueeze(0) < pages.unsqueeze(1)
    return table[valid].to(torch.int32)


def token_causal_pages(positions: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """``ceil((position + 1) / PAGE_SIZE)`` per query token, int32."""
    return (positions[:num_tokens] // PAGE_SIZE + 1).to(torch.int32)


def clamp_topk_to_causal(topk: torch.Tensor, num_valid_pages: torch.Tensor) -> None:
    """Rewrite out-of-range block ids to ``-1`` and move them to the tail.

    ``topk`` is ``[tokens, kv_heads, topk_blocks]`` int32, written in place.
    Entries at or beyond the token's causal page count were selected from
    all-``-inf`` score padding; SAIL's sparse attend contract wants them
    dropped (``-1``) and the remaining ids strictly ascending. The ``-1``s must
    land at the *tail*: the attend kernel stops at the first ``-1``, so a
    leading one makes the token attend nothing at all.
    """
    keep = (topk >= 0) & (topk < num_valid_pages.unsqueeze(-1).unsqueeze(-1))
    key = topk.masked_fill(~keep, _INT32_MAX).sort(dim=-1).values
    topk.copy_(key.masked_fill_(key == _INT32_MAX, -1))
