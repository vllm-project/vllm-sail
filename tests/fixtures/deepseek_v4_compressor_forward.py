# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: F821
"""Bare-CPU donor for the provider-owned forward, AST-checked against the pin.

Copied from vLLM 404c2b510e1d73df151d78e5576e1ee7ca0bcaa3. This fixture
lets CPU tests exercise helper installation without importing torch/vLLM.
"""

from __future__ import annotations


def forward(
    self,
    # [num_tokens, 2 * self.coff * self.head_dim]
    kv_score: torch.Tensor,
    # [num_tokens]
    positions: torch.Tensor,
    rotary_emb,
) -> None:
    # Each of shape [num_tokens, coff * self.head_dim]
    # input bf16, output are fp32
    kv, score = kv_score.split(
        [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
    )

    # Get the metadata and handle dummy profiling run.
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        return

    state_metadata = cast(CompressorMetadata, attn_metadata[self.state_cache.prefix])
    slot_mapping = state_metadata.slot_mapping
    block_size = state_metadata.block_size

    # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
    state_cache = self.state_cache.kv_cache
    # kv_state stored in first half, score_state stored in second half
    state_width = state_cache.shape[-1] // 2
    pdl_kwargs = (
        {}
        if current_platform.is_rocm() or current_platform.is_xpu()
        else {"launch_pdl": False}
    )

    # Store the KV and score (with fused APE addition) in the state.
    # NOTE: PDL is disabled — both this kernel and the compress kernels
    # below depend on preceding kernel outputs (kv/score from the cublas
    # GEMM; state_cache from this kernel) but neither emits/waits on PDL
    # grid dependency primitives, so launch_pdl=True caused a
    # read-after-write race and non-deterministic output.
    _SAVE_PARTIAL_STATES_KERNEL(
        kv=kv,
        score=score,
        ape=self.ape,
        positions=positions,
        state_cache=state_cache,
        slot_mapping=slot_mapping,
        block_size=block_size,
        state_width=state_width,
        compress_ratio=self.compress_ratio,
        pdl_kwargs=pdl_kwargs,
    )

    # A captured graph would replay the capture batch's skip decision.
    if (
        current_platform.is_cuda()
        and self.head_dim == 512
        and self.compress_ratio == 128
        and not torch.cuda.is_current_stream_capturing()
        and state_metadata.c128_boundary is False
    ):
        return

    self._compress_norm_rope_store(
        state_metadata,
        state_cache,
        state_width,
        positions,
        rotary_emb,
        attn_metadata,
        pdl_kwargs,
    )
