# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
"""Kimi K3.1 prefill-context-parallel support for the hybrid MLA/KDA model.

The MLA builder and layer bodies are copied from upstream because their PCP
branches are interleaved with cache insertion and prefill metadata construction.
They stay bound to the upstream module globals, with behavioral changes enclosed
in ``PPU MODIFICATION`` markers. KDA uses additive helpers and thin wrappers so
its existing SAIL PLA prefill/decode dispatch remains in force.
"""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from functools import wraps
from types import ModuleType

import torch
from einops import rearrange
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed.parallel_state import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import mla_attention as _mla_attention
from vllm.models.kimi_k3.common import dsa_cache as _dsa_cache
from vllm.models.kimi_k3.nvidia import dsa as _dsa
from vllm.models.kimi_k3.nvidia import kda as _kda
from vllm.models.kimi_k3.nvidia import mla as _mla
from vllm.models.kimi_k3.nvidia.ops import dsa_indexer_mxfp4 as _mxfp4
from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import _mxfp4_quant
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.v1.worker.gpu import pcp_manager as _pcp_manager

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch, patch_value

_AFFECTED = ">=0.34.0,<0.35.0"
_REASON = (
    "Kimi K3.1 mixes MLA with sequential KDA recurrence and needs PCP-aware "
    "metadata, gathered KV insertion, and globally ordered KDA batches."
)
_REMOVE_WHEN = (
    "Upstream vLLM provides equivalent Kimi K3.1 PCP support, including "
    "MLA split-view metadata, gathered KV insertion, DSA slot gathering, "
    "KDA global-batch reconstruction, and PCP manager global-prefill queries."
)
_MLA_MODULE = "vllm.models.kimi_k3.nvidia.mla"
_KDA_MODULE = "vllm.models.kimi_k3.nvidia.kda"
_KDA_METADATA_MODULE = "vllm.models.kimi_k3.nvidia.kda_metadata"
_DSA_MODULE = "vllm.models.kimi_k3.common.dsa_cache"
_DSA_INDEXER_MODULE = "vllm.models.kimi_k3.nvidia.dsa"
_MLA_COMMON_MODULE = "vllm.model_executor.layers.attention.mla_attention"
_PCP_MODULE = "vllm.v1.worker.gpu.pcp_manager"


# ---------------------------------------------------------------------------
# PCP manager: globally consistent batch state for layer-level collectives.
# ---------------------------------------------------------------------------


@patch(
    _PCP_MODULE,
    "PCPManager.has_global_prefill",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def has_global_prefill(self) -> bool:
    """Whether the global batch schedules any prefill chunk on every rank."""
    global_batch = self._global_batch
    return global_batch is not None and global_batch.has_prefill


patch_value(
    _PCP_MODULE,
    "_current_pcp_manager",
    None,
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)


@patch(
    _PCP_MODULE,
    "get_current_pcp_manager",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def get_current_pcp_manager():
    return getattr(_pcp_manager, "_current_pcp_manager", None)


_upstream_maybe_build_pcp_manager = _pcp_manager.maybe_build_pcp_manager


@patch(
    _PCP_MODULE,
    "maybe_build_pcp_manager",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@wraps(_upstream_maybe_build_pcp_manager)
def maybe_build_pcp_manager(*args, **kwargs):
    manager = _upstream_maybe_build_pcp_manager(*args, **kwargs)
    if manager is not None:
        _pcp_manager._current_pcp_manager = manager
    return manager


# ---------------------------------------------------------------------------
# MLA common metadata: one virtual request per position-consecutive PCP run.
# ---------------------------------------------------------------------------


@patch(
    _MLA_COMMON_MODULE,
    "MLACommonMetadataBuilder._pcp_split_prefill_requests",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _pcp_split_prefill_requests(
    self,
    context_lens_cpu: torch.Tensor,
    prefill_query_start_loc_cpu: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
    num_decode_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Split each PCP prefill request at its position discontinuities.

    Under PCP a rank holds the zigzag chunk pair of each sharded request as
    two position-consecutive runs, while the other ranks' chunks appear in
    the paged cache only after the gathered KV insert. Attention for each
    run therefore decomposes into an unmasked context read of the cache
    rows ``[0, run_start_position)`` (which contains every earlier chunk)
    plus the causal attention among the run's own rows. Returning one
    virtual request per run with ``context_len = run_start_position``
    reuses the chunked-context machinery exactly; without PCP the rows of
    every request are one contiguous run, so this reduces to the identity.

    Returns the split ``context_lens_cpu``, ``prefill_query_start_loc_cpu``
    and the virtual-request -> real block-table-row index, or None when
    positions are unavailable (caller falls back to the unsplit view).
    """
    positions = common_attn_metadata.positions
    if positions is None:
        return None
    prefill_positions = positions[num_decode_tokens:]
    pos_boundary = prefill_positions[1:] - prefill_positions[:-1]
    # One host sync per PCP prefill build; PCP prefill steps are host
    # heavy already and the run layout cannot be derived without it.
    boundary_rows = (
        torch.nonzero(pos_boundary != 1).flatten().add_(1).cpu().tolist()
    )
    boundaries = set(boundary_rows)
    qsl = prefill_query_start_loc_cpu.tolist()
    run_starts: list[int] = []
    run_rows: list[int] = []
    virtual_to_real: list[int] = []
    for real_req, (qs, qe) in enumerate(
        zip(qsl[:-1], qsl[1:]),
    ):
        if qe == qs:
            # Zero-query (graph padding) rows contribute no output; keep
            # them out of the split view entirely.
            continue
        starts = [qs] + [t for t in range(qs + 1, qe) if t in boundaries]
        ends = starts[1:] + [qe]
        run_starts.extend(starts)
        run_rows.extend(end - start for start, end in zip(starts, ends))
        virtual_to_real.extend([real_req] * len(starts))
    if not run_starts:
        return None
    run_start_positions = prefill_positions[
        torch.tensor(run_starts, device=prefill_positions.device)
    ]
    context = run_start_positions.cpu().to(context_lens_cpu.dtype)
    query_start_loc = torch.zeros(
        len(run_rows) + 1, dtype=prefill_query_start_loc_cpu.dtype
    )
    torch.cumsum(
        torch.tensor(run_rows, dtype=query_start_loc.dtype),
        dim=0,
        out=query_start_loc[1:],
    )
    expand_idx = torch.tensor(virtual_to_real, dtype=torch.int64)
    return context, query_start_loc, expand_idx


def _mla_build_body(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> M:
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    max_query_len = common_attn_metadata.max_query_len
    max_seq_len = common_attn_metadata.max_seq_len

    # Note(simon): be careful about the CPU <> GPU memory movement in this
    # function. We should avoid GPU -> CPU sync as much as possible because
    # it blocks on all previous kernels.
    device = self.device
    block_table_tensor = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    query_start_loc = common_attn_metadata.query_start_loc
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    seq_lens = common_attn_metadata.seq_lens
    dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

    non_causal_decode = common_attn_metadata.causal is False
    if non_causal_decode:
        if not (
            self.supports_non_causal_multi_token_decode
            and self.non_causal_multi_token_decode
        ):
            raise ValueError(
                "Non-causal multi-token MLA requires an explicitly supported "
                "attention group."
            )
        # query_start_loc_cpu is host-side and holds one entry per
        # request, so this check costs more in tensor dispatch than in
        # the work it does. Read it out once and validate on Python ints:
        # the block runs on every build, between decode steps.
        qsl = query_start_loc_cpu.tolist()
        query_lens = [qsl[i + 1] - qsl[i] for i in range(len(qsl) - 1)]
        num_active_reqs = 0
        for qlen in query_lens:
            if qlen > 0:
                num_active_reqs += 1
        uniform_active_queries = num_active_reqs > 0 and all(
            qlen == query_lens[0] for qlen in query_lens[:num_active_reqs]
        )
        trailing_graph_padding = all(
            qlen == 0 for qlen in query_lens[num_active_reqs:]
        )
        if not (uniform_active_queries and trailing_graph_padding):
            raise ValueError(
                "Non-causal MLA requires a uniform query block; got query "
                f"lengths {query_lens}."
            )
        # Use exact GPU sequence lengths instead of the prefill path's CPU
        # context-length upper bounds.
        num_decodes = num_reqs
        num_prefills = 0
        num_decode_tokens = num_tokens
        num_prefill_tokens = 0
    else:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
                treat_short_extends_as_decodes=not self.use_pcp,
            )
        )

    assert num_decodes + num_prefills == num_reqs
    assert num_decode_tokens + num_prefill_tokens == num_tokens

    prefill_metadata = None
    if num_prefills > 0:
        reqs_start = num_decodes  # prefill_start

        # Upper bound is exact for prefill rows (no D2H sync).
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        prefill_query_lens_cpu = (
            query_start_loc_cpu[reqs_start + 1 : num_reqs + 1]
            - query_start_loc_cpu[reqs_start:num_reqs]
        )
        context_lens_cpu = torch.empty(
            num_prefills, dtype=seq_lens_cpu.dtype, pin_memory=PIN_MEMORY
        )
        torch.subtract(
            seq_lens_cpu[reqs_start:num_reqs],
            prefill_query_lens_cpu,
            out=context_lens_cpu,
        )
        prefill_query_start_loc = (
            query_start_loc[reqs_start:] - query_start_loc[reqs_start]
        )
        prefill_query_start_loc_cpu = (
            query_start_loc_cpu[reqs_start:] - query_start_loc_cpu[reqs_start]
        )

        # PPU MODIFICATION: begin
        split_view = None
        if self.use_pcp:
            split_view = self._pcp_split_prefill_requests(
                context_lens_cpu,
                prefill_query_start_loc_cpu,
                common_attn_metadata,
                num_decode_tokens,
            )
        if split_view is not None:
            # One virtual request per position run: the chunked context of
            # a run is the cache rows up to the run's first position, and
            # the block table rows duplicate per virtual request.
            # Both the chunked context AND the new-token suffix attention
            # must use the same virtual-run boundaries. If the suffix
            # still used the original (unsplit) qsl, a later run would
            # re-read earlier runs in new-token causal attention *and* in
            # its chunked context, producing duplicate attention mass.
            context_lens_cpu, prefill_query_start_loc_cpu, expand_idx = (
                split_view
            )
            prefill_query_start_loc = prefill_query_start_loc_cpu.to(
                device=prefill_query_start_loc.device
            )
            prefill_query_lens_cpu = prefill_query_start_loc_cpu.diff()
            prefill_block_table = block_table_tensor[reqs_start:][
                expand_idx.to(block_table_tensor.device)
            ]
        else:
            prefill_block_table = block_table_tensor[reqs_start:]
        # PPU MODIFICATION: end

        align_indptr = False
        if (
            self._prefill_backend.get_name() == "TRTLLM_RAGGED"
            and self.vllm_config.attention_config.trtllm_ragged_backend == "cute-dsl"
        ):
            align_indptr = True

        chunked_context_metadata = build_mla_chunked_context_metadata(
            context_lens_cpu=context_lens_cpu,
            prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
            chunked_prefill_workspace=self.chunked_prefill_workspace,
            chunked_prefill_workspace_size=self.chunked_prefill_workspace_size,
            block_size=self.page_size,
            align_chunk_to_block=True,
            device=device,
            dcp_world_size=self.dcp_world_size,
            dcp_local_block_size=self.dcp_local_block_size,
            dcp_virtual_block_size=self.dcp_virtual_block_size,
            dcp_manager=self.dcp_manager,
            align_indptr=align_indptr,
        )

        prefill_metadata = MLACommonPrefillMetadata(
            block_table=prefill_block_table,
            query_start_loc=prefill_query_start_loc,
            max_query_len=max_query_len,
            chunked_context=chunked_context_metadata,
            output_dtype=self.model_config.dtype,
            q_data_type=self.q_data_type,
            prefill_backend=self._prefill_backend,
            query_lens_cpu=prefill_query_lens_cpu,
        )

        self._prefill_backend.prepare_metadata(prefill_metadata)

    decode_metadata = None
    if num_decodes > 0:
        dcp_tot_seq_lens_device = None
        if self.dcp_world_size > 1:
            dcp_tot_seq_lens_device = seq_lens[:num_decodes]
            seq_lens = dcp_local_seq_lens

            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace
            # over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_seq_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

        decode_metadata = self._build_decode(
            block_table_tensor=block_table_tensor[:num_decodes, ...],
            seq_lens_device=seq_lens[:num_decodes],
            max_seq_len=max_seq_len,
            query_start_loc_cpu=query_start_loc_cpu[: num_decodes + 1],
            query_start_loc_device=query_start_loc[: num_decodes + 1],
            num_decode_tokens=num_decode_tokens,
            max_query_len=min(
                common_attn_metadata.max_query_len, self.reorder_batch_threshold
            ),
            dcp_tot_seq_lens_device=dcp_tot_seq_lens_device,
        )

    attn_metadata = self.metadata_cls(
        num_reqs=common_attn_metadata.num_reqs,
        max_query_len=common_attn_metadata.max_query_len,
        max_seq_len=max_seq_len,
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        slot_mapping=slot_mapping,
        head_dim=self.model_config.get_head_size(),
        # MLACommonMetadata Chunk prefill specific
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        causal=not non_causal_decode,
        prefill=prefill_metadata,
        decode=decode_metadata,
    )

    return attn_metadata  # type: ignore[return-value]


patch(
    _MLA_COMMON_MODULE,
    "MLACommonMetadataBuilder.build",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)(bind_body(_mla_build_body, _mla_attention))


# ---------------------------------------------------------------------------
# Kimi dense MLA: gathered cache insertion and concat-only PCP prefill/decode.
# ---------------------------------------------------------------------------


_upstream_mla_init = _mla.MultiHeadLatentAttention.__init__


@patch(
    _MLA_MODULE,
    "MultiHeadLatentAttention.__init__",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@wraps(_upstream_mla_init)
def _mla_init(
    self,
    config,
    hidden_size,
    num_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    q_lora_rank,
    kv_lora_rank,
    use_output_gate=False,
    cache_config=None,
    quant_config=None,
    prefix="",
    aux_stream=None,
    use_rope=False,
    non_causal_multi_token_decode=False,
    run_gemm_rs_ar=False,
) -> None:
    vllm_config = _mla.get_current_vllm_config()
    parallel_config = vllm_config.parallel_config
    use_pcp = parallel_config.prefill_context_parallel_size > 1
    if use_pcp:
        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "Kimi-K3 dense MLA does not support PCP + DCP yet."
            )
        # The clean constructor has not run yet; this is the same value it
        # assigns to self.kv_cache_dtype below.
        cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
        if cache_dtype == "fp8_ds_mla":
            raise NotImplementedError(
                "Kimi-K3 dense MLA does not support PCP with an fp8_ds_mla "
                "cache (the gathered insert uses concat_and_cache_mla)."
            )
        if parallel_config.pcp_shard_decode_requests:
            raise NotImplementedError(
                "Kimi-K3 dense MLA requires replicated decode requests "
                "under PCP (pcp_shard_decode_requests=False)."
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Kimi-K3 does not support speculative decoding with PCP "
                "(the KDA recurrent path assumes spec-free batches)."
            )

        # The clean upstream constructor rejects PCP. Give only this invocation
        # a private config view with PCP disabled; the live config is untouched.
        constructor_config = copy(vllm_config)
        constructor_config.parallel_config = copy(parallel_config)
        constructor_config.parallel_config.prefill_context_parallel_size = 1
        namespace = ModuleType(_upstream_mla_init.__module__)
        namespace.__dict__.update(_upstream_mla_init.__globals__)
        namespace.get_current_vllm_config = lambda: constructor_config
        bind_body(_upstream_mla_init, namespace)(
            self,
            config,
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            use_output_gate,
            cache_config,
            quant_config,
            prefix,
            aux_stream,
            use_rope,
            non_causal_multi_token_decode,
            run_gemm_rs_ar,
        )
    else:
        _upstream_mla_init(
            self,
            config,
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            use_output_gate,
            cache_config,
            quant_config,
            prefix,
            aux_stream,
            use_rope,
            non_causal_multi_token_decode,
            run_gemm_rs_ar,
        )
    self.use_pcp = use_pcp
    if use_pcp:
        # Evaluated lazily on the first PCP step (one sync) and cached.
        self._pcp_skip_k_scale: bool | None = None


@patch(
    _MLA_MODULE,
    "MultiHeadLatentAttention._pcp_gathered_kv_insert",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _pcp_gathered_kv_insert(
    self,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Gather every PCP rank's latents and insert them into the local cache.

    Each rank computes only its zigzag chunks' latents, so a rank's query
    rows (which attend over the whole scheduled query) need the other
    ranks' latents in the paged cache first. Local tensors are padded to
    the per-rank padded token count so the all-gather lines up 1:1 with
    the gathered slot mapping (rank-major, one padded segment per rank).
    Replicated rows are written only by rank 0 (the other ranks' mapping
    entries are PAD_SLOT_ID, which ``concat_and_cache_mla`` skips).
    """
    assert self.impl.__class__.do_kv_cache_update is not None  # type: ignore[attr-defined]
    if self._pcp_skip_k_scale is None:
        self._pcp_skip_k_scale = bool(
            _mla.is_quantized_kv_cache(self.kv_cache_dtype)
            and (self._k_scale == 1.0).all().item()
        )
    pcp_group = get_pcp_group()
    padded = slot_mapping.shape[0] // pcp_group.world_size
    num_tokens = kv_c_normed.shape[0]
    assert num_tokens <= padded

    def pad(t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            (padded, *t.shape[1:]), dtype=t.dtype, device=t.device
        )
        out[:num_tokens] = t
        return out

    k_pe_flat = k_pe.reshape(num_tokens, -1)
    cache_kv_c = pcp_group.all_gather(pad(kv_c_normed).contiguous(), dim=0)
    cache_k_pe = pcp_group.all_gather(pad(k_pe_flat).contiguous(), dim=0)
    cache_positions = pcp_group.all_gather(pad(positions).contiguous(), dim=0)
    if self.rotary_emb is not None:
        # The fused insert kernels apply RoPE on the way in; the unfused
        # gathered insert does it here instead.
        cache_k_pe = self.rotary_emb(cache_positions, cache_k_pe)[0]
    self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
        cache_kv_c,
        cache_k_pe.unsqueeze(1),
        self.kv_cache,
        slot_mapping,
        self.kv_cache_dtype,
        self._k_scale,
        skip_scale=self._pcp_skip_k_scale,
    )


def _attention_body(
    self,
    positions: torch.Tensor,
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    attn_out: torch.Tensor,
) -> None:
    forward_context = get_forward_context()
    attn_metadata_by_layer = forward_context.attn_metadata
    if attn_metadata_by_layer is None:
        attn_out.zero_()
        return
    assert isinstance(attn_metadata_by_layer, dict)
    attn_metadata = cast(
        "MLACommonMetadata", attn_metadata_by_layer[self.layer_name]
    )

    num_actual_toks = attn_metadata.num_actual_tokens
    slot_mapping_by_layer = forward_context.slot_mapping
    assert isinstance(slot_mapping_by_layer, dict)
    slot_mapping = slot_mapping_by_layer[self.layer_name]

    # PPU MODIFICATION: begin
    # Under PCP the layer's slot mapping is the gathered rank-major mapping
    # (one padded segment per rank). When it is expanded, the local latents
    # only cover this rank's zigzag chunks, so every rank's latents must be
    # gathered into the local cache before any attention runs. A collapsed
    # mapping (replicated-only step) keeps the plain local path.
    pcp_gathered = self.use_pcp and slot_mapping.shape[0] > num_actual_toks
    if pcp_gathered:
        self._pcp_gathered_kv_insert(
            kv_c_normed[:num_actual_toks],
            k_pe[:num_actual_toks],
            positions[:num_actual_toks],
            slot_mapping,
        )
    # PPU MODIFICATION: end

    q = q[:num_actual_toks]
    kv_c_normed = kv_c_normed[:num_actual_toks]
    k_pe = k_pe[:num_actual_toks]
    positions = positions[:num_actual_toks]
    attn_out = attn_out[:num_actual_toks]

    cos_sin_cache = None
    rope_positions = None
    if self.rotary_emb is not None:
        # Pass the fp32 cos/sin table straight to the fused epilogue (it reads
        # fp32 and does the RoPE math in fp32) -- no per-forward dtype cast.
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        rope_positions = positions

    # Decode tokens are laid out first, prefill tokens after. The fused
    # prefill covers every supported config (bf16 / plain-fp8 /
    # fp8_ds_mla), so there is no dense-MHA (forward_mha) fallback.
    num_mqa_tokens = attn_metadata.num_decode_tokens
    num_mha_tokens = q.size(0) - num_mqa_tokens

    # Both the prefill and decode fused epilogues write their own cache
    # slice, so there is no separate do_kv_cache_update.

    # ---- Prefill: fused key-concat + cache-insert + attention ----
    if num_mha_tokens > 0:
        self._forward_prefill_fused(
            q[num_mqa_tokens:],
            kv_c_normed[num_mqa_tokens:],
            k_pe[num_mqa_tokens:],
            rope_positions[num_mqa_tokens:] if rope_positions is not None else None,
            cos_sin_cache,
            slot_mapping[num_mqa_tokens:num_actual_toks],
            attn_metadata,
            attn_out[num_mqa_tokens:],
            # PPU MODIFICATION: begin
            pcp_gathered=pcp_gathered,
            # PPU MODIFICATION: end
        )

    # ---- Decode: latent multi-query attention ----
    if num_mqa_tokens > 0:
        mqa_q_nope, mqa_q_pe = q[:num_mqa_tokens].split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        # BMM1: absorb q_nope into latent space. (N,B,P) x (N,P,L) -> (B,N,L)
        ql_nope = torch.bmm(mqa_q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
        # PPU MODIFICATION: begin
        if pcp_gathered:
            # The decode latents were already inserted by the gathered KV
            # insert (replicated decode rows carry rank 0's real slots;
            # other ranks' rows are PAD and skipped), so only build the
            # concatenated MQA query here.
            if rope_positions is not None:
                mqa_q_pe = self.rotary_emb(
                    rope_positions[:num_mqa_tokens], mqa_q_pe
                )[0]
            mqa_q = torch.cat((ql_nope, mqa_q_pe), dim=-1)
        else:
            # Fused: concat mqa_q = [ql_nope | q_pe] and insert the
            # decode-token latent into the paged cache (one launch, right
            # before forward_mqa).
            mqa_q = self._decode_concat_cache(
                ql_nope,
                mqa_q_pe,
                kv_c_normed[:num_mqa_tokens],
                k_pe[:num_mqa_tokens],
                rope_positions[:num_mqa_tokens]
                if rope_positions is not None
                else None,
                cos_sin_cache,
                slot_mapping[:num_mqa_tokens],
            )
        # PPU MODIFICATION: end
        if self.dcp_world_size > 1:
            assert self.dcp_manager is not None
            assert self.dcp_manager.query_gather is not None
            mqa_q = self.dcp_manager.query_gather(mqa_q)
        latent_out, lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
            mqa_q, self._attn_read_kv_cache(), attn_metadata, self
        )
        if self.dcp_world_size > 1:
            assert lse is not None
            assert self.dcp_manager is not None
            assert attn_metadata.decode is not None
            seq_lens = attn_metadata.decode.seq_lens
            query_start_loc = attn_metadata.query_start_loc[
                : attn_metadata.num_decodes + 1
            ]
            if self.W_UV_dcp is not None:
                assert self.dcp_manager.v_up_proj_combine is not None
                self.dcp_manager.v_up_proj_combine(
                    latent_out,
                    lse,
                    self.W_UV_dcp,
                    attn_out[:num_mqa_tokens],
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
                return
            latent_out = self.dcp_manager.combine(
                latent_out,
                lse,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
            )
        self._v_up_proj(latent_out, out=attn_out[:num_mqa_tokens])


patch(
    _MLA_MODULE,
    "MultiHeadLatentAttention._attention",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)(_mla.eager_break_during_capture(bind_body(_attention_body, _mla)))


def _forward_prefill_fused_body(
    self,
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    attn_metadata,
    out: torch.Tensor,
    pcp_gathered: bool = False,
) -> None:
    """Prefill using the fused key-concat + cache-insert kernel.

    Replaces ``_concat_k_nope_k_pe`` and the prefill cache write with one
    fused kernel launch, dispatched by cache dtype. Chunked context runs
    through this layer's ``_compute_prefill_context``, except under DCP where
    it is delegated to the impl.

    Under ``pcp_gathered`` the KV cache was already filled by
    ``_pcp_gathered_kv_insert`` (all ranks' latents), so only the
    concat-only K packing runs here -- no fused cache insert.

    Supported configs (K3 fp8 policy):
      - bf16 cache        -> bf16 prefill query
      - plain fp8 cache   -> fp8 prefill query (unscaled q/k/v; cache _k_scale)
      - fp8_ds_mla cache  -> bf16 prefill query (656B per-tile self-scaled)
    """
    prefill = attn_metadata.prefill
    has_context = prefill.chunked_context is not None
    fp8_prefill = prefill.q_data_type == current_platform.fp8_dtype()

    kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
        -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
    )
    k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    # PPU MODIFICATION: begin
    if pcp_gathered:
        # Cache already holds all ranks' latents (gathered insert); build
        # the new-token K/V without touching the cache.
        if fp8_prefill:
            q = q.to(current_platform.fp8_dtype())
            k, v = fused_mla_kv_pack_quant_fp8(
                kv_nope, k_pe, nope_dim=self.qk_nope_head_dim
            )
        else:
            k = fused_mla_kv_concat(k_nope, k_pe)
    # PPU MODIFICATION: end
    elif self.kv_cache_dtype == "fp8_ds_mla":
        # fp8_ds_mla cache (656B, per-tile self-scaled); bf16 attention.
        assert not fp8_prefill, (
            "Kimi-K3 fp8_ds_mla uses a bf16 prefill query; fp8 prefill "
            "query is not supported with fp8_ds_mla."
        )
        kv_cache = self.kv_cache
        if kv_cache.dtype != torch.uint8:
            kv_cache = kv_cache.view(torch.uint8)
        k = fused_mla_key_concat_ds_mla_insert(
            q,
            k_nope,
            k_pe,
            kv_c_normed,
            kv_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
        )
    elif is_quantized_kv_cache(self.kv_cache_dtype):
        assert fp8_prefill, (
            "Kimi-K3 fp8 KV cache requires an fp8 prefill query; enable "
            "--attention-config '{\"use_prefill_query_quantization\": true}'."
        )
        # Plain per-tensor fp8: quant q/k/v (unscaled, matching forward_mha's
        # unscaled `.to(fp8)`) and insert the fp8 latent (scaled by _k_scale).
        kv_cache = self.kv_cache
        if kv_cache.dtype != torch.float8_e4m3fn:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        q, k, v = fused_mla_qkv_quant_kv_cache_fp8_insert(
            q,
            k_nope,
            k_pe,
            kv_c_normed,
            v,
            kv_cache,
            slot_mapping,
            self._one_scale,
            self._one_scale,
            self._one_scale,
            self._k_scale_inv,
            positions,
            cos_sin_cache,
        )
    else:
        # Concat full K = [k_nope | k_pe] and insert [kv_c_normed | k_pe]
        # into the paged cache for these prefill tokens, in one launch.
        k = fused_mla_key_concat_kv_cache_insert(
            q,
            k_nope,
            k_pe,
            kv_c_normed,
            self.kv_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
        )

    # When there is no chunked context, backends that honor `out` write the
    # attention result straight into it, avoiding a slice+flatten+copy.
    writes_out = not has_context and prefill.prefill_backend.supports_out()
    output_prefill = prefill.prefill_backend.run_prefill_new_tokens(
        q=q,
        k=k,
        v=v,
        return_softmax_lse=has_context,
        out=(
            out.view(-1, self.num_local_heads, self.v_head_dim)
            if writes_out
            else None
        ),
    )

    if has_context:
        if self.dcp_world_size > 1:
            context_output, context_lse = (
                self.impl._context_parallel_compute_prefill_context(  # type: ignore[attr-defined]
                    q,
                    self._attn_read_kv_cache(),
                    attn_metadata,
                    k_scale=self._k_scale,
                    dcp_world_size=self.dcp_world_size,
                    fused_mla_kv_concat_fn=self._fused_mla_kv_concat,
                )
            )
        else:
            context_output, context_lse = self._compute_prefill_context(
                q, attn_metadata
            )
        suffix_output, suffix_lse = output_prefill
        out = out.view(-1, self.num_local_heads, self.v_head_dim)
        merge_attn_states(
            output=out,
            prefix_output=context_output[..., : self.v_head_dim],
            prefix_lse=context_lse,
            suffix_output=suffix_output[..., : self.v_head_dim],
            suffix_lse=suffix_lse,
        )
    elif not writes_out:
        out.copy_(output_prefill[..., : self.v_head_dim].flatten(start_dim=-2))


patch(
    _MLA_MODULE,
    "MultiHeadLatentAttention._forward_prefill_fused",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)(bind_body(_forward_prefill_fused_body, _mla))


# ---------------------------------------------------------------------------
# Kimi DSA indexer: compress the local segment, then gather fixed-size buffers.
# ---------------------------------------------------------------------------


def _dsa_init(self, *args, **kwargs) -> None:
    super(KimiK3IndexerMetadataBuilder, self).__init__(*args, **kwargs)
    # The K3.1 key format, validated as KimiK3DSAIndexer selects it.
    self.indexer_uses_fp4 = kimi_k3_indexer_uses_fp4(self.vllm_config)
    self.slot_mapping_buffer = torch.zeros(
        self.vllm_config.scheduler_config.max_num_batched_tokens,
        dtype=torch.int64,
        device=self.device,
    )


patch(
    _DSA_MODULE,
    "KimiK3IndexerMetadataBuilder.__init__",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)(
    bind_body(_dsa_init, _dsa_cache)
)


def _dsa_build(
    self,
    common_prefix_len: int,
    common_attn_metadata,
    fast_build: bool = False,
):
    slot_mapping = common_attn_metadata.slot_mapping
    # PPU MODIFICATION: begin
    pcp_expanded = self.use_pcp and (
        slot_mapping.shape[0] > common_attn_metadata.num_actual_tokens
    )
    if pcp_expanded:
        # The layer slot mapping is the gathered rank-major mapping (one
        # padded segment per rank); derive this rank's compressed slots
        # from its own segment, then re-assemble the gathered mapping so
        # every rank inserts all ranks' indexer keys (the sparse indexer
        # reads them back from the cache under PCP). This body is rebound to
        # the upstream module, so import patch-only dependencies locally.
        from vllm.distributed.parallel_state import get_pcp_group

        pcp_group = get_pcp_group()
        padded_num_tokens = slot_mapping.shape[0] // pcp_group.world_size
        slot_mapping = slot_mapping[
            self.pcp_rank
            * padded_num_tokens : (self.pcp_rank + 1)
            * padded_num_tokens
        ]
    # PPU MODIFICATION: end
    slot_mapping = get_compressed_slot_mapping(
        common_attn_metadata.num_actual_tokens,
        slot_mapping,
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.block_table_tensor,
        self.kv_cache_spec.block_size,
        compress_ratio=1,
        out=self.slot_mapping_buffer,
        dcp_world_size=self.dcp_world_size,
        cp_interleave=self.cp_kv_cache_interleave_size,
    )
    # PPU MODIFICATION: begin
    if pcp_expanded:
        # Every rank must contribute the same number of rows. The compressed
        # helper fills unused output slots with -1, so gather the padded
        # buffer rather than its rank-dependent actual-token prefix.
        slot_mapping = pcp_group.all_gather(
            self.slot_mapping_buffer[:padded_num_tokens], dim=0
        )
    # PPU MODIFICATION: end
    metadata = super(KimiK3IndexerMetadataBuilder, self).build(
        common_prefix_len,
        replace(common_attn_metadata, slot_mapping=slot_mapping),
        fast_build,
    )
    return metadata


patch(
    _DSA_MODULE,
    "KimiK3IndexerMetadataBuilder.build",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)(
    bind_body(_dsa_build, _dsa_cache)
)


# ---------------------------------------------------------------------------
# Kimi DSA MXFP4: allow the gathered unfused writer for supported PCP configs.
# ---------------------------------------------------------------------------


def _kimi_k3_indexer_uses_fp4(vllm_config: VllmConfig) -> bool:
    """Select the Kimi indexer cache format, including the PCP FP4 writer."""
    if not dsa_indexer_uses_fp4(vllm_config):
        return False
    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    dsa_config = getattr(text_config, "dsa_config", None)
    if dsa_config is not None and not (
        getattr(dsa_config, "index_k_norm_type", "rmsnorm") == "layernorm"
        and getattr(dsa_config, "index_rope_dim", 0) == 0
    ):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' needs the fused Kimi K3 DSA key insert "
            "(index_k_norm_type='layernorm', index_rope_dim=0)."
        )
    # PPU MODIFICATION: begin
    # PCP uses the gathered normalized-key MXFP4 writer below.
    # PPU MODIFICATION: end
    return True


# dsa.py imported this function directly, so both live references must change.
for _fp4_guard_module in (_DSA_MODULE, _DSA_INDEXER_MODULE):
    patch(
        _fp4_guard_module,
        "kimi_k3_indexer_uses_fp4",
        reason=(
            "Remove the stale MXFP4/PCP rejection after adding the gathered "
            "normalized-key MXFP4 cache writer."
        ),
        affected_versions=_AFFECTED,
        remove_when=(
            "upstream Kimi K3.1 natively supports MXFP4 indexer caches with PCP."
        ),
    )(bind_body(_kimi_k3_indexer_uses_fp4, _dsa_cache))


def _dsa_indexer_init(
    self,
    vllm_config: VllmConfig,
    hidden_size: int,
    index_q_input_size: int,
    index_head_dim: int,
    index_num_heads: int,
    index_topk: int,
    rms_norm_eps: float,
    index_rope_dim: int = 0,
    index_k_norm_type: str = "rmsnorm",
    cache_config: CacheConfig | None = None,
    quant_config: QuantizationConfig | None = None,
    topk_indices_buffer: torch.Tensor | None = None,
    prefix: str = "",
) -> None:
    nn.Module.__init__(self)
    self.index_head_dim = index_head_dim
    self.index_num_heads = index_num_heads
    self.indexer_weight_scale = self.index_num_heads**-0.5
    self.topk_tokens = index_topk
    if index_rope_dim < 0 or index_rope_dim % 2 != 0:
        raise ValueError(
            f"dsa_config.index_rope_dim must be even and >= 0, got {index_rope_dim}"
        )
    if index_rope_dim > index_head_dim:
        raise ValueError(
            f"dsa_config.index_rope_dim ({index_rope_dim}) must not "
            f"exceed index_head_dim ({index_head_dim})"
        )
    self.index_rope_dim = index_rope_dim
    self.wq_index = ReplicatedLinear(
        index_q_input_size,
        self.index_num_heads * self.index_head_dim,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wq_index",
    )
    self.wkv_index = MergedColumnParallelLinear(
        hidden_size,
        [self.index_head_dim, self.index_num_heads],
        bias=False,
        quant_config=None,
        disable_tp=True,
        prefix=f"{prefix}.wkv_index",
    )
    self.k_index_norm: nn.Module
    if index_k_norm_type == "layernorm":
        self.k_index_norm = KimiK3DSAIndexerKNorm(
            self.index_head_dim, eps=rms_norm_eps
        )
    elif index_k_norm_type == "rmsnorm":
        self.k_index_norm = RMSNorm(self.index_head_dim, eps=rms_norm_eps)
    else:
        raise ValueError(
            f"dsa_config.index_k_norm_type must be 'rmsnorm' or "
            f"'layernorm', got {index_k_norm_type!r}"
        )

    self.scale_fmt = "ue8m0"
    self.quant_block_size = self.index_head_dim
    self.index_softmax_scale = self.index_head_dim**-0.5
    if self.index_num_heads <= 32:
        self.padded_index_num_heads = 32
    elif self.index_num_heads <= 64:
        self.padded_index_num_heads = 64
    else:
        raise ValueError(
            "Official DeepGEMM DSA indexer supports at most 64 index "
            f"heads, got {self.index_num_heads}."
        )
    self.use_fp4_cache = kimi_k3_indexer_uses_fp4(vllm_config)
    self.k_cache = KimiK3IndexerCache(
        head_dim=indexer_cache_entry_bytes(self.index_head_dim, self.use_fp4_cache),
        dtype=torch.uint8,
        prefix=f"{prefix}.k_cache",
        cache_config=cast(CacheConfig, cache_config),
    )
    self.max_model_len = vllm_config.model_config.max_model_len
    self.max_total_seq_len = v32_indexer.get_max_prefill_buffer_size(vllm_config)
    self.topk_indices_buffer = topk_indices_buffer
    self.use_pcp = vllm_config.parallel_config.prefill_context_parallel_size > 1
    self.dense_mha_metadata_layer_name = ""
    self.topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
    self.use_fused_k_insert = (
        not self.use_pcp
        and index_k_norm_type == "layernorm"
        and index_rope_dim == 0
    )
    # PPU MODIFICATION: begin
    pcp_unfused_fp4 = (
        self.use_pcp
        and index_k_norm_type == "layernorm"
        and index_rope_dim == 0
    )
    if (
        self.use_fp4_cache
        and not self.use_fused_k_insert
        and not pcp_unfused_fp4
    ):
        raise NotImplementedError(
            "MXFP4 indexer keys need the fused key insert, except for the "
            "supported PCP LayerNorm path without indexer RoPE"
        )
    # PPU MODIFICATION: end
    self._parallel_config = vllm_config.parallel_config
    self.dcp_world_size = self._parallel_config.decode_context_parallel_size
    self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
    self._cp_kv_cache_interleave_size: int | None = None
    self.dcp_local_topk = init_dcp_topk_merge(vllm_config, index_topk)


patch(
    _DSA_INDEXER_MODULE,
    "KimiK3DSAIndexer.__init__",
    reason=(
        "Permit only the implemented FP4 unfused configuration: PCP with "
        "LayerNorm and no indexer RoPE. Other unfused FP4 configurations keep "
        "the upstream rejection."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream Kimi K3.1 supports unfused MXFP4 key insertion with PCP.",
)(bind_body(_dsa_indexer_init, _dsa))


# ---------------------------------------------------------------------------
# Kimi DSA MXFP4: gather normalized keys, then quantize and store them.
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_k_quant_store(
    y,
    slot,
    cache_ptr,
    cache_block_size,
    cache_block_stride,
    HEAD_DIM: tl.constexpr,
):
    """Store one normalized key in the shared 68-byte MXFP4 cache layout."""
    NUM_SCALES: tl.constexpr = HEAD_DIM // 32
    packed, ue8m0 = _mxfp4_quant(y, HEAD_DIM)
    block_start = (slot // cache_block_size) * cache_block_stride
    block_off = slot % cache_block_size
    data = cache_ptr + block_start + block_off * (HEAD_DIM // 2)
    tl.store(data + tl.arange(0, HEAD_DIM // 2), packed)
    scale_byte = (
        block_start + cache_block_size * (HEAD_DIM // 2) + block_off * NUM_SCALES
    )
    tl.store(cache_ptr + scale_byte + tl.arange(0, NUM_SCALES), ue8m0)


@patch(
    "vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4",
    "_mxfp4_k_insert",
    reason=(
        "Split the MXFP4 quantization/cache store from LayerNorm so the fused "
        "writer and PCP's gathered unfused writer share identical cache bytes."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream exposes a normalized-key MXFP4 cache writer and shares its "
        "store implementation with the fused writer."
    ),
)
@triton.jit
def _mxfp4_k_insert(
    tok,
    k_ptr,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_block_size,
    cache_block_stride,
    HEAD_DIM: tl.constexpr,
):
    # Padded rows and (under DCP) tokens owned by another rank have no slot.
    slot = tl.load(slot_mapping_ptr + tok, mask=tok < num_slots, other=-1)
    if slot >= 0:
        offs = tl.arange(0, HEAD_DIM)
        x = tl.load(k_ptr + offs).to(tl.float32)
        mean = tl.sum(x, axis=0) / HEAD_DIM
        diff = x - mean
        rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / HEAD_DIM + norm_eps)
        w = tl.load(norm_w_ptr + offs).to(tl.float32)
        b = tl.load(norm_b_ptr + offs).to(tl.float32)
        y = (diff * rstd * w + b).to(k_ptr.dtype.element_ty).to(tl.float32)
        # PPU MODIFICATION: begin
        _mxfp4_k_quant_store(
            y, slot, cache_ptr, cache_block_size, cache_block_stride, HEAD_DIM
        )
        # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4",
    "_mxfp4_k_quant_insert",
    allow_missing=True,
    reason=(
        "PCP gathers keys after Kimi's unfused k_index_norm; this kernel "
        "quantizes those normalized rows and skips negative PAD/ownership slots."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream provides an unfused normalized-key MXFP4 insert kernel.",
)
@triton.jit
def _mxfp4_k_quant_insert(
    k_ptr,
    k_stride,
    slot_mapping_ptr,
    cache_ptr,
    cache_block_size,
    cache_block_stride,
    HEAD_DIM: tl.constexpr,
):
    tok = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + tok)
    if slot >= 0:
        y = tl.load(k_ptr + tok * k_stride + tl.arange(0, HEAD_DIM)).to(tl.float32)
        _mxfp4_k_quant_store(
            y, slot, cache_ptr, cache_block_size, cache_block_stride, HEAD_DIM
        )


@patch(
    "vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4",
    "mxfp4_k_quant_and_cache",
    allow_missing=True,
    reason=(
        "Expose the normalized-key MXFP4 writer needed after PCP gathers the "
        "complete key sequence; unlike the fused entry this does not normalize "
        "again or clear the top-k buffer."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream provides an unfused normalized-key MXFP4 cache writer.",
)
def mxfp4_k_quant_and_cache(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
) -> None:
    """Quantize normalized ``[T, D]`` keys and insert all valid cache slots."""
    num_tokens = slot_mapping.numel()
    if k.ndim != 2 or slot_mapping.ndim != 1 or k.shape[0] != num_tokens:
        raise ValueError("normalized keys and cache slot mapping must align by row")
    if num_tokens == 0:
        return
    head_dim = k.shape[-1]
    cache = _mxfp4._cache_bytes(kv_cache, head_dim)
    _mxfp4_k_quant_insert[(num_tokens,)](
        k,
        k.stride(0),
        slot_mapping,
        cache,
        cache.shape[1],
        cache.stride(0),
        HEAD_DIM=head_dim,
        num_warps=1,
    )


@eager_break_during_capture
def _pcp_mxfp4_gather_and_store(indexer, index_k: torch.Tensor) -> None:
    """Keep PCP collectives and cache writes in the same eager graph segment."""
    # Resolve metadata at execution time: graph replay must use the current
    # step's slot mapping rather than retaining the capture-step metadata.
    attn_metadata = _dsa.get_forward_context().attn_metadata
    # Metadata dictionaries use plain module-name strings. LayerName is an
    # opaque custom-op argument, not a valid dictionary key.
    metadata = attn_metadata[indexer.k_cache.prefix]
    slot_mapping = metadata.slot_mapping
    num_tokens = slot_mapping.shape[0]
    if num_tokens > index_k.shape[0]:
        num_tokens //= get_pcp_group().world_size
    index_k = index_k[:num_tokens]
    cache_k, cache_slot_mapping = maybe_gather_indexer_k(
        index_k,
        slot_mapping,
        metadata.num_decode_tokens,
        indexer.use_pcp,
        pcp_shard_decode_requests=(
            indexer._parallel_config.pcp_shard_decode_requests
        ),
    )
    _mxfp4.mxfp4_k_quant_and_cache(
        cache_k, cache_slot_mapping, indexer.k_cache.kv_cache
    )


def _pcp_mxfp4_k_cache_insert(
    indexer, index_k: torch.Tensor | None
) -> tuple[torch.Tensor | None, bool]:
    """Write gathered FP4 keys and return the inputs for the generic scorer."""
    skip_k_cache_insert = indexer.use_fused_k_insert
    if not (
        indexer.use_fp4_cache
        and indexer.use_pcp
        and not indexer.use_fused_k_insert
        and index_k is not None
    ):
        return index_k, skip_k_cache_insert

    # Profiling/dummy runs have no real metadata. The generic op still reserves
    # its workspaces, but no cache insertion or metadata lookup may happen here.
    attn_metadata = _dsa.get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return index_k, skip_k_cache_insert

    _pcp_mxfp4_gather_and_store(indexer, index_k)
    return None, True


def _indexer_score_body(
    self,
    hidden_states: torch.Tensor,
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    index_k: torch.Tensor | None,
) -> torch.Tensor:
    """Top-k of the current rows into ``topk_indices_buffer``; ``index_k`` is
    None when the key was already inserted (``use_fused_k_insert``)."""
    weight_scale = self.index_softmax_scale * self.indexer_weight_scale
    q_scale: torch.Tensor | None = None
    if self.use_fp4_cache:
        q_quant, q_scale, weights = mxfp4_q_quant(
            index_q, index_weights, weight_scale, self.padded_index_num_heads
        )
    else:
        q_quant, weights = indexer_q_quant_fold_weights(
            index_q, index_weights, weight_scale, self.padded_index_num_heads
        )
    # PPU MODIFICATION: begin
    from vllm_sail.patch.enhancement.models.kimi_k3_pcp import (
        _pcp_mxfp4_k_cache_insert,
    )

    index_k, skip_k_cache_insert = _pcp_mxfp4_k_cache_insert(self, index_k)
    # PPU MODIFICATION: end
    return torch.ops.vllm.sparse_attn_indexer(
        hidden_states,
        _encode_layer_name(self.k_cache.prefix),
        self.k_cache.kv_cache,
        q_quant,
        q_scale,
        index_k,
        weights,
        self.quant_block_size,
        self.scale_fmt,
        self.topk_tokens,
        self.index_head_dim,
        self.max_model_len,
        self.max_total_seq_len,
        self.topk_indices_buffer,
        skip_k_cache_insert,
        self.use_pcp,
        self._parallel_config.pcp_shard_decode_requests,
        _encode_layer_name(self.dense_mha_metadata_layer_name),
        self.use_fp4_cache,
        self.dcp_rank,
        self.dcp_world_size,
        self.cp_kv_cache_interleave_size,
        skip_topk_buffer_clear=self.use_fused_k_insert,
        topk_backend=self.topk_backend,
    )


patch(
    "vllm.models.kimi_k3.nvidia.dsa",
    "KimiK3DSAIndexer.score",
    reason=(
        "Kimi has already normalized index_k before the unfused PCP path. "
        "Gather and MXFP4-store it before scoring, then suppress the generic "
        "FP8-only writer while preserving its top-k buffer clear."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream sparse_attn_indexer supports gathered unfused MXFP4 inserts.",
)(bind_body(_indexer_score_body, _dsa))


# ---------------------------------------------------------------------------
# KDA: reconstruct a de-duplicated global sequence for the ordered recurrence.
# ---------------------------------------------------------------------------


@patch(
    _KDA_METADATA_MODULE,
    "KimiK3KDAAttentionBackend.supports_pcp",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@classmethod
def supports_pcp(cls) -> bool:
    # PCP is implemented by the model-level global recurrence in this patch,
    # rather than by GDNAttentionImpl. The inherited backend probe cannot
    # discover it and would reject Kimi K3 during KV-cache initialization.
    return True


@patch(
    _KDA_MODULE,
    "_reconstruct_pcp_global_batch",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _reconstruct_pcp_global_batch(
    positions: torch.Tensor,
    state_slots: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Rebuild ordered, de-duplicated PCP tokens and their request metadata.

    Returns the selected gathered-row indices, a gathered-row to ordered-token
    inverse map, query start locations, per-request initial-state flags, and
    per-request state slots. Padding rows have state slot -1 and inverse -1.
    """
    # torch has no lexsort. Sorting the secondary key and then stably sorting
    # the primary key gives the same (state slot, position) lexicographic order.
    order = torch.argsort(positions, stable=True)
    order = order[torch.argsort(state_slots[order], stable=True)]
    sorted_state = state_slots[order]
    sorted_pos = positions[order]
    valid = sorted_state >= 0

    # Keep one copy of each (state slot, position). Every duplicate still gets
    # the same inverse ordinal below, which is required for replicated rows.
    token_start = valid.clone()
    token_start[1:] &= (sorted_state[1:] != sorted_state[:-1]) | (
        sorted_pos[1:] != sorted_pos[:-1]
    )
    global_idx = order[token_start]
    sel_pos = sorted_pos[token_start]
    sel_state = sorted_state[token_start]

    token_ordinal = torch.cumsum(token_start.to(torch.int64), dim=0) - 1
    inverse = torch.full(
        state_slots.shape, -1, dtype=torch.int64, device=state_slots.device
    )
    inverse[order[valid]] = token_ordinal[valid]

    # State slots delimit requests; position changes only delimit tokens.
    request_start = torch.ones_like(sel_state, dtype=torch.bool)
    request_start[1:] = sel_state[1:] != sel_state[:-1]
    token_req_ordinal = torch.cumsum(request_start.to(torch.int64), dim=0) - 1
    num_global_reqs = int(request_start.sum().item())
    global_q_lens = torch.bincount(
        token_req_ordinal, minlength=num_global_reqs
    ).to(torch.int32)
    global_qsl = torch.zeros(
        num_global_reqs + 1, dtype=torch.int32, device=positions.device
    )
    torch.cumsum(global_q_lens, dim=0, out=global_qsl[1:])

    has_initial = sel_pos[request_start] > 0
    request_state = sel_state[request_start]
    return global_idx, inverse, global_qsl, has_initial, request_state


_upstream_kda_init = _kda.KimiK3DeltaAttention.__init__
_upstream_kda_forward = _kda.KimiK3DeltaAttention.forward


@patch(
    _KDA_MODULE,
    "KimiK3DeltaAttention.__init__",
    allow_wrapping=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@wraps(_upstream_kda_init)
def _kda_init(
    self,
    config,
    vllm_config,
    prefix: str = "",
    aux_stream=None,
    run_gemm_rs_ar: bool = False,
) -> None:
    _upstream_kda_init(
        self, config, vllm_config, prefix, aux_stream, run_gemm_rs_ar
    )
    self.use_pcp = vllm_config.parallel_config.prefill_context_parallel_size > 1


@patch(
    _KDA_MODULE,
    "KimiK3DeltaAttention._forward_pcp",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _forward_pcp(
    self,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    """Prefill-context-parallel path for the KDA recurrence.

    PCP gives each rank a zigzag slice of every prefill's scheduled tokens,
    but the linear-attention recurrence is inherently sequential over the
    whole sequence. On steps whose global batch contains prefill chunks,
    every rank all-gathers the batch, reconstructs a globally-ordered
    (per-request contiguous) view, runs the full-sequence KDA compute, and
    slices its local rows back out. Replicated decode rows (present on all
    ranks) are de-duplicated by (state slot, position). All ranks then
    write identical recurrent/conv state updates, keeping their state
    pools in sync. Decode-only steps return None and take the local path.
    """
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    pcp_manager = get_current_pcp_manager()
    if pcp_manager is None or not pcp_manager.has_global_prefill():
        return None
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    m = attn_metadata.get(self.prefix)
    if not isinstance(m, _kda.KimiK3KDAMetadata):
        return None

    pcp_group = get_pcp_group()
    pcp_rank = pcp_group.rank_in_group
    padded = hidden_states.size(0)

    # Per-token state-slot identity: groups tokens by request across ranks
    # and marks local padding rows with -1 (excluded after the gather).
    query_start_loc = m.non_spec_query_start_loc
    if query_start_loc is None:
        raise RuntimeError("PCP KDA prefill requires non_spec_query_start_loc")
    num_reqs = query_start_loc.shape[0] - 1
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    state_per_req = m.non_spec_state_indices_tensor
    if state_per_req is None:
        state_per_req = torch.zeros(
            num_reqs, dtype=torch.int64, device=hidden_states.device
        )
    state_tok = torch.repeat_interleave(
        state_per_req[:num_reqs].to(torch.int64), q_lens
    )
    pad_rows = padded - state_tok.shape[0]
    if pad_rows > 0:
        state_tok = torch.cat([state_tok, state_tok.new_full((pad_rows,), -1)])

    gathered_hidden = pcp_group.all_gather(hidden_states.contiguous(), dim=0)
    gathered_pos = pcp_group.all_gather(positions.contiguous(), dim=0)
    gathered_state = pcp_group.all_gather(state_tok.contiguous(), dim=0)

    (
        global_idx,
        inverse,
        global_qsl,
        has_initial,
        request_state,
    ) = _reconstruct_pcp_global_batch(gathered_pos, gathered_state)
    num_global_tokens = int(global_idx.numel())
    if num_global_tokens == 0:
        return None
    num_global_reqs = global_qsl.numel() - 1

    # Projection over the global token set.
    hidden_global = gathered_hidden.index_select(0, global_idx)
    projected_qkvgfab = self.in_proj_qkvgfab(hidden_global)[0]
    split_sizes = [
        3 * self.local_projection_size,
        self.local_projection_size,
        self.head_dim,
        self.local_num_heads,
    ]
    if self.in_proj_padding:
        split_sizes.append(self.in_proj_padding)
    projected = projected_qkvgfab.split(split_sizes, dim=-1)
    mixed_qkv, g_proj_states, f_a, beta = projected[:4]
    g1 = rearrange(
        self.f_b_proj(f_a)[0], "n (h d) -> 1 n h d", d=self.head_dim
    )
    g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
    beta = beta.unsqueeze(0)
    core_attn_out = torch.empty(
        (1, num_global_tokens, self.local_num_heads, self.head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        global_qsl.cpu(), device=hidden_states.device
    )
    flashinfer_qsl = None
    flashinfer_seq_order = None
    if self.kda_prefill_backend == "flashinfer":
        flashinfer_qsl = global_qsl.to(torch.int64)
        if num_global_tokens > num_global_reqs:
            flashinfer_seq_order = torch.argsort(
                flashinfer_qsl.diff(), descending=True
            ).to(torch.int32)

    global_m = _kda.KimiK3KDAMetadata(
        num_prefills=num_global_reqs,
        num_prefill_tokens=num_global_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_global_tokens,
        has_initial_state=has_initial,
        non_spec_query_start_loc=global_qsl,
        non_spec_state_indices_tensor=request_state,
        flashinfer_prefill_query_start_loc=flashinfer_qsl,
        flashinfer_prefill_seq_order=flashinfer_seq_order,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    original_m = attn_metadata.get(self.prefix)
    attn_metadata[self.prefix] = global_m
    try:
        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )
    finally:
        attn_metadata[self.prefix] = original_m

    core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
    local_rows = inverse[pcp_rank * padded : (pcp_rank + 1) * padded].clamp(min=0)
    out_local = core_attn_out[local_rows]
    return self.o_proj(out_local)[0]


@patch(
    _KDA_MODULE,
    "KimiK3DeltaAttention.forward",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@wraps(_upstream_kda_forward)
def _kda_forward(
    self,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if self.use_pcp:
        pcp_out = self._forward_pcp(hidden_states, positions)
        if pcp_out is not None:
            return pcp_out
    return _upstream_kda_forward(self, hidden_states, positions)
