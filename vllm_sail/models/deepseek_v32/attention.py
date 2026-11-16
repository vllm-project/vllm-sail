# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU attention for the new DeepSeek V3.2 main and MTP stack.

The model structure and MLA weight absorption come from upstream vLLM
4ba787ec9959. PPU uses its registered norm/RoPE/linear operations and the
SAIL INT8/FP8 indexer, rather than NVIDIA's fused FP8 producers. The main MLA
cache is BF16; its dtype is independent of checkpoint/indexer quantization.
"""

from typing import ClassVar

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.deepseek_v32.attention import (
    DeepseekV32Attention as UpstreamDeepseekV32Attention,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend

import vllm_sail.ops  # noqa: F401  (register the opaque PPU indexer op)
from vllm_sail.models.deepseek_v32.indexer import DeepseekV32Indexer


class DeepseekV32Attention(UpstreamDeepseekV32Attention):
    indexer_cls = DeepseekV32Indexer
    supports_pcp_dcp: ClassVar[bool] = False
    supports_dense_mha_prefill: ClassVar[bool] = False

    def __init__(
        self,
        vllm_config,
        config,
        prefix,
        topk_indices_buffer=None,
        attn_backend=None,
        index_group_builder=None,
    ):
        if not current_platform.is_ppu():
            raise RuntimeError("The SAIL V3.2 attention requires a PPU platform")
        parallel = vllm_config.parallel_config
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise NotImplementedError("SAIL V3.2 attention requires PCP=DCP=1")
        if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "SAIL V3.2 attention currently requires a BF16 main MLA cache; "
                "FP8/INT8 checkpoint and indexer quantization remain supported"
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("SAIL V3.2 attention requires bfloat16 activations")
        if attn_backend not in (None, FlashMLASparseBackend):
            raise ValueError("SAIL V3.2 attention requires FLASHMLA_SPARSE")
        super().__init__(
            vllm_config,
            config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=FlashMLASparseBackend,
            index_group_builder=index_group_builder,
        )
        if self.hisparse_cache is not None:
            raise NotImplementedError("SAIL V3.2 attention does not support HiSparse")
        # Always publish top-k before sparse MLA. Dense-prefill scoring skip
        # and NVIDIA's packed FP8 MLA query are separate, unsupported routes.
        self._dense_mha_metadata_layer_name = ""

    def _use_sparse_mha(self, attn_metadata) -> bool:
        return False

    def forward(self, positions, hidden_states):
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        kv_c = self.kv_a_layernorm(kv_c)
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        k_pe = k_pe.reshape(-1, self.qk_rope_head_dim)
        ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)

        index_q = index_k = index_weights = None
        if self.indexer is not None and not self.skip_topk:
            indexer = self.indexer
            index_q = indexer.wq_b(q_c)[0].view(-1, indexer.n_head, indexer.head_dim)
            index_k = indexer.k_norm(indexer.wk(hidden_states)[0])
            index_weights = indexer.weights_proj(hidden_states)[0]
            rope_dim = indexer.rope_dim
            q_rope, k_rope = self.indexer_rope_emb(
                positions,
                index_q[..., :rope_dim],
                index_k[:, :rope_dim].unsqueeze(1),
            )
            index_q = torch.cat((q_rope, index_q[..., rope_dim:]), dim=-1)
            index_k = torch.cat(
                (k_rope.reshape(-1, rope_dim), index_k[:, rope_dim:]), dim=-1
            ).contiguous()
            index_q, index_weights = indexer.quantize_query(index_q, index_weights)

        context = get_forward_context()
        slot_mapping = context.slot_mapping
        assert isinstance(slot_mapping, dict)
        attn_metadata, _, _, _ = get_attention_context(self.layer_name)
        self.impl.prepare_for_batch(attn_metadata)
        if context.attn_metadata is not None:
            # The backend's existing cache writer owns its physical layout.
            # Indexer K is inserted by the PPU scorer with its own metadata.
            self.update_kv_cache(
                kv_c,
                k_pe,
                self.kv_cache,
                slot_mapping.get(self.layer_name),
                attn_metadata,
                self.kv_cache_dtype,
                self._k_scale,
            )

        output = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._ppu_indexer_and_attn(
            q_c, index_q, index_k, index_weights, ql_nope, q_pe, output
        )
        return self.o_proj(output)[0]

    @eager_break_during_capture
    def _ppu_indexer_and_attn(
        self, q_c, index_q, index_k, index_weights, ql_nope, q_pe, output
    ):
        if self.indexer is not None and not self.skip_topk:
            indexer = self.indexer
            # Call the opaque PPU op explicitly: upstream's new Attention
            # calls a module function and bypasses CustomOp.forward_ppu.
            torch.ops.vllm.ppu_sparse_attn_indexer(
                q_c,
                _encode_layer_name(indexer.k_cache.prefix),
                indexer.k_cache.kv_cache,
                index_q,
                None,
                index_k,
                index_weights,
                indexer.quant_block_size,
                indexer.scale_fmt,
                indexer.topk_tokens,
                indexer.head_dim,
                indexer.max_model_len,
                indexer.max_total_seq_len,
                self.topk_indices_buffer,
                False,
                topk_backend=indexer.indexer_op.topk_backend,
            )
        self.impl.record_logical_topk_ready()
        metadata, _, kv_cache, _ = get_attention_context(self.layer_name)
        if metadata is None or metadata.num_actual_tokens == 0:
            output.zero_()
            return
        self.impl.prepare_for_batch(metadata)
        num_actual = metadata.num_actual_tokens
        attn_out, _ = self.impl.forward_mqa(
            (ql_nope[:num_actual], q_pe[:num_actual]), kv_cache, metadata, self
        )
        # Preserve upstream MLA's W_UV absorption and unreduced o_proj output.
        values = attn_out.view(
            num_actual, self.num_local_heads, self.kv_lora_rank
        ).transpose(0, 1)
        out = (
            output[:num_actual]
            .view(num_actual, self.num_local_heads, self.v_head_dim)
            .transpose(0, 1)
        )
        torch.bmm(values, self.W_UV, out=out)
