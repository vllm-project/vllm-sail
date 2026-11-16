# SPDX-License-Identifier: Apache-2.0
"""PPU projections and query quantization for the V3.2 indexer.

Weight quantization, indexer Q/K quantization and the main MLA KV-cache dtype
are independent. SM80 scores true INT8 Q/K; SM89 scores FP8 Q/K. Both APIs
consume FP32 weights with the per-query/head Q scale already folded in.
"""

import torch
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.deepseek_v32.attention import (
    DeepseekV32Indexer as UpstreamDeepseekV32Indexer,
)
from vllm.platforms import current_platform


class DeepseekV32Indexer(UpstreamDeepseekV32Indexer):
    def __init__(
        self,
        vllm_config,
        config,
        hidden_size,
        q_lora_rank,
        quant_config,
        cache_config,
        topk_indices_buffer,
        prefix="",
    ):
        if not current_platform.is_ppu():
            raise RuntimeError("The SAIL V3.2 indexer requires a PPU platform")
        if config.index_head_dim != 128:
            raise ValueError("The SAIL V3.2 indexer requires index_head_dim=128")
        super().__init__(
            vllm_config,
            config,
            hidden_size,
            q_lora_rank,
            quant_config,
            cache_config,
            topk_indices_buffer,
            prefix=prefix,
        )
        # Match the quantized checkpoint's WK prefix and scale parameters.
        # The scoring weights remain unquantized, independently of WK.
        del self.wk_weights_proj
        self.wk = ReplicatedLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.use_int8 = not current_platform.supports_fp8()

    def quantize_query(
        self, query: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_query = query.reshape(-1, self.head_dim).contiguous()
        if flat_query.shape[0] == 0:
            dtype = torch.int8 if self.use_int8 else current_platform.fp8_dtype()
            return torch.empty_like(query, dtype=dtype), weights.float()
        if self.use_int8:
            from vllm.model_executor.layers.quantization.utils.int8_utils import (
                per_token_group_quant_int8,
            )

            values, scales = per_token_group_quant_int8(
                flat_query, self.quant_block_size
            )
        else:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                per_token_group_quant_fp8,
            )

            values, scales = per_token_group_quant_fp8(
                flat_query,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=True,
            )
        values = values.view(-1, self.n_head, self.head_dim)
        weights = (
            weights.float()
            * scales.view(-1, self.n_head)
            * self.softmax_scale
            * self.n_head**-0.5
        )
        return values, weights
