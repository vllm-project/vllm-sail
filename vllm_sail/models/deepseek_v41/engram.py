# SPDX-License-Identifier: Apache-2.0
"""Channelwise Engram storage with upstream sharding, UVA and prefetch."""

import torch
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.nvidia.engram import ParallelEngramEmbedding


class ChannelwiseEngramEmbedding(ParallelEngramEmbedding):
    """FP8 embedding values and one FP32 scale per row, with no requantization."""

    def __init__(
        self,
        num_embeddings,
        dim,
        head_sizes,
        *,
        cpu_offload=False,
        dp_shared_memory=False,
    ):
        if dp_shared_memory:
            raise ValueError(
                "Channelwise Engram does not support dp_shared_memory; "
                "use cpu_offload without shared DP storage."
            )
        super().__init__(
            num_embeddings,
            dim,
            head_sizes,
            block_size=dim,
            cpu_offload=cpu_offload,
            dp_shared_memory=False,
        )
        # Keep the upstream parameter name and head-shard loader. The PPU
        # lookup distinguishes FP32 scales from exponent bytes by tensor dtype.
        set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 1.0})

    def _allocate_weights(self):
        placement = {"device": "cpu", "pin_memory": True} if self.cpu_offload else {}
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                **placement,
            ),
            torch.empty(
                self.part_num_embeddings,
                1,
                dtype=torch.float32,
                **placement,
            ),
        )
