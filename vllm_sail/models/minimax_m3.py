# SPDX-License-Identifier: Apache-2.0
"""Registered MiniMax-M3 subclasses with quantization-aware module names."""

from __future__ import annotations

from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.minimax_m3.nvidia.model import (
    MiniMaxM3SparseForCausalLM as _UpstreamMiniMaxM3SparseForCausalLM,
)
from vllm.models.minimax_m3.nvidia.model import (
    MiniMaxM3SparseForConditionalGeneration as _UpstreamMiniMaxM3SparseForConditionalGeneration,
)


class MiniMaxM3SparseForCausalLM(_UpstreamMiniMaxM3SparseForCausalLM, SupportsQuant):
    """Upstream MiniMax-M3 sparse backbone + the fork's SupportsQuant."""


class MiniMaxM3SparseForConditionalGeneration(
    _UpstreamMiniMaxM3SparseForConditionalGeneration, SupportsQuant
):
    """Also map bare vision MLP module names in quantization configs."""

    hf_to_vllm_mapper = (
        _UpstreamMiniMaxM3SparseForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(orig_to_new_suffix={".mlp.fc1": ".fc1", ".mlp.fc2": ".fc2"})
    )
