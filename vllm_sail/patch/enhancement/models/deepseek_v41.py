# SPDX-License-Identifier: Apache-2.0
"""Select the V4.1 PPU FlashMLA subclass at the CUDA selector boundary."""

from vllm.models.deepseek_v41.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention as UpstreamFlashMLAAttention,
)
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v41.flashmla import DeepseekV41FlashMLAAttention
from vllm_sail.patch.utils import PATCH_MARKER, patch

_MODULE = "vllm.models.deepseek_v41.nvidia.model"
_TARGET = f"{_MODULE}._select_dsv4_attn_cls"


@patch(
    _MODULE,
    "_select_dsv4_attn_cls",
    reason=(
        "PPU requires unpadded FlashMLA heads and an MXFP8 BF16 output projection "
        "while retaining V4.1 shared KV/index metadata and CUDA model logic."
    ),
    affected_versions=">=0.30.0,<0.31.0",
    remove_when="vLLM exposes a V4.1 attention-class registry supporting PPU overrides.",
)
def _select_dsv4_attn_cls(vllm_config):
    original = getattr(_select_dsv4_attn_cls, PATCH_MARKER)[_TARGET]
    selected = original(vllm_config)
    if current_platform.is_ppu() and selected is UpstreamFlashMLAAttention:
        return DeepseekV41FlashMLAAttention
    return selected
