# SPDX-License-Identifier: Apache-2.0
"""PPU eligibility for V4.1's native MegaMoE gate branch.

The decode/verify, sequence-parallel and row-count exclusions follow
sglang-for-sail 01a98013bb83. Expert execution remains the upstream native
MegaMoE path, which needs a separate PPU integration before it can be enabled.
"""

import math

import torch
import vllm.envs as vllm_envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.utils.deep_gemm import has_mega_gate, is_deep_gemm_supported

logger = init_logger(__name__)
_MAX_ROWS = 2048


def _fallback_reason(moe, hidden_states):
    if not envs.VLLM_SAIL_DEEPGEMM_MEGA_GATE:
        return "VLLM_SAIL_DEEPGEMM_MEGA_GATE is disabled"
    if not current_platform.is_ppu() or not current_platform.has_device_capability(89):
        return "requires PPU SM89+"
    if not is_deep_gemm_supported() or not has_mega_gate():
        return "SAIL DeepGEMM bf16_mega_gate is unavailable or disabled"
    if not is_forward_context_available():
        return "forward metadata is unavailable"
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return "decode metadata is unavailable"
    batches = [m for m in metadata.values() if hasattr(m, "num_prefill_tokens")]
    if (
        not batches
        or any(m.num_prefill_tokens > 0 for m in batches)
        or not any(getattr(m, "num_decode_tokens", 0) > 0 for m in batches)
    ):
        return "only decode and target verification use MegaGate"
    if moe.use_sequence_parallel or vllm_envs.VLLM_BATCH_INVARIANT:
        return "sequence parallelism or batch-invariant execution is active"
    if (
        moe.scoring_func != "sqrtsoftplus"
        or not moe.renormalize
        or not math.isfinite(moe.routed_scaling_factor)
    ):
        return "requires normalized sqrtsoftplus routing"
    weight = moe.gate.weight
    if (
        hidden_states.ndim != 2
        or not 0 < hidden_states.shape[0] <= _MAX_ROWS
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or hidden_states.shape[1] != weight.shape[1]
        or weight.shape[1] == 0
        or weight.shape[1] % 256
        or not 0 < weight.shape[0] <= 512
        or weight.shape[0] % 4
        or not 0 < moe.n_activated_experts <= min(32, weight.shape[0])
        or weight.device != hidden_states.device
        or weight.data_ptr() % 16
        or hidden_states.data_ptr() % 16
    ):
        return "BF16 gate shape, alignment or row limit is unsupported"
    for bias in (moe.gate.e_score_correction_bias, getattr(moe.gate, "bias_vl", None)):
        if bias is not None and (
            bias.dtype != torch.float32
            or bias.shape != (weight.shape[0],)
            or not bias.is_contiguous()
            or bias.device != hidden_states.device
        ):
            return "gate bias must be contiguous FP32 on the gate device"
    return None


def should_use_mega_gate(moe, hidden_states):
    reason = _fallback_reason(moe, hidden_states)
    if reason is not None:
        if envs.VLLM_SAIL_DEEPGEMM_MEGA_GATE and hidden_states.shape[0] > 0:
            logger.info_once("MegaGate fallback: %s; using the regular gate.", reason)
        return False
    return True
