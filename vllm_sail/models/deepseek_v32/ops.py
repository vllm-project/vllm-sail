# SPDX-License-Identifier: Apache-2.0
"""PPU collective fallback for the new V3.2 target and draft decoders."""

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch


def allreduce_rms_norm(hidden_states, residual, norm):
    # o_proj and MoE deliberately leave partial outputs unreduced. Preserve
    # that ordering without entering FlashInfer's NVIDIA workspace probing.
    return norm(tensor_model_parallel_all_reduce(hidden_states), residual)


def install_allreduce_norm(module) -> None:
    if not current_platform.is_ppu():
        raise RuntimeError("The SAIL V3.2 collective requires a PPU platform")
    if module.fused_allreduce_rms_norm is allreduce_rms_norm:
        return
    patch(
        module.__name__,
        "fused_allreduce_rms_norm",
        reason=(
            "V3.2 decoders keep partial attention/MoE outputs for the next "
            "allreduce+RMSNorm. PPU uses the explicit collective fallback "
            "instead of probing NVIDIA FlashInfer fusion workspaces."
        ),
        affected_versions=">=0.33.0,<0.34.0",
        remove_when=(
            "The common fused_allreduce_rms_norm dispatch excludes PPU from "
            "its NVIDIA FlashInfer fast path."
        ),
    )(allreduce_rms_norm)
