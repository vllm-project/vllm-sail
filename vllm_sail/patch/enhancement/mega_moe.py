# SPDX-License-Identifier: Apache-2.0
"""Redirect the upstream MegaMoE experts on PPU."""

from __future__ import annotations

import sys

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"
_REASON = "SAIL MegaMoE needs MXFP4 dispatch and PPU-specific scale layouts."
_REMOVE_WHEN = "vLLM supports selecting an out-of-tree MegaMoE expert backend."
_EXPERT_CONSUMERS = (
    "vllm.models.deepseek_v4.nvidia.model",
    "vllm.models.kimi_k3.nvidia.model",
    "vllm.models.minimax_m3.nvidia.model",
)
_GATE_CONSUMERS = ("vllm.models.deepseek_v4.nvidia.model",)
_installed = False


def install() -> None:
    """Install the expert redirect once on PPU."""
    global _installed
    from vllm.platforms import current_platform

    if _installed or not current_platform.is_ppu():
        return

    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
        PPUDeepGemmMegaMoEExperts,
    )
    from vllm_sail.model_executor.layers.fused_moe.mega_gate import bf16_mega_gate
    from vllm.utils import deep_gemm

    original = deep_gemm_mega_moe.DeepGemmMegaMoEExperts
    patch(
        deep_gemm_mega_moe.__name__,
        "DeepGemmMegaMoEExperts",
        reason=_REASON,
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )(PPUDeepGemmMegaMoEExperts)

    # Consumers loaded after the provider is patched import the new symbol.
    for module_name in _EXPERT_CONSUMERS:
        consumer = sys.modules.get(module_name)
        if (
            consumer is not None
            and getattr(consumer, "DeepGemmMegaMoEExperts", None) is original
        ):
            patch(
                module_name,
                "DeepGemmMegaMoEExperts",
                reason=_REASON,
                affected_versions=_AFFECTED,
                remove_when=_REMOVE_WHEN,
            )(PPUDeepGemmMegaMoEExperts)
    original_gate = deep_gemm.bf16_mega_gate
    gate_owners = [deep_gemm]
    for module_name in _GATE_CONSUMERS:
        consumer = sys.modules.get(module_name)
        if (
            consumer is not None
            and getattr(consumer, "bf16_mega_gate", None) is original_gate
        ):
            gate_owners.append(consumer)
    for owner in gate_owners:
        patch(
            owner.__name__,
            "bf16_mega_gate",
            reason="MegaGate must exclude padding before validating routing scores.",
            affected_versions=_AFFECTED,
            remove_when="vLLM bf16_mega_gate forwards token validity to DeepGEMM.",
        )(bf16_mega_gate)
    _installed = True
