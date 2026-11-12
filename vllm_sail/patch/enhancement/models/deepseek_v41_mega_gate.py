# SPDX-License-Identifier: Apache-2.0
"""Adapt only the native V4.1 MegaMoE gate path for SAIL."""

from __future__ import annotations

import torch
from vllm.models.deepseek_v4.nvidia import model as _model
from vllm.platforms import current_platform

from vllm_sail.patch._upstream import deepseek_v41_mega_gate as _body
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.33.0,<0.34.0"
_API_MODULE = "vllm.utils.deep_gemm"
_API_TARGET = f"{_API_MODULE}.bf16_mega_gate"
_API_REASON = "Native MegaMoE gate calls on PPU must use the external SAIL DeepGEMM wheel."
_API_REMOVE_WHEN = "vLLM's bf16_mega_gate wrapper supports the PPU DeepGEMM provider."


@patch(
    _API_MODULE,
    "bf16_mega_gate",
    reason=_API_REASON,
    affected_versions=_AFFECTED,
    remove_when=_API_REMOVE_WHEN,
)
def bf16_mega_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_topk: int,
    *,
    scoring_func: str,
    routed_scaling_factor: float,
    ep_rank: int,
    bias: torch.Tensor | None = None,
    image_bias: torch.Tensor | None = None,
    image_token_mask: torch.Tensor | None = None,
    fix_routing_mask: torch.Tensor | None = None,
    unmapped_topk_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    implementation = getattr(bf16_mega_gate, PATCH_MARKER)[_API_TARGET]
    if current_platform.is_ppu():
        from vllm_sail.utils.deep_gemm import bf16_mega_gate as implementation

    return implementation(
        x,
        weight,
        num_topk,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        ep_rank=ep_rank,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        fix_routing_mask=fix_routing_mask,
        unmapped_topk_idx=unmapped_topk_idx,
    )


# The native V4 provider is also the inherited V4.1 forward's global namespace.
if _model.bf16_mega_gate is getattr(bf16_mega_gate, PATCH_MARKER)[_API_TARGET]:
    patch(
        "vllm.models.deepseek_v4.nvidia.model",
        "bf16_mega_gate",
        reason=_API_REASON,
        affected_versions=_AFFECTED,
        remove_when=_API_REMOVE_WHEN,
    )(bf16_mega_gate)


_native_forward = bind_body(_body.forward, _model)
_FORWARD_MODULE = "vllm.models.deepseek_v41.nvidia.model"
_FORWARD_TARGET = f"{_FORWARD_MODULE}.DeepseekV4MoE.forward"


@patch(
    _FORWARD_MODULE,
    "DeepseekV4MoE.forward",
    reason=(
        "The native V4.1 MegaMoE gate needs PPU SM89+ decode/verify guards and "
        "fallback logging without replacing expert execution or the MoE runner."
    ),
    affected_versions=_AFFECTED,
    remove_when="The V4.1 native MoE forward exposes a platform-selectable MegaGate eligibility hook.",
)
def forward(self, hidden_states, input_ids=None, mega_gate_metadata=None):
    if current_platform.is_ppu() and self.use_native_mega_moe:
        return _native_forward(self, hidden_states, input_ids, mega_gate_metadata)
    original = getattr(forward, PATCH_MARKER)[_FORWARD_TARGET]
    return original(self, hidden_states, input_ids, mega_gate_metadata)
