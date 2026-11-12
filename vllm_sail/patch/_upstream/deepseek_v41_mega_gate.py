# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: F821
"""Maintained V4 MoE forward body used by V4.1; not generated.

Source: vllm/models/deepseek_v4/nvidia/model.py:DeepseekV4MoE.forward at
4ba787ec9959e566d3b587a796dff07673324da5. Globals are supplied by bind_body.
Only the marked PPU gate eligibility differs; expert execution stays upstream.
"""

from __future__ import annotations


def forward(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    mega_gate_metadata: MegaGateRoutingMetadata | None = None,
) -> torch.Tensor:
    if self.gate.tid2eid is not None and input_ids is None:
        raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
    # getattr for test doubles that fake the gate module.
    bias_vl = getattr(self.gate, "bias_vl", None)
    if bias_vl is not None and input_ids is None:
        raise ValueError("DeepSeek V4 vision MoE routing requires input_ids.")

    if not self.use_native_mega_moe:
        return self._forward_fused_moe(hidden_states, input_ids)

    org_shape = hidden_states.shape
    # Small local padded batches favor GateLinear; 128-expert gates cross earlier.
    gate_threshold = 1 if self.gate.weight.shape[0] == 128 else 16
    # PPU MODIFICATION: begin
    from vllm_sail.models.deepseek_v41.mega_gate import should_use_mega_gate

    if hidden_states.shape[0] <= gate_threshold or not should_use_mega_gate(
        self, hidden_states
    ):
        # PPU MODIFICATION: end
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
            bias_vl=bias_vl.data if bias_vl is not None else None,
            image_sentinel_lo=self.image_sentinel_lo if bias_vl is not None else 0,
        )
    else:
        if mega_gate_metadata is None:
            raise RuntimeError(
                "MegaMoE routing metadata must be prepared once by the model."
            )

        unmapped_topk_idx = None
        fix_routing_mask = None
        if self.gate.tid2eid is not None:
            if mega_gate_metadata.safe_hash_input_ids is None:
                raise RuntimeError("Hash routing requires prepared input IDs.")
            if mega_gate_metadata.hash_token_mask is None:
                raise RuntimeError("Hash routing requires a prepared routing mask.")
            unmapped_topk_idx = self.gate.tid2eid[
                mega_gate_metadata.safe_hash_input_ids
            ]
            fix_routing_mask = mega_gate_metadata.hash_token_mask

        topk_weights, topk_ids = bf16_mega_gate(
            hidden_states,
            self.gate.weight,
            self.n_activated_experts,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            ep_rank=self.ep_rank,
            bias=self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None,
            image_bias=bias_vl.data if bias_vl is not None else None,
            image_token_mask=mega_gate_metadata.image_token_mask,
            fix_routing_mask=fix_routing_mask,
            unmapped_topk_idx=unmapped_topk_idx,
        )
    final_hidden_states = self.experts(
        hidden_states, topk_weights, topk_ids, self.shared_experts
    )
    return final_hidden_states.view(org_shape)
