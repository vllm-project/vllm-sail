# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: F821

from __future__ import annotations


class Step5DecoderLayer:
    def _forward_ffn(
        self,
        hidden_states: torch.Tensor,
        padding: torch.Tensor | None = None,
    # PPU MODIFICATION: begin
        *,
        _defer_shared: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    # PPU MODIFICATION: end
        if self.use_moe:
            shared_output, moe_output = self.moe(hidden_states, padding=padding)
            # PPU MODIFICATION: begin
            if (
                _defer_shared
                and not self.fp32_residual_connection
                and not self.moe.fuse_all_reduce
                and (self.moe.tp_size == 1 or self.use_sequence_parallel)
                and shared_output is not None
                and shared_output.dtype == moe_output.dtype == torch.bfloat16
                and shared_output.shape == moe_output.shape
                and moe_output.ndim == 2
                and moe_output.shape[1] == 4096
                and moe_output.is_cuda
                and shared_output.device == moe_output.device
                and shared_output.is_contiguous()
                and moe_output.is_contiguous()
                and not (
                    torch.is_grad_enabled()
                    and (shared_output.requires_grad or moe_output.requires_grad)
                )
            ):
                is_ppu = getattr(current_platform, "is_ppu", None)
                if callable(is_ppu) and is_ppu():
                    return moe_output, shared_output
            # PPU MODIFICATION: end
            combined = (
                moe_output
                if shared_output is None
                else self._add_for_residual(moe_output, shared_output)
            )
            # Reduce once when both branches defer reduction; otherwise their
            # outputs are already reduced.
            if self.moe.fuse_all_reduce:
                if self.use_fused_all_reduce:
                    combined = self.tp_group.all_reduce(combined)
                else:
                    combined = tensor_model_parallel_all_reduce(combined)
        # PPU MODIFICATION: begin
            return (combined, None) if _defer_shared else combined
        output = self.mlp(hidden_states)
        return (output, None) if _defer_shared else output
        # PPU MODIFICATION: end

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        padding: torch.Tensor | None = None,
    # PPU MODIFICATION: begin
        *,
        _shared_input: torch.Tensor | None = None,
        _defer_shared: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    ):
        if _shared_input is not None:
            hidden_states, residual = self.input_layernorm._forward_with_shared(
                hidden_states, _shared_input, residual
            )
        elif (
            self.fp32_residual_connection
            or (
                hasattr(current_platform, "is_ppu")
                and current_platform.is_ppu()
                and hidden_states.dtype == torch.bfloat16
                and (residual is None or residual.dtype == torch.bfloat16)
            )
        ) and isinstance(self.input_layernorm, Step5RMSNorm):
    # PPU MODIFICATION: end
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(
                    hidden_states, out_dtype=torch.bfloat16
                )
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual, out_dtype=torch.bfloat16
                )
        else:
            if residual is not None:
                hidden_states = self._add_for_residual(hidden_states, residual)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self._cast_for_param_op(hidden_states)

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.use_sequence_parallel:
            hidden_states = sp_reduce_scatter(hidden_states)
        # PPU MODIFICATION: begin
        if (
            self.fp32_residual_connection
            or (
                hasattr(current_platform, "is_ppu")
                and current_platform.is_ppu()
                and hidden_states.dtype == torch.bfloat16
                and residual.dtype == torch.bfloat16
            )
        ) and isinstance(self.post_attention_layernorm, Step5RMSNorm):
        # PPU MODIFICATION: end
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual, out_dtype=torch.bfloat16
            )
        else:
            hidden_states = self._cast_for_residual(hidden_states)
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self._cast_for_param_op(hidden_states)

        # PPU MODIFICATION: begin
        if _defer_shared:
            if residual.dtype != torch.bfloat16:
                return self._forward_ffn(hidden_states, padding=padding), residual, None
            ffn_output, shared_output = self._forward_ffn(
                hidden_states, padding=padding, _defer_shared=True
            )
            return ffn_output, residual, shared_output
        # PPU MODIFICATION: end
        ffn_output = self._forward_ffn(hidden_states, padding=padding)
        return ffn_output, residual


class Step5Model:
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = self._cast_for_residual(hidden_states)
        padding = None
        if self.use_sequence_parallel:
            padding = _step5_sp_padding(hidden_states)
            hidden_states = sp_shard(hidden_states)
        residual = None
        # PPU MODIFICATION: begin
        pending_shared = None
        is_ppu = getattr(current_platform, "is_ppu", None)
        allow_deferred = (
            callable(is_ppu) and is_ppu() and not self.fp32_residual_connection
        )
        # PPU MODIFICATION: end
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # PPU MODIFICATION: begin
            defer_shared = (
                allow_deferred
                and i + 1 < self.end_layer
                and layer.use_moe
                and not layer.moe.fuse_all_reduce
                and type(self.layers[i + 1].input_layernorm) is Step5RMSNorm
            # PPU MODIFICATION: end
            )
            # PPU MODIFICATION: begin
            kwargs = {}
            if pending_shared is not None:
                kwargs["_shared_input"] = pending_shared
            if defer_shared:
                hidden_states, residual, pending_shared = layer(
                    positions, hidden_states, residual, padding=padding,
                    _defer_shared=True, **kwargs,
                )
            else:
                hidden_states, residual = layer(
                    positions, hidden_states, residual, padding=padding, **kwargs
                )
                pending_shared = None
            # PPU MODIFICATION: end
        if residual is not None:
            hidden_states = layer._add_for_residual(hidden_states, residual)
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        return hidden_states
