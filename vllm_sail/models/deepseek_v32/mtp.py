# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""New V3.2 MTP using the same PPU attention and WK policy as the target.

The loader body is adapted from vLLM 4ba787ec9959
``vllm/models/deepseek_v32/nvidia/mtp.py``. PPU changes are marked. The new
predictor forward, top-k reuse/compaction and post-norm recycling are inherited.
"""

import typing
from collections.abc import Callable, Iterable

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import (
    get_spec_layer_idx_from_weight_name,
)
from vllm.models.deepseek_v32.nvidia import mtp as _upstream

from vllm_sail.models.deepseek_v32.model import install_attention
from vllm_sail.models.deepseek_v32.ops import install_allreduce_norm


class DeepseekV32MTP(_upstream.DeepseekV32MTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        install_attention()
        install_allreduce_norm(_upstream)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            # PPU MODIFICATION: begin
            # WK retains its checkpoint quantization; weights_proj is BF16.
            # Both use their own parameter loaders instead of fused shards.
            # PPU MODIFICATION: end
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # PPU MODIFICATION: begin
        # Separate quantized WK loads weights/scales directly. The upstream
        # FP8-to-BF16 fused-WK pending buffer is not used on this path.
        # PPU MODIFICATION: end
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or self.is_unused_checkpoint_weight(name):
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                # A tied top-level embed_tokens has no spec layer to rewrite
                # from; the draft needs its own copy under PP.
                param = params_dict.get(name) if "embed_tokens" in name else None
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)

            # PPU MODIFICATION: begin
            # Preserve WK's original dtype and scale parameter names.
            # PPU MODIFICATION: end

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                num_chunks = 1

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    is_expert_weight = False
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping  # type: ignore[assignment]
                        if weight_name not in chunk_name:
                            continue
                        is_expert_weight = True
                        name_mapped = chunk_name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    else:
                        if is_expert_weight:
                            continue
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        name = maybe_remap_kv_scale_name(name, params_dict)  # type: ignore[assignment]
                        if name is None:
                            continue
                        if (
                            spec_layer != self.model.mtp_start_layer_idx
                            and ".layers" not in name
                        ):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            loaded_params.add(name)

        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers and is_mtp_completeness_check_enabled():
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint."
                )
        return loaded_params
