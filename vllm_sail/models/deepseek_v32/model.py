# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""New V3.2 model with PPU attention and quantized WK loading.

The loader body is adapted from vLLM 4ba787ec9959
``vllm/models/deepseek_v32/nvidia/model.py``. PPU changes are marked.
"""

import typing
from collections.abc import Callable, Iterable

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import (
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.models.deepseek_v32.nvidia import model as _upstream
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v32.attention import DeepseekV32Attention
from vllm_sail.models.deepseek_v32.ops import install_allreduce_norm
from vllm_sail.patch.utils import patch


def install_attention() -> None:
    """Connect both new main and MTP decoders at their common provider.

    MTP imports the decoder class by value, but its constructor resolves
    Attention in this provider's globals. No global producer/scorer is patched.
    Installation happens explicitly during model construction, not package import.
    """
    if not current_platform.is_ppu():
        raise RuntimeError("The SAIL V3.2 model requires a PPU platform")
    install_allreduce_norm(_upstream)
    if _upstream.DeepseekV32Attention is DeepseekV32Attention:
        return
    patch(
        "vllm.models.deepseek_v32.nvidia.model",
        "DeepseekV32Attention",
        reason=(
            "New V3.2 main/MTP decoders hardcode the Attention provider; PPU "
            "needs its INT8/FP8 producers and SAIL sparse indexer while "
            "retaining the new decoder forward and collective semantics."
        ),
        affected_versions=">=0.33.0,<0.34.0",
        remove_when=(
            "DeepseekV32DecoderLayer exposes an attention_cls factory used "
            "by both main and MTP construction."
        ),
    )(DeepseekV32Attention)


class DeepseekV32Model(_upstream.DeepseekV32Model):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # PPU MODIFICATION: begin
        # DSA-only MLA, separate quantized WK / BF16 weights_proj and experts.
        # PPU MODIFICATION: end
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
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # PPU MODIFICATION: begin
        # Separate quantized WK loads weights/scales directly. The upstream
        # FP8-to-BF16 fused-WK pending buffer is not used on this path.
        # PPU MODIFICATION: end
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # MTP / nextn layers are loaded by the MTP model, not here.
            if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
                continue
            # PPU MODIFICATION: begin
            # Preserve WK's original dtype and scale parameter names.
            # PPU MODIFICATION: end

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Experts are handled below; skip here before the name rewrite.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping  # type: ignore[assignment]
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
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
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DeepseekV32ForCausalLM(_upstream.DeepseekV32ForCausalLM):
    model_cls = DeepseekV32Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        install_attention()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
