# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream DeepSeek V4/V4.1 quantization with PPU mixed-precision extensions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v41.quant_config import (
    DeepseekV4FP8Config as UpstreamDeepseekV4FP8Config,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


class DeepseekV4FP8Config(UpstreamDeepseekV4FP8Config):
    """Keep upstream model recognition, MXFP8 loading and expert dispatch."""

    def __init__(
        self, *args, fp8_channelwise_layers: list[str] | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.fp8_channelwise_layers = fp8_channelwise_layers or []
        self._checkpoint_channelwise_layers = None
        self._checkpoint_ignored_layers = []
        # The V4.1 multimodal wrapper has no class-level packed mapping.
        # Copy rather than mutate the base class's shared default dictionary.
        self.packed_modules_mapping = {
            **getattr(self, "packed_modules_mapping", {}),
            "gate_up_proj": ["w1", "w3"],
            "fused_wqa_wkv": ["wq_a", "wkv"],
            "fused_wkv_wgate": ["wkv", "wgate"],
        }

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        method = super().override_quantization_method(
            hf_quant_cfg, user_quant, hf_config
        )
        if method is not None or not isinstance(hf_quant_cfg, dict):
            return method
        # SpeculativeConfig temporarily rewrites V4.1 to deepseek_mtp before
        # quantization validation, retaining its target architecture until
        # the later DSpark setup restores deepseek_v41.
        model_type = getattr(hf_config, "model_type", None)
        architectures = getattr(hf_config, "architectures", None) or []
        is_v4_like = model_type in (
            "deepseek_v4",
            "deepseek_v4_text",
            "deepseek_v41",
            "deepseek_v41_text",
        ) or (
            model_type == "deepseek_mtp"
            and any(
                name in architectures
                for name in ("DeepSeekV4MTPModel", "DeepseekV41ForCausalLM")
            )
        )
        if (
            current_platform.is_ppu()
            and hf_quant_cfg.get("quant_method") == "mxfp4"
            and is_v4_like
            and hf_quant_cfg.get("fp8_channelwise_layers")
        ):
            return "deepseek_v4_fp8"
        return None

    @classmethod
    def from_config(cls, config: dict) -> DeepseekV4FP8Config:
        if not config.get("fp8_channelwise_layers"):
            return super().from_config(config)
        # PPU mixed-precision exports identify the checkpoint by its MoE dtype.
        # Give the upstream FP8 parser the dense format without mutating HF config.
        dense_config = {
            **config,
            "quant_method": "fp8",
            "activation_scheme": config.get("activation_scheme", "dynamic"),
            "ignored_layers": config.get("ignored_layers")
            or config.get("ignore")
            or [],
        }
        result = super().from_config(dense_config)
        result.fp8_channelwise_layers = list(config["fp8_channelwise_layers"])
        result._checkpoint_channelwise_layers = list(result.fp8_channelwise_layers)
        result._checkpoint_ignored_layers = list(dense_config["ignored_layers"])
        return result

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        if self.fp8_channelwise_layers:
            # Map checkpoint names then strip per-layer/per-MTP-index prefixes
            # so a single short pattern (e.g. ``attn.wkv``) substring-matches
            # every absolute layer index, including MTP draft layers.
            mapped = hf_to_vllm_mapper.apply_list(self.fp8_channelwise_layers)
            layer_index_re = re.compile(r"^(?:model\.)?(?:layers|mtp)\.\d+\.")
            augmented: list[str] = []
            for entry in mapped:
                augmented.append(entry)
                stripped = layer_index_re.sub("", entry)
                if stripped and stripped != entry:
                    augmented.append(stripped)
            self.fp8_channelwise_layers = list(dict.fromkeys(augmented))

    def get_quant_method(self, layer, prefix):
        # FP8 channelwise override for dense layers (quant-config-driven)
        if (
            current_platform.is_ppu()
            and isinstance(layer, LinearBase)
            and self.fp8_channelwise_layers
        ):
            patterns = self.fp8_channelwise_layers
            ignored = self.ignored_layers
            match_mode = "substring"
            if self._checkpoint_channelwise_layers is not None:
                from vllm.config import get_current_vllm_config

                hf = get_current_vllm_config().model_config.hf_config
                if getattr(hf, "model_type", None) in (
                    "deepseek_v41",
                    "deepseek_v41_text",
                ):
                    from vllm_sail.models.deepseek_v41.config import (
                        checkpoint_layer_name,
                    )

                    prefix = checkpoint_layer_name(prefix, hf.num_hidden_layers)
                    patterns = self._checkpoint_channelwise_layers
                    ignored = self._checkpoint_ignored_layers
                    match_mode = "suffix"
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=ignored,
                fused_mapping=self.packed_modules_mapping,
                match_mode=match_mode,
            ):
                return UnquantizedLinearMethod()
            matched = is_layer_skipped(
                prefix=prefix,
                ignored_layers=patterns,
                fused_mapping=self.packed_modules_mapping,
                match_mode=match_mode,
            )
            # PPU MODIFICATION: begin
            # 0.31's cross-layer ``context_wkv_proj`` is populated from every
            # draft layer's ``attn.wkv`` shards, so it must use the same
            # channel-wise scheme. It can never match the checkpoint's
            # ``fp8_channelwise_layers`` entries (those name per-layer
            # ``...attn.wkv`` paths), and ``configure_quant_config`` skips
            # ``apply_vllm_mapper`` for models implementing ``SupportsQuant``
            # (DSpark does), so match it explicitly. Left on the default
            # per-tensor path its scale parameter is
            # ``PerTensorScaleParameter`` and the [N, 1] channel-wise shard
            # from ``_duplicate_context_wkv_weights`` trips
            # ``assert loaded_weight.shape[0] == 1``.
            if not matched and prefix.split(".")[-1] == "context_wkv_proj":
                matched = True
            # PPU MODIFICATION: end
            if matched:
                from compressed_tensors.quantization import (
                    QuantizationArgs,
                    QuantizationStrategy,
                    QuantizationType,
                )
                from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
                    CompressedTensorsLinearMethod,
                )
                from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
                    CompressedTensorsW8A8Fp8,
                )

                channelwise_args = QuantizationArgs(
                    strategy=QuantizationStrategy.CHANNEL,
                    type=QuantizationType.FLOAT,
                    num_bits=8,
                    symmetric=True,
                )
                scheme = CompressedTensorsW8A8Fp8(
                    weight_quant=channelwise_args,
                    is_static_input_scheme=False,
                )
                layer.scheme = scheme
                return CompressedTensorsLinearMethod(self)
            # Mixed exports explicitly enumerate every quantized dense layer.
            # Unlisted layers have BF16 weights and no FP8 scale to load.
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)

    def is_mxfp4_quant(self, prefix, layer):
        if not isinstance(layer, RoutedExperts) or self.expert_dtype != "fp4":
            return False
        return self.moe_quant_algo != "NVFP4"
