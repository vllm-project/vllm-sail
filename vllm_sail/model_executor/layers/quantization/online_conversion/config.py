# SPDX-License-Identifier: Apache-2.0
"""Dependency-free source-format validation and conversion selection."""

from collections.abc import Mapping
from typing import Any

from vllm_sail import envs

CONVERSION_METHODS = ("fp8-to-int8", "fp8-to-bf16")


def validate_fp8_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("FP8 conversion requires checkpoint quantization metadata")
    if config.get("quant_method") != "fp8":
        raise ValueError("FP8 conversion requires an FP8 checkpoint, not BF16 weights")
    if config.get("weight_block_size") != [128, 128]:
        raise ValueError("FP8 conversion requires weight_block_size=[128, 128]")
    if config.get("activation_scheme", "dynamic") != "dynamic":
        raise ValueError("FP8 conversion supports only dynamic activation_scheme")
    if str(config.get("fmt", "e4m3")).lower() != "e4m3":
        raise ValueError("FP8 conversion supports only E4M3 weights")
    if config.get("scales_path"):
        raise ValueError("Cached INT8 scales are not supported by FP8 conversion")
    for key in ("ignored_layers", "modules_to_not_convert"):
        entries = config.get(key)
        if entries is not None and (
            not isinstance(entries, list)
            or any(not isinstance(entry, str) for entry in entries)
        ):
            raise ValueError(f"{key} must be a list of module names")
    if (
        config.get("ignored_layers") is not None
        and config.get("modules_to_not_convert") is not None
        and set(config["ignored_layers"]) != set(config["modules_to_not_convert"])
    ):
        raise ValueError("Conflicting checkpoint exclusion metadata")
    # Legacy mix_layers selects storage, not just execution precision. Without
    # per-layer source metadata it can silently load FP8 values as BF16.
    if config.get("mix_layers"):
        raise ValueError("mix_layers requires explicit per-layer source metadata")


def select_conversion(
    checkpoint_method: str | None, user_quant: str | None, *, is_ppu: bool
) -> str | None:
    if not is_ppu or checkpoint_method != "fp8":
        return None
    if user_quant in CONVERSION_METHODS:
        return user_quant
    if user_quant is not None:
        return None
    policy = envs.VLLM_SAIL_FP8_LOAD_POLICY
    if not envs.is_set("VLLM_SAIL_FP8_LOAD_POLICY") and envs.is_set(
        "VLLM_SAIL_ENABLE_FP8_TO_INT8"
    ):
        policy = "int8" if envs.VLLM_SAIL_ENABLE_FP8_TO_INT8 else "bf16"
    return {"int8": "fp8-to-int8", "bf16": "fp8-to-bf16"}.get(policy)
