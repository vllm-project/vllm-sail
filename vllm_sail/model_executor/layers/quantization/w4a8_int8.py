# SPDX-License-Identifier: Apache-2.0
"""W4A8-INT8 (deepgemm-layout) quantization config for the PPU plugin.

The ``w4a8_int8`` checkpoint (e.g. Qwen3.8-Flash-Next-Quant-W-INT4-PerChannel-
A-INT8-PerToken-deepgemm) stores expert weights as signed-INT4 nibbles in a
``uint8`` container in DeepGemm's portable plain order (even-K low nibble,
odd-K high nibble), NOT the K32-group 2-byte transpose (T) order used by the
``mixed_precision_w4`` checkpoint. Dense layers and per-channel fp32 scales
are byte-identical between the two checkpoints (verified on real weights),
so only the expert container dtype and the weight-preprocess path differ.

This config reuses MixedPrecisionW4Config's dense/ignore handling and
W4AInt8MoEMethod's modular-kernel assembly; ``weight_layout`` routes the
experts class to the plain-order preprocess (no T^-1) and the uint8
container. Registered under the public name ``w4a8_int8`` so
``ModelConfig._verify_quantization`` resolves the checkpoint instead of
raising "Unknown quantization method".
"""

from vllm_sail.model_executor.layers.quantization.mixed_precision_w4 import (
    MixedPrecisionW4Config,
)


class W4A8Int8Config(MixedPrecisionW4Config):
    """DeepGemm-layout W4A8-INT8 checkpoint (portable plain-order uint8)."""

    weight_layout = "deepgemm_plain"

    @classmethod
    def get_name(cls) -> str:
        return "w4a8_int8"