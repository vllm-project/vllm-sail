# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify MiniMax M3 packed_modules_mapping is injected into the quant config
so PPU FP8 channel-wise dense routing sees fused layers correctly.

Regression tests for the silent-fallback bug where
``MiniMaxM3SparseForCausalLM`` declared neither ``packed_modules_mapping``
nor ``SupportsQuant``, leaving ``Mxfp4Config.packed_modules_mapping`` empty:
``is_layer_skipped`` could not expand ``qkv_proj``/``gate_up_proj`` back to
the shard names listed in ``fp8_channelwise_layers``, and fused dense layers
silently fell back to ``UnquantizedLinearMethod``.

Checks the mapping injection through the real ``SupportsQuant.__new__``
path (no GPU, no weights).
"""

import pytest

# This test exercises real vLLM/torch code paths, so it needs both installed.
# tests/ut must remain runnable on a bare CPU runner with neither present
# (see tests/conftest.py), hence the module-level skip rather than an import
# error at collection time.
pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.models.interfaces import SupportsQuant

pytestmark = pytest.mark.requires_runtime

FP8_CHANNELWISE_LAYERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

EXPECTED_FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _model_classes():
    """The classes vLLM loads for MiniMax-M3 with the plugin installed.

    The plugin registers SupportsQuant-augmented subclasses through
    ModelRegistry; raw upstream classes stay unpatched by design, so the
    contract under test is the registered subclass pair.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    from vllm_sail.models import register_model
    from vllm_sail.models.minimax_m3 import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )

    register_model()
    for arch, _cls in (
        ("MiniMaxM3SparseForCausalLM", MiniMaxM3SparseForCausalLM),
        (
            "MiniMaxM3SparseForConditionalGeneration",
            MiniMaxM3SparseForConditionalGeneration,
        ),
    ):
        assert arch in ModelRegistry.models, (
            f"{arch} not registered; plugin model wiring missing"
        )
    return MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration


def _make_quant_config() -> Mxfp4Config:
    return Mxfp4Config(fp8_channelwise_layers=list(FP8_CHANNELWISE_LAYERS))


def test_vision_mlp_quant_exclusions_match_loaded_weights():
    from torch.nn import Linear

    _, model = _model_classes()
    encoder = "vision_tower.vision_model.encoder.layers.0"
    checkpoint_names = [f"{encoder}.mlp.fc1", f"{encoder}.mlp.fc2"]
    quant = CompressedTensorsConfig(
        target_scheme_map={"Linear": {"format": "int-quantized"}},
        ignore=[*checkpoint_names, "multi_modal_projector.linear_1"],
        quant_format="int-quantized",
    )
    model.__new__(model, quant)

    linear = Linear(2, 2)
    mapper = model.hf_to_vllm_mapper
    for name in checkpoint_names:
        runtime_name = mapper.map_name(name + ".weight").removesuffix(".weight")
        assert runtime_name == f"{encoder}.{name.rsplit('.', 1)[1]}"
        assert quant.get_scheme_dict(linear, runtime_name) is None

    assert (
        quant.get_scheme_dict(linear, "vision_tower.multi_modal_projector.linear_1")
        is None
    )
    assert mapper.map_name(f"{encoder}.mlp.fc10") == f"{encoder}.mlp.fc10"
    expert = "language_model.model.layers.3.block_sparse_moe.experts.0.w1"
    assert quant.get_scheme_dict(linear, expert) is not None


class TestPackedModulesMappingInjection:
    """The fix contract: model classes expose the fused mapping and inject it
    into the quant config at instance creation, before any layer is built."""

    def test_model_classes_declare_mapping_and_supports_quant(self):
        for cls in _model_classes():
            assert issubclass(cls, SupportsQuant), (
                f"{cls.__name__} must inherit SupportsQuant so the mapping is "
                "injected in __new__ regardless of loader path"
            )
            assert cls.packed_modules_mapping == EXPECTED_FUSED_MAPPING

    def test_new_injects_mapping_into_quant_config(self):
        """Exercise the real SupportsQuant.__new__ injection path without
        building the full model (no __init__, no weights, no distributed)."""
        causal_lm, _ = _model_classes()
        quant_config = _make_quant_config()
        assert quant_config.packed_modules_mapping == {}

        # SupportsQuant._find_quant_config picks up a QuantizationConfig
        # positional arg, same as it picks vllm_config.quant_config at real
        # model bring-up.
        instance = causal_lm.__new__(causal_lm, quant_config)

        assert instance.quant_config is quant_config
        assert quant_config.packed_modules_mapping == EXPECTED_FUSED_MAPPING
