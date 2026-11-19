# SPDX-License-Identifier: Apache-2.0
"""Check pre-construction quantization mapping without importing torch or vLLM."""

import re
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.source import function


@pytest.mark.upstream_source
@pytest.mark.parametrize("bf16_wkv", [False, True])
def test_checkpoint_quantization_paths_are_mapped_before_construction(
    modules, upstream_source_root, bf16_wkv
):
    # Execute the real upstream mapper and configuration functions. No model or
    # tensor stubs implement the name transformation being checked here.
    source = upstream_source_root / "vllm"
    namespace = dict(__name__=__name__, re=re, field=field, replace=replace)
    namespace["WeightsMapper"] = dataclass(
        function(source / "model_executor/models/utils.py", "WeightsMapper", namespace)
    )
    function(
        source / "models/deepseek_v41/nvidia/model.py",
        "_make_deepseek_v4_weights_mapper",
        namespace,
    )
    factory = function(
        source / "models/deepseek_v41/nvidia/vl_model.py",
        "_make_deepseek_v4_vl_weights_mapper",
        namespace,
    )
    base = type("UpstreamVL", (), {})  # Upstream VL has no class-level mapper.
    modules(
        "vllm.models.deepseek_v41.nvidia.vl_model",
        DeepseekV41ForCausalLM=base,
        _make_deepseek_v4_vl_weights_mapper=factory,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace())
    modules(
        "vllm_sail.models.deepseek_v41.config",
        set_int8_w8a8=lambda value: None,
        uses_int8_w8a8=lambda config: True,
    )
    model_class = load_patch(
        "vllm_sail/models/deepseek_v41/model.py"
    ).DeepseekV41ForCausalLM
    mapper = model_class.hf_to_vllm_mapper
    apply_mapper = function(
        source
        / "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py",
        "CompressedTensorsConfig.apply_vllm_mapper",
        namespace,
    )
    quant_class = type("QuantConfig", (), {"apply_vllm_mapper": apply_mapper})
    quant = quant_class()
    quant.kv_cache_scheme = None
    wkv_names = [f"layers.{layer}.engram.wkv" for layer in (1, 14)]
    checkpoint_ignore = ["layers.0.attn.wq", "mtp.0.head"]
    if bf16_wkv:
        checkpoint_ignore += wkv_names
    quant.ignore = list(checkpoint_ignore)
    # Explicit paths, class targets, and regex targets follow distinct upstream
    # rules. MTP ignores must survive the weight mapper's drop rules too.
    quant.target_scheme_map = {
        "Linear": "w8a8",
        "layers.0.attn.wkv": "w8a8",
        "re:.*ffn.*": "w8a8",
    }
    configure = function(
        source / "model_executor/model_loader/utils.py",
        "configure_quant_config",
        {"SupportsQuant": type("SupportsQuant", (), {})},
    )
    # This is the engine's hook before the first model instance is constructed.
    configure(quant, model_class)
    for name in wkv_names:
        runtime_name = f"language_model.model.{name}"
        assert (runtime_name in quant.ignore) is bf16_wkv
    assert quant.ignore == [f"language_model.model.{n}" for n in checkpoint_ignore]
    assert quant.target_scheme_map == {
        "Linear": "w8a8",
        "language_model.model.layers.0.attn.wkv": "w8a8",
        "re:.*ffn.*": "w8a8",
    }
    assert checkpoint_ignore[0] == "layers.0.attn.wq"
    assert model_class.hf_to_vllm_mapper is mapper
    assert not hasattr(base, "hf_to_vllm_mapper")
