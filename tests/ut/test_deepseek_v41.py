# SPDX-License-Identifier: Apache-2.0
"""V4.1 integration contracts; CPU tests never import torch or vLLM."""

from types import SimpleNamespace

import pytest

from tests.support.source import function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
load_patch = harness.load_patch


@pytest.fixture
def quant_module(modules):
    calls = []

    class Upstream:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def override_quantization_method(cls, *args):
            calls.append(("recognize", args))
            return "upstream-method"

        @classmethod
        def from_config(cls, config):
            calls.append(("config", config))
            return cls()

        def get_quant_method(self, layer, prefix):
            calls.append(("dispatch", layer, prefix))
            return "upstream-kernel"

    modules("vllm.models.deepseek_v41.quant_config", DeepseekV4FP8Config=Upstream)
    modules("vllm.model_executor.layers.fused_moe", RoutedExperts=type("MoE", (), {}))
    modules("vllm.model_executor.layers.linear", LinearBase=type("Linear", (), {}))
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        is_layer_skipped=lambda **kwargs: False,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    return load_patch("vllm_sail/models/deepseek_v4/quant_config.py"), calls


def test_standard_quantization_delegates_without_rewriting_config(quant_module):
    module, calls = quant_module
    cls = module.DeepseekV4FP8Config
    checkpoint = {"quant_method": "fp8", "weight_block_size": [32, 32]}
    hf = SimpleNamespace(model_type="deepseek_v41")
    assert cls.override_quantization_method(checkpoint, None, hf) == "upstream-method"
    config = cls.from_config(checkpoint)
    layer = object()
    assert config.get_quant_method(layer, "model.ffn") == "upstream-kernel"
    assert calls == [
        ("recognize", (checkpoint, None, hf)),
        ("config", checkpoint),
        ("dispatch", layer, "model.ffn"),
    ]
    assert calls[1][1] is checkpoint


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "model_type",
    ["deepseek_v4", "deepseek_v4_text", "deepseek_v41", "deepseek_v41_text"],
)
def test_upstream_recognizes_target_and_text_configs(upstream_source_root, model_type):
    recognize = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.override_quantization_method",
        {},
    )
    assert (
        recognize(
            object,
            {"quant_method": "fp8"},
            None,
            SimpleNamespace(model_type=model_type),
        )
        == "deepseek_v4_fp8"
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
def test_upstream_mxfp4_dispatch_preserves_target_and_draft_moe(
    upstream_source_root, modules, experts, topk
):
    class RoutedExperts:
        moe_config = SimpleNamespace(num_experts=experts, experts_per_token=topk)

    expected = object()
    modules(
        "vllm.model_executor.layers.quantization.mxfp4",
        Mxfp4MoEMethod=lambda cfg: expected,
    )
    dispatch = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.get_quant_method",
        {
            "LinearBase": type("Linear", (), {}),
            "RoutedExperts": RoutedExperts,
            "is_layer_skipped": lambda **kwargs: False,
            "Mxfp4MoEMethod": lambda cfg: (expected, cfg),
        },
    )
    config = SimpleNamespace(
        ignored_layers=[],
        packed_modules_mapping={},
        expert_dtype="fp4",
        moe_quant_algo="",
    )
    layer = RoutedExperts()
    assert dispatch(config, layer, "model.ffn.experts") == (expected, layer.moe_config)


@pytest.mark.parametrize("is_ppu", [True, False])
def test_dense_emulation_reuses_upstream_only_on_ppu(modules, is_ppu):
    class UpstreamEmulation:
        process_weights_after_loading = object()
        apply_weights = object()

    modules(
        "vllm.model_executor.kernels.linear.mxfp8.emulation",
        EmulationMxfp8LinearKernel=UpstreamEmulation,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: is_ppu))
    module = load_patch("vllm_sail/model_executor/kernels/linear/mxfp8.py")
    cls = module.PPUEmulationMxfp8LinearKernel
    assert cls.is_supported()[0] is is_ppu
    assert (
        cls.process_weights_after_loading
        is UpstreamEmulation.process_weights_after_loading
    )
    assert cls.apply_weights is UpstreamEmulation.apply_weights


def test_mxfp8_registration_precedes_cuda_and_supports_explicit_emulation(modules):
    linear_kernels = load_patch("vllm_sail/registry/linear_kernels/__init__.py")

    ppu, cuda = type("PPU", (), {}), type("CUDA", (), {})
    upstream = modules(
        "vllm.model_executor.kernels.linear",
        _POSSIBLE_MXFP8_KERNELS={"cuda": [cuda]},
    )
    modules("vllm.platforms.interface", PlatformEnum=SimpleNamespace(CUDA="cuda"))
    linear_kernels._register_first(ppu, "mxfp8")
    linear_kernels._register_first(ppu, "mxfp8")
    assert upstream._POSSIBLE_MXFP8_KERNELS["cuda"] == [ppu, cuda]
