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


def test_v41_selector_preserves_cuda_and_other_backends(modules):
    upstream, ppu, other = (type(name, (), {}) for name in ("Upstream", "PPU", "Other"))
    platform = SimpleNamespace(ppu=True)
    modules(
        "vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: platform.ppu)
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=upstream
    )
    modules("vllm_sail.models.deepseek_v41.flashmla", DeepseekV41FlashMLAAttention=ppu)

    def original(config):
        return config

    provider = modules(
        "vllm.models.deepseek_v41.nvidia.model", _select_dsv4_attn_cls=original
    )
    replacement = load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")
    assert provider._select_dsv4_attn_cls is replacement._select_dsv4_attn_cls
    marker = getattr(provider._select_dsv4_attn_cls, replacement.PATCH_MARKER)
    assert marker[replacement._TARGET] is original
    assert provider._select_dsv4_attn_cls(upstream) is ppu
    assert provider._select_dsv4_attn_cls(other) is other
    platform.ppu = False
    assert provider._select_dsv4_attn_cls(upstream) is upstream
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")


@pytest.mark.parametrize("dequant_at_load", [True, False])
def test_v41_projection_preserves_weights_and_uses_bf16_helper(
    modules, dequant_at_load
):
    calls = []
    loaded = SimpleNamespace(element_size=lambda: 2 if dequant_at_load else 1)
    bf16 = loaded if dequant_at_load else object()
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda *args: bf16,
    )
    modules(
        "vllm.models.deepseek_v4.nvidia.ops.o_proj",
        deep_gemm_fp8_o_proj=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    class Upstream:
        forward_mqa = object()
        _forward_prefill = object()

    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=Upstream
    )
    modules("vllm_sail.models.deepseek_v4.flashmla", DeepseekV4FlashMLAAttention=object)
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    module = load_patch("vllm_sail/models/deepseek_v41/flashmla.py")
    attn = module.DeepseekV41FlashMLAAttention()
    attn.__dict__.update(
        _o_proj_block_size=32,
        wo_a=SimpleNamespace(weight=loaded, weight_scale=object()),
        wo_b=object(),
        rotary_emb=SimpleNamespace(cos_sin_cache=object()),
        n_local_groups=2,
        n_local_heads=16,
        nope_head_dim=448,
        rope_head_dim=64,
        o_lora_rank=1024,
    )
    o, pos = object(), object()
    attn._o_proj(o, pos)
    args, kwargs = calls.pop()
    assert args[:2] == (o, pos) and args[3].weight is bf16 and args[4] is attn.wo_b
    assert kwargs["heads_per_group"] == 8 and kwargs["einsum_recipe"] == (1, 1, 32)
    assert attn.wo_a.weight is loaded
    assert attn.forward_mqa is Upstream.forward_mqa
    assert attn._forward_prefill is Upstream._forward_prefill
