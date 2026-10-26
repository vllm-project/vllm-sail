# SPDX-License-Identifier: Apache-2.0
"""Real vLLM 0.31 loading interfaces, CPU tensors; no native PPU execution."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")


@pytest.fixture(autouse=True)
def standalone_layer(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter
    from vllm.platforms import current_platform

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(current_platform, "is_ppu", lambda: True, raising=False)


@pytest.mark.parametrize("kind", ["merged", "qkv"])
def test_actual_fused_loaders_keep_scale_grid_offsets(kind):
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )

    from vllm_sail.model_executor.layers.quantization.fp8_to_bf16 import Fp8ToBf16Config

    config = Fp8ToBf16Config.from_config(
        {"quant_method": "fp8", "weight_block_size": [128, 128]}
    )
    common = dict(params_dtype=torch.bfloat16, quant_config=config, bias=False)
    if kind == "merged":
        layer = MergedColumnParallelLinear(128, [128, 128], **common)
        shards = [0, 1]
    else:
        layer = QKVParallelLinear(128, 128, 1, **common)
        shards = ["q", "k", "v"]
    layer._cpu_skip_gemm_dispatch = True
    for index, shard in enumerate(shards):
        layer.weight.weight_loader(
            layer.weight, torch.ones((128, 128)).to(torch.float8_e4m3fn), shard
        )
        layer.weight_scale_inv.weight_loader(
            layer.weight_scale_inv, torch.tensor([[float(index + 1)]]), shard
        )
    layer.quant_method.process_weights_after_loading(layer)
    for index in range(len(shards)):
        assert torch.all(layer.weight[index * 128 : (index + 1) * 128] == index + 1)


def test_bf16_linear_uses_actual_vllm_parameters_and_loader():
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm_sail.model_executor.layers.quantization.online_conversion.linear import (
        Fp8ConversionLinearMethod,
    )

    layer = torch.nn.Module()
    layer._cpu_skip_gemm_dispatch = True
    method = Fp8ConversionLinearMethod("bf16")
    method.create_weights(
        layer,
        128,
        [128, 65],
        128,
        193,
        torch.bfloat16,
        weight_loader=default_weight_loader,
    )
    source = torch.ones((193, 128)).to(torch.float8_e4m3fn)
    layer.weight.weight_loader(layer.weight, source)
    layer.weight_scale_inv.weight_loader(
        layer.weight_scale_inv, torch.tensor([[2.0], [3.0]])
    )
    method.process_weights_after_loading(layer)
    expected = torch.cat([torch.full((128, 128), 2.0), torch.full((65, 128), 3.0)]).to(
        torch.bfloat16
    )
    assert torch.equal(layer.weight, expected)
    pointer = layer.weight.data_ptr()
    method.process_weights_after_loading(layer)
    assert layer.weight.data_ptr() == pointer
    x = torch.ones((2, 128), dtype=torch.bfloat16)
    torch.testing.assert_close(method.apply(layer, x), x @ expected.t(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "sizes,input_part,total_input,total_output",
    [
        ([65, 128], 128, 128, 193),
        ([128], 65, 130, 128),
        ([65], 128, 128, 130),
    ],
)
def test_misaligned_shards_fail_before_allocating(
    sizes, input_part, total_input, total_output
):
    from vllm_sail.model_executor.layers.quantization.online_conversion.linear import (
        Fp8ConversionLinearMethod,
    )

    layer = torch.nn.Module()
    with pytest.raises(ValueError, match="align"):
        Fp8ConversionLinearMethod("int8").create_weights(
            layer, input_part, sizes, total_input, total_output, torch.bfloat16
        )
    assert list(layer.parameters()) == []


def test_real_linear_dtype_guard_and_config_mapping():
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_sail.model_executor.layers.quantization.fp8_to_int8 import Fp8ToInt8Config
    from vllm_sail.model_executor.layers.quantization.online_conversion.loading import (
        CheckedUnquantizedLinearMethod,
    )

    config = Fp8ToInt8Config.from_config(
        {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.language_model.keep"],
        }
    )
    mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
    config.apply_vllm_mapper(mapper)
    config.apply_vllm_mapper(mapper)
    layer = object.__new__(LinearBase)
    assert isinstance(
        config.get_quant_method(layer, "model.keep"), CheckedUnquantizedLinearMethod
    )
    float_layer = torch.nn.Module()
    method = CheckedUnquantizedLinearMethod()
    method.create_weights(
        float_layer,
        128,
        [128],
        128,
        128,
        torch.bfloat16,
        weight_loader=default_weight_loader,
    )
    with pytest.raises(ValueError, match="storage/selection mismatch"):
        float_layer.weight.weight_loader(
            float_layer.weight, torch.ones((128, 128)).to(torch.float8_e4m3fn)
        )


def test_moe_source_allocation_and_int8_conversion():
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm_sail.model_executor.layers.quantization.online_conversion.moe import (
        Fp8SourceMoE,
    )

    source = Fp8SourceMoE()
    source.moe = SimpleNamespace(
        tp_size=1,
        is_act_and_mul=True,
        hidden_dim_unpadded=128,
        intermediate_size_per_partition_unpadded=65,
        has_bias=True,
    )
    layer = torch.nn.Module()
    source.create_weights(
        layer, 2, 128, 65, torch.bfloat16, weight_loader=default_weight_loader
    )
    assert layer.w13_weight.shape == (2, 130, 128)
    assert layer.w13_weight_scale_inv.shape == (2, 2, 1)
    assert layer.w13_bias.dtype == torch.bfloat16
    for name in ("w13", "w2"):
        weight = getattr(layer, name + "_weight")
        weight.weight_loader(weight, torch.ones_like(weight))
        getattr(layer, name + "_weight_scale_inv").data.fill_(2.0)
    source._convert(layer, to_int8=True)
    assert torch.all(layer.w13_weight == 127)
    assert layer.w13_weight_scale.shape == (2, 130, 1)
    assert torch.allclose(layer.w13_weight_scale, torch.full((2, 130, 1), 2 / 127))


def test_int8_moe_uses_late_oracle_and_upstream_kernel_signature(monkeypatch):
    import inspect

    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.oracle import int8 as oracle

    from vllm_sail.model_executor.layers.quantization.online_conversion.moe import (
        Fp8ToInt8MoEMethod,
    )

    events = []
    create_signature = inspect.signature(oracle.make_int8_moe_kernel)
    apply_signature = inspect.signature(FusedMoEKernel.apply)

    class Kernel:
        fused_experts = SimpleNamespace(
            process_weights_after_loading=lambda layer: events.append("pack")
        )

        def apply(self, *args, **kwargs):
            apply_signature.bind(self, *args, **kwargs)
            events.append(kwargs)
            return args[0]

    def create(**kwargs):
        create_signature.bind(**kwargs)
        assert kwargs["int8_backend"] == oracle.Int8MoeBackend.TRITON
        events.append("create")
        return Kernel()

    monkeypatch.setattr(
        oracle,
        "select_int8_moe_backend",
        lambda **kw: (oracle.Int8MoeBackend.TRITON, object),
    )
    monkeypatch.setattr(oracle, "make_int8_moe_kernel", create)
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(
        tp_size=1,
        is_act_and_mul=True,
        hidden_dim_unpadded=128,
        intermediate_size_per_partition_unpadded=128,
        has_bias=False,
    )
    layer._expert_routing_tables = lambda: None
    layer.swiglu_limit = 7.0
    layer.activation = "silu"
    layer.global_num_experts = 2
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    method = Fp8ToInt8MoEMethod(layer)
    method.create_weights(layer, 2, 128, 128, torch.bfloat16)
    for name in ("w13", "w2"):
        getattr(layer, name + "_weight").data.fill_(1)
        getattr(layer, name + "_weight_scale_inv").data.fill_(1)
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)
    assert events == ["create", "pack"]
    assert method.moe_quant_config.gemm1_clamp_limit == 7.0
    shared = object()
    x = torch.ones((1, 128), dtype=torch.bfloat16)
    assert (
        method.apply(
            layer,
            x,
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int32),
            shared,
            x,
        )
        is x
    )
    assert events[-1]["shared_experts"] is shared
    assert events[-1]["shared_experts_input"] is x


@pytest.mark.parametrize(
    "source_float,target_float",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_source_and_target_selection_are_independent(source_float, target_float):
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_sail.model_executor.layers.quantization.fp8_to_int8 import Fp8ToInt8Config
    from vllm_sail.model_executor.layers.quantization.online_conversion.loading import (
        CheckedUnquantizedLinearMethod,
    )

    prefix = "model.layers.0.proj"
    config = Fp8ToInt8Config(
        {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "ignored_layers": [prefix] if source_float else [],
        }
    )
    config.ignored_layers = [prefix] if target_float else []
    method = config.get_quant_method(object.__new__(LinearBase), prefix)
    if source_float and target_float:
        assert isinstance(method, CheckedUnquantizedLinearMethod)
    else:
        assert method.source_dtype == ("bf16" if source_float else "fp8")
        assert method.target_dtype == ("bf16" if target_float else "int8")


def test_recipe_kept_fp8_layer_is_decoded_not_cast():
    from vllm.model_executor.layers.linear import ReplicatedLinear

    from vllm_sail.model_executor.layers.quantization.fp8_to_int8 import Fp8ToInt8Config

    config = Fp8ToInt8Config({"quant_method": "fp8", "weight_block_size": [128, 128]})
    config.maybe_update_config(
        "Qwen3-Next-80B-A3B-Instruct", SimpleNamespace(model_type="qwen3_next")
    )
    layer = ReplicatedLinear(
        128,
        128,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.layers.0.linear_attn.out_proj",
    )
    layer._cpu_skip_gemm_dispatch = True
    layer.weight.weight_loader(
        layer.weight, torch.ones((128, 128)).to(torch.float8_e4m3fn)
    )
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch.tensor([[3.0]]))
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.bfloat16
    assert torch.all(layer.weight == 3)


@pytest.mark.parametrize("kind", ["merged", "qkv", "row"])
def test_bf16_real_fused_loaders_and_conversion(monkeypatch, kind):
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    from vllm_sail.model_executor.kernels.linear.scaled_mm import ppu
    from vllm_sail.model_executor.layers.quantization.bf16_to_int8 import (
        Bf16ToInt8Config,
    )
    from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (
        quantize_int8,
    )

    class Kernel:
        def __init__(self, *args):
            pass

        def process_weights_after_loading(self, layer):
            pass

    monkeypatch.setattr(ppu, "PPUInt8ScaledMMLinearKernel", Kernel)
    config = Bf16ToInt8Config()
    config.maybe_update_config("unknown")
    common = dict(params_dtype=torch.bfloat16, quant_config=config, bias=False)
    if kind == "merged":
        layer = MergedColumnParallelLinear(64, [65, 63], **common)
        shards = [(0, 65), (1, 63)]
    elif kind == "qkv":
        layer = QKVParallelLinear(64, 32, 2, **common)
        shards = [("q", 64), ("k", 64), ("v", 64)]
    else:
        layer = RowParallelLinear(64, 65, **common)
        shards = [(None, 65)]
    expected = []
    for i, (shard, rows) in enumerate(shards):
        tensor = torch.randn(rows, 64, dtype=torch.bfloat16) * (i + 1)
        expected.append(tensor)
        args = () if shard is None else (shard,)
        layer.weight.weight_loader(layer.weight, tensor, *args)
    assert layer.weight_scale_inv is None
    q, scales = quantize_int8(torch.cat(expected))
    layer.quant_method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight, q)
    assert torch.equal(layer.weight_scale, scales)
    layer.quant_method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight, q)


def test_bf16_config_uses_real_upstream_factory(tmp_path):
    from vllm.config import LoadConfig
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.model_loader.weight_utils import get_quant_config

    from vllm_sail.model_executor.layers.quantization.bf16_to_int8 import (
        Bf16ToInt8Config,
    )

    register_quantization_config("bf16-to-int8")(Bf16ToInt8Config)
    model = SimpleNamespace(
        quantization="bf16-to-int8",
        quantization_config=None,
        hf_config=SimpleNamespace(),
        hf_overrides={},
        model=str(tmp_path),
        revision=None,
    )
    config = get_quant_config(model, LoadConfig())
    config.maybe_update_config(model.model, model.hf_config)
    assert config.recipe_id == "_default"
    model.hf_config.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="unquantized"):
        get_quant_config(model, LoadConfig())


def test_bf16_moe_conversion_with_real_loader_metadata():
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm_sail.model_executor.layers.quantization.online_conversion.moe import (
        Fp8SourceMoE,
    )
    from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (
        quantize_int8,
    )

    source = Fp8SourceMoE()
    source.source_dtype = "bf16"
    source.moe = SimpleNamespace(
        tp_size=1,
        is_act_and_mul=True,
        hidden_dim_unpadded=64,
        intermediate_size_per_partition_unpadded=65,
        has_bias=False,
    )
    layer = torch.nn.Module()
    source.create_weights(
        layer, 2, 64, 65, torch.bfloat16, weight_loader=default_weight_loader
    )
    refs = {}
    for name in ("w13", "w2"):
        weight = getattr(layer, name + "_weight")
        data = torch.randn_like(weight)
        weight.weight_loader(weight, data)
        refs[name] = [quantize_int8(expert) for expert in data]
        assert getattr(layer, name + "_weight_scale_inv") is None
    source._convert(layer, to_int8=True)
    for name, values in refs.items():
        assert torch.equal(
            getattr(layer, name + "_weight"), torch.stack([q for q, _ in values])
        )
        assert torch.equal(
            getattr(layer, name + "_weight_scale"), torch.stack([s for _, s in values])
        )


@pytest.mark.parametrize("source_dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_real_routed_experts_checkpoint_loader(
    monkeypatch, source_dtype, tp_size, tp_rank
):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from vllm.model_executor.layers.fused_moe.oracle import int8 as oracle
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    from vllm_sail.model_executor.layers.quantization.bf16_to_int8 import (
        Bf16ToInt8Config,
    )
    from vllm_sail.model_executor.layers.quantization.fp8_to_int8 import Fp8ToInt8Config

    monkeypatch.setattr(
        oracle,
        "select_int8_moe_backend",
        lambda **kw: (oracle.Int8MoeBackend.TRITON, object),
    )
    parallel = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=tp_rank,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    moe = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=256,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    manager = ExpertMapManager(
        max_num_batched_tokens=8,
        top_k=1,
        global_num_experts=2,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    if source_dtype == "bf16":
        config = Bf16ToInt8Config()
    else:
        config = Fp8ToInt8Config(
            {"quant_method": "fp8", "weight_block_size": [128, 128]}
        )
    config.maybe_update_config("unknown")
    layer = RoutedExperts(
        "model.layers.0.mlp.experts", torch.bfloat16, moe, config, manager
    )
    weights = []
    for expert in range(2):
        for proj, shape in (
            ("gate_proj", (256, 128)),
            ("up_proj", (256, 128)),
            ("down_proj", (128, 256)),
        ):
            value = expert + (2 if proj == "up_proj" else 1)
            dtype = torch.bfloat16 if source_dtype == "bf16" else torch.float8_e4m3fn
            name = f"{expert}.{proj}"
            tensor = torch.full(shape, value, dtype=dtype)
            if tp_size == 2:
                if proj == "down_proj":
                    tensor[:, 128:] = value + 4
                else:
                    tensor[128:] = value + 4
            weights.append((name + ".weight", tensor))
            if source_dtype == "fp8":
                weights.append(
                    (
                        name + ".weight_scale_inv",
                        torch.full((shape[0] // 128, shape[1] // 128), 2.0),
                    )
                )
    loaded = list(layer.load_weights(weights))
    assert len(loaded) == len(weights)
    if tp_size == 2:
        for expert in range(2):
            assert torch.all(
                layer.w13_weight[expert, :128].float() == expert + 1 + 4 * tp_rank
            )
            assert torch.all(
                layer.w13_weight[expert, 128:].float() == expert + 2 + 4 * tp_rank
            )
            assert torch.all(
                layer.w2_weight[expert].float() == expert + 1 + 4 * tp_rank
            )
        if source_dtype == "fp8":
            assert layer.w13_weight_scale_inv.shape == (2, 2, 1)
            assert layer.w2_weight_scale_inv.shape == (2, 1, 1)
            assert torch.all(layer.w13_weight_scale_inv == 2)
        return
    layer.quant_method._convert(layer, to_int8=True)
    factor = 2 if source_dtype == "fp8" else 1
    for expert in range(2):
        assert torch.all(layer.w13_weight[expert] == 127)
        torch.testing.assert_close(
            layer.w13_weight_scale[expert, :256],
            torch.full((256, 1), factor * (expert + 1) / 127),
        )
        torch.testing.assert_close(
            layer.w13_weight_scale[expert, 256:],
            torch.full((256, 1), factor * (expert + 2) / 127),
        )
