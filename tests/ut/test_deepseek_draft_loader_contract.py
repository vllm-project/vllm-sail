# SPDX-License-Identifier: Apache-2.0
"""Release draft loaders own TP sink slices and post-load finalization."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function


def check_loader(load, kind, shared):
    calls = []

    class Weight:
        def __init__(self, values):
            self.values = tuple(values)
            self.shape = (len(self.values),)

        def __getitem__(self, index):
            return Weight(self.values[index])

    class Sink:
        def __init__(self):
            self.values = [-float("inf")] * 8

        def __getitem__(self, index):
            assert index == slice(None, 2)
            return NS(copy_=self.copy)

        def copy(self, weight):
            calls.append(("sink", weight.values))
            self.values[:2] = weight.values

    loaded_weight = Weight(range(16))
    sink_name = "model.layers.0.attn.attn_sink"
    shared_name = "model.layers.0.ffn.shared_experts.down_proj.weight"
    sink, shared_param = Sink(), object()

    def default_loader(param, weight):
        assert param is shared_param, (
            "padded sink must be copied into its local head slice"
        )
        assert weight is loaded_weight
        calls.append(("shared", weight.values))

    def unexpected_fused(*args):
        pytest.fail("the release uses a separate shared-expert parameter")

    experts = NS(
        has_fused_shared_experts=True, load_shared_expert_weight=unexpected_fused
    )
    ffn = NS(use_native_mega_moe=True, experts=experts)
    layers = [NS(ffn=ffn)] if kind == "dspark" else {"0": NS(mtp_block=NS(ffn=ffn))}
    model = NS(
        layers=layers, confidence_head=None, mtp_start_layer_idx=0, num_mtp_layers=1
    )
    layer = NS(
        model=model,
        config=NS(n_routed_experts=2, num_attention_heads=16, num_hidden_layers=0),
        pad_shared_expert=False,
        named_parameters=lambda: [(sink_name, sink), (shared_name, shared_param)],
        _remap_dspark_name=lambda name: name,
        _to_spec_layer_name=lambda name: name,
        _rewrite_spec_layer_name=lambda index, name: name,
        process_weights_after_loading=lambda: calls.append(("finalize",)),
    )
    namespace = {
        "make_deepseek_v4_expert_params_mapping": lambda count: [],
        "_duplicate_context_wkv_weights": lambda weights, count: weights,
        "get_spec_layer_idx_from_weight_name": lambda config, name: 0,
        "is_mtp_completeness_check_enabled": lambda: True,
        "default_weight_loader": default_loader,
        "get_tensor_model_parallel_world_size": lambda: 8,
        "get_tensor_model_parallel_rank": lambda: 3,
        "current_platform": NS(is_ppu=lambda: True),
        "logger": NS(info_once=lambda *args: None),
    }
    checkpoint_name = (
        "model.layers.0.ffn.shared_experts.w2.weight" if shared else sink_name
    )
    result = load(namespace)(layer, [(checkpoint_name, loaded_weight)])
    assert result == {shared_name if shared else sink_name}
    assert calls == [
        ("shared", tuple(range(16))) if shared else ("sink", (6, 7)),
        ("finalize",),
    ]
    assert sink.values == (
        [-float("inf")] * 8 if shared else [6, 7] + [-float("inf")] * 6
    )


@pytest.mark.parametrize("kind", ["dspark", "mtp"])
@pytest.mark.parametrize("shared", [False, True])
def test_plugin_loader_preserves_release_tensor_ownership(kind, shared):
    path = f"vllm_sail/models/deepseek_v4/{kind}.py"
    name = "load_weights" if kind == "dspark" else "DeepSeekV4MTP.load_weights"
    check_loader(lambda ns: function(path, name, ns), kind, shared)


@pytest.mark.upstream_source
@pytest.mark.parametrize("kind", ["dspark", "mtp"])
@pytest.mark.parametrize("shared", [False, True])
def test_actual_release_uses_the_same_tensor_ownership(
    upstream_source_root, kind, shared
):
    path = upstream_source_root / f"vllm/models/deepseek_v4/nvidia/{kind}.py"
    cls = "DSparkDeepseekV4ForCausalLM" if kind == "dspark" else "DeepSeekV4MTP"
    check_loader(lambda ns: function(path, cls + ".load_weights", ns), kind, shared)
