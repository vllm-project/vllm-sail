# SPDX-License-Identifier: Apache-2.0
"""Exercise the new loaders without importing torch or vLLM.

Quantized WK weights/scales must reach their own parameter loaders unchanged;
the NVIDIA loader instead maps these names into an unquantized fused shard.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "vllm_sail/models/deepseek_v32"


def _loader(filename, class_name):
    path = _ROOT / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if getattr(node, "name", None) == class_name)
    method = next(
        node for node in cls.body if getattr(node, "name", None) == "load_weights"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, method], type_ignores=[])
    )
    namespace = {
        "fused_moe_make_expert_params_mapping": lambda *args, **kwargs: [],
        "get_spec_layer_idx_from_weight_name": (
            lambda config, name: 1 if ".layers.1." in name else None
        ),
        "is_pp_missing_parameter": lambda name, model: False,
        "maybe_remap_kv_scale_name": lambda name, params: name,
        "default_weight_loader": lambda *args: pytest.fail("Expected parameter loader"),
        "is_mtp_completeness_check_enabled": lambda: True,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["load_weights"]


@pytest.mark.parametrize(
    "filename,class_name,is_mtp",
    [("model.py", "DeepseekV32Model", False), ("mtp.py", "DeepseekV32MTP", True)],
)
def test_wk_weights_and_scales_use_separate_parameter_loaders(
    filename, class_name, is_mtp
):
    layer = 1 if is_mtp else 0
    source_prefix = f"model.layers.{layer}.self_attn.indexer."
    target_prefix = (
        source_prefix.replace(".self_attn.", ".mtp_block.self_attn.")
        if is_mtp
        else source_prefix
    )
    suffixes = ("wk.weight", "wk.weight_scale", "weights_proj.weight")
    loaded = []
    params = {}
    weights = []
    for suffix in suffixes:
        weight = object()
        param = SimpleNamespace()

        def load(parameter, value, *shards):
            assert shards == ()
            loaded.append((parameter, value))

        param.weight_loader = load
        params[target_prefix + suffix] = param
        weights.append((source_prefix + suffix, weight))
    model = SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=0),
        num_redundant_experts=0,
        named_parameters=lambda: params.items(),
        is_unused_checkpoint_weight=lambda name: False,
        _rewrite_spec_layer_name=lambda layer, name: name.replace(
            ".self_attn.", ".mtp_block.self_attn."
        ),
        model=SimpleNamespace(mtp_start_layer_idx=1, num_mtp_layers=1),
    )

    result = _loader(filename, class_name)(model, weights)

    assert result == set(params)
    assert loaded == [
        (params[target_prefix + suffix], weight)
        for suffix, (_, weight) in zip(suffixes, weights, strict=True)
    ]
