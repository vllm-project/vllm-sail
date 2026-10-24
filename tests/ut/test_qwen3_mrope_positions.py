# SPDX-License-Identifier: Apache-2.0
"""Keep mRoPE geometry across the per-layer quantization delegation seam."""

from dataclasses import dataclass, replace
from types import SimpleNamespace as NS

import pytest

from tests.support.source import definition, function

PATCH = "vllm_sail/patch/enhancement/models/qwen3_moe.py"


@dataclass
class Config:
    model_config: object
    cache_config: object
    quant_config: object


@pytest.mark.parametrize("factor", [None, 7])
@pytest.mark.parametrize("quant", ["inherited", "disabled", "overridden"])
def test_decoder_preserves_geometry_and_shallow_quant_override(factor, quant):
    calls = []
    unset = object()
    original_quant, override = object(), object()
    config = Config(NS(hf_text_config=object()), object(), original_quant)

    def upstream(
        self,
        *,
        vllm_config,
        prefix,
        is_fused_checkpoint_transposed,
        mrope_positions_factor=4,
    ):
        calls.append(
            (
                self,
                vllm_config,
                prefix,
                is_fused_checkpoint_transposed,
                mrope_positions_factor,
            )
        )

    namespace = {
        "_UNSET": unset,
        "_upstream_decoder_layer_init": upstream,
        "_with_quant_config": lambda c, q: replace(c, quant_config=q),
    }
    wrapper = function(PATCH, "decoder_layer_init", namespace)
    args = {} if factor is None else {"mrope_positions_factor": factor}
    if quant != "inherited":
        args["quant_config"] = None if quant == "disabled" else override
    instance = object()
    wrapper(
        instance,
        config,
        prefix="model.layers.0",
        is_fused_checkpoint_transposed=True,
        **args,
    )
    assert len(calls) == 1
    own, delegated, prefix, transposed, seen_factor = calls[0]
    assert (own, prefix, transposed, seen_factor) == (
        instance,
        "model.layers.0",
        True,
        4 if factor is None else factor,
    )
    assert delegated.model_config is config.model_config
    assert delegated.cache_config is config.cache_config
    assert delegated.quant_config is (
        original_quant
        if quant == "inherited"
        else None
        if quant == "disabled"
        else override
    )
    assert (delegated is config) == (quant == "inherited")
    assert config.quant_config is original_quant


@pytest.mark.upstream_source
@pytest.mark.parametrize("factor", [None, 7])
@pytest.mark.parametrize("quant", ["inherited", "disabled", "overridden"])
def test_real_decoder_constructor_receives_geometry(
    upstream_source_root, factor, quant
):
    import ast

    source = upstream_source_root / "vllm/model_executor/models/qwen3_moe.py"
    constructor = definition(source, "Qwen3MoeDecoderLayer.__init__")
    cls = ast.ClassDef(
        name="Qwen3MoeDecoderLayer",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[constructor],
        decorator_list=[],
    )
    calls = []

    class Parent:
        pass

    def attention(**kwargs):
        calls.append(kwargs)
        return NS(**kwargs)

    namespace = {
        "Parent": Parent,
        "Qwen3MoeAttention": attention,
        "Qwen3MoeSparseMoeBlock": lambda **kwargs: NS(**kwargs),
        "RMSNorm": lambda *args, **kwargs: object(),
        "extract_layer_index": lambda prefix: 0,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    layer = namespace["Qwen3MoeDecoderLayer"]
    unset, inherited, override = object(), object(), object()
    hf = NS(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        rope_parameters={},
        rms_norm_eps=1e-6,
        num_experts=2,
        decoder_sparse_step=1,
    )
    config = Config(NS(hf_text_config=hf), object(), inherited)
    local = {
        "_UNSET": unset,
        "_upstream_decoder_layer_init": layer.__init__,
        "_with_quant_config": function(
            PATCH, "_with_quant_config", {"replace": replace}
        ),
    }
    layer.__init__ = function(PATCH, "decoder_layer_init", local)
    args = {} if factor is None else {"mrope_positions_factor": factor}
    if quant != "inherited":
        args["quant_config"] = None if quant == "disabled" else override
    obj = layer(config, prefix="model.layers.0", **args)
    assert len(calls) == 1 and calls[0]["mrope_positions_factor"] == (
        4 if factor is None else factor
    )
    expected = (
        inherited if quant == "inherited" else None if quant == "disabled" else override
    )
    assert obj.self_attn.quant_config is expected
    assert obj.mlp.vllm_config.quant_config is expected
    assert obj.mlp.vllm_config.model_config is config.model_config
    assert config.quant_config is inherited
