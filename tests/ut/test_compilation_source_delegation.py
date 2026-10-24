# SPDX-License-Identifier: Apache-2.0
"""The PPU splitting boundary must preserve the current upstream policy."""

import ast
from enum import Enum
from types import SimpleNamespace as NS

import pytest

from tests.support.source import definition, function


class Graph(Enum):
    NONE = "none"
    FULL = "full"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full-and-piecewise"


@pytest.mark.upstream_source
@pytest.mark.parametrize("case", ["default", "custom", "empty", "eager", "deepep-ht"])
def test_indexer_boundary_delegates_updated_upstream_policy(upstream_source_root, case):
    path = upstream_source_root / "vllm/config/compilation.py"
    mode = object()
    original = function(
        path,
        "CompilationConfig.set_splitting_ops_for_v1",
        {
            "CompilationMode": NS(VLLM_COMPILE=mode),
            "CUDAGraphMode": Graph,
            "logger": NS(warning_once=lambda *args: None, info=lambda *args: None),
        },
    )
    attention_ops = next(
        node.value
        for node in definition(path, "CompilationConfig").body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_attention_ops"
    )
    attention_ops = ast.literal_eval(attention_ops)
    replacement = function(
        "vllm_sail/patch/enhancement/residual/compilation_attention_ops.py",
        "install.set_splitting_ops_for_v1",
        {
            "_PPU_ATTN_OP": "vllm::ppu_sparse_attn_indexer",
            "_upstream_set_splitting_ops": original,
        },
    )

    class Config:
        _attention_ops = list(attention_ops)
        set_splitting_ops_for_v1 = replacement

    config = Config()
    config.mode = "eager" if case == "eager" else mode
    config.pass_config = NS(
        fuse_attn_quant=False, fuse_rope_kvcache=False, fuse_qk_norm_rope_kvcache=False
    )
    config.use_inductor_graph_partition = False
    config.cudagraph_mode = Graph.FULL_AND_PIECEWISE
    config.splitting_ops = {"custom": ["custom::op"], "empty": []}.get(case)
    backend = "deepep_high_throughput" if case == "deepep-ht" else "deepep_low_latency"
    for _ in range(2):
        config.set_splitting_ops_for_v1(backend, 8)
    assert Config._attention_ops == attention_ops + ["vllm::ppu_sparse_attn_indexer"]
    if case in ("default", "deepep-ht"):
        assert "vllm::ppu_sparse_attn_indexer" in config.splitting_ops
        assert "vllm::qwen4_exp_qsa_prepare_with_output" in config.splitting_ops
    elif case == "custom":
        assert config.splitting_ops == ["custom::op"]
    else:
        assert config.splitting_ops == []
    expected = {
        "empty": Graph.FULL,
        "deepep-ht": Graph.NONE,
    }.get(case, Graph.FULL_AND_PIECEWISE)
    assert config.cudagraph_mode is expected
