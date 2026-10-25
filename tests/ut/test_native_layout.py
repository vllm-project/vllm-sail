# SPDX-License-Identifier: Apache-2.0
"""Keep extension ownership and authored includes valid after a source move."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vllm_sail.native import manifest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("subtree", ["csrc/moe", "csrc/libtorch_stable/moe"])
def test_moe_sources_and_bindings_stay_in_the_moe_extension(subtree):
    main = manifest.Entry("csrc/cache_kernels.cu", "a")
    bindings = manifest.Entry(f"{subtree}/torch_bindings.cpp", "a")
    kernel = manifest.Entry(f"{subtree}/moe_align_sum_kernels.cu", "a")
    corpus = manifest.Manifest(
        upstream_ref="source-layout-fixture",
        tiers={"a": "compiled"},
        hazards={},
        compiled_tiers=("a",),
        entries=(main, bindings, kernel),
    )

    main_sources, moe_sources = corpus.compiled_sources()

    assert main_sources == (main.vendored,)
    assert moe_sources == (bindings.vendored, kernel.vendored)
    assert bindings.is_bindings
    assert set(main_sources).isdisjoint(moe_sources)
    assert not manifest.Entry(f"{subtree}_extra/kernel.cu", "a").is_moe


def test_group_quantization_has_one_compiled_owner_and_all_three_bindings():
    corpus = manifest.load()
    names = {
        "per_token_group_fp8_quant",
        "per_token_group_fp8_quant_packed",
        "per_token_group_quant_int8",
    }
    owners = [entry for entry in corpus.compiled() if names.intersection(entry.ops)]

    assert len(owners) == 1
    assert names <= set(owners[0].ops)
    sources, moe_sources = corpus.compiled_sources()
    assert (*sources, *moe_sources).count(owners[0].vendored) == 1
    assert not owners[0].is_moe
    bindings = next(
        entry for entry in corpus.compiled() if entry.is_bindings and not entry.is_moe
    )
    body = (ROOT / bindings.vendored).read_text()
    for name in names:
        assert f'"{name}(' in body
        # The schema and CUDA implementation must both survive binding pruning.
        assert re.search(r'ops\.impl\(\s*"' + name + r'"', body)


@pytest.mark.parametrize(
    "relative",
    [
        "csrc/plugin/sampler_bf16.hg",
        "csrc/plugin/libtorch_stable/moe/ep_scatter_kernels.hg",
    ],
)
def test_authored_kernel_headers_resolve_in_the_generated_include_root(relative):
    source = ROOT / relative
    headers = re.findall(r'^#include "([^"]+)"', source.read_text(), re.MULTILINE)

    assert headers
    for header in headers:
        assert any(
            (directory / header).is_file()
            for directory in (
                source.parent,
                ROOT / "csrc/upstream",
                ROOT / "csrc/plugin",
            )
        ), (relative, header)
