# SPDX-License-Identifier: Apache-2.0
"""PPU indexer probes use SAIL capabilities rather than CUDA families."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/attention/mla_indexer.py"


@pytest.mark.parametrize("capability,varlen", [(89, True), (80, False)])
@pytest.mark.parametrize("available", [False, True])
def test_ppu_query_length_capabilities(modules, capability, varlen, available):
    modules(
        "vllm.platforms",
        current_platform=NS(
            is_ppu=lambda: True,
            is_device_capability=lambda cap: cap == capability,
        ),
    )
    namespace = {
        "_ppu_deep_gemm": NS(is_deep_gemm_supported=lambda: available),
        "_upstream_supports_varlen_paged_mqa_logits": lambda: pytest.fail("CUDA probe"),
        "_upstream_supports_flattened_device_query_lens": lambda: pytest.fail(
            "CUDA probe"
        ),
    }
    assert function(PATCH, "_supports_varlen_paged_mqa_logits", namespace)() == (
        varlen and available
    )
    assert (
        function(PATCH, "_supports_flattened_device_query_lens", namespace)()
        == available
    )


@pytest.mark.parametrize(
    "name",
    ["_supports_varlen_paged_mqa_logits", "_supports_flattened_device_query_lens"],
)
def test_non_ppu_query_length_capabilities_delegate(modules, name):
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: False))
    sentinel = object()
    assert function(PATCH, name, {"_upstream" + name: lambda: sentinel})() is sentinel
