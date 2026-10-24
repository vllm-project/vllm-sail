# SPDX-License-Identifier: Apache-2.0
"""Preserve upstream's self-dispatching MoE factory path on the PPU wrapper."""

from types import SimpleNamespace

from tests.support.source import function


def test_factory_retains_passthrough_before_selecting_a_communicator():
    expected = object()
    factory = function(
        "vllm_sail/patch/enhancement/deepep.py",
        "_maybe_make_prepare_finalize_body",
        {"PassThroughPrepareAndFinalize": lambda: expected},
    )
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_passthrough_all2all=True)
    )
    assert factory(moe, None) is expected
