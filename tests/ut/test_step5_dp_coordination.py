# SPDX-License-Identifier: Apache-2.0
"""Check that draft coordination consumes the fused result without a vote."""

import ast
from contextvars import ContextVar
from pathlib import Path

import pytest


@pytest.mark.parametrize("ticket", [False, True])
@pytest.mark.parametrize("match", [False, True])
def test_draft_prediction_fallback(ticket, match):
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_sail/patch/performance/step5_dp_coordination.py"
    )
    function = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "coordinate_draft"
    )
    function.decorator_list = []
    fallbacks = []
    draft = ContextVar("draft", default=None)
    if ticket:
        draft.set(((8 if match else 4, 8, 1), "cached padding", 1))

    def coordinate(*args):
        fallbacks.append(args)
        return "native result"

    # No torch/dist in the namespace: consuming a ticket must not issue any
    # collective, so a stray all_reduce would raise instead of silently voting.
    namespace = dict(
        draft_context=draft,
        coordinate=coordinate,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    result = namespace["coordinate_draft"](8, False, "dp", 8, True, 1)

    assert draft.get() is None
    if ticket and match:
        assert result == (False, "cached padding", 1)
        assert not fallbacks
    else:
        assert result == "native result"
        assert fallbacks == [(8, False, "dp", 8, True, 1)]
