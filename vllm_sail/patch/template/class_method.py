# SPDX-License-Identifier: Apache-2.0
"""TEMPLATE — class-attribute patch. Copy into bugfix/, enhancement/ or
performance/, then replace every TODO. This file is excluded from lint and is
never imported.

Use this form for a method or property. The dotted ``"TargetClass.method"`` path
makes ``@patch`` resolve the class internally, so the patch module does not need
to import it, and descriptor semantics (``staticmethod`` / ``classmethod`` /
``property``) are preserved from the original.
"""

from __future__ import annotations

from vllm_sail.patch.utils import patch


@patch(
    # TODO: module that owns the class.
    "vllm.some_module",
    # TODO: dotted path "TargetClass.attribute". Nested classes work too:
    # "Outer.Inner.method".
    "TargetClass.forward",
    reason="TODO",
    remove_when="TODO",
)
def forward(self, hidden_states):
    # Prefer delegating. Capture the original at module scope before patching:
    #
    #   from vllm.some_module import TargetClass
    #   _upstream_forward = TargetClass.forward
    #
    # then:
    #
    #   from vllm.platforms import current_platform   # lazy: inside the body
    #   if not current_platform.is_ppu():
    #       return _upstream_forward(self, hidden_states)
    #   ...
    raise NotImplementedError("TODO")


# --- descriptor variants ---------------------------------------------------
# @patch must stay OUTSIDE the descriptor:
#
# @patch("vllm.some_module", "TargetClass.from_config", reason=..., ...)
# @classmethod
# def from_config(cls, config): ...
#
# @patch("vllm.some_module", "TargetClass.helper", reason=..., ...)
# @staticmethod
# def helper(x): ...
#
# A property patch replaces only the getter; the original's setter, deleter and
# docstring are preserved automatically:
#
# @patch("vllm.some_module", "TargetClass.value", reason=..., ...)
# def value(self): ...
