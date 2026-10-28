# SPDX-License-Identifier: Apache-2.0
"""Register the PPU sparse-attention indexer op as a piecewise split point.

The fork adds ``"vllm::ppu_sparse_attn_indexer"`` (+1 line) to
``CompilationConfig._attention_ops`` in ``vllm/config/compilation.py`` — the
class-level list of attention ops that ``set_splitting_ops_for_v1`` copies into
``splitting_ops`` for piecewise cudagraphs. Without it, the PPU indexer op
(registered by ``vllm_sail.ops`` and dispatched from the patched
``SparseAttentionIndexer.forward_ppu``) is not treated as an attention
boundary, unlike its upstream sibling ``vllm::sparse_attn_indexer``.

This is distinct from ``PPUPlatform.apply_config_platform_defaults`` appending
``+sparse_attn_indexer`` to ``custom_ops``: that enables the CustomOp dispatch,
while ``_attention_ops`` controls compilation graph splitting.

Delegating patch: the PPU op is appended to the class-level list (idempotently)
just before upstream's method copies it into ``splitting_ops``. No upstream body
is copied.
"""

from __future__ import annotations

from vllm_sail.patch.utils import patch

_MODULE = "vllm.config.compilation"
_TARGET = "CompilationConfig.set_splitting_ops_for_v1"
_PPU_ATTN_OP = "vllm::ppu_sparse_attn_indexer"
REASON = (
    "The PPU sparse-attention indexer torch op must be a piecewise-cudagraph "
    "splitting point exactly like upstream's vllm::sparse_attn_indexer. The "
    "fork adds it to CompilationConfig._attention_ops; without it the op is "
    "traced into a piecewise graph segment instead of being an attention "
    "boundary."
)
AFFECTED_VERSIONS = ">=0.32.0,<0.33.0"
REMOVE_WHEN = (
    "upstream includes vllm::ppu_sparse_attn_indexer in "
    "CompilationConfig._attention_ops itself, or the PPU indexer stops "
    "registering its own torch op and reuses vllm::sparse_attn_indexer."
)

#: (target, reason, affected versions, remove_when) for every patch installed
#: by this module. Read by the metadata-shape unit tests.
METADATA = ((f"{_MODULE}.{_TARGET}", REASON, AFFECTED_VERSIONS, REMOVE_WHEN),)

_upstream_set_splitting_ops = None
_installed = False


def install() -> None:
    """Apply the patch. Requires vLLM to be importable; idempotent."""
    global _upstream_set_splitting_ops, _installed
    if _installed:
        return

    from vllm.config.compilation import CompilationConfig

    if _upstream_set_splitting_ops is None:
        _upstream_set_splitting_ops = CompilationConfig.set_splitting_ops_for_v1

    @patch(
        _MODULE,
        _TARGET,
        reason=REASON,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def set_splitting_ops_for_v1(self, all2all_backend, data_parallel_size=1):
        attention_ops = type(self)._attention_ops
        if _PPU_ATTN_OP not in attention_ops:
            attention_ops.append(_PPU_ATTN_OP)
        return _upstream_set_splitting_ops(self, all2all_backend, data_parallel_size)

    _installed = True
