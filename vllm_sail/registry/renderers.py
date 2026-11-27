# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register the Qwen4Exp chat renderer with vLLM's renderer/tokenizer registries.

Upstream exposes ``RendererRegistry.register`` and ``_TokenizerRegistry.register``
(``vllm/renderers/registry.py``, ``vllm/tokenizers/registry.py``), which add a
mode -> (module, class) entry resolved lazily by ``resolve_obj_by_qualname``. So
the ``qwen4exp`` mode needs **no patch**: the tokenizer stays the standard cached
HF tokenizer and only the renderer (template stage) is replaced, exactly like the
in-tree ``kimi_k3`` / ``inkling`` modes.

The PPU platform selects ``tokenizer_mode="qwen4exp"`` for the Qwen4Exp
architectures in ``PPUPlatform.apply_config_platform_defaults``; this module makes
that mode resolvable.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_registered = False

#: Renderer/tokenizer mode contributed by this plugin.
MODE = "qwen4exp"


def register() -> None:
    """Register the ``qwen4exp`` renderer and tokenizer mode. Idempotent."""
    global _registered
    if _registered:
        return

    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    if MODE not in TokenizerRegistry.tokenizers:
        TokenizerRegistry.register(MODE, "vllm.tokenizers.hf", "CachedHfTokenizer")
    else:
        logger.warning("Tokenizer mode %r already exists; keeping its owner.", MODE)

    if MODE not in RENDERER_REGISTRY.renderers:
        RENDERER_REGISTRY.register(
            MODE, "vllm_sail.renderers.qwen4exp", "Qwen4ExpRenderer"
        )
    else:
        logger.warning("Renderer mode %r already exists; keeping its owner.", MODE)

    _registered = True
