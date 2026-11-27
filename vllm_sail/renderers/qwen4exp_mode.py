# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-free selection of the Qwen4Exp renderer mode.

Upstream's architecture -> ``tokenizer_mode`` table has no Qwen4Exp entry, so a
Qwen4Exp checkpoint resolves to the HF renderer. A template-less Qwen4Exp
checkpoint then fails every chat request, because the HF renderer requires a
resolved Jinja template. The PPU platform defaults these architectures to the
out-of-tree ``qwen4exp`` mode (registered by ``vllm_sail.registry.renderers``),
whose renderer drives the checkpoint's native Python encoder when no template
exists.

This module imports nothing from vLLM so the selection logic stays unit-testable
on a bare CPU runner; ``PPUPlatform.apply_config_platform_defaults`` calls it.
"""

from __future__ import annotations

from typing import Any

#: Architectures that render chat through the Qwen4Exp native encoder.
QWEN4EXP_ARCHITECTURES = (
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
)

#: Renderer/tokenizer mode contributed by ``vllm_sail.registry.renderers``.
QWEN4EXP_MODE = "qwen4exp"


def select_qwen4exp_mode(model_config: Any) -> None:
    """Default Qwen4Exp architectures to the ``qwen4exp`` renderer mode.

    Only an explicit ``"auto"`` is overridden, mirroring upstream's own
    arch-based selection; a user-chosen ``--tokenizer-mode`` is respected.
    """
    if getattr(model_config, "tokenizer_mode", None) != "auto":
        return
    architectures = getattr(model_config, "architectures", None) or ()
    if any(arch in QWEN4EXP_ARCHITECTURES for arch in architectures):
        model_config.tokenizer_mode = QWEN4EXP_MODE
