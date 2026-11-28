# SPDX-License-Identifier: Apache-2.0
"""Keep GLM-5.3-Flash KDA prefill off the NVIDIA-only FlashKDA binary on PPU 1.7.

``vllm.models.glm5next.common.kda._resolve_kda_prefill_backend`` picks FlashKDA
whenever the device reports capability major 9/10/12 with bf16, head_dim 128 and
a bounded gate. PPU shares CUDA identity, and PPU 1.7 (V900) reports capability
(9, 0), so upstream resolves "flashkda" and the layer constructor then runs
``import vllm._flashkda_C`` — an extension PPU does not build — which aborts
model load with ModuleNotFoundError.

PPU 1.0 (8, 0) and 1.5 (8, 9) never enter that branch, so the wrapper delegates
to upstream verbatim there and on every non-PPU platform. On PPU 1.7 the Triton
chunked-prefill kernel (``chunk_kda_with_fused_gate``) is the supported path —
the same kernel the other PPU generations already run — so "auto" resolves to
"triton" and an explicit "flashkda" request raises the same RuntimeError
upstream raises on an unsupported device, instead of crashing on the import.

This replaces passing ``--additional-config '{"kda_prefill_backend":"triton"}'``
on the command line: the platform gate lives in the plugin, so a default launch
works and an explicit FlashKDA request fails with an actionable message.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.glm5next.common.kda"
_META = dict(
    reason=(
        "GLM-5.3-Flash KDA prefill selects the NVIDIA-only FlashKDA extension "
        "on capability major 9, which PPU 1.7 reports; PPU has no "
        "vllm._flashkda_C and must run the Triton chunked-prefill kernel."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when=(
        "Upstream resolves the KDA prefill backend through a platform-aware "
        "kernel registry instead of CUDA capability alone."
    ),
)

logger = init_logger(_MODULE)

# The glm5next model family needs a recent transformers; without it there is no
# resolver to patch, so skip instead of failing the whole plugin install.
try:
    from vllm.models.glm5next.common import kda as _kda
except ImportError as exc:
    _kda = None
    logger.warning(
        "Skipping the glm5next KDA prefill backend patch: the upstream model "
        "cannot be imported (%s).",
        exc,
    )
else:
    _upstream_resolve = _kda._resolve_kda_prefill_backend


def _resolve_kda_prefill_backend(
    backend: str, head_dim: int, dtype: torch.dtype, lower_bound: float | None
) -> str:
    if not (
        current_platform.is_ppu() and current_platform.is_device_capability((9, 0))
    ):
        # PPU 1.0/1.5 and non-PPU: upstream decision, unchanged.
        return _upstream_resolve(backend, head_dim, dtype, lower_bound)

    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    if backend == "flashkda":
        raise RuntimeError(
            "FlashKDA needs the NVIDIA vllm._flashkda_C extension, which PPU "
            "does not build. Use kda_prefill_backend=auto or triton: PPU runs "
            "the Triton chunked-prefill kernel."
        )
    logger.info_once(
        "PPU 1.7 serves the GLM-5.3-Flash KDA prefill with the Triton kernel "
        "(chunk_kda_with_fused_gate); FlashKDA is NVIDIA-only."
    )
    return "triton"


if _kda is not None:
    patch(_MODULE, "_resolve_kda_prefill_backend", **_META)(
        _resolve_kda_prefill_backend
    )