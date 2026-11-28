# SPDX-License-Identifier: Apache-2.0
"""Disable the FlashInfer allreduce+norm fusion workspace on PPU 1.7.

Auto backend selection picks ``mnnvl`` with a ``trtllm`` fallback on a single
node. On PPU 1.7 the trtllm workspace JIT fails to build (the nvfp4 conversion
intrinsics ``__nv_cvt_float2_to_fp4x2`` are not available for the PPU host
compile), which logs an ERROR and leaves the fusion pass in an "ERROR + silent
fallback" state.

Returning ``(None, False)`` is upstream's own "no supported backend" answer, so
the workspace getters return early without logging an error and the fusion pass
deterministically uses the supported allreduce implementation. PPU 1.0/1.5 and
non-PPU platforms delegate to upstream verbatim.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_MODULE = "vllm.distributed.device_communicators.flashinfer_all_reduce"
_META = dict(
    reason=(
        "The trtllm allreduce+norm fusion workspace JIT needs nvfp4 conversion "
        "intrinsics PPU does not provide; leaving it enabled produces an ERROR "
        "plus an implicit fallback on PPU 1.7."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when=(
        "FlashInfer's PPU port builds the trtllm fusion workspace, or upstream "
        "resolves the fusion backend through a platform-aware capability probe."
    ),
)

logger = init_logger(_MODULE)

try:
    from vllm.distributed.device_communicators import flashinfer_all_reduce as _fi_ar
except ImportError as exc:
    _fi_ar = None
    logger.warning(
        "Skipping the FlashInfer allreduce fusion gate: %s cannot be imported (%s).",
        _MODULE,
        exc,
    )
else:
    _upstream_resolve = _fi_ar._resolve_fi_ar_backend


def _resolve_fi_ar_backend(group) -> tuple[str | None, bool]:
    if current_platform.is_ppu() and current_platform.is_device_capability((9, 0)):
        logger.info_once(
            "PPU 1.7 disables the FlashInfer allreduce+norm fusion: its trtllm "
            "workspace JIT requires nvfp4 intrinsics PPU does not provide."
        )
        return None, False
    return _upstream_resolve(group)


if _fi_ar is not None:
    patch(_MODULE, "_resolve_fi_ar_backend", **_META)(_resolve_fi_ar_backend)