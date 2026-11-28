# SPDX-License-Identifier: Apache-2.0
"""Keep top-k/top-p sampling off FlashInfer's JIT kernels on PPU 1.7.

FlashInfer's PPU port ("sailify") generates a ``.ppu_compat`` header that defines
stubs for ``hggcGetCurrentGraphExec`` / ``hggcTriggerProgrammaticLaunchCompletion``.
SDK 2.3.0 already defines them in ``hgrt/hggc_device_functions.h``, so the sampling
JIT module fails to build with a wall of ``redefinition`` errors and the engine
aborts during the sampler warm-up.

The gate lives in the plugin so a default launch works; setting
``VLLM_USE_FLASHINFER_SAMPLER=0`` stays available as a diagnostic override.
PPU 1.0/1.5 and non-PPU platforms delegate to upstream verbatim.
"""

from __future__ import annotations

import importlib

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_MODULE = "vllm.v1.sample.ops.topk_topp_sampler"
_META = dict(
    reason=(
        "FlashInfer's sailify compat header redefines symbols SDK 2.3.0 already "
        "provides, so its sampling JIT cannot build on PPU 1.7; the native "
        "top-k/top-p sampler is the supported path there."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when=(
        "FlashInfer's PPU compat layer stops redefining symbols the installed "
        "SDK already provides, or upstream probes sampler kernels through a "
        "platform-aware registry."
    ),
)

logger = init_logger(_MODULE)

try:
    from vllm.v1.sample.ops import topk_topp_sampler as _sampler
except ImportError as exc:
    _sampler = None
    logger.warning(
        "Skipping the FlashInfer sampler gate: %s cannot be imported (%s).",
        _MODULE,
        exc,
    )
else:
    _upstream_supported = _sampler.flashinfer_sampler_supported


def flashinfer_sampler_supported() -> bool:
    if current_platform.is_ppu() and current_platform.is_device_capability((9, 0)):
        logger.info_once(
            "PPU 1.7 uses the native top-k/top-p sampler: FlashInfer's sampling "
            "JIT conflicts with the installed SDK headers."
        )
        return False
    return _upstream_supported()


if _sampler is not None:
    patch(_MODULE, "flashinfer_sampler_supported", **_META)(flashinfer_sampler_supported)
    # ``vllm.v1.worker.gpu.sample.sampler`` from-imports the probe at module
    # load, so patching only the defining module leaves that alias bound to the
    # original. Gate the alias too, otherwise the GPU sampler still picks
    # FlashInfer.
    _ALIAS_MODULE = "vllm.v1.worker.gpu.sample.sampler"
    try:
        _alias = importlib.import_module(_ALIAS_MODULE)
    except ImportError as exc:
        logger.warning(
            "Skipping the FlashInfer sampler alias gate: %s cannot be imported (%s).",
            _ALIAS_MODULE,
            exc,
        )
    else:
        if hasattr(_alias, "flashinfer_sampler_supported"):
            patch(_ALIAS_MODULE, "flashinfer_sampler_supported", **_META)(
                flashinfer_sampler_supported
            )