# SPDX-License-Identifier: Apache-2.0
"""Keep the MoE router GEMM off the NVIDIA-only CuTe DSL kernels on PPU 1.7.

``vllm.model_executor.layers.fused_moe.router.gate_linear`` enables the
low-latency CuTe DSL router GEMMs (``ll_bf16_gemm`` / ``ll_fp32w_gemm``) when the
device is SM90+ and the CUTLASS Python DSL package imports. PPU shares CUDA
identity and PPU 1.7 (V900) reports capability (9, 0), while the image ships
``nvidia_cutlass_dsl``, so both probes return True and the router calls into
``cuda_dialect_init_library_once``, which fails on PPU with
``cudaErrorCallRequiresNewerDriver``.

PPU 1.0 (8, 0) and 1.5 (8, 9) never satisfy the SM90+ precondition, so the
wrappers delegate to upstream verbatim there and on every non-PPU platform. On
PPU 1.7 the router falls through to the next tier it already supports.
"""

from __future__ import annotations

import importlib

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_META = dict(
    reason=(
        "The CuTe DSL low-latency router GEMMs are gated on SM90+ plus an "
        "importable CUTLASS DSL, both of which hold on PPU 1.7; the kernels "
        "then call NVIDIA driver APIs PPU does not implement."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when=(
        "Upstream probes the router GEMM kernels through a platform-aware "
        "registry instead of CUDA capability plus a package import."
    ),
)

logger = init_logger(__name__)

_TARGETS = (
    ("vllm.model_executor.kernels.linear.cute_dsl.ll_bf16", "ll_bf16_gemm"),
    ("vllm.model_executor.kernels.linear.cute_dsl.ll_fp32w", "ll_fp32w_gemm"),
)


def _make_wrapper(module_name: str, kernel_name: str, upstream):
    def is_available() -> bool:
        if current_platform.is_ppu() and current_platform.is_device_capability((9, 0)):
            logger.info_once(
                "PPU 1.7 disables the CuTe DSL %s router GEMM (NVIDIA-only); "
                "the router uses its next supported tier.",
                kernel_name,
            )
            return False
        return upstream()

    return is_available


for _module_name, _kernel_name in _TARGETS:
    try:
        _mod = importlib.import_module(_module_name)
    except ImportError as exc:
        logger.warning(
            "Skipping the CuTe DSL %s availability patch: %s cannot be imported (%s).",
            _kernel_name,
            _module_name,
            exc,
        )
        continue
    patch(_module_name, "is_available", **_META)(
        _make_wrapper(_module_name, _kernel_name, _mod.is_available)
    )