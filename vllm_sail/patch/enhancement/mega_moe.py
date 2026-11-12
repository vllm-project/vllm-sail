# SPDX-License-Identifier: Apache-2.0
"""MXFP4 MegaMoE (``--moe-backend deep_gemm_mega_moe``) on PPU 1.5.

The PPU implementation lives in
``vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe`` as a
dedicated ``PPUDeepGemmMegaMoEExperts`` subclass of upstream's
``DeepGemmMegaMoEExperts`` (raw MXFP4 weight banks, ``mega_moe_pre_dispatch``,
the ``fp8xmxfp4`` symmetric workspace, ``fp8_fp4_mega_moe`` keywords — see that
module for the full rationale). This patch module only *redirects* the upstream
symbol to it, so every ``from ... import DeepGemmMegaMoEExperts`` consumer
(kimi_k3, deepseek_v4, minimax_m3 nvidia models) constructs the PPU variant on
PPU while upstream stays untouched:

* ``check_deep_gemm_mega_moe_supported`` (module function) — PPU 1.5
  (capability 8.9) instead of SM100, with the W4A8 kernel's shape rules:
  hidden % 256, intermediate % 128. The upstream ``__init__`` calls it, so both
  classes validate through this one implementation.
* ``DeepGemmMegaMoEExperts`` (module symbol) — rebound to the PPU subclass.

Patch module (like ``patch/enhancement/attention/mla_indexer.py``): vLLM must
be importable when ``install()`` runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_sail.patch.utils import patch

if TYPE_CHECKING:
    import torch

_MODULE = "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe"
AFFECTED_VERSIONS = ">=0.32.0,<0.33.0"

REASON_SUPPORT = (
    "PPU's MegaMoE kernels (fp8_fp4_mega_moe) run on PPU 1.5, report capability "
    "(8, 9) instead of SM100, and require the routed hidden size to be a "
    "multiple of 256 rather than 128; upstream's probe rejects every PPU device."
)
REASON_REDIRECT = (
    "PPU's DeepGEMM MegaMoE re-expresses weight layout (raw MXFP4 banks), "
    "dispatch (mega_moe_pre_dispatch), symmetric workspace (mma_type "
    "'fp8xmxfp4', no shared-expert fusion) and kernel keywords "
    "(situ_beta/situ_linear_beta), which upstream's SM100 DeepGemmMegaMoEExperts "
    "cannot express. The upstream symbol is rebound to the PPU subclass so every "
    "model constructs the PPU variant on PPU while upstream stays untouched."
)
REMOVE_WHEN = (
    "vLLM's DeepGEMM MegaMoE experts dispatch on current_platform to a PPU "
    "backend (raw MXFP4 weight layout, mega_moe_pre_dispatch, fp8_fp4_mega_moe "
    "keywords), or --moe-backend deep_gemm_mega_moe is dropped from PPU."
)

#: Modules that bind ``DeepGemmMegaMoEExperts`` with ``from ... import`` at
#: module scope; rebound when already loaded (see ``install()``).
_CONSUMERS = (
    "vllm.models.kimi_k3.nvidia.model",
    "vllm.models.deepseek_v4.nvidia.model",
    "vllm.models.minimax_m3.nvidia.model",
)

METADATA = (
    (
        f"{_MODULE}.check_deep_gemm_mega_moe_supported",
        REASON_SUPPORT,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_MODULE}.DeepGemmMegaMoEExperts",
        REASON_REDIRECT,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
)

_installed = False


def _on_ppu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_ppu()


def install() -> None:
    """Apply the patches. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as mega

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
        PPU_MEGA_APIS,
        PPUDeepGemmMegaMoEExperts,
        resolve_mma_type,
    )

    base_cls = mega.DeepGemmMegaMoEExperts
    _upstream_check = mega.check_deep_gemm_mega_moe_supported

    @patch(
        _MODULE,
        "check_deep_gemm_mega_moe_supported",
        reason=REASON_SUPPORT,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def check_deep_gemm_mega_moe_supported(
        device: torch.device, hidden_size: int, intermediate_size: int
    ) -> None:
        if not _on_ppu():
            return _upstream_check(device, hidden_size, intermediate_size)

        from vllm_sail.utils.deep_gemm import is_ppu_15

        if not is_ppu_15():
            raise NotImplementedError(
                "PPU DeepGEMM MegaMoE requires PPU 1.5 (compute capability 8.9)."
            )

        # The execution precision (VLLM_SAIL_MEGA_MOE_MMA_TYPE) picks both the
        # shape rules and the DeepGEMM entry points that must exist.
        mma_type = resolve_mma_type()
        if mma_type == "mxf4xmxf4":
            shape_error = (
                hidden_size < 1024
                or hidden_size % 512 != 0
                or intermediate_size % 64 != 0
            )
            shape_hint = (
                "hidden >= 1024, hidden % 512 == 0 and intermediate % 64 == 0"
            )
        else:
            shape_error = hidden_size % 256 != 0 or intermediate_size % 128 != 0
            shape_hint = (
                "hidden % 256 == 0 and intermediate % 128 == 0"
                if mma_type == "fp8xmxfp4"
                else "hidden % 256 == 0 and intermediate % 128 == 0 (SiTU only)"
            )
        if shape_error:
            raise ValueError(
                f"PPU DeepGEMM MegaMoE ({mma_type}) requires {shape_hint}."
            )

        # An older DeepGEMM would otherwise fail with an AttributeError deep
        # inside the first forward, long after the backend was accepted.
        import importlib

        try:
            deep_gemm = importlib.import_module("deep_gemm")
        except ImportError as exc:
            raise NotImplementedError(
                "PPU DeepGEMM MegaMoE requires the deep_gemm package."
            ) from exc
        missing = [
            api for api in PPU_MEGA_APIS[mma_type] if not hasattr(deep_gemm, api)
        ]
        if missing:
            raise NotImplementedError(
                f"The installed DeepGEMM lacks PPU MegaMoE ({mma_type}) entry "
                "points: " + ", ".join(missing)
            )

    # Redirect the upstream symbol to the PPU class. Both __init__ paths call
    # the (patched) module-level support check; construction, EPLB and weight
    # loading plumbing stay inherited from upstream.
    patch(
        _MODULE,
        "DeepGemmMegaMoEExperts",
        reason=REASON_REDIRECT,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )(PPUDeepGemmMegaMoEExperts)

    # Consumers bind the class with ``from ... import`` at module scope;
    # rebind the ones already loaded.
    import sys

    for _consumer_name in _CONSUMERS:
        _consumer = sys.modules.get(_consumer_name)
        if (
            _consumer is not None
            and getattr(_consumer, "DeepGemmMegaMoEExperts", None) is base_cls
        ):
            patch(
                _consumer_name,
                "DeepGemmMegaMoEExperts",
                reason="Preloaded MegaMoE consumers must construct the PPU class.",
                affected_versions=AFFECTED_VERSIONS,
                remove_when="Consumers resolve the class through its provider module.",
            )(PPUDeepGemmMegaMoEExperts)

    _installed = True
