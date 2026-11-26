# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Marlin MoE gate — phase-1 decision D2, option (c).

The fork's Marlin support needs in-place edits to upstream CUDA sources
(``csrc/libtorch_stable/quantization/marlin/*``,
``csrc/libtorch_stable/moe/marlin_moe_wna16/*``) that a plugin cannot make.
Until those changes are upstreamed, PPU stays on non-Marlin MoE paths:
``check_moe_marlin_supports_config`` returns False on PPU unless
``VLLM_PPU_ENABLE_MOE_MARLIN`` is explicitly set (default off; declared in
``vllm_sail/envs.py``).

The legacy switch only controls WNA16 layout gating and cannot supply excluded
Marlin compute kernels. Expert and dense compute gates must independently reject
PPU while delegating non-PPU execution verbatim to upstream. Original layout
checks and portable repacking are preserved so PPU DeepGEMM can load weights.
Shared check/verify helpers also validate mixed-precision quant configs and must
not reject before non-Marlin kernel selection. NVFP4 auto-selection keeps the
full upstream body and only disables the forced Marlin shortcut on PPU.
"""

from __future__ import annotations

import sys

from vllm.model_executor.kernels import linear as linear_kernels
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExpertsBase
from vllm.model_executor.layers.quantization.utils import (
    marlin_utils_fp4,
    marlin_utils_fp8,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_config,
)
from vllm.platforms import current_platform

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

# Captured before patching so the replacement can delegate to upstream.
_upstream_check = check_moe_marlin_supports_config
# The layout check is also used by PPU DeepGEMM weight preparation.
supports_marlin_layout = _upstream_check


@patch(
    "vllm.model_executor.layers.quantization.utils.marlin_utils",
    "check_moe_marlin_supports_config",
    reason=(
        "The fork's Marlin kernels are in-place edits of upstream .cu/.h "
        "sources that a plugin cannot ship; v1 restricts PPU to non-Marlin "
        "MoE paths (phase-1 spec 1.8, decision (c)). Delegates to upstream "
        "unless PPU has the opt-out env var left at its default (off)."
    ),
    remove_when=(
        "the Marlin PPU kernel edits are upstreamed (or rebuilt inside "
        "vllm_sail._moe_C) and VLLM_PPU_ENABLE_MOE_MARLIN defaults on."
    ),
)
def gated_check_moe_marlin_supports_config(*args, **kwargs):
    if current_platform.is_ppu() and not ppu_envs.VLLM_SAIL_ENABLE_MOE_MARLIN:
        return False
    return _upstream_check(*args, **kwargs)


for _consumer in (
    "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16",
):
    _module = sys.modules.get(_consumer)
    if (
        _module is not None
        and getattr(_module, "check_moe_marlin_supports_config", None)
        is _upstream_check
    ):
        patch(
            _consumer,
            "check_moe_marlin_supports_config",
            reason="Preloaded WNA16 consumers must observe the PPU Marlin capability gate.",
            remove_when="WNA16 consumers resolve Marlin capabilities through the provider module.",
        )(gated_check_moe_marlin_supports_config)


_COMPUTE_REASON = (
    "PPU's CUDA-compatible identity passes upstream Marlin checks, "
    "but the generated manifest excludes Marlin compute kernels."
)
_COMPUTE_REMOVE_WHEN = (
    "Marlin compute kernels are compiled in the PPU manifest and verified on device."
)
_NO_PPU_MARLIN = (
    "PPU does not ship Marlin compute kernels; portable weight repacking "
    "does not imply Marlin GEMM availability."
)
_upstream_device_check = MarlinExpertsBase._supports_current_device
_upstream_can_implement = MarlinLinearKernel.can_implement.__func__
_upstream_fp4_check = marlin_utils_fp4.is_fp4_marlin_supported
_upstream_fp8_check = marlin_utils_fp8.is_fp8_marlin_supported


@patch(
    "vllm.model_executor.layers.fused_moe.experts.marlin_moe",
    "MarlinExpertsBase._supports_current_device",
    reason=_COMPUTE_REASON,
    remove_when=_COMPUTE_REMOVE_WHEN,
)
def gated_marlin_experts_device():
    if current_platform.is_ppu():
        return False
    return _upstream_device_check()


@patch(
    "vllm.model_executor.kernels.linear.mixed_precision.marlin",
    "MarlinLinearKernel.can_implement",
    reason=_COMPUTE_REASON,
    remove_when=_COMPUTE_REMOVE_WHEN,
)
def gated_marlin_linear_can_implement(cls, *args, **kwargs):
    if current_platform.is_ppu():
        return False, _NO_PPU_MARLIN
    return _upstream_can_implement(cls, *args, **kwargs)


@patch(
    "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4",
    "is_fp4_marlin_supported",
    reason=_COMPUTE_REASON,
    remove_when=_COMPUTE_REMOVE_WHEN,
)
def gated_fp4_marlin_supported(*args, **kwargs):
    if current_platform.is_ppu():
        return False
    return _upstream_fp4_check(*args, **kwargs)


@patch(
    "vllm.model_executor.layers.quantization.utils.marlin_utils_fp8",
    "is_fp8_marlin_supported",
    reason=_COMPUTE_REASON,
    remove_when=_COMPUTE_REMOVE_WHEN,
)
def gated_fp8_marlin_supported(*args, **kwargs):
    if current_platform.is_ppu():
        return False
    return _upstream_fp8_check(*args, **kwargs)


# Only module-level by-value caches need rebinding; function-local imports pick up the patched provider directly.
for _consumer, _name, _original, _replacement in (
    (
        "vllm.model_executor.kernels.linear.nvfp4.marlin",
        "is_fp4_marlin_supported",
        _upstream_fp4_check,
        gated_fp4_marlin_supported,
    ),
    (
        "vllm.model_executor.kernels.linear.scaled_mm.marlin",
        "is_fp8_marlin_supported",
        _upstream_fp8_check,
        gated_fp8_marlin_supported,
    ),
):
    _module = sys.modules.get(_consumer)
    if _module is not None and getattr(_module, _name, None) is _original:
        patch(
            _consumer,
            _name,
            reason="Preloaded dense Marlin consumers must observe the same PPU compute gate.",
            remove_when="Consumers no longer cache Marlin capability probes by value.",
        )(_replacement)


# Full borrowed upstream NVFP4 selector, bound to the provider's live globals.
def init_nvfp4_linear_kernel(
    use_a16: bool = False, act_dtype: torch.dtype | None = None
) -> NvFp4LinearKernel:
    """Select and instantiate the best NVFP4 linear kernel for the
    current platform.

    Args:
        use_a16: Weight-only (W4A16) instead of W4A4.
        act_dtype: Activation dtype of a W4A16 layer, so that BF16-only
            kernels are skipped for FP16 models. None if unknown.

    Returns:
        The kernel, or a placeholder for requantization sources, which have
        no kernel.

    """
    # TODO: migrate to oracle base method.
    if LinearKernelOracle.maybe_skip_for_requantization_source():
        return FakeNvFp4LinearKernel()

    config = NvFp4LinearLayerConfig(act_dtype=act_dtype)
    a16_kernels = (
        B12xNvFp4W4A16LinearKernel,
        FlashInferCuteDslNvFp4W4A16LinearKernel,
        MarlinNvFp4LinearKernel,
        HummingNvFp4LinearKernel,
    )

    # VLLM_BATCH_INVARIANT forces deterministic execution: Humming for W4A16,
    # otherwise CUTLASS, or emulation where CUTLASS is unsupported. It
    # overrides --linear-backend.
    force_kernel: type[NvFp4LinearKernel] | None = None
    quantization = "nvfp4_w4a16" if use_a16 else "nvfp4_w4a4"
    linear_backend = _get_linear_backend(quantization=quantization)
    if envs.VLLM_BATCH_INVARIANT:
        if use_a16:
            force_kernel, bi_backend = HummingNvFp4LinearKernel, "humming"
        else:
            cutlass_supported, reason = CutlassNvFp4LinearKernel.is_supported()
            if cutlass_supported:
                force_kernel, bi_backend = CutlassNvFp4LinearKernel, "cutlass"
            else:
                logger.info_once(
                    "VLLM_BATCH_INVARIANT is set but the batch-invariant NVFP4 "
                    "kernel is not supported on this platform; falling back to "
                    "emulation for deterministic execution. Reason: %s",
                    reason,
                )
                force_kernel, bi_backend = EmulationNvFp4LinearKernel, "emulation"
        if linear_backend not in ("auto", bi_backend):
            logger.warning_once(
                "VLLM_BATCH_INVARIANT overrides --linear-backend=%s; using the "
                "%s backend for deterministic execution.",
                linear_backend,
                bi_backend,
            )
    elif linear_backend == "auto" and use_a16:
        _cc = current_platform.get_device_capability()
        compute_capability = _cc.to_int() if _cc is not None else None
        # Weight-only: prefer FlashInfer CuTe-DSL W4A16 on SM100/103,
        # Humming then Marlin where Humming is preferred, and Marlin elsewhere.
        cutedsl_ok, _ = FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported(
            compute_capability
        )
        cutedsl_ok = (
            cutedsl_ok
            and FlashInferCuteDslNvFp4W4A16LinearKernel.can_implement(config)[0]
        )
        if compute_capability in (100, 103) and cutedsl_ok:
            force_kernel = FlashInferCuteDslNvFp4W4A16LinearKernel
        elif not prefers_humming(compute_capability):
            # PPU MODIFICATION: begin
            # PPU has no Marlin compute kernels; continue registry selection instead of forcing and failing.
            if not current_platform.is_ppu():
                force_kernel = MarlinNvFp4LinearKernel
            # PPU MODIFICATION: end

    if force_kernel is not None:
        if use_a16 and force_kernel not in a16_kernels:
            raise ValueError(f"{force_kernel.__name__} does not support W4A16")
        is_supported, reason = force_kernel.is_supported()
        if is_supported:
            is_supported, reason = force_kernel.can_implement(config)
        if not is_supported:
            raise ValueError(
                f"Forced NVFP4 kernel {force_kernel.__name__} is not "
                f"supported: {reason}"
            )
        logger.debug_once("Using %s for NVFP4 GEMM", force_kernel.__name__)
        return force_kernel(config)

    # Auto-select from registry (or --linear-backend filtered).
    platform = current_platform._enum
    possible = list(_POSSIBLE_NVFP4_KERNELS.get(platform, []))
    if use_a16:
        possible = [kernel for kernel in possible if kernel in a16_kernels]

    # Apply --linear-backend filtering when set.
    possible = _resolve_backend_kernels(
        possible,
        "NVFP4",
        quantization=quantization,
    )

    kernel_cls, failure_reasons = _first_usable_kernel(
        possible, config, None if use_a16 else kNvfp4Dynamic
    )
    if kernel_cls is not None:
        if kernel_cls is EmulationNvFp4LinearKernel and failure_reasons:
            logger.warning_once(
                "NVFP4 linear falling back to the slow and unoptimized "
                "emulation backend as no optimized backend is available "
                "(unavailable reasons:\n - %s\n). "
                "In case you expect one of these backends to be used, "
                "please verify your environment.",
                "\n - ".join(failure_reasons),
            )

        logger.debug_once("Using %s for NVFP4 GEMM", kernel_cls.__name__)
        return kernel_cls(config)

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "NVFP4 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


_upstream_init_nvfp4 = linear_kernels.init_nvfp4_linear_kernel
_ppu_init_nvfp4 = bind_body(init_nvfp4_linear_kernel, linear_kernels)


@patch(
    "vllm.model_executor.kernels.linear",
    "init_nvfp4_linear_kernel",
    reason="PPU NVFP4 auto-selection must not force unavailable Marlin and bypass registry fallback.",
    remove_when="Upstream's auto-force path checks compute kernel support before selecting available backends.",
)
def gated_init_nvfp4_linear_kernel(*args, **kwargs):
    if current_platform.is_ppu():
        return _ppu_init_nvfp4(*args, **kwargs)
    return _upstream_init_nvfp4(*args, **kwargs)


for _consumer in (
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4",
    "vllm.model_executor.layers.quantization.modelopt",
    "vllm.model_executor.layers.quantization.online.nvfp4",
    "vllm.model_executor.layers.quantization.quark.schemes.quark_nvfp4",
):
    _module = sys.modules.get(_consumer)
    if (
        _module is not None
        and getattr(_module, "init_nvfp4_linear_kernel", None) is _upstream_init_nvfp4
    ):
        patch(
            _consumer,
            "init_nvfp4_linear_kernel",
            reason="Preloaded NVFP4 consumers must observe the same PPU auto-selection policy.",
            remove_when="Consumers no longer cache the NVFP4 selector by value.",
        )(gated_init_nvfp4_linear_kernel)
