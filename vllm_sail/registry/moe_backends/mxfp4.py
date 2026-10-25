# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Register PPU backends with the MXFP4 MoE oracle."""

from __future__ import annotations

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "upstream gains a register_moe_backend() extension point mirroring "
    "register_linear_kernel(), which would delete this patch."
)

Mxfp4MoeBackend = oracle.Mxfp4MoeBackend
extend_enum(Mxfp4MoeBackend, "PPU_DEEPGEMM_MXFP4", "PPU_DEEPGEMM_MXFP4")
extend_enum(
    Mxfp4MoeBackend,
    "BATCHED_PPU_DEEPGEMM_MXFP4",
    "BATCHED_PPU_DEEPGEMM_MXFP4",
)

for _name in (
    "PPU_DEEPGEMM_MXFP4_BF16",
    "BATCHED_PPU_DEEPGEMM_MXFP4_BF16",
    "PPU_DEEPGEMM_MXFP4_BF16_MMA",
    "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
):
    extend_enum(Mxfp4MoeBackend, _name, _name)
_PPU_CANDIDATES = tuple(
    getattr(Mxfp4MoeBackend, name)
    for name in (
        "PPU_DEEPGEMM_MXFP4",
        "BATCHED_PPU_DEEPGEMM_MXFP4",
        "PPU_DEEPGEMM_MXFP4_BF16_MMA",
        "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
        "PPU_DEEPGEMM_MXFP4_BF16",
        "BATCHED_PPU_DEEPGEMM_MXFP4_BF16",
    )
)

_upstream_backend_to_kernel_cls = oracle.backend_to_kernel_cls
_upstream_map_mxfp4_backend = oracle.map_mxfp4_backend
_upstream_get_priority_backends_for_gpt_oss = oracle._get_priority_backends_for_gpt_oss
_upstream_get_priority_backends = oracle._get_priority_backends
_upstream_backend_activation_key = oracle._backend_activation_key
_upstream_select_mxfp4_moe_backend = oracle.select_mxfp4_moe_backend


@patch(
    _MODULE,
    "backend_to_kernel_cls",
    reason="The MXFP4 oracle cannot resolve plugin-owned PPU expert classes.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExpertsMXFP4,
        )

        return [PPUDeepGemmExpertsMXFP4]
    if backend == Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4:
        from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            PPUBatchedDeepGemmExpertsMXFP4,
        )

        return [PPUBatchedDeepGemmExpertsMXFP4]
    if backend in _PPU_CANDIDATES[2:]:
        if backend.name.startswith("BATCHED_"):
            from vllm_sail.model_executor.layers.fused_moe.experts import (
                batched_deep_gemm_moe as experts,
            )

            name = "PPUBatchedDeepGemmExperts"
        else:
            from vllm_sail.model_executor.layers.fused_moe.experts import (
                deep_gemm_moe as experts,
            )

            name = "PPUDeepGemmExperts"
        if backend.name.endswith("_MMA"):
            name += "W4FA16MMA"
        return [getattr(experts, name)]
    return _upstream_backend_to_kernel_cls(backend)


@patch(
    _MODULE,
    "map_mxfp4_backend",
    reason="The public ppu_deep_gemm string has no upstream MXFP4 member mapping.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def map_mxfp4_backend(runner_backend: MoEBackend) -> list[Mxfp4MoeBackend]:
    if runner_backend == "ppu_deep_gemm":
        return [Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4]
    if runner_backend == "ppu_deep_gemm_w4a16":
        return [
            Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_BF16_MMA,
            Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_BF16,
        ]
    return _upstream_map_mxfp4_backend(runner_backend)


@patch(
    _MODULE,
    "_get_priority_backends_for_gpt_oss",
    reason="The GPT-OSS MXFP4 candidate list has no PPU DeepGEMM backends.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_priority_backends_for_gpt_oss() -> list[Mxfp4MoeBackend]:
    backends = _upstream_get_priority_backends_for_gpt_oss()
    from vllm.platforms import current_platform

    if current_platform.is_ppu() and Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4 not in backends:
        backends[0:0] = list(_PPU_CANDIDATES)
    return backends


@patch(
    _MODULE,
    "_get_priority_backends",
    reason="The generic MXFP4 candidate list has no PPU DeepGEMM backends.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_priority_backends() -> list[Mxfp4MoeBackend]:
    backends = _upstream_get_priority_backends()
    from vllm.platforms import current_platform

    if current_platform.is_ppu() and Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4 not in backends:
        insertion = backends.index(Mxfp4MoeBackend.DEEPGEMM_MXFP4) + 1
        backends[insertion:insertion] = list(_PPU_CANDIDATES)
    return backends


@patch(
    _MODULE,
    "_backend_activation_key",
    reason="PPU MXFP4 W4A4 backends need the kMxfp4Dynamic activation key.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _backend_activation_key(backend: Mxfp4MoeBackend) -> QuantKey | None:
    if backend in (
        Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4,
        Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4,
    ):
        return oracle.kMxfp4Dynamic
    return _upstream_backend_activation_key(backend)


def select_mxfp4_moe_backend(
    config: FusedMoEConfig,
    activation_key: QuantKey | None = None,
) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts] | None]:
    """Select the primary MXFP4 MoE backend.

    Args:
        config: MoE configuration
        activation_key: Optional activation quantization key declared by
            the checkpoint. Backends that match it are preferred, with BF16
            activation backends as a fallback.
            Use kFp8StaticTensorSym for W4A8 scheme.

    Note: Shape-specific fallbacks may still occur at runtime.

    """
    runner_backend = config.moe_backend
    requested_activation_key = _resolve_activation_key(activation_key)

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    if runner_backend != "auto":
        requested_backends = _get_requested_backends(
            runner_backend, requested_activation_key
        )
        if activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
            # PPU MODIFICATION: begin
            _batched_backend_map = {
                Mxfp4MoeBackend.MARLIN: Mxfp4MoeBackend.BATCHED_MARLIN,
                Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4: (
                    Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4
                ),
                Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_BF16: (
                    Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4_BF16
                ),
                Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_BF16_MMA: (
                    Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA
                ),
            }
            # PPU MODIFICATION: end
            requested_backends = [
                # PPU MODIFICATION: begin
                _batched_backend_map.get(b, b)
                for b in requested_backends
                # PPU MODIFICATION: end
            ]
        if not requested_backends:
            raise ValueError(
                f"moe_backend={runner_backend!r} does not support "
                f"activation={requested_activation_key}"
            )
        last_error: Exception | None = None
        for requested_backend in requested_backends:
            act_key = (
                requested_activation_key
                if requested_backend == Mxfp4MoeBackend.EMULATION
                else _backend_activation_key(requested_backend)
            )
            try:
                selected = _return_or_raise(
                    requested_backend,
                    config,
                    kMxfp4Static,
                    act_key,
                    activation_format,
                )
            except ValueError as e:
                last_error = e
            else:
                _warn_on_bf16_fallback(requested_backend, requested_activation_key)
                return selected
        assert last_error is not None
        raise last_error

    if _requires_qwen38_tep8_emulation(config, requested_activation_key):
        backend = Mxfp4MoeBackend.EMULATION
        logger.warning_once(
            "Using OCP MX emulation for the Qwen3.8 Flash Next TEP8 routed "
            "experts on gfx950 because the native AITER W4A4 kernel is not "
            "numerically reliable for this shape. Performance will be lower."
        )
        return _return_or_raise(
            backend,
            config,
            kMxfp4Static,
            requested_activation_key,
            activation_format,
        )

    # Select kernels in order of backend.
    AVAILABLE_BACKENDS = _filter_by_activation(
        _get_priority_backends_for_gpt_oss(), requested_activation_key
    )

    unsupported_reasons = []
    for backend in AVAILABLE_BACKENDS:
        # EMULATION takes the requested key, other backends use their own
        act_key = (
            requested_activation_key
            if backend == Mxfp4MoeBackend.EMULATION
            else _backend_activation_key(backend)
        )
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, kMxfp4Static, act_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                _warn_on_bf16_fallback(backend, requested_activation_key)
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason))
                unsupported_reasons.append((backend, reason))

    if current_platform.is_xpu():
        backend = Mxfp4MoeBackend.XPU
        logger.info_once(_make_log_backend(backend))
        return _return_or_raise(
            Mxfp4MoeBackend.XPU,
            config,
            kMxfp4Static,
            None,
            activation_format,
        )

    unsupported_log = "; ".join(
        [
            f"backend: {backend.value}, reason: {reason}"
            for backend, reason in unsupported_reasons
        ]
    )
    raise NotImplementedError(
        "No MXFP4 MoE backend supports the deployment configuration. "
        f"weight_key=kMxfp4Static, activation_key={activation_key}. "
        f"Candidate backends were: "
        f"{[backend.value for backend in AVAILABLE_BACKENDS]}. "
        f"Unsupported reasons: {unsupported_log}. "
    )


select_mxfp4_moe_backend = patch(
    _MODULE,
    "select_mxfp4_moe_backend",
    reason="Resolve explicit PPU W4A4 and W4A16 standard/batched candidates with activation filtering.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)(bind_body(select_mxfp4_moe_backend, oracle))


_upstream_select_deepseek_v4 = oracle.select_deepseek_v4_mxfp4_moe_backend


@patch(
    _MODULE,
    "select_deepseek_v4_mxfp4_moe_backend",
    reason="Preserve PPU W4A4/W4A16 selection through the 0.31 MXFP4 kernel factory.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def select_deepseek_v4_mxfp4_moe_backend(config):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_select_deepseek_v4(config)
    activation_key = None
    if not current_platform.is_device_capability((8, 0)) and config.moe_backend not in (
        "marlin",
        "ppu_deep_gemm_w4a16",
    ):
        activation_key = oracle.kMxfp4Dynamic
    return oracle.select_mxfp4_moe_backend(config, activation_key=activation_key)
