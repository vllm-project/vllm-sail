# SPDX-License-Identifier: Apache-2.0
"""Register the PPU W4A8-INT8 backend with the upstream w4a8_int8 MoE oracle.

The public backend name is ``ppu_deep_gemm_w4a8_int8`` and the enum member is
``BATCHED_PPU_DEEPGEMM``, but the expert GEMM is provided by **acext**
``fusedmoe_wrapper(use_int8_w4a8=True)``: DeepGemm has no INT4-per-channel x
INT8-activation entry point. See ``PPUDeepGemmExpertsW4A8Int8`` for the layout
and weight-application contract.

Only the batched (DeepEP low-latency) format is registered. A Standard-format
request deliberately falls through to upstream, which rejects it, rather than
silently selecting a kernel whose activation format does not match.
"""

from __future__ import annotations

from typing import Any

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle import w4a8_int8 as oracle
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

logger = init_logger(__name__)

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.w4a8_int8"
_AFFECTED = ">=0.33.0,<0.34.0"
_REMOVE_WHEN = (
    "upstream gains a register_moe_backend() extension point mirroring "
    "register_linear_kernel(), which would delete this patch."
)

W4A8Int8MoeBackend = oracle.W4A8Int8MoeBackend
extend_enum(W4A8Int8MoeBackend, "BATCHED_PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM")

_upstream_get_priority_backends = oracle._get_priority_backends
_upstream_backend_to_kernel_cls = oracle.backend_to_kernel_cls
_upstream_map_w4a8_int8_backend = oracle.map_w4a8_int8_backend
_upstream_select_w4a8_int8_moe_backend = oracle.select_w4a8_int8_moe_backend
_upstream_convert_to_w4a8_int8_moe_format = oracle.convert_to_w4a8_int8_moe_format


@patch(
    _MODULE,
    "_get_priority_backends",
    reason=(
        "The W4A8-INT8 oracle only offers a CPU backend, so PPU needs its own "
        "candidate list; without it select_w4a8_int8_moe_backend raises "
        "NotImplementedError before an explicit backend is even considered."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_priority_backends(moe_config: FusedMoEConfig) -> list[W4A8Int8MoeBackend]:
    backends = _upstream_get_priority_backends(moe_config)
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return backends

    backends = [
        backend
        for backend in backends
        if backend is not W4A8Int8MoeBackend.BATCHED_PPU_DEEPGEMM
    ]

    from vllm_sail.model_executor.layers.fused_moe.experts.acext import (
        is_acext_supported,
    )

    requested = ppu_envs.VLLM_SAIL_MOE_BACKEND
    # The GEMM is acext's W4A8 JIT kernel, so acext availability is the gate.
    acext_enabled = is_acext_supported() and (not requested or requested == "acext")
    ppu_backends = []
    if acext_enabled and not moe_config.has_bias:
        ppu_backends.append(W4A8Int8MoeBackend.BATCHED_PPU_DEEPGEMM)
    logger.info(
        "vllm-sail: w4a8_int8 MoE priority backends: %s (acext=%s has_bias=%s "
        "requested=%s)",
        [b.name for b in [*ppu_backends, *backends]],
        acext_enabled,
        moe_config.has_bias,
        requested,
    )
    return [*ppu_backends, *backends]


@patch(
    _MODULE,
    "backend_to_kernel_cls",
    reason="The w4a8_int8 oracle cannot resolve plugin-owned PPU expert classes.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def backend_to_kernel_cls(
    backend: W4A8Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend is W4A8Int8MoeBackend.BATCHED_PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExpertsW4A8Int8,
        )

        return [PPUDeepGemmExpertsW4A8Int8]
    return _upstream_backend_to_kernel_cls(backend)


@patch(
    _MODULE,
    "map_w4a8_int8_backend",
    reason="The public PPU backend string has no upstream W4A8-INT8 member.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def map_w4a8_int8_backend(runner_backend: MoEBackend) -> W4A8Int8MoeBackend:
    if runner_backend == "ppu_deep_gemm_w4a8_int8":
        return W4A8Int8MoeBackend.BATCHED_PPU_DEEPGEMM
    return _upstream_map_w4a8_int8_backend(runner_backend)


@patch(
    _MODULE,
    "select_w4a8_int8_moe_backend",
    reason=(
        "Batched W4A8-INT8 activations require the batched PPU expert class; "
        "upstream only knows the monolithic CPU INT4 kernel."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def select_w4a8_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[W4A8Int8MoeBackend, type[mk.FusedMoEExperts]]:
    is_batched = config.moe_parallel_config.use_batched_activation_format
    if not is_batched or config.moe_backend == "auto":
        return _upstream_select_w4a8_int8_moe_backend(
            config, weight_key, activation_key
        )

    requested = map_w4a8_int8_backend(config.moe_backend)
    if requested is not W4A8Int8MoeBackend.BATCHED_PPU_DEEPGEMM:
        return _upstream_select_w4a8_int8_moe_backend(
            config, weight_key, activation_key
        )

    reason: str | None = None
    for kernel_cls in backend_to_kernel_cls(requested):
        supported, reason = kernel_cls.is_supported_config(
            kernel_cls,
            config,
            weight_key,
            activation_key,
            mk.FusedMoEActivationFormat.BatchedExperts,
        )
        if supported:
            logger.info_once(
                "Using %s W4A8 Int8 MoE backend (acext fusedmoe_wrapper "
                "use_int8_w4a8=True)",
                requested.value,
            )
            return requested, kernel_cls
    raise ValueError(
        f"W4A8 Int8 MoE backend {requested.value} is unsupported: {reason}"
    )


@patch(
    _MODULE,
    "convert_to_w4a8_int8_moe_format",
    reason=(
        "Upstream packs INT4 weights into the KleidiAI nibble layout for CPU "
        "experts; acext consumes the checkpoint's [E, N, K/2] int8 pack with "
        "per-channel scales directly, so PPU must pass the weights through."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream accepts out-of-tree W4A8-INT8 backends in its weight-conversion "
        "hook, or exposes a backend-specific conversion registry."
    ),
)
def convert_to_w4a8_int8_moe_format(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    group_size: int,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    from vllm.platforms import current_platform

    if current_platform.is_ppu():
        # Preserve upstream's 6-tuple contract; the trailing four entries are
        # unused placeholders for the KleidiAI scale/bias outputs.
        empty = torch.empty(0)
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            empty,
            empty,
        )
    return _upstream_convert_to_w4a8_int8_moe_format(
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        group_size,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
    )


@patch(
    _MODULE,
    "make_w4a8_int8_moe_kernel",
    reason=(
        "Upstream rejects non-monolithic experts and constructs them without "
        "the max_num_tokens/num_dispatchers arguments the batched PPU expert "
        "requires."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def make_w4a8_int8_moe_kernel(
    moe_quant_config: Any,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    # Resolve lazily so the PPU-patched factory is used, never a captured alias.
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=experts_cls.is_monolithic(),
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")
    logger.info_once("Using %s", experts_cls.__name__, scope="local")

    extra_args: dict[str, Any] = {}
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        extra_args["max_num_tokens"] = max_num_tokens
        extra_args["num_dispatchers"] = prepare_finalize.num_dispatchers()

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        **extra_args,
    )

    return mk.FusedMoEKernel(prepare_finalize, experts)