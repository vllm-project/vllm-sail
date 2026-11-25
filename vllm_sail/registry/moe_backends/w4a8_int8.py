# SPDX-License-Identifier: Apache-2.0
"""Register the PPU W4A8-INT8 backend with the upstream w4a8_int8 MoE oracle.

The public backend name is ``ppu_deep_gemm_w4a8_int8`` and the enum member is
``BATCHED_PPU_DEEPGEMM``. The default expert GEMM is DeepGemm's native masked
``m_grouped_gemm_int8_int4_bf16_nt_masked`` (INT4-per-channel x INT8-per-token,
810E / AC_PPU0010 only), wrapped by ``PPUDeepGemmExpertsW4A8Int8``. Activations
dispatch as bf16 and are quantized to INT8 per-token inside the experts; sail's
DeepEP LL prepare accepts ``defer_input_quant`` for raw bf16 dispatch.

The original acext-based shell is retained as ``PPUAcextW4A8Int8Fallback``
(回退用, NOT selected by default): B1 proved the acext path is structurally
blocked by the DeepEP LL defer contract. See the experts classes for the layout
and weight-application contracts.

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
from vllm_sail.compat import fused_moe_use_batched
from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

logger = init_logger(__name__)

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.w4a8_int8"
_AFFECTED = ">=0.34.0,<0.35.0"
_REMOVE_WHEN = (
    "upstream gains a register_moe_backend() extension point mirroring "
    "register_linear_kernel(), which would delete this patch."
)

W4A8Int8MoeBackend = oracle.W4A8Int8MoeBackend
extend_enum(W4A8Int8MoeBackend, "BATCHED_PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM")

# ---------------------------------------------------------------------------
# u258 P2: layout-driven decision policy.
#
# D1 and D2 are SINGLE-POINT flippable constants (coordinator/group-owner
# rulings). Flip ONLY with coordinator approval.
# D1: mixed_precision_w4 + DeepEP low-latency + no explicit backend.
#   "deepgemm" -> keep the PR's validated default (T^-1 conversion path).
#                 FINAL per group-owner ruling 2026-11-25 11:29 ("按默认来"):
#                 do not raise; "mixed defaults to acext" holds only on the
#                 non-LL geometry (legacy path, untouched).
#   "raise"    -> R2 alternative (strict "mixed defaults to acext" reading):
#                 acext cannot do DeepEP LL batched dispatch (B1), so fail
#                 with guidance. Kept flippable; NOT the shipped default.
#   None       -> undecided state: the combination raises a RuntimeError
#                 naming the pending decision (never a silent default).
D1_MIXED_DEEPEP_LL_AUTO: str | None = "deepgemm"
# D2: mixed_precision_w4 + EXPLICIT ppu_deep_gemm_w4a8_int8 + DeepEP LL.
#   True  -> allow (coordinator ruling 2026-11-25, group-owner confirmed:
#            the PR's T^-1 conversion is a compatible path, removing it
#            would be a regression).
#   False -> treat as incompatible and raise (group-owner veto path).
D2_ALLOW_MIXED_EXPLICIT_DEEPGEMM = True

LAYOUT_QUANT_METHOD = {
    "deepgemm_plain": "w4a8_int8",
    "acext_t_order": "mixed_precision_w4",
}

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

    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    requested = ppu_envs.VLLM_SAIL_MOE_BACKEND
    # The default GEMM is DeepGemm's native masked INT4xINT8, which is 810E-only
    # (AC_PPU0010 == (8, 0)). The acext fallback class is retained but not
    # auto-selected here. u258 P2: the env knob only accepts
    # deepgemm|acext|triton (envs.py GEMM_BACKENDS); the former
    # "ppu_deep_gemm_w4a8_int8" comparison was a dead branch (that spelling
    # is CLI-only). Explicit layout-aware requests are validated in
    # _select_by_layout before this priority list is ever consulted.
    native_enabled = (
        is_deep_gemm_supported()
        and current_platform.is_device_capability((8, 0))
        and (not requested or requested == "deepgemm")
    )
    ppu_backends = []
    if native_enabled and not moe_config.has_bias:
        ppu_backends.append(W4A8Int8MoeBackend.BATCHED_PPU_DEEPGEMM)
    logger.info(
        "vllm-sail: w4a8_int8 MoE priority backends: %s (native_deepgemm=%s "
        "has_bias=%s requested=%s)",
        [b.name for b in [*ppu_backends, *backends]],
        native_enabled,
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


def _resolve_requested(config: FusedMoEConfig) -> tuple[str, str | None]:
    """Resolve the user's MoE backend request: CLI > env > auto (u258 P2).

    Returns (source, raw_value). CLI ``config.moe_backend`` wins when it is
    not "auto"; otherwise ``VLLM_SAIL_MOE_BACKEND`` (deepgemm|acext|triton)
    when set; otherwise ("auto", None).
    """
    cli = config.moe_backend
    if cli != "auto":
        return ("cli", cli)
    env = ppu_envs.VLLM_SAIL_MOE_BACKEND
    if env:
        return ("env", env)
    return ("auto", None)


def _canonical_backend(source: str, raw: str | None) -> str | None:
    """Normalize CLI/env spellings to 'deepgemm' | 'acext' | None (auto).

    Unknown/other-format names return the raw string so the caller can raise
    with the exact requested value.
    """
    if raw is None:
        return None
    if raw in ("ppu_deep_gemm_w4a8_int8", "deepgemm"):
        return "deepgemm"
    if raw in ("ppu_acext", "acext"):
        return "acext"
    return raw


def _incompatible_error(
    quant_method: str, layout: str, requested: str, detail: str, suggestion: str
) -> ValueError:
    """Single raise site for every incompatible request (u258 P2).

    CLI and env spellings of the same backend normalize to the same
    ``requested`` token, so both knobs hit this exact message.
    """
    return ValueError(
        f"MoE backend '{requested}' is incompatible with this checkpoint: "
        f"quantization_config.quant_method='{quant_method}' (weight layout "
        f"'{layout}'). {detail} Fix: {suggestion}"
    )


def _native_backend_or_raise(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[W4A8Int8MoeBackend, type[mk.FusedMoEExperts]]:
    requested = W4A8Int8MoeBackend.BATCHED_PPU_DEEPGEMM
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
                "Using %s W4A8 Int8 MoE backend (DeepGemm native masked "
                "m_grouped_gemm_int8_int4_bf16_nt_masked)",
                requested.value,
            )
            return requested, kernel_cls
    raise ValueError(
        f"W4A8 Int8 MoE backend {requested.value} is unsupported: {reason}"
    )


def _select_by_layout(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    weight_layout: str,
) -> tuple[W4A8Int8MoeBackend, type[mk.FusedMoEExperts]]:
    """u258 P2 decision matrix. Priority: CLI > env > quant_method-derived.

    Incompatible requests raise before weight loading (this runs inside
    W4AInt8MoEMethod.__init__ at model construction). No silent fallback.
    """
    quant_method = LAYOUT_QUANT_METHOD[weight_layout]
    is_batched = fused_moe_use_batched(config)
    source, raw = _resolve_requested(config)
    requested = _canonical_backend(source, raw)

    if weight_layout == "deepgemm_plain":
        if requested in (None, "deepgemm"):
            if not is_batched:
                # R3: masked kernel is batched-only.
                raise _incompatible_error(
                    quant_method,
                    weight_layout,
                    "deepgemm",
                    "DeepGemm native masked W4A8-INT8 requires the batched "
                    "activation format (DeepEP low-latency).",
                    "enable --enable-expert-parallel with --all2all-backend "
                    "deepep_low_latency and dp>1, or choose a standard-format "
                    "backend.",
                )
            return _native_backend_or_raise(config, weight_key, activation_key)
        if requested == "acext":
            # R1: same raise for env VLLM_SAIL_MOE_BACKEND=acext and CLI
            # --moe-backend ppu_acext (canonical token is identical).
            raise _incompatible_error(
                quant_method,
                weight_layout,
                "acext",
                "The acext kernel expects the 'mixed_precision_w4' T-order "
                "layout; this checkpoint stores plain-order packed INT4 for "
                "DeepGemm.",
                "unset VLLM_SAIL_MOE_BACKEND / use --moe-backend "
                "ppu_deep_gemm_w4a8_int8 (or leave 'auto' to select DeepGemm), "
                "or load a mixed_precision_w4 checkpoint.",
            )
        raise _incompatible_error(
            quant_method,
            weight_layout,
            str(requested),
            "This backend does not target the W4A8-INT8 deepgemm layout.",
            "use --moe-backend ppu_deep_gemm_w4a8_int8 or leave 'auto'.",
        )

    # weight_layout == "acext_t_order" (mixed_precision_w4). The sail caller
    # only reaches select() on the MK path, which requires batched format.
    assert is_batched, (
        "mixed_precision_w4 select() reached without batched activations; "
        "the legacy acext path must not call the oracle selector"
    )
    if requested == "acext":
        # R2: acext is structurally blocked on DeepEP LL (B1).
        raise _incompatible_error(
            quant_method,
            weight_layout,
            "acext",
            "The acext MoE backend does not support DeepEP low-latency "
            "batched dispatch (defer_input_quant contract).",
            "drop --all2all-backend deepep_low_latency, or load a w4a8_int8 "
            "(deepgemm-layout) checkpoint, or explicitly opt in with "
            "--moe-backend ppu_deep_gemm_w4a8_int8.",
        )
    if requested == "deepgemm":
        if not D2_ALLOW_MIXED_EXPLICIT_DEEPGEMM:
            raise _incompatible_error(
                quant_method,
                weight_layout,
                "deepgemm",
                "Policy D2 forbids DeepGemm on the T-order layout.",
                "use the default acext path without DeepEP low-latency, or "
                "load a w4a8_int8 (deepgemm-layout) checkpoint.",
            )
        return _native_backend_or_raise(config, weight_key, activation_key)
    if requested is None:
        # D1: single-point flippable policy, see constant definition.
        if D1_MIXED_DEEPEP_LL_AUTO is None:
            raise RuntimeError(
                "D1 policy pending group-owner ruling: quant_method="
                "'mixed_precision_w4' + DeepEP low-latency + no explicit "
                "backend has no approved default. Coordinator must set "
                "D1_MIXED_DEEPEP_LL_AUTO to 'raise' or 'deepgemm'."
            )
        if D1_MIXED_DEEPEP_LL_AUTO == "raise":
            raise _incompatible_error(
                quant_method,
                weight_layout,
                "acext",
                "mixed_precision_w4 defaults to the acext MoE backend, which "
                "does not support DeepEP low-latency batched dispatch "
                "(defer_input_quant contract).",
                "drop --all2all-backend deepep_low_latency, or load a "
                "w4a8_int8 (deepgemm-layout) checkpoint, or explicitly opt "
                "in with --moe-backend ppu_deep_gemm_w4a8_int8.",
            )
        # "deepgemm": keep the PR's validated T^-1 conversion default.
        return _native_backend_or_raise(config, weight_key, activation_key)
    raise _incompatible_error(
        quant_method,
        weight_layout,
        str(requested),
        "This backend does not target the W4A8-INT8 T-order layout.",
        "use the default acext path, or explicitly opt in with --moe-backend "
        "ppu_deep_gemm_w4a8_int8 on DeepEP low-latency.",
    )


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
    *,
    weight_layout: str | None = None,
) -> tuple[W4A8Int8MoeBackend, type[mk.FusedMoEExperts]]:
    # u258 P2: sail callers pass the checkpoint layout; the decision matrix
    # (CLI > env > derived, incompatible -> raise) then applies. Upstream CT
    # consumers pass no layout and keep the original PR behavior.
    if weight_layout is not None:
        return _select_by_layout(config, weight_key, activation_key, weight_layout)

    is_batched = fused_moe_use_batched(config)
    if not is_batched or config.moe_backend == "auto":
        return _upstream_select_w4a8_int8_moe_backend(
            config, weight_key, activation_key
        )

    requested = map_w4a8_int8_backend(config.moe_backend)
    if requested is not W4A8Int8MoeBackend.BATCHED_PPU_DEEPGEMM:
        return _upstream_select_w4a8_int8_moe_backend(
            config, weight_key, activation_key
        )

    return _native_backend_or_raise(config, weight_key, activation_key)


@patch(
    _MODULE,
    "convert_to_w4a8_int8_moe_format",
    reason=(
        "Upstream packs INT4 weights into the KleidiAI nibble layout for CPU "
        "experts; the PPU experts consume the checkpoint's [E, N, K/2] int8 "
        "pack with per-channel scales (native: uint8 view + device permutation; "
        "fallback: acext view relabel), so PPU must pass the weights through "
        "and let the experts class transform them."
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
    *,
    weight_layout: str = "acext_t_order",
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

    # u258 P2: single-line decision evidence, emitted once PER WORKER PROCESS
    # (scope="process"; the default "local" scope would print only on the
    # node-local rank 0, see vllm/logger.py _should_log_with_scope). Grep
    # "[vllm_sail] w4a8 MoE decision:" in the tee'd server log; TP8DP2EP16
    # yields one line per each of the 16 worker processes.
    import os

    _source, _raw = _resolve_requested(moe_config)
    _selected = (
        "BATCHED_PPU_DEEPGEMM"
        if experts_cls.__name__ == "PPUDeepGemmExpertsW4A8Int8"
        else experts_cls.__name__
    )
    logger.info_once(
        "[vllm_sail] w4a8 MoE decision: quant_method=%s layout=%s "
        "requested=%s selected=%s experts=%s prepare=%s pid=%s",
        LAYOUT_QUANT_METHOD.get(weight_layout, "unknown"),
        weight_layout,
        _raw if _raw is not None else "auto",
        _selected,
        experts_cls.__name__,
        prepare_finalize.__class__.__name__,
        os.getpid(),
        scope="process",
    )

    extra_args: dict[str, Any] = {}
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        extra_args["max_num_tokens"] = max_num_tokens
        extra_args["num_dispatchers"] = prepare_finalize.num_dispatchers()
        # u258 P1: route the checkpoint layout to the experts class so its
        # weight preprocess matches the byte order (plain skips T^-1).
        extra_args["weight_layout"] = weight_layout

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        **extra_args,
    )

    return mk.FusedMoEKernel(prepare_finalize, experts)
