# SPDX-License-Identifier: Apache-2.0
"""SAIL runtime environment variables — the single source of truth.

Rules (enforced in review):

1. Every SAIL-specific runtime environment variable is declared here. Do **not**
   read ``VLLM_SAIL_*`` or legacy ``VLLM_PPU_*`` runtime names elsewhere.
2. Access is lazy, via module ``__getattr__``, so a variable changed after
   import is still observed. Use :func:`is_set` to distinguish "explicitly set
   by the user" from "left at the default" — several PPU backend-selection paths
   depend on that difference.
3. ``VLLM_SAIL_*`` names take precedence over their ``VLLM_PPU_*`` aliases,
   including explicit false or empty values. Legacy Python attributes use the
   same resolution. Defaults are unchanged from the in-tree PPU fork.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Declarations for static analysis and IDE completion only.
#
# These MUST stay inside `if TYPE_CHECKING`. An annotated assignment at module
# scope creates a real module attribute, and a real attribute shadows the module
# `__getattr__` below — so the lazy getters would never run and every variable
# would read back as its literal default regardless of the environment. (This is
# also why vllm's own envs.py guards its declarations the same way.)
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    PPU_SDK: str = "/usr/local/PPU_SDK"
    VLLM_SAIL_STRICT_COMPAT: bool = False
    VLLM_SAIL_FUSED_RMSNORM_QUANT: bool = False
    VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT: bool = False
    VLLM_SAIL_DENSE_BF16_DEEPGEMM: bool = False
    VLLM_SAIL_USE_PLA: bool = True
    VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED: bool = False
    VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT: bool = False
    VLLM_SAIL_USE_DEEPSELECT: bool = True
    VLLM_SAIL_MOE_BACKEND: str | None = None
    VLLM_SAIL_DENSE_BACKEND: str | None = None
    VLLM_SAIL_FUSED_GDN_DECODE: bool = True
    VLLM_SAIL_DISABLE_MOE_WNA16_CUDA: bool = False
    VLLM_SAIL_FORCE_MOE_WNA16_CUDA: bool = False
    VLLM_SAIL_ENABLE_MOE_MARLIN: bool = False
    VLLM_SAIL_KIMI_K3_MOE_SEQUENCE_PARALLEL: bool = True
    VLLM_SAIL_MEGA_MOE_MMA_TYPE: str = "fp8xfp4"
    VLLM_SAIL_DISABLE_MEGA_MHC: bool = False
    VLLM_SAIL_USE_TRITON_INT8_QUANT: bool = True
    VLLM_SAIL_FP8_LOAD_POLICY: str | None = None
    VLLM_SAIL_ENABLE_FP8_TO_INT8: bool = False
    VLLM_SAIL_KPOOL_DECODE_LOGITS: str = "non_paged"
    VLLM_SAIL_KPOOL_MAX_DENSE_POOLS: int = 1 << 21
    VLLM_SAIL_QSA_KERNEL: str = "auto"
    VLLM_SAIL_STEP5_DSA_SCORES: bool = False
    VLLM_SAIL_STEP5_DSA_SELECTOR_PPU: bool = False
    VLLM_SAIL_STEP5_DSA_PREFILL_PPU: bool = False
    VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND: str = "flashinfer"
    VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION: bool = False
    VLLM_STEP5_INDEXER_FP8_AS_FP16: bool = False
    VLLM_SAIL_MINIMAX_M3_MSA: bool = True
    VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_TRITON: bool = False
    VLLM_SAIL_MINIMAX_M3_MSA_ATTEND_TRITON: bool = False
    VLLM_SAIL_NVTX_PROFILE: bool = False
    VLLM_SAIL_NVTX_DUMP_DICT: bool = False
    VLLM_SAIL_NVTX_DUMP_TOPK: bool = False
    VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN: bool = False
    VLLM_SAIL_FUSED_NORM_ROPE_ATTN: bool = False
    VLLM_SAIL_PPU_KV_MXFP8: bool = False
    VLLM_SAIL_PPU_KV_NVFP4: bool = False
    VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE: bool = False
    VLLM_DEEPEPLL_RECV_HOOK: bool = True
    VLLM_GDN_DECODE_KERNEL: str = "auto"

    # Backward-compatible Python attributes; all reads resolve SAIL first.
    VLLM_PPU_STRICT_COMPAT: bool = False
    VLLM_PPU_FUSED_RMSNORM_QUANT: bool = False
    VLLM_PPU_USE_OPT_TOKEN_GROUP_QUANT: bool = False
    VLLM_PPU_DENSE_BF16_DEEPGEMM: bool = False
    VLLM_PPU_USE_PLA: bool = True
    VLLM_PPU_DEEPGEMM_MOE_TP_FUSED: bool = False
    VLLM_PPU_DEEPGEMM_MOE_FUSED_ACT_QUANT: bool = False
    VLLM_PPU_USE_DEEPSELECT: bool = True
    VLLM_PPU_MOE_BACKEND: str | None = None
    VLLM_PPU_DENSE_BACKEND: str | None = None
    VLLM_PPU_FUSED_GDN_DECODE: bool = True
    VLLM_PPU_DISABLE_MOE_WNA16_CUDA: bool = False
    VLLM_PPU_FORCE_MOE_WNA16_CUDA: bool = False
    VLLM_PPU_ENABLE_MOE_MARLIN: bool = False
    VLLM_PPU_USE_TRITON_INT8_QUANT: bool = True
    VLLM_PPU_FP8_LOAD_POLICY: str | None = None
    VLLM_PPU_ENABLE_FP8_TO_INT8: bool = False
    VLLM_PPU_STEP5_DSA_SCORES: bool = False
    VLLM_PPU_STEP5_DSA_SELECTOR_PPU: bool = False
    VLLM_PPU_STEP5_DSA_PREFILL_PPU: bool = False
    VLLM_PPU_STEP5_DSA_PREFILL_BACKEND: str = "flashinfer"
    VLLM_PPU_STEP5_DSA_PREFILL_BATCH_ATTENTION: bool = False
    VLLM_PPU_NVTX_PROFILE: bool = False
    VLLM_PPU_NVTX_DUMP_DICT: bool = False
    VLLM_PPU_NVTX_DUMP_TOPK: bool = False
    VLLM_PPU_NVTX_VFA_DUMP_SEQLEN: bool = False
    VLLM_PPU_FUSED_NORM_ROPE_ATTN: bool = False
    VLLM_PPU_PPU_KV_MXFP8: bool = False
    VLLM_PPU_PPU_KV_NVFP4: bool = False
    VLLM_PPU_FUSED_ATTN_PER_TOKEN_SCALE: bool = False

_TRUTHY = ("true", "1", "yes", "on")

#: Backend names accepted by both MoE and dense GEMM selection.
GEMM_BACKENDS = ("deepgemm", "acext", "triton")
GDN_DECODE_KERNELS = ("auto", "cuda", "pla", "triton")
KPOOL_DECODE_LOGITS_BACKENDS = ("non_paged",)
#: Qwen4Exp QSA sparse-attention kernels. ``cuda`` is FA3's QSA path, which only
#: exists on PPU builds carrying the ``qsa_allow_aiu`` argument.
QSA_KERNELS = ("auto", "cuda", "triton")
MEGA_MOE_MMA_TYPES = ("fp4xfp4", "fp8xfp4", "bf16xfp4")
STEP5_PREFILL_BACKENDS = ("flashinfer", "native")


def _env_names(name: str) -> tuple[str, ...]:
    """Canonical name followed by its aliases, in precedence order."""
    name = _canonical_names.get(name, name)
    return (name, *_aliases.get(name, ()))


def _read_env(name: str) -> tuple[str, str | None]:
    for candidate in _env_names(name):
        raw = os.getenv(candidate)
        if raw is not None:
            return candidate, raw
    return name, None


def _bool(name: str, default: bool) -> Callable[[], bool]:
    """Read a boolean env var, falling back to aliases only when unset."""

    def _get() -> bool:
        _, raw = _read_env(name)
        if raw is None:
            return default
        return raw.strip().lower() in _TRUTHY

    return _get


def _int(name: str, default: int) -> Callable[[], int]:
    """Read an integer env var, falling back to aliases only when unset."""

    def _get() -> int:
        source, raw = _read_env(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw.strip())
        except ValueError:
            raise ValueError(
                f"Invalid value {raw!r} for {source}. Expected an integer."
            ) from None

    return _get


def _choice(
    name: str, default: str | None, choices: tuple[str, ...]
) -> Callable[[], str | None]:
    """Read a string env var constrained to ``choices`` (case-insensitive)."""

    def _get() -> str | None:
        source, raw = _read_env(name)
        if raw is None or raw.strip() == "":
            return default
        value = raw.strip().lower()
        if value not in choices:
            raise ValueError(
                f"Invalid value {raw!r} for {source}. Expected one of "
                f"{', '.join(choices)}."
            )
        return value

    return _get


def _positive_int(name: str, default: int) -> Callable[[], int]:
    """Read a positive integer env var."""

    def _get() -> int:
        source, raw = _read_env(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Expected an integer for {source}, got {raw!r}.") from exc
        if value <= 0:
            raise ValueError(f"{source} must be positive, got {value}.")
        return value

    return _get


@dataclass(frozen=True)
class EnvSpec:
    """Declarative metadata for one runtime environment variable."""

    name: str
    section: str
    kind: str
    default: Any
    description: str
    getter: Callable[[], Any]
    choices: tuple[str, ...] = ()
    extra_aliases: tuple[str, ...] = ()
    default_display: str | None = None

    @property
    def aliases(self) -> tuple[str, ...]:
        """Compatibility aliases in precedence order after the canonical name."""
        derived = (
            (self.name.replace("VLLM_SAIL_", "VLLM_PPU_", 1),)
            if self.name.startswith("VLLM_SAIL_")
            else ()
        )
        return (*derived, *self.extra_aliases)


def _str_spec(
    name: str,
    *,
    section: str,
    default: str,
    description: str,
    extra_aliases: tuple[str, ...] = (),
) -> EnvSpec:
    return EnvSpec(
        name=name,
        section=section,
        kind="str",
        default=default,
        description=description,
        getter=lambda: os.environ.get(name, default),
        extra_aliases=extra_aliases,
    )


def _bool_spec(
    name: str,
    *,
    section: str,
    default: bool,
    description: str,
    extra_aliases: tuple[str, ...] = (),
) -> EnvSpec:
    return EnvSpec(
        name=name,
        section=section,
        kind="bool",
        default=default,
        description=description,
        getter=_bool(name, default),
        extra_aliases=extra_aliases,
    )


def _choice_spec(
    name: str,
    *,
    section: str,
    default: str | None,
    choices: tuple[str, ...],
    description: str,
    extra_aliases: tuple[str, ...] = (),
) -> EnvSpec:
    return EnvSpec(
        name=name,
        section=section,
        kind="choice",
        default=default,
        description=description,
        getter=_choice(name, default, choices),
        choices=choices,
        extra_aliases=extra_aliases,
    )


def _positive_int_spec(
    name: str,
    *,
    section: str,
    default: int,
    description: str,
    default_display: str | None = None,
    extra_aliases: tuple[str, ...] = (),
) -> EnvSpec:
    return EnvSpec(
        name=name,
        section=section,
        kind="positive_int",
        default=default,
        description=description,
        getter=_positive_int(name, default),
        default_display=default_display,
        extra_aliases=extra_aliases,
    )


ENVIRONMENT_SPECS: tuple[EnvSpec, ...] = (
    # -- Runtime environment -----------------------------------------------
    _str_spec(
        "PPU_SDK",
        section="Runtime environment",
        default="/usr/local/PPU_SDK",
        description="SAIL SDK root used at runtime.",
    ),
    _bool_spec(
        "VLLM_SAIL_STRICT_COMPAT",
        section="Runtime environment",
        default=False,
        description=(
            "Fail closed when the installed vLLM build does not match the "
            "verified upstream ref in the packaged kernel manifest (default: warn)."
        ),
    ),
    # -- Backend selection -------------------------------------------------
    # PPU MoE group-GEMM backend. Unset means "pick the best available",
    # which is why `is_set()` is meaningful here.
    _choice_spec(
        "VLLM_SAIL_MOE_BACKEND",
        section="Backend selection",
        default=None,
        choices=GEMM_BACKENDS,
        description="Select the PPU MoE group-GEMM backend instead of automatic selection.",
    ),
    # PPU dense-GEMM backend for quantized linear layers.
    _choice_spec(
        "VLLM_SAIL_DENSE_BACKEND",
        section="Backend selection",
        default=None,
        choices=GEMM_BACKENDS,
        description="Select the PPU dense-GEMM backend for quantized linear layers.",
    ),
    _bool_spec(
        "VLLM_SAIL_DENSE_BF16_DEEPGEMM",
        section="Backend selection",
        default=False,
        description="Opt into the Zhenwu M890 BF16 dense DeepGEMM path.",
    ),
    _bool_spec(
        "VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED",
        section="Backend selection",
        default=False,
        description="Enable the supported fused tensor-parallel DeepGEMM MoE path.",
    ),
    # Fuse the second GEMM's input quantization into DeepGEMM's activation
    # kernel (`silu_and_mul` / `situ_and_mul`) instead of running the activation
    # and a separate per-token/per-block quantize pass. The BF16 (w4a16 or
    # unquantized) experts use the same kernel's values-only epilogue. The same
    # epilogue also replaces the standalone `vllm::silu_and_mul` /
    # `vllm::situ_and_mul` that unquantized BF16 dense MLPs and shared experts
    # run on, through SiluAndMul / SituAndMul.
    _bool_spec(
        "VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT",
        section="Backend selection",
        default=False,
        description=(
            "Fuse activation and the second MoE GEMM's input quantization into "
            "DeepGEMM's SiLU/SiTU kernel for FP8 and MXFP4 experts; BF16 "
            "experts, unquantized dense MLPs and shared experts use its "
            "values-only epilogue."
        ),
    ),
    _bool_spec(
        "VLLM_SAIL_USE_PLA",
        section="Backend selection",
        default=True,
        description="Allow PLA linear-attention paths where available.",
    ),
    _bool_spec(
        "VLLM_SAIL_USE_DEEPSELECT",
        section="Backend selection",
        default=True,
        description=(
            "Allow the SAIL DeepSelect top-k backend for sparse indexers on "
            "Zhenwu M890."
        ),
    ),
    # MoE-local sequence sharding for eligible Kimi K3 MegaMoE TP/PP
    # configurations by default; set to 0 to restore the replicated-token
    # path. The decoder and pipeline boundaries still operate on the full
    # sequence.
    _bool_spec(
        "VLLM_SAIL_KIMI_K3_MOE_SEQUENCE_PARALLEL",
        section="Backend selection",
        default=True,
        description=(
            "Shard MoE input tokens across TP ranks for Kimi K3 MegaMoE "
            "PP>1 configurations instead of replicating the full sequence "
            "on every rank; `0` restores the replicated-token path."
        ),
    ),
    # Shared upstream name; PPU extends the choices through its general plugin.
    _choice_spec(
        "VLLM_GDN_DECODE_KERNEL",
        section="Backend selection",
        default="auto",
        choices=GDN_DECODE_KERNELS,
        description=(
            "Select the GDN decode kernel independently of prefill and KDA "
            "(shared upstream name)."
        ),
    ),
    # Qwen4Exp QSA sparse-attention kernel. ``auto`` (default) takes FA3's QSA
    # path when the shape gate, the platform gate and the wheel probe all pass,
    # and falls back to the upstream Triton kernel otherwise; ``cuda`` and
    # ``triton`` pin either side for A/B measurement. The fork hard-wires
    # ``Qwen4ExpQSAImpl`` on the layer, so this is the only selection point.
    _choice_spec(
        "VLLM_SAIL_QSA_KERNEL",
        section="Backend selection",
        default="auto",
        choices=QSA_KERNELS,
        description=(
            "Select the Qwen4Exp QSA sparse-attention kernel (`cuda` selects "
            "FA3's QSA path when supported)."
        ),
    ),
    _bool_spec(
        "VLLM_SAIL_FUSED_RMSNORM_QUANT",
        section="Backend selection",
        default=False,
        description="Enable supported fused RMSNorm and quantization paths.",
    ),
    _bool_spec(
        "VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT",
        section="Backend selection",
        default=False,
        description="Enable the optimized token-group quantization path.",
    ),
    # Triton implementation of dynamic per-token int8 quant; faster than the
    # compiled kernel on PPU.
    _bool_spec(
        "VLLM_SAIL_USE_TRITON_INT8_QUANT",
        section="Backend selection",
        default=True,
        description="Use Triton dynamic per-token INT8 activation quantization.",
    ),
    # DeepEP low-latency deferred recv hook. The hook only pays off on backends
    # that can overlap the deferred receive with other compute; setting this to
    # 0 selects an event-based fallback that is also a simpler debugging
    # baseline for overlap issues.
    _bool_spec(
        "VLLM_DEEPEPLL_RECV_HOOK",
        section="Backend selection",
        default=True,
        description=(
            "Use DeepEP low-latency deferred receive hook; `0` selects the "
            "event-based fallback (shared upstream name)."
        ),
    ),
    # -- DeepSeek V4.1 -----------------------------------------------------
    _choice_spec(
        "VLLM_SAIL_MEGA_MOE_MMA_TYPE",
        section="DeepSeek V4.1",
        default="fp8xfp4",
        choices=MEGA_MOE_MMA_TYPES,
        description=(
            "Activation precision for DeepGEMM MegaMoE MXFP4 routed experts "
            "(`fp4xfp4`=W4A4, `fp8xfp4`=W4A8, `bf16xfp4`=W4A16)."
        ),
    ),
    # Disable the PPU Mega mHC kernel so the upstream TileLang mHC path takes
    # over. The A/B baseline for bisecting mHC regressions: the two paths
    # compute the same shifted post/pre block, so a difference in the model's
    # output points at the Mega mHC kernel.
    _bool_spec(
        "VLLM_SAIL_DISABLE_MEGA_MHC",
        section="DeepSeek V4.1",
        default=False,
        extra_aliases=("SAIL_DISABLE_MEGA_MHC",),
        description=(
            "Disable the PPU DeepGEMM Mega mHC kernel on Zhenwu M890 and fall "
            "back to the upstream TileLang mHC path."
        ),
    ),
    # Master switch for the PPU fused norm-rope sparse-attention path. Default
    # OFF (opt-in): the baseline attention class runs unless the user sets 1.
    # Enabling also requires a compatible flash_mla wheel, PPU SM80/SM89,
    # the 528B KV record switch below, and a compatible V4.1 head/group
    # topology; when any gate is closed the selector falls back to baseline
    # and logs the reason.
    _bool_spec(
        "VLLM_SAIL_FUSED_NORM_ROPE_ATTN",
        section="DeepSeek V4.1",
        default=False,
        description=(
            "Opt into the PPU fused norm-rope sparse-attention path; requires "
            "`VLLM_SAIL_PPU_KV_MXFP8=1`, a compatible `flash_mla` wheel, and "
            "supported V4.1 topology."
        ),
    ),
    # KV record format for DeepSeek V4.1 on PPU: True -> 528B V4.1 MXFP8
    # record (required by the fused decode kernel; opt-in together with the
    # master switch above), False (default) -> 584B V4 record, identical to
    # the upstream non-SM100 default. Read once per process.
    _bool_spec(
        "VLLM_SAIL_PPU_KV_MXFP8",
        section="DeepSeek V4.1",
        default=False,
        description=(
            "Select the 528-byte V4.1 MXFP8 KV record required by the fused "
            "decode kernel instead of the default 584-byte V4 record (read "
            "once per process)."
        ),
    ),
    # Compressed-KV record for the fused V4.1 path: True -> 288B NVFP4 record
    # for the compressed cache only, while the sliding-window cache keeps the
    # 528B MXFP8 record. Requires master switch, VLLM_SAIL_PPU_KV_MXFP8=1,
    # capability 8.9 (ZW-M890P) and sliding_window % 8 == 0.
    _bool_spec(
        "VLLM_SAIL_PPU_KV_NVFP4",
        section="DeepSeek V4.1",
        default=False,
        description=(
            "Opt into a 288-byte NVFP4 record for the compressed KV cache "
            "while keeping 528-byte MXFP8 for the sliding-window cache; "
            "requires `VLLM_SAIL_FUSED_NORM_ROPE_ATTN=1`, "
            "`VLLM_SAIL_PPU_KV_MXFP8=1`, Zhenwu M890 (capability 8.9), and "
            "`sliding_window % 8 == 0` (lossy)."
        ),
    ),
    # Output FP8 scale format of the fused V4.1 norm-rope attention kernel.
    # False (default): num_per_channels=32 with packed UE8M0 scales.
    # True: one exact FP32 dequantization scale per token and WV group
    # (requires capability 8.9 / ZW-M890P and a compatible flash_mla wheel).
    _bool_spec(
        "VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE",
        section="DeepSeek V4.1",
        default=False,
        description=(
            "Use exact per-token FP32 output scales per WV group instead of "
            "packed per-32 UE8M0 scales in fused V4.1 attention; requires "
            "Zhenwu M890 (capability 8.9) and a compatible `flash_mla` wheel."
        ),
    ),
    # -- GLM-5-Next kpool --------------------------------------------------
    # Decode-time logits backend for the glm5_next kpool indexer.
    # ``non_paged`` (default) gathers the pool K cache and uses the non-paged
    # int8 kernel (same one as prefill). The paged INT8 kernel writes only
    # part of the logits on PPU, so selecting it is rejected until fixed.
    _choice_spec(
        "VLLM_SAIL_KPOOL_DECODE_LOGITS",
        section="GLM-5-Next kpool",
        default="non_paged",
        choices=KPOOL_DECODE_LOGITS_BACKENDS,
        description=(
            "Decode logits backend for the Zhenwu 810E INT8 kpool indexer; "
            "only `non_paged` is accepted (`paged` is rejected on PPU)."
        ),
    ),
    # Dense decode fallback bound: batch * ceil(max_seq_len / kpool) pools.
    # Above it the fallback would materialise an O(batch * context) logits
    # matrix (the 1M-context warmup needs ~256 GiB), so fail loudly instead.
    _positive_int_spec(
        "VLLM_SAIL_KPOOL_MAX_DENSE_POOLS",
        section="GLM-5-Next kpool",
        default=1 << 21,
        default_display="`2097152` (`1 << 21`)",
        description=(
            "Upper bound on `batch * ceil(max_seq_len / kpool)` for the kpool "
            "dense decode fallback."
        ),
    ),
    # -- MiniMax M3 SAIL MSA sparse attention ------------------------------
    # Enabled by default: the SAIL path is selected whenever the platform, the
    # model shape and the installed `msa_for_sail` allow it; the dtypes follow
    # the platform and the KV cache dtype (see models/minimax_m3_msa/gating.py).
    # 0 keeps the PPU Triton implementation for both halves.
    _bool_spec(
        "VLLM_SAIL_MINIMAX_M3_MSA",
        section="MiniMax M3 SAIL MSA",
        default=True,
        description=(
            "Enable SAIL MSA sparse attention (`msa_for_sail`) when platform, "
            "shape and wheel gates pass; `0` keeps Triton for both halves."
        ),
    ),
    # Bisection switches: keep one half on the PPU Triton implementation while
    # the other half runs the SAIL kernels.
    _bool_spec(
        "VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_TRITON",
        section="MiniMax M3 SAIL MSA",
        default=False,
        description=(
            "Bisection switch keeping the MiniMax M3 indexer on Triton while "
            "sparse attend uses SAIL MSA."
        ),
    ),
    _bool_spec(
        "VLLM_SAIL_MINIMAX_M3_MSA_ATTEND_TRITON",
        section="MiniMax M3 SAIL MSA",
        default=False,
        description=(
            "Bisection switch keeping MiniMax M3 sparse attend on Triton "
            "while the indexer uses SAIL MSA."
        ),
    ),
    # -- Step-5 DSA --------------------------------------------------------
    # Route the Step5 DSA indexer scoring kernel to the resident-block Triton
    # implementation in vllm_sail/models/step5/ops/. Read on every call. Off by
    # default: upstream's kernel stays the reference until validated per model.
    _bool_spec(
        "VLLM_SAIL_STEP5_DSA_SCORES",
        section="Step-5 DSA",
        default=False,
        description=(
            "Opt into the resident-block Triton indexer scoring kernel on "
            "Zhenwu M890 (capability 8.9)."
        ),
    ),
    # per_row stays the default; the new selector is opt-in only.
    _bool_spec(
        "VLLM_SAIL_STEP5_DSA_SELECTOR_PPU",
        section="Step-5 DSA",
        default=False,
        description=(
            "Opt into the native HGGC FP32 radix region selector (`0 < topk <= 1024`)."
        ),
    ),
    # Defaults to FlashInfer; native opts into the new prefill/decode and old
    # values raise.
    _choice_spec(
        "VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND",
        section="Step-5 DSA",
        default="flashinfer",
        choices=STEP5_PREFILL_BACKENDS,
        description=(
            "Select the Step-5 prefill/decode attention backend (`native` "
            "opts into the plugin's HGGC BF16 paged path)."
        ),
    ),
    # Selects native paged attention during prefill metadata planning.
    _bool_spec(
        "VLLM_SAIL_STEP5_DSA_PREFILL_PPU",
        section="Step-5 DSA",
        default=False,
        description=(
            "Boolean selector for `native` (`1`) vs `flashinfer` (`0`) when "
            "`VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND` is unset."
        ),
    ),
    # The indexer still runs in chunks; only the explicit native route may batch
    # attention at finish.
    _bool_spec(
        "VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION",
        section="Step-5 DSA",
        default=False,
        description=(
            "Enable chunked selection with batched attention at finish on the "
            "`native` Step-5 prefill route."
        ),
    ),
    # 0: E4M3 bytes; 1: E4M3-quantized values stored as FP16, using FP16 MMA.
    # The FP16 summary cache takes twice the memory; keeps the Step5 fork's name.
    # Read lazily; set it before the server starts and keep it fixed at runtime.
    _bool_spec(
        "VLLM_STEP5_INDEXER_FP8_AS_FP16",
        section="Step-5 DSA",
        default=False,
        description=(
            "Store E4M3-quantized indexer summaries, packed pages and queries "
            "as FP16 (2x summary cache memory, FP16 MMA with FP32 "
            "accumulation); set before startup and keep fixed."
        ),
    ),
    # -- Online quantization and compatibility -----------------------------
    _choice_spec(
        "VLLM_SAIL_FP8_LOAD_POLICY",
        section="Online quantization and compatibility",
        default=None,
        choices=("native", "bf16", "int8"),
        description=(
            "Default loading-time FP8 conversion policy when no explicit "
            "`--quantization` argument is supplied."
        ),
    ),
    _bool_spec(
        "VLLM_SAIL_ENABLE_FP8_TO_INT8",
        section="Online quantization and compatibility",
        default=False,
        description=(
            "Legacy loading-time FP8 switch used when "
            "`VLLM_SAIL_FP8_LOAD_POLICY` is unset (`1`=INT8, `0`=BF16 when "
            "explicitly set)."
        ),
    ),
    # Disable / force the wna16 CUDA MoE kernel on PPU. Both exist because the
    # heuristic that picks between the CUDA and Triton kernels is shape
    # dependent and occasionally needs a manual override in either direction.
    _bool_spec(
        "VLLM_SAIL_DISABLE_MOE_WNA16_CUDA",
        section="Online quantization and compatibility",
        default=False,
        description=(
            "Disable the native WNA16 MoE path; takes precedence over "
            "`VLLM_SAIL_FORCE_MOE_WNA16_CUDA`."
        ),
    ),
    _bool_spec(
        "VLLM_SAIL_FORCE_MOE_WNA16_CUDA",
        section="Online quantization and compatibility",
        default=False,
        description=(
            "Request the native WNA16 MoE path; raises if the kernel is unavailable."
        ),
    ),
    # Legacy Marlin selection gate. Native Marlin execution remains excluded
    # by the CUDA-free manifest; this flag cannot supply those kernels.
    _bool_spec(
        "VLLM_SAIL_ENABLE_MOE_MARLIN",
        section="Online quantization and compatibility",
        default=False,
        description=(
            "Legacy selection gate; does not enable excluded native Marlin kernels."
        ),
    ),
    # Legacy getter only. Current PLA/Triton dispatch uses VLLM_SAIL_USE_PLA.
    _bool_spec(
        "VLLM_SAIL_FUSED_GDN_DECODE",
        section="Online quantization and compatibility",
        default=True,
        description=(
            "Legacy getter; GDN decode dispatch now uses "
            "`VLLM_GDN_DECODE_KERNEL` and `VLLM_SAIL_USE_PLA`."
        ),
    ),
    # -- Profiling (opt-in; see vllm_sail/profiling/) ----------------------
    _bool_spec(
        "VLLM_SAIL_NVTX_PROFILE",
        section="Profiling",
        default=False,
        extra_aliases=("SAIL_NVTX_PROFILE",),
        description="Enable optional NVTX instrumentation.",
    ),
    # Step5 forward dict arguments expand only tensor shape/dtype; no device reads.
    _bool_spec(
        "VLLM_SAIL_NVTX_DUMP_DICT",
        section="Profiling",
        default=False,
        description=(
            "Expand Step-5 forward dictionary arguments as tensor shapes and "
            "dtypes in NVTX labels without device reads."
        ),
    ),
    _bool_spec(
        "VLLM_SAIL_NVTX_DUMP_TOPK",
        section="Profiling",
        default=False,
        description="Include top-k details in NVTX labels.",
    ),
    _bool_spec(
        "VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN",
        section="Profiling",
        default=False,
        description="Include sequence lengths in FlashAttention NVTX labels.",
    ),
)

environment_specs: dict[str, EnvSpec] = {spec.name: spec for spec in ENVIRONMENT_SPECS}
environment_variables: dict[str, Callable[[], Any]] = {
    spec.name: spec.getter for spec in ENVIRONMENT_SPECS
}

# All plugin variables retain the corresponding PPU spelling. The shared
# upstream VLLM_DEEPEPLL_RECV_HOOK and VLLM_GDN_DECODE_KERNEL, and the Step5
# fork's VLLM_STEP5_INDEXER_FP8_AS_FP16, keep their names and have no plugin
# aliases.
_aliases: dict[str, tuple[str, ...]] = {
    spec.name: spec.aliases for spec in ENVIRONMENT_SPECS if spec.aliases
}
_canonical_names = {
    alias: name for name, aliases in _aliases.items() for alias in aliases
}


def aliases_of(name: str) -> tuple[str, ...]:
    """Compatibility aliases accepted for ``name``, in precedence order."""
    canonical = _canonical_names.get(name, name)
    return _aliases.get(canonical, ())


def __getattr__(name: str) -> Any:
    name = _canonical_names.get(name, name)
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return [*environment_variables, *_canonical_names]


def is_set(name: str) -> bool:
    """Whether ``name`` or any of its aliases was explicitly set.

    Backend selection distinguishes "user asked for X" from "user did not
    choose", so this is not the same as comparing against the default.
    """
    canonical = _canonical_names.get(name, name)
    return canonical in environment_variables and any(
        candidate in os.environ for candidate in _env_names(canonical)
    )


def snapshot() -> dict[str, Any]:
    """Effective runtime values under canonical names, for ``collect_env``."""
    return {name: getter() for name, getter in environment_variables.items()}
