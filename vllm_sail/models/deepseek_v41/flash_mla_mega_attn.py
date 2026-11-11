# SPDX-License-Identifier: Apache-2.0
"""PPU port of vLLM's DeepSeek V4.1 FlashMLA mega-attention layer.

Upstream ``DeepseekV4MegaAttnAttention`` (SM100/CUDA) fuses Q RoPE, sparse
attention, the output's inverse RoPE and its FP8 cast into one launch of
``torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_{fwd,decode}``.
FlashMLA-for-sail PR#54 (head e06e9d03bf) ports the same kernels to PPU
SM80/SM89 behind the ``flash_mla`` Python package
(``flash_mla.fused_norm_rope_attn_rope_cast.{prefill,decode}``, dispatched
through ``flash_mla_cuda``). This module mirrors the upstream layer for PPU:

* kernel entry points come from ``flash_mla`` (lazy import; a baseline wheel
  without the fused API never breaks module import or the selector);
* the PPU API allocates and *returns* ``(out_fp8, out_sf, ...)`` instead of
  writing in place into caller-provided buffers, so results are copied into
  the pre-allocated ``QuantizedActivation`` token slices (same MN-major
  packed-UE8M0 scale layout on both sides; ``copy_`` is element-correct even
  when the per-chunk aligned strides differ);
* ``backend_cls`` stays ``FLASHMLA_SPARSE_DSV41`` (identical metadata and KV
  geometry per upstream's own note) and ``packed_kv_cache_dtype`` stays
  ``fp8_ds_mla`` — PPU has no ``nvfp4_ds_mla`` compressed record; the fused
  decode kernel reads the 528B V4.1 fp8 record for both caches (enabled by
  the ``dsv41_kv_record`` patch);
* capability gate mirrors the kernel's own hard check: PPU and
  ``(major, minor) in {(8, 0), (8, 9)}`` (PR#54
  csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp ``TORCH_CHECK``); arch
  subdivision (89 vs 80) and fast/generic path selection stay inside the C++
  dispatch;
* forensic ``info_once`` lines on the first real decode/prefill kernel call
  (ledger level L3): s_q / h_q / topk / extra_topk / kv_bytes.

v1.1 (wo_a layout fix, approved 2026-11-08): on PPU the MXFP8 bmm kernel
selection can only land on ``EmulationMxfp8LinearKernel`` (the DeepGEMM bmm
kernel requires capability family 100, vllm
model_executor/kernels/linear/mxfp8/deep_gemm.py:34), which dequantizes
``wo_a`` to BF16 at load time (emulation.py:31-50, env
``VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD`` defaults True). The inherited
upstream ``_o_proj`` then feeds a BF16 2D weight plus E8M0 2D scales into
``fp8_einsum`` (flash_mla_mega_attn.py:221), whose three operands must share
the group batch dim -> DeepGEMM einsum.hpp:68 assertion. Additionally
``self._einsum_recipe`` computed on PPU is (1, 128, 128) (capability major
<= 9 branch, vllm models/deepseek_v4/nvidia/ops/o_proj.py:28-31) while the
fused kernel always emits per-32 packed UE8M0 scales. v1.1 therefore:

* overrides ``__init__`` to pin ``_einsum_recipe = (1, 1, 32)`` and
  ``_tma_aligned_scales = False`` (same convention the vllm_sail baseline
  V4.1 class hardcodes, vllm_sail/models/deepseek_v41/flashmla.py:44-57);
* overrides ``_o_proj`` for the PPU emulation reality: exactly dequantize
  the fused FP8 output (e4m3 x 2^ue8m0, lossless in fp32/bf16) and run the
  grouped BF16 ``torch.bmm`` against ``wo_a`` — mirroring the non-fp8 branch
  of ``deep_gemm_fp8_o_proj`` (vllm models/deepseek_v4/nvidia/ops/o_proj.py:
  96-103) minus the inverse RoPE, which the fused kernel already applied;
* pins real shapes with loud assertions and a first-call ``info_once``
  dumping the full contract (wo_a weight/scale shape+dtype, recipe, block
  size, attn_out data/scale shape+dtype, n_local_groups / n_wv_group /
  o_lora_rank / D). No silent fallback: any shape/dtype deviation raises
  with both sides of the expectation.

Group-dim rationale (self-check item 1, 2026-11-08): the bmm batch dim is
``n_local_groups``, NOT ``n_wv_group``. Upstream's SM100 ``_o_proj`` does
exactly this — flash_mla_mega_attn.py:215 ``groups = self.n_local_groups``
and :223 ``attn_out.data[:, :groups], attn_out.scale[:, :groups]`` — and
``deep_gemm_fp8_o_proj``'s docstring states the kernel writes "one wo_a
group per slot with the live groups first, so only the padding groups are
sliced off". ``n_wv_group = padded_heads // 8 = 8`` counts the *padded*
output slots (TP8: 8 local heads zero-padded to 64); under TP8
``n_local_groups = 1`` and the runtime ``wo_a`` shard is expected to be
``[1 * o_lora_rank, D] = [1024, 4096]`` (ckpt ``[8192, 4096]`` sharded
8-way on the output dim by ColumnParallelLinear). If the real D differs
(the unresolved dump claim of D=1024 vs the D=4096 contract), the
weight-shape assertion below fails loudly printing n_local_groups /
n_wv_group / weight.shape / data.shape — never a silent 1-group slice.

v1.2 (decode regression stage-1 fixes F1+F2, per
decode_regression_rootcause_fixplan_20261111): the FUSED=1 decode path paid
+78.3k us/window in six per-call layout-copy kernels and +5.7k us/window in
two D2D memcpys relative to the unfused baseline, because the PPU API
allocates its own outputs and ``_o_proj`` dequantizes through a 6-step ATen
chain. v1.2 removes both, purely on the vllm_sail side:

* F1 zero-copy output (env ``VLLM_SAIL_FUSED_ZERO_COPY_OUT``, default on):
  on decode-only steps (``num_prefills == 0``) stash the kernel-owned
  ``(out_fp8, out_sf)`` and let ``_o_proj`` consume them directly instead of
  ``copy_``-ing into the shared step buffer. Mixed steps and warmup dummy
  runs keep the copy path; ``forward_mqa`` clears the stash on entry so a
  stale segment can never leak into the next step. Under CUDA-graph capture
  the stash references are pinned at capture time and the fork op rewrites
  the same graph-pool addresses on replay, so semantics are unchanged
  (fixplan §二.F1; piecewise-graph segmentation check listed in §四.7).
* F2 fused dequant (env ``VLLM_SAIL_FUSED_DEQUANT_MODE``, default
  ``triton``): replace the e8m0->f32 / repeat_interleave / fp8->f32 / mul /
  bf16 ATen chain with a single Triton kernel that extracts the four UE8M0
  bytes per int32 word by bit arithmetic and writes BF16 directly. The
  result is bit-identical to the legacy chain for exponent bytes 1..254
  (e4m3 x power-of-two is exact in f32/bf16; byte 0 -> 2^-127 and byte 255
  -> NaN are special-cased to match the e8m0 semantics). If the Triton path
  raises (e.g. fp8e4m3 load unsupported on this Triton/PPU build — fixplan
  §四.4 lists it as needs-measurement), the module logs a loud warning and
  degrades to ``torch`` mode (broadcast-view, 4 kernels, no
  repeat_interleave) for the rest of the process; ``legacy`` restores the
  pre-v1.2 chain for A/B attribution.
* F3 (positions int32 cache) is intentionally NOT implemented: the fixplan
  marks it conditional on the per-layer ``positions.to(torch.int32)``
  proving to be a standalone kernel (needs-measurement item §四.1/§七.B.4).

Stage-2 fixes F4 (metadata plan reuse) and F5 (clamp sinking) live in the
FlashMLA-for-sail fork wheel, not in vllm_sail, and are out of scope here.

Selection is gated by ``fused_gate`` (env switch, KV-record switch, platform,
capability, wheel capability probe, V4.1 head/group topology); the selector
patch in ``vllm_sail/patch/enhancement/models/deepseek_v41.py`` logs which
gate closed when the class is not chosen.
"""

from __future__ import annotations

import importlib.util

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v41.common.ops.fused_layout import WV_GROUP_SIZE
from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
    _HEAD_DIM_V,
    _QUANT_GROUP,
    _ROPE_DIM,
    _token_slice,
    DeepseekV4MegaAttnAttention,
)
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLABackend
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.worker.workspace import current_workspace_manager

import vllm_sail.envs as ppu_envs

logger = init_logger(__name__)

#: PPU capabilities the PR#54 fused kernels accept (kernel-side TORCH_CHECK:
#: major == 8 && (minor == 0 || minor == 9); minor 9 -> arch 89, else 80).
_PPU_FUSED_CAPABILITIES = ((8, 0), (8, 9))

#: v1.1: the fused kernel always emits per-32 packed UE8M0 output scales, so
#: the PPU o_proj contract is the per-32 recipe without TMA-aligned scales
#: (mirrors vllm_sail/models/deepseek_v41/flashmla.py and the SM100 MXFP8 bmm
#: kernel's fixed recipe in vllm .../mxfp8/deep_gemm.py:26).
_PPU_EINSUM_RECIPE = (1, 1, 32)

_API = None  # lazy flash_mla.fused_norm_rope_attn_rope_cast module handle

#: v1.2 F2: set True after the Triton dequant path raised once; the process
#: then stays on the broadcast fallback (loud warning, no silent retry).
_DEQUANT_TRITON_BROKEN = False

#: lazily built @triton.jit kernel handle (import stays lazy: CPU-only
#: environments and the baseline wheel must never touch triton at import).
_DEQUANT_KERNEL = None


def _fused_api():
    """Lazily fetch the PR#54 Python API; never called unless fused_gate passed."""
    global _API
    if _API is None:
        import flash_mla

        _API = flash_mla.fused_norm_rope_attn_rope_cast
    return _API


def _check_dequant_contract(
    data: torch.Tensor, scale: torch.Tensor
) -> tuple[int, int, int]:
    tokens, groups, dim = data.shape
    if scale.shape != (tokens, groups, dim // 128):
        raise AssertionError(
            f"fused output scale shape {tuple(scale.shape)} != expected "
            f"{(tokens, groups, dim // 128)} for data shape {tuple(data.shape)}"
        )
    return tokens, groups, dim


def _dequant_fused_output_legacy(
    data: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Pre-v1.2 6-kernel ATen chain, kept verbatim for A/B attribution."""
    # int32 -> 4 UE8M0 bytes -> float (2^exponent), per 32 dims.
    # Byte order (self-check item 4): ``Tensor.view(torch.uint8)`` reinterprets
    # each int32 word in the platform's native byte order, which is
    # little-endian on both x86-64 hosts and the PPU device — byte j of word i
    # covers dims [128*i + 32*j, +32). This matches DeepGEMM's packed-UE8M0
    # convention (four per-32 scales per int32, lowest block in the LSB), the
    # same convention alloc_mega_attn_output's int32 scale buffer and the
    # fused kernel's use_packed_ue8m0=True output use on both SM100 and PPU.
    scale_bytes = scale.contiguous().view(torch.uint8)  # [T, G, D//32]
    scale_f32 = scale_bytes.view(torch.float8_e8m0fnu).to(torch.float32)
    scale_expanded = scale_f32.repeat_interleave(_QUANT_GROUP, dim=-1)  # [T,G,D]
    return (data.to(torch.float32) * scale_expanded).to(torch.bfloat16)


def _dequant_fused_output_torch(
    data: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """v1.2 degrade tier: broadcast-view dequant, 4 kernels instead of 6.

    Same exact arithmetic as the legacy chain, but the per-32 scale is
    broadcast over a [T, G, D//32, 32] view of the data instead of being
    materialized at [T, G, D] by ``repeat_interleave`` (fixplan §二.F2
    fallback: drops the 32MB f32 intermediate write and two kernels).
    """
    tokens, groups, dim = _check_dequant_contract(data, scale)
    scale_bytes = scale.contiguous().view(torch.uint8)  # [T, G, D//32]
    scale_f32 = scale_bytes.view(torch.float8_e8m0fnu).to(torch.float32)
    x = data.reshape(tokens, groups, dim // 32, _QUANT_GROUP).to(torch.float32)
    y = x * scale_f32.unsqueeze(-1)
    return y.to(torch.bfloat16).reshape(tokens, groups, dim)


def _dequant_kernel():
    """Build (once) the Triton kernel for the fused exact dequant."""
    global _DEQUANT_KERNEL
    if _DEQUANT_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _dequant_mxfp8_packed_ue8m0_kernel(
            data_ptr,
            scale_ptr,
            out_ptr,
            n_groups,
            stride_dt,
            stride_dg,
            stride_dd,
            stride_st,
            stride_sg,
            stride_sd,
            stride_ot,
            stride_og,
            stride_od,
            BLOCK_D: tl.constexpr,
        ):
            # One program per (token, group) row: BLOCK_D fp8e4m3 values and
            # BLOCK_D//128 int32 words (four packed UE8M0 bytes each).
            pid = tl.program_id(0)
            t = pid // n_groups
            g = pid % n_groups

            offs_w = tl.arange(0, BLOCK_D // 128)
            words = tl.load(
                scale_ptr + t * stride_st + g * stride_sg + offs_w * stride_sd
            )
            # Little-endian word -> bytes: byte j covers dims [128*i + 32*j,
            # +32), i.e. flat per-32 block index k = 4*i + j (same convention
            # as the legacy chain's view(torch.uint8)).
            byte_idx = tl.arange(0, 4)
            e = (words[:, None] >> (8 * byte_idx)[None, :]) & 0xFF  # [NW, 4]
            e = tl.reshape(e, (BLOCK_D // 32,))
            # UE8M0 -> f32 by exponent-field construction: bits = e << 23
            # gives 2^(e-127) for 1 <= e <= 254; e == 0 is the denormal
            # 2^-127 (0x00400000) and e == 255 is NaN (0x7FC00000), matching
            # torch's float8_e8m0fnu -> float32 conversion.
            bits = tl.where(
                e == 0, 0x00400000, tl.where(e == 255, 0x7FC00000, e << 23)
            )
            scale_f32 = tl.reshape(
                tl.broadcast_to(bits[:, None], (BLOCK_D // 32, 32)), (BLOCK_D,)
            ).to(tl.float32, bitcast=True)

            offs_d = tl.arange(0, BLOCK_D)
            x = tl.load(
                data_ptr + t * stride_dt + g * stride_dg + offs_d * stride_dd
            ).to(tl.float32)
            y = x * scale_f32
            tl.store(
                out_ptr + t * stride_ot + g * stride_og + offs_d * stride_od,
                y.to(tl.bfloat16),
            )

        _DEQUANT_KERNEL = _dequant_mxfp8_packed_ue8m0_kernel
    return _DEQUANT_KERNEL


def _dequant_fused_output_triton(
    data: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """v1.2 default: single fused kernel, bit-identical to the legacy chain.

    Raises (and the dispatcher degrades to the broadcast tier) when the row
    width is not a power of two, when Triton is unavailable, or when this
    Triton/PPU build cannot compile the fp8e4m3 load — fixplan §四.4 lists
    PPU Triton fp8/e8m0 support as needs-measurement.
    """
    tokens, groups, dim = _check_dequant_contract(data, scale)
    n_words = dim // 128
    if tokens == 0:
        return torch.empty(
            (0, groups, dim), dtype=torch.bfloat16, device=data.device
        )
    if dim & (dim - 1) != 0 or n_words & (n_words - 1) != 0:
        raise ValueError(
            f"Triton fused dequant needs power-of-two D and D//128, got D={dim}"
        )
    if data.dtype != torch.float8_e4m3fn or scale.dtype != torch.int32:
        raise AssertionError(
            f"Triton fused dequant expects (float8_e4m3fn, int32), got "
            f"({data.dtype}, {scale.dtype})"
        )
    out = torch.empty(
        (tokens, groups, dim), dtype=torch.bfloat16, device=data.device
    )
    kernel = _dequant_kernel()
    kernel[(tokens * groups,)](
        data,
        scale,
        out,
        groups,
        data.stride(0),
        data.stride(1),
        data.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=dim,
        num_warps=8 if dim >= 4096 else 4,
    )
    return out


def _dequant_fused_output(
    data: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Exact BF16 dequantization of the fused kernel's packed-MXFP8 output.

    ``data``: ``[T, G, D]`` float8_e4m3fn. ``scale``: ``[T, G, D // 128]``
    int32, each word holding four UE8M0 bytes, one per 32-dim block, in
    little-endian byte order (byte j covers dims ``[128*i + 32*j, +32)``).
    e4m3 (3-bit mantissa) times a power of two is exact in fp32 and in bf16,
    so this dequantization introduces no rounding of its own — all three
    v1.2 tiers (triton / torch / legacy) are bit-identical for exponent
    bytes 1..254.
    """
    _check_dequant_contract(data, scale)
    mode = ppu_envs.VLLM_SAIL_FUSED_DEQUANT_MODE
    if mode == "legacy":
        return _dequant_fused_output_legacy(data, scale)
    if mode == "torch":
        return _dequant_fused_output_torch(data, scale)
    global _DEQUANT_TRITON_BROKEN
    if _DEQUANT_TRITON_BROKEN:
        return _dequant_fused_output_torch(data, scale)
    try:
        return _dequant_fused_output_triton(data, scale)
    except Exception as exc:  # noqa: BLE001 - degrade loudly, never crash decode
        _DEQUANT_TRITON_BROKEN = True
        logger.warning(
            "[vllm_sail] fused dequant Triton path failed (%r); degrading to "
            "the broadcast fallback (VLLM_SAIL_FUSED_DEQUANT_MODE=torch "
            "semantics) for the rest of this process. Set "
            "VLLM_SAIL_FUSED_DEQUANT_MODE=legacy for the pre-v1.2 chain.",
            exc,
        )
        return _dequant_fused_output_torch(data, scale)


def fused_gate(vllm_config) -> tuple[bool, str]:
    """Whether the PPU fused norm-rope path may serve this configuration.

    Returns ``(ok, report)``; ``report`` names the closing gate so the
    selector can log exactly which of the two env switches, the wheel, the
    capability or the topology declined. Order: cheap env/platform checks
    first, wheel import probe last.
    """
    if not ppu_envs.VLLM_SAIL_FUSED_NORM_ROPE_ATTN:
        return False, "gate-env: VLLM_SAIL_FUSED_NORM_ROPE_ATTN=0"
    if not ppu_envs.VLLM_SAIL_PPU_KV_MXFP8:
        return False, (
            "gate-kvrecord: VLLM_SAIL_PPU_KV_MXFP8=0 selects the 584B V4 "
            "record; the fused decode kernel only reads the 528B V4.1 record"
        )
    if not current_platform.is_ppu():
        return False, "gate-platform: not PPU"
    cap = current_platform.get_device_capability()
    if cap is None or (cap.major, cap.minor) not in _PPU_FUSED_CAPABILITIES:
        return False, f"gate-cap: capability {cap} not in {_PPU_FUSED_CAPABILITIES}"
    if importlib.util.find_spec("flash_mla") is None:
        return False, "gate-wheel: flash_mla not installed"
    try:
        import flash_mla

        api = getattr(flash_mla, "fused_norm_rope_attn_rope_cast", None)
        if api is None or not all(
            hasattr(api, name) for name in ("prefill", "decode")
        ):
            return False, (
                "gate-wheel: flash_mla lacks fused_norm_rope_attn_rope_cast "
                "(baseline wheel b1b0e1f); fused path disabled"
            )
    except Exception as exc:  # noqa: BLE001 - probe must never crash selection
        return False, f"gate-wheel: flash_mla import failed: {exc!r}"

    # Topology: mirrors upstream DeepseekV4MegaAttnAttention.is_available_for
    # minus the SM100/torch.ops checks (replaced by the gates above).
    config = vllm_config.model_config.hf_config
    config = getattr(config, "text_config", config)
    n_heads = getattr(config, "num_attention_heads", 0) or 0
    n_groups = getattr(config, "o_groups", 0) or 0
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    if n_heads % tp_size or n_groups % tp_size:
        return False, (
            f"gate-topology: heads={n_heads} o_groups={n_groups} not "
            f"divisible by tp={tp_size}"
        )
    n_local_heads, n_local_groups = n_heads // tp_size, n_groups // tp_size
    if (
        not n_local_groups
        or n_local_heads % WV_GROUP_SIZE
        or n_local_heads // n_local_groups != WV_GROUP_SIZE
    ):
        return False, (
            f"gate-topology: local_heads={n_local_heads} "
            f"local_groups={n_local_groups} incompatible with "
            f"WV_GROUP_SIZE={WV_GROUP_SIZE} (need local_heads % 8 == 0 and "
            f"exactly 8 heads per wo_a group)"
        )
    return True, (
        f"all gates open: cap={cap.major}.{cap.minor} tp={tp_size} "
        f"local_heads={n_local_heads} padded_heads="
        f"{64 if n_local_heads <= 64 else 128}"
    )


class PPUDeepseekV4MegaAttnAttention(DeepseekV4MegaAttnAttention):
    """FlashMLA fused norm-rope sparse MLA layer for DeepSeek V4.1 (PPU)."""

    # Same metadata / KV geometry as FLASHMLA_SPARSE_DSV41 (upstream note on
    # FlashMLAMegaAttnBackend); avoids the SM100-only mega backend whose
    # supports_compute_capability requires major == 10 and whose dtype list
    # would resolve --kv-cache-dtype auto to nvfp4_ds_mla.
    backend_cls = DeepseekV4FlashMLABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # v1.1: the inherited __init__ computed the recipe from the device
        # capability, which on PPU (major 8 <= 9) yields (1, 128, 128) — the
        # wrong convention for the fused kernel's fixed per-32 packed UE8M0
        # output scales. Pin the PPU contract (same values the vllm_sail
        # baseline V4.1 class hardcodes).
        self._einsum_recipe = _PPU_EINSUM_RECIPE
        self._tma_aligned_scales = False
        # v1.2 F1: per-step (out_fp8, out_sf) stash handed from
        # _forward_decode_mega to _o_proj on decode-only steps; None means
        # "consume attn_out.data/.scale as written by the copy path".
        self._last_decode_out = None

    def forward_mqa(self, q, kv, positions, output) -> None:
        # Stash hygiene: clear before anything else so a stale zero-copy
        # segment can never leak into this step's _o_proj — notably the
        # warmup dummy run (attn_metadata is None) returns inside super()
        # without ever reaching the decode branch, and mixed prefill+decode
        # steps must present a None stash to _o_proj.
        self._last_decode_out = None
        super().forward_mqa(q, kv, positions, output)

    @classmethod
    def is_available_for(cls, vllm_config) -> bool:
        return fused_gate(vllm_config)[0]

    @property
    def packed_kv_cache_dtype(self):
        # PPU has no NVFP4 compressed record; both the SWA and the compressed
        # cache keep the 528B V4.1 fp8 record the fused decode kernel reads.
        return "fp8_ds_mla"

    # ---- output projection (v1.1, PPU emulation reality) -------------------

    def _o_proj(
        self, attn_out: QuantizedActivation, positions: torch.Tensor
    ) -> torch.Tensor:
        """Grouped BF16 ``wo_a`` bmm then ``wo_b`` over the fused output.

        The inverse RoPE and the FP8 cast already happened inside the fused
        attention kernel, so ``positions`` is unused here. On PPU ``wo_a`` is
        a BF16 2D shard (EmulationMxfp8LinearKernel dequantized it at load
        time), which the upstream fp8_einsum cannot consume; dequantize the
        kernel output exactly and run the same grouped bmm the baseline
        non-fp8 path runs (deep_gemm_fp8_o_proj's bmm branch), with no second
        inverse RoPE. Every shape/dtype contract is asserted loudly; there is
        no silent fallback.
        """
        weight = self.wo_a.weight
        weight_scale = getattr(self.wo_a, "weight_scale", None)
        # v1.2 F1: on decode-only steps the fused kernel's own outputs were
        # stashed instead of copied into the step buffer; consume (and clear)
        # the stash here. A shape disagreement with attn_out is a contract
        # violation — fail loudly, never fall back to the unwritten buffer.
        stashed = self._last_decode_out
        self._last_decode_out = None
        if stashed is not None:
            data, scale = stashed
            zero_copy = True
            if data.shape != attn_out.data.shape or (
                scale.shape != attn_out.scale.shape
            ):
                raise AssertionError(
                    f"zero-copy stash shapes {tuple(data.shape)}/"
                    f"{tuple(scale.shape)} != attn_out "
                    f"{tuple(attn_out.data.shape)}/"
                    f"{tuple(attn_out.scale.shape)}; the stashed decode "
                    "segment must cover exactly this step's tokens (the "
                    "stash is only taken when num_prefills == 0)"
                )
        else:
            data, scale = attn_out.data, attn_out.scale
            zero_copy = False
        dim = data.shape[2] if data.ndim == 3 else None
        logger.info_once(
            "[vllm_sail] fused norm-rope O_PROJ first call: "
            "wo_a.weight.shape=%s dtype=%s | wo_a.weight_scale.shape=%s "
            "dtype=%s | einsum_recipe=%s o_proj_block_size=%s | "
            "attn_out.data.shape=%s dtype=%s | attn_out.scale.shape=%s "
            "dtype=%s | n_local_groups=%s n_wv_group=%s o_lora_rank=%s D=%s "
            "| zero_copy=%s dequant_mode=%s",
            tuple(weight.shape), weight.dtype,
            tuple(weight_scale.shape) if weight_scale is not None else None,
            weight_scale.dtype if weight_scale is not None else None,
            self._einsum_recipe, self._o_proj_block_size,
            tuple(data.shape), data.dtype,
            tuple(scale.shape), scale.dtype,
            self.n_local_groups, self.n_wv_group, self.o_lora_rank, dim,
            zero_copy, ppu_envs.VLLM_SAIL_FUSED_DEQUANT_MODE,
        )

        # Batch dim of the grouped bmm is n_local_groups, per upstream's own
        # SM100 _o_proj (flash_mla_mega_attn.py:215,223): the fused kernel
        # writes live groups first into the n_wv_group padded slots, so the
        # padding groups (zeros) are sliced off — not a silent 1-group slice
        # of live data. The shape ladder below fails loudly if that contract
        # does not hold on this runtime.
        groups = self.n_local_groups
        if not isinstance(attn_out, QuantizedActivation):
            raise AssertionError(
                f"fused _o_proj requires a QuantizedActivation, got "
                f"{type(attn_out).__name__}"
            )
        if data.dtype != torch.float8_e4m3fn:
            raise AssertionError(
                f"fused attn_out.data dtype must be float8_e4m3fn, got {data.dtype}"
            )
        if scale.dtype != torch.int32:
            raise AssertionError(
                f"fused attn_out.scale dtype must be int32 (packed UE8M0), "
                f"got {scale.dtype}"
            )
        if data.ndim != 3 or data.shape[1] != self.n_wv_group:
            raise AssertionError(
                f"fused attn_out.data shape {tuple(data.shape)} != expected "
                f"(tokens, n_wv_group={self.n_wv_group}, D) "
                f"[n_local_groups={groups}, o_lora_rank={self.o_lora_rank}]"
            )
        dim = data.shape[2]
        if scale.shape != (data.shape[0], data.shape[1], dim // 128):
            raise AssertionError(
                f"fused attn_out.scale shape {tuple(scale.shape)} != expected "
                f"{(data.shape[0], data.shape[1], dim // 128)} "
                f"[n_local_groups={groups}, n_wv_group={self.n_wv_group}, "
                f"data.shape={tuple(data.shape)}]"
            )
        if weight.ndim != 2:
            raise AssertionError(
                f"PPU fused _o_proj expects a 2D wo_a weight shard "
                f"(emulation-dequantized or fp8), got ndim={weight.ndim} "
                f"shape={tuple(weight.shape)}; grouped 3D layouts belong to "
                f"the SM100 DeepGEMM bmm kernel, not PPU "
                f"[n_local_groups={groups}, n_wv_group={self.n_wv_group}, "
                f"data.shape={tuple(data.shape)}]"
            )
        # Expected runtime shard under TP8: [n_local_groups * o_lora_rank, D]
        # = [1 * 1024, 4096] (ckpt [8192, 4096] sharded 8-way on the output
        # dim). D is taken from data.shape — shape-driven, not hardcoded; if
        # the real D is 1024 as one dump claimed, this raises with all four
        # values printed so the first call pins the truth.
        if weight.shape != (groups * self.o_lora_rank, dim):
            raise AssertionError(
                f"wo_a weight shape {tuple(weight.shape)} != expected "
                f"{(groups * self.o_lora_rank, dim)} "
                f"[n_local_groups={groups}, n_wv_group={self.n_wv_group}, "
                f"o_lora_rank={self.o_lora_rank}, D={dim}, "
                f"data.shape={tuple(data.shape)}, "
                f"scale.shape={tuple(scale.shape)}]"
            )

        if weight.element_size() == 1:
            # Defensive branch: emulation dequant-at-load disabled or a
            # checkpoint variant kept wo_a quantized. Same helper the
            # vllm_sail baseline V4.1 class uses.
            if weight.dtype != torch.float8_e4m3fn or weight_scale is None:
                raise AssertionError(
                    f"1-byte wo_a weight must be float8_e4m3fn with a "
                    f"weight_scale, got dtype={weight.dtype} "
                    f"scale={'None' if weight_scale is None else weight_scale.dtype}"
                )
            weight = dequant_mxfp8_to_bf16(weight, weight_scale)
        if weight.dtype != torch.bfloat16:
            raise AssertionError(
                f"wo_a weight must be BF16 after load-time dequant, got {weight.dtype}"
            )

        # Exact dequant of the fused output, live groups first (padding
        # groups are zero-filled by the kernel and sliced off here).
        o = _dequant_fused_output(data[:, :groups], scale[:, :groups])

        # Grouped bmm, mirroring deep_gemm_fp8_o_proj's non-fp8 branch
        # (vllm models/deepseek_v4/nvidia/ops/o_proj.py:96-103); the inverse
        # RoPE is intentionally absent — the fused kernel already applied it.
        # Self-check item 2: upstream writes ``out=z.transpose(0, 1)`` (a
        # non-contiguous view) — verified CUDA-side, NOT verifiable offline
        # for the torch 2.13 PPU backend, so use the allocation-free-risk
        # form: bmm into its own [G, T, R] result, then transpose+contiguous.
        grouped_weight = weight.view(groups, self.o_lora_rank, dim)
        z = torch.bmm(
            o.transpose(0, 1), grouped_weight.transpose(1, 2)
        ).transpose(0, 1).contiguous()  # [T, groups, o_lora_rank]
        return self._wo_b_proj(z.flatten(1))

    # ---- forward -----------------------------------------------------------

    def _forward_decode_mega(
        self,
        q: torch.Tensor,
        positions_int32: torch.Tensor,
        flashmla_metadata,
        swa_metadata,
        out,
    ) -> None:
        """Fused decode over the paged 528B records, copied into ``out``."""
        extra_cache, extra_idx, extra_len = self._decode_compressed_kv_and_topk(
            flashmla_metadata, swa_metadata
        )
        assert swa_metadata.decode_swa_indices is not None
        swa_kv_cache = self.swa_cache_layer.kv_cache
        indices = swa_metadata.decode_swa_indices.view(q.shape[0], -1)
        logger.info_once(
            "[vllm_sail] fused norm-rope DECODE invoked: s_q=%d h_q=%d "
            "topk=%d extra_topk=%d kv_bytes=%d",
            q.shape[0],
            q.shape[1],
            indices.shape[-1],
            0 if extra_idx is None else extra_idx.shape[-1],
            swa_kv_cache.shape[-1],
        )
        out_fp8, out_sf, _lse = _fused_api().decode(
            enable_q_norm=False,  # V4.1 norms qr before wq_b
            rms_norm_eps=0.0,
            token_positions=positions_int32,
            is_rope_neox_style=False,  # GPT-J, kernel-enforced
            rope_dim=_ROPE_DIM,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            n_wv_group=self.n_wv_group,
            num_per_channels=_QUANT_GROUP,
            use_tma_aligned_col_major_sf=True,
            round_sf=True,
            use_packed_ue8m0=True,
            q=q,
            k_cache=swa_kv_cache.unsqueeze(-2),
            indices_in_kvcache=indices,
            sm_scale=self.scale,
            d_v=_HEAD_DIM_V,
            attn_sink=self.attn_sink,
            topk_length=swa_metadata.decode_swa_lens,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=extra_idx,
            extra_topk_length=extra_len,
        )
        # The PPU API allocates its outputs; the contract with _o_proj is the
        # shared step buffer, so copy this segment in (layouts match; strides
        # may differ per alignment, copy_ handles that element-wise).
        # v1.2 F1: on a decode-only step the segment IS the whole step buffer
        # (forward_mqa slices output[0:num_decode_tokens] and there are no
        # prefill tokens), so the two D2D copies can be skipped entirely by
        # handing the kernel-owned tensors to _o_proj through the stash.
        # Any eligibility doubt (mixed step, env off, shape disagreement)
        # deterministically keeps the pre-v1.2 copy path — no silent change.
        if (
            ppu_envs.VLLM_SAIL_FUSED_ZERO_COPY_OUT
            and swa_metadata.num_prefills == 0
            and out_fp8.shape == out.data.shape
            and out_sf.shape == out.scale.shape
        ):
            self._last_decode_out = (out_fp8, out_sf)
            return
        self._last_decode_out = None
        out.data.copy_(out_fp8)
        out.scale.copy_(out_sf)

    def _forward_prefill_mega(
        self,
        q: torch.Tensor,
        positions_int32: torch.Tensor,
        flashmla_metadata,
        swa_metadata,
        out,
        token_base: int,
    ) -> None:
        """Fused prefill over gathered bf16 KV, per chunk, copied into ``out``.

        Body mirrors upstream ``_forward_prefill_mega`` (same chunk plan,
        workspace and index combining); only the kernel call and the output
        handling are PPU-specific.
        """
        swa_only = self.compress_ratio == 0
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:][
            : swa_metadata.num_prefill_tokens
        ]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size, 128)
        for chunk_start, chunk_end, chunk_n, chunk_m in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv_ws, idx_ws, lens_ws = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            if not swa_only:
                assert flashmla_metadata is not None
                dequantize_and_gather_k_cache(
                    kv_ws[:chunk_size, :chunk_n],
                    self._compressed_kv_cache(),
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=flashmla_metadata.block_table[num_decodes:][
                        chunk_start:chunk_end
                    ],
                    block_size=flashmla_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
            dequantize_and_gather_k_cache(
                kv_ws[:chunk_size],
                self.swa_cache_layer.kv_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_metadata.block_table[num_decodes:][
                    chunk_start:chunk_end
                ],
                block_size=swa_metadata.block_size,
                offset=chunk_n,
            )
            first, last = num_decodes + chunk_start, num_decodes + chunk_end
            qs = int(query_start_loc_cpu[first] - prefill_token_base)
            qe = int(query_start_loc_cpu[last] - prefill_token_base)
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[qs:qe],
                query_start_loc[first : last + 1],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_m,
                chunk_n,
                out=(idx_ws[: qe - qs], lens_ws[: qe - qs]),
            )
            chunk_out = _token_slice(out, token_base + qs, token_base + qe)
            logger.info_once(
                "[vllm_sail] fused norm-rope PREFILL invoked: s_q=%d h_q=%d "
                "topk=%d extra_topk=%d kv_bytes=%d",
                qe - qs,
                q.shape[1],
                combined_indices.shape[-1],
                top_k,
                self.swa_cache_layer.kv_cache.shape[-1],
            )
            out_fp8, out_sf, _max_logits, _lse = _fused_api().prefill(
                enable_q_norm=False,
                rms_norm_eps=0.0,
                token_positions=positions_int32[qs:qe],
                is_rope_neox_style=False,
                rope_dim=_ROPE_DIM,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                n_wv_group=self.n_wv_group,
                num_per_channels=_QUANT_GROUP,
                use_tma_aligned_col_major_sf=True,
                round_sf=True,
                use_packed_ue8m0=True,
                q=q[qs:qe],
                kv=kv_ws.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=_HEAD_DIM_V,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )
            chunk_out.data.copy_(out_fp8)
            chunk_out.scale.copy_(out_sf)
