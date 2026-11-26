# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in lane-streaming and launch tuning for Step5 joint preparation."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from vllm_sail import envs
from vllm_sail.models.step5.ops.soft_fp8 import ByteFP8Launch, _indexer_norm_rope

logger = init_logger(__name__)


class JointPrepareLaunch(ByteFP8Launch):
    """Select lane-streaming and cache/token-aware warps for validated TP4 inputs."""

    def __init__(self, upstream, ppu, byte_args, *, optimized=None):
        super().__init__(upstream, ppu, byte_args)
        self.optimized = (
            ByteFP8Launch(upstream, optimized, byte_args)
            if optimized is not None
            else None
        )
        self._reported = set()

    @staticmethod
    def _fallback_reason(grid, args, kwargs):
        if not envs.VLLM_SAIL_STEP5_JOINT_PREPARE:
            return "disabled"
        if not current_platform.is_ppu():
            return "platform"
        if len(args) != 60 or not isinstance(grid, tuple) or len(grid) != 2:
            return "launch_contract"
        tokens = grid[0]
        if type(tokens) is not int or not 8 <= tokens <= 1024:
            return "tokens"
        if grid[1] != 12:
            return "grid"
        if kwargs != {"num_warps": 4, "enable_fp_fusion": False}:
            return "launch_options"
        if args[16] is not False or any(value is not None for value in args[11:16]):
            return "csa"
        constants = {
            17: 0,
            18: 0,
            19: 0,
            20: 8,
            21: 0,
            22: 8,
            32: 1e-5,
            50: tokens,
            51: 64,
            55: 1e-5,
        }
        if any(args[index] != value for index, value in constants.items()):
            return "constants"
        if args[26:30] != (4, 8, 1, tokens) or args[30] != 32:
            return "indexer_geometry"
        if args[33:35] != (1.0, False) or args[45:47] != (16, 1):
            return "main_geometry"
        if args[48] != 64 or args[56] != 1.0 or type(args[59]) is not bool:
            return "main_rope_or_cache_mode"
        cache_dtype = torch.float8_e4m3fn if args[59] else torch.bfloat16
        expected = {
            0: (torch.bfloat16, (tokens, 1024)),
            1: (torch.bfloat16, (tokens, 256)),
            2: (torch.bfloat16, (tokens, 4)),
            3: (torch.bfloat16, (256,)),
            4: (torch.bfloat16, (256,)),
            5: (torch.bfloat16, (256,)),
            7: (torch.int64, (tokens,)),
            8: (torch.float8_e4m3fn, (1, tokens, 8, 256)),
            9: (torch.bfloat16, (tokens, 1, 256)),
            10: (torch.float32, (1, tokens, 8)),
            35: (torch.bfloat16, (tokens, 3456)),
            36: (torch.float32, (192,)),
            37: (torch.float32, (192,)),
            39: (torch.int64, (tokens,)),
            40: (torch.bfloat16, (tokens, 3072)),
            41: (torch.bfloat16, (tokens, 192)),
            42: (torch.bfloat16, (tokens, 192)),
            57: (torch.float32, ()),
            58: (torch.float32, ()),
        }
        query = args[0]
        if not isinstance(query, torch.Tensor) or not query.is_cuda:
            return "query_device"
        for index, (dtype, shape) in expected.items():
            tensor = args[index]
            if not isinstance(tensor, torch.Tensor):
                return f"tensor_{index}"
            if tensor.device != query.device or tensor.dtype != dtype:
                return f"dtype_or_device_{index}"
            if tuple(tensor.shape) != shape or (shape and tensor.stride(-1) != 1):
                return f"shape_or_stride_{index}"
        for index, width, stride_index in ((6, 32, 31), (38, 64, 49)):
            rope = args[index]
            if not isinstance(rope, torch.Tensor) or rope.device != query.device:
                return f"rope_device_{index}"
            if rope.dtype != torch.bfloat16 or rope.ndim != 2:
                return f"rope_dtype_{index}"
            if rope.shape[1] != width or rope.stride(1) != 1:
                return f"rope_layout_{index}"
            if args[stride_index] != rope.stride(0):
                return f"rope_stride_{index}"
        if any(not args[index].is_contiguous() for index in (8, 9, 10, 40, 41, 42)):
            return "output_layout"
        if args[23] != query.stride(0) or args[24] != args[1].stride(0):
            return "input_stride"
        if args[25] != args[2].stride(0) or args[47] != args[35].stride(0):
            return "input_stride"
        for index in (43, 44):
            cache = args[index]
            if not isinstance(cache, torch.Tensor) or cache.device != query.device:
                return f"cache_device_{index}"
            if cache.dtype != cache_dtype or cache.ndim != 4:
                return f"cache_dtype_{index}"
            if tuple(cache.shape[1:]) != (64, 1, 192) or cache.stride(-1) != 1:
                return f"cache_layout_{index}"
            if tuple(cache.stride(i) for i in range(3)) != args[52:55]:
                return f"cache_stride_{index}"
        if tuple(torch.cuda.get_device_capability(query.device)) != (8, 9):
            return "capability"
        return None

    @classmethod
    def _eligible(cls, grid, args, kwargs):
        return cls._fallback_reason(grid, args, kwargs) is None

    @staticmethod
    def _num_warps(tokens, fp8_cache):
        return (
            8
            if fp8_cache or tokens <= 13 or 23 <= tokens <= 25 or 63 <= tokens <= 65
            else 4
        )

    def _report(self, grid, reason, kwargs, compiled, cache_mode):
        if reason == "disabled" or len(self._reported) >= 64:
            return
        key = (str(grid), reason, kwargs.get("num_warps"), cache_mode)
        if key in self._reported:
            return
        self._reported.add(key)
        metadata = getattr(compiled, "metadata", None)
        kernel = (
            self.optimized.ppu
            if reason is None and self.optimized is not None
            else self.ppu
            if current_platform.is_ppu()
            else self.upstream
        )
        logger.info(
            "Step5 joint prepare: wrapper=%s kernel=%s grid=%s route=%s "
            "reason=%s selected_num_warps=%s compiled_num_warps=%s main_cache=%s "
            "implementation=%s",
            type(self).__name__,
            getattr(kernel, "__name__", type(kernel).__name__),
            grid,
            "tuned" if reason is None else "baseline",
            reason or "eligible",
            kwargs.get("num_warps"),
            getattr(metadata, "num_warps", None),
            cache_mode,
            "lane_streaming"
            if reason is None and self.optimized is not None
            else "original",
        )

    def __getitem__(self, grid):
        baseline = super().__getitem__(grid)

        def run(*args, **kwargs):
            reason = self._fallback_reason(grid, args, kwargs)
            selected = baseline
            if reason is None:
                kwargs = dict(kwargs, num_warps=self._num_warps(grid[0], args[59]))
                if self.optimized is not None:
                    selected = self.optimized[grid]
            compiled = selected(*args, **kwargs)
            cache_mode = (
                str(getattr(args[43], "dtype", None)) if len(args) > 43 else "unknown"
            )
            self._report(grid, reason, kwargs, compiled, cache_mode)
            return compiled

        return run


_UPSTREAM_REVISION = "4ba787ec9959e566d3b587a796dff07673324da5"


@triton.jit(do_not_specialize=["SLOT_ROWS"])
def _prepare_main(
    QKV,
    QWeight,
    KWeight,
    Rope,
    Positions,
    Slots,
    QOut,
    KOut,
    VOut,
    KCache,
    VCache,
    token,
    group,
    QH: tl.constexpr,
    KH: tl.constexpr,
    STRIDE: tl.constexpr,
    ROTARY: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    SLOT_ROWS,
    BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    CACHE_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    BIAS: tl.constexpr,
    KScale,
    VScale,
    FP8_CACHE: tl.constexpr,
):
    DIM: tl.constexpr = 192
    LANES: tl.constexpr = 32 if ROTARY > 128 else 16
    VECTOR: tl.constexpr = 8 if ROTARY > 128 else 4
    GROUP: tl.constexpr = 128 // LANES
    VALUES: tl.constexpr = triton.cdiv(DIM, LANES * VECTOR) * VECTOR
    h = group * GROUP + tl.arange(0, GROUP)
    d = tl.arange(0, 256)
    valid = (h < QH + 2 * KH)[:, None] & (d < DIM)[None, :]
    # PPU MODIFICATION: begin
    lane = tl.arange(0, LANES)
    # Match the native QK kernel's sequential FMA and ascending XOR reduction.
    partial = tl.full((GROUP, LANES), 0, tl.float32)
    for i in tl.static_range(VALUES):
        col = lane * VECTOR + i % VECTOR + i // VECTOR * (LANES * VECTOR)
        value = tl.load(
            QKV + token * STRIDE + h[:, None] * DIM + col[None, :],
            (h < QH + 2 * KH)[:, None] & (col < DIM)[None, :],
            other=0,
        ).to(tl.float32)
        partial = tl.fma(value, value, partial)
    for i in tl.static_range(5 if LANES == 32 else 4):
        other = tl.broadcast_to((lane ^ (1 << i))[None, :], (GROUP, LANES))
        partial = partial + tl.gather(partial, other, 1)
    sums = tl.sum(tl.where(lane[None, :] == 0, partial, 0.0), 1)
    inverse = tl.rsqrt(tl.div_rn(sums, 192.0) + EPS)
    raw = tl.load(QKV + token * STRIDE + h[:, None] * DIM + d[None, :], valid, other=0)
    x = raw.to(tl.float32)
    # PPU MODIFICATION: end
    weight_ptr = tl.where((h < QH)[:, None], QWeight + d[None, :], KWeight + d[None, :])
    weight = (
        tl.load(weight_ptr, (h < QH + KH)[:, None] & (d < DIM)[None, :], other=0).to(
            tl.float32
        )
        + BIAS
    )
    normalized = (x * inverse[:, None]) * weight
    # Norm is rounded before RoPE; its second half uses an explicit FMA.
    normalized = normalized.to(QKV.dtype.element_ty).to(tl.float32)
    if ROTARY > 0:
        half: tl.constexpr = ROTARY // 2
        partner = tl.where(d < half, d + half, tl.where(d < ROTARY, d - half, d))
        partner_x = tl.gather(
            normalized, tl.broadcast_to(partner[None, :], (GROUP, 256)), 1
        )
        position = tl.load(Positions + token)
        cos = tl.load(Rope + position * ROPE_STRIDE + d % half).to(tl.float32)
        sin = tl.load(Rope + position * ROPE_STRIDE + half + d % half).to(tl.float32)
        low = normalized * cos[None, :] - partner_x * sin[None, :]
        high = tl.fma(partner_x, sin[None, :], normalized * cos[None, :])
        normalized = tl.where(
            (d < ROTARY)[None, :], tl.where((d < half)[None, :], low, high), normalized
        )
    result = tl.where((h < QH + KH)[:, None], normalized.to(QKV.dtype.element_ty), raw)
    target = tl.where(
        (h < QH)[:, None],
        QOut + token * QH * DIM + h[:, None] * DIM + d[None, :],
        tl.where(
            (h < QH + KH)[:, None],
            KOut + token * KH * DIM + (h[:, None] - QH) * DIM + d[None, :],
            VOut + token * KH * DIM + (h[:, None] - QH - KH) * DIM + d[None, :],
        ),
    )
    tl.store(target, result, valid)
    slot = tl.load(Slots + token, token < SLOT_ROWS, other=-1)
    cache_head = tl.where(h < QH + KH, h - QH, h - QH - KH)
    offset = (
        (slot // BLOCK_SIZE) * CACHE_BLOCK
        + (slot % BLOCK_SIZE) * CACHE_TOKEN
        + cache_head[:, None] * CACHE_HEAD
        + d[None, :]
    )
    cache_ptr = tl.where((h < QH + KH)[:, None], KCache + offset, VCache + offset)
    if FP8_CACHE:
        scale = tl.where((h < QH + KH)[:, None], tl.load(KScale), tl.load(VScale))
        result = tl.div_rn(result.to(tl.float32), scale)
        result = tl.minimum(tl.maximum(result, -448.0), 448.0)
    tl.store(cache_ptr, result, valid & (h >= QH)[:, None] & (slot >= 0))


@triton.jit(do_not_specialize=["CSA_ROWS", "TOKENS", "MAIN_SLOT_ROWS"])
def _joint_prepare(
    Q,
    K,
    W,
    QWeight,
    KWeight,
    KBias,
    Rope,
    Positions,
    QOut,
    KOut,
    WOut,
    Z,
    Slots,
    Mean,
    Ring,
    RingBase,
    UPDATE_CSA: tl.constexpr,
    CSA_ROWS,
    Z_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    MEAN_BLOCK_SIZE: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    TOKENS,
    ROTARY: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_BIAS: tl.constexpr,
    NATIVE_ROPE: tl.constexpr,
    MainQKV,
    MainQWeight,
    MainKWeight,
    MainRope,
    MainSlots,
    MainQ,
    MainK,
    MainV,
    MainKC,
    MainVC,
    MAIN_QH: tl.constexpr,
    MAIN_KH: tl.constexpr,
    MAIN_STRIDE: tl.constexpr,
    MAIN_ROTARY: tl.constexpr,
    MAIN_ROPE_STRIDE: tl.constexpr,
    MAIN_SLOT_ROWS,
    MAIN_BLOCK_SIZE: tl.constexpr,
    MAIN_CACHE_BLOCK: tl.constexpr,
    MAIN_CACHE_TOKEN: tl.constexpr,
    MAIN_CACHE_HEAD: tl.constexpr,
    MAIN_EPS: tl.constexpr,
    MAIN_BIAS: tl.constexpr,
    MainKScale,
    MainVScale,
    FP8_CACHE: tl.constexpr,
):
    if tl.program_id(1) <= GROUPS * PAD_HEADS:
        _indexer_norm_rope(
            Q,
            K,
            W,
            QWeight,
            KWeight,
            KBias,
            Rope,
            Positions,
            QOut,
            KOut,
            WOut,
            Z,
            Slots,
            Mean,
            Ring,
            RingBase,
            UPDATE_CSA,
            CSA_ROWS,
            Z_STRIDE,
            SLOT_STRIDE,
            RING_CAPACITY,
            MEAN_BLOCK,
            MEAN_BLOCK_SIZE,
            Q_STRIDE,
            K_STRIDE,
            W_STRIDE,
            HEADS,
            PAD_HEADS,
            GROUPS,
            TOKENS,
            ROTARY,
            ROPE_STRIDE,
            EPS,
            WEIGHT_BIAS,
            NATIVE_ROPE,
        )
    else:
        _prepare_main(
            MainQKV,
            MainQWeight,
            MainKWeight,
            MainRope,
            Positions,
            MainSlots,
            MainQ,
            MainK,
            MainV,
            MainKC,
            MainVC,
            tl.program_id(0),
            tl.program_id(1) - GROUPS * PAD_HEADS - 1,
            MAIN_QH,
            MAIN_KH,
            MAIN_STRIDE,
            MAIN_ROTARY,
            MAIN_ROPE_STRIDE,
            MAIN_SLOT_ROWS,
            MAIN_BLOCK_SIZE,
            MAIN_CACHE_BLOCK,
            MAIN_CACHE_TOKEN,
            MAIN_CACHE_HEAD,
            MAIN_EPS,
            MAIN_BIAS,
            MainKScale,
            MainVScale,
            FP8_CACHE,
        )
