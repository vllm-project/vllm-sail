# SPDX-License-Identifier: Apache-2.0
"""Bare-CPU tests for MXFP4 E2M1 rounding and E8M0 scale kernels.

Executes the real ``_compute_quant_and_scale`` and ``_downcast_to_mxfp4_u16``
bodies from ``vllm_sail/model_executor/layers/quantization/utils/mxfp4_utils.py``
against a minimal pure-Python ``tl`` double so CI catches rounding and scale
regressions without requiring torch, triton, vLLM, or a device.
"""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace

import pytest

from tests.support.source import function

MXFP4_UTILS = "vllm_sail/model_executor/layers/quantization/utils/mxfp4_utils.py"
E2M1_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_to_u32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _u32_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _coerce(value: int | float | bool, dtype: str) -> int | float | bool:
    if dtype == "float32":
        return _f32(float(value))
    if dtype == "int64":
        return int(value)
    if dtype == "uint32":
        return int(value) & 0xFFFFFFFF
    if dtype == "uint16":
        return int(value) & 0xFFFF
    if dtype == "uint8":
        return int(value) & 0xFF
    if dtype == "bool":
        return bool(value)
    raise AssertionError(f"Unsupported dtype {dtype!r}")


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def to(self, dtype: str) -> int:
        return int(_coerce(self.value, dtype))


class _FakePtr:
    def __init__(self, buf: list, dtype: str, offset: int = 0) -> None:
        self.buf = buf
        self.dtype = SimpleNamespace(element_ty=dtype)
        self.offset = offset

    def __add__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeAddr(self.buf, self.dtype.element_ty, self.offset, other)
        return _FakePtr(self.buf, self.dtype.element_ty, self.offset + int(other))


class _FakeAddr:
    def __init__(
        self, buf: list, dtype: str, base_offset: int, offsets: _FakeTensor
    ) -> None:
        self.buf = buf
        self.dtype = dtype
        self.base_offset = base_offset
        self.offsets = offsets


def _expand_to_shape(tensor: _FakeTensor, out_shape: tuple[int, ...]) -> list:
    if tensor.shape == out_shape:
        return tensor.data
    assert len(tensor.shape) == len(out_shape)
    strides = []
    stride = 1
    for dim in reversed(tensor.shape):
        strides.append(stride if dim > 1 else 0)
        stride *= dim
    strides.reverse()
    out = []
    for flat in range(math.prod(out_shape)):
        rem = flat
        coords = []
        for dim in reversed(out_shape):
            coords.append(rem % dim)
            rem //= dim
        coords.reverse()
        idx = sum(c * st for c, st in zip(coords, strides, strict=False))
        out.append(tensor.data[idx])
    return out


class _FakeTensor:
    """Minimal multidimensional tensor matching the Triton ops in the kernel."""

    def __init__(self, data: list, shape: tuple[int, ...], dtype: str) -> None:
        self.data = [_coerce(v, dtype) for v in data]
        self.shape = tuple(shape)
        self.dtype = dtype
        assert len(self.data) == math.prod(self.shape)

    def __getitem__(self, key) -> _FakeTensor:
        if key == (None, slice(None)) and len(self.shape) == 1:
            return _FakeTensor(self.data, (1, self.shape[0]), self.dtype)
        if key == (slice(None), None) and len(self.shape) == 1:
            return _FakeTensor(self.data, (self.shape[0], 1), self.dtype)
        raise AssertionError(f"Unsupported slice {key!r} for shape {self.shape}")

    def to(self, dtype: str, bitcast: bool = False) -> _FakeTensor:
        if bitcast:
            if self.dtype == "float32" and dtype == "uint32":
                return _FakeTensor(
                    [_f32_to_u32(v) for v in self.data], self.shape, "uint32"
                )
            if self.dtype == "uint32" and dtype == "float32":
                return _FakeTensor(
                    [_u32_to_f32(v) for v in self.data], self.shape, "float32"
                )
            raise AssertionError(f"Unsupported bitcast {self.dtype} -> {dtype}")
        return _FakeTensor(self.data, self.shape, dtype)

    def reshape(self, shape: list[int] | tuple[int, ...]) -> _FakeTensor:
        return _FakeTensor(self.data, tuple(shape), self.dtype)

    def _broadcast(self, other: _FakeTensor | int | float) -> tuple[list, list, tuple]:
        if not isinstance(other, _FakeTensor):
            return self.data, [_coerce(other, self.dtype)] * len(self.data), self.shape
        if self.shape == other.shape:
            return self.data, other.data, self.shape
        assert len(self.shape) == len(other.shape)
        out_shape = tuple(
            max(a, b) for a, b in zip(self.shape, other.shape, strict=False)
        )
        return (
            _expand_to_shape(self, out_shape),
            _expand_to_shape(other, out_shape),
            out_shape,
        )

    def _binop(self, other, fn, out_dtype: str | None = None) -> _FakeTensor:
        lhs, rhs, shape = self._broadcast(other)
        dtype = out_dtype or self.dtype
        return _FakeTensor(
            [fn(a, b) for a, b in zip(lhs, rhs, strict=False)], shape, dtype
        )

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binop(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binop(
            other,
            lambda a, b: math.copysign(float("inf"), b) if a == 0.0 else b / a,
        )

    def __and__(self, other):
        return self._binop(other, lambda a, b: a & b)

    def __or__(self, other):
        return self._binop(other, lambda a, b: a | b)

    def __ror__(self, other):
        return self._binop(other, lambda a, b: b | a)

    def __rshift__(self, other):
        def shr(a: int, b: int) -> int:
            assert 0 <= b < 32, f"Out-of-range uint32 shift: {b}"
            return a >> b

        return self._binop(other, shr)

    def __lshift__(self, other):
        def shl(a: int, b: int) -> int:
            assert 0 <= b < 32, f"Out-of-range uint32 shift: {b}"
            return (a << b) & 0xFFFFFFFF

        return self._binop(other, shl)

    def __lt__(self, other):
        return self._binop(other, lambda a, b: a < b, out_dtype="bool")

    def __gt__(self, other):
        return self._binop(other, lambda a, b: a > b, out_dtype="bool")

    def __eq__(self, other):
        return self._binop(other, lambda a, b: a == b, out_dtype="bool")

    def __ne__(self, other):
        return self._binop(other, lambda a, b: a != b, out_dtype="bool")


def _make_fake_tl(pid0: int = 0, pid1: int = 0):
    def tl_abs(x: _FakeTensor) -> _FakeTensor:
        return _FakeTensor([abs(v) for v in x.data], x.shape, x.dtype)

    def tl_where(cond: _FakeTensor, x, y) -> _FakeTensor:
        tensors = [t for t in (cond, x, y) if isinstance(t, _FakeTensor)]
        out_shape = tuple(
            max(t.shape[dim] for t in tensors) for dim in range(len(cond.shape))
        )
        dtype = x.dtype if isinstance(x, _FakeTensor) else y.dtype
        cs = _expand_to_shape(cond, out_shape)
        xs = (
            _expand_to_shape(x, out_shape)
            if isinstance(x, _FakeTensor)
            else [_coerce(x, dtype)] * len(cs)
        )
        ys = (
            _expand_to_shape(y, out_shape)
            if isinstance(y, _FakeTensor)
            else [_coerce(y, dtype)] * len(cs)
        )
        return _FakeTensor(
            [a if c else b for c, a, b in zip(cs, xs, ys, strict=False)],
            out_shape,
            dtype,
        )

    def tl_reshape(x: _FakeTensor, shape: list[int]) -> _FakeTensor:
        return x.reshape(shape)

    def tl_max(x: _FakeTensor, axis: int, keep_dims: bool = False) -> _FakeTensor:
        assert axis == 2 and len(x.shape) == 3
        d0, d1, d2 = x.shape
        out = [
            max(x.data[(i * d1 + j) * d2 : (i * d1 + j + 1) * d2])
            for i in range(d0)
            for j in range(d1)
        ]
        shape = (d0, d1, 1) if keep_dims else (d0, d1)
        return _FakeTensor(out, shape, x.dtype)

    def tl_maximum(x: _FakeTensor, other) -> _FakeTensor:
        return x._binop(other, max)

    def tl_minimum(x: _FakeTensor, other) -> _FakeTensor:
        return x._binop(other, min)

    def tl_split(x: _FakeTensor) -> tuple[_FakeTensor, _FakeTensor]:
        assert x.shape[-1] == 2
        evens = x.data[0::2]
        odds = x.data[1::2]
        out_shape = x.shape[:-1]
        return _FakeTensor(evens, out_shape, x.dtype), _FakeTensor(
            odds, out_shape, x.dtype
        )

    def tl_program_id(axis: int) -> _FakeScalar:
        assert axis in (0, 1)
        return _FakeScalar(pid0 if axis == 0 else pid1)

    def tl_arange(start: int, end: int) -> _FakeTensor:
        return _FakeTensor(list(range(start, end)), (end - start,), "int64")

    def tl_load(addr: _FakeAddr, mask: _FakeTensor, other: float = 0.0) -> _FakeTensor:
        out = [
            addr.buf[addr.base_offset + off] if m else other
            for off, m in zip(addr.offsets.data, mask.data, strict=False)
        ]
        return _FakeTensor(out, addr.offsets.shape, addr.dtype)

    def tl_store(addr: _FakeAddr, val: _FakeTensor, mask: _FakeTensor) -> None:
        for off, v, m in zip(addr.offsets.data, val.data, mask.data, strict=False):
            if m:
                addr.buf[addr.base_offset + off] = _coerce(v, addr.dtype)

    def tl_inline_asm_elementwise(
        asm: str,
        constraints: str,
        args: list[_FakeTensor],
        dtype: str,
        is_pure: bool,
        pack: int,
    ) -> _FakeTensor:
        assert "cvt.rn.satfinite.e2m1x2.f32" in asm
        assert dtype == "uint8" and len(args) == 2
        hi_f32, lo_f32 = args
        packed = [
            _ref_rtne_e2m1(lo) | (_ref_rtne_e2m1(hi) << 4)
            for hi, lo in zip(hi_f32.data, lo_f32.data, strict=False)
        ]
        return _FakeTensor(packed, lo_f32.shape, "uint8")

    def core_sub(lhs, rhs: _FakeTensor, sanitize_overflow: bool = True) -> _FakeTensor:
        assert sanitize_overflow is False
        return rhs._binop(lhs, lambda right, left: (left - right) & 0xFFFFFFFF)

    return SimpleNamespace(
        constexpr=object,
        float32="float32",
        bfloat16="bfloat16",
        float16="float16",
        int64="int64",
        uint32="uint32",
        uint16="uint16",
        uint8="uint8",
        float8e4nv="float8e4nv",
        float8e5="float8e5",
        abs=tl_abs,
        where=tl_where,
        reshape=tl_reshape,
        max=tl_max,
        maximum=tl_maximum,
        minimum=tl_minimum,
        split=tl_split,
        program_id=tl_program_id,
        arange=tl_arange,
        load=tl_load,
        store=tl_store,
        inline_asm_elementwise=tl_inline_asm_elementwise,
        static_assert=lambda cond, msg="": (_ for _ in ()).throw(AssertionError(msg))
        if not cond
        else None,
        core=SimpleNamespace(sub=core_sub),
    )


def _load_compute_quant_and_scale():
    tl = _make_fake_tl()
    namespace = {"tl": tl, "MXFP_BLOCK_SIZE": 32}
    function(MXFP4_UTILS, "_get_max_quant_val", namespace)
    return function(MXFP4_UTILS, "_compute_quant_and_scale", namespace), tl


def _run_downcast_u16(
    rows: list[list[float]],
    rounding_mode: int = 0,
    block_out: int = 4,
    block_quant: int = 128,
) -> tuple[list[int], list[int]]:
    """Run the real ``_downcast_to_mxfp4_u16`` kernel on CPU over ``rows``."""
    outer_dim = len(rows)
    orig_quant_dim = len(rows[0])
    assert all(len(row) == orig_quant_dim for row in rows)
    padded_quant_dim = ((orig_quant_dim + 63) // 64) * 64
    s_groups_pairs = padded_quant_dim // 64

    src_buf = [v for row in rows for v in row]
    quant_buf = [0xAA] * (outer_dim * (orig_quant_dim // 2))
    # Col-major physical layout [S_pairs, N], matching _downcast_to_mxfp4_u16_wrapper.
    scale_buf = [0xBEEF] * (s_groups_pairs * outer_dim)

    grid_out = (outer_dim + block_out - 1) // block_out
    grid_quant = (padded_quant_dim + block_quant - 1) // block_quant

    for pid0 in range(grid_out):
        for pid1 in range(grid_quant):
            tl = _make_fake_tl(pid0=pid0, pid1=pid1)
            namespace = {"tl": tl, "MXFP_BLOCK_SIZE": 32}
            kernel = function(MXFP4_UTILS, "_downcast_to_mxfp4_u16", namespace)
            kernel(
                _FakePtr(quant_buf, "uint8"),
                orig_quant_dim // 2,
                1,
                _FakePtr(scale_buf, "uint16"),
                outer_dim,
                1,
                _FakePtr(src_buf, "float32"),
                orig_quant_dim,
                1,
                outer_dim,
                padded_quant_dim,
                orig_quant_dim,
                BLOCK_SIZE_OUT_DIM=block_out,
                BLOCK_SIZE_QUANT_DIM=block_quant,
                DEQUANT_SCALE_ROUNDING_MODE=rounding_mode,
            )

    # Unpack col-major [S_pairs, N] uint16 scales into logical row-major uint8 bytes.
    scale_bytes: list[int] = []
    for row_idx in range(outer_dim):
        for pair_idx in range(s_groups_pairs):
            packed_u16 = scale_buf[pair_idx * outer_dim + row_idx]
            scale_bytes.append(packed_u16 & 0xFF)
            scale_bytes.append((packed_u16 >> 8) & 0xFF)
    return quant_buf, scale_bytes


def _unpack_e2m1_codes(packed: list[int]) -> list[int]:
    return [code for byte in packed for code in (byte & 0x0F, byte >> 4)]


def _ref_rtne_e2m1(val: float) -> int:
    bits = _f32_to_u32(val)
    sign = (bits >> 31) & 1
    mag = abs(_u32_to_f32(bits))
    if mag >= 6.0:
        return (sign << 3) | 7
    best_code = 0
    best_dist = float("inf")
    for code, candidate in enumerate(E2M1_TABLE):
        dist = abs(mag - candidate)
        if dist < best_dist or (dist == best_dist and (code & 1) == 0):
            best_dist = dist
            best_code = code
    return (sign << 3) | best_code


def test_issue_399_midpoints_round_ties_to_even() -> None:
    """Issue #399 reproducer: exact midpoints must match ties-to-even (UINT16/.rn)."""
    compute_fn, tl = _load_compute_quant_and_scale()

    values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
    expected_pos = [0, 2, 2, 4, 4, 6, 6, 7]
    expected_neg = [code | 0x8 for code in expected_pos]

    row = values * 4 + [-v for v in values] * 4
    src = _FakeTensor(row, (1, 64), "float32")
    mask = _FakeTensor([True] * 64, (1, 64), "bool")

    packed, scales = compute_fn(src, mask, tl.uint8, DEQUANT_SCALE_ROUNDING_MODE=0)
    assert scales.data == [127, 127]
    assert _unpack_e2m1_codes(packed.data) == expected_pos * 4 + expected_neg * 4


@pytest.mark.parametrize("rounding_mode", [0, 1])
def test_midpoint_neighbors_zeros_and_subnormals_match_rtne(rounding_mode: int) -> None:
    """Verify exact RTNE across +/-1 ULP neighbors (including 0.25 + 1 ULP) and zeros."""
    compute_fn, tl = _load_compute_quant_and_scale()

    # Anchor each 32-element block with 6.0 so E8M0 scale is 1.0 in both modes.
    probe_anchors = [
        0.0,
        1e-45,
        1e-38,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
    ]
    probes: list[float] = []
    for base in probe_anchors:
        for sign in (1.0, -1.0):
            signed = math.copysign(base, sign)
            bits0 = _f32_to_u32(signed)
            for delta in (-1, 0, 1):
                bits = bits0 + delta
                if 0 <= bits <= 0xFFFFFFFF:
                    val = _u32_to_f32(bits)
                    if math.isfinite(val) and abs(val) <= 6.0:
                        probes.append(val)

    # Pack probes into 32-element groups, placing 6.0 at slot 0 so scale == 1.0.
    blocks: list[float] = []
    for start in range(0, len(probes), 31):
        chunk = probes[start : start + 31]
        blocks.extend([6.0] + chunk + [0.0] * (31 - len(chunk)))

    src = _FakeTensor(blocks, (1, len(blocks)), "float32")
    mask = _FakeTensor([True] * len(blocks), (1, len(blocks)), "bool")

    packed, scales = compute_fn(
        src, mask, tl.uint8, DEQUANT_SCALE_ROUNDING_MODE=rounding_mode
    )
    assert all(scale == 127 for scale in scales.data)

    got_codes = _unpack_e2m1_codes(packed.data)
    expected_codes = [_ref_rtne_e2m1(v) for v in blocks]
    assert got_codes == expected_codes


def test_padding_mask_and_saturation_with_round_down() -> None:
    """Masked padding lanes emit 0 and values > 6.0 saturate to +/-6.0."""
    compute_fn, tl = _load_compute_quant_and_scale()

    # With ROUND_DOWN, max_abs=11.0 gives dequant_scale=11/6 in [1, 2) -> scale=1.0,
    # so 11.0 and -11.0 enter E2M1 conversion > 6.0 and must saturate to codes 7 / 15.
    row = [11.0, -11.0, 2.5, -2.5, 0.25, -0.25] + [0.0] * 24 + [999.0, -999.0]
    valid = [True] * 30 + [False, False]

    src = _FakeTensor(row, (1, 32), "float32")
    mask = _FakeTensor(valid, (1, 32), "bool")

    packed, scales = compute_fn(src, mask, tl.uint8, DEQUANT_SCALE_ROUNDING_MODE=1)
    assert scales.data == [127]
    codes = _unpack_e2m1_codes(packed.data)
    assert codes[:6] == [7, 15, 4, 12, 0, 8]
    assert codes[30:] == [0, 0]


@pytest.mark.parametrize(
    ("exp", "expected_scale_byte", "expected_code_mag"),
    [
        (-40, 85, 6),  # 2^-40 / 2^-42 == 4.0 (code 6)
        (-32, 93, 6),  # 2^-32 / 2^-34 == 4.0 (code 6)
        (-120, 5, 6),  # 2^-120 / 2^-122 == 4.0 (code 6)
        (-126, 0, 4),  # 2^-126 / 2^-127 == 2.0 (code 4)
        (-128, 0, 1),  # 2^-128 / 2^-127 == 0.5 (code 1)
    ],
)
def test_issue_397_u16_small_magnitudes_match_u8(
    exp: int, expected_scale_byte: int, expected_code_mag: int
) -> None:
    """Issue #397: UINT16 kernel must preserve small magnitudes and match UINT8."""
    compute_fn, tl = _load_compute_quant_and_scale()
    row = [2.0**exp] * 32 + [-(2.0**exp)] * 32

    q_u16, s_u16 = _run_downcast_u16([row], rounding_mode=0)
    src = _FakeTensor(row, (1, 64), "float32")
    mask = _FakeTensor([True] * 64, (1, 64), "bool")
    q_u8, s_u8 = compute_fn(src, mask, tl.uint8, DEQUANT_SCALE_ROUNDING_MODE=0)

    assert s_u16 == [expected_scale_byte, expected_scale_byte]
    assert s_u8.data == s_u16
    assert q_u16 == q_u8.data

    codes = _unpack_e2m1_codes(q_u16)
    assert codes == [expected_code_mag] * 32 + [expected_code_mag | 0x8] * 32
    decoded = [
        (-1 if code & 8 else 1) * E2M1_TABLE[code & 7] * (2.0 ** (s_u16[i // 32] - 127))
        for i, code in enumerate(codes)
    ]
    assert decoded == row


def test_min_e8m0_scale_round_down_exact_decoding() -> None:
    """ROUND_DOWN with 6 * 2^-127 must use E8M0 byte 0 (2^-127) and decode exactly."""
    compute_fn, tl = _load_compute_quant_and_scale()
    val = 6.0 * (2.0**-127)
    row = [val] * 32 + [-val] * 32

    q_u16, s_u16 = _run_downcast_u16([row], rounding_mode=1)
    src = _FakeTensor(row, (1, 64), "float32")
    mask = _FakeTensor([True] * 64, (1, 64), "bool")
    q_u8, s_u8 = compute_fn(src, mask, tl.uint8, DEQUANT_SCALE_ROUNDING_MODE=1)

    assert s_u16 == [0, 0]
    assert s_u8.data == [0, 0]
    assert q_u16 == q_u8.data

    codes = _unpack_e2m1_codes(q_u16)
    assert codes == [7] * 32 + [15] * 32
    decoded = [
        (-1 if code & 8 else 1) * E2M1_TABLE[code & 7] * (2.0 ** (s_u16[i // 32] - 127))
        for i, code in enumerate(codes)
    ]
    assert decoded == row


@pytest.mark.parametrize("rounding_mode", [0, 1])
def test_u16_zero_and_padded_groups_match_u8(rounding_mode: int) -> None:
    """All-zero and padded groups emit zero scales and codes without div-by-zero."""
    compute_fn, tl = _load_compute_quant_and_scale()
    rows = [
        [0.0] * 64,
        [2.0**-40] * 32 + [0.0] * 32,
    ]
    q_u16, s_u16 = _run_downcast_u16(
        rows, rounding_mode=rounding_mode, block_out=4, block_quant=128
    )

    flat = [v for row in rows for v in row]
    src = _FakeTensor(flat, (2, 64), "float32")
    mask = _FakeTensor([True] * 128, (2, 64), "bool")
    q_u8, s_u8 = compute_fn(
        src, mask, tl.uint8, DEQUANT_SCALE_ROUNDING_MODE=rounding_mode
    )

    assert s_u16 == s_u8.data
    assert q_u16 == q_u8.data
    assert s_u16[:2] == [0, 0]
    assert s_u16[3] == 0
    assert q_u16[:32] == [0] * 32
