# SPDX-License-Identifier: Apache-2.0
"""Offline guard for the V4.1 paged KV record layout and software gather.

The 584B V4 record is planar (a page holds its whole value plane, then its whole
scale plane); the 528B MXFP8 and 288B NVFP4 V4.1 records are interleaved (each
token's scales follow its own values). Reading an interleaved record with the
planar formula does not raise -- it silently misreads every token, which shows up
as degenerate generation rather than a crash. These tests pin the address
arithmetic, the launch geometry and the e2m1 decoder without torch, vLLM or a
device.

Writer contract: vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py
``_rope_quant_insert_kernel`` (584B), ``_rope_quant_insert_mxfp8_kernel`` (528B)
and ``_rope_quant_insert_nvfp4_kernel`` (288B).
"""

import ast
import types

import pytest

from tests.support.source import ROOT, definition, function

GATHER = "vllm_sail/models/deepseek_v41/ops/cache_gather.py"
PLANAR = "vllm_sail/models/deepseek_v4/ops/cache.py"
WIDTHS = [584, 528, 288]
BLOCK_SIZES = [16, 32, 64]


def _layout():
    return function(GATHER, "record_layout", {})


def _offsets():
    return function(GATHER, "token_offsets", {})


def writer_offsets(record_bytes, block_size, t):
    """Independent transcription of the upstream writer's address arithmetic."""
    if record_bytes == 584:
        # values = page + t * 576; scales = page + B * 576 + t * 8
        return (t * 576, block_size * 576 + t * 8)
    if record_bytes == 528:
        # values = page + t * 528; scales = values + 512
        return (t * 528, t * 528 + 512)
    if record_bytes == 288:
        # values = page + t * 288; scales = values + 256
        return (t * 288, t * 288 + 256)
    raise AssertionError(f"unknown record width {record_bytes}")


@pytest.mark.parametrize("record_bytes", WIDTHS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_reader_addresses_match_writer_for_every_token(record_bytes, block_size):
    """reader == writer for every token of every page, not just the first."""
    layout, offsets = _layout(), _offsets()
    contract = layout(record_bytes)
    for t in range(block_size):
        assert offsets(contract, block_size, t) == writer_offsets(
            record_bytes, block_size, t
        ), f"record={record_bytes} B={block_size} t={t}"


@pytest.mark.parametrize("record_bytes", WIDTHS)
def test_record_geometry_matches_the_published_contract(record_bytes):
    contract = _layout()(record_bytes)
    expected = {
        584: (True, 576, None, 8, 64, 7),
        528: (False, 528, 512, 16, 32, 16),
        288: (False, 288, 256, 32, 16, 32),
    }[record_bytes]
    assert (
        contract["planar"],
        contract["token_stride"],
        contract["scale_offset"],
        contract["scale_bytes"],
        contract["quant_block"],
        contract["num_scales"],
    ) == expected
    assert contract["head_dim"] == 512
    # The planar V4 record keeps 64 RoPE dims in bf16, outside the scaled region.
    scaled_dims = 448 if contract["planar"] else 512
    assert contract["num_scales"] * contract["quant_block"] == scaled_dims
    if not contract["planar"]:
        stride = contract["token_stride"]
        assert contract["scale_offset"] + contract["scale_bytes"] == stride


def test_unknown_record_width_is_rejected():
    with pytest.raises(ValueError, match="unsupported paged KV record width"):
        _layout()(132)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_planar_formula_on_528_is_the_regression(block_size):
    """The guard has teeth: the shipped #214 formula matches almost nothing.

    #214 set token_data_size=512 for the 528B record and kept the planar scale
    base ``B * token_data_size + t * scale_dim``. Data addresses then coincide
    only at t=0 and scale addresses only at t=B-1.
    """
    buggy_data = [t for t in range(block_size) if t * 512 == t * 528]
    buggy_scale = [
        t for t in range(block_size) if block_size * 512 + t * 16 == t * 528 + 512
    ]
    assert buggy_data == [0]
    assert buggy_scale == [block_size - 1]
    fixed = _offsets()(_layout()(528), block_size, 0)
    assert fixed == writer_offsets(528, block_size, 0)


def _kernel_source(path, name):
    """Unparsed kernel body without its docstring: prose may name fp8 dtypes."""
    node = definition(path, name)
    stmts = node.body
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]
    return ast.unparse(ast.Module(body=stmts, type_ignores=[]))


@pytest.mark.parametrize(
    "kernel_name", ["_gather_mxfp8_sm80_kernel", "_gather_nvfp4_sm80_kernel"]
)
def test_kernels_use_the_interleaved_address_formula(kernel_name):
    body = _kernel_source(ROOT / GATHER, kernel_name)
    assert "token_ptr = page + pos_in_block * token_stride" in body
    assert "tl.load(token_ptr + scale_offset + s)" in body
    # The planar scale base must not reappear in an interleaved kernel.
    assert "cache_block_size * token_data_size" not in body
    assert "token_data_size" not in body


@pytest.mark.parametrize(
    "kernel_name", ["_gather_mxfp8_sm80_kernel", "_gather_nvfp4_sm80_kernel"]
)
def test_kernels_avoid_hardware_fp8_and_unproven_casts(kernel_name):
    body = _kernel_source(ROOT / GATHER, kernel_name)
    assert "float8e4nv" not in body
    assert "float8e4b8" not in body
    assert "tl.join" not in body


def test_planar_kernel_keeps_its_own_formula():
    body = _kernel_source(ROOT / PLANAR, "_dequantize_and_gather_k_kernel_sm80")
    assert "token_data_ptr = cache_block_ptr + pos_in_block * token_data_size" in body
    assert "cache_block_size * token_data_size" in body


@pytest.mark.parametrize("record_bytes", [528, 288])
def test_launch_passes_the_layout_as_constexpr(record_bytes):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    launch = function(
        GATHER,
        "_launch",
        {"record_layout": _layout(), "_NUM_WORKERS": 128},
    )
    out = types.SimpleNamespace(stride=lambda dim: (102400, 1024)[dim])
    cache = types.SimpleNamespace(
        shape=(4, 64, record_bytes), stride=lambda dim: 64 * record_bytes + 512
    )
    seq_lens = types.SimpleNamespace(shape=(2,))
    table = types.SimpleNamespace(shape=(2, 4))
    kernel = Kernel()
    launch(kernel, record_bytes, out, cache, seq_lens, None, table, 64, 3)
    grid, args, kwargs = calls.pop()
    contract = _layout()(record_bytes)
    assert grid == (2, 128)
    assert args == (out, 102400, 1024, cache, seq_lens, table, 3, None)
    assert kwargs == dict(
        max_blocks_per_seq=4,
        cache_block_size=64,
        block_stride=64 * record_bytes + 512,
        token_stride=contract["token_stride"],
        scale_offset=contract["scale_offset"],
        head_dim=512,
        scale_dim=contract["num_scales"],
        quant_block=contract["quant_block"],
    )
    assert not calls
    # A mismatched record must not be launched with another width's geometry.
    cache.shape = (4, 64, 584 if record_bytes == 528 else 528)
    with pytest.raises(ValueError, match="byte record"):
        launch(kernel, record_bytes, out, cache, seq_lens, None, table, 64, 3)
    assert not calls


def test_launch_refuses_the_planar_record():
    launch = function(
        GATHER, "_launch", {"record_layout": _layout(), "_NUM_WORKERS": 128}
    )
    cache = types.SimpleNamespace(shape=(4, 64, 584), stride=lambda dim: 0)
    with pytest.raises(ValueError, match="planar V4 record"):
        launch(None, 584, None, cache, None, None, None, 64, 0)


class _Scalar:
    """Minimal scalar shim so a Triton body can be evaluated on the CPU."""

    def __init__(self, value):
        self.value = value

    def to(self, dtype):
        return _Scalar(float(self.value))

    def __rshift__(self, other):
        return _Scalar(self.value >> other)

    def __and__(self, other):
        return _Scalar(self.value & other)

    def __sub__(self, other):
        return _Scalar(self.value - _unwrap(other))

    def __add__(self, other):
        return _Scalar(self.value + _unwrap(other))

    __radd__ = __add__

    def __mul__(self, other):
        return _Scalar(self.value * _unwrap(other))

    __rmul__ = __mul__

    def __neg__(self):
        return _Scalar(-self.value)

    def __ne__(self, other):
        return _Scalar(self.value != _unwrap(other))


def _unwrap(value):
    return value.value if isinstance(value, _Scalar) else value


def _fake_tl():
    return types.SimpleNamespace(
        float32="fp32",
        exp2=lambda x: _Scalar(2.0 ** _unwrap(x)),
        where=lambda cond, a, b: _Scalar(_unwrap(a) if _unwrap(cond) else _unwrap(b)),
    )


def test_e2m1_decoder_matches_the_format_truth_table():
    decode = function(GATHER, "_decode_e2m1", {"tl": _fake_tl()})
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for nibble in range(16):
        expected = magnitudes[nibble & 7]
        if nibble & 8:
            expected = -expected
        assert _unwrap(decode(_Scalar(nibble))) == pytest.approx(expected), (
            f"nibble {nibble:#x}"
        )


@pytest.mark.parametrize("record_bytes", [528, 288])
def test_scale_grouping_covers_each_dim_exactly_once(record_bytes):
    """The kernel's (num_scales, quant_block) reshape matches the writer tiling."""
    contract = _layout()(record_bytes)
    groups = [d // contract["quant_block"] for d in range(contract["head_dim"])]
    assert groups == sorted(groups)
    assert set(groups) == set(range(contract["num_scales"]))
    if record_bytes == 288:
        # Byte b packs dims 2b (low nibble) and 2b+1; both share scale b // 8.
        for b in range(256):
            assert (2 * b) // 16 == (2 * b + 1) // 16 == b // 8
