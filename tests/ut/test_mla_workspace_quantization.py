# SPDX-License-Identifier: Apache-2.0
"""MLA profiling reserves workspace and honors the selected output quantizer."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/attention/mla_attention.py"
UPSTREAM = "vllm/model_executor/layers/attention/mla_attention.py"


class Tensor:
    def __init__(self, shape=(4, 8), *, dtype="bf16", root=None, events=None):
        self.shape = shape
        self.dtype = dtype
        self.device = "host-double"
        self.root = self if root is None else root
        self.events = events

    def __getitem__(self, index):
        first = index[0] if isinstance(index, tuple) else index
        start, stop, step = first.indices(self.shape[0])
        assert step == 1
        return Tensor(
            (stop - start, *self.shape[1:]),
            dtype=self.dtype,
            root=self.root,
            events=self.events,
        )

    def size(self, axis):
        return self.shape[axis]

    def fill_(self, value):
        self.events.append(("fill", self.root, value))
        return self

    def copy_(self, value):
        self.events.append(("copy", self.root, value))
        return self


def _make_case(ppu, quantized):
    events, allocated = [], []
    output = Tensor(dtype="fp8" if quantized else "bf16", events=events)

    def empty(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        result = Tensor(shape, dtype=kwargs["dtype"], events=events)
        allocated.append(result)
        events.append(("empty", result))
        return result

    namespace = {
        "torch": NS(empty=empty),
        "current_platform": NS(is_ppu=lambda: ppu),
        "_detect_output_quant_key": lambda *args: "static" if quantized else None,
    }
    layer = NS(
        num_heads=2,
        v_head_dim=4,
        qk_nope_head_dim=2,
        chunked_prefill_workspace_size=128,
        impl=NS(reserve_workspace=lambda: events.append(("reserve",))),
    )
    return layer, namespace, output, events, allocated


def check_profile(path, name, ppu, quantized, copied):
    layer, namespace, output, events, allocated = _make_case(ppu, quantized)
    tensor = Tensor()
    result = function(path, name, namespace)(
        layer, tensor, tensor, tensor, tensor, None, output
    )
    assert result is output
    assert [e[0] for e in events][-2:] == ["reserve", "fill"]
    assert events[-1] == ("fill", output, 0)
    assert sum(e[0] == "reserve" for e in events) == 1
    # Keep the existing PPU scratch-allocation workaround. The provider has no
    # such workaround, but both bodies must still reserve backend workspace.
    assert len(allocated) == int(quantized) + int(not (copied and ppu))
    if not (copied and ppu):
        assert allocated[-1].shape == (128, 2, 6)


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_profile_reserves_before_zero_fill(ppu, quantized):
    check_profile(PATCH, "_forward_impl_body", ppu, quantized, True)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_actual_profile_workspace_lifecycle(
    upstream_source_root, copied, ppu, quantized
):
    path = PATCH if copied else upstream_source_root / UPSTREAM
    name = "_forward_impl_body" if copied else "MLAAttention.forward_impl"
    check_profile(path, name, ppu, quantized, copied)


def check_quantization(path, name, aiter):
    layer, namespace, output, events, allocated = _make_case(False, True)
    scale, fp8_data = object(), object()

    def per_tensor(*args):
        events.append(("aiter", *args))

    def fallback(actual, used_scale):
        events.append(("fallback", actual, used_scale))
        return fp8_data, object()

    layer.impl = NS(
        is_sparse=False,
        dcp_world_size=1,
        forward_mha=lambda *args, **kwargs: events.append(("mha", kwargs["output"])),
    )
    layer.kv_cache_dtype = "auto"
    layer.use_pcp = False
    layer.prefill_backend = None
    layer._k_scale = object()
    layer._aiter_output_quant = aiter
    layer._quant_fp8_op = fallback
    namespace.update(
        torch=NS(
            empty=namespace["torch"].empty,
            ops=NS(vllm=NS(rocm_aiter_per_tensor_quant=per_tensor)),
        ),
        is_quantized_kv_cache=lambda dtype: False,
        kNvfp4Dynamic="nvfp4",
        kFp8Dynamic128Sym="fp8-128",
        kFp8Dynamic64Sym="fp8-64",
        kFp8StaticTensorSym="static",
    )
    metadata = NS(
        num_actual_tokens=2, num_decodes=0, num_prefills=1, num_decode_tokens=0
    )
    q = Tensor((4, 2, 4))
    result = function(path, name, namespace)(
        layer, q, q, q, q, metadata, output, output_scale=scale
    )
    assert result is output
    assert [e[0] for e in events] == (
        ["empty", "mha", "aiter"] if aiter else ["empty", "mha", "fallback", "copy"]
    )
    quant_event = events[2]
    actual = quant_event[2] if aiter else quant_event[1]
    assert actual.root is allocated[0] and actual.shape == (2, 8)
    if aiter:
        destination = quant_event[1]
        assert destination.root is output and destination.shape == (2, 8)
        assert quant_event[3:] == (scale, False)
    else:
        assert quant_event[2] is scale
        assert events[3] == ("copy", output, fp8_data)


@pytest.mark.parametrize("aiter", [False, True])
def test_selected_per_tensor_quantizer_receives_actual_rows(aiter):
    check_quantization(PATCH, "_forward_impl_body", aiter)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("aiter", [False, True])
def test_actual_per_tensor_quantizer_contract(upstream_source_root, copied, aiter):
    path = PATCH if copied else upstream_source_root / UPSTREAM
    name = "_forward_impl_body" if copied else "MLAAttention.forward_impl"
    check_quantization(path, name, aiter)
