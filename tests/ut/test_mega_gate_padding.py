"""MegaGate contracts without importing torch, vLLM, or device libraries."""

from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch


class Mask:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return Mask(self.values[index])

    def __invert__(self):
        return Mask([not value for value in self.values])

    def __bool__(self):
        raise AssertionError("Routing must not read a device mask on the host")


@pytest.fixture
def gate_harness(modules):
    allocations = []
    calls = []
    context = SimpleNamespace(available=True, is_padding=None)

    def empty(shape, **kwargs):
        output = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(output)
        return output

    def gate(*args, **kwargs):
        calls.append((args, kwargs))

    def initialize():
        provider._bf16_mega_gate_impl = gate

    modules("torch", empty=empty, int64="int64", float32="float32")
    modules(
        "vllm.forward_context",
        is_forward_context_available=lambda: context.available,
        get_forward_context=lambda: context,
    )
    provider = modules(
        "vllm.utils.deep_gemm",
        _lazy_init=initialize,
        _bf16_mega_gate_impl=None,
    )
    implementation = load_patch(
        "vllm_sail/model_executor/layers/fused_moe/mega_gate.py"
    )
    return SimpleNamespace(
        run=implementation.bf16_mega_gate,
        context=context,
        provider=provider,
        allocations=allocations,
        calls=calls,
    )


@pytest.mark.parametrize(
    "padding,expected",
    [
        (None, None),
        ([False, False, False], [True, True, True]),
        ([True, True, True], [False, False, False]),
        ([False, True, False, True, True], [True, False, True]),
    ],
)
def test_mask_polarity_extent_and_output_contract(gate_harness, padding, expected):
    harness = gate_harness
    harness.context.is_padding = Mask(padding) if padding is not None else None
    hidden = SimpleNamespace(shape=(3, 5120), device="ppu")
    weight = object()
    optional = {
        name: object()
        for name in (
            "bias",
            "image_bias",
            "image_token_mask",
            "fix_routing_mask",
            "unmapped_topk_idx",
        )
    }
    outputs = harness.run(
        hidden,
        weight,
        6,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        ep_rank=7,
        **optional,
    )
    args, kwargs = harness.calls[0]
    assert args == (hidden, weight, 6)
    assert (
        kwargs["mask"].values if expected is not None else kwargs["mask"]
    ) == expected
    assert kwargs["out"] == tuple(harness.allocations)
    assert outputs == tuple(reversed(harness.allocations))
    assert [output.dtype for output in harness.allocations] == ["int64", "float32"]
    assert all(output.shape == (3, 6) for output in outputs)
    assert all(output.device == "ppu" for output in outputs)
    assert kwargs["ep_rank"] == 7
    assert kwargs["scoring_func"] == "sqrtsoftplus"
    assert kwargs["routed_scaling_factor"] == 1.5
    assert kwargs["num_shared_experts"] == 0
    assert kwargs["use_shared_as_routed"] is False
    for name, value in optional.items():
        assert kwargs[name] is value


def test_no_forward_context_and_missing_backend(gate_harness):
    harness = gate_harness
    harness.context.available = False
    harness.context.is_padding = Mask([True])
    hidden = SimpleNamespace(shape=(1, 5120), device="ppu")
    arguments = dict(scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, ep_rank=0)
    harness.run(hidden, object(), 6, **arguments)
    assert harness.calls[0][1]["mask"] is None
    harness.provider._lazy_init = lambda: None
    harness.provider._bf16_mega_gate_impl = None
    with pytest.raises(RuntimeError, match="requires DeepGEMM"):
        harness.run(hidden, object(), 6, **arguments)
