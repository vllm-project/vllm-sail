# SPDX-License-Identifier: Apache-2.0
"""Worked examples for the independent oracle; torch is an optional CPU extra."""

import math

import pytest

pytestmark = pytest.mark.requires_runtime

torch = pytest.importorskip("torch")

from tests.e2e.gdn._gdn_mtp_reference import gdn_mtp_reference  # noqa: E402
from tests.e2e.gdn._gdn_mtp_utils import assert_unchanged  # noqa: E402


def _inputs(*, lengths=(1,), indices=((1,),), accepted=(1,), state_dtype=None):
    tokens = sum(lengths)
    return dict(
        mixed_qkv=torch.zeros(tokens, 6, dtype=torch.bfloat16),
        a=torch.zeros(tokens, 1, dtype=torch.bfloat16),
        b=torch.zeros(tokens, 1, dtype=torch.bfloat16),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        state_indices=torch.tensor(indices, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        state=torch.zeros(5, 1, 2, 2, dtype=state_dtype or torch.float32),
        output_gate=torch.ones(tokens, 1, 2, dtype=torch.bfloat16),
        norm_weight=torch.ones(2),
        scale=1.0,
        norm_eps=0.0,
        output_gate_activation="sigmoid",
    )


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_one_hot_update_and_gated_norm_have_a_closed_form(activation):
    inp = _inputs()
    inp["mixed_qkv"][0] = torch.tensor([1, 0, 1, 0, 2, 4])
    inp["output_gate_activation"] = activation
    before = inp["state"].clone()
    actual = gdn_mtp_reference(**inp)

    # q=k=e0, beta=1/2, initial state=0: h=(1,2) outer normalized e0.
    expected_state = torch.tensor([[1.0, 0.0], [2.0, 0.0]]) / math.sqrt(1 + 1e-6)
    torch.testing.assert_close(inp["state"][1, 0], expected_state)
    gate = 1 / (1 + math.exp(-1))  # SiLU(1) and sigmoid(1) coincide.
    expected = (torch.tensor([1.0, 2.0]) / math.sqrt(2.5) * gate).to(torch.bfloat16)
    torch.testing.assert_close(actual[0, 0], expected, atol=0, rtol=0)
    torch.testing.assert_close(inp["state"][[0, 2, 3, 4]], before[[0, 2, 3, 4]])


def test_rewind_reads_selected_slot_and_saves_every_candidate():
    inp = _inputs(lengths=(3,), indices=((3, 1, 2),), accepted=(2,))
    inp["norm_eps"] = 1e-6
    inp["state"][1].fill_(8)
    inp["state"][2].fill_(100)
    inp["state"][3].fill_(-100)
    output = gdn_mtp_reference(**inp)
    # q=k=0, decay=1/2: slot 1 is the source, independent of slot order.
    assert torch.equal(inp["state"][3], torch.full_like(inp["state"][3], 4))
    assert torch.equal(inp["state"][1], torch.full_like(inp["state"][1], 2))
    assert torch.equal(inp["state"][2], torch.full_like(inp["state"][2], 1))
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("null_slot", [0, -1])
def test_null_source_produces_zero_output_without_any_state_write(null_slot):
    inp = _inputs(lengths=(2,), indices=((null_slot, 2),))
    inp["state"].fill_(7)
    before = inp["state"].clone()
    assert torch.count_nonzero(gdn_mtp_reference(**inp)) == 0
    assert torch.equal(inp["state"], before)


def test_null_destinations_do_not_stop_the_recurrence():
    inp = _inputs(lengths=(3,), indices=((1, 0, -1),))
    inp["norm_eps"] = 1e-6
    inp["state"].fill_(8)
    before = inp["state"].clone()
    gdn_mtp_reference(**inp)
    assert torch.equal(inp["state"][1], torch.full_like(inp["state"][1], 4))
    assert torch.equal(inp["state"][[0, 2, 3, 4]], before[[0, 2, 3, 4]])


def test_bf16_state_storage_does_not_round_the_running_fp32_state():
    inp = _inputs(lengths=(3,), indices=((1, 2, 3),), state_dtype=torch.bfloat16)
    inp["norm_eps"] = 1e-6
    inp["a"].fill_(0.75)
    inp["state"][1].fill_(1)
    gdn_mtp_reference(**inp)
    decay = 1 / (1 + math.exp(0.75))
    for step, slot in enumerate((1, 2, 3), start=1):
        expected = torch.full((1, 2, 2), decay**step).to(torch.bfloat16)
        assert torch.equal(inp["state"][slot], expected)


def test_zero_length_request_preserves_its_slots():
    inp = _inputs(lengths=(0, 1), indices=((3,), (1,)), accepted=(1, 1))
    inp["norm_eps"] = 1e-6
    inp["state"].fill_(8)
    before = inp["state"].clone()
    assert torch.count_nonzero(gdn_mtp_reference(**inp)) == 0
    assert torch.equal(inp["state"][1], before[1] / 2)
    assert torch.equal(inp["state"][3], before[3])


def test_output_is_rounded_to_bf16_before_rmsnorm():
    inp = _inputs()
    inp["mixed_qkv"][0] = torch.tensor([1, 0, 1, 0, 1, 2.5])
    inp["b"].fill_(-2)
    actual = gdn_mtp_reference(**inp)
    beta = 1 / (1 + math.exp(2))
    raw = torch.tensor([1.0, 2.5]) * beta / (1 + 1e-6)
    rounded = raw.to(torch.bfloat16).float()
    gate = 1 / (1 + math.exp(-1))
    expected = rounded / rounded.square().mean().sqrt() * inp["norm_weight"] * gate
    torch.testing.assert_close(
        actual[0, 0], expected.to(torch.bfloat16), atol=0, rtol=0
    )
    unrounded = raw / raw.square().mean().sqrt() * gate
    assert not torch.equal(expected.to(torch.bfloat16), unrounded.to(torch.bfloat16))


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_gate_zero_and_negative_values_distinguish_activations(activation):
    inp = _inputs()
    inp["mixed_qkv"][0] = torch.tensor([1, 0, 1, 0, 2, 4])
    inp["output_gate"][0, 0] = torch.tensor([-2.0, 0.0])
    inp["output_gate_activation"] = activation
    actual = gdn_mtp_reference(**inp)
    gates = torch.tensor([1 / (1 + math.exp(2)), 0.5])
    if activation == "silu":
        gates *= torch.tensor([-2.0, 0.0])
    expected = torch.tensor([1.0, 2.0]) / math.sqrt(2.5) * gates
    torch.testing.assert_close(
        actual[0, 0], expected.to(torch.bfloat16), atol=0, rtol=0
    )


def test_grouped_value_heads_use_the_corresponding_key_head():
    inp = _inputs()
    values = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]])
    inp["mixed_qkv"] = (
        torch.cat((torch.tensor([1.0, 0.0, 0.0, 1.0] * 2), values.flatten()))
        .reshape(1, -1)
        .to(torch.bfloat16)
    )
    inp["a"] = inp["b"] = torch.zeros(1, 4, dtype=torch.bfloat16)
    inp["A_log"] = inp["dt_bias"] = torch.zeros(4)
    inp["output_gate"] = torch.ones(1, 4, 2, dtype=torch.bfloat16)
    inp["state"] = torch.zeros(5, 4, 2, 2)
    actual = gdn_mtp_reference(**inp)
    expected_state = torch.zeros(4, 2, 2)
    expected_state[:2, :, 0] = values[:2] / (2 * math.sqrt(1 + 1e-6))
    expected_state[2:, :, 1] = values[2:] / (2 * math.sqrt(1 + 1e-6))
    torch.testing.assert_close(inp["state"][1], expected_state)
    raw = values / 2
    expected = raw / raw.square().mean(-1, keepdim=True).sqrt() / (1 + math.exp(-1))
    torch.testing.assert_close(actual[0], expected.to(torch.bfloat16), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bitwise_guards_detect_signed_zero_changes(dtype):
    before = torch.zeros(4, dtype=dtype)
    after = before.clone()
    assert_unchanged(after, before, "guard")
    after[0] = -0.0
    assert torch.equal(after, before)  # Numerical equality would miss the write.
    with pytest.raises(AssertionError, match="guard"):
        assert_unchanged(after, before, "guard")
