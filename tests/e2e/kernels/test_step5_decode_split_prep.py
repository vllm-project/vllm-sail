# SPDX-License-Identifier: Apache-2.0
"""Decode split preparation must match the joint kernel, including CSA state."""

import pytest

pytestmark = pytest.mark.ppu


def _inputs(tokens, half, csa, device="cuda"):
    import torch

    dtype = torch.bfloat16
    torch.manual_seed(7)
    qkv = torch.randn(tokens, 18 * 192, dtype=dtype, device=device)
    qw = torch.randn(192, dtype=dtype, device=device) * 0.1
    kw = torch.randn(192, dtype=dtype, device=device) * 0.1
    phase = torch.randn(4096, 32, device=device)
    rope_main = torch.cat((phase.cos(), phase.sin()), dim=1).to(dtype)
    positions = torch.arange(tokens, device=device, dtype=torch.int64) % 4096
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    kc = torch.randn(4, 64, 1, 192, dtype=dtype, device=device)
    vc = torch.randn_like(kc)
    q = torch.randn(tokens, 4 * 256, dtype=dtype, device=device)
    k = torch.randn(tokens, 256, dtype=dtype, device=device)
    w = torch.randn(tokens, 4, device=device)
    iqw = torch.randn(256, dtype=dtype, device=device) * 0.1
    ikw = torch.randn(256, dtype=dtype, device=device) * 0.1
    ikb = torch.randn(256, dtype=dtype, device=device) * 0.1
    iphase = torch.randn(4096, 16, device=device)
    rope_idx = torch.cat((iphase.cos(), iphase.sin()), dim=1).to(dtype)
    main_inputs = (
        qkv,
        qw,
        kw,
        rope_main,
        slots,
        kc,
        vc,
        16,
        1,
        64,
        1e-6,
        1.0,
        None,
        None,
    )
    state = None
    if csa:
        from types import SimpleNamespace

        mean = torch.zeros(
            (1, 32, 256),
            dtype=torch.float16 if half else torch.uint8,
            device=device,
        )
        ring = torch.zeros(256, 512, dtype=dtype, device=device)
        bases = torch.zeros(tokens, device=device, dtype=torch.int64)
        state = SimpleNamespace(
            cache=SimpleNamespace(mean=mean),
            ring=SimpleNamespace(
                rows=ring,
                capacity=256,
                metadata=lambda: SimpleNamespace(base=bases),
            ),
            z=k,
            slots=slots,
        )
    return (q, k, w, iqw, ikw, ikb, rope_idx, positions, 1e-6, True), main_inputs, state


@pytest.mark.parametrize("tokens", [4, 32, 64, 128])
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("csa", [False, True])
def test_split_matches_joint(monkeypatch, tokens, half, csa):
    import torch
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu() or not current_platform.has_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", str(int(half)))
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia.ops import qkv_indexer_norm_rope as joint_mod

    args, main_inputs, state = _inputs(tokens, half, csa)
    kwargs = {}
    if state is not None:
        kwargs = {
            "decode_cache": state.cache,
            "decode_ring": state.ring,
            "decode_z": state.z,
            "decode_slots": state.slots,
        }
    monkeypatch.setenv("VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV", "0")
    if state is not None:
        # Warm the ring first: the CSA replay reads the previous seven rows,
        # which production decode wrote in earlier launches. On a fresh ring
        # a single batch would race its own replays in either path.
        joint_mod.qkv_indexer_norm_rope(*args, main_inputs=main_inputs, **kwargs)
        state.cache.mean.zero_()
    expected = joint_mod.qkv_indexer_norm_rope(*args, main_inputs=main_inputs, **kwargs)
    exp_cache = (
        main_inputs[5].clone().view(torch.uint8),
        main_inputs[6].clone().view(torch.uint8),
    )
    exp_state = None
    if state is not None:
        exp_state = (
            state.cache.mean.clone().view(torch.uint8),
            state.ring.rows.clone(),
        )
        # The CSA replay reads the previous seven ring rows, which production
        # decode wrote in earlier launches. Keep the reference run's ring so
        # the replay is deterministic; zero only the summary being compared.
        state.cache.mean.zero_()

    monkeypatch.setenv("VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV", "1")
    from vllm_sail.models.step5.ops import decode_split_prep as split_mod

    calls = []
    original_split = split_mod._split_prepare

    def spy(*a, **kw):
        calls.append(1)
        return original_split(*a, **kw)

    monkeypatch.setattr(split_mod, "_split_prepare", spy)
    actual = joint_mod.qkv_indexer_norm_rope(*args, main_inputs=main_inputs, **kwargs)
    assert calls, "decode shape must take the split path"

    assert actual[0].dtype == (torch.float16 if half else torch.float8_e4m3fn)
    for got, want in zip(actual[:3], expected[:3], strict=True):
        torch.testing.assert_close(
            got.view(torch.uint8), want.view(torch.uint8), atol=0, rtol=0
        )
    # The main op and the joint main half round FP32 reductions differently;
    # the microbench bound is a few bf16 ulp (normwise rel <= 1.72e-5).
    torch.testing.assert_close(actual[3], expected[3], atol=1e-3, rtol=1.6e-2)
    torch.testing.assert_close(actual[4], expected[4], atol=0, rtol=0)
    torch.testing.assert_close(actual[5], expected[5], atol=0, rtol=0)
    torch.testing.assert_close(
        main_inputs[5].view(torch.uint8), exp_cache[0], atol=0, rtol=0
    )
    torch.testing.assert_close(
        main_inputs[6].view(torch.uint8), exp_cache[1], atol=0, rtol=0
    )
    if state is not None:
        torch.testing.assert_close(
            state.cache.mean.view(torch.uint8), exp_state[0], atol=0, rtol=0
        )
        torch.testing.assert_close(state.ring.rows, exp_state[1], atol=0, rtol=0)


@pytest.mark.parametrize("half", [False, True])
def test_split_graph_replay_with_csa(monkeypatch, half):
    import torch
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu() or not current_platform.has_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", str(int(half)))
    monkeypatch.setenv("VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV", "1")
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia.ops import qkv_indexer_norm_rope as joint_mod

    args, main_inputs, state = _inputs(64, half, True)
    kwargs = {
        "decode_cache": state.cache,
        "decode_ring": state.ring,
        "decode_z": state.z,
        "decode_slots": state.slots,
    }

    def prepare():
        return joint_mod.qkv_indexer_norm_rope(*args, main_inputs=main_inputs, **kwargs)

    prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = prepare()
    state.cache.mean.zero_()
    state.ring.rows.zero_()
    main_inputs[5].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert out[0].dtype == (torch.float16 if half else torch.float8_e4m3fn)
    assert torch.count_nonzero(state.ring.rows[:64]) > 0
    assert torch.count_nonzero(state.cache.mean.view(torch.uint8)) > 0
    assert torch.count_nonzero(main_inputs[5].view(torch.uint8)) > 0


def test_large_shape_keeps_joint(monkeypatch):
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu() or not current_platform.has_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    monkeypatch.setenv("VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV", "1")
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia.ops import qkv_indexer_norm_rope as joint_mod

    from vllm_sail.models.step5.ops import decode_split_prep as split_mod

    args, main_inputs, _ = _inputs(256, False, False)
    calls = []
    monkeypatch.setattr(split_mod, "_split_prepare", lambda *a, **kw: calls.append(1))
    joint_mod.qkv_indexer_norm_rope(*args, main_inputs=main_inputs)
    assert not calls
