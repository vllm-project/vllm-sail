"""Exercise padding validity changes during PPU graph replay."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


def test_mega_gate_masks_nonfinite_padding_during_graph_replay(monkeypatch):
    import torch
    import vllm.forward_context as forward_context

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.model_executor.layers.fused_moe.mega_gate import bf16_mega_gate

    torch.manual_seed(0)
    hidden = torch.randn(64, 5120, device="cuda", dtype=torch.bfloat16) * 0.01
    weight = torch.randn(384, 5120, device="cuda", dtype=torch.bfloat16) * 0.01
    padding = torch.zeros(64, device="cuda", dtype=torch.bool)
    context = SimpleNamespace(is_padding=padding)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    arguments = dict(scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, ep_rank=0)
    expected_weights, expected_indices = bf16_mega_gate(hidden, weight, 6, **arguments)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, indices = bf16_mega_gate(hidden, weight, 6, **arguments)
    clean_hidden = hidden.clone()
    for active_rows in (32, 48, 0, 64):
        hidden.copy_(clean_hidden)
        hidden[active_rows:] = float("nan")
        padding.fill_(False)
        padding[active_rows:] = True
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            weights[:active_rows], expected_weights[:active_rows]
        )
        torch.testing.assert_close(
            indices[:active_rows], expected_indices[:active_rows]
        )
        assert torch.all(indices[active_rows:] == -1)
        assert torch.all(weights[active_rows:] == 0)
