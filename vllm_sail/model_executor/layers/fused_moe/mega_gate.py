"""PPU MegaGate routing with graph-safe token validity masking."""

from __future__ import annotations


def bf16_mega_gate(
    x,
    weight,
    num_topk,
    *,
    scoring_func,
    routed_scaling_factor,
    ep_rank,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    unmapped_topk_idx=None,
):
    import torch
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )
    from vllm.utils import deep_gemm

    deep_gemm._lazy_init()
    gate = deep_gemm._bf16_mega_gate_impl
    if gate is None:
        raise RuntimeError("PPU MegaMoE requires DeepGEMM with bf16_mega_gate.")

    padding = (
        get_forward_context().is_padding if is_forward_context_available() else None
    )
    token_mask = ~padding[: x.shape[0]] if padding is not None else None
    shape = (x.shape[0], num_topk)
    indices = torch.empty(shape, dtype=torch.int64, device=x.device)
    weights = torch.empty(shape, dtype=torch.float32, device=x.device)
    gate(
        x,
        weight,
        num_topk,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        ep_rank=ep_rank,
        scoring_func=scoring_func,
        mask=token_mask,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        fix_routing_mask=fix_routing_mask,
        unmapped_topk_idx=unmapped_topk_idx,
        out=(indices, weights),
    )
    return weights, indices
