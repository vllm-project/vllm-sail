# SPDX-License-Identifier: Apache-2.0
"""Optimize Step5 independent and split-K routing on M890."""

from vllm_sail import envs

_installed = False


def _install_independent_topk() -> None:
    import functools

    import torch
    from vllm import envs as vllm_envs
    from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router

    from vllm_sail.models.step5.ops.fused_router import _reduce_router_kernel
    from vllm_sail.patch.utils import patch

    original = fused_topk_bias_router.vllm_topk_sigmoid

    @patch(
        fused_topk_bias_router.__name__,
        "vllm_topk_sigmoid",
        reason="Reduce latency of independent Step5 sigmoid Top8 at small M on M890.",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="Native biased sigmoid TopK matches small-M Triton latency.",
    )
    @functools.wraps(original)
    def topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
    ):
        rows = gating_output.shape[0]
        bias = e_score_correction_bias
        if (
            vllm_envs.VLLM_BATCH_INVARIANT
            or not 0 < rows <= 512
            or gating_output.shape != (rows, 352)
            or topk_weights.shape != (rows, 8)
            or gating_output.dtype != torch.float32
            or not gating_output.is_cuda
            or bias is None
            or bias.shape != (352,)
            or bias.dtype != torch.float32
            or not all(
                t.is_contiguous()
                for t in (
                    gating_output,
                    bias,
                    topk_weights,
                    topk_indices,
                    token_expert_indices,
                )
            )
            or topk_indices.dtype not in (torch.int32, torch.int64)
        ):
            return original(
                topk_weights,
                topk_indices,
                token_expert_indices,
                gating_output,
                renormalize,
                bias,
                routed_scaling_factor,
            )
        sms = torch.cuda.get_device_properties(
            gating_output.device
        ).multi_processor_count
        warps = 8 if rows <= sms else 4 if rows <= 4 * sms else 1
        padding = fused_topk_bias_router._get_padding_mask(rows)
        if padding is not None:
            padding = padding.contiguous()
        _reduce_router_kernel[(rows,)](
            gating_output,
            bias,
            topk_weights,
            topk_indices,
            padding,
            352,
            0,
            SPLIT_K=1,
            RENORMALIZE=renormalize,
            SCALING=routed_scaling_factor,
            IDX64=topk_indices.dtype == torch.int64,
            PARALLEL_TOPK=False,
            HAS_PADDING=padding is not None,
            source_rows_ptr=token_expert_indices,
            num_warps=warps,
            num_stages=2,
        )
        return topk_weights, topk_indices


def install() -> None:
    global _installed
    if _installed:
        return

    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or not current_platform.is_device_capability(89):
        return

    _install_independent_topk()
    fuse_router = envs.VLLM_SAIL_STEP5_FUSED_ROUTER
    optimize_dispatch = (
        envs.VLLM_SAIL_STEP5_MEGA_MOE_OVERLAP or envs.VLLM_SAIL_MEGA_MOE_ULTRA_EP
    )
    if not (fuse_router or optimize_dispatch):
        _installed = True
        return

    import functools
    from weakref import ref

    import torch
    from vllm import envs as vllm_envs
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.models.step5.nvidia import model

    from vllm_sail.models.step5.mega_moe import PPUStep5MegaMoEExperts
    from vllm_sail.models.step5.ops.fused_router import register_ops
    from vllm_sail.patch.utils import patch

    meta = dict(
        reason="Fuse Step5 MegaMoE split-K router reduction and sigmoid TopK on M890.",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="Step5 MegaMoE exposes a fused gate and TopK router entry point.",
    )
    if fuse_router:
        register_ops()
    original_forward = model.Step5MoE.forward
    original_load = model.Step5ForCausalLM.load_weights
    original_init = model.Step5MoE.__init__

    if optimize_dispatch:
        from vllm_sail.models.step5 import mega_moe_dispatch

        mega_moe_dispatch.register_ops()

        @patch(model.__name__, "Step5MoE.__init__", **meta)
        @functools.wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # PPU MODIFICATION: begin — the opaque dispatch resolves the current layer.
            experts = getattr(self.experts, "routed_experts", None)
            if isinstance(experts, PPUStep5MegaMoEExperts):
                experts._sail_shared_expert = ref(self.share_expert)
                experts._sail_sequence_parallel = self.use_sequence_parallel
            # PPU MODIFICATION: end

    @patch(model.__name__, "Step5ForCausalLM.load_weights", **meta)
    @functools.wraps(original_load)
    def load_weights(self, weights):
        loaded = original_load(self, weights)
        if not fuse_router:
            return loaded
        for layer in self.modules():
            if not isinstance(layer, model.Step5MoE):
                continue
            experts = getattr(layer.experts, "routed_experts", None)
            if not isinstance(experts, PPUStep5MegaMoEExperts):
                continue
            weight = layer.gate.weight.detach()
            bf16_weight = None
            if weight.shape == (352, 4096) and weight.dtype == torch.float32:
                candidate = weight.to(torch.bfloat16).contiguous()
                if torch.equal(weight, candidate.float()):
                    bf16_weight = candidate
            layer.register_buffer(
                "_sail_router_bf16_weight", bf16_weight, persistent=False
            )
        return loaded

    @patch(model.__name__, "Step5MoE.forward", **meta)
    @functools.wraps(original_forward)
    def forward(self, hidden_states, padding=None):
        experts = self.experts
        routed_experts = getattr(experts, "routed_experts", None)
        if not isinstance(routed_experts, PPUStep5MegaMoEExperts):
            return original_forward(self, hidden_states, padding)
        bf16_weight = getattr(self, "_sail_router_bf16_weight", None)
        router = experts.router
        fused = (
            bf16_weight is not None
            and hidden_states.dtype == torch.bfloat16
            and bf16_weight.device == hidden_states.device
            and not self.enable_eplb
            and not vllm_envs.VLLM_BATCH_INVARIANT
            and router.scoring_func == "sigmoid"
            and router.top_k == 8
            and router.capture_fn is None
        )
        dispatch_enabled = optimize_dispatch and (
            routed_experts.ultra_ep is not None
            or mega_moe_dispatch.can_overlap(self, hidden_states)
        )
        if not fused and not dispatch_enabled:
            return original_forward(self, hidden_states, padding)

        # PPU MODIFICATION: begin — bypass materialized logits for the MegaMoE path.
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if not dispatch_enabled:
            shared = self.share_expert(hidden_states)
        if fused:
            # Under sequence parallel ``padding`` is the rank-local mask; pass it
            # explicitly so the fused op masks the sharded rows correctly instead
            # of reading the global forward-context prefix. Non-SP passes None and
            # the op reads that global prefix itself.
            weights, ids = torch.ops.vllm.ppu_step5_fused_router(
                hidden_states,
                self.gate.weight,
                bf16_weight,
                self.router_bias,
                router.renormalize,
                router.routed_scaling_factor,
                padding,
            )
        else:
            logits, _ = self.gate(hidden_states)
            weights, ids = router.select_experts(
                hidden_states, logits, topk_indices_dtype=torch.int64
            )
        if (
            padding is None
            and not experts.sequence_parallel
            and vllm_envs.VLLM_MOE_SKIP_PADDING
            and is_forward_context_available()
        ):
            mask = get_forward_context().is_padding
            padding = mask[: hidden_states.shape[0]] if mask is not None else None
        # The fused router already masks padding rows (ids=-1, weights=0);
        # passing padding downstream would re-mask in MegaMoE (4 extra kernels).
        masked_padding = None if fused else padding
        if dispatch_enabled:
            return mega_moe_dispatch.dispatch(
                self, hidden_states, weights, ids, masked_padding
            )
        return shared, routed_experts(hidden_states, weights, ids, masked_padding)
        # PPU MODIFICATION: end

    _installed = True


install()
