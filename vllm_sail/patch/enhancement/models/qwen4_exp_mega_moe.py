# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp MegaMoE: route PPU MoE layers through DeepGEMM MegaMoE.

Upstream Qwen4Exp has no ``deep_gemm_mega_moe`` branch (only DeepSeek V4 /
V4.1, Kimi K3 and MiniMax M3 instantiate ``DeepGemmMegaMoEExperts``), so this
patch installs one on ``Qwen4ExpSparseMoeBlock`` on PPU:

* ``__init__`` — under ``--moe-backend deep_gemm_moe`` with EP on, replaces
  the ``FusedMoEFactory`` runner with the (already PPU-redirected)
  ``DeepGemmMegaMoEExperts`` from ``vllm_sail.patch.enhancement.mega_moe``.
  The SAIL experts bank matches the qwen4-max MXFP4 checkpoint contract
  exactly: U8 packed weights plus 1x32 UE8M0 scales (verified on m890-57,
  2026-11-23). The shared expert (with its sigmoid gate) stays outside the
  routed kernel and is added by the experts' ``forward``.
* ``forward`` — computes softmax+renormalize routing with the standard
  ``FusedTopKRouter`` (``bf16_mega_gate`` supports sqrtsoftplus/sigmoid
  only) and calls the experts with explicit ``topk_weights``/``topk_ids``.

Expert checkpoint weights (``mlp.experts.<id>.{gate,up,down}_proj``) reach
the experts module through ``AutoWeightsLoader`` recursion, which delegates
to any submodule with a ``load_weights``; the mega experts only expose the
per-parameter MoE loader, so the patch attaches a small
``load_weights`` that maps ``<id>.gate_proj.weight`` onto
``w13_/w2_`` shards (the same mapping DeepSeek V4 builds by hand).

v1 rejections: EPLB (W4A8 rebalancing changes the kernel's captured inverse
weight scale), MoE/HC sequence parallelism (the HC-SP path reads
``experts.moe_config``, which the mega experts do not expose — the kernel's
routed output is already complete per rank), TP > 1, and speculative
decoding. ``mtp.py`` imports the class by value, but the class-attribute
patches mutate the shared class object, so the draft model sees them
automatically.
"""

from __future__ import annotations

import torch
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MegaMoEActivation,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.qwen4_exp.common.hyperconnection import (
    is_hc_sequence_parallel_enabled,
)
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpSparseMoeBlock
from vllm.platforms import current_platform

from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.34.0,<0.35.0"

_MODULE = "vllm.models.qwen4_exp.nvidia.model"
_INIT_TARGET = f"{_MODULE}.Qwen4ExpSparseMoeBlock.__init__"
_FORWARD_TARGET = f"{_MODULE}.Qwen4ExpSparseMoeBlock.forward"

# Checkpoint projection name -> (shard id, parameter stem).
_PROJ_TO_SHARD = {
    "gate_proj": ("w1", "w13"),
    "up_proj": ("w3", "w13"),
    "down_proj": ("w2", "w2"),
}


def _make_expert_load_weights(experts):
    """Map ``<expert_id>.<proj>.<weight|weight_scale>`` onto the mega banks.

    ``AutoWeightsLoader`` recursion reaches the experts module with names
    relative to it (the ``experts.`` prefix already consumed), and delegates
    to this attached ``load_weights`` instead of walking the raw parameters.
    """

    def load_weights(weights):
        for name, loaded_weight in weights:
            parts = name.split(".")
            if len(parts) != 3 or parts[1] not in _PROJ_TO_SHARD:
                # Non per-expert tensors never reach the experts module.
                continue
            expert_id_s, proj, leaf = parts
            if not expert_id_s.isdigit() or leaf not in (
                "weight",
                "weight_scale",
            ):
                continue
            shard_id, stem = _PROJ_TO_SHARD[proj]
            # E8M0 scales stored as float8_e8m0fnu must be reinterpreted as
            # raw exponent bytes; a numeric copy_ would destroy them.
            if (
                leaf == "weight_scale"
                and loaded_weight.dtype == torch.float8_e8m0fnu
            ):
                loaded_weight = loaded_weight.view(torch.uint8)
            param_name = f"{stem}_{leaf}"
            param = getattr(experts, param_name)
            loaded = experts.weight_loader(
                param,
                loaded_weight,
                f"experts.{param_name}",
                shard_id,
                int(expert_id_s),
                return_success=True,
            )
            if loaded:
                yield param_name

    return load_weights


@patch(
    _MODULE,
    "Qwen4ExpSparseMoeBlock.__init__",
    reason=(
        "Qwen4Exp has no upstream MegaMoE branch; on PPU with "
        "--moe-backend deep_gemm_mega_moe and EP on, build the SAIL "
        "PPUDeepGemmMegaMoEExperts instead of the FusedMoEFactory runner. "
        "The qwen4-max MXFP4 checkpoint (1x32 UE8M0, verified m890-57) "
        "loads through the experts' shard-id weight loader."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vLLM Qwen4Exp gains a use_native_mega_moe branch equivalent to "
        "DeepSeek V4's _init_mega_moe_experts."
    ),
)
def moe_block_init(
    self,
    vllm_config,
    prefix: str = "",
    reduce_results: bool = True,
) -> None:
    # PPU MODIFICATION: begin — resolve the mega branch before upstream
    # __init__ so the FusedMoEFactory runner is never constructed for it.
    use_native_mega_moe = (
        current_platform.is_ppu()
        and vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
    )
    if use_native_mega_moe:
        parallel_config = vllm_config.parallel_config
        if not parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a "
                "different moe backend."
            )
        if parallel_config.enable_eplb:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE does not support EPLB on PPU: SAIL W4A8 "
                "rebalancing can change the inverse weight scale captured "
                "by the kernel."
            )
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE does not support MoE sequence parallelism "
                "in v1."
            )
        if is_hc_sequence_parallel_enabled(vllm_config):
            raise NotImplementedError(
                "Qwen4Exp MegaMoE does not support HC sequence parallelism "
                "in v1: the HC-SP reduce path reads experts.moe_config, "
                "which the mega experts do not expose."
            )
        if get_tensor_model_parallel_world_size() > 1:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE requires TP=1 in v1; TP2 additionally "
                "triggers the framework MoE sequence-parallel path, whose "
                "reduce semantics read experts.moe_config, which the mega "
                "experts do not expose."
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE does not support speculative decoding "
                "(MTP) in v1."
            )
    self.use_native_mega_moe = use_native_mega_moe
    # PPU MODIFICATION: end

    if not use_native_mega_moe:
        _upstream_init(
            self,
            vllm_config=vllm_config,
            prefix=prefix,
            reduce_results=reduce_results,
        )
        return

    # PPU MODIFICATION: begin — MegaMoE branch.
    # Rebuild only what the mega path consumes: the gate, the sigmoid-gated
    # shared MLP (kept outside the megakernel), and the mega experts. The
    # field set mirrors what Qwen4ExpMixtureOfExperts metadata reads.
    config = vllm_config.model_config.hf_text_config
    torch.nn.Module.__init__(self)
    self.prefix = prefix
    self.tp_size = 1
    self.ep_group = get_ep_group().device_group
    self.ep_size = self.ep_group.size()
    self.ep_rank = get_ep_group().rank_in_group
    self.n_routed_experts = config.num_experts
    self.is_sequence_parallel = False
    self.is_fused_shared_expert_enabled = False
    self.replicate_shared_expert = False
    self.enable_eplb = False

    self.n_logical_experts = self.n_routed_experts
    self.n_redundant_experts = (
        vllm_config.parallel_config.eplb_config.num_redundant_experts
    )
    self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
    if self.n_physical_experts % self.ep_size:
        raise NotImplementedError(
            f"n_physical_experts={self.n_physical_experts} must be divisible "
            f"by ep_size={self.ep_size} for Qwen4Exp MegaMoE."
        )
    self.n_local_physical_experts = self.n_physical_experts // self.ep_size
    self.experts_start_idx = self.ep_rank * self.n_local_physical_experts
    self.experts_end_idx = self.experts_start_idx + self.n_local_physical_experts

    self.gate = GateLinear(
        config.hidden_size,
        config.num_experts,
        prefix=f"{prefix}.gate",
    )
    self.router = FusedTopKRouter(
        top_k=config.num_experts_per_tok,
        global_num_experts=self.n_routed_experts,
        scoring_func="softmax",
        renormalize=getattr(config, "norm_topk_prob", True),
    )

    self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)
    if self.n_shared_experts:
        from vllm.model_executor.models.qwen2_moe import (
            Qwen2MoeMLP as Qwen3NextMLP,
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = Qwen3NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            reduce_results=False,
            expert_gate=self.shared_expert_gate,
            is_sequence_parallel=False,
            disable_tp=False,
            prefix=f"{prefix}.shared_expert",
        )
    else:
        self.shared_expert_gate = None
        self.shared_expert = None

    # Deferred import: the PPU redirect (enhancement/mega_moe install) has
    # replaced this symbol by the time a model is built; an eager top-level
    # import would capture the pre-redirect CUDA class.
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        DeepGemmMegaMoEExperts,
    )

    self.experts = DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=self.n_physical_experts,
        num_local_experts=self.n_local_physical_experts,
        experts_start_idx=self.experts_start_idx,
        num_logical_experts=self.n_logical_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        activation=MegaMoEActivation("swiglu"),
        weight_format=MEGA_MOE_MXFP4,
        shared_experts=None,
        prefix=f"{prefix}.experts",
    )
    # AutoWeightsLoader delegates expert checkpoint tensors here; the mega
    # experts only expose the per-parameter shard-id loader.
    self.experts.load_weights = _make_expert_load_weights(self.experts)
    # PPU MODIFICATION: end


_upstream_init = getattr(moe_block_init, PATCH_MARKER)[_INIT_TARGET]


@patch(
    _MODULE,
    "Qwen4ExpSparseMoeBlock.forward",
    reason=(
        "The mega experts consume explicit topk_weights/topk_ids instead of "
        "router_logits, and Qwen4Exp softmax routing cannot use bf16_mega_gate "
        "(sqrtsoftplus/sigmoid only). Route with the standard fused top-k, "
        "then launch the MegaMoE experts with the shared expert attached."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vLLM Qwen4Exp gains a MegaMoE forward equivalent to DeepSeek V4's "
        "use_native_mega_moe branch, or bf16_mega_gate supports softmax "
        "scoring."
    ),
)
def moe_block_forward(
    self,
    hidden_states: torch.Tensor,
    already_sequence_parallel: bool = False,
) -> torch.Tensor:
    if not getattr(self, "use_native_mega_moe", False):
        return _upstream_forward(
            self, hidden_states, already_sequence_parallel=already_sequence_parallel
        )

    # PPU MODIFICATION: begin — explicit softmax routing + MegaMoE launch.
    orig_shape = hidden_states.shape
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    if isinstance(router_logits, tuple):
        router_logits = router_logits[0]
    router_logits = router_logits.to(torch.float32)
    topk_weights, topk_ids = self.router.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_indices_dtype=torch.int32,
    )

    final_hidden_states = self.experts(
        hidden_states,
        topk_weights.to(torch.float32),
        topk_ids,
        self.shared_expert,
    )
    return final_hidden_states.view(orig_shape)
    # PPU MODIFICATION: end


_upstream_forward = getattr(moe_block_forward, PATCH_MARKER)[_FORWARD_TARGET]


# The __init__/forward replacements are class-attribute patches on the
# shared Qwen4ExpSparseMoeBlock class object, so mtp.py's by-value import
# (mtp.py:62) already observes them; no consumer rebinding is required.
