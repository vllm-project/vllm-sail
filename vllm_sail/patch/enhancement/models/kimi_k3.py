# SPDX-License-Identifier: Apache-2.0
"""Quantized routed up-projection support for the Kimi K3 latent-MoE tail.

The fused latent-MoE path assumes the replicated routed up-projection is a
plain dense weight: tier 0's CuTeDSL operator and tier 2's ``narrow()+addmm_``
read ``up_proj.weight`` directly, and tier 1 multiplies it with ``torch.mm``.
A PPU checkpoint may quantize that projection, and then only the layer's quant
method can run its GEMM.

These patches record whether the projection is quantized, force every batch
onto tier 1 when it is, and route both the tier-1 GEMM and
``KimiRoutedOutputTransform``'s residual epilogue through the layer. An
unquantized up-projection keeps delegating to upstream unchanged.

They also restore the missing tensor-parallel reduction on the eager MegaMoE
path: ``KimiMoE.forward`` calls the shared expert directly there, and sequence
parallelism -- the only reason the shared expert is replicated -- is disabled
whenever pipeline parallelism is on, so nothing summed the TP-sharded
shared-expert partial. ``KimiMoE.__init__`` records when that sum is owed and
``KimiRoutedOutputTransform`` is where the shared output re-enters the residual
stream, so the collective lives there.

A default-on, opt-out MoE-local sequence-parallel path removes replicated
routed-expert work when pipeline parallelism disables the model-level SP path.
Only the MoE sees a TP token shard; its output is gathered before the
surrounding decoder layer.
"""

from __future__ import annotations

import inspect
from dataclasses import fields as dataclass_fields

from vllm import envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.common.dsa_cache import SelfAddressedMLASpec
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
    LatentMoERunner,
    LatentTailTier,
)
from vllm.models.kimi_k3.nvidia.model import KimiMoE, KimiRoutedOutputTransform
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

from vllm_sail import envs as sail_envs
from vllm_sail.patch.utils import patch

# Captured before patching so every replacement can fall back to upstream.
_upstream_runner_init = LatentMoERunner.__init__
_upstream_select_tail_tier = LatentMoERunner._select_tail_tier
_upstream_overlap_allreduce_tail = LatentMoERunner._overlap_allreduce_tail
_upstream_transform_forward = KimiRoutedOutputTransform.forward
_upstream_moe_init = KimiMoE.__init__
_upstream_moe_forward = KimiMoE.forward


@patch(
    "vllm.models.kimi_k3.nvidia.model",
    "KimiMoE.__init__",
    reason=(
        "KimiMoE.forward calls the shared expert directly on the eager MegaMoE "
        "path, so no MoERunner tail all-reduces the TP-sharded shared-expert "
        "partial. Also enable MoE-local SP when PP disables model-level SP; "
        "delegates to upstream elsewhere."
    ),
    remove_when=(
        "the eager MegaMoE path reduces the shared-expert output itself and "
        "upstream provides MoE-local sequence sharding with pipeline parallelism."
    ),
)
def moe_init(self, *args, **kwargs) -> None:
    # PPU MODIFICATION: begin — model-level SP is disabled for PP>1. Opt into
    # the same contract only inside this MoE, which both shards MegaMoE's token
    # capacity and gives the shared expert the correct SP weight/collective
    # semantics. Do not enable the surrounding decoder or pipeline SP path.
    is_ppu = current_platform.is_ppu()
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is None and len(args) > 1:
        vllm_config = args[1]
    use_sequence_parallel = kwargs.get(
        "use_sequence_parallel", args[5] if len(args) > 5 else False
    )
    use_moe_sequence_parallel = (
        is_ppu
        and sail_envs.VLLM_SAIL_KIMI_K3_MOE_SEQUENCE_PARALLEL
        and vllm_config is not None
        and not use_sequence_parallel
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and vllm_config.parallel_config.enable_expert_parallel
        and vllm_config.parallel_config.tensor_parallel_size > 1
        and vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
    )
    if use_moe_sequence_parallel:
        if len(args) > 5:
            args = (*args[:5], True, *args[6:])
        else:
            kwargs["use_sequence_parallel"] = True
        # The surrounding PP>1 model may have initialized GEMM-AR for dense
        # layers. A sharded SP shared expert needs RS instead, so use its plain
        # AG/RS collectives rather than entering the incompatible singleton.
        if len(args) > 6:
            args = (*args[:6], False, *args[7:])
        else:
            kwargs["run_gemm_rs_ar"] = False
    # PPU MODIFICATION: end
    _upstream_moe_init(self, *args, **kwargs)
    if not is_ppu:
        return
    self._moe_sequence_parallel = use_moe_sequence_parallel
    # PPU MODIFICATION: begin — sequence parallelism keeps the shared expert
    # replicated (both projections are disable_tp), which is why the eager
    # MegaMoE path needs no collective at all. It turns off whenever pipeline
    # parallelism is enabled, and then the shared expert falls back to a
    # TP-sharded down_proj whose partial output nobody sums: the routed branch
    # is already complete after the kernel's combine, so the missing rows
    # corrupt every following layer.
    transform = self.routed_output_transform
    if self.use_mega_moe and transform is not None:
        transform._mega_shared_partial = (
            self.tp_size > 1 and not self.use_sequence_parallel
        )
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.model",
    "KimiMoE.forward",
    reason=(
        "Pipeline parallelism disables Kimi K3 model-level SP, leaving every TP "
        "rank a full token copy for MegaMoE. The opt-out wrapper shards only the "
        "MoE input and gathers its output, preserving decoder and PP interfaces."
    ),
    remove_when=(
        "upstream supports an equivalent MoE-local sequence-sharding path for "
        "Kimi K3 MegaMoE when pipeline parallelism is enabled."
    ),
)
def moe_forward(self, hidden_states):
    if not getattr(self, "_moe_sequence_parallel", False):
        return _upstream_moe_forward(self, hidden_states)
    # PPU MODIFICATION: begin — the upstream MoE runs on this rank's token
    # shard. Padding rows need the same local mask, and the original mask must
    # be restored before the gathered output rejoins the full residual stream.
    num_tokens = hidden_states.shape[0]
    forward_context = None
    is_padding = None
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        forward_context = get_forward_context()
        is_padding = forward_context.is_padding
        forward_context.is_padding = sp_padding_mask(
            is_padding[:num_tokens] if is_padding is not None else None,
            hidden_states,
        )
    try:
        local_output = _upstream_moe_forward(self, sp_shard(hidden_states))
    finally:
        if forward_context is not None:
            forward_context.is_padding = is_padding
    return sp_all_gather(local_output)[:num_tokens]
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.latent_moe_runner",
    "LatentMoERunner.__init__",
    reason=(
        "PPU checkpoints can quantize the replicated routed up-projection, and "
        "the latent-MoE tail can only run its GEMM through the layer's quant "
        "method. Record whether up_proj is quantized on PPU; delegates to "
        "upstream elsewhere."
    ),
    remove_when=(
        "LatentMoERunner derives tail eligibility from the up-projection's "
        "quant method instead of assuming a plain weight."
    ),
)
def runner_init(self, *args, **kwargs) -> None:
    _upstream_runner_init(self, *args, **kwargs)
    if not current_platform.is_ppu():
        return
    # PPU MODIFICATION: begin — a quantized up-proj cannot use the raw-weight
    # tiers, so the tail needs to know about it.
    assert self.enable_k3_latent_moe_tail_fusion is False
    transform = self.routed_output_transform
    self._up_proj_is_quantized = transform is not None and not isinstance(
        transform.up_proj.quant_method, UnquantizedLinearMethod
    )
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.latent_moe_runner",
    "LatentMoERunner._select_tail_tier",
    reason=(
        "A quantized routed up-projection cannot run tier 0's CuTeDSL operator "
        "or tier 2's narrow()+addmm_ shard, which both read the packed weight "
        "directly; force tier 1, the only tier that goes through the layer's "
        "quant method. Delegates to upstream for an unquantized weight."
    ),
    remove_when=(
        "LatentMoERunner selects its tail from the up-projection quant method, "
        "so a quantized weight never reaches the raw-weight tiers."
    ),
)
def select_tail_tier(self, fused_output, shared_output):
    if not getattr(self, "_up_proj_is_quantized", False):
        return _upstream_select_tail_tier(self, fused_output, shared_output)
    # PPU MODIFICATION: begin — only tier 1 runs the up-projection through its
    # quant method. The attribute is PPU-only, so non-PPU always delegates.
    return LatentTailTier.ALLREDUCE_OVERLAP
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.latent_moe_runner",
    "LatentMoERunner._overlap_allreduce_tail",
    reason=(
        "Tier 1 normally multiplies the replicated weight with torch.mm, which "
        "cannot run a quantized up-projection; call the layer so its quant "
        "method owns the GEMM. Quantization forces every batch onto this tier, "
        "so the small-batch token guard no longer applies."
    ),
    remove_when=(
        "upstream runs the tier-1 up-projection through the linear layer (or "
        "its quant method) instead of torch.mm on the raw weight."
    ),
)
def overlap_allreduce_tail(self, fused_output, shared_output, trunc_size):
    if not getattr(self, "_up_proj_is_quantized", False):
        return _upstream_overlap_allreduce_tail(
            self, fused_output, shared_output, trunc_size
        )
    # PPU MODIFICATION: begin — verbatim upstream body, with the up-projection
    # GEMM routed through the layer and the small-batch guard dropped.
    transform = self.routed_output_transform
    assert transform is not None
    if transform.norm is not None:
        fused_latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
    else:
        fused_latent = tensor_model_parallel_all_reduce(fused_output)

    # Overlap the shared-expert all-reduce with the up-projection GEMM.
    result, shared_output = maybe_execute_in_parallel(
        lambda: transform.up_proj(fused_latent)[0],
        lambda: tensor_model_parallel_all_reduce(shared_output),
        self._shared_ar_events[0],
        self._shared_ar_events[1],
        aux_stream(),
    )
    result.add_(shared_output)

    # Output is already fully reduced; this only strips padding.
    return self._maybe_reduce_final_output(result, trunc_size, output_is_reduced=True)
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.model",
    "KimiRoutedOutputTransform.forward",
    reason=(
        "KimiRoutedOutputTransform folds the shared output into the "
        "up-projection GEMM's beta-add epilogue through residual.addmm_ on the "
        "raw weight, which only works for an unquantized up-projection. On "
        "PPU, call the linear layer for a quantized weight and keep the "
        "epilogue otherwise."
    ),
    remove_when=(
        "upstream folds the residual through the layer's quant method, or "
        "exposes an equivalent epilogue for a quantized up-projection."
    ),
)
def transform_forward(self, hidden_states, residual=None):
    # PPU MODIFICATION: begin — sum the TP-sharded shared-expert partial before
    # the up-projection epilogue folds the routed result into it. Only the eager
    # MegaMoE path sets the flag (see ``KimiMoE.__init__``): it is the one caller
    # whose shared output arrives unreduced and un-run through a MoERunner tail.
    if residual is not None and getattr(self, "_mega_shared_partial", False):
        residual = tensor_model_parallel_all_reduce(residual)
    # PPU MODIFICATION: end
    if (
        residual is None
        or not current_platform.is_ppu()
        or isinstance(self.up_proj.quant_method, UnquantizedLinearMethod)
    ):
        return _upstream_transform_forward(self, hidden_states, residual)
    # PPU MODIFICATION: begin — the beta-add epilogue reads the raw weight; a
    # quantized up-projection must run its own GEMM and be added afterwards.
    if self.norm is not None:
        hidden_states = self.norm(hidden_states)
    return residual.add_(self.up_proj(hidden_states)[0])
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.common.dsa_cache",
    "SelfAddressedMLASpec.from_spec",
    reason=(
        "The fork extends MLAAttentionSpec with indexer_n_head/indexer_q_head_dim "
        "field descriptors (attention/kv_cache_interface.py), and a dataclass "
        "subclass generated after that injection lists them in __dataclass_fields__ "
        "while its generated __init__ still rejects them. from_spec copies every "
        "init field of the source spec, so binding Kimi's self-addressed MLA cache "
        "crashes with an unexpected-keyword TypeError. Keep the geometry out: the "
        "self-addressed cache is sized by the packed layout itself, and the DSA "
        "indexer metadata builder reads its geometry from the model config. On "
        "builds where MLAAttentionSpec natively declares (or lacks) those fields, "
        "the filter is a no-op against upstream's behavior."
    ),
    remove_when=(
        "upstream MLAAttentionSpec declares the indexer geometry fields itself "
        "(so no injection happens), or from_spec stops copying unknown fields."
    ),
)
@classmethod
def from_spec(cls, spec):
    accepted = set(inspect.signature(cls.__init__).parameters)
    return cls(
        **{
            f.name: getattr(spec, f.name)
            for f in dataclass_fields(spec)
            if f.init and f.name in accepted
        }
    )
