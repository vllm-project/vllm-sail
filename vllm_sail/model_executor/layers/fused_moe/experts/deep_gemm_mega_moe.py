# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU variants of upstream's ``DeepGemmMegaMoEExperts``.

Upstream's class targets SM100: parameters in the kernel's interleaved layout,
Triton input preparation (``prepare_megamoe_inputs``) and
``deep_gemm.<format>_mega_moe`` with CUDA's keyword names. PPU's DeepGEMM
implements the same fused dispatch/GEMM/combine contract, but with different
weight, dispatch and activation arguments, so the stages it shapes differently
are re-expressed here as a dedicated subclass. The patch layer
(``vllm_sail.patch.enhancement.mega_moe``) redirects the upstream symbol to
this class on PPU.

Execution precision: PPU's DeepGEMM MegaMoE runs several MMA modes over the
same raw MXFP4 checkpoint banks, selected by ``VLLM_SAIL_MEGA_MOE_MMA_TYPE``
(see :data:`_FORMATS`):

* ``fp8xmxfp4`` (W4A8, default) — ``fp8_fp4_mega_moe``, E4M3 activations.
* ``mxf4xmxf4`` (W4A4) — ``fp4_fp4_mega_moe``, MXFP4 activations. The
  strictest shapes: H >= 1024, H % 512, I % 64.
* ``bf16xmxfp4`` (W4A16) — ``bf16_fp4_mega_moe``, BF16 activations, SiTU-only.

Other overrides relative to upstream:

* ``_fusable_shared_experts`` — PPU has no shared-expert fusion
  (``num_shared_experts`` is always 0); Kimi K3 already runs its shared MLP on
  the un-projected hidden states.
* ``_create_weights`` / ``_load_weight`` — raw MXFP4 banks (``w13``
  ``(E, 2I, H/2)`` and ``w2`` ``(E, H, I/2)`` uint8, with uint8 UE8M0 group-32
  scales) instead of the interleaved, int32-packed kernel layout. All gate rows
  precede all up rows, which is what every PPU transform expects. The same
  pass rejects more than 128 local experts per rank: PPU's dispatch layout is
  sized from that count, so a wider expert shard can never build its symmetric
  buffer.

Weight preprocessing is deliberately lazy: PPU wants the raw banks at load time
and the transformed banks at first use, so the one-off transform runs on the
first forward and is invalidated whenever a checkpoint shard is written into
the banks. That keeps a reload correct without a post-load hook (these modules
carry no ``quant_method``, so ``process_weights_after_loading`` never reaches
them). EPLB moves expert bytes through the tensors ``get_expert_weights``
handed out, which bypasses the loaders, so while it is active the transform is
rebuilt per step instead of reused. That guard is defensive: Kimi K3 leaves
``update_physical_experts_metadata`` at its ``MixtureOfExperts`` stub, so its
EPLB state is never wired up and the cache is not dropped at runtime.

The layer's ``MegaMoEWeightFormat`` is still ``MEGA_MOE_MXFP4``
(``mma_type="fp8xfp4"``); the PPU path never reads it -- it takes the geometry,
activation and expert ranges from the module itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
)

from vllm_sail.utils.deep_gemm import (
    bf16_fp4_mega_moe,
    fp4_fp4_mega_moe,
    fp8_fp4_mega_moe,
    get_symm_buffer_for_mega_moe,
    mega_moe_pre_dispatch,
    transform_weights_for_mega_moe,
    transform_weights_for_w4a16_mega_moe,
    transform_weights_for_w4a8_mega_moe,
)

logger = init_logger(__name__)

#: Most local experts one rank may own. DeepGEMM's dispatch layout is
#: dimensioned by the per-rank expert count (``checked_routed_layout``), so the
#: symmetric buffer cannot be built beyond it.
PPU_MAX_LOCAL_EXPERTS = 128

#: DeepGEMM entry points per MMA mode the PPU path calls. A missing one means
#: the installed package predates that mode.
PPU_MEGA_APIS: dict[str, tuple[str, ...]] = {
    "fp8xmxfp4": (
        "get_symm_buffer_for_mega_moe",
        "transform_weights_for_w4a8_mega_moe",
        "mega_moe_pre_dispatch",
        "fp8_fp4_mega_moe",
    ),
    "mxf4xmxf4": (
        "get_symm_buffer_for_mega_moe",
        "transform_weights_for_mega_moe",
        "mega_moe_pre_dispatch",
        "fp4_fp4_mega_moe",
    ),
    "bf16xmxfp4": (
        "get_symm_buffer_for_mega_moe",
        "transform_weights_for_w4a16_mega_moe",
        "mega_moe_pre_dispatch",
        "bf16_fp4_mega_moe",
    ),
}


@dataclass(frozen=True)
class PPUMegaMoEFormat:
    """One PPU MegaMoE execution precision over an MXFP4 checkpoint."""

    mma_type: str
    #: Symmetric-buffer tag and pre-dispatch mode.
    kernel: str
    #: Complete-compute DeepGEMM entry.
    transform: str
    #: Name of the ``transform_weights_for_*`` entry preparing the banks.
    activations: frozenset[str]
    #: Activations the kernel accepts (checked against the layer's).
    hidden_multiple: int
    inter_multiple: int
    min_hidden: int


_FORMATS: dict[str, PPUMegaMoEFormat] = {
    fmt.mma_type: fmt
    for fmt in (
        PPUMegaMoEFormat(
            mma_type="fp8xmxfp4",
            kernel="fp8_fp4_mega_moe",
            transform="transform_weights_for_w4a8_mega_moe",
            activations=frozenset({"swiglu", "situ", "swiglu_oai"}),
            hidden_multiple=256,
            inter_multiple=128,
            min_hidden=0,
        ),
        PPUMegaMoEFormat(
            mma_type="mxf4xmxf4",
            kernel="fp4_fp4_mega_moe",
            transform="transform_weights_for_mega_moe",
            activations=frozenset({"swiglu", "situ", "swiglu_oai"}),
            hidden_multiple=512,
            inter_multiple=64,
            min_hidden=1024,
        ),
        PPUMegaMoEFormat(
            mma_type="bf16xmxfp4",
            kernel="bf16_fp4_mega_moe",
            transform="transform_weights_for_w4a16_mega_moe",
            # W4A16 runs the BF16 activations through an MX quantizing
            # epilogue that only exists for SiTU.
            activations=frozenset({"situ"}),
            hidden_multiple=256,
            inter_multiple=128,
            min_hidden=0,
        ),
    )
}


def resolve_mma_type() -> str:
    """The configured PPU MegaMoE execution precision (default W4A8)."""
    import vllm_sail.envs as ppu_envs

    return ppu_envs.VLLM_SAIL_MEGA_MOE_MMA_TYPE


def _eplb_active(module: Any) -> bool:
    """Whether EPLB may have rewritten the raw banks behind the weight loaders."""
    eplb_state = getattr(module, "eplb_state", None)
    return eplb_state is not None and eplb_state.logical_to_physical_map is not None


def _transformed_weights(module: Any, fmt: PPUMegaMoEFormat) -> tuple[Any, Any]:
    """PPU's transformed weight banks, built on first use.

    The transform walks both banks once, so its result is cached on the module
    and dropped by ``_load_weight`` when a checkpoint shard rewrites a bank.
    EPLB rearranges experts by copying into the tensors ``get_expert_weights``
    returned, never through the loaders, so its rearrangements invalidate the
    cache as well: while it is active the transform is rebuilt per step rather
    than reused.
    """
    cached = getattr(module, "_ppu_mega_moe_weights", None)
    if cached is not None:
        if not _eplb_active(module):
            return cached
        logger.warning_once(
            "MegaMoE runs with EPLB enabled, so the PPU weight transform is "
            "re-run every step; disable EPLB to preprocess the banks once."
        )

    activation = module.activation.name
    banks = (
        (module.w13_weight.view(torch.uint8), module.w13_weight_scale.view(torch.uint8)),
        (module.w2_weight.view(torch.uint8), module.w2_weight_scale.view(torch.uint8)),
    )
    transform = {
        "transform_weights_for_w4a8_mega_moe": lambda: transform_weights_for_w4a8_mega_moe(
            *banks
        ),
        "transform_weights_for_mega_moe": lambda: transform_weights_for_mega_moe(
            *banks, activation
        ),
        "transform_weights_for_w4a16_mega_moe": lambda: transform_weights_for_w4a16_mega_moe(
            *banks, activation
        ),
    }[fmt.transform]
    transformed = transform()
    module._ppu_mega_moe_weights = transformed
    return transformed


def _activation_kwargs(activation: Any) -> dict[str, Any]:
    """Map the layer's fused activation onto PPU's kernel keywords.

    PPU rejects a SiTU linear scale of zero and takes ``None`` for "no up
    softcap", while the layer keeps Kimi K3's ``beta=0`` as its default.
    """
    if activation.name == "situ":
        return {
            "activation": "situ",
            "situ_beta": float(activation.alpha) if activation.alpha > 0 else 1.0,
            "situ_linear_beta": float(activation.beta) if activation.beta > 0 else None,
        }
    return {"activation": activation.name, "activation_clamp": activation.clamp}


class PPUDeepGemmMegaMoEExperts(DeepGemmMegaMoEExperts):
    """Upstream MegaMoE experts re-expressed for PPU's MegaMoE kernels.

    Construction, weight loading plumbing, EPLB and CUDA-graph capture are
    inherited; the stages PPU's kernels shape differently are overridden. The
    execution precision comes from ``VLLM_SAIL_MEGA_MOE_MMA_TYPE``.
    """

    # PPU's workspace is keyed differently from upstream's (no shared experts,
    # PPU mma type), so keep a separate cache.
    _symm_buffers: ClassVar[dict[tuple, Any]] = {}

    @property
    def ppu_format(self) -> PPUMegaMoEFormat:
        return _FORMATS[resolve_mma_type()]

    def _fusable_shared_experts(
        self, shared_experts: nn.Module | None
    ) -> tuple[int, Any]:
        # PPU cannot fuse a shared expert into the launch (the symmetric buffer
        # and kernel keywords have no shared-expert slots). A model passing one
        # still works: forward() runs it as a separate MLP.
        return 0, None

    def _create_weights(self) -> None:
        fmt = self.ppu_format
        if self.weight_format.name != "mxfp4":
            raise NotImplementedError(
                "PPU DeepGEMM MegaMoE supports the MXFP4 (W4A8) weight format "
                f"only, not {self.weight_format.name!r}."
            )
        if self.activation.name not in fmt.activations:
            raise NotImplementedError(
                f"PPU DeepGEMM MegaMoE {fmt.mma_type} supports "
                f"{sorted(fmt.activations)}, not {self.activation.name!r}."
            )
        # Fail here, before a single bank is allocated: the symmetric buffer is
        # sized from the per-rank expert count and cannot be built past the cap.
        experts_per_rank = self.num_local_experts
        if experts_per_rank > PPU_MAX_LOCAL_EXPERTS:
            ep_size = max(self.num_experts // experts_per_rank, 1)
            min_ep = -(-self.num_experts // PPU_MAX_LOCAL_EXPERTS)
            feasible = [
                ep
                for ep in range(min_ep, self.num_experts + 1)
                if self.num_experts % ep == 0
            ][:3]
            raise ValueError(
                f"PPU DeepGEMM MegaMoE supports at most {PPU_MAX_LOCAL_EXPERTS} "
                f"local experts per rank, but num_experts={self.num_experts} over "
                f"expert parallelism {ep_size} gives {experts_per_rank}. Use an "
                f"expert-parallel world size of at least {min_ep} that divides "
                f"{self.num_experts} (for example {feasible}), e.g. a larger "
                f"--tensor-parallel-size or --data-parallel-size."
            )
        h, i = self.hidden_size, self.intermediate_size
        if (
            h % fmt.hidden_multiple
            or i % fmt.inter_multiple
            or h < fmt.min_hidden
        ):
            raise ValueError(
                f"PPU DeepGEMM MegaMoE {fmt.mma_type} requires hidden % "
                f"{fmt.hidden_multiple} (at least {fmt.min_hidden}) and "
                f"intermediate % {fmt.inter_multiple}; got hidden={h}, "
                f"intermediate={i}."
            )
        e = self.num_local_experts
        # Raw checkpoint layout: packed uint8 values and uint8 UE8M0 scales, all
        # gate rows then all up rows. Every PPU transform consumes this as-is.
        for name, rows, cols in (("w13", 2 * i, h), ("w2", h, i)):
            weight = torch.zeros(e, rows, cols // 2, dtype=torch.uint8)
            scale = torch.zeros(e, rows, cols // 32, dtype=torch.uint8)
            self.register_parameter(
                f"{name}_weight", nn.Parameter(weight, requires_grad=False)
            )
            self.register_parameter(
                f"{name}_weight_scale", nn.Parameter(scale, requires_grad=False)
            )

    def _load_weight(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        shard_id: str,
        is_scale: bool,
    ) -> None:
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        src = src.view(dst.dtype)
        rows = dst.shape[0] // (1 if shard_id == "w2" else 2)
        if shard_id != "w2":
            # Gate occupies the first slot, up the second.
            dst = dst[:rows] if shard_id == "w1" else dst[rows:]
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(
                f"MegaMoE {shard_id} checkpoint shard {tuple(src.shape)} does not "
                f"fit the raw MXFP4 slot {tuple(dst.shape)}"
            )
        dst.copy_(src)
        self._ppu_mega_moe_weights = None

    def get_symm_buffer(self):
        fmt = self.ppu_format
        from vllm.distributed import get_ep_group

        group = get_ep_group().device_group
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            fmt.mma_type,
            self.activation,
        )
        symm_buffer = self._symm_buffers.get(key)
        if symm_buffer is None:
            symm_buffer = get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                num_shared_experts=0,
                mma_type=fmt.mma_type,
                activation=self.activation.name,
            )
            self._symm_buffers[key] = symm_buffer
        return symm_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        fmt = self.ppu_format
        import vllm.envs as envs
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            eplb_map_to_physical_and_record,
        )
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"MegaMoE got {num_tokens} tokens, but the symmetric buffer was "
                f"sized for {self.max_num_tokens}."
            )
        self._synchronize_first_launch()

        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        if is_padding is not None:
            # -1 is the global masked-route sentinel and PPU allows it to
            # repeat, so padding rows cost no dispatch or expert compute.
            # Upstream only folds the mask in while EPLB is active because its
            # Triton prepare step takes the mask itself; the C++ entry has no
            # mask argument, so folding it into the ids keeps one path.
            topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)

        # EPLB: map logical expert IDs to physical replicas and record load.
        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        # Pre-dispatch is a C++ entry with hard dtype and layout asserts, so the
        # routing tensors are normalised here instead of trusting the router.
        topk_weights = topk_weights.to(torch.float32)
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()
        if topk_ids.dtype not in (torch.int32, torch.int64):
            topk_ids = topk_ids.to(torch.int64)
        if not topk_ids.is_contiguous():
            topk_ids = topk_ids.contiguous()
        x = hidden_states
        if not x.is_contiguous():
            x = x.contiguous()

        symm_buffer = self.get_symm_buffer()
        mega_moe_pre_dispatch(
            x,
            topk_ids,
            topk_weights,
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            num_tokens,
            32,
            fmt.mma_type,
        )

        l1_weights, l2_weights = _transformed_weights(self, fmt)
        y = torch.empty(
            num_tokens, self.hidden_size, dtype=torch.bfloat16, device=x.device
        )
        {
            "fp8_fp4_mega_moe": fp8_fp4_mega_moe,
            "fp4_fp4_mega_moe": fp4_fp4_mega_moe,
            "bf16_fp4_mega_moe": bf16_fp4_mega_moe,
        }[fmt.kernel](
            y,
            l1_weights,
            l2_weights,
            symm_buffer,
            fast_math=self.fast_math,
            **_activation_kwargs(self.activation),
        )
        if shared_experts is not None:
            y += shared_experts(hidden_states)
        return y


__all__ = [
    "PPU_MEGA_APIS",
    "PPU_MAX_LOCAL_EXPERTS",
    "PPUDeepGemmMegaMoEExperts",
    "PPUMegaMoEFormat",
    "resolve_mma_type",
]
