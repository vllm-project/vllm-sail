# SPDX-License-Identifier: Apache-2.0
"""SAIL MegaMoE expert adaptation for Step5."""

from contextlib import nullcontext
from copy import copy
from inspect import signature
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import replace_parameter
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.models.step5.nvidia.mega_moe import _MegaMoEMethod

from vllm_sail import envs
from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
    PPUDeepGemmMegaMoEExperts,
)
from vllm_sail.utils.deep_gemm_mega import mega_moe_api

_FP8_CHANNELWISE = "fp8_channelwise"
_MMA_TYPES = {
    "fp8xfp8": "fp8xfp8",
    "mxf4xmxf4": "fp4xfp4",
    "fp8xmxfp4": "fp8xfp4",
    "bf16xmxfp4": "bf16xfp4",
}
logger = init_logger(__name__)


def _fp8_channelwise_config(quant_config, prefix, experts):
    from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

    scheme = quant_config.get_scheme_dict(experts, layer_name=prefix)
    if scheme is None:
        scheme = quant_config.target_scheme_map.get("RoutedExperts")
    if scheme is None:
        scheme = quant_config.target_scheme_map.get("Linear", {})
    weight, activation = scheme.get("weights"), scheme.get("input_activations")
    return (
        weight is not None
        and activation is not None
        and weight.type == activation.type == QuantizationType.FLOAT
        and weight.num_bits == activation.num_bits == 8
        and weight.symmetric
        and activation.symmetric
        and not weight.dynamic
        and activation.dynamic
        and weight.strategy == QuantizationStrategy.CHANNEL
        and activation.strategy == QuantizationStrategy.TOKEN
    )


class PPUStep5BF16MoERunner(MoERunner):
    """Tail layers stay BF16 and accept in-stage SP padding."""

    def __init__(self, *, sequence_parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.sequence_parallel = sequence_parallel

    def forward(self, hidden_states, router_logits, padding=None):
        context = nullcontext()
        if self.sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)
            router_logits = sp_all_gather(router_logits)
            if is_forward_context_available():
                gathered = copy(get_forward_context())
                if gathered.dp_metadata is not None:
                    tp_size = get_tensor_model_parallel_world_size()
                    metadata = copy(gathered.dp_metadata)
                    # TP gather keeps aligned rows; DP dispatch/combine must
                    # use the same sizes.
                    counts = metadata.num_tokens_across_dp_cpu
                    metadata.num_tokens_across_dp_cpu = (
                        (counts + tp_size - 1) // tp_size
                    ) * tp_size
                    gathered.dp_metadata = metadata
                gathered.is_padding = (
                    sp_all_gather(padding[:, None].to(hidden_states.dtype))
                    .flatten()
                    .bool()
                    if padding is not None
                    else None
                )
                context = override_forward_context(gathered)
        with context:
            output = super().forward(hidden_states, router_logits)
        if self.sequence_parallel:
            output = sp_shard(output)
        if padding is not None:
            output = output.masked_fill(padding[: output.shape[0], None], 0)
        return output


class PPUStep5MegaMoEExperts(PPUDeepGemmMegaMoEExperts):
    """Reuses original MXFP4 weight loading; EP workspaces are isolated per PP stage."""

    def __init__(
        self,
        vllm_config,
        *,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        quant_config,
        activation,
        activation_clamp,
        prefix,
        max_num_tokens=None,
    ):
        from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
            MEGA_MOE_MXFP4,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
            CompressedTensorsConfig,
        )
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

        parallel = vllm_config.parallel_config
        if not parallel.enable_expert_parallel:
            raise ValueError("SAIL Step5 MegaMoE requires expert parallelism")
        if parallel.enable_eplb or parallel.eplb_config.num_redundant_experts:
            raise ValueError("SAIL Step5 MegaMoE does not support EPLB")
        if parallel.enable_dbo or getattr(parallel, "ubatch_size", 1) > 1:
            raise ValueError("SAIL Step5 MegaMoE requires serial microbatches")
        if isinstance(quant_config, Mxfp4Config):
            weight_format = MEGA_MOE_MXFP4
        elif isinstance(quant_config, CompressedTensorsConfig) and (
            _fp8_channelwise_config(quant_config, prefix, self)
        ):
            if not 1 <= top_k <= 16:
                raise ValueError("Step5 FP8 MegaMoE requires top-k in [1, 16]")
            weight_format = _FP8_CHANNELWISE
        else:
            raise ValueError(
                "SAIL Step5 MegaMoE requires MXFP4 or channelwise FP8 weights"
            )
        if activation not in ("silu", "swiglustep"):
            raise ValueError(f"Unsupported Step5 MegaMoE activation: {activation}")
        if activation == "swiglustep" and activation_clamp != 7.0:
            raise ValueError("Step5 swiglustep requires activation_clamp=7.0")
        group = get_ep_group().device_group
        if num_experts % group.size():
            raise ValueError("MegaMoE experts must divide evenly across EP ranks")
        local_experts = num_experts // group.size()
        super().__init__(
            vllm_config,
            num_experts=num_experts,
            num_local_experts=local_experts,
            experts_start_idx=group.rank() * local_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=SimpleNamespace(name="swiglu", clamp=activation_clamp),
            weight_format=weight_format,
            max_num_tokens=max_num_tokens,
            prefix=prefix,
        )
        if self.mma_type == "bf16xmxfp4":
            raise ValueError("Step5 requires a MegaMoE mode supporting SwiGLU")
        self.activation.name = "swiglu_step" if activation == "swiglustep" else "swiglu"
        self.quant_method = _MegaMoEMethod()
        self.ready = False
        self.pool = None
        self.register_buffer("_padding_id", None, persistent=False)
        self.register_buffer("_padding_weight", None, persistent=False)
        self.replica_pool = None
        self.ultra_ep = None
        transfer = vllm_config.kv_transfer_config
        self.ultra_ep_requested = (
            envs.VLLM_SAIL_MEGA_MOE_ULTRA_EP
            and self.mma_type == "mxf4xmxf4"
            and parallel.data_parallel_size == 1
            and vllm_config.speculative_config is None
            and (
                transfer is None
                or transfer.kv_connector is None
                or transfer.kv_role == "kv_producer"
            )
        )
        self.ultra_ep_min_tokens = envs.VLLM_SAIL_MEGA_MOE_ULTRA_EP_MIN_TOKENS
        if self.ultra_ep_requested:
            self._ultra_ep_compilation = vllm_config.compilation_config
            self._ultra_ep_sp_size = parallel.tensor_parallel_size

    def prepare_weights(self):
        if not self.ready:
            api = mega_moe_api(self.mma_type)
            weights = self._kernel_weights(api)
            if self.mma_type == "fp8xfp8":
                # Release checkpoint-layout banks after the one-time transform.
                for bank, (weight, scale) in zip(("w13", "w2"), weights, strict=True):
                    getattr(self, f"{bank}_weight").data = weight
                    getattr(self, f"{bank}_weight_scale").data = scale
            if self.ultra_ep_requested:
                self._prepare_ultra_ep(api, weights)
            self.ready = True

    def _ultra_ep_compilation_rejection(self):
        compilation = self._ultra_ep_compilation
        mode = compilation.cudagraph_mode
        if mode not in (
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ):
            return "prefill requires eager or piecewise CUDA graphs"
        if mode != CUDAGraphMode.NONE:
            capture = max(
                compilation.max_cudagraph_capture_size or 0,
                max(compilation.cudagraph_capture_sizes or [0]),
            )
            local_capture = (
                capture + self._ultra_ep_sp_size - 1
            ) // self._ultra_ep_sp_size
            if local_capture >= self.ultra_ep_min_tokens:
                return "MIN_TOKENS must exceed the largest local CUDA graph capture"
        return None

    def _prepare_ultra_ep(self, api, weights):
        from vllm_sail.models.step5.ultra_ep import (
            PPUMegaMoEUltraEP,
            all_ranks_supported,
            available,
        )

        group = get_ep_group()
        kernel = getattr(api, self.mma_format.kernel)
        supported = False
        if available():
            try:
                supported = {"l1_weights_replicas", "l2_weights_replicas"} <= signature(
                    kernel
                ).parameters.keys()
            except (TypeError, ValueError):
                pass
        if supported:
            reason = self._ultra_ep_compilation_rejection()
            if reason:
                logger.info_once("Step5 UltraEP disabled: %s", reason, scope="process")
                supported = False
        if not all_ranks_supported(group, supported):
            logger.info_once(
                "Step5 UltraEP unavailable; using regular MegaMoE", scope="process"
            )
            return
        redundant = envs.VLLM_SAIL_MEGA_MOE_ULTRA_EP_REDUNDANT_EXPERTS
        if (
            self.ultra_ep_min_tokens <= 0
            or redundant <= 0
            or self.num_local_experts + redundant > 128
            or (self.num_local_experts + redundant) * group.world_size > 2048
        ):
            raise ValueError("Invalid Step5 UltraEP token threshold or replica count")
        self.ultra_ep = PPUMegaMoEUltraEP(
            group.device_group, extract_layer_index(self.prefix), *weights, redundant
        )
        # These banks keep the original logical layout for ordinary MegaMoE.
        # Replace weights after registration to release the original allocation.
        for bank, (weight, scale) in (
            ("w13", self.ultra_ep.l1_weights),
            ("w2", self.ultra_ep.l2_weights),
        ):
            replace_parameter(self, f"{bank}_weight", weight)
            replace_parameter(self, f"{bank}_weight_scale", scale)
        logger.info_once("Step5 UltraEP replica weights enabled", scope="process")

    def _resolve_mma_type(self, weight_format):
        if weight_format == _FP8_CHANNELWISE:
            return "fp8xfp8"
        return super()._resolve_mma_type(weight_format)

    def _register_weight(self, name, shape, spec):
        if self.mma_type != "fp8xfp8":
            return super()._register_weight(name, shape, spec)
        experts, rows, _ = shape
        self.register_parameter(
            f"{name}_weight",
            torch.nn.Parameter(
                torch.empty(shape, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
        )
        self.register_parameter(
            f"{name}_weight_scale",
            torch.nn.Parameter(
                torch.empty((experts, rows, 1), dtype=torch.float32),
                requires_grad=False,
            ),
        )

    def _load_weight(self, dst, src, shard_id, is_scale):
        if self.mma_type != "fp8xfp8":
            return super()._load_weight(dst, src, shard_id, is_scale)
        if shard_id != "w2":
            dst = dst.chunk(2, dim=0)[0 if shard_id == "w1" else 1]
        if is_scale and src.ndim == 1:
            src = src.unsqueeze(-1)
        if src.shape != dst.shape or src.dtype != dst.dtype:
            raise ValueError(
                f"Invalid Step5 FP8 {shard_id} tensor: expected "
                f"{tuple(dst.shape)} {dst.dtype}, got {tuple(src.shape)} {src.dtype}"
            )
        dst.copy_(src)
        self._transformed_weights = None

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name=None,
        shard_id=None,
        expert_id=None,
        return_success=False,
    ):
        if self.ready:
            raise RuntimeError(
                "Reloading transformed Step5 MegaMoE weights is unsupported"
            )
        if expert_id is not None:
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )
        # Fused checkpoints keep the original gate/up layout.
        weight_name = next(
            name for name, value in self.named_parameters() if value is param
        )
        bank = "w2" if weight_name.startswith("w2_") else "w13"
        count = loaded_weight.shape[0]
        if count not in (self.num_experts, self.num_local_experts):
            raise ValueError("Invalid fused Step5 MegaMoE expert count")
        start = 0 if count == self.num_experts else self.experts_start_idx
        for index in range(count):
            expert_id = start + index
            value = loaded_weight[index]
            shards = (
                (("w2", value),)
                if bank == "w2"
                else zip(("w1", "w3"), value.chunk(2, dim=0), strict=True)
            )
            for shard, value in shards:
                super().weight_loader(param, value, weight_name, shard, expert_id)
        return True if return_success else None

    weight_loader.supports_moe_loading = True

    def get_symm_buffer(self, use_replicas=False):
        pool = self.replica_pool if use_replicas else self.pool
        if pool is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up Step5 MegaMoE before capturing a CUDA graph"
                )
            self._synchronize_first_launch()
            group = get_ep_group().device_group
            num_experts = (
                self.ultra_ep.num_physical_experts if use_replicas else self.num_experts
            )
            key = (
                id(group),
                torch.cuda.current_device(),
                num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                self.mma_type,
                self.activation.name,
            )
            if key not in self._symm_buffers:
                self._symm_buffers[key] = mega_moe_api(
                    self.mma_type
                ).get_symm_buffer_for_mega_moe(
                    group,
                    num_experts,
                    self.max_num_tokens,
                    self.top_k,
                    self.hidden_size,
                    self.intermediate_size,
                    mma_type=_MMA_TYPES[self.mma_type],
                    activation=self.activation.name,
                )
            self._padding_id = torch.full(
                (), -1, dtype=torch.int64, device=self.w13_weight.device
            )
            self._padding_weight = torch.zeros(
                (), dtype=torch.float32, device=self.w13_weight.device
            )
            pool = self._symm_buffers[key]
            if use_replicas:
                self.replica_pool = pool
            else:
                self.pool = pool
        return pool

    def _use_replicas(self, tokens):
        if (
            self.ultra_ep is None
            or tokens < self.ultra_ep_min_tokens
            or not is_forward_context_available()
        ):
            return False
        context = get_forward_context()
        if (
            context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        key = "sail_step5_ultra_ep_pure_prefill"
        if key not in context.additional_kwargs:
            prefill = False
            if isinstance(context.attn_metadata, dict):
                for metadata in context.attn_metadata.values():
                    if getattr(metadata, "step5_runtime", None) is not None:
                        count = metadata.num_actual_reqs
                        starts = metadata.query_start_loc_cpu[: count + 1].tolist()
                        prefill = all(
                            b - a > 1
                            for a, b in zip(starts[:-1], starts[1:], strict=True)
                        )
                        break
            context.additional_kwargs[key] = prefill
        return context.additional_kwargs[key]

    def forward(self, hidden_states, topk_weights, topk_ids, padding=None):
        if not self.ready:
            raise RuntimeError(
                "Step5 MegaMoE weights must be prepared before inference"
            )
        tokens = hidden_states.shape[0]
        if tokens > self.max_num_tokens:
            raise ValueError("Step5 MegaMoE input exceeds its allocated token capacity")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("Step5 MegaMoE expects BF16 hidden states")
        use_replicas = self._use_replicas(tokens)
        buffer = self.get_symm_buffer(use_replicas)
        if padding is not None:
            if (
                topk_ids.dtype == torch.int64
                and topk_weights.dtype == torch.float32
                and topk_ids.shape == topk_weights.shape == (tokens, self.top_k)
                and not topk_ids.is_neg()
                and not topk_weights.is_neg()
                and topk_ids.is_contiguous()
                and topk_weights.is_contiguous()
                and padding.ndim == 1
                and padding.dtype == torch.bool
                and padding.shape[0] >= tokens
                and padding.device == topk_ids.device == topk_weights.device
                and not (torch.is_grad_enabled() and topk_weights.requires_grad)
            ):
                from vllm_sail.models.step5.ops.route_padding import mask_routes

                topk_ids, topk_weights = mask_routes(topk_ids, topk_weights, padding)
            elif topk_ids.dtype == torch.int64 and topk_weights.dtype == torch.float32:
                topk_ids = torch.where(
                    padding[:tokens, None], self._padding_id, topk_ids
                )
                topk_weights = torch.where(
                    padding[:tokens, None], self._padding_weight, topk_weights
                )
            else:
                topk_ids = torch.where(padding[:tokens, None], -1, topk_ids)
                topk_weights = torch.where(padding[:tokens, None], 0, topk_weights)
        api = mega_moe_api(self.mma_type)
        l1, l2 = self._kernel_weights(api)
        replica_kwargs = {}
        if use_replicas:
            logger.info_once("Step5 UltraEP prefill dispatch active", scope="process")
            topk_ids = self.ultra_ep.prepare_topk(topk_ids.contiguous())
            replica_kwargs = {
                "l1_weights_replicas": self.ultra_ep.l1_weights_replicas,
                "l2_weights_replicas": self.ultra_ep.l2_weights_replicas,
            }
        x = hidden_states.contiguous()
        api.mega_moe_pre_dispatch(
            x,
            topk_ids.to(torch.int64).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            buf_x=buffer.x,
            buf_x_sf=buffer.x_sf,
            buf_topk_idx=buffer.topk_idx,
            buf_topk_weights=buffer.topk_weights,
            num_tokens=tokens,
            mma_type=_MMA_TYPES[self.mma_type],
        )
        output = torch.empty_like(x)
        getattr(api, self.mma_format.kernel)(
            output,
            l1,
            l2,
            buffer,
            activation=self.activation.name,
            activation_clamp=self.activation.clamp,
            fast_math=self.fast_math,
            **replica_kwargs,
            **({"recipe": (1, 1, 32)} if self.mma_format.transform is None else {}),
        )
        return output
