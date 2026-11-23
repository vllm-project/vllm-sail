# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms import current_platform

from vllm_sail.models.step5.ops.exact_cast import exact_bf16_to_fp32


class PPUStep5GateLinear(GateLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = get_current_vllm_config().model_config
        self._step5_exact_cast = (
            current_platform.is_ppu()
            and model_config is not None
            and model_config.hf_text_config.model_type == "step5"
            and self.is_unquantized
            and self.weight.dtype == torch.float32
            and tuple(self.weight.shape) == (352, 4096)
            and self.out_dtype == torch.float32
            and not any(
                (
                    self.allow_ll_bf16_gemm,
                    self.allow_rocm_bf16x3_router_gemm,
                    self.allow_fp32_router_gemm,
                    self.allow_ll_fp32w_gemm,
                    self.allow_bf16x3_router_gemm,
                    self.allow_cublas_router_gemm,
                )
            )
        )

    def forward(self, x):
        if (
            self._step5_exact_cast
            and self.weight.dtype == torch.float32
            and self.out_dtype == torch.float32
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and not x.is_neg()
            and not (
                torch.is_grad_enabled()
                and (x.requires_grad or self.weight.requires_grad)
            )
        ):
            x = exact_bf16_to_fp32(x)
        return super().forward(x)
