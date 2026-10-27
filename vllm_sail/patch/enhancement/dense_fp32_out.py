# SPDX-License-Identifier: Apache-2.0
"""Route upstream bf16->fp32 dense GEMMs through PPU DeepGEMM.

Upstream computes BF16 x BF16 products with an fp32 output as
``torch.mm(x, w.T, out_dtype=torch.float32)`` -- the cuBLAS/hipBLASLt out_dtype
epilogue -- and therefore bypasses ``dispatch_unquantized_gemm`` entirely:

* ``LogitsProcessor._apply_head`` -- an fp32 lm_head
  (``--hf-overrides '{"head_dtype": "float32"}'``) accumulates logits in fp32
  instead of casting both operands to fp32 on every step.
* ``GateLinear.forward`` tier 4 -- fp32 MoE router logits for gate layers whose
  ``out_dtype`` is fp32 (Kimi K3, MiniMax-M2/M3 style routers). On PPU the
  specialized tiers 1-3 are never eligible (they need SM90+/SM100), so a K3
  decoder layer always lands on this ``torch.mm(out_dtype=float32)`` right
  after the post-attention ``attn_res`` kernel.

Both replacements only pre-empt upstream when
``should_use_deepgemm_for_bf16_linear(..., out_dtype=torch.float32)`` agrees and
otherwise delegate to the patched original, so kernel selection stays behind the
explicit ``VLLM_SAIL_DENSE_BF16_FP32_OUT_DEEPGEMM=1`` opt-in (independent of the
BF16-output switch) and ``VLLM_USE_DEEP_GEMM``.

They also need a DeepGEMM that exposes ``gemm_bf16_bf16_fp32_nt``: the BF16
kernel ``gemm_bf16_bf16_bf16_nt`` asserts ``d.scalar_type() == kBFloat16``
(``csrc/apis/gemm.hpp``) and cannot write an fp32 ``D``. That symbol doubles as
the version probe: it is resolved with ``getattr`` and a missing one merely
leaves this path disabled, so an older build keeps upstream's ``torch.mm`` and
never fails. See ``vllm_sail/patch/enhancement/dense.py`` for the custom op,
which keeps the launch compile-safe.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from vllm_sail.patch.utils import original_of, patch

logger = init_logger(__name__)
_META = dict(
    reason=(
        "PPU BF16 DeepGEMM also serves the bf16->fp32 GEMMs upstream runs as "
        "torch.mm(out_dtype=float32) (fp32 lm_head, fp32 router logits)."
    ),
    affected_versions=">=0.31.0,<0.32.0",
    remove_when="Unquantized GEMM dispatch grows an out_dtype parameter.",
)

_LOGITS_TARGET = (
    "vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head"
)
_GATE_TARGET = (
    "vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear.forward"
)


def _use_deep_gemm_fp32(x: torch.Tensor, weight: torch.Tensor) -> bool:
    from vllm_sail.utils.deep_gemm import should_use_deepgemm_for_bf16_linear

    return should_use_deepgemm_for_bf16_linear(x, weight, out_dtype=torch.float32)


def _is_unquantized_head(lm_head: torch.nn.Module) -> bool:
    """Mirror upstream's precondition for an fp32 head projection.

    Upstream raises ``ValueError`` for a quantized lm_head; never route such a
    head through DeepGEMM, which would silently use the raw weight.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    return isinstance(
        lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
    )


@patch(
    "vllm.model_executor.layers.logits_processor",
    "LogitsProcessor._apply_head",
    **_META,
)
def _apply_head(
    self,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None,
) -> torch.Tensor:
    original = original_of(_apply_head, _LOGITS_TARGET)
    if (
        self.head_dtype == torch.float32
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_cuda
        and _is_unquantized_head(lm_head)
    ):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if _use_deep_gemm_fp32(flat, lm_head.weight):
            logger.info_once("Using PPU DeepGEMM for the fp32 lm_head projection")
            bias = None if embedding_bias is None else embedding_bias.to(torch.float32)
            logits = torch.ops.vllm.ppu_bf16_deepgemm_linear_fp32(
                flat, lm_head.weight, bias
            )
            return logits.reshape(*hidden_states.shape[:-1], -1)
    return original(self, lm_head, hidden_states, embedding_bias)


@patch(
    "vllm.model_executor.layers.fused_moe.router.gate_linear",
    "GateLinear.forward",
    **_META,
)
def _gate_forward(self, x: torch.Tensor):
    original = original_of(_gate_forward, _GATE_TARGET)
    # Mirror tier 4's position: the specialized router kernels win when their
    # own eligibility flags are set, and a bias-only/short-sequence case falls
    # through to upstream untouched.
    if (
        self.allow_cublas_router_gemm
        # Tiers 1-3 keep their own eligibility, token-count and dtype rules.
        and not (self.allow_ll_bf16_gemm and x.shape[0] <= self.LL_BF16_MAX_TOKENS)
        and not self.allow_fp32_router_gemm
        and not self.allow_bf16x3_router_gemm
        and x.dtype == torch.bfloat16
        and _use_deep_gemm_fp32(x, self.weight)
    ):
        logger.info_once("Using PPU DeepGEMM for fp32 MoE router logits")
        return self._return(
            torch.ops.vllm.ppu_bf16_deepgemm_linear_fp32(x, self.weight, None)
        )
    return original(self, x)
