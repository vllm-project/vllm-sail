# SPDX-License-Identifier: Apache-2.0
"""Enable the upstream Triton gated SWA decode kernel on PPU.

Upstream gates ``fuse_swa_gate`` on device capability 103 and FA version 2.
The kernel itself is portable Triton; M890 fails only those two gates (it
selects FA version 3, whose dispatched kernels are the same sm80-style
cutlass family, and head 192 / BF16 hold). Per-batch fallbacks inside
``forward_with_headwise_gate`` still cover prefill and non-uniform batches.
The fallback gate is also rewritten in place so its follow-up ``copy_`` into
the same output buffer becomes a no-op (prefill otherwise pays one extra
allocation + DtoD per layer).
"""

from vllm_sail import envs

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or not envs.VLLM_SAIL_STEP5_FUSED_SWA_DECODE:
        return

    import functools

    import torch
    from vllm.models.step5.nvidia import model

    from vllm_sail.patch.utils import patch

    original_init = model.Step5Attention.__init__

    @patch(
        model.__name__,
        "Step5Attention.__init__",
        reason="Run the upstream Triton gated SWA decode kernel on M890.",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="Upstream enables fuse_swa_gate beyond device capability 103.",
    )
    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # PPU MODIFICATION: begin — M890 fails only the capability-103 and
        # FA-version-2 gates; head 192 and the paged KV layout still hold.
        impl = self.attn.impl
        if (
            not self.fuse_swa_gate
            and not self.use_dsa_backend
            and self.use_head_wise_attn_gate
            and self.params_dtype == torch.bfloat16
            and impl.head_size == 192
            and impl.vllm_flash_attn_version in (2, 3)
        ):
            self.fuse_swa_gate = True
        # PPU MODIFICATION: end

    from vllm.models.step5.nvidia import sparse_attention
    from vllm.models.step5.nvidia.ops import attention_gate
    from vllm.triton_utils import triton

    original_gate = attention_gate.headwise_sigmoid_gate

    @patch(
        sparse_attention.__name__,
        "headwise_sigmoid_gate",
        reason="Gate the SWA/DSA fallback output in place; the callers' copy_ back into the same buffer becomes a no-op.",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="Upstream writes the fallback gate result in place.",
    )
    @functools.wraps(original_gate)
    def headwise_sigmoid_gate(attention, gate, active=None, *, slot_mapping=None):
        # PPU MODIFICATION: begin — alias Output with Attention. Both callers
        # (SWA and DSA fallbacks) copy the result back into this same buffer;
        # torch self-copy_ emits no kernel, so this drops one allocation and
        # one DtoD per layer. Kernel blocks are disjoint and load before
        # store, so the aliasing is numerically exact.
        if active is not None and slot_mapping is not None:
            raise ValueError("Pass either active flags or decode slots")
        mask = active if active is not None else slot_mapping
        head_dim = attention.shape[1] // gate.shape[1]
        attention_gate._headwise_sigmoid_gate[
            (attention.shape[0], triton.cdiv(attention.shape[1], 1024))
        ](
            attention,
            gate,
            attention,
            mask,
            mask.stride(0) if mask is not None else 0,
            mask is not None,
            slot_mapping is not None,
            mask.numel() if mask is not None else 0,
            attention.shape[1],
            head_dim,
            *attention.stride(),
            *gate.stride(),
            1024,
            enable_fp_fusion=False,
        )
        return attention
        # PPU MODIFICATION: end

    _installed = True


install()