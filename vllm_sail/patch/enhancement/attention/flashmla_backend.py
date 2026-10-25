# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.v1.attention.backends.mla.flashmla`` (dense FlashMLA).

The fork makes four changes, all PPU-gated; each is applied here by delegation
rather than a body copy:

* ``FlashMLABackend.supports_compute_capability`` accepts capability major 8 on
  PPU (upstream: 9 and 10).
* ``FlashMLAMetadataBuilder.__init__`` sizes the CUDA-graph tile-scheduler
  buffer from a hard-coded ``num_sms = 320`` on PPU ("PPU FIXME (kai):
  temporarily use hard-coded ppu magic number, this is align with ppu mla
  algo"). The upstream ``__init__`` is run first, then the buffer is
  re-allocated with the PPU size.
* ``FlashMLAImpl.forward_mqa`` gains optional NVTX ranges gated by
  ``VLLM_PPU_NVTX_PROFILE``. The range is pushed around the upstream method
  instead of around the kernel call inside it; the range message is therefore
  built from the pre-reshape query (the fork builds it mid-method after
  ``reshape_query_for_spec_decode``), which only changes the numbers in the
  profiling label.

The fork reads the NVTX env vars once at module import; the plugin reads
``vllm_sail.envs`` per call, which also observes late env changes.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import flashmla as _flashmla_backend

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.backends.mla.flashmla"

#: Hard-coded SM count the fork uses for the PPU MLA scheduler buffers.
_PPU_NUM_SMS = 320

_upstream_supports_compute_capability = (
    _flashmla_backend.FlashMLABackend.supports_compute_capability
)
_upstream_builder_init = _flashmla_backend.FlashMLAMetadataBuilder.__init__
_upstream_forward_mqa = _flashmla_backend.FlashMLAImpl.forward_mqa


@patch(
    _MODULE,
    "FlashMLABackend.supports_compute_capability",
    reason=(
        "PPU runs FlashMLA on capability major 8. Upstream accepts only majors "
        "9 and 10, so without this PPU could never select the FlashMLA backend."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream accepts PPU's capability for FlashMLA natively.",
)
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    if current_platform.is_ppu():
        return capability.major in [8]
    return _upstream_supports_compute_capability(capability)


@patch(
    _MODULE,
    "FlashMLAMetadataBuilder.__init__",
    reason=(
        "PPU FIXME (kai): the fork hard-codes num_sms = 320 for the PPU MLA "
        "tile-scheduler buffers ('temporarily use hard-coded ppu magic number, "
        "this is align with ppu mla algo') instead of querying the device."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "PPU reports a compute-unit count the FlashMLA scheduler can use "
        "directly, and the fork's hard-coded 320 is removed."
    ),
)
def builder_init(
    self,
    kv_cache_spec,
    layer_names,
    vllm_config,
    device: torch.device,
) -> None:
    _upstream_builder_init(self, kv_cache_spec, layer_names, vllm_config, device)
    if current_platform.is_ppu() and self.cg_buf_tile_scheduler_metadata is not None:
        self.cg_buf_tile_scheduler_metadata = torch.zeros(
            (_PPU_NUM_SMS, 8),
            device=self.device,
            dtype=torch.int32,
        )


def _nvtx_push(label: str) -> bool:
    try:
        from torch.cuda.nvtx import range_push
    except ImportError:
        return False
    range_push(label)
    return True


def _nvtx_pop() -> None:
    from torch.cuda.nvtx import range_pop

    range_pop()


def _fm_fmha_nvtx_message(
    self,
    q: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata,
) -> str:
    # Fork's [FM_FMHA] profiling label. Built from the pre-reshape q; the fork
    # builds it after reshape_query_for_spec_decode, so the reported shapes can
    # differ slightly from the fork's.
    tmp_k_cache = kv_c_and_k_pe_cache.unsqueeze(-2)  # Add head dim of 1
    batch_size = q.shape[0]
    if len(q.shape) == 4:
        max_seqlen_q = q.shape[-3]
    else:
        max_seqlen_q = 1
    if not torch.cuda.is_current_stream_capturing() and ppu_envs.VLLM_SAIL_NVTX_DUMP_TOPK:
        cache_seqlens = attn_metadata.decode.seq_lens
        cu_seqlens_k_list = (
            cache_seqlens.flatten().cpu().tolist()
            if cache_seqlens is not None
            else "[]"
        )
        return f"[FM_FMHA] --format=MLA,Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{self.kv_lora_rank},num_heads_kv:{tmp_k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:True,num_blocks:{tmp_k_cache.shape[-4]},page_block_size:{tmp_k_cache.shape[-3]},cu_seqlens_k:{cu_seqlens_k_list}"
    return f"[FM_FMHA] --format=MLA,Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{self.kv_lora_rank},num_heads_kv:{tmp_k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:True"


@patch(
    _MODULE,
    "FlashMLAImpl.forward_mqa",
    reason=(
        "PPU NVTX profiling: the fork wraps the FlashMLA decode kernel call in "
        "forward_mqa with an [FM_FMHA] NVTX range gated by "
        "VLLM_PPU_NVTX_PROFILE. The plugin pushes the range around the upstream "
        "method instead; the label contents follow the fork."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU profiling moves to a mechanism that does not need NVTX ranges here.",
)
def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
    if not ppu_envs.VLLM_SAIL_NVTX_PROFILE:
        return _upstream_forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
    if not _nvtx_push(_fm_fmha_nvtx_message(self, q, kv_c_and_k_pe_cache, attn_metadata)):
        return _upstream_forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
    try:
        return _upstream_forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
    finally:
        _nvtx_pop()
