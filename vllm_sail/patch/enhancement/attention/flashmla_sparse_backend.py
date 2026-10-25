# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.v1.attention.backends.mla.flashmla_sparse``.

Two PPU-gated fork changes, both applied by delegation:

* ``FlashMLASparseBackend.supports_compute_capability`` accepts capability
  majors 8, 9 and 10 on PPU (upstream: 9 and 10).
* ``FlashMLASparseMetadataBuilder.__init__`` uses the full SM count for
  ``max_num_sm_parts`` on PPU ("PPU uses full SM count"). That value only feeds
  the ``tile_scheduler_metadata_buffer`` allocation, so the upstream ``__init__``
  runs first and the buffer is re-allocated with the PPU size afterwards.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backends.mla import flashmla_sparse as _flashmla_sparse_backend

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.backends.mla.flashmla_sparse"

_upstream_supports_compute_capability = (
    _flashmla_sparse_backend.FlashMLASparseBackend.supports_compute_capability
)
_upstream_builder_init = _flashmla_sparse_backend.FlashMLASparseMetadataBuilder.__init__


@patch(
    _MODULE,
    "FlashMLASparseBackend.supports_compute_capability",
    reason=(
        "PPU runs sparse FlashMLA on capability major 8. Upstream accepts only "
        "majors 9 and 10, so without this PPU could never select the sparse "
        "FlashMLA backend."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream accepts PPU's capability for sparse FlashMLA natively.",
)
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    if current_platform.is_ppu():
        return capability.major in [8, 9, 10]
    return _upstream_supports_compute_capability(capability)


@patch(
    _MODULE,
    "FlashMLASparseMetadataBuilder.__init__",
    reason=(
        "PPU uses the full SM count for max_num_sm_parts (the fork: 'PPU uses "
        "full SM count') instead of the SM90/SM100 formulas. The value only "
        "sizes tile_scheduler_metadata_buffer, so this delegates to upstream and "
        "re-allocates that buffer on PPU."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream grows a PPU branch in the max_num_sm_parts formula, or the "
        "PPU scheduler stops needing the full SM count."
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
    if current_platform.is_ppu():
        sm_count = num_compute_units(device.index)
        self.tile_scheduler_metadata_buffer = torch.empty(
            # TileSchedulerMetaDataSize = 8
            # see: FlashMLA/csrc/params.h
            (sm_count, 8),
            dtype=torch.int32,
            device=device,
        )
