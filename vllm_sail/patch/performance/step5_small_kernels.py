# SPDX-License-Identifier: Apache-2.0
"""PPU Step5 residual fusion and deferred shared-output wiring."""

from __future__ import annotations

import functools

import torch
from vllm.distributed.device_communicators import base_device_communicator as _comm
from vllm.models.step5.nvidia import model as _model
from vllm.platforms import current_platform

from vllm_sail.patch._upstream import step5_small_kernels as _bodies
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"
_REASON = (
    "Step5 BF16 residuals and same-stage shared outputs need fused norm wiring on PPU."
)
_REMOVE_WHEN = "Step5 exposes BF16 residual and deferred shared-output fusion through a platform hook."

_upstream_ffn = _model.Step5DecoderLayer._forward_ffn
_upstream_decoder = _model.Step5DecoderLayer.forward
_upstream_model = _model.Step5Model.forward
_upstream_all_gather = _comm.DeviceCommunicatorBase.all_gather
_ffn = bind_body(_bodies.Step5DecoderLayer._forward_ffn, _model)
_decoder = bind_body(_bodies.Step5DecoderLayer.forward, _model)
_model_forward = bind_body(_bodies.Step5Model.forward, _model)


@patch(
    "vllm.models.step5.nvidia.model",
    "Step5DecoderLayer._forward_ffn",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@functools.wraps(_upstream_ffn)
def _forward_ffn(self, *args, **kwargs):
    if current_platform.is_ppu():
        return _ffn(self, *args, **kwargs)
    return _upstream_ffn(self, *args, **kwargs)


@patch(
    "vllm.models.step5.nvidia.model",
    "Step5DecoderLayer.forward",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@functools.wraps(_upstream_decoder)
def decoder_forward(self, *args, **kwargs):
    if current_platform.is_ppu():
        return _decoder(self, *args, **kwargs)
    return _upstream_decoder(self, *args, **kwargs)


@patch(
    "vllm.models.step5.nvidia.model",
    "Step5Model.forward",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@functools.wraps(_upstream_model)
def model_forward(self, *args, **kwargs):
    if current_platform.is_ppu():
        return _model_forward(self, *args, **kwargs)
    return _upstream_model(self, *args, **kwargs)


@patch(
    "vllm.distributed.device_communicators.base_device_communicator",
    "DeviceCommunicatorBase.all_gather",
    reason="Avoid the strided reshape copy for Step5 TP4 BF16 vocabulary gathering on PPU.",
    affected_versions=_AFFECTED,
    remove_when="DeviceCommunicatorBase exposes a platform-specific all-gather layout hook.",
)
@functools.wraps(_upstream_all_gather)
def all_gather(self, input_, dim=-1):
    if (
        current_platform.is_ppu()
        and input_.ndim == 2
        and dim in (1, -1)
        and self.world_size == 4
        and input_.shape[0] >= 7
        and input_.shape[1] == 32224
        and input_.dtype == torch.bfloat16
    ):
        shape = input_.shape
        output = torch.empty(
            (self.world_size * shape[0], shape[1]),
            dtype=input_.dtype,
            device=input_.device,
        )
        _comm.dist.all_gather_into_tensor(output, input_, group=self.device_group)
        return torch.cat(output.reshape((self.world_size,) + shape).unbind(0), dim=1)
    return _upstream_all_gather(self, input_, dim)
