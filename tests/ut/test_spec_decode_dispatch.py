# SPDX-License-Identifier: Apache-2.0
"""Check copied sampler calls against the selected upstream kernel signature."""

from __future__ import annotations

import inspect
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function


class Tensor:
    def __init__(self, *shape, dtype="float32", contiguous=True):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self._contiguous = contiguous

    def stride(self, axis):
        return 1 if self._contiguous else 2

    def new_empty(self, *shape, dtype):
        return Tensor(*shape, dtype=dtype)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return Tensor(*self.shape, dtype=self.dtype)

    def view(self, dtype):
        return Tensor(*self.shape, dtype=dtype)


class Kernel:
    def __init__(self, call=lambda *a, **kw: None):
        self.call = call

    def __getitem__(self, grid):
        return self.call


@pytest.mark.upstream_source
@pytest.mark.parametrize("mask_layout", [None, "contiguous", "strided"])
def test_resample_dispatch_binds_the_upstream_mask_argument(
    upstream_source_root, mask_layout
):
    upstream = function(
        upstream_source_root
        / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
        "_resample_kernel",
        {},
    )
    calls = []

    def resample(*args, **kwargs):
        calls.append(inspect.signature(upstream).bind(*args, **kwargs).arguments)

    namespace = dict(
        torch=NS(
            **{
                name: name
                for name in ("int32", "int64", "float32", "float64", "bool", "uint8")
            }
        ),
        triton=NS(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
        _compute_local_logits_stats_kernel=Kernel(),
        _rejection_kernel=Kernel(),
        _resample_kernel=Kernel(resample),
        _insert_resampled_kernel=Kernel(),
    )
    sampler = function(
        "vllm_sail/patch/enhancement/runtime.py", "rejection_sample", namespace
    )
    masked = mask_layout is not None
    sampler(
        Tensor(2, 32),
        None,
        Tensor(2),
        Tensor(2),
        Tensor(2),
        Tensor(1),
        Tensor(2),
        Tensor(2),
        Tensor(1),
        Tensor(1),
        1,
        contexts=Tensor(2, 2, dtype="int32") if masked else None,
        watermarking=Tensor(1, dtype="bool") if masked else None,
        watermarking_skip_mask=Tensor(
            2, dtype="bool", contiguous=mask_layout != "strided"
        )
        if masked
        else None,
        watermark_key=123 if masked else None,
    )
    assert len(calls) == 1
    bound = calls[0]
    assert bound["DEDUPLICATE_CONTEXTS"] is masked
    pointer = bound["watermarking_skip_mask_ptr"]
    if masked:
        assert pointer.shape == (2,)
        assert pointer.dtype == "uint8" and pointer.is_contiguous()
    else:
        assert pointer is None
