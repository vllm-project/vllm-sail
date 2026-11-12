# SPDX-License-Identifier: Apache-2.0
"""PPU DeepSelect (``deep_select`` package) wrapper for the DSA indexer top-k.

Upstream's ``deep_select`` backend calls ``torch.ops.deep_select`` from
``vllm._deepselect_C``, which is built for SM100 only. The PPU port ships as
the separate ``deep_select`` wheel (``deep_select.deep_select_ppu``, built for
``ppu_15``, i.e. SM 8.9) with these constraints, checked by
:func:`unsupported_reason` so callers can fall back to the per-row kernel
(``topKPerRowDecode``) instead of tripping a ``TORCH_CHECK``:

* ``0 < topk <= 4096``; BF16 input only; ``vocab < 2**23``.
* input / output index: contiguous last dimension, 16-byte aligned base,
  ``stride(0)`` aligned to ``get_stride_requirement()`` bytes (1024 / 32).
* output index: int32 or int64, exactly ``(rows, topk)`` after slicing.
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_TOPK = 4096
MAX_VOCAB = 2**23
BASE_ALIGNMENT_BYTES = 16
# Matches the -1 fill convention used for topk_indices_buffer upstream.
IDX_OOB_FILL_VALUE = -1
# Only the ppu_15 (SM 8.9) build exists; PPU 1.0 (SM 8.0) is unsupported.
SUPPORTED_CAPABILITY = 89


@functools.cache
def _module() -> ModuleType | None:
    try:
        return importlib.import_module("deep_select")
    except ImportError as exc:
        logger.debug("PPU DeepSelect unavailable: %s", exc)
        return None


@functools.cache
def _device_supported() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_ppu() and current_platform.is_device_capability(
        SUPPORTED_CAPABILITY
    )


def is_available() -> bool:
    """Whether this is an SM 8.9 PPU with the ``deep_select`` wheel."""
    return _device_supported() and _module() is not None


@functools.cache
def get_stride_requirement() -> tuple[int, int]:
    """Row stride alignment (input, output) in bytes."""
    module = _module()
    assert module is not None
    return tuple(module.get_stride_requirement())


def _layout_reason(
    name: str, tensor: torch.Tensor, stride_alignment: int
) -> str | None:
    if tensor.dim() != 2:
        return f"{name} must be 2-D, got {tensor.dim()}-D"
    if tensor.shape[1] != 1 and tensor.stride(1) != 1:
        return f"{name}.stride(1) must be 1, got {tensor.stride(1)}"
    if tensor.stride(0) * tensor.element_size() % stride_alignment:
        return (
            f"{name}.stride(0) must be a multiple of {stride_alignment} bytes, "
            f"got {tensor.stride(0)} x {tensor.element_size()} bytes"
        )
    if tensor.data_ptr() % BASE_ALIGNMENT_BYTES:
        return f"{name} base address must be {BASE_ALIGNMENT_BYTES}-byte aligned"
    return None


def unsupported_reason(
    input: torch.Tensor, topk: int, output_idx: torch.Tensor | None = None
) -> str | None:
    """Why PPU DeepSelect cannot run these tensors (``None`` when it can)."""
    if not _device_supported():
        return "requires an SM 8.9 PPU"
    if _module() is None:
        return "the deep_select package could not be imported"
    if not 0 < topk <= MAX_TOPK:
        return f"topk must be in [1, {MAX_TOPK}], got {topk}"
    if input.dtype != torch.bfloat16:
        return f"requires bfloat16 input, got {input.dtype}"
    input_stride, output_stride = get_stride_requirement()
    reason = _layout_reason("input", input, input_stride)
    if reason is not None:
        return reason
    if input.shape[1] >= MAX_VOCAB:
        return f"vocab must be < 2**23, got {input.shape[1]}"
    if output_idx is not None:
        if output_idx.dtype not in (torch.int32, torch.int64):
            return f"output_idx must be int32 or int64, got {output_idx.dtype}"
        reason = _layout_reason("output_idx", output_idx, output_stride)
        if reason is not None:
            return reason
        if output_idx.shape[0] < input.shape[0] or output_idx.shape[1] < topk:
            return (
                f"output_idx shape {tuple(output_idx.shape)} is smaller than "
                f"({input.shape[0]}, {topk})"
            )
    return None


def is_supported(
    input: torch.Tensor, topk: int, output_idx: torch.Tensor | None = None
) -> bool:
    return unsupported_reason(input, topk, output_idx) is None


def resolve_backend(
    input: torch.Tensor,
    topk: int,
    backend: str,
    output_idx: torch.Tensor | None = None,
) -> str:
    """PPU top-k backend for ``auto`` / ``deep_select``.

    Returns ``deep_select`` when PPU DeepSelect accepts the tensors. Otherwise
    an explicit ``deep_select`` falls back to ``per_row`` (topKPerRowDecode)
    and ``auto`` is returned unchanged for the caller's own auto chain. Other
    backends, and non-PPU platforms, are returned unchanged.
    """
    if backend not in ("auto", "deep_select"):
        return backend
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return backend
    reason = unsupported_reason(input, topk, output_idx)
    if reason is None:
        return "deep_select"
    if backend == "deep_select":
        logger.warning_once(
            "sparse_indexer_topk_backend='deep_select' falls back to per_row "
            "(topKPerRowDecode) on PPU: %s",
            reason,
        )
        return "per_row"
    return backend


def topk(
    input: torch.Tensor,
    topk: int,
    end: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    indices_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Upstream ``deep_select_topk`` on the PPU ``deep_select`` package.

    Unsorted top-k indices per row of ``input`` within ``[0, end[i])``; short
    rows are filled with -1. ``output_idx`` may be a wider buffer whose
    leading ``(rows, topk)`` slice receives the result.
    """
    module = _module()
    assert module is not None, "the deep_select package is not installed"
    rows = input.shape[0]
    if output_idx is not None:
        indices_dtype = output_idx.dtype
        output_idx = output_idx[:rows, :topk]
    if end is not None:
        end = end.to(torch.int32).contiguous()
    _, indices = module.topk(
        input,
        topk,
        end=end,
        indices_type=indices_dtype,
        output_idx=output_idx,
        idx_oob_fill_value=IDX_OOB_FILL_VALUE,
        value_oob_fill_value=float("-inf"),
        return_value=False,
        # Dummy/capture passes can feed NaN logits from uninitialized KV
        # cache; trapping would abort CUDA graph capture (as upstream).
        abort_when_nan_found=False,
    )
    return indices
