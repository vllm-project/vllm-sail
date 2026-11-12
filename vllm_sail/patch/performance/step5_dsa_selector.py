# SPDX-License-Identifier: Apache-2.0
"""仅显式启用时替换 Step5 region selector；不改变 indexer、MTP 或 attention。"""
from __future__ import annotations

import functools
import importlib
import sys

from vllm.platforms import current_platform
from vllm_sail import envs
from vllm_sail.patch.utils import patch

_TARGET = "vllm.models.step5.nvidia.ops.sparse_indexer"
_REASON = "Step5 selector_v2 精确 FP32 radix 候选，复用稳定化；默认关闭，按 UT 性能选择。"
_REMOVE = "上游 per_row 提供等效或更快的流式 radix 选择器。"
_installed = False


def install():
    global _installed
    if _installed:
        return
    try:
        module = importlib.import_module(_TARGET)
    except ModuleNotFoundError as exc:
        if exc.name and (_TARGET == exc.name or _TARGET.startswith(exc.name + ".")):
            return
        raise
    original = module.select_regions

    @functools.wraps(original)
    def select(scores, current, topk, backend="auto", **kwargs):
        if (not envs.VLLM_SAIL_STEP5_DSA_SELECTOR_PPU or not current_platform.is_ppu()
                or backend not in ("auto", "per_row") or not 0 < topk <= 1024
                or not scores.is_contiguous()
                or str(scores.dtype) != "torch.float32"):
            return original(scores, current, topk, backend, **kwargs)
        from vllm_sail.models.step5.ops.selector_v2 import select_regions
        return select_regions(scores, current, topk, backend, **kwargs)

    for name in (_TARGET, "vllm.models.step5.nvidia.flashinfer_sparse"):
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, "select_regions", None) is original:
            patch(name, "select_regions", reason=_REASON,
                  affected_versions=">=0.33.0,<0.34.0", remove_when=_REMOVE)(select)
    _installed = True


install()
