# SPDX-License-Identifier: Apache-2.0
"""支持的 PPU 输入默认融合 Step5 RMS 核心；public Norm 契约仍由兼容层维护。"""

from __future__ import annotations

import functools
import importlib
import math

from vllm.platforms import current_platform

from vllm_sail.patch.utils import PATCH_MARKER, patch

_MODULE = "vllm.models.step5.nvidia.layernorm"
_NAME = "_step5_rms_norm"
_REASON = "支持的 PPU 输入默认使用融合 FP32 Step5 RMS 归约与缩放。"
_AFFECTED = ">=0.33.0,<0.34.0"
_REMOVE_WHEN = "Upstream provides a PPU-compatible fused Step5 RMS core with the same contract."


def install() -> bool:
    """安装惰性选路，不加载 Triton；幂等且不覆盖其他 patch。"""
    try:
        module = importlib.import_module(_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name and (_MODULE == exc.name or _MODULE.startswith(exc.name + ".")):
            return False
        raise
    original = getattr(module, _NAME, None)
    if original is None:
        return False
    if getattr(original, "_sail_step5_fused_norm", False):
        return True
    if getattr(original, PATCH_MARKER, None) is not None:
        return False
    candidate = None

    @functools.wraps(original)
    def _step5_rms_norm(x, weight, variance_epsilon, zero_centered, out_dtype=None):
        if not current_platform.is_ppu():
            return original(x, weight, variance_epsilon, zero_centered, out_dtype)
        nonlocal candidate
        if candidate is None:
            try:
                from vllm_sail.models.step5.ops import rms_norm
            except ImportError:
                return original(x, weight, variance_epsilon, zero_centered, out_dtype)
            candidate = rms_norm
        if (not candidate.supports_fused_rms_norm(x, weight, out_dtype)
                or not math.isfinite(variance_epsilon) or variance_epsilon <= 0):
            return original(x, weight, variance_epsilon, zero_centered, out_dtype)
        # 编译/执行错误不能被 eager 静默掩盖。
        return candidate.fused_rms_norm(x, weight, variance_epsilon, zero_centered, out_dtype)

    _step5_rms_norm._sail_step5_fused_norm = True
    patch(_MODULE, _NAME, reason=_REASON, affected_versions=_AFFECTED,
          remove_when=_REMOVE_WHEN)(_step5_rms_norm)
    return True


install()
