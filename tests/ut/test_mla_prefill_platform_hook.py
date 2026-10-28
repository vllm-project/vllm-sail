# SPDX-License-Identifier: Apache-2.0
"""Check the inherited MLA hook with real source and CPU dependency fixtures."""

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function


@pytest.mark.upstream_source
@pytest.mark.parametrize("capability", [80, 89])
@pytest.mark.parametrize("rejection", [None, "configuration", "import"])
def test_ppu_inherits_validating_mla_prefill_hook(
    monkeypatch, upstream_source_root, capability, rejection
):
    cuda_source = upstream_source_root / "vllm/platforms/cuda.py"
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "get_mla_prefill_backend_cls"
        for node in ast.walk(ast.parse(cuda_source.read_text()))
    )
    device = NS(major=capability // 10, minor=capability % 10)
    seen = []

    class Backend:
        @staticmethod
        def validate_configuration(cap, config):
            seen.append((cap, config))
            return ["fixture configuration rejected"] if rejection else []

    class Candidate:
        name = "FLASH_ATTN"

        def get_class(self):
            if rejection == "import":
                raise ImportError("fixture optional backend missing")
            return Backend

    hook = function(
        upstream_source_root / "vllm/platforms/interface.py",
        "Platform.get_mla_prefill_backend_cls",
        {"logger": NS(debug_once=lambda *args: None)},
    )

    class NvmlCudaPlatform:
        get_mla_prefill_backend_cls = classmethod(hook)

        @classmethod
        def get_device_capability(cls):
            return device

    modules = {
        "torch": NS(backends=NS(cuda=NS(enable_cudnn_sdp=lambda value: None))),
        "vllm.logger": NS(init_logger=logging.getLogger),
        "vllm.platforms.cuda": NS(NvmlCudaPlatform=NvmlCudaPlatform),
        "vllm.platforms.interface": NS(
            DeviceCapability=tuple, PlatformEnum=NS(CUDA=object())
        ),
        "vllm.v1.attention.backends.registry": NS(AttentionBackendEnum=object),
        "vllm.v1.attention.backends.mla.prefill.base": NS(
            MLADimensions=lambda **kw: NS(**kw)
        ),
        "vllm.v1.attention.backends.mla.prefill.registry": NS(
            MLAPrefillBackendEnum=NS(FLASH_ATTN=Candidate())
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "vllm_sail/platform.py"
    spec = importlib.util.spec_from_file_location("_ppu_mla_hook_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    platform = module.PPUPlatform
    assert "get_mla_prefill_backend_cls" not in platform.__dict__
    assert platform.is_ppu()
    config = NS(mla_dimensions=NS())
    if rejection:
        with pytest.raises(ValueError, match="FLASH_ATTN"):
            platform.get_mla_prefill_backend_cls(config)
    else:
        assert platform.get_mla_prefill_backend_cls(config) is Backend
    assert seen == ([] if rejection == "import" else [(device, config)])
