# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (``glm5_next``) PPU gap patch.

``enhancement/attention/flashmla_sparse_backend`` lets the sparse FlashMLA
backend accept the rope-free (NoPE) ``head_size == 512`` combination on PPU
capability major 8, keeping upstream's kv-cache dtype and rope checks.

The ``layer_types`` allow-list this module used to cover is gone: upstream
removed the allow-list table and the pinned transformers release already
accepts ``deepseek_sparse_attention`` for ``glm5_next``.

Everything here needs an importable vLLM; on a bare CPU runner the whole module
skips.
"""

from __future__ import annotations

import inspect
import types

import pytest

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

import torch  # noqa: E402
from vllm.platforms.interface import DeviceCapability  # noqa: E402

from tests.conftest import FakePPUPlatform  # noqa: E402
from vllm_sail.patch.utils import PATCH_MARKER  # noqa: E402

_COMBINATION_TARGET = (
    "vllm.v1.attention.backends.mla.flashmla_sparse."
    "FlashMLASparseBackend.supports_combination"
)


@pytest.fixture(scope="module")
def installed():
    import vllm_sail.patch as patch_pkg

    patch_pkg.install()
    return patch_pkg.PATCH_REGISTRY


# ---------------------------------------------------------------------------
# NoPE-512 supports_combination gate
# ---------------------------------------------------------------------------


@pytest.fixture
def backend_cls():
    from vllm.v1.attention.backends.mla import flashmla_sparse

    return flashmla_sparse.FlashMLASparseBackend


@pytest.fixture
def patch_module():
    from vllm_sail.patch.enhancement.attention import flashmla_sparse_backend

    return flashmla_sparse_backend


@pytest.fixture
def ppu_platform(monkeypatch, patch_module):
    monkeypatch.setattr(patch_module, "current_platform", FakePPUPlatform("810e"))
    return FakePPUPlatform("810e")


def _install_rope_dim(monkeypatch, rope_dim: int) -> None:
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(qk_rope_head_dim=rope_dim)
        )
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: vllm_config, raising=False
    )


def _supports_combination(
    backend_cls,
    *,
    head_size: int = 512,
    kv_cache_dtype: str | None = "auto",
    capability: tuple[int, int] = (8, 0),
) -> str | None:
    return backend_cls.supports_combination(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=DeviceCapability(*capability),
    )


def test_combination_target_is_patched(installed, backend_cls) -> None:
    records = {record.target: record for record in installed}
    record = records[_COMBINATION_TARGET]
    assert record.kind == "class"
    implementation = inspect.getattr_static(backend_cls, "supports_combination")
    if isinstance(implementation, classmethod):
        implementation = implementation.__func__
    assert getattr(implementation, PATCH_MARKER, None), (
        "supports_combination not patched"
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", None])
def test_nope_512_is_allowed_on_ppu(
    monkeypatch, ppu_platform, backend_cls, kv_cache_dtype
) -> None:
    _install_rope_dim(monkeypatch, 0)
    assert _supports_combination(backend_cls, kv_cache_dtype=kv_cache_dtype) is None


def test_rope_carrying_512_is_still_rejected_on_ppu(
    monkeypatch, ppu_platform, backend_cls
) -> None:
    _install_rope_dim(monkeypatch, 64)
    reason = _supports_combination(backend_cls)
    assert reason is not None and "rope-free (NoPE)" in reason


def test_quantized_512_cache_is_still_rejected_on_ppu(
    monkeypatch, ppu_platform, backend_cls
) -> None:
    _install_rope_dim(monkeypatch, 0)
    reason = _supports_combination(backend_cls, kv_cache_dtype="fp8_ds_mla")
    assert reason is not None and "bf16 kv-cache" in reason


def test_ppu_sm89_capability_hits_the_ppu_gate(
    monkeypatch, patch_module, backend_cls
) -> None:
    """890P reports major 8 minor 9; the workaround keys off major 8.

    Upstream rejects major 8 outright, so ``None`` proves the PPU branch
    answered. The kv-cache dtype and rope gates must still fire there.
    """
    monkeypatch.setattr(patch_module, "current_platform", FakePPUPlatform("890p"))
    _install_rope_dim(monkeypatch, 0)
    assert _supports_combination(backend_cls, capability=(8, 9)) is None

    reason = _supports_combination(
        backend_cls, kv_cache_dtype="fp8_ds_mla", capability=(8, 9)
    )
    assert reason is not None and "bf16 kv-cache" in reason

    _install_rope_dim(monkeypatch, 64)
    reason = _supports_combination(backend_cls, capability=(8, 9))
    assert reason is not None and "rope-free (NoPE)" in reason


def test_non_ppu_major8_still_uses_the_upstream_gate(
    monkeypatch, patch_module, backend_cls
) -> None:
    """The workaround must stay PPU-only; non-PPU keeps the upstream answer."""
    monkeypatch.setattr(
        patch_module, "current_platform", types.SimpleNamespace(is_ppu=lambda: False)
    )
    _install_rope_dim(monkeypatch, 0)
    reason = _supports_combination(backend_cls)
    # Some fork builds already accept major 8 in their upstream gate; whatever
    # that gate answers must be passed through unchanged.
    upstream = patch_module._upstream_supports_combination(
        head_size=512,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=DeviceCapability(8, 0),
    )
    assert reason == upstream


def test_ppu_576_keeps_the_upstream_answer(
    monkeypatch, ppu_platform, backend_cls
) -> None:
    """DeepSeek-style 576 heads are outside this gate and still resolve."""
    _install_rope_dim(monkeypatch, 64)
    assert _supports_combination(backend_cls, head_size=576) is None
