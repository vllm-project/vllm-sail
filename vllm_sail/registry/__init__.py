# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree registration into vLLM's documented extension points.

Everything here is a *call into an API upstream provides*, not a patch. That
distinction matters: a registration cannot drift when upstream refactors, so each
item that lives here instead of in ``patch/`` is permanently cheaper to maintain.

Four of the in-tree fork's file edits are eliminated outright by this package:

| Fork edit | Replaced by |
|---|---|
| ``kernels/linear/__init__.py`` (+45) | ``register_linear_kernel`` -> ``linear_kernels`` |
| ``kernels/linear/scaled_mm/__init__.py`` (+10) | same |
| ``quantization/__init__.py`` (+4) | ``register_quantization_config`` -> ``quant_config`` |
| ``fused_moe/__init__.py`` (+14, re-exports only) | nothing needed; the plugin owns the classes |

``moe_backends/`` is the exception: upstream has no ``register_moe_backend``, so
those carry real patches. They are the top candidate for an upstream contribution.

Call :func:`register` to perform every registration. Importing this package has
no side effects, so dependency-free helpers below it remain importable on bare
CPU test runners; :func:`register` is idempotent.

Registration mutates vLLM-owned global registries. A partial failure therefore
cannot be rolled back safely; subsequent calls fail with a restart-required
error instead of pretending that retrying in the same process is transactional.
"""

from __future__ import annotations

__all__ = ["register"]

_registered = False
_registration_error: Exception | None = None


def register() -> None:
    """Perform every out-of-tree registration. Idempotent."""
    global _registered, _registration_error
    if _registered:
        return
    if _registration_error is not None:
        raise RuntimeError(
            "PPU registry registration previously failed after possible "
            "partial mutation; restart this process before retrying"
        ) from _registration_error

    from vllm_sail.registry import (
        linear_kernels,
        moe_backends,
        quant_config,
        renderers,
        tuned_configs,
    )

    try:
        # Tuned configs first: the kernels registered below consult them, so a
        # config-directory problem is reported before kernel selection happens.
        tuned_configs.register()
        linear_kernels.register()
        quant_config.register()
        renderers.register()

        # Imported last because the oracles resolve expert classes that the
        # kernel registrations above make selectable.
        moe_backends.register()
    except Exception as exc:
        _registration_error = exc
        raise RuntimeError(
            "PPU registry registration failed after possible partial mutation; "
            f"restart this process before retrying: {exc}"
        ) from exc
    _registered = True
