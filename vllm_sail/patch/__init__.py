# SPDX-License-Identifier: Apache-2.0
"""PPU runtime patches for vLLM.

Call :func:`install` to apply every patch. Importing this package has **no side
effects** — that is deliberate and differs from vLLM-MetaX, whose patch package
installs on import.

Why explicit installation matters:

* ``from vllm_sail.patch.utils import patch`` is what every patch module does. If
  importing the package installed the patches, that line would trigger a partial
  circular import of the package from inside its own initialisation.
* It keeps ``vllm_sail.patch.utils`` importable without vLLM installed, so the
  patch framework itself is unit-testable on a bare CPU runner.
* Installation order and failure handling become visible and testable instead of
  being an emergent property of import order.

Load order is ``bugfix`` -> ``enhancement`` -> ``performance``:

* ``bugfix/`` — defects in upstream vLLM that break PPU. Should disappear as
  upstream fixes land.
* ``enhancement/`` — PPU compatibility shims. The Phase-2 patches live here; they
  are what let PPU reuse upstream's CUDA paths.
* ``performance/`` — replacements that are functionally equivalent but faster on
  PPU. Safe to disable when debugging.

Note the relative imports below. vLLM-MetaX's equivalent module ships
``from utils import patch`` — an *absolute* import that raises
``ModuleNotFoundError: No module named 'utils'`` on a clean interpreter and stops
its patches installing at all.
"""

from __future__ import annotations

from vllm_sail.patch.utils import PATCH_REGISTRY, patch, patch_value

__all__ = ["PATCH_REGISTRY", "install", "installed", "patch", "patch_value"]

#: Categories applied in order. `performance` is last so that disabling it (for
#: debugging) cannot change whether the others applied.
_CATEGORIES = ("bugfix", "enhancement", "performance")

_installed = False


def installed() -> bool:
    """Whether :func:`install` has already run in this process."""
    return _installed


def install() -> None:
    """Apply every PPU patch. Idempotent.

    Called from :func:`vllm_sail.register_out_of_tree`. Requires vLLM to be
    importable; the individual patch modules import their targets.
    """
    global _installed
    if _installed:
        return

    import importlib

    for category in _CATEGORIES:
        module = importlib.import_module(f"{__name__}.{category}")
        if install_category := getattr(module, "install", None):
            install_category()

    _installed = True
