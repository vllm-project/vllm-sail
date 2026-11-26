# SPDX-License-Identifier: Apache-2.0
"""Tests for ``vllm_sail.patch.utils.patch_value``.

``patch_value`` is the imperative counterpart to the ``@patch`` decorator, for
targets that are data rather than callables (module constants, dataclass
instances, lookup tables). Its semantics deliberately differ from ``@patch`` in
one place: re-installing an *equal* value is a no-op rather than an error,
because upstream defining a constant identically is a signal to delete the patch,
not a crash.

Runs with no vLLM and no torch: targets are throwaway modules.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from vllm_sail.patch.utils import PATCH_REGISTRY, patch_value

META = {
    "reason": "test",
    "remove_when": "never, this is a test",
}


@pytest.fixture
def target_module() -> Iterator[types.ModuleType]:
    """A throwaway importable module to patch."""
    name = "vllm_sail_test_value_target"
    module = types.ModuleType(name)
    module.EXISTING = "original"
    sys.modules[name] = module
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def new_records() -> Iterator[list]:
    """Isolate PATCH_REGISTRY assertions from patches installed by other tests.

    PATCH_REGISTRY is process-global and other modules (e.g. test_patch_install)
    legitimately populate it, so assert on the records *this* test added rather
    than on the whole registry.
    """
    before = len(PATCH_REGISTRY)
    added: list = []
    yield added
    added.extend(PATCH_REGISTRY[before:])
    del PATCH_REGISTRY[before:]


def test_installs_new_value(target_module: types.ModuleType) -> None:
    returned = patch_value(
        target_module.__name__, "NEW_CONST", 42, allow_missing=True, **META
    )
    assert returned == 42
    assert target_module.NEW_CONST == 42


def test_records_in_registry(target_module: types.ModuleType) -> None:
    before = len(PATCH_REGISTRY)
    patch_value(target_module.__name__, "NEW_CONST", 42, allow_missing=True, **META)

    assert len(PATCH_REGISTRY) == before + 1
    record = PATCH_REGISTRY[-1]
    assert record.target == f"{target_module.__name__}.NEW_CONST"
    assert record.kind == "value"
    assert record.was_missing is True
    # Data has no source to hash, so drift detection treats it as additive.
    assert record.original_source is None


def test_missing_target_without_allow_missing_raises(
    target_module: types.ModuleType,
) -> None:
    with pytest.raises(AttributeError, match="does not exist"):
        patch_value(target_module.__name__, "ABSENT", 1, **META)


def test_equal_existing_value_is_a_noop(target_module: types.ModuleType) -> None:
    """Upstream defining the same constant must be a clean removal signal."""
    before = len(PATCH_REGISTRY)
    returned = patch_value(target_module.__name__, "EXISTING", "original", **META)

    assert returned == "original"
    assert target_module.EXISTING == "original"
    # A no-op is not a patch, so nothing new is recorded.
    assert len(PATCH_REGISTRY) == before


def test_conflicting_existing_value_raises(target_module: types.ModuleType) -> None:
    with pytest.raises(RuntimeError, match="already exists with a different value"):
        patch_value(target_module.__name__, "EXISTING", "different", **META)


@pytest.mark.parametrize("field", ["reason", "remove_when"])
def test_metadata_is_mandatory(target_module: types.ModuleType, field: str) -> None:
    meta = dict(META) | {field: "   "}
    with pytest.raises(ValueError, match=field):
        patch_value(target_module.__name__, "NEW_CONST", 1, allow_missing=True, **meta)


def test_installs_unhashable_value(target_module: types.ModuleType) -> None:
    """Lookup tables are a real use case, so dicts and lists must work."""
    table = {"a": [1, 2], "b": [3]}
    patch_value(target_module.__name__, "TABLE", table, allow_missing=True, **META)
    assert target_module.TABLE is table
