# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_sail.compat.

``vllm_sail.compat`` imports ``vllm.logger``, which is not installed in the
CPU-only test environment, so a minimal stub is installed for the duration of
each test rather than at collection time. The stub is a real module object with
exactly the one attribute compat needs — not a MagicMock — so an unexpected new
dependency on vLLM surfaces as an AttributeError here instead of being silently
absorbed.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import types
from collections.abc import Iterator

import pytest

PIN = "528772a4bf589fd52eec40bc8ca79c84d99206b5"


@pytest.fixture
def compat(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import ``vllm_sail.compat`` against a stubbed ``vllm.logger``."""
    vllm = sys.modules.get("vllm")
    if vllm is None:
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []  # mark as a package so submodule imports resolve
        monkeypatch.setitem(sys.modules, "vllm", vllm)

    if "vllm.logger" not in sys.modules:
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = logging.getLogger
        monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)

    monkeypatch.delenv("VLLM_SAIL_STRICT_COMPAT", raising=False)
    monkeypatch.delenv("VLLM_PPU_STRICT_COMPAT", raising=False)
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.delitem(sys.modules, "vllm_sail.compat", raising=False)
    module = importlib.import_module("vllm_sail.compat")

    from vllm_sail.native import manifest

    monkeypatch.setattr(
        manifest, "load", lambda: types.SimpleNamespace(upstream_ref=PIN)
    )
    yield module
    # Leave no cached copy behind: the next test may want a different stub.
    sys.modules.pop("vllm_sail.compat", None)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.31.0", "0.31"),
        ("v0.31.0", "0.31"),
        ("0.31", "0.31"),
        ("0.31.1+cu128", "0.31"),
        ("0.31.0rc1", "0.31"),
        ("0.1.dev1+g4bdc8a788", "0.1"),
    ],
)
def test_release_line_parses(compat, version: str, expected: str) -> None:
    assert compat._release_line(version) == expected


@pytest.mark.parametrize("version", ["unknown", "", "0", "dev", "abc", "0.²"])
def test_release_line_unparseable(compat, version: str) -> None:
    assert compat._release_line(version) is None


def test_release_tag_uses_version_env_override(
    compat, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compat, "_upstream_ref", lambda: "v0.33.0")
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.1.dev1")
    monkeypatch.setenv("VLLM_VERSION", " 0.33.0 ")
    compat.check_vllm_compatibility(strict=True)
    assert compat._last_matched


@pytest.mark.parametrize("override", ["", "   "])
def test_release_tag_blank_version_env_override_falls_back(
    compat, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setattr(compat, "_upstream_ref", lambda: "v0.33.0")
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.33.1")
    monkeypatch.setenv("VLLM_VERSION", override)
    compat.check_vllm_compatibility(force=True, strict=True)
    assert compat._last_matched


def test_named_ref_ignores_version_env_override(compat, monkeypatch) -> None:
    monkeypatch.setattr(compat, "_upstream_ref", lambda: "main")
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.33.0")
    monkeypatch.setenv("VLLM_VERSION", "main")
    with pytest.raises(RuntimeError, match="vLLM ref 'main'"):
        compat.check_vllm_compatibility(strict=True)


def test_no_hardcoded_version_range_constants(compat) -> None:
    """Compatibility derives solely from manifest upstream_ref, not static bounds."""
    for removed in (
        "MIN_VLLM_VERSION",
        "MAX_VLLM_VERSION_EXCLUSIVE",
        "vllm_version_is",
        "vllm_version_at_least",
    ):
        assert not hasattr(compat, removed)


def test_exact_main_pin_reads_packaged_kernel_metadata(compat, monkeypatch) -> None:
    from vllm_sail.native import manifest

    monkeypatch.setattr(
        manifest, "load", lambda: types.SimpleNamespace(upstream_ref=PIN)
    )
    assert compat._upstream_ref() == PIN
    assert compat._is_commit_sha(PIN)


@pytest.mark.parametrize("dirty", [False, True])
def test_exact_pin_accepts_generated_scm_version(
    compat, monkeypatch, tmp_path, dirty: bool
) -> None:
    """Exercise upstream's full-hash SCM command, including dirty build inputs."""
    from setuptools_scm import get_version

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git("config", "user.name", "SCM test")
    git("config", "user.email", "scm@example.com")
    source = tmp_path / "source.py"
    source.write_text("initial\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "initial")
    git("tag", "v0.33.0")
    source.write_text("checkpoint\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "checkpoint")
    pin = git("rev-parse", "HEAD")
    if dirty:
        source.write_text("prepared build inputs\n")
    for name in (
        "VLLM_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM",
    ):
        monkeypatch.delenv(name, raising=False)
    version = get_version(
        root=str(tmp_path),
        git_describe_command=[
            "git",
            "describe",
            "--dirty",
            "--tags",
            "--long",
            "--abbrev=40",
            "--match",
            "v[0-9]*",
        ],
    )
    monkeypatch.setattr(compat, "_upstream_ref", lambda: pin)
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: version)
    compat.check_vllm_compatibility(force=True, strict=True)
    assert compat._checked
    assert compat._last_matched


def test_exact_main_pin_accepts_its_real_scm_version(compat, monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "installed_vllm_version",
        lambda: f"0.33.0rc1.dev672+g{PIN}.d20261106.empty",
    )
    compat.check_vllm_compatibility(force=True, strict=True)
    assert compat._checked
    assert compat._last_matched


@pytest.mark.parametrize(
    "version",
    ["0.33.0", "0.33.1.dev1+g528772a", "0.33.1.dev1+g" + "a" * 40],
)
def test_default_mode_warns_without_raising_on_commit_mismatch(
    compat,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    version: str,
) -> None:
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: version)
    with caplog.at_level(logging.WARNING):
        compat.check_vllm_compatibility(force=True)
    assert PIN in caplog.text
    assert version in caplog.text
    assert compat._checked is True
    assert compat._last_matched is False


@pytest.mark.parametrize(
    "version",
    ["0.33.0", "0.33.1.dev1+g528772a", "0.33.1.dev1+g" + "a" * 40],
)
def test_strict_mode_rejects_missing_or_different_commit(
    compat, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: version)
    with pytest.raises(RuntimeError, match="verified vLLM commit"):
        compat.check_vllm_compatibility(force=True, strict=True)
    assert compat._checked is False
    assert compat._last_matched is False


@pytest.mark.parametrize(
    "env_name", ["VLLM_SAIL_STRICT_COMPAT", "VLLM_PPU_STRICT_COMPAT"]
)
def test_strict_env_var_enables_fail_closed_check(
    compat, monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    monkeypatch.setenv(env_name, "1")
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.33.0")
    with pytest.raises(RuntimeError, match="verified vLLM commit"):
        compat.check_vllm_compatibility(force=True)
    # Explicit strict=False still overrides the env var.
    compat.check_vllm_compatibility(force=True, strict=False)


@pytest.mark.parametrize("override", ["0.33.0", f"0.33.0.dev1+g{PIN}"])
def test_exact_main_pin_ignores_version_env_override(
    compat, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("VLLM_VERSION", override)
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.33.0")
    with pytest.raises(RuntimeError, match="verified vLLM commit"):
        compat.check_vllm_compatibility(strict=True)
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: f"0.33.0.dev1+g{PIN}")
    compat.check_vllm_compatibility(strict=True)


@pytest.mark.parametrize(
    "version",
    ["0.33.0", "0.33.1+cu128", "0.33.0rc1", "0.33.1.dev5+g1234567"],
)
def test_release_tag_manifest_accepts_matching_minor_line(
    compat,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    version: str,
) -> None:
    monkeypatch.setattr(compat, "_upstream_ref", lambda: "v0.33.0")
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: version)
    with caplog.at_level(logging.WARNING):
        compat.check_vllm_compatibility(force=True, strict=True)
    assert not caplog.text
    assert compat._checked is True
    assert compat._last_matched is True


@pytest.mark.parametrize(
    "version",
    ["0.32.0", "0.34.0", "0.1.dev1+g4bdc8a788", "dev", "unknown"],
)
def test_release_tag_manifest_warns_by_default_and_raises_in_strict_mode(
    compat,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    version: str,
) -> None:
    monkeypatch.setattr(compat, "_upstream_ref", lambda: "v0.33.0")
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: version)
    with caplog.at_level(logging.WARNING):
        compat.check_vllm_compatibility(force=True)
    assert "0.33.x" in caplog.text
    with pytest.raises(RuntimeError, match="expected 0.33.x"):
        compat.check_vllm_compatibility(force=True, strict=True)


def test_default_warn_is_cached_but_strict_still_raises(
    compat, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Default mode warns once per process; strict mode still fails without force=True."""
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.33.0")
    with caplog.at_level(logging.WARNING):
        compat.check_vllm_compatibility()
        compat.check_vllm_compatibility()
    assert len(caplog.records) == 1

    with pytest.raises(RuntimeError, match="verified vLLM commit"):
        compat.check_vllm_compatibility(strict=True)
    assert compat._checked is False


def test_failed_strict_check_is_not_cached(
    compat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caught strict failure must not make the next call skip validation."""
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.33.0")
    with pytest.raises(RuntimeError, match="verified vLLM commit"):
        compat.check_vllm_compatibility(force=True, strict=True)

    monkeypatch.setattr(
        compat, "installed_vllm_version", lambda: f"0.33.0rc1.dev1+g{PIN}"
    )
    compat.check_vllm_compatibility(strict=True)
    assert compat._checked is True
    assert compat._last_matched is True


def test_matched_check_is_cached_without_force(
    compat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A passing check runs once per process unless force=True."""
    monkeypatch.setattr(
        compat, "installed_vllm_version", lambda: f"0.33.0rc1.dev1+g{PIN}"
    )
    compat.check_vllm_compatibility(force=True, strict=True)

    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.99.0")
    compat.check_vllm_compatibility(strict=True)  # cached pass

    with pytest.raises(RuntimeError):
        compat.check_vllm_compatibility(force=True, strict=True)
    assert compat._checked is False
