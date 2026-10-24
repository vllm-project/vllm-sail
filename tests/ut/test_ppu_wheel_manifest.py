# SPDX-License-Identifier: Apache-2.0
"""Exercise artifact handoff without torch, vLLM, Rust or a device."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/ppu_wheel_manifest.py"
spec = importlib.util.spec_from_file_location("ppu_wheel_manifest", SCRIPT)
assert spec is not None and spec.loader is not None
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def make_wheels(
    directory: Path, *, rust: bool = True, parser_bridge: bool = True
) -> Path:
    for name, version in (("vllm", "0.30.0+empty"), ("vllm_sail", "0.1.dev1")):
        target = directory / name
        target.mkdir(exist_ok=True)
        wheel = target / f"{name}-{version}-cp312-cp312-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                f"{name}-{version}.dist-info/METADATA",
                f"Name: {name}\nVersion: {version}\n",
            )
            if name == "vllm" and rust:
                archive.writestr("vllm/vllm-rs", b"test executable")
                if parser_bridge:
                    archive.writestr(
                        "vllm/_rust_tool_parser.abi3.so", b"test extension"
                    )
            elif name == "vllm_sail":
                for module in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C"):
                    archive.writestr(f"vllm_sail/{module}.abi3.so", b"test extension")
    return next(directory.rglob("vllm-*.whl"))


def record(directory: Path) -> dict:
    data = {"schema": 1, "wheels": manifest.wheel_records(directory)}
    (directory / manifest.MANIFEST).write_text(json.dumps(data))
    return data


def test_preserves_real_wheel_versions_and_verifies_hashes(tmp_path: Path) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    assert data["wheels"]["vllm"]["version"] == "0.30.0+empty"
    assert manifest.verify(tmp_path) == data


def test_rejects_missing_rust_even_when_wheel_exists(tmp_path: Path) -> None:
    make_wheels(tmp_path, rust=False)
    with pytest.raises(ValueError, match="required Rust artifacts"):
        manifest.wheel_records(tmp_path)


def test_accepts_upstream_frontend_without_removed_parser_bridge(
    tmp_path: Path,
) -> None:
    make_wheels(tmp_path, parser_bridge=False)
    data = record(tmp_path)
    assert manifest.verify(tmp_path) == data


@pytest.mark.parametrize("binary", [None, b""])
def test_parser_bridge_does_not_replace_required_frontend(
    tmp_path: Path, binary: bytes | None
) -> None:
    wheel = make_wheels(tmp_path, rust=False)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("vllm/_rust_tool_parser.abi3.so", b"test extension")
        if binary is not None:
            archive.writestr("vllm/vllm-rs", binary)
    with pytest.raises(ValueError, match="required Rust artifacts"):
        manifest.wheel_records(tmp_path)


def test_rejects_ambiguous_wheel_set(tmp_path: Path) -> None:
    wheel = make_wheels(tmp_path)
    (wheel.parent / "vllm-extra.whl").write_bytes(wheel.read_bytes())
    with pytest.raises(ValueError, match="exactly one vllm wheel"):
        manifest.wheel_records(tmp_path)


def test_rejects_changed_wheel_bytes(tmp_path: Path) -> None:
    wheel = make_wheels(tmp_path)
    record(tmp_path)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("vllm/changed.py", "# substituted wheel\n")
    with pytest.raises(ValueError, match="differ from build manifest"):
        manifest.verify(tmp_path)


def test_rejects_manifest_version_override(tmp_path: Path) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    data["wheels"]["vllm"]["version"] = "0.27.1"
    (tmp_path / manifest.MANIFEST).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differ from build manifest"):
        manifest.verify(tmp_path)


def test_legacy_artifact_without_manifest_fails_closed(tmp_path: Path) -> None:
    make_wheels(tmp_path)
    with pytest.raises(FileNotFoundError):
        manifest.verify(tmp_path)


@pytest.mark.parametrize("stale_package", [None, "vllm", "vllm-sail", "missing"])
def test_cli_requires_the_pr_and_upstream_commits(
    tmp_path: Path, stale_package: str | None
) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    commits = {"vllm": "a" * 40, "vllm-sail": "b" * 40}
    data["commits"] = commits.copy()
    if stale_package == "missing":
        del data["commits"]
    elif stale_package:
        data["commits"][stale_package] = "c" * 40
    (tmp_path / manifest.MANIFEST).write_text(json.dumps(data))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            str(tmp_path),
            "--vllm-commit",
            commits["vllm"],
            "--sail-commit",
            commits["vllm-sail"],
        ],
        capture_output=True,
        text=True,
    )
    if stale_package:
        assert result.returncode != 0
        assert "source commit differs" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("different_owner", [False, True])
def test_cli_records_source_commits(tmp_path: Path, different_owner: bool) -> None:
    make_wheels(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "create",
            str(tmp_path),
            "--vllm-source",
            str(ROOT),
            "--sail-source",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_TEST_ASSUME_DIFFERENT_OWNER": str(int(different_owner)),
        },
    )
    assert result.returncode == 0, result.stderr
    data = manifest.verify(tmp_path)
    assert (
        data["commits"]["vllm"]
        == subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    )
