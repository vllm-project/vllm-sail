# SPDX-License-Identifier: Apache-2.0
"""Exercise local build dispatch without torch, a toolchain or network access."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_wheels.sh"


@pytest.fixture
def local_builder(tmp_path):
    root = tmp_path / "checkout with spaces"
    (root / "scripts/ci").mkdir(parents=True)
    (root / ".github").mkdir()
    shutil.copy2(SCRIPT, root / "scripts/build_wheels.sh")
    pin = root / ".github/vllm-main-verified.commit"
    pin.write_text("a" * 40 + "\n")
    # Replace only the expensive backend; run the real public entry point.
    (root / "scripts/ci/build_ppu_wheels.sh").write_text(
        'exec "$TEST_PYTHON" -c '
        "'import json, os, sys; print(json.dumps(dict("
        'args=sys.argv[1:], ref=os.environ.get("VLLM_REF"), '
        'reuse=os.environ.get("VLLM_SAIL_NATIVE_REUSE_DIR"), '
        'output=os.environ["ARTIFACT_DIR"])))\' "$@"\n'
    )

    def run(*args):
        return subprocess.run(
            ["bash", str(root / "scripts/build_wheels.sh"), *args],
            cwd=tmp_path,
            env={
                **os.environ,
                "TEST_PYTHON": sys.executable,
                "VLLM_REF": "b" * 40,
                "VLLM_SAIL_NATIVE_REUSE_DIR": "/unwanted-donor",
                "ARTIFACT_DIR": str(tmp_path / "default-output"),
            },
            capture_output=True,
            text=True,
        )

    return run, pin


def test_local_build_dispatches_exact_pin_and_fresh_compile(local_builder, tmp_path):
    run, _ = local_builder
    result = run("--output-dir", "wheel output")
    assert result.returncode == 0, result.stderr
    dispatched = json.loads(result.stdout.splitlines()[-1])
    assert dispatched == {
        "args": ["--prepared-env"],
        "ref": "a" * 40,
        "reuse": None,
        "output": str(tmp_path / "wheel output"),
    }


def test_local_build_preserves_existing_artifacts(local_builder, tmp_path):
    run, _ = local_builder
    output = tmp_path / "old-output"
    output.mkdir()
    wheel = output / "old.whl"
    wheel.write_bytes(b"preserve")
    result = run("--output-dir", str(output))
    assert result.returncode == 2
    assert "output directory must be empty" in result.stderr
    assert wheel.read_bytes() == b"preserve"
    assert list(output.iterdir()) == [wheel]


@pytest.mark.parametrize("pin", ["main", "abc123", "a" * 40 + "\nb" * 40])
def test_local_build_rejects_unpinned_revisions(local_builder, pin):
    run, pin_file = local_builder
    pin_file.write_text(pin)
    result = run()
    assert result.returncode == 2
    assert "expected a full vLLM commit" in result.stderr


def test_local_build_help_does_not_need_build_environment(local_builder):
    run, pin_file = local_builder
    pin_file.unlink()
    result = run("--help")
    assert result.returncode == 0
    assert "VLLM_TARGET_DEVICE=empty" in result.stdout


@pytest.mark.parametrize("args", [("--output-dir",), ("--output-dir", ""), ("--typo",)])
def test_local_build_rejects_invalid_arguments(local_builder, args):
    run, _ = local_builder
    assert run(*args).returncode == 2


def test_prepared_backend_uses_active_sdk_python_and_pip_settings(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    compiler = tools / "custom-hgcc"
    compiler.write_text('#!/bin/sh\nprintf "mock hgcc\\n"\n')
    compiler.chmod(0o755)
    (tools / "uname").write_text('#!/bin/sh\nprintf "Linux\\n"\n')
    (tools / "uname").chmod(0o755)
    (tools / "python").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    print('3.10')\n"
        "else:\n"
        "    keys = ('PPU_SDK', 'HGCC', 'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL')\n"
        "    print('PREFLIGHT=' + json.dumps({k: os.environ[k] for k in keys}))\n"
        "    sys.exit(17)\n"
    )
    (tools / "python").chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/build_ppu_wheels.sh"), "--prepared-env"],
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "PPU_SDK": str(sdk),
            "HGCC": str(compiler),
            "ARTIFACT_DIR": str(output),
            "PIP_INDEX_URL": "https://index.example/simple",
            "PIP_EXTRA_INDEX_URL": "https://extra.example/simple",
        },
        capture_output=True,
        text=True,
    )
    # Stop before dependency installation or fetching any source. The first
    # Python preflight must see the caller's SDK, compiler and pip indexes.
    assert result.returncode == 17, result.stdout + result.stderr
    line = next(s for s in result.stdout.splitlines() if s.startswith("PREFLIGHT="))
    env = json.loads(line.removeprefix("PREFLIGHT="))
    assert env["PPU_SDK"] == str(sdk)
    assert env["HGCC"] == str(compiler)
    assert env["PIP_INDEX_URL"] == "https://index.example/simple"
    assert env["PIP_EXTRA_INDEX_URL"] == "https://extra.example/simple"
    assert "status=failed\nexit_code=17" in (output / "build-manifest.txt").read_text()
