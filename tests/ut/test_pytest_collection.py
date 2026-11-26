# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for collecting CPU and device tests together."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_same_basename_modules_collect_together(tmp_path: Path) -> None:
    files = []
    for tier in ("ut", "e2e"):
        path = tmp_path / tier / "test_shared_name.py"
        path.parent.mkdir()
        path.write_text(
            "from pathlib import Path\n\n"
            f"def test_origin():\n    assert Path(__file__).parent.name == {tier!r}\n",
            encoding="utf-8",
        )
        files.append(str(path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ROOT / "pyproject.toml"),
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-q",
            *files,
        ],
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


def test_ut_partition_is_disjoint_complete_and_source_marked() -> None:
    code = """
import re
import shlex
from pathlib import Path
import pytest
import yaml
from _pytest.mark.expression import Expression

workflow = yaml.safe_load(Path(".github/workflows/ut.yaml").read_text())
command = next(
    shlex.split(step["run"])
    for step in workflow["jobs"]["cpu-unit-tests"]["steps"]
    if "pytest" in step.get("run", "")
)
cpu = Expression.compile(command[command.index("-m", 3) + 1])
pin_script = Path("scripts/ci/ppu_pin_ut.sh").read_text()
pin = Expression.compile(re.search(r'^\\s+-m "([^"]+)"', pin_script, re.M).group(1))

class UTPartition:
    def pytest_collection_modifyitems(self, items):
        missing = [
            item.nodeid for item in items
            if "upstream_source_root" in item.fixturenames
            and not item.get_closest_marker("upstream_source")
        ]
        if missing:
            raise pytest.UsageError("Missing upstream_source markers: " + ", ".join(missing))
        cpu_ids = {
            item.nodeid for item in items
            if cpu.evaluate(lambda mark: mark in item.keywords)
        }
        pin_ids = {
            item.nodeid for item in items
            if pin.evaluate(lambda mark: mark in item.keywords)
        }
        all_ids = {item.nodeid for item in items}
        assert not cpu_ids & pin_ids, sorted(cpu_ids & pin_ids)
        assert cpu_ids | pin_ids == all_ids, sorted(all_ids - cpu_ids - pin_ids)
        print(f"UT partition: CPU={len(cpu_ids)}, pin={len(pin_ids)}, overlap=0")

raise SystemExit(pytest.main(
    ["tests/ut", "--collect-only", "-qq", "-p", "no:cacheprovider", "-s"],
    plugins=[UTPartition()],
))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UT partition:" in result.stdout
