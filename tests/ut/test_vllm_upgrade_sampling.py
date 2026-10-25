# SPDX-License-Identifier: Apache-2.0
"""Freeze calendar boundaries, release topology and immutable output behavior."""

from __future__ import annotations

import runpy
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SAMPLER = (
    Path(__file__).resolve().parents[2]
    / ".agents/skills/vllm-main-upgrade/scripts/sample_commits.py"
)
ZONE = ZoneInfo("Asia/Shanghai")


def epoch(value):
    return int(datetime.fromisoformat(value).timestamp())


def test_calendar_eod_uses_timezone_and_keeps_off_main_release_endpoint():
    api = runpy.run_path(str(SAMPLER))
    times = {
        "start": epoch("2026-10-03T12:00:00+08:00"),
        "release": epoch("2026-10-08T18:00:00+08:00"),
        "target": epoch("2026-10-10T10:00:00+08:00"),
    }
    history = [
        ("start", times["start"]),
        ("late05", epoch("2026-10-05T23:59:59+08:00")),
        ("midnight06", epoch("2026-10-06T00:00:00+08:00")),
        ("late07", epoch("2026-10-07T23:59:59+08:00")),
        ("main08", epoch("2026-10-08T17:00:00+08:00")),
        ("late09", epoch("2026-10-09T23:59:59+08:00")),
        ("target", times["target"]),
        ("later10", epoch("2026-10-10T11:00:00+08:00")),
    ]
    points = api["samples"](history, times, "start", "release", "target", ZONE)
    assert [p["sha"] for p in points] == [
        "start",
        "late05",
        "late07",
        "release",
        "late09",
        "target",
    ]
    assert [p["index"] for p in points] == list(range(6))
    assert points[1]["date"] == "2026-10-05"
    assert points[3]["roles"] == ["release"]
    assert points[-1]["roles"] == ["calendar-sample", "frozen-main"]


def test_sparse_days_deduplicate_sha_without_losing_endpoint_roles():
    api = runpy.run_path(str(SAMPLER))
    times = {
        "start": epoch("2026-10-03T12:00:00+08:00"),
        "release": epoch("2026-10-08T18:00:00+08:00"),
        "target": epoch("2026-10-10T10:00:00+08:00"),
    }
    history = [
        ("start", times["start"]),
        ("only", epoch("2026-10-04T00:00:00+08:00")),
        ("target", times["target"]),
    ]
    points = api["samples"](history, times, "start", "release", "target", ZONE)
    assert [p["sha"] for p in points] == ["start", "only", "release", "target"]
    assert points[1]["date"] == "2026-10-05"
    assert points[-1]["roles"] == ["calendar-sample", "frozen-main"]


@pytest.mark.parametrize("before,after", [(0, 1), (2, 0), (-1, 1)])
def test_nonpositive_intervals_are_rejected(before, after):
    api = runpy.run_path(str(SAMPLER))
    with pytest.raises(ValueError, match="positive calendar days"):
        api["samples"]([], {"a": 1, "b": 2, "c": 3}, "a", "b", "c", ZONE, before, after)


def test_out_of_order_release_timestamps_are_rejected():
    api = runpy.run_path(str(SAMPLER))
    with pytest.raises(ValueError, match="precede the frozen target"):
        api["samples"]([], {"a": 3, "b": 2, "c": 4}, "a", "b", "c", ZONE)


def test_cli_does_not_overwrite_a_preexisting_plan(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return (
            subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.PIPE
            )
            .decode()
            .strip()
        )

    git("init", "-q", "-b", "codex/fixture")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", str(root / ".git/unused-hooks"))
    (root / "source").write_text("original\n")
    git("add", ".")
    git("commit", "-q", "-m", "Fixture")
    sha = git("rev-parse", "HEAD")
    plan = tmp_path / "plan.json"
    sentinel = b"previous evidence must survive\n"
    plan.write_bytes(sentinel)
    result = subprocess.run(
        [
            sys.executable,
            str(SAMPLER),
            "--repo",
            str(root),
            "--start",
            sha,
            "--release",
            sha,
            "--target",
            sha,
            "--output",
            str(plan),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert plan.read_bytes() == sentinel
    assert git("rev-parse", "HEAD") == sha
    assert not git("status", "--porcelain", "--untracked-files=all")
