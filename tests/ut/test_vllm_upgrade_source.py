# SPDX-License-Identifier: Apache-2.0
"""Exercise preservation boundaries using disposable real Git histories."""

from __future__ import annotations

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[2] / ".agents/skills/vllm-main-upgrade/scripts"
)


@pytest.fixture(scope="module")
def source_api():
    helper = SCRIPTS / "owned_source.py"
    transition = SCRIPTS / "source_transition.py"
    return (
        helper,
        hashlib.sha256(helper.read_bytes()).hexdigest(),
        runpy.run_path(str(helper)),
        runpy.run_path(str(transition))["restore_reviewed_wip"],
    )


def git(root, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], stderr=subprocess.PIPE
    )


def changed_repo(tmp_path, api):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "codex/fixture")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "config", "core.hooksPath", str(root / ".git/unused-hooks"))
    originals = {
        "csrc/upstream/manifest.toml": b"old pin\n",
        "csrc/upstream/old.cu": b"old native\n",
        "removed/empty": b"",
        "kept.py": b"kept\n",
    }
    for name, data in originals.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Fixture")
    base = git(root, "rev-parse", "HEAD").decode().strip()
    (root / "csrc/upstream/manifest.toml").write_text("new pin\n")
    (root / "csrc/upstream/old.cu").unlink()
    (root / "removed/empty").unlink()
    (root / "removed").rmdir()
    for name in ("new/test.py", "csrc/upstream/new.cu"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new source\n")
    return root, base, originals, api["snapshot"](root, base)


def test_preserve_deleted_modified_and_new_sources_before_restore(tmp_path, source_api):
    helper, digest, api, restore = source_api
    root, base, originals, expected = changed_repo(tmp_path, api)
    backup = tmp_path / "backup"
    receipt = restore(root, expected, backup, contract=helper, contract_sha256=digest)
    assert receipt["base_commit"] == base
    assert not git(root, "status", "--porcelain", "--untracked-files=all")
    for name, data in originals.items():
        assert (root / name).read_bytes() == data
    for name, wanted in expected["files"].items():
        assert api["sha"]((backup / "present" / name).read_bytes()) == wanted
    for name, wanted in expected["deleted_files"].items():
        assert api["sha"]((backup / "deleted-from-head" / name).read_bytes()) == wanted
    assert not (root / "new/test.py").exists()
    assert not (root / "csrc/upstream/new.cu").exists()


@pytest.mark.parametrize(
    "conflict",
    [
        "foreign-untracked",
        "ignored-native",
        "changed-source",
        "staged-source",
        "existing-backup",
        "symlink-parent",
    ],
)
def test_refuse_conflicts_before_any_source_or_backup_write(
    tmp_path, source_api, conflict
):
    helper, digest, api, restore = source_api
    root, base, _, expected = changed_repo(tmp_path, api)
    backup = tmp_path / "backup"
    if conflict == "foreign-untracked":
        (root / "foreign.txt").write_text("foreign work\n")
    elif conflict == "ignored-native":
        (root / ".git/info/exclude").write_text("csrc/upstream/foreign.cu\n")
        (root / "csrc/upstream/foreign.cu").write_text("foreign work\n")
    elif conflict == "changed-source":
        (root / "new/test.py").write_text("changed after review\n")
    elif conflict == "staged-source":
        git(root, "add", "csrc/upstream/manifest.toml")
    elif conflict == "existing-backup":
        backup.mkdir()
        (backup / "keep").write_text("keep previous evidence\n")
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "removed").symlink_to(outside, target_is_directory=True)
        (root / ".git/info/exclude").write_text("removed\n")
    before_diff = git(root, "diff", "--binary")
    before_index = git(root, "diff", "--cached", "--binary")
    before_files = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts and not path.is_symlink()
    }
    with pytest.raises(AssertionError):
        restore(root, expected, backup, contract=helper, contract_sha256=digest)
    assert git(root, "rev-parse", "HEAD").decode().strip() == base
    assert git(root, "diff", "--binary") == before_diff
    assert git(root, "diff", "--cached", "--binary") == before_index
    assert all(
        (root / name).read_bytes() == data for name, data in before_files.items()
    )
    assert backup.exists() == (conflict == "existing-backup")
    if conflict == "existing-backup":
        assert (backup / "keep").read_text() == "keep previous evidence\n"
    if conflict == "symlink-parent":
        assert not list(outside.iterdir())


@pytest.mark.parametrize("name", ["owned_source.py", "source_transition.py"])
def test_optimized_python_refuses_to_disable_preservation_guards(name):
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "import runpy;runpy.run_path(" + repr(str(SCRIPTS / name)) + ")",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "Source preservation guards require Python without optimization"
        in result.stderr
    )
