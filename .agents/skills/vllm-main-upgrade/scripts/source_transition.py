# SPDX-License-Identifier: Apache-2.0
"""Preserve a reviewed source snapshot before restoring its owned paths."""

from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

if not __debug__:
    raise RuntimeError("Source preservation guards require Python without optimization")


def restore_reviewed_wip(root, expected, destination, *, contract, contract_sha256):
    """Preserve an exact owned WIP, then restore only its reviewed paths to HEAD."""
    root, destination, contract = Path(root), Path(destination), Path(contract)
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == contract_sha256
    api = runpy.run_path(str(contract))
    git, digest, checked = api["git"], api["sha"], api["checked_path"]
    assert root.is_absolute() and destination.is_absolute()
    assert destination.parent.is_dir() and not destination.parent.is_symlink()
    assert not destination.resolve().is_relative_to(root.resolve())
    assert not destination.exists() and not destination.is_symlink()
    assert api["snapshot"](root, expected["base_commit"]) == expected

    def generated_tree_is_reviewed():
        relative = "csrc/upstream"
        folder = checked(root, relative)
        tracked = {
            name.decode()
            for name in git(root, "ls-files", "-z", "--", relative).split(b"\0")
            if name
        }
        present = (tracked - set(expected["deleted_files"])) | {
            name
            for name in expected["untracked_files"]
            if name.startswith(relative + "/")
        }
        actual = set()
        for path in folder.rglob("*"):
            assert not path.is_symlink(), path
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
        assert actual == present, {
            "unowned": sorted(actual - present),
            "missing": sorted(present - actual),
        }

    generated_tree_is_reviewed()
    api["preserve"](root, expected, destination)
    # Recheck the full contract after the complete backup and before cleanup.
    assert api["snapshot"](root, expected["base_commit"]) == expected
    generated_tree_is_reviewed()
    for name, wanted in expected["untracked_files"].items():
        assert digest(checked(root, name).read_bytes()) == wanted, name
        assert digest((destination / "present" / name).read_bytes()) == wanted, name
    restore_paths = sorted(
        set(expected["tracked_files"]) | set(expected["deleted_files"])
    )
    if restore_paths:
        git(root, "restore", "--source=HEAD", "--worktree", "--", *restore_paths)
    for name, wanted in expected["untracked_files"].items():
        path = checked(root, name)
        assert path.is_file() and digest(path.read_bytes()) == wanted, name
        assert digest((destination / "present" / name).read_bytes()) == wanted, name
        path.unlink()
    assert git(root, "rev-parse", "HEAD").decode().strip() == expected["base_commit"]
    assert not git(root, "status", "--porcelain", "--untracked-files=all")
    api["assert_closed_generated_tree"](root, "csrc/upstream")
    return {
        "status": "owned-wip-preserved-and-restored-to-base",
        "base_commit": expected["base_commit"],
        "backup": str(destination),
        "backup_snapshot_sha256": digest((destination / "snapshot.json").read_bytes()),
        "restored_tracked_paths": restore_paths,
        "preserved_then_removed_untracked": sorted(expected["untracked_files"]),
    }
