# SPDX-License-Identifier: Apache-2.0
"""Capture source changes and preserve tracked, new and deleted inputs."""

import hashlib
import json
import subprocess
from pathlib import Path

if not __debug__:
    raise RuntimeError("Source preservation guards require Python without optimization")


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(root), "-C", str(root), *args]
    )


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_relative(name):
    assert isinstance(name, str) and name and "\0" not in name
    path = Path(name)
    assert path.as_posix() == name
    assert not path.is_absolute() and ".." not in path.parts and path.parts
    return path


def checked_path(root, name):
    """Reject symlink ancestors even when a reviewed deleted file is absent."""
    root = Path(root)
    assert root.is_dir() and not root.is_symlink(), root
    relative = checked_relative(name)
    cursor = root
    for component in relative.parts[:-1]:
        cursor = cursor / component
        assert not cursor.is_symlink(), cursor
        assert not cursor.exists() or cursor.is_dir(), cursor
    path = root / relative
    assert not path.is_symlink(), path
    return path


def snapshot(root, base):
    root = Path(root)
    assert git(root, "rev-parse", "HEAD").decode().strip() == base
    assert not git(root, "diff", "--cached", "--name-only")
    statuses = git(root, "diff", "--no-renames", "--name-status", "-z").split(b"\0")
    assert statuses.pop() == b"" and len(statuses) % 2 == 0
    tracked, deleted = {}, {}
    for code, raw in zip(statuses[::2], statuses[1::2], strict=True):
        name = raw.decode()
        path = checked_path(root, name)
        assert not path.is_symlink(), name
        if code == b"D":
            assert not path.exists(), name
            deleted[name] = sha(git(root, "show", base + ":" + name))
        else:
            assert code == b"M" and path.is_file(), (code, name)
            tracked[name] = sha(path.read_bytes())
    untracked = {}
    for raw in git(root, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\0"
    ):
        if not raw:
            continue
        name = raw.decode()
        path = checked_path(root, name)
        assert path.is_file() and not path.is_symlink(), name
        untracked[name] = sha(path.read_bytes())
    assert not set(tracked) & set(untracked)
    return {
        "base_commit": base,
        "diff_sha256": sha(git(root, "diff", "--binary")),
        "tracked_files": tracked,
        "untracked_files": untracked,
        "files": {**tracked, **untracked},
        "deleted_files": deleted,
    }


def preserve(root, expected, destination):
    """Write a new backup only after proving every current source input matches."""
    root, destination = Path(root), Path(destination)
    assert snapshot(root, expected["base_commit"]) == expected
    assert not destination.exists() and not destination.is_symlink()
    destination.mkdir()
    for name, wanted in expected["files"].items():
        data = checked_path(root, name).read_bytes()
        assert sha(data) == wanted
        target = destination / "present" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    for name, wanted in expected["deleted_files"].items():
        data = git(root, "show", expected["base_commit"] + ":" + name)
        assert sha(data) == wanted
        target = destination / "deleted-from-head" / checked_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    patch = git(root, "diff", "--binary")
    assert sha(patch) == expected["diff_sha256"]
    with (destination / "tracked.patch").open("xb") as stream:
        stream.write(patch)
    assert snapshot(root, expected["base_commit"]) == expected
    with (destination / "snapshot.json").open("x") as stream:
        json.dump(expected, stream, indent=2)
        stream.write("\n")


def assert_closed_generated_tree(root, relative):
    """A generator may rebuild only a tree containing known tracked files."""
    root = Path(root)
    folder = checked_path(root, relative)
    assert not folder.is_symlink()
    tracked = {
        raw.decode()
        for raw in git(root, "ls-files", "-z", "--", relative).split(b"\0")
        if raw
    }
    actual = set()
    for path in folder.rglob("*"):
        assert not path.is_symlink(), path
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    assert actual == tracked, {
        "unowned": sorted(actual - tracked),
        "missing": sorted(tracked - actual),
    }
