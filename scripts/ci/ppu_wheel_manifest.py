# SPDX-License-Identifier: Apache-2.0
"""Record and verify the exact wheels passed from the build to PPU E2E."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from email.parser import BytesParser
from pathlib import Path

MANIFEST = "wheel-manifest.json"


def wheel_records(directory: Path) -> dict:
    records = {}
    for name, pattern in (("vllm", "vllm-*.whl"), ("vllm-sail", "vllm_sail-*.whl")):
        wheels = list(directory.rglob(pattern))
        if len(wheels) != 1:
            raise ValueError(f"expected exactly one {name} wheel, found {len(wheels)}")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            metadata_paths = [p for p in members if p.endswith(".dist-info/METADATA")]
            if len(metadata_paths) != 1:
                raise ValueError(f"invalid wheel metadata: {wheel.name}")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            if metadata["Name"].replace("_", "-") != name or not metadata["Version"]:
                raise ValueError(f"unexpected package metadata: {wheel.name}")
            if name == "vllm":
                # Current upstream builds the frontend executable without the
                # removed PyO3 tool-parser bridge. Older wheels may retain it.
                if (
                    "vllm/vllm-rs" not in members
                    or archive.getinfo("vllm/vllm-rs").file_size == 0
                ):
                    raise ValueError("vLLM wheel is missing required Rust artifacts")
            else:
                for module in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C"):
                    if not any(
                        p.startswith(f"vllm_sail/{module}.") and p.endswith(".so")
                        for p in members
                    ):
                        raise ValueError(f"SAIL wheel is missing {module}")
        digest = hashlib.sha256()
        with wheel.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records[name] = {
            "file": wheel.relative_to(directory).as_posix(),
            "version": metadata["Version"],
            "sha256": digest.hexdigest(),
        }
    return records


def verify(
    directory: Path,
    *,
    vllm_commit: str | None = None,
    sail_commit: str | None = None,
) -> dict:
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest.get("schema") != 1:
        raise ValueError("unsupported wheel manifest schema")
    if manifest["wheels"] != wheel_records(directory):
        raise ValueError(
            "wheel filenames, versions or hashes differ from build manifest"
        )
    for name, expected in (("vllm", vllm_commit), ("vllm-sail", sail_commit)):
        if expected is not None and manifest.get("commits", {}).get(name) != expected:
            raise ValueError(
                f"{name} source commit differs from required revision {expected}"
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--sail-source", type=Path)
    parser.add_argument("--vllm-commit", help="require this upstream source commit")
    parser.add_argument("--sail-commit", help="require this plugin source commit")
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify(
            args.directory, vllm_commit=args.vllm_commit, sail_commit=args.sail_commit
        )
    else:
        if args.vllm_source is None or args.sail_source is None:
            parser.error("create requires --vllm-source and --sail-source")
        manifest = {
            "schema": 1,
            "commits": {
                name: subprocess.check_output(
                    [
                        "git",
                        "-c",
                        f"safe.directory={source.resolve()}",
                        "-C",
                        str(source),
                        "rev-parse",
                        "HEAD",
                    ],
                    text=True,
                ).strip()
                for name, source in (
                    ("vllm", args.vllm_source),
                    ("vllm-sail", args.sail_source),
                )
            },
            "wheels": wheel_records(args.directory),
        }
        (args.directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
