#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare patched upstream source with the committed per-target baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "patch_baseline.json"
ADDITIVE = "additive"
sys.path.insert(0, str(ROOT))

from tools._common import PatchEnvironmentError, load_patch_records  # noqa: E402


def opaque_target_id(target: str) -> str:
    """Return a deterministic, public-safe opaque identifier for a patch target."""
    return f"t_{hashlib.sha256(target.encode('utf-8')).hexdigest()[:24]}"


def _fingerprints(records: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        result[record.target] = (
            ADDITIVE
            if record.original_source is None
            else hashlib.sha256(record.original_source.encode("utf-8")).hexdigest()
        )
    return dict(sorted(result.items()))


def _read_mapping_file(path: Path, label: str) -> dict[str, str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if isinstance(data, dict):
        for envelope_key in ("opaque_fingerprints", "patch_fingerprints"):
            if isinstance(data.get(envelope_key), dict):
                fp = data[envelope_key]
                if not fp:
                    return None
                data = fp
                break
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ValueError(f"{label} must map target strings to hashes")
    return data


def _read_baseline() -> dict[str, str]:
    try:
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            "tools/patch_baseline.json is missing; run this tool with --update"
        ) from exc
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ValueError("tools/patch_baseline.json must map target strings to hashes")
    return data


def _describe(record: Any) -> str:
    return f"reason={record.reason}; remove when={record.remove_when}"


def evaluate_drift(
    records: list[Any],
    baseline: dict[str, str],
    previous: dict[str, str] | None = None,
) -> dict[str, Any]:
    current = _fingerprints(records)
    added = sorted(current.keys() - baseline.keys())
    removed = sorted(baseline.keys() - current.keys())
    drifted = sorted(
        target
        for target in current.keys() & baseline.keys()
        if current[target] != baseline[target]
    )
    if previous is None:
        incremental_drifted = list(drifted)
        known_soft_drifted: list[str] = []
    else:

        def _prev_hash(target: str) -> str | None:
            if target in previous:
                return previous[target]
            return previous.get(opaque_target_id(target))

        incremental_drifted = sorted(
            target for target in drifted if _prev_hash(target) != current[target]
        )
        known_soft_drifted = sorted(
            target for target in drifted if _prev_hash(target) == current[target]
        )
    if not (added or removed or drifted):
        status = "clean"
    elif added or removed or incremental_drifted:
        status = "incremental_drift"
    else:
        status = "known_soft_drift"
    opaque = dict(sorted((opaque_target_id(k), v) for k, v in current.items()))
    return {
        "status": status,
        "targets_count": len(current),
        "added": added,
        "removed": removed,
        "drifted": drifted,
        "incremental_drifted": incremental_drifted,
        "known_soft_drifted": known_soft_drifted,
        "fingerprints": current,
        "opaque_fingerprints": opaque,
    }


def compare(
    records: list[Any],
    baseline: dict[str, str],
    *,
    previous: dict[str, str] | None = None,
    allow_known_soft_drift: bool = False,
) -> tuple[bool, dict[str, Any]]:
    report = evaluate_drift(records, baseline, previous=previous)
    by_target = {record.target: record for record in records}
    added = report["added"]
    removed = report["removed"]
    drifted = report["drifted"]

    if not (added or removed or drifted):
        print(
            f"OK: {report['targets_count']} patch targets match tools/patch_baseline.json"
        )
        return False, report

    if (
        allow_known_soft_drift
        and previous is not None
        and report["status"] == "known_soft_drift"
    ):
        print(
            f"KNOWN_SOFT_DRIFT: {len(report['known_soft_drifted'])} target(s) "
            "unchanged since previous checked SHA:"
        )
        for target in report["known_soft_drifted"]:
            print(f"SOFT    {target}")
        return False, report

    print("Patch drift detected. Review each target against the pinned upstream:")
    for target in added:
        print(f"ADDED   {target}")
        print(f"        {_describe(by_target[target])}")
    for target in removed:
        print(f"REMOVED {target}")
        print("        delete its stale baseline entry or restore the patch")
    for target in drifted:
        prefix = (
            "SOFT   "
            if previous is not None and target in report["known_soft_drifted"]
            else "DRIFTED"
        )
        print(f"{prefix} {target}")
        print(f"        {_describe(by_target[target])}")
        print(
            "        compare the upstream body, update/remove the patch, then "
            "rerun with --update"
        )
    return True, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace the baseline with hashes from the installed vLLM",
    )
    parser.add_argument(
        "--previous-baseline",
        type=Path,
        default=None,
        help="fingerprints JSON from the previous checked upstream SHA",
    )
    parser.add_argument(
        "--allow-known-soft-drift",
        action="store_true",
        help="exit 0 when drift matches --previous-baseline with no new/removed targets",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="write structured drift evaluation JSON to this file",
    )
    args = parser.parse_args(argv)
    try:
        records = load_patch_records("patch drift checking")
        if args.update:
            current = _fingerprints(records)
            BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {BASELINE} ({len(current)} patch targets)")
            return 0
        previous = (
            _read_mapping_file(args.previous_baseline, "--previous-baseline")
            if args.previous_baseline is not None
            else None
        )
        failed, report = compare(
            records,
            _read_baseline(),
            previous=previous,
            allow_known_soft_drift=args.allow_known_soft_drift,
        )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return 1 if failed else 0
    except (PatchEnvironmentError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
