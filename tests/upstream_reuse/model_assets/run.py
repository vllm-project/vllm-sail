# SPDX-License-Identifier: Apache-2.0
"""Run verified model-asset pytest nodes on one M890 CI shard."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path, PurePosixPath

from tests.upstream_reuse._reuse_harness import UpstreamRunner

HERE = Path(__file__).resolve().parent
MODULE_SECONDS = 240
SHARD_SECONDS = 110 * 60


def load_modules(manifest: Path, pin: str) -> dict[str, list[str]]:
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    if plan["schema"] != 2 or plan["upstream_sha"] != pin:
        raise ValueError(
            "model-asset probe manifest must use schema 2 and match the CI upstream pin"
        )
    modules = plan["modules"]
    if not modules:
        raise ValueError("model-asset probe manifest must select at least one module")
    for module, nodeids in modules.items():
        path = PurePosixPath(module)
        if path.parts[:1] != ("tests",) or ".." in path.parts or path.suffix != ".py":
            raise ValueError(f"invalid upstream test path: {module}")
        if not nodeids or len(set(nodeids)) != len(nodeids):
            raise ValueError(f"empty or duplicated selection: {module}")
        for nodeid in nodeids:
            definition, params, _ = nodeid.partition("[")
            if not all(part.isidentifier() for part in definition.split("::")) or (
                params and not nodeid.endswith("]")
            ):
                raise ValueError(f"invalid module-relative pytest node ID: {module}")
    return modules


def shard_modules(modules: dict[str, list[str]], count: int, index: int) -> list[str]:
    if not 0 <= index < count:
        raise ValueError(f"invalid probe shard {index} for {count} shards")
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0] * count
    for module in sorted(modules, key=lambda name: (-len(modules[name]), name)):
        target = min(range(count), key=loads.__getitem__)
        shards[target].append(module)
        loads[target] += 10 + len(modules[module])
    return shards[index]


def _kill_process_tree(proc: subprocess.Popen) -> None:
    # Upstream server fixtures can start their own process groups. Capture and
    # stop their descendants before killing the parent that owns them.
    import psutil

    with suppress(psutil.NoSuchProcess):
        for child in reversed(psutil.Process(proc.pid).children(recursive=True)):
            with suppress(psutil.NoSuchProcess):
                child.kill()
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _passed_nodeids(events: Path) -> set[str]:
    phases: dict[str, set[str]] = {}
    nonpassing = set()
    if not events.is_file():
        return set()
    with events.open(encoding="utf-8") as stream:
        for line in stream:
            report = json.loads(line)
            if report["event"] != "test_report":
                continue
            nodeid = report["nodeid"]
            if report["outcome"] == "passed" and not report["xfail"]:
                phases.setdefault(nodeid, set()).add(report["phase"])
            else:
                nonpassing.add(nodeid)
    return {
        nodeid
        for nodeid, passed in phases.items()
        if passed == {"setup", "call", "teardown"} and nodeid not in nonpassing
    }


def run_module(
    view: Path,
    module: str,
    nodeids: list[str],
    output: Path,
    timeout: float,
) -> dict:
    expected_nodeids = {f"{module}::{nodeid}" for nodeid in nodeids}
    events = output / "events.jsonl"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join((str(view), str(HERE))),
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        SAIL_ASSET_NODEIDS=json.dumps(sorted(expected_nodeids)),
        SAIL_ASSET_EVENTS=str(events),
    )
    # ppu_upstream_reuse.sh verifies the wheels and device in a separate process.
    # Leave CUDA uninitialized here for upstream tests that fork before using it.
    command = [
        sys.executable,
        "-m",
        "pytest",
        module,
        "-p",
        "_sail_asset_cases",
        "-p",
        "no:cacheprovider",
        "-q",
        "-rs",
        "--optional",
        "--tb=short",
        "--continue-on-collection-errors",
    ]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "pytest.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=view,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "seconds": timeout}
        finally:
            if proc.poll() is None:
                _kill_process_tree(proc)
    if code == 0:
        missing_passes = expected_nodeids - _passed_nodeids(events)
        if missing_passes:
            return {"status": "not_passed", "nodeids": sorted(missing_passes)}
    return {"status": "completed", "exit_code": code}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()

    pin = os.environ["EXPECTED_VLLM_COMMIT"]
    modules = load_modules(HERE / "manifest.json", pin)
    assigned = shard_modules(modules, args.shards, args.shard)
    output = Path(os.environ["REUSE_RESULTS_DIR"]) / "model-assets"
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "upstream_sha": pin,
        "sail_sha": os.environ["TESTED_SAIL_COMMIT"],
        "shard": args.shard,
        "shards": args.shards,
        "modules": [
            {"module": module, "nodeids": modules[module], "artifact": str(index)}
            for index, module in enumerate(assigned)
        ],
    }
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    runner = UpstreamRunner(Path(os.environ["VLLM_SOURCE_ROOT"]), strict=True)
    counts = Counter()
    deadline = time.monotonic() + SHARD_SECONDS
    try:
        view = runner.runtime_root()
        # Some upstream tests read chat templates relative to the test tree.
        (view / "examples").symlink_to(
            runner.root / "examples", target_is_directory=True
        )
        with (output / "module-results.jsonl").open("a", encoding="utf-8") as progress:
            for row in plan["modules"]:
                module = row["module"]
                remaining = deadline - time.monotonic()
                if remaining < 1:
                    result = {"status": "not_run", "reason": "shard time budget"}
                else:
                    result = run_module(
                        view,
                        module,
                        row["nodeids"],
                        output / row["artifact"],
                        min(MODULE_SECONDS, remaining),
                    )
                progress.write(json.dumps({"module": module, **result}) + "\n")
                progress.flush()
                result_status = result["status"]
                if result_status == "completed":
                    result_status = (
                        "pytest_ok" if result["exit_code"] == 0 else "pytest_error"
                    )
                counts[result_status] += 1
                print(f"model-assets {args.shard}: {module} {result}", flush=True)
    finally:
        runner.close()
    print(f"model-asset module results: {dict(counts)}", flush=True)
    return int(counts["pytest_ok"] != len(assigned))


if __name__ == "__main__":
    raise SystemExit(main())
