# SPDX-License-Identifier: Apache-2.0
"""Run installed-package PPU operator tests and validate evidence on CPU; load device dependencies only in isolated subprocesses."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select = load(Path(__file__).with_name("ppu_ops_select.py"))
require = select.require


def result_path(run_id, attempt, group_id, *, pod):
    select.decimal(run_id)
    select.decimal(attempt)
    require(group_id in {b + "-0" for b in select.BOARDS}, "Invalid execution group")
    return (
        Path("/mnt/wl_nas" if pod else "/wl_nas")
        / "devops"
        / f"{run_id}-{attempt}"
        / "ops-tests"
        / group_id
    )


def require_nas(mount):
    require(
        any(os.path.ismount(p) for p in (mount, mount / "devops")), "Missing NAS mount"
    )
    require(
        (mount / "devops").is_dir() and os.access(mount / "devops", os.W_OK),
        "NAS is not writable",
    )


def copy_test_tree(repo, isolated):
    repo, isolated = Path(repo).resolve(), Path(isolated).resolve()
    require(
        not isolated.is_relative_to(repo),
        "Isolated test tree must be outside the checkout",
    )
    for path in (repo / "tests").rglob("*"):
        require(not path.is_symlink(), "Test tree must not contain symbolic links")
    shutil.copytree(
        repo / "tests",
        isolated / "tests",
        ignore=shutil.ignore_patterns("ut", "__pycache__", "*.pyc"),
    )
    shutil.copyfile(repo / "pyproject.toml", isolated / "pyproject.toml")


def clean_environment(environment, repo, isolated):
    env = {
        k: v
        for k, v in environment.items()
        if k
        not in {
            "PYTHONPATH",
            "PYTEST_ADDOPTS",
            "VLLM_VERSION",
            "VLLM_VERSION_OVERRIDE",
            "VLLM_PPU_USE_PLA",
            "VLLM_SOURCE_ROOT",
            "VLLM_USE_PRECOMPILED",
            "VLLM_USE_PRECOMPILED_RUST",
            "VLLM_PRECOMPILED_WHEEL_LOCATION",
            "PYTHONSTARTUP",
            "PYTHONHOME",
        }
        and not k.startswith("SETUPTOOLS_SCM_PRETEND_VERSION")
    }
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        VLLM_SAIL_USE_PLA="1",
        VLLM_PLUGINS="ppu",
        VLLM_USE_RUST_FRONTEND="0",
        VLLM_USE_RUST_BENCH="0",
        PPU_CI_CHECKOUT=str(Path(repo).resolve()),
        PPU_CI_ISOLATED_ROOT=str(Path(isolated).resolve()),
    )
    return env


def effective_selection(config, original, original_bytes, board, expanded):
    groups = copy.deepcopy(original["groups"])
    if expanded:
        all_groups = select.route_tests(
            config, [t["id"] for t in config["test_catalog"]]
        )["groups"]
        replacement = next(g for g in all_groups if g["board"] == board)
        groups = [replacement if g["board"] == board else g for g in groups]
    return {
        "schema": 1,
        "original_selection_sha256": select.checksum(original_bytes),
        "original_mode": original["mode"],
        "expanded_boards": [board] if expanded else [],
        "groups": groups,
        "matrix": copy.deepcopy(original["matrix"]),
        "test_ids": sorted({t for g in groups for t in g["test_ids"]}),
        "files": sorted({f for g in groups for f in g["files"]}),
    }


def validate_file_evidence(directory, entry, board):
    directory = Path(directory)
    node_path = directory / "nodes" / (entry["id"] + ".json")
    report = json.loads(node_path.read_bytes())
    load(ROOT / "tests/support/ppu_ci.py").validate_node_report(report, entry, board)
    log = directory / "logs" / (entry["id"] + ".log")
    require(log.is_file() and log.stat().st_size > 0, "Missing test file log")
    try:
        tree = ET.parse(directory / "junit" / (entry["id"] + ".xml"))
    except ET.ParseError as error:
        raise ValueError("Malformed JUnit XML") from error
    cases = tree.findall(".//testcase")
    require(
        not any(tree.findall(".//" + tag) for tag in ("failure", "error", "skipped")),
        "JUnit contains failures, errors, or skips",
    )
    actual = [(case.get("classname"), case.get("name")) for case in cases]
    expected = []
    for node in report["applicable"]:
        parts = node.split("::")
        expected.append(
            (".".join([parts[0][:-3].replace("/", "."), *parts[1:-1]]), parts[-1])
        )
    require(sorted(actual) == sorted(expected), "JUnit node coverage mismatch")
    return report


def run_files(
    entries, board, config_path, isolated, output, env, *, clock=time.monotonic
):
    results = []
    for kind in ("logs", "nodes", "junit"):
        (output / kind).mkdir(exist_ok=True)
    for entry in entries:
        started = clock()
        test_id = entry["id"]
        log_path = output / "logs" / (test_id + ".log")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(isolated / "pyproject.toml"),
            "-p",
            "tests.support.ppu_ci",
            entry["test"],
            "-q",
            "-rs",
            "--ppu-ci-config",
            str(config_path),
            "--ppu-ci-board",
            board,
            "--ppu-ci-report",
            str(output / "nodes" / (test_id + ".json")),
            "--junitxml",
            str(output / "junit" / (test_id + ".xml")),
        ]
        outcome = "failure"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=isolated,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                require(result.returncode == 0, f"pytest exit={result.returncode}")
                log.flush()
                validate_file_evidence(output, entry, board)
                outcome = "success"
            except (OSError, ValueError, KeyError, TypeError) as error:
                log.write(f"\nCI evidence validation failed: {error}\n")
        results.append(
            {
                "test_id": test_id,
                "file": entry["test"],
                "duration_seconds": clock() - started,
                "outcome": outcome,
            }
        )
    return results


def runtime_profile(runtime, packages, sdk_output, smi_output):
    driver = re.search(r"Driver Version:\s*([0-9][0-9.]*)", smi_output)
    require(driver is not None, "Unable to determine the driver identity reliably")
    profile = {
        "capability": runtime["capability"],
        "runtime_env": runtime["runtime_env"],
        "packages": packages,
        "sdk_identity": sdk_output.strip(),
        "driver_identity": driver.group(1),
    }
    select.validate_runtime_profile(profile)
    return profile


def gate_passes(selector, mode, device, *, has_tests, reports_complete):
    if selector != "success":
        return False
    if mode == "none":
        return not has_tests and device == "skipped"
    return (
        mode in ("subset", "full")
        and has_tests
        and device == "success"
        and reports_complete
    )


def validate_group(
    config, original_bytes, directory, run_id, attempt, *, require_success=True
):
    original = json.loads(original_bytes)
    directory = Path(directory)

    def read(name):
        return json.loads((directory / name).read_bytes())

    summary = read("summary.json")
    group_id = summary["group_id"]
    planned = next((g for g in original["groups"] if g["group_id"] == group_id), None)
    require(planned is not None, "Unknown execution group")
    board = planned["board"]
    identity = {
        "run_id": select.decimal(run_id),
        "run_attempt": select.decimal(attempt),
        "group_id": group_id,
        "board": board,
        "tested_sha": original["tested_sha"],
    }
    for name in ("summary.json", "environment.json", "timings.json"):
        value = read(name)
        require(
            value.get("schema") == 1
            and all(value.get(k) == v for k, v in identity.items()),
            f"Report identity mismatch: {name}",
        )
    if require_success:
        require(
            summary["status"] == "success"
            and summary["failure_stage"] is None
            and summary["reason"] is None,
            "Execution group did not succeed",
        )
    require(read("selection.json") == original, "Original selection was modified")
    environment = read("environment.json")
    select.validate_runtime_profile(environment["profile"])
    require(
        environment["runtime_image"] == original["runtime_image"],
        "Runtime image evidence mismatch",
    )
    require(type(environment["expanded"]) is bool, "Invalid expansion flag")
    require(
        environment["expansion_reason"]
        == ("runtime_environment_changed" if environment["expanded"] else None),
        "Expansion reason mismatch",
    )
    expected = effective_selection(
        config, original, original_bytes, board, environment["expanded"]
    )
    effective = read("effective-selection.json")
    require(
        effective == expected,
        "Effective selection is missing, duplicated, or does not preserve the original plan",
    )
    group = next(g for g in effective["groups"] if g["group_id"] == group_id)
    catalog = {t["id"]: t for t in config["test_catalog"]}
    timings = read("timings.json")
    for key in ("config_sha256", "build_fingerprint", "runtime_image"):
        require(timings[key] == original[key], "Timing metadata identity mismatch")
    require(timings["nproc_per_node"] == 1, "Timing device count mismatch")
    select.keys(
        timings["phases"],
        (
            "queue_seconds",
            "install_seconds",
            "preflight_seconds",
            "device_occupied_seconds",
            "total_seconds",
        ),
    )
    for value in timings["phases"].values():
        require(
            (value is None and bool(timings.get("timing_error")))
            or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
            "Missing or invalid timing",
        )
    require(
        [r["test_id"] for r in timings["files"]] == group["test_ids"],
        "Missing or duplicate test file execution",
    )
    for record in timings["files"]:
        entry = catalog[record["test_id"]]
        require(
            record["file"] == entry["test"] and record["outcome"] == "success",
            "Test file did not succeed",
        )
        duration = record["duration_seconds"]
        require(
            type(duration) in (int, float)
            and math.isfinite(duration)
            and duration >= 0,
            "Invalid test file timing",
        )
        validate_file_evidence(directory, entry, board)
    fingerprint = select.build_fingerprint(
        select.read_diagnostics(directory / "build-manifest.txt"),
        read("wheel-manifest.json"),
        tested_sha=original["tested_sha"],
        expected_vllm_commit=original["build_fingerprint"]["vllm_commit"],
        build_image=original["build_fingerprint"]["build_image"],
    )
    require(fingerprint == original["build_fingerprint"], "Build fingerprint mismatch")
    return group_id


def validate_results(config, original_bytes, directories, run_id, attempt):
    expected = [g["group_id"] for g in json.loads(original_bytes)["groups"]]
    require(bool(expected), "Summary is missing execution groups")
    actual = [
        validate_group(config, original_bytes, d, run_id, attempt) for d in directories
    ]
    require(
        sorted(actual) == sorted(expected),
        "Missing, duplicate, or unknown execution groups",
    )


def read_inputs(env):
    required = (
        "WHEELS_DIR",
        "BUILD_MANIFEST_FILE",
        "SELECTION_FILE",
        "DEVICE_RESULTS_DIR",
        "DEVICE_BOARD",
        "DEVICE_GROUP_ID",
        "EXPECTED_SAIL_COMMIT",
        "EXPECTED_VLLM_COMMIT",
        "EXPECTED_BUILD_RUN_ID",
        "EXPECTED_BUILD_IMAGE",
        "CI_RUN_ID",
        "CI_RUN_ATTEMPT",
        "RUNTIME_IMAGE",
    )
    require(all(env.get(k) for k in required), "Missing device execution inputs")
    require(
        env["EXPECTED_BUILD_RUN_ID"] == env["CI_RUN_ID"],
        "Device workflow must not reuse a previous build",
    )
    selection_file = Path(env["SELECTION_FILE"]).resolve()
    config_file = selection_file.with_name("ppu_ops_tests.json")
    environment_file = selection_file.with_name("ppu_ops_environment.json")
    raw, config_raw, environment_raw = (
        selection_file.read_bytes(),
        config_file.read_bytes(),
        environment_file.read_bytes(),
    )
    original, config, environment = (
        json.loads(raw),
        json.loads(config_raw),
        json.loads(environment_raw),
    )
    require(
        config_raw == (ROOT / "scripts/ci/ppu_ops_tests.json").read_bytes(),
        "Artifact configuration differs from the checkout",
    )
    require(
        environment_raw == (ROOT / "scripts/ci/ppu_ops_environment.json").read_bytes(),
        "Artifact environment differs from the checkout",
    )
    select.validate_config(config, ROOT)
    select.validate_selection(
        original,
        config_raw,
        environment_raw,
        tested_sha=env["EXPECTED_SAIL_COMMIT"],
        build_run_id=env["EXPECTED_BUILD_RUN_ID"],
        build_image=env["EXPECTED_BUILD_IMAGE"],
        runtime_image=env["RUNTIME_IMAGE"],
        expected_vllm_commit=env["EXPECTED_VLLM_COMMIT"],
        repo=ROOT,
    )
    if "DISPATCH_MATRIX" in env:
        require(
            json.loads(env["DISPATCH_MATRIX"]) == original["matrix"],
            "Dispatch matrix does not match the selection",
        )
    require(
        env["DEVICE_GROUP_ID"] == env["DEVICE_BOARD"] + "-0",
        "Board or group identity mismatch",
    )
    group = next(
        (g for g in original["groups"] if g["group_id"] == env["DEVICE_GROUP_ID"]), None
    )
    require(
        group is not None and bool(group["files"]), "Unknown or empty execution group"
    )
    expected_path = result_path(
        env["CI_RUN_ID"], env["CI_RUN_ATTEMPT"], env["DEVICE_GROUP_ID"], pod=True
    )
    require(
        env["DEVICE_RESULTS_DIR"] == str(expected_path),
        "Device results directory mismatch",
    )
    select.validate_fingerprint(original["build_fingerprint"])
    manifest = select.load_sibling("ppu_wheel_manifest").verify(
        Path(env["WHEELS_DIR"]),
        vllm_commit=env["EXPECTED_VLLM_COMMIT"],
        sail_commit=env["EXPECTED_SAIL_COMMIT"],
    )
    actual = select.build_fingerprint(
        select.read_diagnostics(env["BUILD_MANIFEST_FILE"]),
        manifest,
        tested_sha=env["EXPECTED_SAIL_COMMIT"],
        expected_vllm_commit=env["EXPECTED_VLLM_COMMIT"],
        build_image=env["EXPECTED_BUILD_IMAGE"],
    )
    require(
        actual == original["build_fingerprint"],
        "Build fingerprint mismatch before execution",
    )
    return raw, config_file, config, environment, manifest


PREFLIGHT = """
import importlib, importlib.metadata, json, os, shutil, subprocess
from tests.support.ppu_ci import inspect_runtime
runtime = inspect_runtime(os.environ["PPU_CI_ISOLATED_ROOT"], os.environ["PPU_CI_CHECKOUT"], json.loads(os.environ["PPU_CI_REQUIRES"]))
packages = {}
for name in ("torch", "triton", "pla"):
    module = importlib.import_module(name)
    packages[name] = str(getattr(module, "__version__", None) or importlib.metadata.version(name))
def command(args):
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip() or result.stderr.strip()
smi = shutil.which("ppu-smi") or shutil.which("nvidia-smi")
if not smi:
    raise RuntimeError("Missing SMI driver probe")
observed = dict(runtime=runtime, packages=packages, sdk_output=command(["hgcc", "--version"]), smi_output=command([smi]))
with open(os.environ["PPU_CI_OBSERVATION"], "w") as stream:
    json.dump(observed, stream, indent=2)
"""


def execute(env):
    # Verify the shared mount and CPU probe before creating any local results tree.
    output = result_path(
        env["CI_RUN_ID"], env["CI_RUN_ATTEMPT"], env["DEVICE_GROUP_ID"], pod=True
    )
    require(str(output) == env["DEVICE_RESULTS_DIR"], "Invalid results path")
    require_nas(Path("/mnt/wl_nas"))
    probe = json.loads((output / "cpu-probe.json").read_bytes())
    require(
        probe["run_id"] == env["CI_RUN_ID"]
        and probe["run_attempt"] == env["CI_RUN_ATTEMPT"]
        and probe["group_id"] == env["DEVICE_GROUP_ID"],
        "NAS probe identity mismatch",
    )
    select.write_json(output / "pod-probe.json", probe)
    identity = {
        "run_id": env["CI_RUN_ID"],
        "run_attempt": env["CI_RUN_ATTEMPT"],
        "group_id": env["DEVICE_GROUP_ID"],
        "board": env["DEVICE_BOARD"],
        "tested_sha": env["EXPECTED_SAIL_COMMIT"],
    }
    summary = {
        "schema": 1,
        **identity,
        "status": "failure",
        "failure_stage": "inputs",
        "reason": "Not completed",
    }
    timings = {
        "schema": 1,
        **identity,
        "nproc_per_node": 1,
        "files": [],
        "phases": {
            k: None
            for k in (
                "queue_seconds",
                "install_seconds",
                "preflight_seconds",
                "device_occupied_seconds",
                "total_seconds",
            )
        },
        "timing_error": "Scheduling events are unavailable; device monotonic clocks cannot be compared across hosts",
    }
    started = time.monotonic()
    stage = "inputs"
    try:
        raw, config_path, config, baseline, manifest = read_inputs(env)
        original = json.loads(raw)
        for key in ("config_sha256", "build_fingerprint", "runtime_image"):
            timings[key] = original[key]
        (output / "selection.json").write_bytes(raw)
        shutil.copyfile(env["BUILD_MANIFEST_FILE"], output / "build-manifest.txt")
        select.write_json(output / "wheel-manifest.json", manifest)
        with tempfile.TemporaryDirectory(prefix="ppu-ops-", dir="/tmp") as temporary:
            isolated = (Path(temporary) / "isolated").resolve()
            copy_test_tree(ROOT, isolated)
            # pytest's node guard needs the same discovered catalog as selection.
            config_path = isolated / "ppu_ops_tests.json"
            select.write_json(config_path, config)
            child_env = clean_environment(env, ROOT, isolated)
            stage = "install"
            phase = time.monotonic()
            wheels = [
                str(Path(env["WHEELS_DIR"]) / record["file"])
                for record in manifest["wheels"].values()
            ]
            child_env["PIP_CONSTRAINT"] = str(output / "image-constraints.txt")
            stack_script = str(ROOT / "scripts/ci/ppu_image_stack.py")
            stack_args = [
                "--sdk",
                "/usr/local/PPU_SDK",
                "--output",
                str(output / "image-stack.json"),
            ]
            with (output / "install.log").open("w") as log:
                for command in (
                    [
                        sys.executable,
                        stack_script,
                        "capture",
                        *stack_args,
                        "--constraints",
                        child_env["PIP_CONSTRAINT"],
                    ],
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--force-reinstall",
                        *wheels,
                    ],
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        # Set the index only for test tools; install this build's wheels with --no-deps.
                        "--index-url",
                        env.get("SAIL_PIP_INDEX_URL")
                        or "https://mirrors.aliyun.com/pypi/simple/",
                        "-r",
                        str(ROOT / "requirements/dev.txt"),
                    ],
                    [sys.executable, stack_script, "verify", *stack_args],
                ):
                    subprocess.run(
                        command,
                        check=True,
                        cwd=isolated,
                        env=child_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            timings["phases"]["install_seconds"] = time.monotonic() - phase
            stage = "preflight"
            phase = time.monotonic()
            # Preflight all dependencies for this board to allow safe expansion after environment changes.
            entries = [
                t
                for t in config["test_catalog"]
                if identity["board"] in select.runnable_boards(config, t)
            ]
            child_env["PPU_CI_REQUIRES"] = json.dumps(
                sorted({r for t in entries for r in t["requires"]})
            )
            child_env["PPU_CI_OBSERVATION"] = str(output / "runtime-observation.json")
            with (output / "preflight.log").open("w") as log:
                subprocess.run(
                    [sys.executable, "-c", PREFLIGHT],
                    cwd=isolated,
                    env=child_env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            observed = json.loads((output / "runtime-observation.json").read_bytes())
            profile = runtime_profile(**observed)
            expanded = profile != baseline["profiles"].get(identity["board"])
            environment = {
                "schema": 1,
                **identity,
                "profile": profile,
                "expanded": expanded,
                "expansion_reason": "runtime_environment_changed" if expanded else None,
                "runtime_image": original["runtime_image"],
            }
            select.write_json(output / "environment.json", environment)
            effective = effective_selection(
                config, original, raw, identity["board"], expanded
            )
            select.write_json(output / "effective-selection.json", effective)
            timings["phases"]["preflight_seconds"] = time.monotonic() - phase
            stage = "pytest"
            group = next(
                g for g in effective["groups"] if g["group_id"] == identity["group_id"]
            )
            catalog = {t["id"]: t for t in config["test_catalog"]}
            timings["files"] = run_files(
                [catalog[t] for t in group["test_ids"]],
                identity["board"],
                config_path,
                isolated,
                output,
                child_env,
            )
        stage = "evidence"
        select.write_json(output / "timings.json", timings)
        select.write_json(output / "summary.json", summary)
        validate_group(
            config,
            raw,
            output,
            env["CI_RUN_ID"],
            env["CI_RUN_ATTEMPT"],
            require_success=False,
        )
        summary.update(status="success", failure_stage=None, reason=None)
    except Exception as error:
        summary.update(failure_stage=stage, reason=f"{type(error).__name__}: {error}")
    finally:
        timings["device_script_seconds"] = time.monotonic() - started
        select.write_json(output / "timings.json", timings)
        select.write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "success" else 1


def scheduler_names(groups, owner, run_id, attempt):
    # Match scripts/submit.sh in pinned action a4e03cb: 20-character owner/suffix, 52-character full name.
    owner = re.sub(r"[^a-z0-9-]", "-", owner.lower()).strip("-")[:20].rstrip("-")
    require(bool(owner), "Job name is missing the repository owner")
    names = []
    for group in groups:
        suffix = "ops-" + group["group_id"]
        require(
            len(suffix) <= 20 and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", suffix),
            "Invalid job name suffix",
        )
        name = f"ppu-{owner}-{run_id}-{attempt}-{suffix}"[:52].rstrip("-")
        require(
            name.endswith("-" + suffix),
            "Truncating the job name would remove the execution group suffix",
        )
        names.append(name)
    require(len(names) == len(set(names)), "Job names are not unique")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "prepare", "finish", "summary"))
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    env = os.environ
    if args.command == "run":
        raise SystemExit(execute(env))
    raw, _, config, _, _ = read_inputs(env)
    if args.command == "prepare":
        scheduler_names(
            json.loads(raw)["groups"],
            env.get("GITHUB_REPOSITORY_OWNER", ""),
            env["CI_RUN_ID"],
            env["CI_RUN_ATTEMPT"],
        )
        output = result_path(
            env["CI_RUN_ID"], env["CI_RUN_ATTEMPT"], env["DEVICE_GROUP_ID"], pod=False
        )
        require_nas(Path("/wl_nas"))
        output.mkdir(parents=True, exist_ok=False)
        probe = {
            "run_id": env["CI_RUN_ID"],
            "run_attempt": env["CI_RUN_ATTEMPT"],
            "group_id": env["DEVICE_GROUP_ID"],
            "nonce": uuid.uuid4().hex,
        }
        select.write_json(output / "cpu-probe.json", probe)
        select.write_json(output / "orchestration.json", {"started": time.monotonic()})
        profile = config["execution_profiles"][env["DEVICE_BOARD"]]
        select.github_outputs(
            {"node_selector": "board-type=" + profile["node_selector"]["board-type"]}
        )
    elif args.command == "finish":
        output = result_path(
            env["CI_RUN_ID"], env["CI_RUN_ATTEMPT"], env["DEVICE_GROUP_ID"], pod=False
        )
        require(
            (output / "cpu-probe.json").read_bytes()
            == (output / "pod-probe.json").read_bytes(),
            "CPU and Pod NAS probes do not match",
        )
        timings = json.loads((output / "timings.json").read_bytes())
        started = json.loads((output / "orchestration.json").read_bytes())["started"]
        timings["phases"]["total_seconds"] = time.monotonic() - started
        timings["timing_error"] = (
            "Scheduling action did not provide device start/end events; queue_seconds/device_occupied_seconds are null"
        )
        select.write_json(output / "timings.json", timings)
        validate_group(config, raw, output, env["CI_RUN_ID"], env["CI_RUN_ATTEMPT"])
    else:
        require(args.results is not None, "Missing artifact results directory")
        directories = [p for p in args.results.iterdir() if p.is_dir()]
        validate_results(
            config, raw, directories, env["CI_RUN_ID"], env["CI_RUN_ATTEMPT"]
        )


if __name__ == "__main__":
    main()
