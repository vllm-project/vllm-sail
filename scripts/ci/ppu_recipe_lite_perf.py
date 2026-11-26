# SPDX-License-Identifier: Apache-2.0
"""Run and report recipe-backed lightweight performance regression tests."""

import argparse
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests/e2e/perf"))
ITERS = 20
SAMPLES = 5
BENCHMARK_TIMEOUT_S = 45 * 60
VLLM_CLI_MODULE = "vllm.entrypoints.cli.main"
REGRESSION_LIMIT = Decimal("1.05")
MAX_CV = 0.15
CASE_NAME = re.compile(r"[a-z0-9_]{1,64}")
BASELINE_CONFIG_FIELDS = (
    "benchmark_args",
    "checkpoint_files_sha256",
    "devices",
    "groups",
    "recipe_config_digest",
    "sdk_sha256",
    "torch_sha256",
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def snapshot_provenance(checkpoint):
    cache_ref = checkpoint.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", cache_ref):
        raise ValueError("Invalid offline checkpoint snapshot reference")
    provenance = {"checkpoint_source": "hf_snapshot", "checkpoint_revision": None}
    if re.fullmatch(r"[0-9a-f]{40}", cache_ref):
        provenance["checkpoint_revision"] = cache_ref
    else:
        provenance["checkpoint_cache_ref"] = cache_ref
    return provenance


def commands(resolved, groups, checkpoint, results_dir):
    from ppu_recipe import _parse_flag_entries, _render_entries

    ignored = {"--enable-auto-tool-choice", "--tool-call-parser", "--reasoning-parser"}
    result = {}
    for name, group in groups.items():
        limits = {
            "--" + key.replace("_", "-"): str(value)
            for key, value in group.items()
            if key != "cases"
        }
        flags = _render_entries(
            [
                entry
                for entry in _parse_flag_entries(resolved["flags"])
                if entry[0] not in ignored and entry[0] not in limits
            ]
        )
        result[name] = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            # c10d also advertises a bootstrap address to the local ranks.
            "--local-addr=127.0.0.1",
            # Give the elastic agent time to terminate its worker groups.
            "--shutdown-timeout=20",
            f"--nproc-per-node={resolved['ppu_count']}",
            "-m",
            VLLM_CLI_MODULE,
            "bench",
            "forward",
            "--model",
            str(checkpoint),
            *flags,
            *(f"{key}={value}" for key, value in limits.items()),
            "--seed",
            "0",
            "--kv-len-jitter",
            "0",
            "--routing",
            "zipf",
            "--exclude-sampler",
            "--iters",
            str(ITERS),
            "--samples",
            str(SAMPLES),
        ]
        for case in group["cases"]:
            result[name].extend(("--case", case))
        result[name].extend(("--output-json", str(results_dir / f"{name}.json")))
    return result


def benchmark_args(argv, model_id):
    args = argv[argv.index(VLLM_CLI_MODULE) + 1 : -2]
    args[args.index("--model") + 1] = model_id
    # Earlier vLLM builds excluded sampling by default when recording the baseline.
    args.remove("--exclude-sampler")
    return args


def run_benchmark(argv, env, log):
    from ppu_recipe_runner import _stop_server as stop_process_group

    from scripts.eval.ppu_model_sanity import defer_termination_signals

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        process = None
        try:
            with defer_termination_signals() as launch_signals:
                process = subprocess.Popen(
                    argv,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            if launch_signals:
                raise SystemExit(128 + launch_signals[-1])
            returncode = process.wait(timeout=BENCHMARK_TIMEOUT_S)
        finally:
            if process is not None and process.returncode != 0:
                with defer_termination_signals():
                    stop_process_group(process)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, argv)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def read_measurements(path, expected, rank_count):
    from _baseline_store import compute_cv

    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) != len(expected):
        raise ValueError("Expected one benchmark result per case")
    metrics = {}
    for row in data:
        name = row["case"]
        if name in metrics or name not in expected or row["requests"] != expected[name]:
            raise ValueError("Unexpected benchmark case or request shape")
        ranks = row["ranks"]
        median = row["median_us"]
        if not positive(median) or len(ranks) != rank_count:
            raise ValueError("Invalid benchmark median or rank count")
        for rank in ranks:
            latency = rank["latency_us"]
            if (
                rank["finite"] is not True
                or rank["routing"] != "zipf"
                or latency["timer"] != "graph_events_rank_max"
                or latency["median"] != median
                or latency["count"] != ITERS * SAMPLES
                or len(latency["samples_us"]) != ITERS * SAMPLES
                or not all(positive(sample) for sample in latency["samples_us"])
            ):
                raise ValueError("Invalid benchmark rank evidence")
        metrics[name] = {
            "median_us": median,
            "cv": compute_cv(ranks[0]["latency_us"]["samples_us"]),
        }
    return metrics


def assess(current, baseline):
    if baseline["schema_version"] != 1 or baseline["config"] != current["config"]:
        return {
            "status": "fail",
            "failure_reason": "lite_perf_config_mismatch",
            "cases": [],
            "config_mismatch": [
                field
                for field in BASELINE_CONFIG_FIELDS
                if baseline["config"].get(field) != current["config"].get(field)
            ],
        }
    current_metrics, baseline_metrics = current["metrics"], baseline["metrics"]
    if current_metrics.keys() != baseline_metrics.keys():
        raise ValueError("Baseline must contain exactly the measured cases")
    rows = []
    for name, metric in current_metrics.items():
        reference = baseline_metrics[name]
        baseline_us, cv = reference["median_us"], reference["cv"]
        median_us = metric["median_us"]
        if (
            not positive(baseline_us)
            or type(cv) not in (int, float)
            or not 0 <= cv <= MAX_CV
        ):
            raise ValueError("Invalid lite perf baseline metric")
        if Decimal(str(median_us)) > Decimal(str(baseline_us)) * REGRESSION_LIMIT:
            status = "FAIL"
        elif metric["cv"] > MAX_CV:
            status = "UNSTABLE"
        else:
            status = "PASS"
        rows.append(
            {
                "case": name,
                "status": status,
                "median_us": median_us,
                "baseline_us": baseline_us,
                "delta": (median_us - baseline_us) / baseline_us,
            }
        )
    failed = {row["status"] for row in rows}
    reason = (
        "lite_perf_regression"
        if "FAIL" in failed
        else "lite_perf_measurement_unstable"
        if "UNSTABLE" in failed
        else "not_applicable"
    )
    return {
        "status": "pass" if reason == "not_applicable" else "fail",
        "failure_reason": reason,
        "cases": rows,
    }


def run(args):
    from dataclasses import asdict

    import torch
    from huggingface_hub import snapshot_download
    from ppu_recipe import _load_yaml, redact_and_verify_public_payload, resolve_recipe
    from vllm.benchmarks.forward.shapes import parse_case
    from vllm.platforms import current_platform

    output = args.results_dir
    output.mkdir(parents=True, exist_ok=True)
    policy = _load_yaml(args.policy)
    resolved = resolve_recipe(args.recipe, policy_path=args.policy, tier="lite_perf")
    groups = policy["tiers"]["lite_perf"]["groups"]
    count = resolved["ppu_count"]
    devices = [
        torch.cuda.get_device_name(rank) for rank in range(torch.cuda.device_count())
    ]
    if not current_platform.is_ppu() or devices != ["ZW-M890P"] * count or count != 8:
        raise ValueError("Lightweight performance regression requires eight M890 PPUs")
    if args.local_checkpoint is not None:
        if not args.capture_baseline or not args.local_checkpoint.is_absolute():
            raise ValueError("Local checkpoint paths require explicit baseline capture")
        checkpoint = args.local_checkpoint
        checkpoint_info = {
            "checkpoint_source": "local_export",
            "checkpoint_revision": None,
        }
    else:
        checkpoint = Path(
            snapshot_download(resolved["resolved_model"], local_files_only=True)
        )
        checkpoint_info = snapshot_provenance(checkpoint)
    # The mounted export and the CI cache can use different directory layouts.
    checkpoint_files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    checkpoint_hashes = {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in checkpoint_files
    }
    provenance = json.loads((output.parent / "provenance.json").read_text())
    stack = json.loads(args.image_stack.read_text())
    cmd = commands(resolved, groups, checkpoint, output)
    current = {
        "schema_version": 1,
        "config": {
            "recipe_config_digest": resolved["config_digest"],
            "checkpoint_files_sha256": checkpoint_hashes,
            "devices": devices,
            "sdk_sha256": stack["sdk_sha256"],
            "torch_sha256": stack["torch_sha256"],
            "groups": groups,
            "benchmark_args": {
                name: benchmark_args(argv, resolved["hf_model_id"])
                for name, argv in cmd.items()
            },
        },
        "provenance": {
            "commits": provenance["commits"],
            **checkpoint_info,
            "runtime_image_digest": provenance.get("runtime_image_digest"),
            "runtime_image_id": provenance.get("runtime_image_id"),
        },
        "metrics": {},
    }
    baseline_name = f"{args.policy.stem}-lite-perf.json"
    baseline = None
    if not args.capture_baseline:
        path = ROOT / "tests/e2e/perf/baselines/890p" / baseline_name
        baseline = json.loads(path.read_text())
        if baseline["schema_version"] != 1 or baseline["config"] != current["config"]:
            redact_and_verify_public_payload(current)
            write_json(output / "measurement.json", current)
            write_json(output / "summary.json", assess(current, baseline))
            return 1

    write_json(output / "commands.json", {"argv": cmd, "env": resolved["server_env"]})
    env = {**os.environ, **resolved["server_env"]}
    for name, argv in cmd.items():
        expected = {}
        for raw in groups[name]["cases"]:
            case = parse_case(raw)
            expected[case.name] = [asdict(req) for req in case.requests]
        try:
            with (output / f"{name}.log").open("w") as log:
                run_benchmark(argv, env, log)
        except subprocess.TimeoutExpired:
            print(
                f"Lite perf benchmark exceeded {BENCHMARK_TIMEOUT_S} seconds",
                file=sys.stderr,
            )
            write_json(
                output / "summary.json",
                {
                    "status": "fail",
                    "failure_reason": "lite_perf_measurement_timeout",
                    "cases": [],
                },
            )
            return 1
        current["metrics"].update(
            read_measurements(output / f"{name}.json", expected, count)
        )
    redact_and_verify_public_payload(current)
    write_json(output / "measurement.json", current)

    if baseline is not None:
        summary = assess(current, baseline)
    else:
        if any(metric["cv"] > MAX_CV for metric in current["metrics"].values()):
            raise ValueError("Cannot capture unstable lite perf measurements")
        write_json(output / baseline_name, current)
        summary = {"status": "pass", "failure_reason": "not_applicable", "cases": []}
    write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "pass" else 1


def publish():
    # The runner host can have an older system Python than the runtime image.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ppu_recipe_diagnostics import failure_reason

    artifacts = Path(os.environ["PRIVATE_DIR"])
    private = artifacts / "results"
    public = Path(os.environ["PUBLIC_DIR"])
    status = "fail"
    reason = failure_reason(artifacts, "lite_perf")
    rows = []
    config_mismatch = []
    try:
        path = private / "summary.json"
        if not stat.S_ISDIR(private.lstat().st_mode):
            raise ValueError("Expected regular result directory")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024:
            raise ValueError("Expected bounded regular summary")
        report = json.loads(path.read_text())
        if report["status"] == "fail" and report["failure_reason"] in {
            "lite_perf_config_mismatch",
            "lite_perf_regression",
            "lite_perf_measurement_timeout",
            "lite_perf_measurement_unstable",
        }:
            reason = report["failure_reason"]
        if reason == "lite_perf_config_mismatch":
            config_mismatch = [
                field
                for field in BASELINE_CONFIG_FIELDS
                if field in report.get("config_mismatch", [])
            ]
        cases = report["cases"]
        if not isinstance(cases, list):
            raise ValueError("Expected a list of lite perf results")
        for row in cases:
            if (
                CASE_NAME.fullmatch(row["case"])
                and row["status"] in {"PASS", "FAIL", "UNSTABLE"}
                and positive(row["median_us"])
                and positive(row["baseline_us"])
                and type(row["delta"]) in (int, float)
                and math.isfinite(row["delta"])
            ):
                rows.append(
                    {
                        key: row[key]
                        for key in (
                            "case",
                            "status",
                            "median_us",
                            "baseline_us",
                            "delta",
                        )
                    }
                )
        if (
            os.environ["RECIPE_OUTCOME"] == "success"
            and report["status"] == "pass"
            and report["failure_reason"] == "not_applicable"
            and rows
            and len(cases) == len(rows)
            and all(row["status"] == "PASS" for row in rows)
        ):
            status = "pass"
            reason = "not_applicable"
    except (OSError, KeyError, ValueError, TypeError, OverflowError, RecursionError):
        pass
    if os.environ["RETENTION_OUTCOME"] == "failure":
        status = "fail"
        reason = "private_retention_failed"
    summary = {"status": status, "failure_reason": reason, "cases": rows}
    if config_mismatch:
        summary["config_mismatch"] = config_mismatch
    write_json(public / "summary.json", summary)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
        stream.write(
            f"### Lightweight Performance Regression\n\nResult: **{status}** ({reason})\n\n"
            "A case fails when its median latency is more than 5% above baseline.\n\n"
        )
        if config_mismatch:
            stream.write(
                f"Baseline configuration differs in {', '.join(config_mismatch)}.\n\n"
            )
        if rows:
            stream.write(
                "| Case | Current (µs) | Baseline (µs) | Change | Result |\n"
                "| --- | ---: | ---: | ---: | --- |\n"
            )
            for row in rows:
                stream.write(
                    f"| {row['case']} | {row['median_us']:.2f} | "
                    f"{row['baseline_us']:.2f} | {row['delta']:+.2%} | {row['status']} |\n"
                )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write(f"status={status}\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action")
    run_parser = sub.add_parser("run")
    for name in ("recipe", "policy", "results-dir", "image-stack"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--capture-baseline", action="store_true")
    run_parser.add_argument("--local-checkpoint", type=Path)
    sub.add_parser("publish")
    args = parser.parse_args()
    if args.action is None:
        parser.error("the following arguments are required: action")
    return run(args) if args.action == "run" else publish()


if __name__ == "__main__":
    raise SystemExit(main())
