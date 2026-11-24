# SPDX-License-Identifier: Apache-2.0
"""Run and report recipe-backed offline forward performance measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests/e2e/perf"))
ITERS = 20
SAMPLES = 5
REGRESSION_LIMIT = Decimal("1.05")
MAX_CV = 0.15
CASE_NAME = re.compile(r"[a-z0-9_]{1,64}")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


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
            f"--nproc-per-node={resolved['ppu_count']}",
            "-m",
            "vllm.entrypoints.cli.main",
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
            "--iters",
            str(ITERS),
            "--samples",
            str(SAMPLES),
        ]
        for case in group["cases"]:
            result[name].extend(("--case", case))
        result[name].extend(("--output-json", str(results_dir / f"{name}.json")))
    return result


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def read_measurements(path, expected, rank_count):
    from _baseline_store import compute_cv

    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) != len(expected):
        raise ValueError("Expected one forward result per case")
    metrics = {}
    for row in data:
        name = row["case"]
        if name in metrics or name not in expected or row["requests"] != expected[name]:
            raise ValueError("Unexpected forward case or request shape")
        ranks = row["ranks"]
        median = row["median_us"]
        if not positive(median) or len(ranks) != rank_count:
            raise ValueError("Invalid forward median or rank count")
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
                raise ValueError("Invalid forward rank evidence")
        metrics[name] = {
            "median_us": median,
            "cv": compute_cv(ranks[0]["latency_us"]["samples_us"]),
        }
    return metrics


def assess(current, baseline):
    if baseline["schema_version"] != 1 or baseline["config"] != current["config"]:
        return {
            "status": "fail",
            "failure_reason": "forward_config_mismatch",
            "cases": [],
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
            raise ValueError("Invalid forward baseline metric")
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
        "forward_regression"
        if "FAIL" in failed
        else "forward_measurement_unstable"
        if "UNSTABLE" in failed
        else "not_applicable"
    )
    return {
        "status": "pass" if reason == "not_applicable" else "fail",
        "failure_reason": reason,
        "cases": rows,
    }


def run(args):
    import torch
    from huggingface_hub import snapshot_download
    from ppu_recipe import _load_yaml, redact_and_verify_public_payload, resolve_recipe
    from vllm.benchmarks.forward.shapes import parse_case
    from vllm.platforms import current_platform

    output = args.results_dir
    output.mkdir(parents=True, exist_ok=True)
    policy = _load_yaml(args.policy)
    resolved = resolve_recipe(args.recipe, policy_path=args.policy, tier="forward_perf")
    groups = policy["tiers"]["forward_perf"]["groups"]
    count = resolved["ppu_count"]
    devices = [
        torch.cuda.get_device_name(rank) for rank in range(torch.cuda.device_count())
    ]
    if not current_platform.is_ppu() or devices != ["ZW-M890P"] * count or count != 8:
        raise ValueError("Forward performance requires eight M890 PPUs")
    revision = None
    if args.local_checkpoint is not None:
        if not args.capture_baseline or not args.local_checkpoint.is_absolute():
            raise ValueError("Local checkpoint paths require explicit baseline capture")
        checkpoint = args.local_checkpoint
    else:
        checkpoint = Path(
            snapshot_download(resolved["resolved_model"], local_files_only=True)
        )
        revision = checkpoint.name
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Expected the offline cache's checkpoint revision")
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
                name: [*argv[7:10], resolved["hf_model_id"], *argv[11:-2]]
                for name, argv in cmd.items()
            },
        },
        "provenance": {
            "commits": provenance["commits"],
            "checkpoint_source": "local_export" if revision is None else "hf_snapshot",
            "checkpoint_revision": revision,
            "runtime_image_digest": provenance.get("runtime_image_digest"),
            "runtime_image_id": provenance.get("runtime_image_id"),
        },
        "metrics": {},
    }
    baseline = None
    if not args.capture_baseline:
        path = (
            ROOT / "tests/e2e/perf/baselines/890p" / f"{args.policy.stem}-forward.json"
        )
        baseline = json.loads(path.read_text())
        if baseline["schema_version"] != 1 or baseline["config"] != current["config"]:
            write_json(output / "summary.json", assess(current, baseline))
            return 1

    write_json(output / "commands.json", {"argv": cmd, "env": resolved["server_env"]})
    env = {**os.environ, **resolved["server_env"]}
    for name, argv in cmd.items():
        expected = {}
        for raw in groups[name]["cases"]:
            case = parse_case(raw)
            expected[case.name] = [asdict(req) for req in case.requests]
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        current["metrics"].update(
            read_measurements(output / f"{name}.json", expected, count)
        )
    redact_and_verify_public_payload(current)
    write_json(output / "measurement.json", current)

    if baseline is not None:
        summary = assess(current, baseline)
    else:
        if any(metric["cv"] > MAX_CV for metric in current["metrics"].values()):
            raise ValueError("Cannot capture unstable forward measurements")
        write_json(output / f"{args.policy.stem}-forward.json", current)
        summary = {"status": "pass", "failure_reason": "not_applicable", "cases": []}
    write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "pass" else 1


def publish():
    private = Path(os.environ["PRIVATE_DIR"]) / "results"
    public = Path(os.environ["PUBLIC_DIR"])
    status = "fail"
    reason = "forward_measurement_failed"
    rows = []
    try:
        path = private / "summary.json"
        if not stat.S_ISDIR(private.lstat().st_mode):
            raise ValueError("Expected regular result directory")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024:
            raise ValueError("Expected bounded regular summary")
        report = json.loads(path.read_text())
        if report["failure_reason"] in {
            "not_applicable",
            "forward_config_mismatch",
            "forward_regression",
            "forward_measurement_unstable",
        }:
            reason = report["failure_reason"]
        for row in report["cases"]:
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
            and rows
        ):
            status = "pass"
    except (OSError, KeyError, ValueError, TypeError, RecursionError):
        pass
    if os.environ["RETENTION_OUTCOME"] == "failure":
        status = "fail"
        reason = "private_retention_failed"
    write_json(
        public / "summary.json",
        {"status": status, "failure_reason": reason, "cases": rows},
    )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write(
            f"### Forward performance\n\nResult: **{status}** ({reason})\n\n"
            "A case fails when its median latency is more than 5% above baseline.\n\n"
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
    return 0 if status == "pass" or os.environ["RECIPE_OUTCOME"] != "success" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    for name in ("recipe", "policy", "results-dir", "image-stack"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--capture-baseline", action="store_true")
    run_parser.add_argument("--local-checkpoint", type=Path)
    sub.add_parser("publish")
    args = parser.parse_args()
    return run(args) if args.action == "run" else publish()


if __name__ == "__main__":
    raise SystemExit(main())
