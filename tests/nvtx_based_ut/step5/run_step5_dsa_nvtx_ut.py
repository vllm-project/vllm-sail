from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from importlib import import_module, metadata
from importlib.machinery import EXTENSION_SUFFIXES
from itertools import count
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
bench = import_module("benchmark_step5_sparse_nvtx")

SUPPORTED_BACKENDS = {
    "score": bench.default_backends({"kind": "direct"})
    + bench.default_backends({"kind": "grouped"}),
    "topk": bench.default_backends({"kind": "topk"}),
    "attention": list(
        dict.fromkeys(
            bench.default_backends({"kind": "attention", "pd_type": "decode"})
            + bench.default_backends({"kind": "attention", "pd_type": "prefill"})
        )
    ),
}
BACKEND_HELP = (
    "Supported backends (subject to per-case contract admission):\n"
    + "\n".join(
        f"  {op}: {', '.join(backends)}" for op, backends in SUPPORTED_BACKENDS.items()
    )
)


class CaseArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        super().error(f"{message}\n{BACKEND_HELP}")


def check_vllm_version(dry_only):
    try:
        installed = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        if dry_only:
            return {
                "vllm_version": None,
                "version_check": "skipped",
                "reason": "vLLM is not installed; dry-only does not execute device ops",
            }
        parser.error("vLLM is not installed; device UT requires vLLM 0.33.x")
    try:
        release = Version(installed).release
    except InvalidVersion:
        parser.error(f"Invalid vLLM version: {installed!r}; UT requires vLLM 0.33.x")
    if release[:2] != (0, 33):
        parser.error(f"Unsupported vLLM version: {installed}; UT requires vLLM 0.33.x")
    return {"vllm_version": installed, "version_check": "passed"}


def installed_extensions(required):
    for distribution in metadata.distributions(name="vllm-sail"):
        for file in distribution.files or ():
            if (
                len(file.parts) == 2
                and file.parts[0] == "vllm_sail"
                and file.name.endswith(tuple(EXTENSION_SUFFIXES))
            ):
                extension = Path(distribution.locate_file(file)).resolve()
                if extension.is_file():
                    return extension.parent
    if required:
        parser.error(
            f"No installed vllm-sail native extensions found for {sys.executable}"
        )
    return None


def create_run_output(root, started_at):
    root.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%d-%H%M%S-%f")
    for attempt in count():
        folder = root / (timestamp if attempt == 0 else f"{timestamp}-{attempt}")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder


def print_case_timing(entry, backends):
    def percent(value):
        return f"{value:.3f}" if value is not None else "N/A"

    memory = next((result.get("memory") or {} for result in entry["results"]), {})
    arms = {
        arm["backend"]: arm for result in entry["results"] for arm in result["arms"]
    }
    for backend in backends:
        arm = arms.get(backend, {})
        median_us = (arm.get("timing") or {}).get("median_us")
        latency = f"{median_us:.3f}" if median_us is not None else "N/A"
        status = arm.get("status") or entry["status"]
        utilization = arm.get("utilization") or {}
        print(
            f"case_id={entry['case_id']} op={entry['op']} scope={entry['scope']} "
            f"backend={backend} median_us={latency} status={status} "
            f"mfu_pct={percent(utilization.get('mfu_pct'))} "
            f"logical_bandwidth_utilization_pct={percent(utilization.get('logical_bandwidth_utilization_pct'))} "
            f"case_peak_allocated_capacity_pct={percent(memory.get('peak_allocated_capacity_pct'))} "
            f"device_memory_used_pct={percent(memory.get('device_memory_used_pct'))}",
            flush=True,
        )


parser = CaseArgumentParser(
    epilog=BACKEND_HELP, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("output"),
    help=(
        "Output root (default: ./output); each run uses a timestamp subdirectory; "
        "relative paths use the current directory"
    ),
)
parser.add_argument(
    "--nvtx",
    "--caselist",
    nargs="+",
    type=Path,
    required=True,
    help="Input NVTX TXT/caselist files; non-selected ops are filtered out",
)
parser.add_argument(
    "--workload-config",
    type=Path,
    help="Synthetic content overrides; default uses geometry-derived capacity-causal data",
)
parser.add_argument(
    "--topk-tensor-dir",
    type=Path,
    help="Directory containing topk_tensor files (default: source TXT directory); missing files warn and use synthetic inputs",
)
parser.add_argument("--op", choices=tuple(SUPPORTED_BACKENDS), required=True)
parser.add_argument(
    "--backends", nargs="+", required=True, help="Explicit backends for the selected op"
)
parser.add_argument(
    "--scope",
    choices=("all", "prepared-op", "kernel"),
    help="Default: prepared-op for topk, all for score/attention",
)
parser.add_argument("--pilot", action="store_true")
parser.add_argument("--dry-only", action="store_true")
bench.add_utilization_arguments(parser)
args = parser.parse_args()
try:
    peaks = bench.utilization_peaks(args)
except ValueError as exc:
    parser.error(str(exc))
started_at = datetime.now().astimezone()
unsupported = set(args.backends) - set(SUPPORTED_BACKENDS[args.op])
if unsupported:
    parser.error(
        f"Unsupported backends for {args.op}: {', '.join(sorted(unsupported))}"
    )
if len(args.backends) != len(set(args.backends)):
    parser.error("Duplicate backend selection")
output_root = args.output.resolve()
args.nvtx = [path.resolve() for path in args.nvtx]
for path in args.nvtx:
    if not path.is_file():
        parser.error(f"Input file does not exist: {path}")
try:
    config = (
        json.loads(args.workload_config.read_text())
        if args.workload_config
        else {"default": {"policy": "capacity-causal"}, "cases": {}}
    )
    if (
        not isinstance(config, dict)
        or set(config) - {"default", "cases"}
        or not isinstance(config.get("default", {}), dict)
        or not isinstance(config.get("cases", {}), dict)
    ):
        parser.error("Workload config must contain default/cases mappings")
    cases = bench.extract_cases(args.nvtx)["cases"]
except (OSError, ValueError) as exc:
    parser.error(str(exc))
parsed_cases = cases
op_cases = sorted(
    (case for case in parsed_cases if case["op"] == args.op),
    key=lambda case: case["sources"][0]["line"],
)
scope = args.scope or ("prepared-op" if args.op == "topk" else "all")
label_scope = "op" if scope == "prepared-op" else "kernel"
cases = [case for case in op_cases if scope == "all" or case["scope"] == label_scope]
if args.pilot:
    cases = [case for case in cases if case["scope"] != "kernel"][:1]
if not cases:
    parser.error(f"No cases match --op {args.op} --scope {scope}")
runtime = check_vllm_version(args.dry_only)
extensions = installed_extensions(required=not args.dry_only)
runtime["extension_root"] = str(extensions) if extensions is not None else None
try:
    output = create_run_output(output_root, started_at)
except OSError as exc:
    parser.error(f"Cannot create output run directory under {output_root}: {exc}")
selected_ids = {case["id"] for case in cases}
selection = {
    "output_root": str(output_root),
    "output_dir": str(output),
    "started_at": started_at.isoformat(),
    "runtime": runtime,
    "utilization_peaks": peaks,
    "nvtx": [str(path) for path in args.nvtx],
    "parsed_cases": len(parsed_cases),
    "filtered_op_counts": dict(
        collections.Counter(
            case["op"] for case in parsed_cases if case["op"] != args.op
        )
    ),
    "op": args.op,
    "backends": args.backends,
    "scope": scope,
    "pilot": args.pilot,
    "selected_case_ids": [case["id"] for case in cases],
    "excluded_cases": [
        {
            "case_id": case["id"],
            "scope": case["scope"],
            "reason": (
                "Outside selected scope"
                if scope != "all" and case["scope"] != label_scope
                else "Outside pilot subset"
            ),
        }
        for case in op_cases
        if case["id"] not in selected_ids
    ],
}
bench.save(output / "selection.json", selection)
bench.save(output / "workload.json", config)
bench.save(output / "cases.json", cases)
print(json.dumps({"stage": "selection", **selection}), flush=True)

dry = []
for case in cases:
    settings = argparse.Namespace(
        scope="kernel" if case["scope"] == "kernel" else "prepared-op",
        max_memory_mib=102400,
        backends=args.backends,
        topk_tensor_dir=args.topk_tensor_dir,
        peak_compute_tflops=args.peak_compute_tflops,
        peak_compute_dtype=args.peak_compute_dtype,
        peak_hbm_gbps=args.peak_hbm_gbps,
    )
    try:
        entry = bench.dry_case(case, config, settings)
        blocked = [
            arm
            for arm in entry.get("backend_support", [])
            if arm["status"] == "unsupported"
        ]
        if blocked:
            entry.update(
                status="unsupported",
                reason="; ".join(
                    f"{arm['backend']}: {arm['reason']}" for arm in blocked
                ),
            )
        if entry["status"] == "ready" and entry["geometry"]["kind"] == "attention":
            if entry["workload"].get("scale", case["scalars"].get("scale")) is None:
                entry.update(
                    status="unsupported", reason="Missing explicit attention scale"
                )
    except ValueError as exc:
        entry = {
            "case_id": case["id"],
            "op": case["op"],
            "status": "unsupported",
            "reason": str(exc),
        }
    prefix = f"{case['op']}-{case['scope']}-{case['id']}"
    entry["output_dir"] = str(output)
    entry["output_prefix"] = prefix
    bench.save(output / f"{prefix}-case.json", case)
    bench.save(output / f"{prefix}-dry-run.json", entry)
    dry.append(entry)
bench.save(output / "dry-run.json", dry)
print(
    json.dumps(
        {
            "stage": "dry",
            "counts": dict(collections.Counter(entry["status"] for entry in dry)),
            "max_estimated_gib": max(
                (entry.get("estimated_peak_bytes", 0) / 2**30 for entry in dry),
                default=0,
            ),
        },
        ensure_ascii=False,
    ),
    flush=True,
)
if args.dry_only:
    bench.save(output / "summary.json", dry)
    raise SystemExit(0 if all(entry["status"] == "ready" for entry in dry) else 2)

bench.check_device_occupancy(allow_busy=True)

env = os.environ.copy()
for name in (
    "VLLM_SAIL_NVTX_PROFILE",
    "VLLM_PPU_NVTX_PROFILE",
    "SAIL_NVTX_PROFILE",
    "VLLM_SAIL_LAYERWISE_NVTX",
    "VLLM_PPU_LAYERWISE_NVTX",
):
    env[name] = "0"
env.update(
    CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES", "0"),
    VLLM_PLUGINS="ppu",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    VLLM_STEP5_INDEXER_FP8_AS_FP16="0",
    VLLM_SAIL_STEP5_DSA_SCORES="v1",
    VLLM_SAIL_STEP5_DSA_SCORE_DECODE="v2",
)
summary = []
for case, admission in zip(cases, dry, strict=True):
    op, scope = case["op"], case["scope"]
    prefix = admission["output_prefix"]
    if admission["status"] != "ready":
        entry = {
            "case_id": case["id"],
            "op": op,
            "scope": scope,
            "status": "unsupported",
            "reason": admission["reason"],
            "output_dir": str(output),
            "output_prefix": prefix,
            "results": [],
            **{
                name: admission[name]
                for name in (
                    "estimated_peak_bytes",
                    "memory_budget_bytes",
                    "recorded_storage_bytes",
                    "memory_components_per_backend",
                    "memory_components_by_backend",
                )
                if name in admission
            },
        }
        bench.save(output / f"{prefix}-result.json", entry)
        summary.append(entry)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"stage": "unsupported", **entry}), flush=True)
        print_case_timing(entry, args.backends)
        continue
    command = [
        sys.executable,
        str(ROOT / "benchmark_step5_sparse_nvtx.py"),
        "compare",
        "--nvtx",
        *(str(path) for path in args.nvtx),
        "--op",
        op,
        "--backends",
        *args.backends,
        "--workload-config",
        str(output / "workload.json"),
        "--scope",
        "kernel" if scope == "kernel" else "prepared-op",
        "--timer",
        "graph-events",
        "--allow-busy-device",
        "--max-memory-mib",
        "102400",
        "--extension-root",
        str(extensions),
        "--warmup",
        "2",
        "--samples",
        "3",
        "--iterations",
        "5",
        "--output-dir",
        str(output),
        "--output-prefix",
        f"{prefix}-replay",
    ]
    command.extend(bench.utilization_cli_args(args))
    command.extend(["--case-id", case["id"]])
    if args.topk_tensor_dir is not None:
        command.extend(["--topk-tensor-dir", str(args.topk_tensor_dir.resolve())])
    print(
        json.dumps(
            {
                "stage": "execute",
                "op": op,
                "scope": scope,
                "case_id": case["id"],
                "output": str(output),
                "output_prefix": prefix,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    start = time.monotonic()
    with (output / f"{prefix}-run.log").open("x") as log:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    report_path = output / f"{prefix}-replay-report.json"
    entry = {
        "case_id": case["id"],
        "op": op,
        "scope": scope,
        "output_dir": str(output),
        "output_prefix": prefix,
        "report_path": str(report_path),
        "returncode": result.returncode,
        "elapsed_s": time.monotonic() - start,
        "command": command,
        "results": [],
    }
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for item in report["results"]:
            record = {
                "case_id": item["case_id"],
                "status": item["status"],
                "memory": item.get("memory"),
                "error": item.get("error"),
                "synthetic": item.get("synthetic"),
                "topk_tensor_fallback": item.get("topk_tensor_fallback"),
                "arms": [
                    {
                        key: arm.get(key)
                        for key in (
                            "backend",
                            "actual_backend",
                            "implementation",
                            "synthetic",
                            "topk_tensor_fallback",
                            "status",
                            "error",
                            "reason",
                            "config",
                            "metrics",
                            "correctness",
                            "graph_correctness",
                            "batch_graph_correctness",
                            "post_timing_correctness",
                            "timing",
                            "work",
                            "estimated_tflops",
                            "estimated_gbps",
                            "utilization",
                        )
                    }
                    for arm in item["arms"]
                ],
            }
            for arm in record["arms"]:
                if arm["config"] is not None:
                    arm["config"] = {
                        key: value
                        for key, value in arm["config"].items()
                        if key != "normalized_workload"
                    }
            entry["results"].append(record)
    entry["status"] = (
        "completed"
        if result.returncode == 0
        and len(entry["results"]) == 1
        and entry["results"][0]["case_id"] == case["id"]
        and entry["results"][0]["status"] == "completed"
        else "failed"
    )
    bench.save(output / f"{prefix}-result.json", entry)
    summary.append(entry)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stage": "finished",
                "case_id": case["id"],
                "status": entry["status"],
                "op": op,
                "scope": scope,
                "returncode": result.returncode,
                "elapsed_s": entry["elapsed_s"],
                "counts": dict(
                    collections.Counter(item["status"] for item in entry["results"])
                ),
                "max_peak_allocated_gib": max(
                    (
                        item["memory"]["peak_allocated_bytes"] / 2**30
                        for item in entry["results"]
                        if item.get("memory")
                    ),
                    default=0,
                ),
            }
        ),
        flush=True,
    )
    print_case_timing(entry, args.backends)
records = summary
print(
    json.dumps(
        {
            "stage": "total",
            "cases": len(records),
            "counts": dict(collections.Counter(item["status"] for item in records)),
        }
    ),
    flush=True,
)
raise SystemExit(
    1
    if any(entry["status"] == "failed" for entry in summary)
    else 2
    if any(entry["status"] == "unsupported" for entry in summary)
    else 0
)
