# SPDX-License-Identifier: Apache-2.0
"""Independent Step5 sparse UT from NVTX TXT, with optional real input dumps."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from step5_sparse_cases import (  # noqa: E402
    DTYPES,
    estimate_memory,
    extract_cases,
)
from step5_sparse_replay import (  # noqa: E402
    Unsupported,
    fallback_topk_workload,
    normalize_workload,
    resolve_topk_tensor,
)


def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def positive_peak(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Hardware peaks must be finite and positive")
    return value


def add_utilization_arguments(command):
    command.add_argument(
        "--peak-compute-tflops",
        type=positive_peak,
        help="Per-device theoretical compute peak in decimal TFLOP/s; no default",
    )
    command.add_argument(
        "--peak-compute-dtype",
        choices=("fp32", "tf32", "bf16", "fp16", "fp8"),
        help="Precision of the supplied compute peak; not inferred from KV cache dtype",
    )
    command.add_argument(
        "--peak-hbm-gbps",
        type=positive_peak,
        help="Per-device theoretical HBM bandwidth in decimal GB/s; no default",
    )


def utilization_peaks(args):
    compute = getattr(args, "peak_compute_tflops", None)
    dtype = getattr(args, "peak_compute_dtype", None)
    hbm = getattr(args, "peak_hbm_gbps", None)
    if (compute is None) != (dtype is None):
        raise ValueError(
            "Supply --peak-compute-tflops and --peak-compute-dtype together"
        )
    for value in (compute, hbm):
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("Hardware peaks must be finite and positive")
    return dict(
        compute_tflops=compute,
        compute_dtype=dtype,
        hbm_gbps=hbm,
        source="user-supplied per-device theoretical peaks",
    )


def utilization_cli_args(args):
    utilization_peaks(args)
    return [
        part
        for name in ("peak_compute_tflops", "peak_compute_dtype", "peak_hbm_gbps")
        if (value := getattr(args, name, None)) is not None
        for part in ("--" + name.replace("_", "-"), str(value))
    ]


def case_utilization(work, timing, peaks):
    us = timing["median_us"]
    if not math.isfinite(us) or us <= 0:
        raise ValueError("Utilization requires a finite positive device latency")
    tflops = work.get("flops", 0) / us / 1e6
    gbps = work["bytes"] / us / 1000
    compute, hbm = peaks["compute_tflops"], peaks["hbm_gbps"]
    mfu = 100 * tflops / compute if work.get("flops") and compute else None
    bandwidth = 100 * gbps / hbm if hbm else None
    values = dict(
        scope="case-effective, not end-to-end model MFU",
        mfu_pct=mfu,
        mfu_reason=(
            "no-floating-point-work-accounting"
            if not work.get("flops")
            else "compute-peak-not-configured"
            if compute is None
            else None
        ),
        logical_bandwidth_utilization_pct=bandwidth,
        bandwidth_reason="hbm-peak-not-configured" if hbm is None else None,
        peaks=peaks,
        work_accounting=work.get("accounting", "Replay logical bytes and FLOPs"),
        bandwidth_basis="logical minimum traffic, not measured HBM traffic",
        timing_stable=timing["range_pct"] <= 5,
        warnings=[],
    )
    for name, value in (
        ("mfu_pct", mfu),
        ("logical_bandwidth_utilization_pct", bandwidth),
    ):
        if value is not None and value > 100:
            values["warnings"].append(
                name
                + " exceeds 100%; check peaks, accounting, sparsity and warm-cache reuse"
            )
    return values


def case_memory(torch, budget, total_memory):
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    result = dict(
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
        allocator_budget_bytes=budget,
        device_total_bytes=total_memory,
        peak_allocated_capacity_pct=100 * allocated / total_memory,
        peak_reserved_capacity_pct=100 * reserved / total_memory,
        peak_allocator_budget_util_pct=100 * allocated / budget,
        scope="case aggregate: all backends, initialization, correctness and graphs",
        allocator_basis="PyTorch allocator only; native allocations may be excluded",
        device_used_bytes=None,
        device_memory_used_pct=None,
        device_memory_snapshot_total_bytes=None,
        device_memory_basis="end-of-case driver snapshot, not a sampled high-water mark",
    )
    try:
        free, total = torch.cuda.mem_get_info(0)
        if total <= 0 or not 0 <= free <= total:
            raise ValueError("Invalid driver memory counters")
        result.update(
            device_used_bytes=total - free,
            device_memory_used_pct=100 * (total - free) / total,
            device_memory_snapshot_total_bytes=total,
        )
    except (AttributeError, RuntimeError, NotImplementedError, ValueError) as exc:
        result["device_memory_reason"] = "Driver memory snapshot unavailable: " + str(
            exc
        )
    return result


def reject_profiling():
    names = (
        "VLLM_SAIL_NVTX_PROFILE",
        "VLLM_PPU_NVTX_PROFILE",
        "SAIL_NVTX_PROFILE",
        "VLLM_SAIL_LAYERWISE_NVTX",
        "VLLM_PPU_LAYERWISE_NVTX",
    )
    if any(
        os.environ.get(name, "").lower() in ("1", "true", "yes", "on") for name in names
    ):
        raise ValueError(
            "Disable NVTX and profiling before starting the comparison process"
        )
    for name in ("torch", "vllm_sail.profiling.step5_sparse"):
        if name in sys.modules:
            raise ValueError(
                "Comparison must start in a fresh process without torch/profiling"
            )


def workload_for(case, config, *, synthetic=False):
    if case.get("topk_tensor") and not synthetic:
        return {"policy": "tensor-dump", "synthetic": False}
    defaults, cases = config.get("default", {}), config.get("cases", {})
    if (
        not isinstance(defaults, dict)
        or not isinstance(cases, dict)
        or not isinstance(cases.get(case["id"], {}), dict)
    ):
        raise ValueError("Workload default and per-case entries must be mappings")
    values = {
        **({"policy": "capacity-causal"} if synthetic else {}),
        **defaults,
        **cases.get(case["id"], {}),
    }
    if values.get("policy") not in ("capacity-causal", "explicit"):
        raise Unsupported(
            "Select an explicit or capacity-causal synthetic policy in --workload-config"
        )
    for key in ("k_scale", "v_scale", "scale"):
        if key in values and (
            type(values[key]) not in (float, int)
            or not math.isfinite(values[key])
            or values[key] <= 0
        ):
            raise Unsupported("Synthetic scales must be finite and positive")
    return values


def geometry(case, work):
    t, s = case["tensors"], case["scalars"]

    def shape(name, rank):
        item = t.get(name)
        if (
            item is None
            or len(item["shape"]) != rank
            or any(v <= 0 for v in item["shape"])
        ):
            raise Unsupported("Missing or unsupported " + name)
        return item["shape"]

    op = case["op"]
    if op == "topk":
        if case.get("topk_tensor"):
            if case["impl"].endswith("select_all_regions"):
                (rows,) = shape("current", 1)
                k = s.get("topk")
                if type(k) is not int or not 0 <= k <= 4096:
                    raise Unsupported("Missing select-all topk")
                return dict(kind="topk", rows=rows, regions=k, topk=k, select_all=True)
            raw_selector = case["impl"].endswith("raw_select")
            if raw_selector:
                queries, heads, regions = shape("scores", 3)
                if (
                    shape("positions", 1) != [queries]
                    or t["positions"]["dtype"] != "torch.int64"
                ):
                    raise Unsupported("Raw selector requires per-query int64 positions")
                rows = queries * heads
            else:
                rows, regions = shape("scores", 2)
                queries, heads = rows, 1
                if case["scope"] == "op":
                    if (
                        shape("current", 1) != [rows]
                        or t["current"]["dtype"] != "torch.int64"
                    ):
                        raise Unsupported(
                            "Prepared TopK requires per-row int64 current"
                        )
            if t["scores"]["dtype"] not in ("torch.float32", "torch.bfloat16"):
                raise Unsupported("TopK dump requires FP32/BF16 scores")
            k = s.get("topk")
            if type(k) is not int or not 0 < k <= (1024 if raw_selector else 4096):
                raise Unsupported("Missing/unsupported recorded topk")
            return dict(
                kind="topk",
                rows=rows,
                regions=regions,
                topk=k if case["scope"] == "kernel" else min(k, regions),
                raw_selector=raw_selector,
                queries=queries,
                heads=heads,
            )
        rows, regions = shape("scores", 2)
        if t["scores"]["dtype"] != "torch.float32" or shape("current", 1) != [rows]:
            raise Unsupported("Topk requires FP32 scores and per-row current")
        k = s.get("topk")
        if type(k) is not int or not 0 < k <= 1024:
            raise Unsupported("Missing/unsupported topk (1..1024)")
        return dict(kind="topk", rows=rows, regions=regions, topk=min(k, regions))
    if op == "score":
        q = shape("query", 4)
        if q[-1] != 256 or q[2] not in (4, 8, 16):
            raise Unsupported("Unsupported score query geometry")
        regions = s.get("max_seq_len")
        if type(regions) is not int or regions <= 0:
            raise Unsupported("Missing max_seq_len for scoring")
        if t.get("mean"):
            mean = shape("mean", 3)
            reqs = shape("block_table", 2)[0]
            qlen = s.get("query_len") or 1
            if type(qlen) is not int or qlen <= 0:
                raise Unsupported("Invalid direct query_len")
            if q[0] != reqs * qlen or mean[-1] != 256 or mean[1] not in (2, 4, 8):
                raise Unsupported("Direct decode request/summary geometry mismatch")
            return dict(
                kind="direct",
                rows=q[0] * q[1],
                tokens=q[0],
                groups=q[1],
                requests=reqs,
                query_len=qlen,
                regions=regions,
                rpb=mean[1],
                heads=q[2],
            )
        packed = shape("packed", 4)
        if packed[1:] != [32, 1, 260] or t["packed"]["dtype"] != "torch.uint8":
            raise Unsupported("Expected packed FP8 summary pages [P,32,1,260]")
        grouped = t.get("spans") is not None
        next_n = s.get("next_n", q[1] if not grouped else 1)
        if next_n not in (1, 2):
            raise Unsupported("Packed replay supports next_n=1/2")
        rows = q[0] if grouped else q[0] * next_n
        return dict(
            kind="grouped" if grouped else "packed",
            rows=rows,
            groups=q[1] if grouped else 1,
            regions=regions,
            heads=q[2],
            next_n=next_n,
        )
    if case["impl"].endswith(("_sparse_decode_split", "_sparse_decode_combine")):
        raise Unsupported(
            "Raw attention split/combine has no complete fixed-selection boundary"
        )
    rows, heads, dim = shape("query", 3)
    cache = shape("kv_cache", 4)
    table = shape("block_table", 2)
    indices = shape("indices", 2)
    if dim != 192 or heads not in (4, 8, 16) or cache[-1] != 384 or indices[0] != rows:
        raise Unsupported("Unsupported DSA attention geometry")
    if t["query"]["dtype"] != "torch.bfloat16":
        raise Unsupported("Attention replay requires BF16 queries")
    phase = case["pd_type"] if case["pd_type"] != "unknown" else work.get("pd_type")
    if phase not in ("prefill", "decode"):
        raise Unsupported("Attention requires a recorded or explicit phase")
    groups = cache[1]
    if phase == "prefill" and groups != 1:
        raise Unsupported("Prefill op replay requires a single-group cache slice")
    if case.get("topk_tensor"):
        for name, dtype in (
            ("positions", "torch.int64"),
            ("indices", "torch.int64"),
            ("block_table", "torch.int32"),
        ):
            if t.get(name) is None or t[name]["dtype"] != dtype:
                raise Unsupported("Recorded attention requires " + dtype + " " + name)
        if shape("positions", 1) != [rows]:
            raise Unsupported("Recorded attention positions/rows mismatch")
        name, dtype = (
            ("active", "torch.bool") if phase == "decode" else ("owners", "torch.int64")
        )
        if shape(name, 1) != [rows] or t[name]["dtype"] != dtype:
            raise Unsupported(
                "Recorded attention requires per-row " + dtype + " " + name
            )
    qlen = case["scalars"].get("query_len", work.get("query_len", 1))
    if type(qlen) is not int or qlen <= 0:
        raise Unsupported("Attention query_len must be a positive integer")
    if phase == "decode" and rows != table[0] * groups * qlen:
        raise Unsupported("Folded decode rows/request/group/query_len mismatch")
    return dict(
        kind="attention",
        rows=rows,
        heads=heads,
        groups=groups,
        requests=table[0],
        tokens=rows // groups,
        query_len=qlen,
        block_size=cache[2],
        pages=cache[0],
        topk=indices[1],
        pd_type=phase,
    )


def dry_case(case, config, args):
    result = dict(
        case_id=case["id"],
        op=case["op"],
        source=case["sources"],
        synthetic=not bool(case.get("topk_tensor")),
        memory_budget_bytes=args.max_memory_mib * (1 << 20),
        utilization_peaks=utilization_peaks(args),
    )
    try:
        recorded_storage = estimate_memory(case)
        result.update(
            recorded_storage_bytes=recorded_storage,
            estimated_peak_bytes=recorded_storage,
        )
        if recorded_storage > result["memory_budget_bytes"]:
            raise Unsupported("Recorded storage exceeds conservative memory budget")
        path = None
        work = workload_for(case, config)
        if case.get("topk_tensor"):
            path = resolve_topk_tensor(case, getattr(args, "topk_tensor_dir", None))
            if path is None:
                work = fallback_topk_workload(
                    case, workload_for(case, config, synthetic=True)
                )
                result.update(
                    synthetic=True, topk_tensor_fallback=work["topk_tensor_fallback"]
                )
            else:
                result["content_validation"] = "deferred-until-torch-load"
            expected_scope = "kernel" if case["scope"] == "kernel" else "prepared-op"
            if args.scope != expected_scope:
                raise Unsupported("Tensor dump must replay its recorded scope")
        geo = geometry(case, work)
        work = normalize_workload(case, geo, work)
        if path is not None:
            work["topk_tensor_path"] = str(path)
            defaults, overrides = config.get("default", {}), config.get("cases", {})
            work["synthetic_fallback_workload"] = {
                "policy": "capacity-causal",
                **defaults,
                **overrides.get(case["id"], {}),
            }
        output = geo["rows"] * (
            geo.get("regions", 0) * 4 * geo.get("groups", 1)
            or geo.get("heads", 1) * 192 * 2
        )
        backends = args.backends or default_backends(geo)
        gathered = (
            geo["rows"] * geo.get("topk", 0) * 8 * 384 * 2
            if geo["kind"] == "attention"
            else 0
        )
        logical_inputs = sum(
            math.prod(t["shape"]) * DTYPES[t["dtype"]]
            for t in case["tensors"].values()
            if t is not None
        )
        memory_components = dict(
            recorded_storage_bytes=recorded_storage,
            initialization_budget_bytes=logical_inputs * 4,
            output_budget_bytes=output * 4,
            gathered_budget_bytes=gathered * 3,
            workspace_budget_bytes=256 << 20,
        )
        memory_components_by_backend = {
            backend: {
                **memory_components,
                "gathered_budget_bytes": (
                    0
                    if geo["kind"] == "attention" and backend == "native"
                    else gathered * 3
                ),
            }
            for backend in backends
        }
        estimate = sum(
            sum(components.values())
            for components in memory_components_by_backend.values()
        )
        # Keep the legacy per-backend upper bound for mixed-backend reports.
        memory_components = {
            name: max(parts[name] for parts in memory_components_by_backend.values())
            for name in memory_components
        }
        result.update(
            geometry=geo,
            workload=work,
            estimated_peak_bytes=estimate,
            memory_components_per_backend=memory_components,
            memory_components_by_backend=memory_components_by_backend,
            scope=args.scope,
            backends=backends,
            backend_support=[
                static_support(case, geo, name, args.scope) for name in backends
            ],
        )
        if estimate > result["memory_budget_bytes"]:
            raise Unsupported(
                "Conservative inputs/outputs/workspace estimate exceeds --max-memory-mib"
            )
        result["status"] = "ready"
    except Unsupported as exc:
        result.update(status="unsupported", reason=str(exc))
    return result


def default_backends(geo):
    if geo["kind"] == "topk":
        return ["per_row", "deep_select", "selector_v2"]
    if geo["kind"] == "direct":
        return ["direct_baseline", "direct_resident", "direct_mtp"]
    if geo["kind"] in ("packed", "grouped"):
        return ["upstream_paged", "resident", "compact_grouped"]
    return (
        ["upstream_triton", "native"]
        if geo["pd_type"] == "decode"
        else ["flashinfer_paged", "flashinfer_ragged", "native"]
    )


def static_support(case, geo, backend, scope):
    reason = None
    if backend not in default_backends(geo):
        reason = "Backend does not implement this contract group"
    elif case.get("topk_tensor") and geo.get("select_all"):
        pass
    elif case.get("topk_tensor") and geo["kind"] == "topk" and geo.get("raw_selector"):
        if backend != "selector_v2" or scope != "kernel":
            reason = "Raw selector dump requires selector_v2 kernel scope"
    elif (
        case.get("topk_tensor")
        and geo["kind"] == "topk"
        and (
            (
                case["tensors"]["scores"]["dtype"] != "torch.float32"
                and (backend != "deep_select" or scope != "kernel")
            )
            or ("deep_select_topk" in case["impl"] and backend != "deep_select")
            or (backend == "selector_v2" and geo["topk"] > 1024)
        )
    ):
        reason = "Backend does not implement this recorded dtype/entry contract"
    elif backend == "compact_grouped" and (
        geo["kind"] != "grouped" or scope == "kernel"
    ):
        reason = "Requires recorded grouped spans and prepared-op boundary"
    elif backend == "selector_v2" and (
        scope == "kernel" or case["tensors"]["scores"]["stride"] != [geo["regions"], 1]
    ):
        reason = "Requires contiguous scores and complete selection boundary"
    elif geo["kind"] == "attention" and scope == "kernel" and backend != "native":
        reason = "No identical single-kernel native input/output boundary"
    elif (
        backend == "flashinfer_paged"
        and case["tensors"]["kv_cache"]["dtype"] != "torch.bfloat16"
    ):
        reason = "Requires BF16 paged KV"
    return dict(
        backend=backend,
        status="unsupported" if reason else "requires-device-admission",
        reason=reason,
        contract_group=geo["kind"] + ":" + scope,
    )


def provenance(torch):
    def git(*parts):
        try:
            result = subprocess.run(
                ["git", "-C", str(ROOT), *parts],
                text=True,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    sources = []
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and (
            name.startswith(("vllm_sail", "vllm.models.step5"))
            or Path(path).resolve().parent == ROOT
        ):
            path = Path(path).resolve()
            if path.is_file():
                sources.append(
                    dict(
                        module=name,
                        path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    packages = {}
    for name in ("torch", "vllm", "vllm-sail", "triton", "flashinfer-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    props = torch.cuda.get_device_properties(0)
    return dict(
        revision=git("rev-parse", "HEAD"),
        dirty=git("status", "--porcelain"),
        sources=sources,
        versions=packages,
        device=dict(
            name=props.name,
            uuid=str(getattr(props, "uuid", "unknown")),
            capability=list(torch.cuda.get_device_capability(0)),
        ),
        env={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_SAIL_", "VLLM_PPU_", "VLLM_STEP5_"))
            or k in ("CUDA_VISIBLE_DEVICES", "VLLM_PLUGINS")
        },
    )


def summarize(samples):
    if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise RuntimeError("Invalid timer samples")
    ordered = sorted(samples)
    median = statistics.median(samples)
    return dict(
        median_us=median,
        min_us=min(samples),
        max_us=max(samples),
        p90_us=ordered[math.ceil(0.9 * len(ordered)) - 1],
        range_pct=100 * (max(samples) - min(samples)) / median,
        samples_us=samples,
    )


def check_device_occupancy(allow_busy=False):
    from step5_sparse_support import DeviceBusyError, idle_gate

    try:
        return idle_gate()
    except DeviceBusyError as exc:
        if not allow_busy:
            raise
        print(
            "WARNING: Other GPU compute processes detected; "
            "performance measurements may be inaccurate. UT will continue.",
            file=sys.stderr,
            flush=True,
        )
        return exc.snapshot


def compare_case(case, dry, args, runtime, idle_gate):
    from step5_sparse_replay import prepare_arms

    torch = runtime
    result = {
        **dry,
        "arms": [],
        "timer": args.timer,
        "cache": "cold-l2" if args.timer == "cupti" else "warm",
    }
    arms, blocked = prepare_arms(
        torch, case, dry["geometry"], dry["workload"], dry["backends"], args
    )
    result["arms"].extend(blocked)
    normalized = (
        arms[0].config.get("normalized_workload", dry["workload"])
        if arms
        else blocked[0].get("normalized_workload", dry["workload"])
        if blocked
        else dry["workload"]
    )
    result["workload"] = normalized
    result["synthetic"] = normalized.get(
        "synthetic", dry.get("synthetic", not bool(case.get("topk_tensor")))
    )
    if normalized.get("topk_tensor_fallback"):
        result["topk_tensor_fallback"] = normalized["topk_tensor_fallback"]

    def check_correctness(arm, item, stage):
        try:
            item[stage] = arm.check()
        except AssertionError as exc:
            item[stage] = dict(
                status="failed", error=str(exc), metrics=getattr(exc, "metrics", None)
            )
            item["status"] = "correctness_failed"
            item.setdefault("error", str(exc))
            item.setdefault("metrics", getattr(exc, "metrics", None))

    valid = []
    for arm in arms:
        item = dict(
            backend=arm.backend,
            requested_backend=arm.backend,
            actual_backend=arm.config.get("actual_backend", arm.backend),
            implementation=arm.impl,
            config=arm.config,
            contract=arm.contract,
            synthetic=arm.config.get("normalized_workload", {}).get(
                "synthetic", result["synthetic"]
            ),
            status="passed",
        )
        if arm.config.get("normalized_workload", {}).get("topk_tensor_fallback"):
            item["topk_tensor_fallback"] = arm.config["normalized_workload"][
                "topk_tensor_fallback"
            ]
        result["arms"].append(item)
        try:
            idle_gate()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                arm.call()
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(args.warmup):
                arm.call()
            torch.cuda.synchronize()
            check_correctness(arm, item, "correctness")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                arm.call()
            graph.replay()
            torch.cuda.synchronize()
            check_correctness(arm, item, "graph_correctness")
            if args.timer == "graph-events":
                batch_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(batch_graph):
                    for _ in range(args.iterations):
                        arm.call()
                batch_graph.replay()
                torch.cuda.synchronize()
                check_correctness(arm, item, "batch_graph_correctness")
                graph = batch_graph
            valid.append((arm, graph, item, []))
        except Exception as exc:
            item.update(status="execution_failed", error=str(exc))
            result["status"] = "failed"
            return result
    if not valid:
        result["status"] = (
            "failed"
            if any(
                item["status"] in ("correctness_failed", "execution_failed")
                for item in result["arms"]
            )
            else "unsupported"
        )
        return result
    for sample in range(args.samples):
        for arm, graph, item, samples in valid if sample % 2 == 0 else reversed(valid):
            try:
                idle_gate()
                if args.timer == "graph-events":
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    value = start.elapsed_time(end) * 1000 / args.iterations
                else:
                    from flashinfer.testing import bench_gpu_time_with_cupti

                    values = bench_gpu_time_with_cupti(
                        arm.call,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                        dry_run_iters=args.warmup,
                        repeat_iters=args.iterations,
                    )
                    value = statistics.median(values) * 1000
                    item.setdefault("cupti_raw_ms", []).append(list(map(float, values)))
                idle_gate()
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError("Invalid timer sample")
                samples.append(value)
            except Exception as exc:
                item.update(
                    status="execution_failed", error=str(exc), failed_round=sample
                )
                for _, _, previous, raw in valid:
                    previous["samples_us"] = raw
                result["status"] = "failed"
                return result
    for arm, _graph, item, samples in valid:
        try:
            check_correctness(arm, item, "post_timing_correctness")
        except Exception as exc:
            item.update(status="execution_failed", error=str(exc), samples_us=samples)
            continue
        item["timing"] = {
            **summarize(samples),
            "time_domain": "device",
            "method": args.timer,
            "valid_for_comparison": item["status"] != "correctness_failed",
            "op_repeats_per_graph": args.iterations
            if args.timer == "graph-events"
            else None,
        }
        if item["status"] == "correctness_failed":
            continue
        stable = item["timing"]["range_pct"] <= 5
        item.update(status="stable" if stable else "unstable", work=arm.work)
        us = item["timing"]["median_us"]
        item["estimated_gbps"] = arm.work["bytes"] / us / 1000
        if arm.work.get("flops"):
            item["estimated_tflops"] = arm.work["flops"] / us / 1e6
        item["utilization"] = case_utilization(
            arm.work, item["timing"], utilization_peaks(args)
        )
    stable = [item for item in result["arms"] if item["status"] == "stable"]
    baseline = next(
        (item for item in stable if item["backend"] == dry["backends"][0]), None
    )
    if baseline:
        for item in stable:
            if item["contract"] != baseline["contract"]:
                continue
            item["speedup_vs"] = baseline["backend"]
            item["speedup"] = (
                baseline["timing"]["median_us"] / item["timing"]["median_us"]
            )
    result["status"] = (
        "failed"
        if any(
            item["status"] in ("correctness_failed", "execution_failed")
            for item in result["arms"]
        )
        else "completed"
    )
    return result


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--nvtx", nargs="+", required=True)
    extract.add_argument("--output", required=True, type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("--nvtx", nargs="+", required=True)
    compare.add_argument(
        "--op", choices=("score", "topk", "attention", "all"), default="all"
    )
    compare.add_argument("--case-id", action="append")
    compare.add_argument("--backends", nargs="+")
    compare.add_argument("--workload-config", type=Path)
    compare.add_argument(
        "--topk-tensor-dir",
        type=Path,
        help="Directory for topk_tensor files; default is the source TXT directory",
    )
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument(
        "--scope", choices=("prepared-op", "kernel"), default="prepared-op"
    )
    compare.add_argument(
        "--timer", choices=("auto", "cupti", "graph-events"), default="auto"
    )
    compare.add_argument("--warmup", type=int, default=5)
    compare.add_argument("--samples", type=int, default=12)
    compare.add_argument("--iterations", type=int, default=20)
    compare.add_argument("--max-memory-mib", type=int, default=2048)
    compare.add_argument("--extension-root", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument(
        "--output-prefix",
        default="",
        help="Filename prefix for sharing an output directory without overwriting files",
    )
    compare.add_argument("--dry-run", action="store_true")
    compare.add_argument(
        "--allow-busy-device",
        action="store_true",
        help=(
            "Warn and continue UT when other GPU compute processes are present; "
            "performance measurements may be inaccurate"
        ),
    )
    add_utilization_arguments(compare)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    extracted = extract_cases(args.nvtx)
    if not extracted["cases"]:
        raise ValueError("No versioned Step5 sparse op labels found")
    if args.command == "extract":
        save(args.output, extracted)
        return 0
    peaks = utilization_peaks(args)
    reject_profiling()
    if any(
        v <= 0
        for v in (args.warmup, args.samples, args.iterations, args.max_memory_mib)
    ):
        raise ValueError("Warmup/samples/iterations/memory budget must be positive")
    if args.backends and len(args.backends) != len(set(args.backends)):
        raise ValueError("Duplicate backend selection")
    if args.case_id and set(args.case_id) - {c["id"] for c in extracted["cases"]}:
        raise ValueError("Unknown case ID")
    config = (
        json.loads(args.workload_config.read_text()) if args.workload_config else {}
    )
    if not isinstance(config, dict) or set(config) - {"default", "cases"}:
        raise ValueError("Workload config must contain default/cases mappings")
    cases = [
        c
        for c in extracted["cases"]
        if (args.op == "all" or c["op"] == args.op)
        and (not args.case_id or c["id"] in args.case_id)
    ]
    if not cases:
        raise ValueError("No cases match the selection")
    if args.output_prefix and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.output_prefix
    ):
        raise ValueError("Output prefix must be a plain filename prefix")

    def output_path(name):
        filename = f"{args.output_prefix}-{name}" if args.output_prefix else name
        return args.output_dir / filename

    if (
        not args.output_prefix
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise ValueError("Output directory is not empty")
    names = ["cases.json", "dry-run.json", "report.json", "results.csv"]
    names.extend(case["id"] + ".json" for case in cases)
    for name in names:
        path = output_path(name)
        if path.exists():
            raise ValueError(f"Output file already exists: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dry = [dry_case(c, config, args) for c in cases]
    save(output_path("cases.json"), {**extracted, "cases": cases})
    save(output_path("dry-run.json"), dry)
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return 0 if all(c["status"] == "ready" for c in dry) else 2
    if any(c["status"] != "ready" for c in dry):
        raise ValueError(
            "Resolve unsupported cases in dry-run.json before device execution"
        )
    check_device_occupancy(args.allow_busy_device)
    idle_gate = (lambda: None) if args.allow_busy_device else check_device_occupancy
    from step5_sparse_replay import initialize_runtime

    torch = initialize_runtime(args.extension_root)
    budget = args.max_memory_mib * (1 << 20)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total_memory), 0)
    torch.cuda.reset_peak_memory_stats()
    name = torch.cuda.get_device_name(0).upper()
    ppu = any(word in name for word in ("PPU", "ZW-", "ZW8", "M890"))
    if args.timer == "auto":
        args.timer = "graph-events" if ppu else "cupti"
    if ppu and args.timer == "cupti":
        raise ValueError(
            "PPU CUPTI/cold-L2 is not verified; explicitly use graph-events warm-cache"
        )
    report = dict(
        schema=1,
        synthetic=all(c["synthetic"] for c in dry),
        results=[],
        command=sys.argv,
        seed=args.seed,
        scope=args.scope,
        timer=args.timer,
        allocator_budget_bytes=budget,
        utilization_peaks=peaks,
        note="Existing launch configurations, not tuned backend optima or serving performance.",
    )
    try:
        for case, entry in zip(cases, dry, strict=False):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                result = compare_case(case, entry, args, torch, idle_gate)
            except Exception as exc:
                result = {**entry, "status": "failed", "arms": [], "error": str(exc)}
            result["memory"] = case_memory(torch, budget, total_memory)
            report["results"].append(result)
            save(output_path(case["id"] + ".json"), result)
            print(
                json.dumps(dict(case_id=case["id"], status=result["status"])),
                flush=True,
            )
            if torch.cuda.max_memory_allocated() > args.max_memory_mib * (1 << 20):
                raise RuntimeError("Actual peak allocation exceeded memory budget")
        idle_gate()
    finally:
        report["synthetic"] = all(r["synthetic"] for r in report["results"])
        report["provenance"] = provenance(torch)
        save(output_path("report.json"), report)
    with output_path("results.csv").open("x", newline="") as stream:
        metric_columns = [
            "median_us",
            "range_pct",
            "speedup",
            "estimated_gbps",
            "estimated_tflops",
            "mfu_pct",
            "logical_bandwidth_utilization_pct",
        ]
        memory_columns = [
            "peak_allocated_capacity_pct",
            "peak_reserved_capacity_pct",
            "peak_allocator_budget_util_pct",
            "device_memory_used_pct",
        ]
        columns = [
            "case_id",
            "op",
            "backend",
            "actual_backend",
            "status",
            "synthetic",
            *metric_columns,
            *("case_" + name for name in memory_columns),
            "utilization_json",
            "memory_json",
            "seed",
            "timer",
            "cache",
            "scope",
            "implementation",
            "contract",
            "source_json",
            "workload_json",
            "config_json",
            "correctness_json",
            "provenance_json",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in report["results"]:
            for arm in case["arms"]:
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "op": case["op"],
                        "backend": arm["backend"],
                        "actual_backend": arm.get("actual_backend", ""),
                        "status": arm["status"],
                        "synthetic": arm.get("synthetic", case["synthetic"]),
                        **{
                            k: arm.get("utilization", {}).get(
                                k, arm.get("timing", {}).get(k, arm.get(k, ""))
                            )
                            for k in metric_columns
                        },
                        **{
                            "case_" + k: case.get("memory", {}).get(k)
                            for k in memory_columns
                        },
                        "utilization_json": json.dumps(arm.get("utilization", {})),
                        "memory_json": json.dumps(case.get("memory", {})),
                        "seed": args.seed,
                        "timer": args.timer,
                        "cache": case.get("cache", ""),
                        "scope": args.scope,
                        "implementation": arm.get("implementation", ""),
                        "contract": arm.get("contract", ""),
                        "source_json": json.dumps(case["source"]),
                        "workload_json": json.dumps(case.get("workload", {})),
                        "config_json": json.dumps(arm.get("config", {})),
                        "correctness_json": json.dumps(arm.get("correctness", {})),
                        "provenance_json": json.dumps(report["provenance"]),
                    }
                )
    return (
        1
        if any(r["status"] == "failed" for r in report["results"])
        else 2
        if all(r["status"] == "unsupported" for r in report["results"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
