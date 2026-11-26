# SPDX-License-Identifier: Apache-2.0
"""Real Step5 attention standalone replay; device dependencies are imported lazily and compute functions are never modified."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from tests.support.step5_attention_cases import (
    attention_work,
    effective_tokens,
    mutable_state_rows,
    peak,
    positive,
)

PROFILE_SWITCHES = (
    "VLLM_SAIL_NVTX_PROFILE",
    "VLLM_PPU_NVTX_PROFILE",
    "SAIL_NVTX_PROFILE",
)


def reject_instrumentation():
    from vllm_sail import envs

    if envs.VLLM_SAIL_NVTX_PROFILE or any(
        os.environ.get(name, "").lower() in ("1", "true", "yes", "on")
        for name in PROFILE_SWITCHES
    ):
        raise ValueError(
            "formal performance UTs must disable NVTX and its aliases; profiling runs in a separate pass"
        )


@contextmanager
def override_step5_environment(values):
    """Temporarily set Step5 environment values for standalone A/B scopes."""
    from vllm_sail import envs

    values = {
        envs._canonical_names.get(name, name): value for name, value in values.items()
    }
    if any(
        name not in envs.environment_variables
        or not (
            name.startswith("VLLM_SAIL_STEP5_")
            or name == "VLLM_STEP5_INDEXER_FP8_AS_FP16"
        )
        for name in values
    ):
        raise ValueError("Only Step5 settings may be overridden in this test scope")
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def require_runtime(implementation="default", extension_root=None):
    reject_instrumentation()
    import torch

    import vllm_sail

    if extension_root is not None:
        directory = str(Path(extension_root).resolve())
        if not Path(directory).is_dir():
            raise ValueError("extension-root directory does not exist")
        if directory not in vllm_sail.__path__:
            # Only extend the binary search path; Python sources from older wheels must not shadow the current package.
            vllm_sail.__path__.append(directory)
    if not torch.cuda.is_available():
        raise RuntimeError("no PPU device available")
    name = torch.cuda.get_device_name(0)
    if not any(word in name.upper() for word in ("PPU", "ZW-", "ZW8", "M890")):
        raise RuntimeError(f"this UT requires PPU, actual device: {name}")
    if implementation not in ("default", "flashinfer", "native"):
        raise ValueError("unknown implementation")
    os.environ["VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND"] = (
        "flashinfer" if implementation == "default" else implementation
    )
    os.environ["VLLM_SAIL_STEP5_DSA_PREFILL_PPU"] = str(int(implementation == "native"))
    vllm_sail.register()
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia import sparse_attention
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        raise RuntimeError(
            "PPU platform not registered; set VLLM_PLUGINS=ppu before process start"
        )
    package_root = os.environ.get("VLLM_SAIL_TEST_PACKAGE_ROOT")
    if package_root:
        import sys

        root = Path(package_root).resolve()
        for module_name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if module_name.startswith("vllm_sail") and path and path.endswith(".py"):
                if not Path(path).resolve().is_relative_to(root):
                    raise RuntimeError(
                        f"isolated wheel acceptance loaded a worktree module: {module_name}: {path}"
                    )
    return torch, sparse_attention


def prepare_case(case, implementation="default", extension_root=None):
    torch, module = require_runtime(implementation, extension_root)
    from vllm.forward_context import ForwardContext
    from vllm.models.step5.nvidia.csa_tail_ring import (
        Step5CSATailRing,
        Step5CSATailRingMetadata,
    )
    from vllm.models.step5.nvidia.sparse_summary_cache import Step5SparseSummaryCache

    from vllm_sail.models.step5.ops import soft_fp8

    geo, syn = case["geometry"], case["synthetic"]
    required = case["memory_estimate"]["estimated_peak_bytes"]
    free, _ = torch.cuda.mem_get_info()
    if required > free * 0.9:
        raise MemoryError(
            f"case budget {required / 2**30:.2f} GiB > safe free-memory budget {free * 0.9 / 2**30:.2f} GiB; shrinking shapes is refused"
        )
    prefix = "dsa_proxy" if case["pd_type"] == "prefill" else "proxy"
    expected_proxy = str(soft_fp8.indexer_query_dtype())
    if case["tensors"][prefix + "_query"]["dtype"] != expected_proxy:
        raise ValueError(
            f"log proxy dtype mismatches the indexer mode: expected {expected_proxy}; explicitly select the mode matching the original log"
        )
    torch.manual_seed(syn["seed"])
    rng = random.Random(syn["seed"])
    pages = rng.sample(range(geo["pages"]), sum(syn["page_counts"]))
    tables, cursor = [], 0
    for count in syn["page_counts"]:
        tables.append(pages[cursor : cursor + count])
        cursor += count
    owners, positions, slots = [], [], []
    for request, (seq, q) in enumerate(
        zip(syn["seq_lens"], syn["query_lens"], strict=False)
    ):
        for pos in range(seq - q, seq):
            owners.append(request)
            positions.append(pos)
            slots.append(
                tables[request][pos // geo["block_size"]] * geo["block_size"]
                + pos % geo["block_size"]
            )

    def allocate(name, output=False):
        item = case["tensors"].get(name)
        if item is None:
            return None
        dtype = getattr(torch, item["dtype"].removeprefix("torch."))
        shape = item["shape"]
        if output:
            return torch.empty(shape, device="cuda", dtype=dtype)
        return (torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.2).to(dtype)

    tensors = {
        name: allocate(
            name,
            name
            in (
                "output",
                "kv_cache",
                "block_table",
                "seq_lens",
                "slot_mapping",
                "query_start_loc",
            ),
        )
        for name in case["tensors"]
    }
    cache = tensors["kv_cache"]
    outer = case.get("entry") == "forward"
    if outer:
        # Match production indexer group-major storage without duplicating K.
        for name in ("dsa_proxy_query", "dsa_proxy_weights"):
            tensors[name] = tensors[name].transpose(0, 1).contiguous().transpose(0, 1)
    # The cache is fully allocated; only live pages that will be read are initialized, and unused pages never enter the table.
    storage = (
        cache.view(torch.uint8)
        if cache.dtype in (torch.float8_e4m3fn, torch.uint8)
        else cache
    )
    for start in range(0, len(pages), 32):
        ids = torch.tensor(pages[start : start + 32], device="cuda", dtype=torch.int64)
        values = (
            torch.randn(
                (len(ids), *cache.shape[1:]), dtype=torch.bfloat16, device="cuda"
            )
            * 0.2
        )
        if cache.dtype in (torch.float8_e4m3fn, torch.uint8):
            values = values.to(torch.float8_e4m3fn).view(torch.uint8)
        storage.index_copy_(0, ids, values)
    tensors["block_table"].fill_(-1)
    for request, table in enumerate(tables):
        tensors["block_table"][request, : len(table)].copy_(
            torch.tensor(table, dtype=torch.int32, device="cuda")
        )
    for name, values in (
        ("seq_lens", syn["seq_lens"]),
        ("slot_mapping", slots),
        ("query_start_loc", syn["query_start_loc"]),
    ):
        tensors[name].copy_(
            torch.tensor(values, dtype=tensors[name].dtype, device="cuda")
        )
    for name in ("k_scale", "v_scale"):
        if tensors.get(name) is not None:
            tensors[name].fill_(syn[name])
    metadata = SimpleNamespace(
        block_table=tensors["block_table"],
        seq_lens=tensors["seq_lens"],
        slot_mapping=tensors["slot_mapping"],
        query_start_loc=tensors["query_start_loc"],
        query_start_loc_cpu=torch.tensor(syn["query_start_loc"], dtype=torch.int32),
        num_actual_reqs=geo["requests"],
        num_actual_tokens=geo["tokens"],
        max_query_len=syn["query_len"],
        step5_runtime=None,
    )
    # Build only the real components the entry needs; never load the full model or bypass operator execution.
    impl = object.__new__(module.Step5DSAAttentionImpl)
    impl.scale, impl.sparse_topk = 192**-0.5, syn["topk"]
    impl.num_speculative_tokens = syn["num_speculative_tokens"]
    impl.indexer_heads, impl.max_num_seqs = (
        4 if case.get("paired") else geo["indexer_heads"],
        geo["requests"],
    )
    impl.max_num_regions = (max(syn["seq_lens"]) + 7) // 8
    impl.adapter = module._shared_adapter(torch.device("cuda:0"))
    summary_dtype = torch.float16 if expected_proxy == "torch.float16" else torch.uint8
    mean = torch.empty(
        (geo["pages"], geo["block_size"] // 8, 256), device="cuda", dtype=summary_dtype
    )
    for start in range(0, len(pages), 64):
        ids = torch.tensor(pages[start : start + 64], device="cuda", dtype=torch.int64)
        values = (
            torch.randn(
                (len(ids), geo["block_size"] // 8, 256),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        values = values.to(torch.float8_e4m3fn)
        mean.index_copy_(
            0,
            ids,
            values.to(torch.float16)
            if summary_dtype == torch.float16
            else values.view(torch.uint8),
        )
    impl.summary_cache = Step5SparseSummaryCache(mean)
    ring = Step5CSATailRing(
        prefix="attention_ut_ring",
        main_layer_name="attention_ut",
        num_speculative_tokens=impl.num_speculative_tokens,
        static_forward_context={},
    )
    capacity = max(8, ring.required_rows, geo["block_size"] * geo["groups"])
    ring.bind_kv_cache(
        torch.randn(
            (geo["requests"] + 1, capacity, 512), device="cuda", dtype=torch.bfloat16
        )
        * 0.2
    )
    impl.csa_ring = ring
    base = [(owner + 1) * capacity for owner in owners]
    ring_slots = [
        b + p % capacity if p >= syn["seq_lens"][owner] - capacity else -1
        for b, p, owner in zip(base, positions, owners, strict=False)
    ]
    ring_meta = Step5CSATailRingMetadata(
        torch.tensor(base, device="cuda", dtype=torch.int64),
        torch.tensor(ring_slots, device="cuda", dtype=torch.int64),
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={ring.layer_name: ring_meta},
        slot_mapping={},
    )
    if case["pd_type"] == "prefill":
        import numpy as np
        from vllm.models.step5.nvidia.ops.sparse_prefill_pages import kv_page_stride

        layout = module._PrefillLayout(
            geo["heads"],
            192,
            geo["groups"] if outer else 1,
            torch.bfloat16,
            cache.dtype != torch.bfloat16,
            geo["block_size"],
            kv_page_stride(cache),
        )
        metadata.step5_runtime = impl.plan_prefill_step(
            metadata, geo["tokens"], np.array(syn["seq_lens"]), layout
        )
        if metadata.step5_runtime is None:
            raise RuntimeError("the real planner cannot build the prefill runtime")
    # Following the real summary launcher's write contract, restore only rows this step may have written.
    summary_rows, ring_rows = mutable_state_rows(case, tables, capacity)
    summary_ids = torch.tensor(summary_rows, device="cuda", dtype=torch.int64)
    ring_ids = torch.tensor(ring_rows, device="cuda", dtype=torch.int64)
    flat_mean = mean.view(-1, 256)
    initial_mean = flat_mean.index_select(0, summary_ids).clone()
    initial_ring = ring.rows.index_select(0, ring_ids).clone()

    def reset():
        if summary_rows:
            flat_mean.index_copy_(0, summary_ids, initial_mean)
        if ring_rows:
            ring.rows.index_copy_(0, ring_ids, initial_ring)

    def call():
        if outer:
            return impl.forward(
                SimpleNamespace(
                    _k_scale=tensors.get("k_scale"), _v_scale=tensors.get("v_scale")
                ),
                tensors["query"],
                None,
                None,
                cache,
                metadata,
                output=tensors["output"],
                dsa_proxy_query=tensors["dsa_proxy_query"],
                dsa_proxy_key=tensors["dsa_proxy_key"],
                dsa_proxy_weights=tensors["dsa_proxy_weights"],
                dsa_proxy_z=tensors["dsa_proxy_z"],
            )
        args = (
            tensors["query"],
            cache,
            metadata,
            tensors["output"],
            *(
                tensors[prefix + suffix]
                for suffix in ("_query", "_key", "_weights", "_z")
            ),
        )
        kwargs = {"kv_scales": (tensors.get("k_scale"), tensors.get("v_scale"))}
        if case["pd_type"] == "decode":
            kwargs.update(query_len=syn["query_len"], gate=tensors.get("gate"))
        return getattr(impl, "_forward_" + case["pd_type"])(*args, **kwargs)

    torch.cuda.synchronize()
    from vllm_sail.models.step5.ops.dsa_native import NativePrefill

    paged = metadata.step5_runtime[5] if metadata.step5_runtime else None
    route = (
        "decode"
        if case["pd_type"] == "decode"
        else (
            "native"
            if isinstance(paged, NativePrefill)
            else "paged"
            if paged is not None
            else "ragged"
        )
    )
    return SimpleNamespace(
        case=case,
        torch=torch,
        module=module,
        impl=impl,
        tensors=tensors,
        metadata=metadata,
        context=context,
        reset=reset,
        call=call,
        route=route,
        host_tables=tables,
        host_owners=owners,
        host_positions=positions,
        implementation=implementation,
        restore_scope={
            "summary_rows": summary_rows,
            "ring_rows": ring_rows,
            "ring_capacity_source": "explicit synthetic capacity, not recovered from the online KV spec",
        },
    )


def benchmark(torch, call, reset, groups=7, iterations=20):
    """Each Event pair wraps only the entry; reset is explicitly before the start marker."""
    reject_instrumentation()
    positive(groups, "groups")
    positive(iterations, "iterations")
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    for start, end in pairs:
        start.record()
        end.record()
    torch.cuda.synchronize()
    samples, host_samples, raw_samples = [], [], []
    for _ in range(groups):
        torch.cuda.synchronize()
        began = time.perf_counter()
        for start, end in pairs:
            reset()
            start.record()
            call()
            end.record()
        torch.cuda.synchronize()
        host_samples.append((time.perf_counter() - began) * 1e6 / iterations)
        durations = [start.elapsed_time(end) * 1000 for start, end in pairs]
        raw_samples.append(durations)
        samples.append(sum(durations) / iterations)
    median = peak(statistics.median(samples), "Event duration")
    return {
        "median_us": median,
        "min_us": min(samples),
        "max_us": max(samples),
        "range_pct": (max(samples) - min(samples)) / median * 100,
        "event_group_us": samples,
        "raw_event_us": raw_samples,
        "host_us_including_reset_and_events": host_samples,
        "groups": groups,
        "iterations": iterations,
        "timing_scope": "full eager entry without profiling",
    }


def measure_attention(torch, call, reset=lambda: None, groups=5, iterations=5):
    """Event and synchronized wall times are reported separately; per-group arithmetic means, then median; reset stays outside timing."""
    reject_instrumentation()
    positive(groups, "groups")
    positive(iterations, "iterations")
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    raw_device, raw_wall = [], []
    for _ in range(groups):
        durations, wall = [], []
        for start, end in events:
            reset()
            torch.cuda.synchronize()
            began = time.perf_counter()
            start.record()
            call()
            end.record()
            end.synchronize()
            wall.append(peak((time.perf_counter() - began) * 1e6, "wall duration"))
            durations.append(peak(start.elapsed_time(end) * 1000, "Event duration"))
        raw_device.append(durations)
        raw_wall.append(wall)
    device = [statistics.mean(values) for values in raw_device]
    wall = [statistics.mean(values) for values in raw_wall]
    median_device, median_wall = statistics.median(device), statistics.median(wall)
    return {
        "median_device_us": median_device,
        "median_sync_wall_us": median_wall,
        "device_group_us": device,
        "sync_wall_group_us": wall,
        "raw_device_us": raw_device,
        "raw_sync_wall_us": raw_wall,
        "group_statistic": "arithmetic_mean",
        "aggregate_statistic": "median",
        "range_pct": (max(device) - min(device)) / median_device * 100,
        "sync_wall_range_pct": (max(wall) - min(wall)) / median_wall * 100,
        "groups": groups,
        "iterations": iterations,
        "reset_oracle_jit_included": False,
        "wall_includes_events_and_sync": True,
    }


def attention_eligibility(
    consumer_rounds, full_rounds, name, *, reference="flashinfer"
):
    """Both orders must pass both gates; produces advice only, never modifies the production allowlist."""
    if len(consumer_rounds) != 2 or len(full_rounds) != 2:
        raise ValueError("must include both forward and reverse orders")
    for rounds in (consumer_rounds, full_rounds):
        orders = [r["order"] for r in rounds]
        if (
            orders[1] != list(reversed(orders[0]))
            or len(set(orders[0])) != len(orders[0])
            or reference not in orders[0]
            or name not in orders[0]
        ):
            raise ValueError(
                "must be forward/reverse orders containing both reference and candidate"
            )
    if consumer_rounds[0]["order"] != full_rounds[0]["order"]:
        raise ValueError("consumer and full must use the same order")
    speedups, ratios = [], []
    for consumer, full in zip(consumer_rounds, full_rounds, strict=False):
        c = consumer["routes"]
        f = full["routes"]
        speedups.append(
            peak(c[reference]["median_sync_wall_us"], "reference")
            / peak(c[name]["median_sync_wall_us"], name)
        )
        ratios.append(
            peak(f[name]["median_sync_wall_us"], name)
            / peak(f[reference]["median_sync_wall_us"], "reference")
        )
    return {
        "eligible": min(speedups) >= 1.05 and max(ratios) <= 1.05,
        "consumer_wall_speedups": speedups,
        "full_wall_ratios": ratios,
        "consumer_wall_latency_reduction_pct": [(1 - 1 / s) * 100 for s in speedups],
        "full_wall_latency_reduction_pct": [(1 - r) * 100 for r in ratios],
        "scope": "experimental advice usable only after numerical acceptance; not injected into the production allowlist",
    }


def attention_request_spans(starts, begin, end):
    """Pure host-side span splitting; imports no production native or device dependencies."""
    if (
        not starts
        or starts[0] != 0
        or any(b < a for a, b in zip(starts, starts[1:], strict=False))
        or not 0 <= begin <= end <= starts[-1]
    ):
        raise ValueError("invalid request/chunk boundaries")
    for req, (a, b) in enumerate(zip(starts, starts[1:], strict=False)):
        left, right = max(a, begin), min(b, end)
        if left < right:
            yield req, left, right


def query_union_diagnostics(
    rows, positions, starts, chunks, table_rows, block_size, pages
):
    """Host-only diagnostics, splitting into 8-row groups per request/chunk; logical reuse does not imply measured HBM."""
    positive(block_size, "block_size")
    positive(pages, "pages")
    if len(rows) != len(positions) or not starts or starts[-1] != len(rows):
        raise ValueError("row count inconsistent with request boundaries")
    cursor = 0
    for begin, end in chunks:
        if begin != cursor or end < begin:
            raise ValueError("chunks must cover all rows sequentially and completely")
        cursor = end
    if cursor != len(rows):
        raise ValueError("incomplete chunk coverage")
    groups = []
    for chunk_id, (begin, end) in enumerate(chunks):
        for req, left, right in attention_request_spans(starts, begin, end):
            for first in range(left, right, 8):
                last = min(first + 8, right)
                sets = []
                for row in range(first, last):
                    sets.append(
                        {
                            r
                            for r in rows[row]
                            if 0 <= r < 8192
                            and r * 8 <= positions[row]
                            and r * 8 // block_size < len(table_rows[req])
                            and 0 <= table_rows[req][r * 8 // block_size] < pages
                        }
                    )
                union = set().union(*sets)
                total = sum(map(len, sets))
                groups.append(
                    {
                        "request": req,
                        "chunk": chunk_id,
                        "start": first,
                        "end": last,
                        "full_q8_tile": last - first == 8,
                        "per_row_regions": list(map(len, sets)),
                        "union_regions": len(union),
                        "overlap_fraction": 1 - len(union) / total if total else 0,
                    }
                )
    return {
        "groups": groups,
        "full_q8_tiles": sum(g["full_q8_tile"] for g in groups),
        "tail_rows": sum(
            g["end"] - g["start"] for g in groups if not g["full_q8_tile"]
        ),
        "scope": "logical overlap of per-query effective region sets, not measured HBM traffic",
    }


def calibrate_bf16(torch):
    """Standalone GEMM calibration; must not be called the official theoretical peak."""
    size = 4096
    a = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
    b, out = torch.randn_like(a), torch.empty_like(a)

    def call():
        return torch.mm(a, b, out=out)

    for _ in range(30):
        call()
    result = benchmark(torch, call, lambda: None, groups=5, iterations=20)
    tflops = 2 * size**3 / (result["min_us"] * 1e-6) / 1e12
    return {
        "bf16_tflops": tflops,
        "source": "mean of the fastest group of this standalone 4096x4096 BF16 GEMM; measured calibration, not theoretical peak",
        "calibration": result,
    }


@contextmanager
def observe(data, records, profiling=False):
    """Only the diagnostic pass installs process-local delegating wrappers; original attributes are restored exactly on exit."""
    torch = data.torch
    from vllm.models.step5.nvidia import flashinfer_sparse

    from vllm_sail.models.step5.ops.dsa_native import NativePrefill

    changes = []
    data.selection_metadata = {}
    if data.case["pd_type"] == "prefill":
        runtime = data.metadata.step5_runtime
        data.selection_metadata.update(owners=runtime[0], positions=runtime[1])
    if (
        data.case["pd_type"] == "prefill"
        and data.metadata.step5_runtime[2] <= data.impl.sparse_topk
    ):
        for meta in data.metadata.step5_runtime[4]:
            records.append(meta.selected)

    def hook(owner, name, stage, capture=None):
        original = getattr(owner, name)
        had = name in vars(owner)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if profiling:
                with torch.profiler.record_function("step5_stage:" + stage):
                    result = original(*args, **kwargs)
            else:
                result = original(*args, **kwargs)
            if capture is not None:
                capture(args, kwargs, result)
            return result

        changes.append((owner, name, original, had))
        setattr(owner, name, wrapper)

    def selected(args, kwargs, result):
        records.append(result)

    def decoded(args, kwargs, result):
        decode = result[1]
        data.selection_metadata.update(positions=decode.positions, active=decode.active)

    for name in ("update_prefill", "update_decode", "update_verify"):
        hook(data.impl.summary_cache, name, "summary")
    for name, stage in (
        ("pack_summaries", "pack"),
        ("pack_decode_summaries", "pack"),
        ("logits", "score"),
        ("decode_logits", "decode_score_pipeline"),
    ):
        hook(
            data.impl.adapter, name, stage, decoded if name == "decode_logits" else None
        )
    hook(data.impl.adapter, "select_regions", "select", selected)
    hook(data.impl.adapter, "run_attention", "attention_pipeline")
    hook(flashinfer_sparse, "gather_prefill_kv", "staging")
    hook(data.impl.adapter._attention, "plan", "host_attention_plan")
    hook(data.impl.adapter._attention, "run", "attention")
    hook(
        flashinfer_sparse,
        "sparse_decode_attention",
        "attention_fused_gate" if data.tensors.get("gate") is not None else "attention",
    )
    hook(data.module, "stage_prefill_pages", "staging")
    paged = data.metadata.step5_runtime[5] if data.metadata.step5_runtime else None
    if paged is not None:
        native = isinstance(paged, NativePrefill)
        chunk_attention = native and not paged.batch_attention
        hook(paged, "prepare_chunk", "attention" if chunk_attention else "compaction")
        hook(paged, "finish", "host_release" if chunk_attention else "attention")
    try:
        yield
    finally:
        for owner, name, original, had in reversed(changes):
            if had:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def capture_selection(data, profiling=False, trace_path=None):
    from vllm.forward_context import override_forward_context

    records = []
    data.reset()
    data.torch.cuda.synchronize()
    with override_forward_context(data.context):
        with observe(data, records, profiling):
            if profiling:
                with data.torch.profiler.profile(
                    activities=[
                        data.torch.profiler.ProfilerActivity.CPU,
                        data.torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    data.call()
                    data.torch.cuda.synchronize()
                prof.export_chrome_trace(str(trace_path))
            else:
                data.call()
                data.torch.cuda.synchronize()
    if not records:
        raise RuntimeError(
            "no real selection captured; refusing to estimate sparse FLOPs"
        )
    rows = [row for tensor in records for row in tensor.cpu().tolist()]
    if len(rows) != data.case["geometry"]["rows"]:
        raise RuntimeError("selection row count inconsistent with the entry query rows")
    digest = hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()
    observed = {
        name: tensor.cpu().tolist() for name, tensor in data.selection_metadata.items()
    }
    geo = data.case["geometry"]
    if data.case.get("entry") == "forward":
        for name in ("owners", "positions"):
            observed[name] *= geo["groups"]
    if "active" not in observed:
        observed["active"] = [True] * geo["rows"]
    if "owners" not in observed:
        observed["owners"] = [
            data.host_owners[row // geo["groups"]] for row in range(geo["rows"])
        ]
    if any(len(values) != geo["rows"] for values in observed.values()):
        raise RuntimeError(
            "actual positions/active/owners inconsistent with the query row count"
        )
    data.observed_selection = observed
    work = summarize_selection(data, rows, observed)
    return rows, {**work, "selection_sha256": digest, "observed_metadata": observed}


def summarize_selection(data, rows, observed=None):
    geo, syn = data.case["geometry"], data.case["synthetic"]
    counts = []
    for row, regions in enumerate(rows):
        token = (
            row % geo["tokens"]
            if data.case.get("entry") == "forward"
            else row // geo["groups"]
            if data.case["pd_type"] == "decode"
            else row
        )
        owner = observed["owners"][row] if observed else data.host_owners[token]
        pos = observed["positions"][row] if observed else data.host_positions[token]
        active = observed["active"][row] if observed else True
        counts.append(
            effective_tokens(
                regions,
                pos,
                syn["seq_lens"][owner],
                data.host_tables[owner],
                geo["block_size"],
                geo["pages"],
                active,
            )
        )
    return {
        **attention_work(data.case, sum(counts), sum(value > 0 for value in counts)),
        "per_row_selected_tokens": counts,
    }


def correctness(data, sample_rows=8, *, repeat_atol=0):
    """CPU FP32 oracle; repeats are bitwise identical by default, atomic-reordering experiments may supply an explicit absolute error bound."""
    if not math.isfinite(repeat_atol) or repeat_atol < 0:
        raise ValueError("repeat_atol must be a finite non-negative number")
    torch = data.torch
    records, work = capture_selection(data)
    row_ids = check_selection_oracle(
        data, records, sample_rows, work["observed_metadata"]
    )
    baseline = data.tensors["output"].clone()
    repeated, repeated_work = capture_selection(data)
    if (
        work["selection_sha256"] != repeated_work["selection_sha256"]
        or records != repeated
    ):
        raise AssertionError("selection changed after state restoration")
    torch.testing.assert_close(
        data.tensors["output"], baseline, atol=repeat_atol, rtol=0
    )
    repeat_error = float(
        (data.tensors["output"].float() - baseline.float()).abs().max()
    )
    return {
        "status": "passed",
        "sample_rows": row_ids,
        "repeated_output_exact": repeat_error == 0,
        "repeated_max_abs_error": repeat_error,
        "repeat_atol": repeat_atol,
        **work,
    }


def check_selection_oracle(data, records, sample_rows=26, observed=None):
    """Standalone FP32 oracle reads the original per-row selection without relying on Q8 unions; fixed selections use this entry too."""
    torch, geo, syn = data.torch, data.case["geometry"], data.case["synthetic"]
    count = min(positive(sample_rows, "sample_rows"), geo["rows"])
    row_ids = sorted(
        {round(i * (geo["rows"] - 1) / max(count - 1, 1)) for i in range(count)}
    )
    for row in row_ids:
        outer = data.case.get("entry") == "forward"
        token = (
            row % geo["tokens"]
            if outer
            else row // geo["groups"]
            if data.case["pd_type"] == "decode"
            else row
        )
        group = (
            row // geo["tokens"]
            if outer
            else row % geo["groups"]
            if data.case["pd_type"] == "decode"
            else 0
        )
        owner = observed["owners"][row] if observed else data.host_owners[token]
        pos = observed["positions"][row] if observed else data.host_positions[token]
        active = observed["active"][row] if observed else True
        tokens = sorted(
            {
                region * 8 + j
                for region in records[row]
                if active and region >= 0
                for j in range(8)
                if region * 8 + j <= pos and region * 8 + j < syn["seq_lens"][owner]
            }
        )
        tokens = [
            t
            for t in tokens
            if t // geo["block_size"] < len(data.host_tables[owner])
            and 0 <= data.host_tables[owner][t // geo["block_size"]] < geo["pages"]
        ]
        head_slice = slice(group * geo["heads"], (group + 1) * geo["heads"])
        q = (
            (
                data.tensors["query"][token, head_slice]
                if outer
                else data.tensors["query"][row]
            )
            .float()
            .cpu()
        )
        if tokens:
            physical = [data.host_tables[owner][t // geo["block_size"]] for t in tokens]
            offsets = [t % geo["block_size"] for t in tokens]
            cache = data.tensors["kv_cache"]
            if cache.dtype in (torch.uint8, torch.float8_e4m3fn):
                packed = (
                    cache.view(torch.uint8)[physical, group, offsets]
                    .cpu()
                    .view(torch.float8_e4m3fn)
                    .float()
                )
                packed[:, :192] *= syn["k_scale"]
                packed[:, 192:] *= syn["v_scale"]
                packed = packed.to(torch.bfloat16).float()
            else:
                packed = cache[physical, group, offsets].float().cpu()
            expected = (q @ packed[:, :192].T * data.impl.scale).softmax(-1) @ packed[
                :, 192:
            ]
        else:
            expected = torch.zeros_like(q)
        gate = data.tensors.get("gate")
        if gate is not None:
            expected *= (
                gate[token, group * geo["heads"] : (group + 1) * geo["heads"]]
                .float()
                .cpu()
                .sigmoid()
                .unsqueeze(-1)
            )
        actual = (
            data.tensors["output"][token, head_slice]
            if outer
            else data.tensors["output"][row]
        )
        torch.testing.assert_close(
            actual.float().cpu(), expected, atol=0.025, rtol=0.025
        )
    return row_ids


def trace_summary(path):
    """Attributes GPU kernels via launch correlation and the innermost CPU phase, avoiding double counting across nested phases."""
    events = json.loads(Path(path).read_text(encoding="utf-8"))["traceEvents"]
    phases = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("name", "").startswith("step5_stage:")
    ]
    mapping = {}
    for event in events:
        if event.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is None:
            continue
        candidates = [
            p
            for p in phases
            if p.get("pid") == event.get("pid")
            and p.get("tid") == event.get("tid")
            and p["ts"] <= event["ts"] < p["ts"] + p.get("dur", 0)
        ]
        if candidates:
            mapping[correlation] = min(candidates, key=lambda e: e["dur"])[
                "name"
            ].split(":", 1)[1]
    kernels, stages = [], defaultdict(list)
    for event in events:
        if str(event.get("cat", "")).lower() != "kernel":
            continue
        stage = mapping.get(event.get("args", {}).get("correlation"), "unattributed")
        row = {
            "name": event["name"],
            "duration_us": float(event["dur"]),
            "stage": stage,
            "ts": event["ts"],
            "grid": event.get("args", {}).get("grid"),
            "block": event.get("args", {}).get("block"),
            "stream": event.get("args", {}).get("stream"),
        }
        kernels.append(row)
        stages[stage].append(row)
    if not kernels:
        raise RuntimeError(
            "torch profiler returned no kernel events; stage metrics unavailable"
        )
    total = sum(row["duration_us"] for row in kernels)
    return {
        "trace": str(path),
        "kernels": kernels,
        "kernel_count": len(kernels),
        "stage_calls": dict(Counter(p["name"].split(":", 1)[1] for p in phases)),
        "attention_coverage_us": interval_coverage(
            [r for r in kernels if r["stage"] in ("attention", "attention_fused_gate")]
        ),
        "stages": {
            name: {
                "kernel_us": sum(r["duration_us"] for r in rows),
                "kernel_count": len(rows),
            }
            for name, rows in stages.items()
        },
        "stage_coverage": 1
        - sum(r["duration_us"] for r in stages.get("unattributed", [])) / total,
        "kernel_active_us": total,
        "timing_scope": "standalone profiler pass, not the formal uninstrumented timing",
    }


def interval_coverage(rows):
    """Union of GPU kernel time intervals, avoiding double counting when multiple streams overlap."""
    end, covered = -math.inf, 0.0
    for row in sorted(rows, key=lambda r: r["ts"]):
        stop = row["ts"] + row["duration_us"]
        covered += max(0.0, stop - max(end, row["ts"]))
        end = max(end, stop)
    return covered
