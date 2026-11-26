# SPDX-License-Identifier: Apache-2.0
"""Step5 attention log protocol, synthetic workloads, and traffic model; standard library only."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

DTYPE_BYTES = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.float8_e4m3fn": 1,
    "torch.uint8": 1,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.bool": 1,
}
PREFIX = "[FW_SPARSE] op:_forward_"
OPTION_KEYS = {
    "context_len",
    "topk",
    "seed",
    "query_len",
    "query_lens",
    "query_start_loc",
    "seq_lens",
    "k_scale",
    "v_scale",
}


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def peak(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def parse_label(label):
    """Splits on field boundaries rather than commas inside tuples; never evaluates expressions from the log."""
    label = label.strip()
    if not label.startswith(PREFIX):
        raise ValueError("not a Step5 attention label")
    parts = re.split(r",(?=[A-Za-z_][\w.]*:)", label[len("[FW_SPARSE] ") :])
    raw = {}
    for part in parts:
        key, separator, value = part.partition(":")
        if not separator or not re.fullmatch(r"[A-Za-z_][\w.]*", key) or key in raw:
            raise ValueError("duplicate field or malformed field format")
        raw[key] = value
    kind = raw.get("pd_type")
    if (
        kind not in ("prefill", "decode")
        or raw.get("op") != f"_forward_{kind}"
        or raw.get("model") != "step5"
        or raw.get("fw") != "vllm"
        or "backend" not in raw
    ):
        raise ValueError("op/pd_type or fixed protocol fields inconsistent")
    fixed = {"op", "pd_type", "model", "fw", "backend"}
    tensors = {}
    for key, value in raw.items():
        if key in fixed or key.endswith("_dtype"):
            continue
        # Normalize dict-style metadata and plain object labels; keep the original fields in raw.
        name = key
        for prefix in ("metadata.", "attn_metadata."):
            if key.startswith(prefix):
                name = key[len(prefix) :]
        name = {
            "kv_scales.0": "k_scale",
            "kv_scales.1": "v_scale",
            "kv_scales.k_scale": "k_scale",
            "kv_scales.v_scale": "v_scale",
        }.get(name, name)
        if value == "None":
            if key + "_dtype" in raw:
                raise ValueError("None tensors must not carry a dtype")
            item = None
        else:
            if not re.fullmatch(r"\(\s*(?:\d+\s*(?:,\s*\d+\s*)*,?\s*)?\)", value):
                raise ValueError(
                    f"{key} is not shape/None; extra scalar values are rejected"
                )
            shape = [int(v) for v in re.findall(r"\d+", value)]
            dtype = raw.get(key + "_dtype")
            if dtype not in DTYPE_BYTES:
                raise ValueError(f"{key} missing or unsupported dtype: {dtype}")
            item = {"shape": shape, "dtype": dtype}
        if name in tensors:
            raise ValueError(f"duplicate field after normalization: {name}")
        tensors[name] = item
    for key in raw:
        if key.endswith("_dtype") and key[:-6] not in raw:
            raise ValueError(f"orphan dtype: {key}")
    case = {
        "schema": 1,
        "pd_type": kind,
        "backend_label": raw["backend"],
        "tensors": tensors,
        "raw_label": label,
    }
    geometry(case)
    canonical = json.dumps(
        {k: case[k] for k in ("pd_type", "backend_label", "tensors")},
        sort_keys=True,
        separators=(",", ":"),
    )
    case["id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return case


def geometry(case):
    tensors, kind = case["tensors"], case["pd_type"]

    def shape(name, rank=None, dtype=None):
        item = tensors.get(name)
        if item is None:
            raise ValueError(f"missing required tensor {name}")
        result = item["shape"]
        if rank is not None and len(result) != rank:
            raise ValueError(f"{name} has the wrong rank")
        if any(v <= 0 for v in result) or (dtype and item["dtype"] not in dtype):
            raise ValueError(f"{name} shape/dtype unsupported")
        return result

    outer = case.get("entry") == "forward"
    q = shape("query", 3, {"torch.bfloat16"})
    if (
        shape("output", 3, {"torch.bfloat16"}) != q
        or q[2] != 192
        or (not outer and q[1] not in (4, 8, 16))
    ):
        raise ValueError("query/output/head geometry unsupported")
    cache = shape(
        "kv_cache", 4, {"torch.bfloat16", "torch.float8_e4m3fn", "torch.uint8"}
    )
    if cache[3] != 384 or cache[2] % 8:
        raise ValueError("packed KV geometry unsupported")
    prefix = "dsa_proxy" if kind == "prefill" else "proxy"
    pq = shape(prefix + "_query", 4, {"torch.float8_e4m3fn", "torch.float16"})
    pk = shape(prefix + "_key", 3, {"torch.bfloat16"})
    pz = shape(prefix + "_z", 3, {"torch.bfloat16"})
    pw = shape(prefix + "_weights", 3, {"torch.float32"})
    n, groups = pk[0], cache[1] if outer else pk[1]
    if (
        pk != pz
        or pk[2] != 256
        or (outer and pk[1] != 1)
        or pq != [n, groups, pq[2], 256]
        or pw != pq[:3]
        or pq[2] not in ((4, 8, 16) if outer else (4, 8))
    ):
        raise ValueError("proxy geometry inconsistent")
    if outer:
        if (
            kind != "prefill"
            or q[0] != n
            or q[1] % groups
            or q[1] // groups not in (4, 8, 16)
        ):
            raise ValueError("outer prefill token/group geometry inconsistent")
    elif kind == "prefill":
        if groups != 1 or cache[1] != 1 or q[0] != n:
            raise ValueError(
                "prefill entry must be a single KV-group slice; unrecorded group strides cannot be recovered"
            )
    elif cache[1] != groups or q[0] != n * groups:
        raise ValueError("folded decode token/group geometry inconsistent")
    table = shape("block_table", 2, {"torch.int32"})
    reqs = table[0]
    if (
        shape("seq_lens", 1, {"torch.int32"}) != [reqs]
        or shape("query_start_loc", 1, {"torch.int32"}) != [reqs + 1]
        or shape("slot_mapping", 1, {"torch.int64"}) != [n]
    ):
        raise ValueError("request metadata geometry inconsistent")
    if tensors.get("gate") is not None:
        if kind != "decode" or shape("gate", 2, {"torch.bfloat16"}) != [
            n,
            q[1] * groups,
        ]:
            raise ValueError("gate geometry inconsistent")
    quantized = tensors["kv_cache"]["dtype"] != "torch.bfloat16"
    for name in ("k_scale", "v_scale"):
        if quantized and tensors.get(name) is None:
            raise ValueError(
                "quantized KV lacks scale shape/dtype; re-export a valid NVTX log"
            )
        if tensors.get(name) is not None:
            item = tensors[name]
            if math.prod(item["shape"]) != 1 or item["dtype"] != "torch.float32":
                raise ValueError("KV scale must be a single-element FP32")
    return {
        "tokens": n,
        "rows": n * groups if outer else q[0],
        "groups": groups,
        "heads": q[1] // groups if outer else q[1],
        "requests": reqs,
        "block_size": cache[2],
        "pages": cache[0],
        "table_width": table[1],
        "indexer_heads": pq[2],
    }


def load_nvtx(path, unique=False, case_id=None):
    path = Path(path).resolve()
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    cases, seen = [], set()
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if PREFIX not in line:
            continue
        try:
            case = parse_label(line[line.index(PREFIX) :])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if case_id and case["id"] != case_id:
            continue
        if unique and case["id"] in seen:
            continue
        seen.add(case["id"])
        cases.append(
            {
                **case,
                "source": str(path),
                "source_sha256": digest,
                "line": line_number,
                "observation_id": f"{case['id']}_line{line_number}",
            }
        )
    if not cases:
        raise ValueError("no matching attention case")
    return cases


def resolve_case(case, options=None, overrides=None):
    options, overrides = dict(options or {}), dict(overrides or {})
    for supplied in (options, overrides):
        if supplied.keys() - OPTION_KEYS:
            raise ValueError(f"unknown override: {supplied.keys() - OPTION_KEYS}")
    values = {
        "context_len": 8192,
        "topk": 512,
        "seed": 128,
        "k_scale": 1.0,
        "v_scale": 1.0,
        **options,
        **overrides,
    }
    geo = geometry(case)
    n, reqs = geo["tokens"], geo["requests"]
    positive(values["context_len"], "context_len")
    positive(values["topk"], "topk")
    if values["topk"] > 1024 or type(values["seed"]) is not int or values["seed"] < 0:
        raise ValueError("top-k must be <=1024 and seed must be a non-negative integer")
    starts = values.get("query_start_loc")
    lengths = values.get("query_lens")
    if starts is not None:
        if (
            not isinstance(starts, list)
            or len(starts) != reqs + 1
            or any(type(x) is not int for x in starts)
            or starts[0] != 0
            or starts[-1] != n
        ):
            raise ValueError("query_start_loc length/boundaries invalid")
        derived = [b - a for a, b in zip(starts, starts[1:], strict=False)]
        if lengths is not None and lengths != derived:
            raise ValueError("query_lens inconsistent with starts")
        lengths = derived
    if lengths is None:
        if case["pd_type"] == "decode":
            if n % reqs:
                raise ValueError(
                    "decode tokens do not divide evenly per request; supply full metadata"
                )
            lengths = [n // reqs] * reqs
        else:
            lengths = [n // reqs + int(i < n % reqs) for i in range(reqs)]
    if not isinstance(lengths, list) or len(lengths) != reqs or sum(lengths) != n:
        raise ValueError("query_lens length or total token count invalid")
    for value in lengths:
        positive(value, "query_lens")
    query_len = max(lengths)
    if case["pd_type"] == "decode" and (min(lengths) != query_len or query_len > 4):
        raise ValueError("decode supports uniform query_len=1..4")
    if "query_len" in values and values["query_len"] != query_len:
        raise ValueError("query_len inconsistent with the token/request allocation")
    seq_lens = values.get("seq_lens", [max(values["context_len"], q) for q in lengths])
    if not isinstance(seq_lens, list) or len(seq_lens) != reqs:
        raise ValueError("seq_lens length invalid")
    for seq, q in zip(seq_lens, lengths, strict=False):
        positive(seq, "seq_lens")
        if seq < q or seq > geo["table_width"] * geo["block_size"] or seq >= 2**31:
            raise ValueError(
                "seq_lens smaller than query or beyond table/int32 capacity"
            )
    page_counts = [
        (seq + geo["block_size"] - 1) // geo["block_size"] for seq in seq_lens
    ]
    if sum(page_counts) > geo["pages"]:
        raise ValueError(
            "real cache pages insufficient for a mapping without cross-request sharing"
        )
    starts = [0]
    for q in lengths:
        starts.append(starts[-1] + q)
    for name in ("k_scale", "v_scale"):
        values[name] = peak(values[name], name)
    result = {
        **case,
        "geometry": geo,
        "synthetic": {
            **values,
            "query_len": query_len,
            "query_lens": lengths,
            "query_start_loc": starts,
            "seq_lens": seq_lens,
            "page_counts": page_counts,
            "num_speculative_tokens": query_len - 1
            if case["pd_type"] == "decode"
            else 0,
        },
        "input_values": "shape-matched synthetic; contiguous strides assumed, not recorded",
    }
    result["workload_id"] = hashlib.sha256(
        json.dumps(
            {"id": case["id"], "synthetic": result["synthetic"]}, sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    result["memory_estimate"] = memory_estimate(result)
    return result


def memory_estimate(case):
    geo, syn = case["geometry"], case["synthetic"]
    tensor_bytes = sum(
        math.prod(t["shape"]) * DTYPE_BYTES[t["dtype"]]
        for t in case["tensors"].values()
        if t is not None
    )
    pq = case["tensors"][
        "dsa_proxy_query" if case["pd_type"] == "prefill" else "proxy_query"
    ]
    summary = geo["pages"] * (geo["block_size"] // 8) * 256 * DTYPE_BYTES[pq["dtype"]]
    capacity = max(
        8, 7 + syn["num_speculative_tokens"], geo["block_size"] * geo["groups"]
    )
    ring = (geo["requests"] + 1) * capacity * 1024
    # Conservatively counts two live summary/reset copies, KV staging, logits/splits, and workspace.
    live_summary = (
        sum(syn["page_counts"])
        * geo["block_size"]
        // 8
        * 256
        * DTYPE_BYTES[pq["dtype"]]
    )
    scratch = (768 << 20) + geo["rows"] * (syn["topk"] + 1) * 16
    return {
        "tensor_bytes": tensor_bytes,
        "summary_bytes": summary,
        "ring_bytes": ring,
        "scratch_budget_bytes": scratch,
        "estimated_peak_bytes": tensor_bytes
        + summary
        + 2 * live_summary
        + 2 * ring
        + scratch,
        "note": "conservative budget, not allocator-measured; log shapes are never shrunk",
    }


def mutable_state_rows(case, tables, capacity):
    """Write set of the synthetic fixed step; used only for restoration outside timing, never reads device state."""
    geo, syn = case["geometry"], case["synthetic"]
    positive(capacity, "ring capacity")
    summaries, ring_rows = set(), set()
    for request, (seq, q) in enumerate(
        zip(syn["seq_lens"], syn["query_lens"], strict=False)
    ):
        for position in range(seq - q, seq):
            if position % 8 == 7:
                page = tables[request][position // geo["block_size"]]
                summaries.add(
                    page * (geo["block_size"] // 8) + position % geo["block_size"] // 8
                )
            if case["pd_type"] == "prefill":
                keep = position >= seq - capacity
            else:
                keep = syn["query_len"] > 1 or position % 8 != 7
            if keep:
                ring_rows.add((request + 1) * capacity + position % capacity)
    return sorted(summaries), sorted(ring_rows)


def effective_tokens(regions, position, seq_len, table, block_size, pages, active=True):
    """Region-level clipped counting without expanding eight-token lists; invalid pages and duplicate regions contribute no useful FLOPs."""
    if not active or position < 0:
        return 0
    total, limit = 0, min(position + 1, seq_len)
    for region in set(regions):
        if region < 0:
            continue
        begin, end = region * 8, min(region * 8 + 8, limit)
        if begin >= end:
            continue
        page = begin // block_size
        if page < len(table) and 0 <= table[page] < pages:
            total += end - begin
    return total


def attention_work(case, selected_tokens, active_rows=None):
    geo, tensors = case["geometry"], case["tensors"]
    rows, heads = geo["rows"], geo["heads"]
    active_rows = rows if active_rows is None else active_rows
    if (
        type(selected_tokens) is not int
        or selected_tokens < 0
        or type(active_rows) is not int
        or not 0 <= active_rows <= rows
    ):
        raise ValueError("invalid effective token/row counts")
    query = active_rows * heads * 192 * DTYPE_BYTES[tensors["query"]["dtype"]]
    kv = selected_tokens * 384 * DTYPE_BYTES[tensors["kv_cache"]["dtype"]]
    output = active_rows * heads * 192 * DTYPE_BYTES[tensors["output"]["dtype"]]
    gate = (
        active_rows * heads * DTYPE_BYTES[tensors["gate"]["dtype"]]
        if tensors.get("gate")
        else 0
    )
    metadata = {
        name: math.prod(item["shape"]) * DTYPE_BYTES[item["dtype"]]
        for name, item in tensors.items()
        if item is not None
        and name
        in (
            "block_table",
            "seq_lens",
            "slot_mapping",
            "query_start_loc",
            "k_scale",
            "v_scale",
        )
    }
    return {
        "selected_tokens": selected_tokens,
        "active_rows": active_rows,
        "attention_useful_flops": 2 * heads * 384 * selected_tokens,
        "logical_read_bytes": query + kv + gate,
        "logical_write_bytes": output,
        "logical_detail": {
            "query_bytes": query,
            "kv_bytes": kv,
            "gate_bytes": gate,
            "output_bytes": output,
            "auxiliary_input_bytes_once": metadata,
            "auxiliary_scope": "metadata/scales input capacity listed separately; not per-kernel measured traffic and not added to the main attention traffic",
        },
        "traffic_scope": "attention algorithm traffic, per-query MQA KV read; not measured HBM",
    }


def utilization(work, elapsed_us, bf16_tflops=None, hbm_gbps=None):
    seconds = peak(elapsed_us, "elapsed_us") * 1e-6
    result = {
        "case_attention_effective_mfu_pct": None,
        "logical_read_gbps": work["logical_read_bytes"] / seconds / 1e9,
        "logical_write_gbps": work["logical_write_bytes"] / seconds / 1e9,
        "logical_read_util_pct": None,
        "logical_write_util_pct": None,
    }
    if bf16_tflops is not None:
        result["case_attention_effective_mfu_pct"] = (
            work["attention_useful_flops"]
            / seconds
            / (peak(bf16_tflops, "BF16 peak") * 1e12)
            * 100
        )
    if hbm_gbps is not None:
        for direction in ("read", "write"):
            result[f"logical_{direction}_util_pct"] = (
                result[f"logical_{direction}_gbps"] / peak(hbm_gbps, "HBM peak") * 100
            )
    return result
