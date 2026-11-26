# SPDX-License-Identifier: Apache-2.0
"""Fixed-input sparse op arms, including tensor dumps; device imports are lazy."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from step5_sparse_cases import DTYPES, dump_output_names, storage_elements


class Unsupported(ValueError):
    """Admission failure, not a swallowed launch error."""


@dataclass
class Arm:
    backend: str
    impl: str
    config: dict
    call: object
    check: object
    work: dict
    contract: str


def initialize_runtime(extension_root=None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/PPU device available")
    name = torch.cuda.get_device_name(0).upper()
    if any(word in name for word in ("PPU", "ZW-", "ZW8", "M890")):
        from step5_sparse_support import require_runtime

        return require_runtime("flashinfer", extension_root)[0]
    if extension_root is not None:
        raise Unsupported("extension-root is only supported for PPU")
    return torch


def _vector(work, name, count, default, low=0, high=2147483647):
    value = work.get(name, default)
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(type(v) is not int or not low <= v <= high for v in value)
    ):
        raise Unsupported("Invalid synthetic " + name)
    return value


def _table(work, requests, columns, pages):
    value = work.get(
        "block_table",
        [[(i * columns + j) % pages for j in range(columns)] for i in range(requests)],
    )
    if (
        not isinstance(value, list)
        or len(value) != requests
        or any(
            not isinstance(row, list)
            or len(row) != columns
            or any(type(v) is not int or not -1 <= v < pages for v in row)
            for row in value
        )
    ):
        raise Unsupported(
            "block_table must be a bounded rectangular physical-page mapping"
        )
    return value


def resolve_topk_tensor(case, directory=None):
    """Resolve a plain dump filename without importing torch."""
    filename = case.get("topk_tensor", "")
    if not filename:
        return None
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise Unsupported("Invalid topk_tensor filename")
    candidates = (
        {Path(directory).resolve() / filename}
        if directory is not None
        else {
            Path(s["path"]).resolve().parent / filename for s in case.get("sources", [])
        }
    )
    existing = {p.resolve() for p in candidates if p.is_file()}
    if not existing:
        logging.getLogger(__name__).warning(
            "topk_tensor %s not found for case %s; falling back to synthetic inputs. "
            "Searched: %s",
            filename,
            case.get("id", "unknown"),
            ", ".join(str(p) for p in sorted(candidates)) or "no source directories",
        )
        return None
    if len(existing) != 1:
        raise Unsupported(
            f"topk_tensor {filename}: expected one existing file; use --topk-tensor-dir"
        )
    return existing.pop()


def fallback_topk_workload(case, work):
    """Keep the recorded case identity while switching only the input source."""
    work = dict(work)
    for key in ("topk_tensor_path", "topk_tensor_sha256"):
        work.pop(key, None)
    if work.get("policy") in (None, "tensor-dump"):
        work["policy"] = "capacity-causal"
    work.update(
        synthetic=True,
        topk_tensor_fallback={
            "filename": case["topk_tensor"],
            "reason": "file-not-found",
        },
    )
    return work


def load_topk_tensor(torch, case, path):
    """Load trusted CPU snapshots; reject stale or mismatched dump/label pairs."""
    try:
        with Path(path).open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
            stream.seek(0)
            record = torch.load(stream, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Unsupported(f"Cannot load topk_tensor {path}: {exc}") from exc
    kind = "TopK" if case["op"] == "topk" else "Attention"
    if not isinstance(record, dict) or record.get("schema") != 1:
        raise Unsupported(f"Unsupported {kind} dump schema")
    if record.get("op", "topk") != case["op"]:
        raise Unsupported(f"{kind} dump/label mismatch: op")
    if case["op"] == "attention" and record.get("parameters"):
        raise Unsupported(
            "Structured Attention prefill_chunks are saved but not replayable yet"
        )
    for name in ("impl", "backend", "scope", "pd_type"):
        if record.get(name) != case[name]:
            raise Unsupported(f"{kind} dump/label mismatch: " + name)
    inputs, metadata, scalars = (
        record.get("inputs"),
        record.get("metadata"),
        record.get("scalars"),
    )
    if not all(isinstance(v, dict) for v in (inputs, metadata, scalars)):
        raise Unsupported(f"{kind} dump requires inputs, metadata and scalars mappings")
    outputs = dump_output_names(case["op"], case["impl"])
    for name, item in case["tensors"].items():
        if item is None:
            if name in metadata or inputs.get(name) is not None:
                raise Unsupported("Unexpected dump tensor: " + name)
            continue
        meta = metadata.get(name)
        if not isinstance(meta, dict) or any(meta.get(k) != v for k, v in item.items()):
            raise Unsupported(f"{kind} dump layout mismatch: " + name)
        if name not in outputs and name not in inputs:
            raise Unsupported(f"Missing {kind} input tensor: " + name)
    for name, value in inputs.items():
        item = case["tensors"].get(name)
        if (
            name in outputs
            or item is None
            or not isinstance(value, torch.Tensor)
            or str(value.device) != "cpu"
            or list(value.shape) != item["shape"]
            or str(value.dtype) != item["dtype"]
        ):
            raise Unsupported(f"Invalid {kind} input tensor: " + name)
    for name, value in case["scalars"].items():
        if name != "capture" and scalars.get(name) != value:
            raise Unsupported(f"{kind} dump scalar mismatch: " + name)
    return inputs, digest


def _attention_tensor_workload(case, geo, inputs):
    rows, requests = geo["rows"], geo["requests"]
    table = inputs["block_table"].tolist()
    columns = case["tensors"]["block_table"]["shape"][1]
    table = _table({"block_table": table}, requests, columns, geo["pages"])
    positions = inputs["positions"].reshape(-1).tolist()
    active = (
        inputs["active"].reshape(-1).tolist() if "active" in inputs else [True] * rows
    )
    if len(active) != rows or any(type(v) is not bool for v in active):
        raise Unsupported("Invalid recorded active")
    if geo["pd_type"] == "decode":
        owners = [i // (geo["groups"] * geo["query_len"]) for i in range(rows)]
    else:
        owners = inputs["owners"].reshape(-1).tolist()
        _vector({"owners": owners}, "owners", rows, [], high=requests - 1)
    if len(positions) != rows or any(
        type(p) is not int or (live and not 0 <= p < columns * geo["block_size"])
        for p, live in zip(positions, active, strict=True)
    ):
        raise Unsupported("Invalid recorded positions")
    selected = inputs["indices"].tolist()
    if len(selected) != rows:
        raise Unsupported("Invalid recorded indices rows")
    for ids, pos, live in zip(selected, positions, active, strict=True):
        valid = [v for v in ids if v >= 0]
        if len(ids) != geo["topk"] or any(type(v) is not int for v in ids):
            raise Unsupported("Invalid recorded indices")
        if live and (
            any(not -1 <= v <= pos // 8 for v in ids)
            or len(valid) != len(set(valid))
            or pos // 8 not in valid
        ):
            raise Unsupported("Invalid recorded history/current/duplicate selection")
    lengths = [geo["query_len"]] * requests
    for owner, pos, live in zip(owners, positions, active, strict=True):
        if live:
            lengths[owner] = max(lengths[owner], pos + 1)
    work = dict(
        policy="tensor-dump",
        synthetic=False,
        block_table=table,
        positions=positions,
        owners=owners,
        active=active,
        indices=selected,
        seq_lens=lengths,
        scale=case["scalars"].get("scale"),
    )
    if case["tensors"]["kv_cache"]["dtype"] in ("torch.uint8", "torch.float8_e4m3fn"):
        for name in ("k_scale", "v_scale"):
            tensor = inputs.get(name)
            if tensor is None or tensor.numel() != 1:
                raise Unsupported("Recorded quantized cache requires scalar " + name)
            value = tensor.reshape(-1).tolist()[0]
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise Unsupported("Invalid recorded " + name)
            work[name] = value
    return work


def tensor_workload(case, geo, work, inputs):
    """Use recorded lengths and mappings, never synthetic content overrides."""
    if geo["kind"] == "attention":
        return {**work, **_attention_tensor_workload(case, geo, inputs)}
    work = {**work, "synthetic": False}
    rows, width = geo["rows"], geo["regions"]

    def vector(name, count, low, high):
        value = inputs[name].reshape(-1).tolist()
        if len(value) != count or any(
            type(v) is not int or not low <= v <= high for v in value
        ):
            raise Unsupported("Invalid recorded " + name)
        return value

    if geo.get("select_all"):
        work["current"] = vector("current", rows, -1, width)
        return work
    if geo.get("raw_selector"):
        positions = vector("positions", geo["queries"], -2147483648, 2147483647)
        block = case["scalars"].get("region_block", 8)
        if type(block) is not int or block <= 0:
            raise Unsupported("Invalid recorded region_block")
        work["current"] = [
            max(0, min(p // block, width - 1))
            for p in positions
            for _ in range(geo["heads"])
        ]
        return work
    if "current" in inputs:
        work["current"] = vector("current", rows, -1, width - 1)
        work["topk_lengths"] = (
            vector("topk_lengths", rows, 0, width)
            if "topk_lengths" in inputs
            else [max(0, v) for v in work["current"]]
        )
    else:
        if "topk_lengths" in inputs:
            if (
                case["impl"].startswith("vllm_sail.")
                and "deep_select_topk" in case["impl"]
            ):
                end = inputs["topk_lengths"]
                if end.numel() != rows:
                    raise Unsupported("Recorded DeepSelect end/rows mismatch")
                import torch

                # Match the SAIL helper's conversion without changing its input.
                ends = end.reshape(-1).to(torch.int32).clamp(0, width).tolist()
            else:
                ends = vector("topk_lengths", rows, 0, width)
        elif "lengths" in inputs:
            n = case["scalars"].get("next_n", 1)
            count = inputs["lengths"].numel()
            ends = vector("lengths", count, 0, width)
            if count != rows:
                if type(n) is not int or n <= 0 or count * n != rows:
                    raise Unsupported("Recorded lengths/next_n/rows mismatch")
                ends = [max(0, end - n + 1 + i) for end in ends for i in range(n)]
        else:
            ends = [width] * rows
        work["current"] = ends
        work["topk_lengths"] = ends
    if case["tensors"].get("pages") is not None:
        table = inputs["block_table"].tolist()
        work["owners"] = vector("owners", rows, 0, len(table) - 1)
        work["block_table"] = table
        work["indptr"] = vector(
            "indptr", rows + 1, 0, case["tensors"]["pages"]["shape"][0]
        )
        expected = [0]
        for current in work["current"]:
            expected.append(expected[-1] + max(0, min(current, geo["topk"]) + 1))
        if work["indptr"] != expected:
            raise Unsupported("Recorded fused page indptr/count mismatch")
        block = case["scalars"].get("block_size")
        if block not in (16, 32, 64, 128) or not case["scalars"].get("page_stride"):
            raise Unsupported("Invalid recorded fused page geometry")
        for owner, current in zip(work["owners"], work["current"], strict=True):
            needed = current // (block // 8) + 1
            if needed > len(table[owner]) or any(v < 0 for v in table[owner][:needed]):
                raise Unsupported("Recorded fused page table does not cover current")
    return work


def normalize_workload(case, geo, work):
    """Resolve every content-dependent field before importing torch."""
    work = dict(work)
    if case.get("topk_tensor") and not work.get("topk_tensor_fallback"):
        return {"policy": "tensor-dump", "synthetic": False}
    kind = geo["kind"]
    explicit = work["policy"] == "explicit"
    needed = {
        "topk": {"current"},
        "direct": {"seq_lens", "slots", "block_table"},
        "packed": {"lengths", "block_table"},
        "grouped": {"lengths", "owners", "spans"},
        "attention": {"seq_lens", "positions", "active", "block_table", "indices"},
    }[kind]
    if kind == "attention" and geo["pd_type"] == "prefill":
        needed.add("owners")
    if explicit and needed - work.keys():
        raise Unsupported(
            "Explicit policy requires: " + ", ".join(sorted(needed - work.keys()))
        )
    rows = geo["rows"]
    tensors = case["tensors"]
    if kind == "topk":
        width = geo["regions"]
        raw = bool(case.get("topk_tensor") and case["scope"] == "kernel")
        default_current = [max(0, width - 1 - i % 3) for i in range(rows)]
        if geo.get("raw_selector"):
            capacity = width
            if case["scalars"].get("packed_mode"):
                block = case["scalars"].get("region_block", 8)
                page = case["scalars"].get("page_size", 64)
                if (
                    type(block) is not int
                    or block <= 0
                    or type(page) is not int
                    or page < block
                    or page % block
                ):
                    raise Unsupported("Invalid selector page geometry")
                capacity = min(
                    width, tensors["block_table"]["shape"][-1] * (page // block)
                )
            default_current = [
                max(0, capacity - 1 - q % 3)
                for q in range(geo["queries"])
                for _ in range(geo["heads"])
            ]
        raw_lengths = tensors.get("lengths") if raw else None
        if raw_lengths:
            count = math.prod(raw_lengths["shape"])
            n = case["scalars"].get("next_n", 1)
            if count != rows and (type(n) is not int or n <= 0 or count * n != rows):
                raise Unsupported("Synthetic lengths/next_n/rows mismatch")
            defaults = (
                (work["current"] if count == rows else work["current"][n - 1 :: n])
                if "current" in work
                else [width] * count
            )
            work["lengths"] = _vector(work, "lengths", count, defaults, high=width)
            default_current = (
                work["lengths"]
                if count == rows
                else [
                    max(0, end - n + 1 + i) for end in work["lengths"] for i in range(n)
                ]
            )
        elif (
            raw
            and "deep_select_topk" in case["impl"]
            and not tensors.get("topk_lengths")
        ):
            default_current = [width] * rows
        elif geo.get("select_all"):
            default_current = [max(0, width - i % 3) for i in range(rows)]
        work["current"] = _vector(
            work,
            "current",
            rows,
            default_current,
            low=-1 if geo.get("select_all") else 0,
            high=width if raw or geo.get("select_all") else width - 1,
        )
        if raw_lengths and work["current"] != default_current:
            raise Unsupported("Synthetic current must match lengths/next_n expansion")
        if (
            raw
            and "deep_select_topk" in case["impl"]
            and not tensors.get("topk_lengths")
            and work["current"] != [width] * rows
        ):
            raise Unsupported("DeepSelect without end must use the full score width")
        if geo.get("raw_selector"):
            block = case["scalars"].get("region_block", 8)
            if type(block) is not int or block <= 0:
                raise Unsupported("Invalid selector region_block")
            defaults = [
                work["current"][q * geo["heads"]] * block for q in range(geo["queries"])
            ]
            work["positions"] = _vector(work, "positions", geo["queries"], defaults)
            expanded = [
                min(p // block, width - 1)
                for p in work["positions"]
                for _ in range(geo["heads"])
            ]
            if work["current"] != expanded:
                raise Unsupported("Synthetic selector current/positions mismatch")
            shape = tensors["block_table"]["shape"]
            if len(shape) not in (1, 2) or (
                len(shape) == 2 and shape[0] != geo["queries"]
            ):
                raise Unsupported("Unsupported selector table geometry")
            requests, columns = (1, shape[0]) if len(shape) == 1 else shape
            table_work = dict(work)
            if len(shape) == 1 and "block_table" in work:
                table_work["block_table"] = [work["block_table"]]
            table = _table(table_work, requests, columns, max(1, requests * columns))
            work["block_table"] = table[0] if len(shape) == 1 else table
        work["topk_lengths"] = _vector(
            work,
            "topk_lengths",
            rows,
            work["current"],
            low=-1 if geo.get("select_all") else 0,
            high=geo["regions"],
        )
        if work["topk_lengths"] != work["current"]:
            raise Unsupported(
                "Replay requires topk_lengths equal current for complete Step5 selection"
            )
        if tensors.get("pages") is not None:
            if geo["regions"] <= geo["topk"]:
                raise Unsupported("Fused pages require scored topk")
            if explicit and {"owners", "block_table"} - work.keys():
                raise Unsupported("Fused pages require explicit owners/block_table")
            block_size, page_stride = (
                case["scalars"].get("block_size"),
                case["scalars"].get("page_stride"),
            )
            table_meta = tensors.get("block_table")
            if (
                block_size not in (16, 32, 64, 128)
                or type(page_stride) is not int
                or page_stride <= 0
                or table_meta is None
            ):
                raise Unsupported("Missing fused page geometry")
            requests, columns = table_meta["shape"]
            work["owners"] = _vector(
                work,
                "owners",
                rows,
                [i % requests for i in range(rows)],
                high=requests - 1,
            )
            work["block_table"] = _table(
                work, requests, columns, max(1, requests * columns)
            )
            if any(v < 0 for row in work["block_table"] for v in row) or any(
                v // (block_size // 8) >= columns for v in work["current"]
            ):
                raise Unsupported("Fused page tables do not cover current regions")
            ptr = [0]
            for cur in work["current"]:
                ptr.append(ptr[-1] + min(cur, geo["topk"]) + 1)
            if tensors["pages"]["shape"] != [ptr[-1]]:
                raise Unsupported(
                    "Recorded fused page capacity differs from synthetic counts"
                )
            work["indptr"] = ptr
        if work.get("pattern", "random") not in ("random", "ties", "near-ties"):
            raise Unsupported("Unknown score pattern")
    elif kind == "direct":
        columns = tensors["block_table"]["shape"][1]
        pages = tensors["mean"]["shape"][0]
        capacity = min(geo["regions"], columns * geo["rpb"])
        if capacity < 1 or capacity * 8 < geo["query_len"]:
            raise Unsupported("Insufficient direct causal capacity")
        work["seq_lens"] = _vector(
            work,
            "seq_lens",
            geo["requests"],
            [capacity * 8] * geo["requests"],
            low=geo["query_len"],
            high=capacity * 8,
        )
        work["slots"] = _vector(
            work, "slots", geo["tokens"], [0] * geo["tokens"], low=-1
        )
        work["block_table"] = _table(work, geo["requests"], columns, pages)
    elif kind in ("packed", "grouped"):
        n = geo["next_n"]
        count = rows // n
        if rows % n:
            raise Unsupported("Unpaired packed rows")
        work["lengths"] = _vector(
            work, "lengths", count, [geo["regions"]] * count, high=geo["regions"]
        )
        pages = tensors["packed"]["shape"][0]
        columns = math.ceil(geo["regions"] / 32)
        if kind == "packed":
            shape = tensors.get("block_table")
            if (
                shape is None
                or shape["shape"][0] != count
                or shape["shape"][1] < columns
            ):
                raise Unsupported("Missing packed page table capacity")
            work["block_table"] = _table(work, count, shape["shape"][1], pages)
        else:
            page_count = case["scalars"].get("page_count")
            if (
                type(page_count) is not int
                or page_count < columns
                or pages % page_count
            ):
                raise Unsupported(
                    "Grouped replay needs page_count and owner-contiguous packed pages"
                )
            requests = pages // page_count
            work["owners"] = _vector(
                work,
                "owners",
                rows,
                [(i // n) % requests for i in range(rows)],
                high=requests - 1,
            )
            if any(
                work["owners"][i] != work["owners"][i // n * n] for i in range(rows)
            ):
                raise Unsupported("Paired grouped rows must share their owner")
            group_rows = case["scalars"].get("group_rows") or 64 // geo["heads"]
            if group_rows not in (4, 8, 16):
                raise Unsupported(
                    "Grouped score requires a supported production group_rows (4/8/16)"
                )
            spans = work.get(
                "spans",
                [
                    [i, min(group_rows, rows - i), work["owners"][i]]
                    for i in range(0, rows, group_rows)
                ],
            )
            covered = []
            if not isinstance(spans, list):
                raise Unsupported("Invalid grouped spans")
            for span in spans:
                if (
                    not isinstance(span, list)
                    or len(span) != 3
                    or any(type(v) is not int for v in span)
                    or not 0 <= span[0] < rows
                    or not 1 <= span[1] <= group_rows
                    or span[0] + span[1] > rows
                    or not 0 <= span[2] < requests
                    or any(
                        work["owners"][i] != span[2]
                        for i in range(span[0], span[0] + span[1])
                    )
                ):
                    raise Unsupported(
                        "Grouped spans must cover one owner within the configured tile"
                    )
                covered.extend(range(span[0], span[0] + span[1]))
            if (
                covered != list(range(rows))
                or len(spans) != tensors["spans"]["shape"][0]
            ):
                raise Unsupported(
                    "Grouped spans must cover each row exactly once and match the recorded shape"
                )
            work["spans"] = spans
            work["block_table"] = [
                [work["owners"][i * n] * page_count + j for j in range(columns)]
                for i in range(count)
            ]
        if any(v < 0 for row in work["block_table"] for v in row):
            raise Unsupported(
                "Packed scorers require valid physical pages (no masked -1 table reads)"
            )
        scales = work.get("summary_scale", 1.0)
        if type(scales) not in (int, float) or not math.isfinite(scales) or scales <= 0:
            raise Unsupported("summary_scale must be finite and positive")
        work["summary_scale"] = scales
    else:
        requests, groups = geo["requests"], geo["groups"]
        columns = tensors["block_table"]["shape"][1]
        capacity = columns * geo["block_size"]
        if capacity < geo["query_len"] or columns < 1:
            raise Unsupported("Insufficient attention page capacity")
        work["seq_lens"] = _vector(
            work,
            "seq_lens",
            requests,
            [capacity] * requests,
            low=geo["query_len"],
            high=capacity,
        )
        owners = (
            [i // (groups * geo["query_len"]) for i in range(rows)]
            if geo["pd_type"] == "decode"
            else [i % requests for i in range(rows)]
        )
        work["owners"] = _vector(work, "owners", rows, owners, high=requests - 1)
        if geo["pd_type"] == "decode" and owners != work["owners"]:
            raise Unsupported(
                "Decode owners must follow folded request/token/group rows"
            )
        positions = [
            work["seq_lens"][o]
            - 1
            - (geo["query_len"] - 1 - (i // groups) % geo["query_len"])
            if geo["pd_type"] == "decode"
            else work["seq_lens"][o] - 1
            for i, o in enumerate(work["owners"])
        ]
        work["positions"] = _vector(
            work, "positions", rows, positions, high=capacity - 1
        )
        if any(
            p >= work["seq_lens"][o]
            for p, o in zip(work["positions"], work["owners"], strict=False)
        ):
            raise Unsupported("Attention positions exceed the explicit sequence length")
        active = work.get("active", [True] * rows)
        if (
            not isinstance(active, list)
            or len(active) != rows
            or any(type(v) is not bool for v in active)
        ):
            raise Unsupported("active requires one boolean per row")
        if geo["pd_type"] == "prefill" and not all(active):
            raise Unsupported("Prefill has no inactive-row public contract")
        work["active"] = active
        work["block_table"] = _table(work, requests, columns, geo["pages"])
        selected = work.get(
            "indices",
            [
                list(range(max(0, p // 8 - geo["topk"] + 1), p // 8 + 1))
                for p in work["positions"]
            ],
        )
        if not isinstance(selected, list) or len(selected) != rows:
            raise Unsupported("indices must contain one fixed selection per row")
        records = []
        for row, pos in zip(selected, work["positions"], strict=False):
            if (
                not isinstance(row, list)
                or len(row) > geo["topk"]
                or any(type(v) is not int or not -1 <= v <= pos // 8 for v in row)
                or len([v for v in row if v >= 0]) != len(set(v for v in row if v >= 0))
                or pos // 8 not in row
            ):
                raise Unsupported("Invalid history/current/duplicate selected regions")
            records.append([-1] * (geo["topk"] - len(row)) + row)
        work["indices"] = records
        for key in ("k_scale", "v_scale"):
            if (
                tensors["kv_cache"]["dtype"] in ("torch.uint8", "torch.float8_e4m3fn")
                and key not in work
            ):
                raise Unsupported("Quantized cache requires explicit synthetic " + key)
    work["synthetic"] = True
    return work


class Views:
    """Budgeted exact recorded storage geometry; never silently contiguous."""

    def __init__(self, torch, case, budget, inputs=None):
        self.torch, self.case, self.budget = torch, case, budget
        self.inputs = inputs or {}
        self.allocated = 0

    def workspace(self, shape, dtype):
        required = math.prod(shape) * dtype.itemsize
        if self.allocated + required > self.budget:
            raise Unsupported("Derived workspace exceeds allocation budget")
        self.allocated += required
        return self.torch.empty(shape, dtype=dtype, device="cuda")

    def tensor(self, name, *, shape=None, dtype=None, values=None, random=False):
        torch = self.torch
        item = self.case["tensors"].get(name)
        if item is None:
            if shape is None or dtype is None:
                raise Unsupported("Missing recorded tensor " + name)
            stride, span = [], 1
            for size in reversed(shape):
                stride.insert(0, span)
                span *= size
            item = dict(shape=list(shape), stride=stride, offset=0, dtype=dtype)
        if shape is not None and list(shape) != item["shape"]:
            raise Unsupported("Recorded shape mismatch: " + name)
        if dtype is not None and dtype != item["dtype"]:
            raise Unsupported("Recorded dtype mismatch: " + name)
        elements = storage_elements(item)
        required = elements * DTYPES[item["dtype"]]
        if self.allocated + required > self.budget:
            raise Unsupported("Exact recorded views exceed allocation budget")
        self.allocated += required
        typ = getattr(torch, item["dtype"].removeprefix("torch."))
        storage = torch.empty(elements, dtype=typ, device="cuda")
        view = storage.as_strided(item["shape"], item["stride"], item["offset"])
        if name in self.inputs:
            view.copy_(self.inputs[name])
        elif values is not None:
            value = (
                values
                if isinstance(values, torch.Tensor)
                else torch.tensor(values, dtype=typ, device="cuda")
            )
            view.copy_(value.reshape(item["shape"]))
        elif random:
            value = (
                torch.randn(item["shape"], dtype=torch.float32, device="cuda") * 0.125
            )
            if typ == torch.uint8:
                value = value.to(torch.float8_e4m3fn).view(torch.uint8)
            view.copy_(value.to(typ))
        return view


def _original(call, target):
    from vllm_sail.patch.utils import PATCH_MARKER, original_of

    while target in getattr(call, PATCH_MARKER, {}):
        call = original_of(call, target)
    return call


def _entry(call):
    return (
        getattr(call, "__module__", type(call).__module__)
        + "."
        + getattr(call, "__qualname__", type(call).__qualname__)
    )


def _raw_selector(torch, case, geo, work, backend, views, args):
    from vllm_sail.models.step5.ops import selector_v2

    if backend != "selector_v2" or args.scope != "kernel":
        raise Unsupported("Raw selector dump requires selector_v2 kernel scope")
    scores = views.tensor("scores", random=True)
    if work.get("pattern") == "ties":
        scores.fill_(1.0)
    elif work.get("pattern") == "near-ties":
        scores.copy_(1 + torch.arange(geo["regions"], device="cuda").float() * 2**-23)
    positions = views.tensor("positions", values=work.get("positions"))
    table = views.tensor("block_table", values=work.get("block_table"))
    k, width = geo["topk"], geo["regions"]
    shape = [geo["queries"], geo["heads"], k + 1]
    candidates = views.tensor("indices", shape=shape, dtype="torch.int32")
    starts = views.tensor("starts", shape=shape, dtype="torch.int32")
    capacity = selector_v2.workspace_capacity(width, k)
    workspace = views.tensor("workspace", dtype="torch.int32")
    if workspace.shape[:2] != scores.shape[:2] or workspace.shape[2] < capacity:
        raise Unsupported("Recorded raw selector workspace is too small")
    block, page = (
        case["scalars"].get("region_block", 8),
        case["scalars"].get("page_size", 64),
    )
    if (
        type(block) is not int
        or block <= 0
        or type(page) is not int
        or page < block
        or page % block
    ):
        raise Unsupported("Invalid recorded selector page geometry")
    packed = bool(case["scalars"].get("packed_mode", 0))
    mapping = table.cpu().tolist()
    if table.ndim == 1:
        mapping = [mapping] * geo["queries"]
    if packed:
        for row, current in enumerate(work["current"]):
            needed = current // (page // block) + 1
            table_row = mapping[row // geo["heads"]]
            if len(table_row) < needed or any(p < 0 for p in table_row[:needed]):
                raise Unsupported(
                    "Packed selector table must cover all recorded live regions"
                )
    host = scores.cpu().reshape(geo["rows"], width).tolist()
    call = partial(
        selector_v2.raw_select,
        scores,
        positions,
        table,
        workspace,
        candidates,
        starts,
        region_block=block,
        page_size=page,
        topk=k,
        packed=packed,
        threads=case["scalars"].get("threads", 0),
    )

    def check():
        ids = candidates.cpu().reshape(geo["rows"], k + 1).tolist()
        offsets = starts.cpu().reshape(geo["rows"], k + 1).tolist()
        for row, (got, start, current) in enumerate(
            zip(ids, offsets, work["current"], strict=True)
        ):
            logical = [v // block for v in start[:-1] if v >= 0]
            wanted = sorted(range(current), key=lambda i: (-host[row][i], i))[:k]
            if (
                len(logical) != min(k, current)
                or len(set(logical)) != len(logical)
                or any(not 0 <= v < current for v in logical)
                or sorted(host[row][i] for i in logical)
                != sorted(host[row][i] for i in wanted)
                or start[-1] != current * block
            ):
                raise AssertionError("Raw selector cutoff/current contract failed")
            for index, offset in zip(got, start, strict=True):
                region = offset // block if offset >= 0 else -1
                expected = region
                if packed and region >= 0:
                    expected = mapping[row // geo["heads"]][
                        region // (page // block)
                    ] * (page // block) + region % (page // block)
                if index != expected or (offset >= 0 and offset % block):
                    raise AssertionError("Raw selector region/start mapping failed")
        return dict(status="passed", rows=geo["rows"], cutoff_ties=True, exact=False)

    return Arm(
        backend,
        _entry(selector_v2.raw_select),
        dict(scope=args.scope, topk=k),
        call,
        check,
        dict(
            bytes=scores.numel() * scores.element_size() + geo["rows"] * (k + 1) * 8,
            flops=0,
        ),
        "selector-raw-cutoff-current-starts",
    )


def _topk(torch, case, geo, work, backend, views, args):
    from vllm.models.step5.nvidia.ops import sparse_indexer as indexer

    from vllm_sail.models.step5.ops import selector_v2

    if geo.get("select_all"):
        current = views.tensor("current", dtype="torch.int64", values=work["current"])
        holder = {}

        def call():
            holder["output"] = indexer.select_all_regions(current, geo["topk"])

        def check():
            expected = [
                [-1] * (geo["topk"] - max(0, c)) + list(range(c + 1))
                if c >= 0
                else [-1] * (geo["topk"] + 1)
                for c in work["current"]
            ]
            if holder["output"].cpu().tolist() != expected:
                raise AssertionError("Select-all padding/current contract failed")
            return dict(status="passed", rows=geo["rows"], exact=True)

        return Arm(
            backend,
            _entry(indexer.select_all_regions),
            dict(actual_backend="select_all"),
            call,
            check,
            dict(bytes=geo["rows"] * (geo["topk"] + 2) * 8, flops=0),
            "topk-complete-stable",
        )
    if geo.get("raw_selector"):
        return _raw_selector(torch, case, geo, work, backend, views, args)
    scores = views.tensor("scores", random=True)
    # Identical seed and data for every arm, including difficult cutoff ties.
    if work.get("pattern") == "ties":
        scores.fill_(1.0)
    elif work.get("pattern") == "near-ties":
        scores.copy_(
            1 + torch.arange(geo["regions"], device="cuda").float()[None, :] * 2**-23
        )
    current = views.tensor(
        "current", shape=[geo["rows"]], dtype="torch.int64", values=work["current"]
    )
    direct_deep_select = bool(
        case.get("topk_tensor") and "deep_select_topk" in case["impl"]
    )
    lengths = views.tensor(
        "topk_lengths",
        dtype=None
        if case.get("topk_tensor") and case["tensors"].get("topk_lengths")
        else "torch.int32",
        values=work["topk_lengths"],
        shape=case["tensors"].get("topk_lengths", {}).get("shape", [geo["rows"]])
        if case["tensors"].get("topk_lengths")
        else [geo["rows"]],
    )
    k, rows = geo["topk"], geo["rows"]
    if scores.dtype != torch.float32 and (
        backend != "deep_select" or args.scope != "kernel"
    ):
        raise Unsupported("Recorded BF16 scores require raw DeepSelect")
    if backend not in ("per_row", "deep_select", "selector_v2"):
        raise Unsupported("Unknown topk backend")
    if backend == "deep_select":
        from vllm.platforms import current_platform

        if current_platform.is_ppu():
            from vllm_sail.utils.deep_select import can_use_deep_select

            supported = can_use_deep_select(scores, k)
        else:
            from vllm.model_executor.layers.indexer_topk import is_deep_select_supported

            supported = is_deep_select_supported(scores, k)
        if not supported:
            raise Unsupported("DeepSelect device/library/layout admission failed")
    if backend == "selector_v2" and (
        not scores.is_contiguous() or not current.is_contiguous()
    ):
        raise Unsupported("selector_v2 requires contiguous scores/current")
    sort = bool(case["scalars"].get("sort", 1))
    holder = {}
    paged = None
    if case["tensors"].get("pages") is not None:
        if args.scope == "kernel":
            raise Unsupported("Raw topk does not produce fused prefill pages")
        owners = views.tensor("owners", values=work["owners"])
        table = views.tensor("block_table", values=work["block_table"])
        indptr = views.tensor("indptr", values=work["indptr"])
        pages = views.tensor("pages")
        paged = indexer.PagedPrefillMetadata(
            owners,
            table,
            pages,
            indptr,
            case["scalars"]["block_size"],
            case["scalars"]["page_stride"],
        )
    if args.scope == "kernel":
        if backend == "selector_v2" or (
            not case.get("topk_tensor") and geo["regions"] <= k
        ):
            raise Unsupported(
                "No identical raw output boundary for selector_v2/select-all"
            )
        run = None if direct_deep_select else indexer.get_indexer_topk(backend)
        stride = math.ceil(k / 8) * 8
        output_dtype = getattr(
            torch,
            (case["scalars"].get("output_index_dtype") or "torch.int32").removeprefix(
                "torch."
            ),
        )
        output = (
            views.tensor("indices")
            if case.get("topk_tensor") and case["tensors"].get("indices")
            else torch.empty((rows, stride), dtype=output_dtype, device="cuda")[:, :k]
        )
        if direct_deep_select:
            from vllm.platforms import current_platform

            if backend != "deep_select":
                raise Unsupported("Direct DeepSelect dump requires deep_select backend")
            if current_platform.is_ppu():
                from vllm_sail.utils.deep_select import deep_select_topk as run
            else:
                from vllm.model_executor.layers.indexer_topk import (
                    deep_select_topk as run,
                )
            ends = lengths if case["tensors"].get("topk_lengths") else None
            recorded_output = output if case["tensors"].get("indices") else None

            def call():
                holder["output"] = run(
                    scores,
                    k,
                    end=ends,
                    output_idx=recorded_output,
                    indices_dtype=output.dtype,
                )
        else:
            raw_lengths = (
                views.tensor("lengths", values=work.get("lengths"))
                if case["tensors"].get("lengths")
                else lengths
            )
            call = partial(
                run,
                scores,
                raw_lengths,
                case["scalars"].get("next_n", 1),
                output,
                k,
                case["scalars"].get("max_seq_len", geo["regions"]),
            )
        holder["output"] = output[:rows, :k]
        contract = "topk-raw-cutoff-ties"
    else:
        run = (
            selector_v2.select_regions
            if backend == "selector_v2"
            else indexer.select_regions
        )
        if backend != "selector_v2":
            run = _original(
                run, "vllm.models.step5.nvidia.ops.sparse_indexer.select_regions"
            )

        def call():
            holder["output"] = run(
                scores,
                current,
                k,
                "per_row" if backend == "selector_v2" else backend,
                topk_lengths=lengths,
                paged_prefill=paged,
                sort=sort,
            )

        contract = (
            "topk-complete-stable"
            if sort or backend != "deep_select"
            else "topk-complete-unsorted"
        )
    host = scores.cpu().tolist()
    expected = []
    for row, length in enumerate(work["current"]):
        chosen = sorted(range(length), key=lambda i: (-host[row][i], i))[:k]
        expected.append([-1] * (k - len(chosen)) + sorted(chosen) + [length])

    def check():
        actual = holder["output"].cpu().tolist()
        for row, got in enumerate(actual):
            if args.scope == "kernel":
                chosen = [i for i in got if 0 <= i < work["current"][row]]
                wanted = expected[row][:-1]
                if (
                    len(chosen) != len(set(chosen))
                    or any(i != -1 and not 0 <= i < work["current"][row] for i in got)
                    or len(chosen) != min(k, max(0, work["current"][row]))
                    or sorted(host[row][i] for i in chosen)
                    != sorted(host[row][i] for i in wanted if i >= 0)
                ):
                    raise AssertionError(
                        "Raw topk score set/validity/uniqueness failed"
                    )
            elif contract.endswith("unsorted"):
                if (
                    got[-1] != expected[row][-1]
                    or sorted(got[:-1]) != expected[row][:-1]
                ):
                    raise AssertionError("Unsorted complete topk contract failed")
            elif got != expected[row]:
                raise AssertionError(
                    "Complete topk padding/current/stable-tie contract failed"
                )
        if paged is not None:
            want = []
            for row, selected in enumerate(actual):
                owner = work["owners"][row]
                want.extend(
                    work["block_table"][owner][i // (paged.block_size // 8)]
                    * paged.page_stride
                    + i % (paged.block_size // 8)
                    for i in selected
                    if i >= 0
                )
            if paged.indices.cpu().tolist() != want:
                raise AssertionError("Fused page output mismatch")
        return dict(
            status="passed",
            exact=args.scope != "kernel" and not contract.endswith("unsorted"),
            rows=rows,
            cutoff_ties=True,
        )

    return Arm(
        backend,
        _entry(run),
        dict(topk=k, sort=sort, scope=args.scope),
        call,
        check,
        dict(bytes=rows * geo["regions"] * 4 + rows * (k + 1) * 8, flops=0),
        contract,
    )


def _direct(torch, case, geo, work, backend, views, args):
    from step5_sparse_support import direct_cpu_oracle, error_metrics

    from vllm_sail.models.step5.ops import soft_fp8, sparse_attention_opt

    q = views.tensor("query", random=True)
    w = views.tensor("weights", shape=q.shape[:3], dtype="torch.float32", random=True)
    mean = views.tensor("mean", random=True)
    table = views.tensor("block_table", dtype="torch.int32", values=work["block_table"])
    slots = views.tensor(
        "slots", shape=[geo["tokens"]], dtype="torch.int64", values=work["slots"]
    )
    seq = views.tensor("lengths", shape=[geo["requests"]], values=work["seq_lens"])
    if (
        q.stride(3) != 1
        or q.stride(2) != 256
        or w.stride(2) != 1
        or mean.stride(1) != 256
        or mean.stride(2) != 1
        or table.stride(1) != 1
        or q.dtype not in (torch.float16, torch.float8_e4m3fn)
        or mean.dtype != (torch.float16 if q.dtype == torch.float16 else torch.uint8)
    ):
        raise Unsupported("Direct launcher does not support this recorded dtype/stride")
    capacity, rows = geo["regions"], geo["rows"]
    output = views.tensor("scores", shape=[rows, capacity], dtype="torch.float32")
    if output.stride(1) != 1:
        raise Unsupported("Direct output requires contiguous columns")
    output.fill_(float("nan"))
    active = views.tensor("active", shape=[rows], dtype="torch.bool")
    pos = views.tensor("positions", shape=[rows], dtype="torch.int64")
    current = views.tensor("current", shape=[rows], dtype="torch.int64")
    lengths = views.tensor("context_lengths", shape=[rows], dtype="torch.int32")
    topk_lengths = views.tensor("topk_lengths", shape=[rows], dtype="torch.int32")
    if any(v.stride(0) != 1 for v in (active, pos, current, lengths, topk_lengths)):
        raise Unsupported("Direct metadata outputs require unit strides")
    host = []
    for row in range(rows):
        token, group = divmod(row, geo["groups"])
        owner, offset = divmod(token, geo["query_len"])
        live = work["slots"][token] >= 0
        position = work["seq_lens"][owner] - geo["query_len"] + offset if live else 0
        cur = position >> 3
        written = min(capacity, math.ceil(min(max(cur + 1, 0), capacity) / 128) * 128)
        host.append((live, position, cur, cur + 1, cur, owner, token, group, written))
    data = SimpleNamespace(
        torch=torch,
        query=q,
        mean=mean,
        weights=w,
        host=host,
        host_table=work["block_table"],
        spec=dict(rpb=geo["rpb"], heads=geo["heads"]),
    )
    oracle = direct_cpu_oracle(data, list(range(rows)))
    tile = 128
    from vllm.triton_utils import triton

    tiles = math.ceil(capacity / tile)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    default_workers = min(tiles, max(1, triton.cdiv(sms, max(rows, 1))))
    workers = case["scalars"].get("workers", work.get("workers", default_workers))
    if type(workers) is not int or not 1 <= workers <= tiles:
        raise Unsupported("Direct workers must be within the tile capacity")
    rawq = q if q.dtype == torch.float16 else q.view(torch.uint8)
    launch_args = (
        rawq,
        w,
        mean,
        table,
        slots,
        seq,
        output,
        active,
        pos,
        current,
        lengths,
        topk_lengths,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        w.stride(1),
        table.shape[1],
        table.stride(0),
        slots.stride(0),
        seq.stride(0),
        mean.shape[0],
        geo["rpb"],
        mean.stride(0),
        capacity,
        output.stride(0),
        geo["heads"],
        tile,
        workers,
        geo["groups"],
        geo["query_len"],
    )
    config = dict(tile=tile, workers=workers, num_warps=4)
    if backend == "direct_baseline":
        kernel, grid = soft_fp8._direct_scores, (workers, rows)
    elif backend in ("direct_resident", "direct_mtp"):
        if (
            tuple(torch.cuda.get_device_capability(0)) != (8, 9)
            or geo["heads"] != 8
            or q.dtype != torch.float8_e4m3fn
            or mean.dtype != torch.uint8
        ):
            raise Unsupported(
                "Direct resident requires the production SM89/H8/FP8 contract"
            )
        kernel = sparse_attention_opt._direct_scores_resident
        config.update(HOIST=True, SPLIT=True)
        if backend == "direct_mtp":
            if not soft_fp8.DirectScoresLaunch._mtp_eligible(launch_args):
                raise Unsupported(
                    "Direct MTP admission failed; no baseline substitution"
                )
            grid = (workers, rows)
            config["GUARD_MTP"] = True
        else:
            grid = (workers * rows,)
    else:
        raise Unsupported("Unknown direct backend")
    call = partial(
        kernel[grid],
        *launch_args,
        **{k: v for k, v in config.items() if k not in ("tile", "workers")},
    )

    def check():
        for tensor, index in zip(
            (active, pos, current, lengths, topk_lengths), range(5), strict=False
        ):
            torch.testing.assert_close(
                tensor.cpu(),
                torch.tensor([h[index] for h in host], dtype=tensor.dtype),
                atol=0,
                rtol=0,
            )
        metrics = error_metrics(
            torch,
            torch.cat([output[i, : h[-1]].cpu() for i, h in enumerate(host)]),
            torch.cat(oracle),
            atol=1e-4,
            rtol=1e-4,
        )
        for row, h in enumerate(host):
            if not torch.isnan(output[row, h[-1] :]).all():
                raise AssertionError("Direct scorer wrote unspecified tail")
        return dict(
            status="passed", mask="written-prefix/untouched-NaN-tail", **metrics
        )

    keys = set()
    valid_pairs = 0
    for h in host:
        for region in range(h[-1]):
            col = region // geo["rpb"]
            physical = work["block_table"][h[5]][col] if col < table.shape[1] else -1
            if 0 <= physical < mean.shape[0]:
                keys.add((physical, region % geo["rpb"]))
                valid_pairs += 1
    return Arm(
        backend,
        _entry(kernel),
        config,
        call,
        check,
        dict(
            bytes=q.numel() * q.element_size()
            + w.numel() * 4
            + len(keys) * 256 * mean.element_size()
            + sum(h[-1] for h in host) * 4,
            flops=valid_pairs * geo["heads"] * 256 * 2,
            accounting="Unique valid summaries read once; written prefix output; excludes backend workspace traffic.",
        ),
        "score-direct-prefix",
    )


def _packed(torch, case, geo, work, backend, views, args):
    from step5_sparse_support import direct_cpu_oracle, error_metrics
    from vllm.models.step5.nvidia.ops import sparse_indexer_scores as upstream

    from vllm_sail.models.step5.ops import sparse_attention_opt as opt

    q = views.tensor("query", dtype="torch.float8_e4m3fn", random=True)
    w = views.tensor("weights", dtype="torch.float32", random=True)
    packed = views.tensor("packed", dtype="torch.uint8")
    if not packed.is_contiguous() or packed.storage_offset() % 4 or q.stride(-1) != 1:
        raise Unsupported(
            "Packed page ABI needs contiguous pages and 4-byte aligned scale offset"
        )
    pages = packed.shape[0]
    keys = (torch.randn((pages, 32, 256), device="cuda") * 0.125).to(
        torch.float8_e4m3fn
    )
    flat = packed.view(pages, 32 * 260)
    flat[:, : 32 * 256].copy_(keys.view(torch.uint8).reshape(pages, -1))
    flat[:, 32 * 256 :].view(torch.float32).fill_(work["summary_scale"])
    rows, groups, n, heads = geo["rows"], geo["groups"], geo["next_n"], geo["heads"]
    lengths = views.tensor(
        "lengths", shape=[rows // n], dtype="torch.int32", values=work["lengths"]
    )
    table = (
        views.tensor("block_table", dtype="torch.int32", values=work["block_table"])
        if geo["kind"] == "packed"
        else torch.tensor(work["block_table"], dtype=torch.int32, device="cuda")
    )
    if geo["kind"] == "packed":
        if list(w.shape) != [rows, heads]:
            raise Unsupported("Packed weights must be [batch*next_n,heads]")
        query = q.reshape(rows, 1, heads, 256)
        weights = w.reshape(rows, 1, heads)
    else:
        if list(w.shape) != list(q.shape[:3]):
            raise Unsupported("Grouped weights geometry mismatch")
        query, weights = q, w
    host = [
        (True, 0, 0, 0, 0, i // n, i, g, geo["regions"])
        for i in range(rows)
        for g in range(groups)
    ]
    data = SimpleNamespace(
        torch=torch,
        query=query,
        weights=weights,
        mean=keys.view(torch.uint8),
        host=host,
        host_table=work["block_table"],
        spec=dict(rpb=32, heads=heads),
    )
    oracle = (
        torch.stack(direct_cpu_oracle(data, list(range(rows * groups))))
        .reshape(rows, groups, geo["regions"])
        .transpose(0, 1)
        * work["summary_scale"]
    )
    for row in range(rows):
        oracle[:, row, work["lengths"][row // n] :] = float("-inf")
    holder = {}
    config = dict(next_n=n, heads=heads, scope=args.scope)
    if backend == "compact_grouped":
        if geo["kind"] != "grouped":
            raise Unsupported(
                "Compact/grouped requires recorded spans; never fabricate missing request spans"
            )
        if args.scope == "kernel":
            raise Unsupported(
                "Grouped and packed kernels have different physical output boundaries"
            )
        spans = views.tensor("spans", dtype="torch.int32", values=work["spans"])
        run = opt.grouped_paged_logits

        def call():
            holder["output"] = run(
                q,
                w,
                packed,
                lengths,
                spans,
                geo["regions"],
                case["scalars"]["page_count"],
                heads,
                n,
                group_rows=case["scalars"].get("group_rows"),
                groups=groups,
            )

        config.update(
            tile=32,
            num_warps=2,
            grid_cap=32,
            group_rows=case["scalars"].get("group_rows") or 64 // heads,
            actual_backend="compact_grouped",
            actual_leaf=_entry(opt._grouped_scores_sm89),
        )
    elif backend in ("upstream_paged", "resident"):
        run = (
            _original(
                upstream.paged_mqa_logits_sm90,
                "vllm.models.step5.nvidia.ops.sparse_indexer_scores.paged_mqa_logits_sm90",
            )
            if backend == "upstream_paged"
            else opt.paged_mqa_logits_sm90
        )
        if backend == "resident" and tuple(torch.cuda.get_device_capability(0)) < (
            8,
            9,
        ):
            raise Unsupported("Resident scorer needs hardware E4M3")
        leaf = (
            upstream._paged_scores_sm90
            if backend == "upstream_paged"
            else opt._resident_scores_sm90
        )
        actual_backend = backend
        if backend == "upstream_paged":
            from vllm.platforms import current_platform

            from vllm_sail.models.step5.ops.soft_fp8 import ByteFP8Launch

            if isinstance(leaf, ByteFP8Launch) and current_platform.is_ppu():
                leaf, actual_backend = leaf.ppu, "software_paged"
        config.update(actual_backend=actual_backend, actual_leaf=_entry(leaf))
        # Group-major staging is a production cost for this baseline, inside each call.
        qstage = [
            views.workspace((rows // n, n, heads, 256), q.dtype) for _ in range(groups)
        ]
        wstage = [views.workspace((rows, heads), w.dtype) for _ in range(groups)]
        if args.scope == "kernel":
            if groups != 1 or geo["kind"] != "packed":
                raise Unsupported(
                    "Kernel-only packed replay needs the exact single-group packed boundary"
                )
            output = torch.empty(
                (rows, geo["regions"]), dtype=torch.float32, device="cuda"
            )
            tile = 32 if backend == "upstream_paged" else 128
            workers = opt._resident_workers(rows, math.ceil(geo["regions"] / tile))
            kernel = (
                upstream._paged_scores_sm90
                if backend == "upstream_paged"
                else opt._resident_scores_sm90
            )
            launch = (
                q,
                packed,
                w,
                lengths,
                table,
                output,
                *q.stride()[:3],
                *w.stride(),
                packed.stride(0),
                *table.stride(),
                lengths.stride(0),
                geo["regions"],
                n,
                heads,
                32,
                tile,
            )
            grid = (
                (rows, math.ceil(geo["regions"] / tile))
                if backend == "upstream_paged"
                else (rows * workers,)
            )
            if backend == "resident":
                launch += (workers,)
            call = partial(kernel[grid], *launch, num_warps=4)
            holder["output"] = output
            config.update(tile=tile, workers=workers, num_warps=4)
        else:

            def call():
                outputs = []
                for group in range(groups):
                    if geo["kind"] == "packed":
                        qq, ww = q, w
                    else:
                        qstage[group].copy_(
                            q[:, group].reshape(rows // n, n, heads, 256)
                        )
                        wstage[group].copy_(w[:, group])
                        qq, ww = qstage[group], wstage[group]
                    outputs.append(run(qq, packed, ww, table, lengths, geo["regions"]))
                holder["output"] = outputs

            tile = 32 if backend == "upstream_paged" else opt._DEFAULT_TILE
            config.update(tile=tile, num_warps=4)
            if backend == "resident":
                config["workers"] = opt._resident_workers(
                    rows, math.ceil(geo["regions"] / tile)
                )
    else:
        raise Unsupported("Unknown packed score backend")

    def check():
        out = holder["output"]
        if isinstance(out, list):
            actual = torch.stack(out).cpu()
        else:
            actual = out.cpu().reshape(groups, rows, geo["regions"])
        return dict(
            status="passed",
            **error_metrics(torch, actual, oracle, atol=1e-4, rtol=1e-4),
        )

    key_ids = {
        (work["block_table"][i][region // 32], region % 32)
        for i, length in enumerate(work["lengths"])
        for region in range(length)
    }
    return Arm(
        backend,
        _entry(run),
        config,
        call,
        check,
        dict(
            bytes=q.numel()
            + w.numel() * 4
            + len(key_ids) * 260
            + rows * groups * geo["regions"] * 4,
            flops=sum(work["lengths"]) * n * groups * heads * 256 * 2,
            accounting="Unique selected summaries/scales read once; full masked output; excludes staging/workspace traffic.",
        ),
        "score-packed-full-mask",
    )


def _attention(torch, case, geo, work, backend, views, args):
    from step5_sparse_support import check_selection_oracle

    from vllm_sail.models.step5.ops import dsa_native as native

    q = views.tensor("query", random=True)
    cache = views.tensor("kv_cache", random=True)
    table = views.tensor("block_table", dtype="torch.int32", values=work["block_table"])
    regions = views.tensor("indices", dtype="torch.int64", values=work["indices"])
    pos = views.tensor(
        "positions", shape=[geo["rows"]], dtype="torch.int64", values=work["positions"]
    )
    active = views.tensor(
        "active", shape=[geo["rows"]], dtype="torch.bool", values=work["active"]
    )
    output = views.tensor("output", shape=q.shape, dtype="torch.bfloat16")
    output.zero_()
    gate = views.tensor("gate", random=True) if case["tensors"].get("gate") else None
    scale = work.get("scale", case["scalars"].get("scale"))
    if scale is None:
        raise Unsupported("Missing explicit attention scale")
    k_scale = v_scale = None
    if cache.dtype in (torch.uint8, torch.float8_e4m3fn):

        def scale_tensor(name):
            item = case["tensors"].get(name)
            if item is not None and math.prod(item["shape"]) != 1:
                raise Unsupported("Only scalar quantized scales are replayable")
            tensor = views.tensor(
                name, shape=item["shape"] if item else [], dtype="torch.float32"
            )
            if name not in views.inputs:
                tensor.fill_(work[name])
            return tensor

        k_scale, v_scale = scale_tensor("k_scale"), scale_tensor("v_scale")
    config = dict(
        scale=scale, query_len=geo["query_len"], fixed_selection=True, scope=args.scope
    )
    observed = dict(
        owners=work["owners"], positions=work["positions"], active=work["active"]
    )
    data = SimpleNamespace(
        torch=torch,
        case=dict(geometry=geo, synthetic=work, pd_type=geo["pd_type"]),
        tensors=dict(query=q, kv_cache=cache, output=output, gate=gate),
        impl=SimpleNamespace(scale=scale),
        host_tables=work["block_table"],
    )
    if geo["pd_type"] == "decode":
        clear = bool(case["scalars"].get("clear_inactive", 1))
        if not clear and not all(work["active"]):
            raise Unsupported(
                "Inactive output preservation needs explicit initial output values"
            )
        if backend == "native":
            reason = native.decode_reason(
                q,
                cache,
                regions,
                table,
                pos,
                active,
                output,
                scale,
                geo["query_len"],
                gate,
            )
            if reason:
                raise Unsupported("Native decode: " + reason)
            run = native.compiled_op("decode")
            if run is None:
                raise Unsupported("Native decode compiled op unavailable")
            call = partial(
                run,
                q,
                cache,
                regions,
                table,
                pos,
                active,
                output,
                scale,
                geo["query_len"],
                clear,
                gate,
            )
        elif backend == "upstream_triton":
            from vllm.models.step5.nvidia.ops.sparse_decode_attention import (
                sparse_decode_attention,
                sparse_decode_splits,
            )

            run = _original(
                sparse_decode_attention,
                "vllm.models.step5.nvidia.ops.sparse_decode_attention.sparse_decode_attention",
            )
            if args.scope == "kernel":
                raise Unsupported(
                    "Upstream split/combine public path has no identical native single-kernel boundary"
                )
            if (
                q.stride(-1) != 1
                or cache.stride(-1) != 1
                or output.stride(-1) != 1
                or regions.stride(-1) != 1
            ):
                raise Unsupported("Unsupported upstream attention column strides")
            splits = case["scalars"].get("splits") or sparse_decode_splits(geo["rows"])
            config["splits"] = splits
            call = partial(
                run,
                q,
                cache,
                regions,
                table,
                pos,
                active,
                output,
                scale=scale,
                query_len=geo["query_len"],
                splits=splits,
                clear_inactive=clear,
                gate=gate,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        else:
            raise Unsupported("Unknown decode attention backend")
    else:
        if gate is not None:
            raise Unsupported("Prefill gate is outside the public attention boundary")
        owners = views.tensor(
            "owners", shape=[geo["rows"]], dtype="torch.int64", values=work["owners"]
        )
        if backend == "native":
            if "tiles" in views.inputs:
                tiles = views.tensor("tiles", dtype="torch.int32")
                query_tile = case["scalars"].get("query_tile")
                logical = case["scalars"].get("max_seq_len")
                if type(query_tile) is not int or type(logical) is not int:
                    raise Unsupported("Missing recorded native prefill tile parameters")
            else:
                tiles = torch.tensor(
                    [
                        [i for i in range(geo["rows"])],
                        [i + 1 for i in range(geo["rows"])],
                    ],
                    dtype=torch.int32,
                    device="cuda",
                )
                query_tile = 0
                logical = max(work["positions"]) // 8 + 1
            reason = native.prefill_reason(
                q, cache, regions, table, owners, pos, tiles, output, logical, scale
            )
            if reason:
                raise Unsupported("Native prefill: " + reason)
            run = native.compiled_op("prefill")
            if run is None:
                raise Unsupported("Native prefill compiled op unavailable")
            call = partial(
                run,
                q,
                cache,
                regions,
                table,
                owners,
                pos,
                tiles,
                output,
                logical,
                scale,
                query_tile,
            )
            config.update(query_tile=query_tile, max_seq_len=logical)
            if "tiles" not in views.inputs:
                config["tile_rows"] = 1
        elif backend == "flashinfer_paged":
            if args.scope == "kernel":
                raise Unsupported(
                    "Paged FlashInfer and native have different prepared input boundaries"
                )
            from vllm.models.step5.nvidia.ops.sparse_prefill_pages import (
                CompactPagedPrefill,
                fill_prefill_pages,
                kv_page_stride,
            )

            if (
                cache.dtype != torch.bfloat16
                or geo["heads"] not in (8, 16)
                or not all(v >= 0 for row in work["block_table"] for v in row)
            ):
                raise Unsupported(
                    "Paged FlashInfer requires dense BF16 single-group valid pages"
                )
            plan = CompactPagedPrefill(views.workspace([128 << 20], torch.uint8))
            plan.plan(
                q.shape,
                q.dtype,
                q.device,
                work["positions"],
                geo["topk"],
                scale,
                geo["block_size"],
                contiguous_query=True,
            )
            plan.fixup_positions(pos)
            counts = [min(p // 8 + 1, geo["topk"]) for p in work["positions"]]
            if any(
                sum(v >= 0 for v in row) != count
                for row, count in zip(work["indices"], counts, strict=False)
            ):
                raise Unsupported(
                    "Paged plan counts must equal fixed selected-region counts"
                )
            run = plan.finish
            staged_query = views.workspace(q.shape, q.dtype)

            def call():
                fill_prefill_pages(
                    regions,
                    table,
                    owners,
                    plan.indptr,
                    plan.pages,
                    geo["block_size"],
                    q,
                    staged_query,
                    kv_page_stride(cache),
                )
                plan.finish(staged_query, cache, output)
        elif backend == "flashinfer_ragged":
            if args.scope == "kernel":
                raise Unsupported(
                    "Ragged gather and native have different prepared input boundaries"
                )
            if not all(v >= 0 for row in work["block_table"] for v in row):
                raise Unsupported("Ragged prefill requires valid physical pages")
            from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
            from vllm.models.step5.nvidia.ops.sparse_prefill import gather_prefill_kv

            counts = [
                sum(min(8, p - region * 8 + 1) for region in row if region >= 0)
                for row, p in zip(work["indices"], work["positions"], strict=False)
            ]
            ptr = [0]
            for count in counts:
                ptr.append(ptr[-1] + count)
            host = torch.tensor(ptr, dtype=torch.int32).pin_memory()
            device = host.to("cuda")
            plan = BatchPrefillWithRaggedKVCacheWrapper(
                views.workspace([128 << 20], torch.uint8), "NHD", backend="fa2"
            )
            plan.plan(
                torch.arange(geo["rows"] + 1, dtype=torch.int32).pin_memory(),
                host,
                geo["heads"],
                1,
                192,
                head_dim_vo=192,
                causal=False,
                sm_scale=scale,
                q_data_type=q.dtype,
                kv_data_type=q.dtype,
                disable_split_kv=True,
            )
            run = plan.run

            def call():
                key, value = gather_prefill_kv(
                    cache,
                    regions,
                    table,
                    owners,
                    pos,
                    device,
                    ptr[-1],
                    max(counts),
                    output_dtype=q.dtype,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
                plan.run(q.contiguous(), key, value, out=output)
        else:
            raise Unsupported("Unknown prefill attention backend")

    def check():
        ids = check_selection_oracle(
            data, work["indices"], sample_rows=geo["rows"], observed=observed
        )
        if not torch.isfinite(output).all():
            raise AssertionError("Nonfinite attention output")
        return dict(
            status="passed",
            atol=0.025,
            rtol=0.025,
            oracle_rows=ids,
            fixed_selection=True,
            inactive_checked=not all(work["active"]),
            gate_checked=gate is not None,
        )

    count, key_ids = 0, set()
    for row_id, (selected_row, host_position, live, owner) in enumerate(
        zip(
            work["indices"],
            work["positions"],
            work["active"],
            work["owners"],
            strict=False,
        )
    ):
        if not live:
            continue
        group = row_id % geo["groups"] if geo["pd_type"] == "decode" else 0
        for region in selected_row:
            if region < 0:
                continue
            for token in range(region * 8, min(region * 8 + 8, host_position + 1)):
                physical = work["block_table"][owner][token // geo["block_size"]]
                if physical >= 0:
                    count += 1
                    key_ids.add((physical, group, token % geo["block_size"]))
    return Arm(
        backend,
        _entry(run),
        config,
        call,
        check,
        dict(
            bytes=q.numel() * 2
            + output.numel() * 2
            + len(key_ids) * 384 * cache.element_size(),
            flops=count * geo["heads"] * 192 * 4,
            selected_token_pairs=count,
            accounting="Unique valid selected KV tokens read once; causal/inactive/page masks; excludes backend staging/workspace traffic.",
        ),
        "attention-fixed-selection",
    )


def prepare_arms(torch, case, geo, work, backends, args):
    inputs, path = None, None
    fallback_work = work.get("synthetic_fallback_workload", work)
    if case.get("topk_tensor") and not work.get("topk_tensor_fallback"):
        path = resolve_topk_tensor(case, getattr(args, "topk_tensor_dir", None))
        if path is None:
            work = fallback_topk_workload(case, fallback_work)
    work = normalize_workload(case, geo, work)
    if path is not None:
        try:
            inputs, digest = load_topk_tensor(torch, case, path)
        except Unsupported as exc:
            if not isinstance(exc.__cause__, FileNotFoundError) or Path(path).is_file():
                raise
            logging.getLogger(__name__).warning(
                "topk_tensor %s disappeared before loading; falling back to synthetic inputs",
                path,
            )
            work = normalize_workload(
                case, geo, fallback_topk_workload(case, fallback_work)
            )
        else:
            work = tensor_workload(case, geo, work, inputs)
            work.update(topk_tensor_path=str(path), topk_tensor_sha256=digest)
    arms, blocked = [], []
    factories = {
        "topk": _topk,
        "direct": _direct,
        "packed": _packed,
        "grouped": _packed,
        "attention": _attention,
    }
    budget = args.max_memory_mib * (1 << 20)
    # Re-seeding restores exactly the same canonical data for every arm.
    for backend in backends:
        torch.manual_seed(args.seed)
        try:
            views = Views(torch, case, budget, inputs)
            arm = factories[geo["kind"]](torch, case, geo, work, backend, views, args)
            arm.config["normalized_workload"] = work
            arms.append(arm)
        except Unsupported as exc:
            blocked.append(dict(backend=backend, status="unsupported", reason=str(exc)))
        except (ImportError, ModuleNotFoundError) as exc:
            blocked.append(
                dict(
                    backend=backend,
                    status="unsupported",
                    reason="Missing dependency: " + str(exc),
                )
            )
        except Exception as exc:
            blocked.append(
                dict(
                    backend=backend,
                    status="execution_failed",
                    error=str(exc),
                    metrics=getattr(exc, "metrics", None),
                )
            )
    for item in blocked:
        item["synthetic"] = work["synthetic"]
        item["normalized_workload"] = work
        if work.get("topk_tensor_fallback"):
            item["topk_tensor_fallback"] = work["topk_tensor_fallback"]
    return arms, blocked
