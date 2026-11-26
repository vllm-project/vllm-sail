# SPDX-License-Identifier: Apache-2.0
"""Score pipeline acceptance against the real adapter; device dependencies load lazily, synthetic data is not an online replay."""

from __future__ import annotations

import math
import random
import statistics
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial
from types import MethodType, SimpleNamespace

from tests.support.step5_attention_testing import (
    override_step5_environment,
    reject_instrumentation,
)
from vllm_sail.models.step5.ops import score_pipeline as score

ROUTES = ("reference", "resident", "compact", "mqa2", "mqa4", "mqa8")
_BQ = ContextVar("step5_score_testing_bq", default=None)
_COMPACT_CONFIG = ContextVar("step5_score_testing_compact_config", default=None)


class ScoreDependencyBlocked(RuntimeError):
    """An explicitly requested device dependency is missing; never disguise it as another backend passing."""


def direct_cases(large=False, *, workload="regression"):
    """Synthetic decode/MTP inputs; HTTP concurrency is not the actual DP-local batch size."""
    if workload not in ("regression", "tp4-dp2"):
        raise ValueError("unknown direct workload")
    cases = [
        dict(name="one_region", capacity=1, requests=1),
        dict(name="short_invalid", capacity=35, requests=3, invalid=True),
        dict(name="tile127", capacity=127, requests=4, rpb=2),
        dict(name="tile128", capacity=128, requests=4, rpb=4, pattern="ties"),
        dict(name="tile129", capacity=129, requests=4, pattern="near_ties"),
        dict(
            name="mtp_groups",
            capacity=641,
            requests=4,
            groups=3,
            query_len=3,
            layout="group",
            invalid=True,
        ),
        dict(
            name="causal_edges",
            capacity=129,
            requests=5,
            query_len=2,
            seqs=[-16, 0, 1, 9, 1032],
            inactive=True,
        ),
        dict(name="heads4", capacity=129, requests=2, heads=4),
        dict(name="heads16", capacity=129, requests=2, heads=16),
        dict(name="fp16_storage", capacity=641, requests=4, storage="fp16"),
        dict(name="decode8_1k", capacity=1024, requests=8),
        dict(name="decode32_8k", capacity=8192, requests=32),
    ]
    if large:
        cases += [
            dict(name=f"decode{n}_8k", capacity=8192, requests=n)
            for n in (1, 8, 64, 128)
        ]
        cases += [
            dict(
                name="mtp2_8k",
                capacity=8192,
                requests=16,
                groups=2,
                query_len=2,
                layout="group",
            )
        ]
        cases += [
            dict(
                name="mtp4_full_h8_groups",
                capacity=8193,
                requests=8,
                groups=3,
                query_len=4,
                layout="group",
                invalid=True,
            ),
            dict(
                name="mtp4_causal_edges",
                capacity=129,
                requests=8,
                query_len=4,
                seqs=[-16, 0, 1, 7, 8, 9, 1025, 1032],
                inactive=True,
            ),
            dict(
                name="mtp4_ties", capacity=129, requests=8, query_len=4, pattern="ties"
            ),
            dict(
                name="mtp4_near_ties",
                capacity=8193,
                requests=16,
                query_len=4,
                pattern="near_ties",
                invalid=True,
            ),
            dict(
                name="mtp4_guard_mixed",
                capacity=131072,
                requests=8,
                query_len=4,
                layout="production",
                seqs=[4096, 65536, 5596, 69632, 65535, 65537, 69633, 7],
            ),
            dict(
                name="mtp4_guard_inactive",
                capacity=131072,
                requests=16,
                query_len=4,
                seqs=[65536] * 16,
                inactive=True,
            ),
        ]
    if workload == "tp4-dp2":
        cases = [
            dict(
                name=f"tp4_dp2_r{requests}_q{query_len}_s{seq}",
                capacity=131072,
                requests=requests,
                query_len=query_len,
                seqs=[seq] * requests,
                effective_heads=4,
                layout="production",
                http_concurrency_hint=2 * requests,
                dp_partition="assumed_balanced_not_observed",
                scheduler_token_budget=8192,
            )
            for requests in (8, 16)
            for query_len in (1, 4)
            for seq in (4096, 5596, 65536, 69632)
        ]
        if large:
            cases += [
                dict(
                    name=f"tp4_dp2_tail_r{requests}_q{query_len}",
                    capacity=131072,
                    requests=requests,
                    query_len=query_len,
                    seqs=[(4096, 5596, 65536, 69632)[i % 4] for i in range(requests)],
                    effective_heads=4,
                    layout="production",
                    inactive=True,
                )
                for requests in (1, 7, 9, 15, 17, 32)
                for query_len in (1, 2, 3, 4)
            ]
            cases += [
                cases[6]
                | dict(name="tp4_dp2_shared_pages_control", page_layout="shared")
            ]
            cases += [
                dict(
                    name=f"tp4_dp2_boundary_s{seq}",
                    capacity=131072,
                    requests=8,
                    query_len=4,
                    seqs=[seq] * 8,
                    effective_heads=4,
                    layout="token",
                )
                for seq in (7, 8, 9, 1023, 1024, 1025, 65535, 65537)
            ]
        for case in cases:
            case.setdefault("scheduler_token_budget", 8192)
            case.setdefault("dp_partition", "synthetic_not_observed")
    normalized = []
    for row in cases:
        spec = (
            dict(
                groups=1,
                query_len=1,
                rpb=8,
                heads=8,
                storage="fp8",
                layout="token",
                pattern="random",
                invalid=False,
                inactive=False,
                seed=128,
                page_layout="independent",
            )
            | row
        )
        spec.setdefault("effective_heads", spec["heads"])
        normalized.append(spec)
    return normalized


def direct_mtp_cases():
    """Full H8 synthetic MTP sweep; seq is tokens and capacity is summary regions."""
    base = direct_cases()[0] | dict(query_len=4)
    cases = [
        dict(
            name=f"mtp4_r{requests}_s{seq}",
            requests=requests,
            seqs=[seq] * requests,
            capacity=131072 if requests == 48 else seq // 8 + 1,
            layout="production" if requests == 48 else "token",
        )
        for requests in (1, 7, 8, 16, 17, 32, 48)
        for seq in (4096, 131072)
    ]
    cases += [
        dict(
            name=f"mtp4_r48_s{seq}",
            requests=48,
            seqs=[seq] * 48,
            capacity=131072,
            layout="production",
        )
        for seq in (131073, 133120, 135167)
    ]
    cases += [
        dict(name=f"mtp4_boundary_s{seq}", seqs=[seq], capacity=max(1, seq // 8 + 1))
        for seq in (
            4,
            7,
            8,
            9,
            1023,
            1024,
            1025,
            5596,
            65535,
            65536,
            65537,
            69632,
            69633,
            131071,
        )
    ]
    cases += [
        dict(
            name="mtp4_r48_s135168",
            requests=48,
            seqs=[135168] * 48,
            capacity=131072,
            layout="production",
        ),
        dict(
            name="mtp4_mixed_safety",
            requests=8,
            seqs=[-16, 0, 1, 3, 4, 9, 4096, 131072],
            capacity=16385,
            invalid=True,
        ),
        dict(
            name="mtp4_inactive_each_offset",
            requests=7,
            seqs=[1025] * 7,
            capacity=129,
            inactive_tokens=[4, 9, 14, 19],
        ),
        dict(name="mtp4_small_capacity", seqs=[4096], capacity=1),
        dict(name="mtp4_ties_small", seqs=[1032], capacity=129, pattern="ties"),
        dict(
            name="mtp4_near_ties_small", seqs=[1032], capacity=129, pattern="near_ties"
        ),
    ]
    return [base | case for case in cases]


def select_direct_cases(large=False, requested=None, *, workload="regression"):
    if workload == "mtp":
        base = next(c for c in direct_mtp_cases() if c["name"] == "mtp4_r48_s131072")
        cases = [
            base | dict(name=f"mtp4_r48_s{seq}", seqs=[seq] * 48)
            for seq in (4096, 65536, 131072, 133120, 135168)
        ]
    else:
        cases = direct_cases(large, workload=workload)
    if requested is not None:
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("empty or duplicated direct case selection")
        cases = [case for case in cases if case["name"] in requested]
        if len(cases) != len(requested):
            raise ValueError("unknown direct case; check workload and --large")
    return cases


def direct_host_rows(spec):
    """Original decode causal metadata and the 128-region written prefix."""
    for key in ("requests", "groups", "query_len", "capacity", "rpb", "heads"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError("invalid direct score geometry: " + key)
    if spec["heads"] not in (4, 8, 16) or spec["rpb"] not in (2, 4, 8):
        raise ValueError("unsupported direct score heads or summary block")
    effective = spec.get("effective_heads", spec["heads"])
    if (
        type(effective) is not int
        or effective not in (4, 8, 16)
        or effective > spec["heads"]
    ):
        raise ValueError("invalid direct effective heads")
    if spec.get("page_layout", "independent") not in ("independent", "shared"):
        raise ValueError("invalid direct page layout")
    if spec["layout"] not in ("token", "group", "production"):
        raise ValueError("invalid direct tensor layout")
    seqs = spec.get(
        "seqs", [spec["capacity"] * 8 - i % 7 for i in range(spec["requests"])]
    )
    if len(seqs) != spec["requests"] or any(type(seq) is not int for seq in seqs):
        raise ValueError("direct sequence/request count mismatch or noninteger length")
    rows = []
    for token in range(spec["requests"] * spec["query_len"]):
        owner, offset = divmod(token, spec["query_len"])
        active = not (
            (spec["inactive"] and token % 3 == 0)
            or token in spec.get("inactive_tokens", ())
        )
        seq = seqs[owner] - (spec["query_len"] - 1 - offset)
        position = seq - 1 if active else 0
        current = position >> 3
        length = current + 1
        written = min(
            spec["capacity"],
            math.ceil(min(max(length, 0), spec["capacity"]) / 128) * 128,
        )
        for group in range(spec["groups"]):
            rows.append(
                (
                    active,
                    position,
                    current,
                    length,
                    current,
                    owner,
                    token,
                    group,
                    written,
                )
            )
    return rows, seqs


def direct_host_pages(spec):
    """Independent requests do not share physical pages; allocate K only for the prefix writable by the original kernel."""
    host, _ = direct_host_rows(spec)
    columns = math.ceil(spec["capacity"] / spec["rpb"])
    used = [0] * spec["requests"]
    for row in host:
        used[row[5]] = max(used[row[5]], math.ceil(row[-1] / spec["rpb"]))
    if spec.get("page_layout", "independent") == "shared":
        pages = columns + spec["requests"]
        table = [
            [(column + owner) % pages for column in range(columns)]
            for owner in range(spec["requests"])
        ]
    else:
        pages = max(1, sum(used))
        physical = list(range(pages))
        random.Random(spec["seed"]).shuffle(physical)
        table, offset = [], 0
        for count in used:
            table.append(physical[offset : offset + count] + [-1] * (columns - count))
            offset += count
    if spec["invalid"]:
        table = [
            [
                (-1 if column % 3 == 0 else pages + 7 if column % 3 == 1 else page)
                if page >= 0
                else -1
                for column, page in enumerate(row)
            ]
            for row in table
        ]
    return SimpleNamespace(table=table, pages=pages, used_columns=used)


def direct_pad_heads(query, weights, effective_heads):
    """Preserve H8 storage; zero padded Q/W heads only under the explicit test contract."""
    query[..., effective_heads:, :] = 0
    weights[..., effective_heads:] = 0


def prepare_direct_case(spec, extension_root=None):
    from tests.support.step5_attention_testing import require_runtime

    host, seqs = direct_host_rows(spec)
    page_plan = direct_host_pages(spec)
    torch, module = require_runtime("flashinfer", extension_root)
    from vllm_sail.models.step5.ops import soft_fp8, sparse_attention_opt

    if tuple(torch.cuda.get_device_capability(0)) != (8, 9):
        raise ScoreDependencyBlocked("direct score optimization requires SM89")
    torch.manual_seed(spec["seed"])
    tokens, groups, heads = (
        spec["requests"] * spec["query_len"],
        spec["groups"],
        spec["heads"],
    )
    columns = math.ceil(spec["capacity"] / spec["rpb"])
    pages = page_plan.pages
    query = (torch.randn((tokens, groups, heads, 256), device="cuda") * 0.125).to(
        torch.float8_e4m3fn
    )
    key = (torch.randn((pages, spec["rpb"], 256), device="cuda") * 0.125).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn((tokens, groups, heads), device="cuda") * 0.2
    if spec["pattern"] != "random":
        query.fill_(0.125)
        key.fill_(0.125)
        weights.fill_(0.2)
        weights[..., 0] = -0.1
        if spec["pattern"] == "near_ties":
            key.view(torch.uint8)[:, 1::2, 0] += 1
    direct_pad_heads(query, weights, spec.get("effective_heads", heads))
    if spec["storage"] == "fp16":
        query, key = query.half(), key.half()
    dtype = query.dtype
    if spec["layout"] == "production":
        q = torch.empty(
            (groups, tokens, heads, 256), dtype=dtype, device="cuda"
        ).transpose(0, 1)
        w = torch.empty((groups, tokens, heads), device="cuda").transpose(0, 1)
    elif spec["layout"] == "group":
        q = torch.empty(
            (groups + 1, tokens + 1, heads, 256), dtype=dtype, device="cuda"
        ).transpose(0, 1)[1:, 1:]
        w = torch.empty((groups + 1, tokens + 1, heads), device="cuda").transpose(0, 1)[
            1:, 1:
        ]
    else:
        q = torch.empty(
            (tokens + 1, groups + 1, heads, 256), dtype=dtype, device="cuda"
        )[1:, 1:]
        w = torch.empty((tokens + 1, groups + 1, heads), device="cuda")[1:, 1:]
    q.copy_(query)
    w.copy_(weights)
    mean_dtype = torch.float16 if spec["storage"] == "fp16" else torch.uint8
    padding = int(spec["layout"] != "production")
    mean = torch.empty(
        (pages, spec["rpb"] + padding, 256), dtype=mean_dtype, device="cuda"
    )[:, : spec["rpb"]]
    mean.copy_(key if mean_dtype == torch.float16 else key.view(torch.uint8))
    host_table = page_plan.table
    table = torch.full(
        (spec["requests"], columns + 3), -1, dtype=torch.int32, device="cuda"
    )[:, :columns]
    table.copy_(torch.tensor(host_table, dtype=torch.int32, device="cuda"))
    slots = torch.empty(tokens * 2 + 1, dtype=torch.int64, device="cuda")[1::2]
    slots.copy_(
        torch.tensor(
            [0 if host[token * groups][0] else -1 for token in range(tokens)],
            device="cuda",
        )
    )
    seq = torch.empty(spec["requests"] * 2 + 1, dtype=torch.int64, device="cuda")[1::2]
    seq.copy_(torch.tensor(seqs, device="cuda"))
    if spec["layout"] == "production":
        table, slots, seq = table.contiguous(), slots.contiguous(), seq.contiguous()
    return SimpleNamespace(
        torch=torch,
        module=module,
        spec=spec,
        host=host,
        host_table=host_table,
        query=q,
        weights=w,
        mean=mean,
        table=table,
        slots=slots,
        seq=seq,
        soft=soft_fp8,
        kernels=sparse_attention_opt,
        adapter=module._shared_adapter(torch.device("cuda:0")),
    )


def direct_capture(data, config=None, *, initialize=True, reference_warps=4):
    """Capture one real decode launcher; metadata and scores are preallocated for replay."""
    from vllm.models.step5.nvidia.ops import sparse_indexer_scores as upstream

    from vllm_sail.models.step5.ops.soft_fp8 import DirectScoresLaunch

    torch, spec = data.torch, data.spec
    guarded = config is not None and config.get("guard_mtp", False)
    calls = []

    class Recorder:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                args = list(args)
                if args[0].dtype == torch.float8_e4m3fn:
                    args[0] = args[0].view(torch.uint8)
                # The public launcher is H8. H4/H16 are low-level tests only.
                args[25] = spec["heads"]
                if initialize:
                    args[6].fill_(float("nan"))
                if config is None or (
                    guarded and not DirectScoresLaunch._mtp_eligible(args)
                ):
                    kernel, launch_grid = data.soft._direct_scores, grid
                    kwargs["num_warps"] = reference_warps
                else:
                    kernel = data.kernels._direct_scores_resident
                    launch_grid = grid if guarded else (grid[0] * grid[1],)
                    kwargs.update(
                        HOIST=config["hoist"],
                        SPLIT=config["split"],
                        num_warps=config["warps"],
                    )
                    if guarded:
                        kwargs["GUARD_MTP"] = True
                    elif config.get("pack_queries", False):
                        kwargs["PACK_QUERIES"] = True
                launch = kernel[launch_grid]
                compiled = launch(*args, **kwargs)
                calls.append(
                    SimpleNamespace(
                        launch=launch,
                        args=tuple(args),
                        kwargs=kwargs,
                        compiled=compiled,
                        grid=launch_grid,
                    )
                )
                return compiled

            return run

    saved = upstream._direct_scores
    upstream._direct_scores = Recorder()
    try:
        if spec["heads"] == 8:
            scores, metadata = data.adapter.decode_logits(
                data.query,
                data.weights,
                data.mean,
                data.table,
                spec["capacity"],
                data.slots,
                data.seq,
                row_groups=spec["groups"],
                query_len=spec["query_len"],
            )
        else:
            scores, metadata = upstream.decode_summary_scores(
                data.query,
                data.weights,
                data.mean,
                data.table,
                data.slots,
                data.seq,
                spec["capacity"],
                num_sms=data.adapter._num_sms,
                row_groups=spec["groups"],
                query_len=spec["query_len"],
            )
    finally:
        upstream._direct_scores = saved
    if len(calls) != 1:
        raise AssertionError(
            "direct decode must submit exactly one direct score kernel"
        )
    call = calls[0]
    call.scores, call.metadata = scores, metadata
    call.replay = partial(call.launch, *call.args, **call.kwargs)
    return call


def direct_launch_capture(data, mode, *, warps=None):
    """Replay the real v0/v2 DirectScoresLaunch through adapter.decode_logits."""
    from vllm.models.step5.nvidia.ops import sparse_indexer_scores as upstream

    from vllm_sail.models.step5.ops.soft_fp8 import DirectScoresLaunch

    if mode not in ("v0", "v2"):
        raise ValueError("MTP acceptance compares v0 and v2 only")
    if warps is not None and warps not in (2, 4, 8):
        raise ValueError("unsupported MTP benchmark warp count")
    calls = []

    class RecordingLaunch(DirectScoresLaunch):
        @staticmethod
        def _submit(call, args, kwargs, backend, source_args=None):
            if warps is not None:
                kwargs = dict(kwargs, num_warps=warps)
            args[6].fill_(float("nan"))
            compiled = call(*args, **kwargs)
            calls.append(
                SimpleNamespace(
                    args=tuple(args),
                    kwargs=dict(kwargs),
                    compiled=compiled,
                    grid=(args[27], args[0].shape[0] * args[28]),
                    backend=backend,
                    replay=partial(call, *args, **kwargs),
                )
            )
            return compiled

    saved = upstream._direct_scores
    upstream._direct_scores = RecordingLaunch(
        data.soft._direct_scores, data.soft._direct_scores, (0,)
    )
    try:
        with override_step5_environment({"VLLM_SAIL_STEP5_DSA_SCORE_DECODE": mode}):
            scores, metadata = data.adapter.decode_logits(
                data.query,
                data.weights,
                data.mean,
                data.table,
                data.spec["capacity"],
                data.slots,
                data.seq,
                row_groups=data.spec["groups"],
                query_len=data.spec["query_len"],
            )
    finally:
        upstream._direct_scores = saved
    if len(calls) != 1:
        raise AssertionError("decode must submit exactly one direct score kernel")
    call = calls[0]
    call.scores, call.metadata = scores, metadata
    return call


def direct_cpu_oracle(data, ids):
    """Independent CPU FP32 dot and the original four FMA-chain epilogue."""
    torch = data.torch
    q = data.query.cpu().float()
    raw = data.mean.cpu()
    key = (
        raw.float()
        if raw.dtype == torch.float16
        else raw.view(torch.float8_e4m3fn).float()
    )
    weights = data.weights.cpu().float()
    expected = []
    for row in ids:
        _, _, _, _, _, owner, token, group, written = data.host[row]
        regions = torch.arange(written)
        col = regions // data.spec["rpb"]
        physical = torch.tensor(data.host_table[owner])[col]
        valid = (physical >= 0) & (physical < key.shape[0])
        keys = key[
            physical.clamp(0, key.shape[0] - 1), regions % data.spec["rpb"]
        ].clone()
        keys[~valid] = 0
        dot = (keys @ q[token, group].T).clamp_min(0)
        sums = torch.zeros((written, 4), dtype=torch.float32)
        for head_group in range(data.spec["heads"] // 4):
            first = head_group * 4
            sums = (
                dot[:, first : first + 4].double()
                * weights[token, group, first : first + 4].double()
                + sums.double()
            ).float()
        expected.append(((sums[:, 0] + sums[:, 1]) + sums[:, 2]) + sums[:, 3])
    return expected


def direct_check(data, old, new, oracle_samples=26):
    torch = data.torch
    for call in (old, new):
        for index in range(5):
            wanted = torch.tensor(
                [row[index] for row in data.host],
                device="cuda",
                dtype=call.args[7 + index].dtype,
            )
            torch.testing.assert_close(call.args[7 + index], wanted, atol=0, rtol=0)
        if call.metadata.block_table is not data.table:
            raise AssertionError(
                "direct metadata must preserve the original page table"
            )
    pieces, reference = [], []
    for row, host in enumerate(data.host):
        written = host[-1]
        pieces.append(new.scores[row, :written])
        reference.append(old.scores[row, :written])
        for call in (old, new):
            if not bool(torch.isnan(call.scores[row, written:]).all()):
                raise AssertionError(
                    "direct kernel unexpectedly wrote the unspecified tail"
                )
        count = min(max(host[4], 0), data.spec["capacity"]) if host[0] else 0
        if count:
            k = min(16, count)
            a = torch.topk(old.scores[row, :count], k).indices.sort().values
            b = torch.topk(new.scores[row, :count], k).indices.sort().values
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    equivalence = error_metrics(torch, torch.cat(pieces), torch.cat(reference))
    ids = sorted(
        {
            round(
                i
                * (len(data.host) - 1)
                / max(1, min(oracle_samples, len(data.host)) - 1)
            )
            for i in range(min(oracle_samples, len(data.host)))
        }
    )
    expected = direct_cpu_oracle(data, ids)
    oracle = {}
    for name, call in (("reference", old), ("candidate", new)):
        actual = torch.cat(
            [call.scores[row, : data.host[row][-1]].cpu() for row in ids]
        )
        try:
            oracle[name] = dict(
                status="passed", **error_metrics(torch, actual, torch.cat(expected))
            )
        except ScoreNumericalError as exc:
            oracle[name] = dict(status="failed", error=str(exc), **exc.metrics)
    saved = new.scores.clone()
    new.replay()
    torch.testing.assert_close(new.scores, saved, atol=0, rtol=0, equal_nan=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new.replay()
    graph.replay()
    torch.testing.assert_close(new.scores, saved, atol=0, rtol=0, equal_nan=True)
    original_key = data.mean.clone()
    data.mean.copy_(
        torch.ones_like(data.mean)
        if data.spec["storage"] == "fp16"
        else torch.full_like(data.mean, 56)
    )
    try:
        old.replay()
        graph.replay()
        error_metrics(
            torch,
            torch.cat([new.scores[r, : h[-1]] for r, h in enumerate(data.host)]),
            torch.cat([old.scores[r, : h[-1]] for r, h in enumerate(data.host)]),
        )
        if any(h[-1] for h in data.host) and not data.spec["invalid"]:
            if torch.equal(new.scores.nan_to_num(), saved.nan_to_num()):
                raise AssertionError("changed shared K must recompute direct scores")
    finally:
        data.mean.copy_(original_key)
    old.replay()
    new.replay()
    return dict(
        equivalence=equivalence,
        fp32_oracle=oracle,
        oracle_rows=ids,
        metadata_exact=True,
        topk_sets_exact=True,
        repeated_graph_exact=True,
        changed_K_recomputed=True,
        unspecified_tail_preserved=True,
    )


def direct_memory(torch, calls):
    """Measure peak allocation after stable reclamation so asynchronous frees cannot fabricate savings."""
    import gc

    result = {}
    for name, call in calls.items():
        gc.collect()
        torch.cuda.synchronize()
        warmup = call()
        torch.cuda.synchronize()
        del warmup
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        output = call()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        output_bytes = output[0].untyped_storage().nbytes()
        if peak - base < output_bytes:
            raise AssertionError(
                "allocation measurement must cover the live score output"
            )
        result[name] = dict(
            peak_allocated_bytes=peak,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            incremental_allocated_bytes=peak - base,
            score_storage_bytes=output_bytes,
        )
        del output
    return result


def direct_oracle_passed(check):
    return all(v["status"] == "passed" for v in check["fp32_oracle"].values())


def direct_sweep(data, old, mode, rounds, iterations, oracle_samples):
    from itertools import product

    reference, candidate = [], []
    configs = [(None, w) for w in (2, 4)] + [
        (dict(hoist=h, split=s, warps=w), 4)
        for h, s, w in product((False, True), (False, True), (2, 4))
    ]
    for config, warps in configs:
        call = direct_capture(data, config, reference_warps=warps)
        check = direct_check(data, old, call, oracle_samples)
        item = dict(
            config=config or dict(warps=warps),
            correctness=check,
            numerical_valid=direct_oracle_passed(check),
        )
        target, name = (
            (reference, "reference") if config is None else (candidate, "candidate")
        )
        target.append(item)
        if mode == "direct-ab" and not item["numerical_valid"]:
            item["status"] = "blocked_by_oracle"
            continue
        item["timing"] = direct_timing(
            data.torch, {name: call.replay}, rounds, iterations
        )
    return reference, candidate


def direct_launcher(data, config=None, *, reference_warps=4):
    """Bind the unchanged public allocation launcher without a global patch or recorder."""
    from types import FunctionType

    from vllm.models.step5.nvidia.ops import sparse_indexer_scores as upstream

    from vllm_sail.models.step5.ops.soft_fp8 import DirectScoresLaunch

    guarded = config is not None and config.get("guard_mtp", False)

    class Delegate:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                args = list(args)
                args[25] = data.spec["heads"]
                if config is None or (
                    guarded and not DirectScoresLaunch._mtp_eligible(args)
                ):
                    kernel = data.soft._direct_scores
                    kwargs["num_warps"] = reference_warps
                else:
                    kernel = data.kernels._direct_scores_resident
                    grid_local = grid if guarded else (grid[0] * grid[1],)
                    kwargs.update(
                        HOIST=config["hoist"],
                        SPLIT=config["split"],
                        num_warps=config["warps"],
                    )
                    if guarded:
                        kwargs["GUARD_MTP"] = True
                    elif config.get("pack_queries", False):
                        kwargs["PACK_QUERIES"] = True
                    return kernel[grid_local](*args, **kwargs)
                return kernel[grid](*args, **kwargs)

            return launch

    function = upstream.decode_summary_scores
    namespace = dict(function.__globals__)
    if guarded:

        class GuardedLaunch(DirectScoresLaunch):
            @staticmethod
            def _resident_kernel():
                return data.kernels._direct_scores_resident

        namespace["_direct_scores"] = GuardedLaunch(
            None, data.soft._direct_scores, (0,), guarded_mtp=True
        )
    else:
        namespace["_direct_scores"] = data.soft.ByteFP8Launch(
            None, Delegate(), byte_args=(0,)
        )
    isolated = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    isolated.__kwdefaults__ = function.__kwdefaults__
    return partial(
        isolated,
        data.query,
        data.weights,
        data.mean,
        data.table,
        data.slots,
        data.seq,
        data.spec["capacity"],
        num_sms=data.adapter._num_sms,
        row_groups=data.spec["groups"],
        query_len=data.spec["query_len"],
    )


def direct_profile(torch, calls, directory):
    """Independent device trace; profiler durations never enter A/B timing."""
    import json
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    result = {}
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            output = call()
            torch.cuda.synchronize()
        trace = directory / (name + ".trace.json")
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
        direct = [n for n in kernels if "_direct_scores" in n]
        if len(direct) != 1 or len(kernels) != 1:
            raise AssertionError(
                "direct launcher trace must contain exactly one direct kernel: "
                + repr(kernels)
            )
        result[name] = dict(
            trace=str(trace), kernel_names=kernels, direct_kernel_hits=len(direct)
        )
        del output
    return result


def direct_timing(torch, calls, rounds=6, iterations=32):
    """CUPTI/cold-L2 preferred; the explicit fallback is event/graph warm-cache."""
    import importlib

    try:
        bench = importlib.import_module("flashinfer.testing").bench_gpu_time_with_cupti
        importlib.import_module("cupti")
    except (ImportError, AttributeError) as exc:
        result = diagnostic_graph_ab(torch, calls, rounds, iterations)
        result["timer"] = "CUDA event / CUDA graph"
        result["fallback_reason"] = str(exc)
        return result
    samples, orders = {name: [] for name in calls}, []
    for call in calls.values():
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    for index in range(rounds):
        order = tuple(calls) if index % 2 == 0 else tuple(reversed(calls))
        orders.append(list(order))
        for name in order:
            values = bench(
                calls[name],
                dry_run_iters=3,
                repeat_iters=iterations,
                use_cuda_graph=True,
                cold_l2_cache=True,
            )
            samples[name].append(float(statistics.median(values)) * 1000)
    return dict(
        orders=orders,
        timer="CUPTI / CUDA graph",
        cache_policy="cold L2",
        routes={
            name: dict(
                median_device_us=statistics.median(values),
                device_group_us=values,
                range_pct=(max(values) - min(values)) / statistics.median(values) * 100,
            )
            for name, values in samples.items()
        },
    )


def direct_mtp_launch_info(call):
    return dict(
        backend=call.backend,
        grid=list(call.grid),
        kwargs=call.kwargs,
        workers=call.args[27],
        row_groups=call.args[28],
        query_len=call.args[29],
    )


def direct_mtp_timing(torch, calls, rounds, iterations):
    result = direct_timing(
        torch, {n: c.replay for n, c in calls.items()}, rounds, iterations
    )
    arms = result["routes"]
    ratios = [
        a / b
        for a, b in zip(
            arms["reference"]["device_group_us"],
            arms["candidate"]["device_group_us"],
            strict=True,
        )
    ]
    result["speedup"] = (
        arms["reference"]["median_device_us"] / arms["candidate"]["median_device_us"]
    )
    result["min_round_speedup"] = min(ratios)
    result["max_round_speedup"] = max(ratios)
    return result


def direct_mtp_matrix(
    specs, extension_root, mode, rounds, iterations, oracle_samples, directory
):
    import json
    from pathlib import Path

    from step5_attention_harness import provenance

    reject_instrumentation()
    rows = []
    for spec in specs:
        data = prepare_direct_case(spec, extension_root)
        old = direct_launch_capture(data, "v0")
        new = direct_launch_capture(data, "v2")
        if (
            old.backend != "direct_baseline"
            or new.backend != "direct_mtp"
            or not new.kwargs.get("GUARD_MTP")
            or old.grid != new.grid
        ):
            raise AssertionError("MTP timing requires real v0/v2 on the original grid")
        row = dict(
            case=spec,
            correctness=direct_check(data, old, new, oracle_samples),
            production_launch={
                "reference": direct_mtp_launch_info(old),
                "candidate": direct_mtp_launch_info(new),
            },
            strides={
                n: list(getattr(data, n).stride())
                for n in ("query", "weights", "mean", "table", "slots", "seq")
            },
            production_dispatch_eligible=True,
            performance_valid=False,
            timing_scope="direct score kernel only; allocation/JIT/checks/host submission excluded",
            work_estimate=dict(
                unique_key_bytes=data.mean.numel() * data.mean.element_size(),
                output_written_bytes=sum(h[-1] for h in data.host) * 4,
                dot_flops=2 * sum(h[-1] for h in data.host) * spec["heads"] * 256,
                convention="rounded writable prefix, unique K read once; not measured HBM traffic",
            ),
        )
        row["numerical_valid"] = direct_oracle_passed(row["correctness"])
        rows.append(row)
        if mode != "direct-numerical" and (
            row["numerical_valid"] or mode == "direct-diagnostic-ab"
        ):
            row["kernel_only"] = direct_mtp_timing(
                data.torch, {"reference": old, "candidate": new}, rounds, iterations
            )
            print(
                f"MTP {spec['name']}: production speedup={row['kernel_only']['speedup']:.4f}, "
                f"oracle={row['numerical_valid']}",
                flush=True,
            )
            sweep = []
            for warps in (2, 4, 8):
                reference = direct_launch_capture(data, "v0", warps=warps)
                candidate = direct_launch_capture(data, "v2", warps=warps)
                checks = {
                    "reference": direct_check(data, old, reference, oracle_samples),
                    "candidate": direct_check(data, old, candidate, oracle_samples),
                }
                item = dict(
                    warps=warps,
                    correctness=checks,
                    launch={
                        "reference": direct_mtp_launch_info(reference),
                        "candidate": direct_mtp_launch_info(candidate),
                    },
                )
                sweep.append(item)
                if mode == "direct-ab" and not all(
                    map(direct_oracle_passed, checks.values())
                ):
                    item["status"] = "blocked_by_oracle"
                    continue
                item["timing"] = direct_mtp_timing(
                    data.torch,
                    {"reference": reference, "candidate": candidate},
                    rounds,
                    iterations,
                )
            row["sweep"] = sweep
            available = [s for s in sweep if "timing" in s]
            if available:
                winners = {
                    name: min(
                        available,
                        key=lambda s: s["timing"]["routes"][name]["median_device_us"],
                    )["warps"]
                    for name in ("reference", "candidate")
                }
                row["best_warps"] = winners
                row["both_tuned"] = direct_mtp_timing(
                    data.torch,
                    {
                        "reference": direct_launch_capture(
                            data, "v0", warps=winners["reference"]
                        ),
                        "candidate": direct_launch_capture(
                            data, "v2", warps=winners["candidate"]
                        ),
                    },
                    rounds,
                    iterations,
                )
            row["performance_valid"] = (
                mode == "direct-ab" and row["numerical_valid"] and "both_tuned" in row
            )
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / (spec["name"] + ".json")).open("x") as stream:
                json.dump(row, stream, indent=2, allow_nan=False)
        print(f"MTP {spec['name']}: complete", flush=True)
    valid = all(r["numerical_valid"] for r in rows)
    return dict(
        cases=rows,
        provenance=provenance(SimpleNamespace(module=data.module)),
        arms={"reference": "v0_direct_baseline", "candidate": "v2_guarded_mtp"},
        numerical_valid=valid,
        performance_valid=all(r["performance_valid"] for r in rows),
        status="diagnosed"
        if mode == "direct-diagnostic-ab"
        else "passed"
        if valid
        else "failed",
        scope="synthetic single-device H8 MTP direct score; not 128K prefill/4K generation/full model/HTTP throughput",
    )


def direct_score_matrix(
    extension_root=None,
    mode="direct-numerical",
    *,
    large=False,
    requested=None,
    rounds=6,
    iterations=32,
    oracle_samples=26,
    diagnostic_dir=None,
    workload="regression",
):
    """Original/candidate A/B; inherited oracle errors never enable dispatch."""
    from pathlib import Path

    from step5_attention_harness import provenance

    if mode not in ("direct-numerical", "direct-ab", "direct-diagnostic-ab"):
        raise ValueError("unknown direct acceptance mode")
    if rounds < 6 or iterations <= 0 or oracle_samples <= 0:
        raise ValueError(
            "direct A/B requires six rounds and positive iterations/oracle rows"
        )
    specs = select_direct_cases(large, requested, workload=workload)
    if workload == "mtp":
        return direct_mtp_matrix(
            specs,
            extension_root,
            mode,
            rounds,
            iterations,
            oracle_samples,
            diagnostic_dir,
        )
    rows = []
    for spec in specs:
        data = prepare_direct_case(spec, extension_root)
        torch = data.torch
        old = direct_capture(data)
        default = dict(hoist=True, split=True, warps=4)
        new = direct_capture(data, default)
        row = dict(
            case=spec,
            correctness=direct_check(data, old, new, oracle_samples),
            default_config=default,
            production_dispatch_eligible=False,
            strides={
                n: list(getattr(data, n).stride())
                for n in ("query", "weights", "mean", "table", "slots", "seq")
            },
        )
        row["numerical_valid"] = direct_oracle_passed(row["correctness"])
        rows.append(row)
        if mode != "direct-numerical" and (
            row["numerical_valid"] or mode == "direct-diagnostic-ab"
        ):
            reference_sweep, sweep = direct_sweep(
                data, old, mode, rounds, iterations, oracle_samples
            )
            row["reference_sweep"], row["sweep"] = reference_sweep, sweep
            best = min(
                (s for s in sweep if "timing" in s),
                key=lambda s: s["timing"]["routes"]["candidate"]["median_device_us"],
            )
            best_reference = min(
                (s for s in reference_sweep if "timing" in s),
                key=lambda s: s["timing"]["routes"]["reference"]["median_device_us"],
            )
            winner = direct_capture(data, best["config"])
            tuned_reference = direct_capture(
                data, reference_warps=best_reference["config"]["warps"]
            )
            row["best_config"], row["best_reference_config"] = (
                best["config"],
                best_reference["config"],
            )
            row["kernel_only"] = direct_timing(
                torch,
                {"reference": old.replay, "candidate": new.replay},
                rounds,
                iterations,
            )
            row["both_tuned"] = direct_timing(
                torch,
                {"reference": tuned_reference.replay, "candidate": winner.replay},
                rounds,
                iterations,
            )
            calls = {
                "reference": direct_launcher(data),
                "candidate": direct_launcher(data, default),
            }
            row["decode_launcher"] = direct_timing(torch, calls, rounds, iterations)
            row["decode_launcher"]["scope"] = (
                "unchanged public decode_summary_scores allocation launcher; no Recorder; graph device time excludes host allocation latency"
            )
            row["memory"] = direct_memory(torch, calls)
            for metric in ("kernel_only", "both_tuned", "decode_launcher"):
                arms = row[metric]["routes"]
                ratios = [
                    a / b
                    for a, b in zip(
                        arms["reference"]["device_group_us"],
                        arms["candidate"]["device_group_us"],
                        strict=True,
                    )
                ]
                row[metric]["speedup"] = (
                    arms["reference"]["median_device_us"]
                    / arms["candidate"]["median_device_us"]
                )
                row[metric]["min_round_speedup"] = min(ratios)
        if diagnostic_dir:
            row["profile"] = direct_profile(
                torch,
                {
                    "reference": direct_launcher(data),
                    "candidate": direct_launcher(data, default),
                },
                Path(diagnostic_dir) / (spec["name"] + "-trace"),
            )
            path = Path(diagnostic_dir) / spec["name"]
            path.mkdir(parents=True, exist_ok=False)
            for name, call in (("reference", old), ("candidate", new)):
                stages = {}
                for stage, text in call.compiled.asm.items():
                    if isinstance(text, str):
                        with (path / (name + "." + stage)).open("x") as stream:
                            stream.write(text)
                        stages[stage] = str(path / (name + "." + stage))
                row.setdefault("generated_code", {})[name] = stages
        row["performance_valid"] = (
            mode == "direct-ab" and row["numerical_valid"] and "both_tuned" in row
        )
        print(
            f"direct {spec['name']}: oracle={row['numerical_valid']} "
            f"kernel speedup={row.get('kernel_only', {}).get('speedup')}",
            flush=True,
        )
    valid = all(row["numerical_valid"] for row in rows)
    return dict(
        cases=rows,
        provenance=provenance(SimpleNamespace(module=data.module)),
        arms={"reference": "original_direct", "candidate": "r19_direct_resident"},
        numerical_valid=valid,
        performance_valid=all(row["performance_valid"] for row in rows),
        status="diagnosed"
        if mode == "direct-diagnostic-ab"
        else "passed"
        if valid
        else "failed",
        scope="direct decode score kernel/launcher only; not prefill/full DSA/model/HTTP acceptance",
    )


def supported_routes(storage, requested=None):
    available = ROUTES if storage == "fp8" else ("reference", "mqa2", "mqa4", "mqa8")
    routes = (
        tuple(requested)
        if requested
        else tuple(n for n in available if not n.startswith("mqa"))
    )
    if any(n not in available for n in routes) or len(set(routes)) != len(routes):
        raise ValueError("score route unsupported for this storage or duplicated")
    return routes


@contextmanager
def selector_evidence(adapter, selector):
    """Only the diagnostic pass records real wheel calls; formal performance timing stays uninstrumented."""
    calls = []
    if selector == "per_row":
        yield calls
        return
    from vllm_sail.utils import deep_select

    if not deep_select.is_deep_select_available():
        raise ScoreDependencyBlocked(
            "explicit DeepSelect wheel unavailable or disabled"
        )
    wheel = deep_select._wheel()
    original = wheel.topk

    def traced(logits, *args, **kwargs):
        calls.append(
            {
                "dtype": str(logits.dtype),
                "shape": list(logits.shape),
                "provider": str(wheel.__file__),
            }
        )
        return original(logits, *args, **kwargs)

    wheel.topk = traced
    try:
        yield calls
    finally:
        wheel.topk = original
    if not calls:
        raise AssertionError(
            "explicit DeepSelect never hit the real wheel; passing via per_row is not allowed"
        )


def score_cases(large=False):
    cases = [
        dict(
            name="small_scored",
            query_lens=[4, 4],
            seq_lens=[32, 35],
            regions=5,
            topk=2,
            regions_per_block=2,
            pages=16,
            table_columns=4,
        ),
        dict(name="mixed_empty_spans", query_lens=[3, 0, 11, 5], regions=35, topk=8),
        dict(name="paired_tail", query_lens=[6, 10], regions=35, topk=8, paired=True),
        dict(name="group_strided", query_lens=[9, 7], regions=257, topk=16, groups=3),
        dict(
            name="invalid_pages", query_lens=[5, 7], regions=35, topk=8, invalid="mixed"
        ),
        dict(
            name="all_invalid_pages",
            query_lens=[3, 5],
            regions=35,
            topk=8,
            invalid="all",
        ),
        dict(name="empty_lengths", query_lens=[3, 5], regions=35, topk=8, empty_rows=2),
        dict(name="single_long", query_lens=[129], regions=1024, topk=512),
        dict(
            name="compact_spans",
            query_lens=[273, 0, 259],
            regions=641,
            topk=512,
            groups=3,
            empty_rows=2,
            invalid="mixed",
        ),
        dict(
            name="compact_paired",
            query_lens=[258, 270],
            regions=641,
            topk=512,
            paired=True,
        ),
        dict(
            name="compact_heads16",
            query_lens=[257, 259],
            regions=641,
            topk=512,
            heads=16,
        ),
        dict(
            name="compact_forward",
            query_lens=[256, 256],
            regions=1024,
            topk=512,
            seq_lens=[8192, 8192],
        ),
    ]
    if large:
        cases.append(
            dict(
                name="logged8192_synthetic",
                query_lens=[631, 631] + [630] * 11,
                regions=1024,
                topk=512,
                table_columns=16384,
                pages=43224,
                seq_lens=[8192] * 13,
            )
        )
        cases.append(
            dict(
                name="paired_chunk_boundary",
                query_lens=[4102, 4094],
                regions=514,
                topk=16,
                paired=True,
            )
        )
        for rows in (4096, 8192):
            cases.append(
                dict(
                    name="context64k_q" + str(rows),
                    query_lens=[rows],
                    regions=8192,
                    topk=512,
                    seq_lens=[65536],
                )
            )
        cases.append(
            dict(
                name="context64k_irregular",
                query_lens=[631, 631] + [630] * 11,
                regions=8192,
                topk=512,
                seq_lens=[65536] * 13,
            )
        )
    return [
        {
            "seed": 128,
            "groups": 1,
            "regions_per_block": 8,
            "paired": False,
            "invalid": None,
            "empty_rows": 0,
            "heads": 8,
            **case,
        }
        for case in cases
    ]


@contextmanager
def route(name, config=None):
    """Only flips switches in the acceptance process; legacy MQA still needs the experimental context."""
    if name not in ROUTES:
        raise ValueError("unknown score route")
    modes = {"reference": "v0", "resident": "v1", "compact": "v2"}
    values = {"VLLM_SAIL_STEP5_DSA_SCORES": modes.get(name, "v0")}
    token = _BQ.set(int(name[-1]) if name.startswith("mqa") else None)
    config_token = _COMPACT_CONFIG.set(config or {})
    try:
        with override_step5_environment(values), score._experimental():
            yield
    finally:
        _BQ.reset(token)
        _COMPACT_CONFIG.reset(config_token)


@contextmanager
def experimental_forward(data, name):
    """Only the standalone acceptance instance temporarily adopts the candidate; the instance is restored on exit, never touching production classes or modules."""
    with route(name):
        if not name.startswith("mqa"):
            yield
            return
        impl, adapter = data.impl, data.impl.adapter
        targets = (
            (impl, "_forward_prefill"),
            (adapter, "pack_summaries"),
            (adapter, "logits"),
        )
        saved = [
            (owner, attr, attr in vars(owner), vars(owner).get(attr))
            for owner, attr in targets
        ]
        original_pack, original_logits = adapter.pack_summaries, adapter.logits
        original_forward = impl._forward_prefill
        plan_present = hasattr(data.metadata, score._PLAN_ATTRIBUTE)
        original_plan = getattr(data.metadata, score._PLAN_ATTRIBUTE, None)

        def packed(mean, table, regions):
            return score.pack_or_reference(
                lambda owner, *args: original_pack(*args), adapter, mean, table, regions
            )

        def logits(query, weights, packed, pages, requests, lengths, regions, **kwargs):
            if isinstance(packed, score.RawSummary):
                result = score.raw_logits(
                    packed,
                    adapter,
                    query,
                    weights,
                    lengths,
                    regions,
                    kwargs.get("pairs"),
                    kwargs.get("paired_heads", 0),
                )
                if result is not None:
                    return result
                packed, pages = packed.materialize()
            return original_logits(
                query, weights, packed, pages, requests, lengths, regions, **kwargs
            )

        try:
            score.attach_plan(
                data.metadata, data.metadata.step5_runtime, True, bq=int(name[-1])
            )
            wrapped = score.wrap_forward(
                lambda owner, *args, **kwargs: original_forward(*args, **kwargs),
                SimpleNamespace(is_ppu=lambda: True),
            )
            impl._forward_prefill = MethodType(wrapped, impl)
            adapter.pack_summaries, adapter.logits = packed, logits
            yield
        finally:
            for owner, attr, present, value in saved:
                if present:
                    setattr(owner, attr, value)
                elif attr in vars(owner):
                    delattr(owner, attr)
            if plan_present:
                setattr(data.metadata, score._PLAN_ATTRIBUTE, original_plan)
            elif hasattr(data.metadata, score._PLAN_ATTRIBUTE):
                delattr(data.metadata, score._PLAN_ATTRIBUTE)


@contextmanager
def layer_route(name, budget=1 << 30, *, grouped=False):
    """Production modes, not the legacy experimental context; same-scorer budget A/B."""
    if name not in ("reference", "resident", "compact") or budget < 0:
        raise ValueError("invalid production layer route or budget")
    if grouped and (name != "compact" or budget != 0):
        raise ValueError("grouped A/B requires the compact chunk route")
    mode = (
        "v3"
        if grouped
        else {"reference": "v0", "resident": "v1", "compact": "v2"}[name]
    )
    values = {"VLLM_SAIL_STEP5_DSA_SCORES": mode}
    if not grouped:
        values["VLLM_SAIL_STEP5_DSA_SCORE_MAX_BYTES"] = str(budget)
    with override_step5_environment(values):
        yield


def capture_layer_scores(data):
    """Observe the real outer forward; CPU copies are outside all formal timings."""
    from tests.support.step5_attention_testing import capture_selection

    adapter = data.impl.adapter
    original = adapter.logits
    present, saved = "logits" in vars(adapter), vars(adapter).get("logits")
    scores = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        scores.append(result.detach().cpu())
        return result

    adapter.logits = capture
    try:
        selected, work = capture_selection(data)
    finally:
        if present:
            adapter.logits = saved
        else:
            delattr(adapter, "logits")
    runtime = data.metadata.step5_runtime
    expected = data.case["geometry"]["groups"] * len(runtime[4])
    if len(scores) != expected:
        raise AssertionError(
            "outer forward did not consume every group/chunk score slice"
        )
    return SimpleNamespace(
        scores=scores,
        selected=selected,
        work=work,
        output=data.tensors["output"].detach().cpu().clone(),
        mean=data.impl.summary_cache.mean.detach().cpu().clone(),
        ring=data.impl.csa_ring.rows.detach().cpu().clone(),
    )


def check_layer_scores(data, actual, baseline, oracle_count=26):
    """Every slice and mask is compared; sampled FP32 oracle retains 1e-4 gates."""
    torch, geo = data.torch, data.case["geometry"]
    if len(actual.scores) != len(baseline.scores) or len(actual.selected) != len(
        baseline.selected
    ):
        raise AssertionError("incomplete layer score/selection coverage")
    metrics = [
        error_metrics(torch, a, b)
        for a, b in zip(actual.scores, baseline.scores, strict=True)
    ]
    if any(
        set(a) != set(b)
        for a, b in zip(actual.selected, baseline.selected, strict=True)
    ):
        raise AssertionError("outer forward TopK sets differ")
    torch.testing.assert_close(actual.output, baseline.output, atol=0.025, rtol=0.025)
    torch.testing.assert_close(actual.mean, baseline.mean, atol=0, rtol=0)
    torch.testing.assert_close(actual.ring, baseline.ring, atol=0, rtol=0)
    runtime = data.metadata.step5_runtime
    oracle_data = SimpleNamespace(
        torch=torch,
        mean=actual.mean,
        query=data.tensors["dsa_proxy_query"],
        weights=data.tensors["dsa_proxy_weights"],
        host_table=data.host_tables,
        host_owners=data.host_owners,
        host_positions=data.host_positions,
        spec={
            "shared_mean": True,
            "regions": runtime[2],
            "paired": runtime[8] is not None,
            "heads": geo["indexer_heads"],
            "regions_per_block": geo["block_size"] // 8,
        },
    )
    oracle_metrics = []
    for group in range(geo["groups"]):
        begin = 0
        for index, meta in enumerate(runtime[4]):
            size = meta.current.numel()
            ids = sorted(
                {
                    round(i * (size - 1) / max(min(oracle_count, size) - 1, 1))
                    for i in range(min(oracle_count, size))
                }
            )
            expected = oracle_rows(oracle_data, group, [begin + row for row in ids])
            oracle_metrics.append(
                error_metrics(
                    torch, actual.scores[group * len(runtime[4]) + index][ids], expected
                )
            )
            begin += size
    return {
        "status": "passed",
        "slices": metrics,
        "fp32_oracle": oracle_metrics,
        "same_topk_sets": True,
        "summary_ring_exact": True,
        "score_atol": 1e-4,
        "score_rtol": 1e-4,
    }


def layer_score_replay(data, rebuild=False):
    """Score pipeline ablation from a real forward's shared updated K; no attention or K swapping."""
    impl, metadata = data.impl, data.metadata
    q, w = data.tensors["dsa_proxy_query"], data.tensors["dsa_proxy_weights"]
    runtime, groups = metadata.step5_runtime, data.case["geometry"]["groups"]
    if rebuild:
        # Includes consolidated metadata costs, not the unchanged attention planner.
        setattr(metadata, score._LAYER_ATTRIBUTE, None)
        setattr(metadata, score._PACKED_ATTRIBUTE, None)
    with score.bind_layer(impl, metadata, q, w, groups) as context:
        for group in range(groups):
            query, weights = q[:, group : group + 1], w[:, group : group + 1]
            accepted = context.enter_group(metadata, query, weights)
            if not accepted and score.compact_enabled():
                plan = getattr(metadata, score._PACKED_ATTRIBUTE, None)
                if plan is None:
                    score.attach_packed_plan(
                        metadata,
                        runtime,
                        True,
                        4 if runtime[8] is not None else q.shape[2],
                    )
            with (
                nullcontext()
                if accepted
                else score.bind_packed(
                    metadata, runtime, impl.adapter, query, weights, True
                )
            ):
                packed, pages = impl.adapter.pack_summaries(
                    impl.summary_cache.mean,
                    metadata.block_table[: metadata.num_actual_reqs],
                    runtime[2],
                )
                begin = 0
                for index, meta in enumerate(runtime[4]):
                    end = begin + meta.current.numel()
                    logits = impl.adapter.logits(
                        query[begin:end],
                        weights[begin:end],
                        packed,
                        pages,
                        runtime[0][begin:end],
                        meta.lengths,
                        runtime[2],
                        block_table=meta.block_table,
                        schedule_meta=meta.schedule,
                        pairs=runtime[8][index] if runtime[8] is not None else None,
                        paired_heads=impl.indexer_heads,
                    )
                    impl.adapter.select_regions(
                        logits,
                        meta.current,
                        impl.sparse_topk,
                        topk_lengths=meta.topk_lengths,
                        sort=False,
                    )
                    begin = end


@contextmanager
def capture_score_submissions():
    """Host submission observation is distinct from profiler device kernel hits."""
    from vllm.models.step5.nvidia.ops import sparse_indexer_scores as upstream

    from vllm_sail.models.step5.ops import sparse_attention_opt as kernels

    calls, saved = [], {}

    class LaunchObserver:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def record(*args, **kwargs):
                calls.append((launch, args, kwargs))
                return launch(*args, **kwargs)

            return record

        def __getattr__(self, name):
            return getattr(self.kernel, name)

    targets = (
        (upstream, "_paged_scores_sm90"),
        (kernels, "_resident_scores_sm90"),
        (kernels, "_grouped_scores_sm89"),
    )
    try:
        for module, name in targets:
            original = saved[module, name] = getattr(module, name)
            setattr(module, name, LaunchObserver(original))
        yield calls
    finally:
        for (module, name), original in saved.items():
            setattr(module, name, original)


def layer_score_only_calls(data):
    """Exact kernel replay excludes pack/normalization/select and output allocation."""
    with capture_score_submissions() as calls:
        layer_score_replay(data)
    if not calls:
        raise AssertionError("no actual score launcher captured")

    def replay():
        for function, args, kwargs in calls:
            function(*args, **kwargs)

    return replay, len(calls)


def profile_layer_scores(data, trace, expected):
    """Exactly one actual outer forward per profile, no warmup/replay aggregation."""
    from tests.support.step5_attention_testing import capture_selection, trace_summary

    before = score.stats()
    with capture_score_submissions() as submissions:
        _, work = capture_selection(data, True, trace)
    after = score.stats()
    profile = trace_summary(trace)
    kernels = [
        row
        for row in profile["kernels"]
        if any(
            name in row["name"]
            for name in (
                "_paged_scores_sm90",
                "_resident_scores_sm90",
                "_grouped_scores_sm89",
            )
        )
    ]
    if len(submissions) != expected:
        raise AssertionError(
            f"expected {expected} host score submissions, got {len(submissions)}"
        )
    if len(kernels) != expected:
        raise AssertionError(
            f"one layer forward: expected {expected} device score kernels, got {len(kernels)}"
        )
    delta = {
        key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)
    }
    return {
        "device_score_kernel_hits": len(kernels),
        "score_kernels": kernels,
        "host_submissions": len(submissions),
        "layer_host_submissions": delta.get("layer_launches", 0),
        "slice_consumptions": delta.get("layer_slices", 0),
        "route_stats": delta,
        "selection_sha256": work["selection_sha256"],
        "profile": profile,
        "scope": "one real Step5DSAAttentionImpl.forward; diagnostics only",
    }


def host_geometry(spec):
    """Checks the synthetic sample's request/causal ranges before device initialization."""
    sizes = spec["query_lens"]
    seqs = spec.get("seq_lens", [spec["regions"] * 8 - 3] * len(sizes))
    if (
        len(seqs) != len(sizes)
        or not sizes
        or spec["regions"] <= 0
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in sizes)
        or any(
            not isinstance(seq, int)
            or isinstance(seq, bool)
            or seq < size
            or seq > spec["regions"] * 8
            for size, seq in zip(sizes, seqs, strict=True)
        )
        or not 0 <= spec["empty_rows"] <= sum(sizes)
    ):
        raise ValueError("query/seq/region ranges invalid")
    if spec["paired"] and any(n % 2 for n in sizes):
        raise ValueError("paired queries must not pair across requests")
    starts, owners, positions = [0], [], []
    for owner, size in enumerate(spec["query_lens"]):
        starts.append(starts[-1] + size)
        owners.extend([owner] * size)
        seq = seqs[owner]
        positions.extend(range(seq - size, seq))
    positions[: spec["empty_rows"]] = [-1] * spec["empty_rows"]
    return starts, owners, positions


def prepare_score_case(spec, extension_root=None, storage="fp8", selector="per_row"):
    from tests.support.step5_attention_testing import require_runtime

    starts, owners, positions = host_geometry(spec)
    torch, module = require_runtime("flashinfer", extension_root)
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError(
            "multi-query score numerical and performance acceptance supports SM89 only"
        )
    torch.manual_seed(spec["seed"])
    rows, groups, rpb = starts[-1], spec["groups"], spec["regions_per_block"]
    heads = spec.get("heads", 8)
    columns = math.ceil(spec["regions"] / rpb)
    pages = spec.get("pages", max(1, len(spec["query_lens"]) * columns))
    table = torch.full(
        (len(spec["query_lens"]), spec.get("table_columns", columns) + 3),
        -1,
        dtype=torch.int32,
        device="cuda",
    )[:, :-3]
    host_table = []
    for owner in range(len(spec["query_lens"])):
        ids = [(owner * columns + col) % pages for col in range(columns)]
        if spec["invalid"]:
            ids = (
                [-1 if col % 2 else pages + 7 for col in range(columns)]
                if spec["invalid"] == "all"
                else [
                    -1 if col % 3 == 0 else pages + 7 if col % 3 == 1 else value
                    for col, value in enumerate(ids)
                ]
            )
        host_table.append(ids)
        table[owner, :columns].copy_(
            torch.tensor(ids, device="cuda", dtype=torch.int32)
        )
    query = (torch.randn((rows, groups, heads, 256), device="cuda") * 0.125).to(
        torch.float8_e4m3fn
    )
    values = (torch.randn((pages, groups, rpb, 256), device="cuda") * 0.125).to(
        torch.float8_e4m3fn
    )
    if storage == "fp16":
        query, values = query.half(), values.half()
    elif storage == "fp8":
        values = values.view(torch.uint8)
    else:
        raise ValueError("storage supports fp8/fp16 only")
    # Padded heads are explicitly non-zero; the paired baseline still reads only the first four heads.
    weights = (
        torch.randn((rows, groups, heads), device="cuda", dtype=torch.float32) * 0.2
    )
    adapter = module._shared_adapter(torch.device("cuda:0"))
    adapter._topk_backend = selector
    if selector == "deep_select":
        from vllm_sail.utils.deep_select import is_deep_select_available

        if not is_deep_select_available():
            raise ScoreDependencyBlocked(
                "explicit DeepSelect wheel unavailable or disabled"
            )
    metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        num_actual_reqs=len(spec["query_lens"]),
        slot_mapping=torch.empty(rows, device="cuda", dtype=torch.int64),
    )
    return SimpleNamespace(
        torch=torch,
        module=module,
        adapter=adapter,
        spec=spec,
        query=query,
        weights=weights,
        mean=values,
        table=table,
        metadata=metadata,
        host_table=host_table,
        starts=starts,
        host_owners=owners,
        host_positions=positions,
    )


def build_metadata(data):
    """The primary metric includes metadata allocation, upload, and paired construction; no baseline work is removed."""
    torch, spec = data.torch, data.spec
    owners = torch.tensor(data.host_owners, device="cuda", dtype=torch.int64)
    positions = torch.tensor(data.host_positions, device="cuda", dtype=torch.int64)
    indexer = data.adapter.prefill_metadata(
        owners, positions, spec["regions"], spec["topk"]
    )
    pairs = data.adapter.prefill_pair_inputs(indexer) if spec["paired"] else None
    runtime = (
        owners,
        positions,
        spec["regions"],
        None,
        indexer,
        None,
        None,
        None,
        pairs,
    )
    data.metadata.step5_runtime = runtime
    if score.compact_enabled():
        score.attach_packed_plan(
            data.metadata,
            runtime,
            True,
            4 if spec["paired"] else data.query.shape[2],
            group_rows=(_COMPACT_CONFIG.get() or {}).get("group_rows"),
        )
    if _BQ.get() is not None:
        score.attach_plan(data.metadata, runtime, True, bq=_BQ.get())
    else:
        setattr(data.metadata, score._PLAN_ATTRIBUTE, None)
    return runtime


class ChunkCapture:
    """Persists per chunk and loads only the chunk being compared, avoiding caching all large matrices."""

    def __init__(self, torch, directory):
        from pathlib import Path

        self.torch, self.directory, self.count = torch, Path(directory), 0
        self.directory.mkdir(parents=True, exist_ok=False)

    def append(self, value):
        self.torch.save(value, self.directory / (str(self.count) + ".pt"))
        self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.torch.load(self.directory / (str(index) + ".pt"), weights_only=True)


def capture_target(data, label):
    root = getattr(data, "capture_root", None)
    return root / label if root is not None else True


def pipeline(data, rebuild=False, capture=False):
    """No summary update and no attention; covers every group/chunk and the real selector."""
    runtime = build_metadata(data) if rebuild else data.metadata.step5_runtime
    if capture and capture is not True:
        outputs = ChunkCapture(data.torch, capture / "scores")
        selections = ChunkCapture(data.torch, capture / "selection")
    else:
        outputs, selections = [], []
    last = None
    for group in range(data.spec["groups"]):
        q, w, mean = (
            data.query[:, group : group + 1],
            data.weights[:, group : group + 1],
            data.mean[:, group],
        )
        with (
            score.bind(
                data.metadata,
                runtime,
                data.adapter,
                q,
                w,
                mean,
                data.table,
                4 if data.spec["paired"] else q.shape[2],
                True,
            ),
            score.bind_packed(data.metadata, runtime, data.adapter, q, w, True),
        ):
            packed, pages = (
                score.pack_or_reference(
                    type(data.adapter).pack_summaries,
                    data.adapter,
                    mean,
                    data.table,
                    data.spec["regions"],
                )
                if _BQ.get() is not None
                else data.adapter.pack_summaries(mean, data.table, data.spec["regions"])
            )
            begin = 0
            for index, meta in enumerate(runtime[4]):
                end = begin + meta.current.numel()
                scores = None
                if isinstance(packed, score.RawSummary):
                    scores = score.raw_logits(
                        packed,
                        data.adapter,
                        q[begin:end],
                        w[begin:end],
                        meta.lengths,
                        data.spec["regions"],
                        runtime[8][index] if runtime[8] is not None else None,
                        4 if data.spec["paired"] else 8,
                    )
                    if scores is None:
                        packed, pages = packed.materialize()
                if scores is None and score.compact_enabled():
                    scores = score.packed_logits(
                        data.adapter,
                        q[begin:end],
                        w[begin:end],
                        packed,
                        pages,
                        runtime[0][begin:end],
                        meta.lengths,
                        data.spec["regions"],
                        pairs=runtime[8][index] if runtime[8] is not None else None,
                        paired_heads=4 if data.spec["paired"] else 0,
                        block_table=meta.block_table,
                        config={
                            k: v
                            for k, v in (_COMPACT_CONFIG.get() or {}).items()
                            if k != "group_rows"
                        },
                    )
                if scores is None:
                    with score.bind_packed(None, None, None, None, None, False):
                        scores = data.adapter.logits(
                            q[begin:end],
                            w[begin:end],
                            packed,
                            pages,
                            runtime[0][begin:end],
                            meta.lengths,
                            data.spec["regions"],
                            block_table=meta.block_table,
                            schedule_meta=meta.schedule,
                            pairs=runtime[8][index] if runtime[8] is not None else None,
                            paired_heads=4 if data.spec["paired"] else q.shape[2],
                        )
                last = data.adapter.select_regions(
                    scores,
                    meta.current,
                    data.spec["topk"],
                    topk_lengths=meta.topk_lengths,
                    sort=False,
                )
                if capture:
                    outputs.append(scores.detach().cpu())
                    selections.append(last.detach().cpu())
                del scores
                begin = end
    return (outputs, selections) if capture else last


class ScoreNumericalError(AssertionError):
    """Keeps the error under the fixed tolerance; failed results must not feed the performance gate."""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def error_metrics(torch, actual, expected, atol=1e-4, rtol=1e-4):
    if not torch.equal(torch.isneginf(actual), torch.isneginf(expected)):
        raise AssertionError("score -inf masks are not equivalent")
    mask = torch.isfinite(expected)
    if not bool(torch.isfinite(actual[mask]).all()):
        raise AssertionError("NaN/+inf in valid scores")
    delta = (actual[mask] - expected[mask]).abs()
    relative = delta / expected[mask].abs().clamp_min(1e-12)
    result = {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "max_rel": float(relative.max()) if delta.numel() else 0.0,
        "mean_rel": float(relative.mean()) if delta.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
    }
    result["outside_tolerance"] = int(
        (delta > atol + rtol * expected[mask].abs()).sum()
    )
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise ScoreNumericalError(str(exc), result) from exc
    return result


def oracle_rows(data, group, row_ids):
    """CPU FP32 dot oracle; the four-chain FMA uses FP64 intermediates to emulate one FP32 rounding."""
    torch, spec = data.torch, data.spec
    mean = (data.mean if spec.get("shared_mean") else data.mean[:, group]).cpu()
    if mean.dtype == torch.uint8:
        mean = mean.view(torch.float8_e4m3fn)
    mean = mean.float()
    query = data.query[:, group].float().cpu()[row_ids]
    weights = data.weights[row_ids, group].cpu()
    regions, heads = spec["regions"], 4 if spec["paired"] else spec.get("heads", 8)
    output = torch.full((len(row_ids), regions), -float("inf"))
    for index, row in enumerate(row_ids):
        owner = data.host_owners[row]
        key = torch.zeros((regions, 256))
        for region in range(regions):
            physical = data.host_table[owner][region // spec["regions_per_block"]]
            if 0 <= physical < mean.shape[0]:
                key[region] = mean[physical, region % spec["regions_per_block"]]
        dot = (key @ query[index, :heads].T).clamp_min(0)
        sums = torch.zeros((regions, 4))
        for head in range(heads):
            lane = head % 4
            sums[:, lane] = (
                dot[:, head].double() * weights[index, head].double()
                + sums[:, lane].double()
            ).float()
        result = ((sums[:, 0] + sums[:, 1]) + sums[:, 2]) + sums[:, 3]
        source = row // 2 * 2 + 1 if spec["paired"] else row
        length = data.host_positions[source] // 8 + 1
        output[index, :length] = result[:length]
    return output


def compare_selection(
    torch,
    baseline,
    candidate,
    ref_scores,
    candidate_scores,
    topk,
    currents,
    oracle=None,
):
    """Compares sets when there are no cutoff ties; near-cutoff cases are recorded and the candidate must stay within the oracle tolerance boundary."""
    ties, changed = [], []
    for row, current in enumerate(currents):
        expected = {int(x) for x in baseline[row] if x >= 0}
        observed = {int(x) for x in candidate[row] if x >= 0}
        if current >= 0 and current not in observed:
            raise AssertionError("selector dropped the current region")
        if any(x > current or x < 0 for x in observed):
            raise AssertionError("selector output invalid or from future regions")
        if expected == observed:
            continue
        changed.append(row)
        if oracle is None:
            raise AssertionError(
                "selection boundary changes require an additional FP32 oracle check"
            )
        oracle_score = oracle(row)
        error_metrics(torch, candidate_scores[row], oracle_score)
        error_metrics(torch, ref_scores[row], oracle_score)
        values = oracle_score[: max(current, 0)]
        k = min(topk, values.numel())
        if k == 0:
            raise AssertionError("empty-history selection sets disagree")
        ordered = values.sort(descending=True).values
        cutoff = float(ordered[k - 1])
        margin = float(ordered[k - 1] - ordered[k]) if k < values.numel() else None
        tol = 1e-4 + abs(cutoff) * 1e-4
        different = (expected ^ observed) - {current}
        if (margin is not None and margin > 2 * tol) or any(
            abs(float(values[x]) - cutoff) > 2 * tol for x in different
        ):
            raise AssertionError("top-k set changed far from the cutoff")
        # Candidate scores were already compared item by item against the FP32 oracle; only boundary contention within the same tolerance is allowed.
        ties.append(
            {
                "row": row,
                "cutoff": cutoff,
                "margin": margin,
                "different_regions": sorted(different),
                "oracle_checked": True,
            }
        )
    return {"same_sets": not changed, "changed_rows": changed, "boundary_cases": ties}


def numerical(data, name, baseline, sample_rows=26, config=None):
    torch = data.torch
    with (
        route(name, config),
        selector_evidence(
            getattr(data, "adapter", None),
            getattr(getattr(data, "adapter", None), "_topk_backend", "per_row"),
        ) as provider_calls,
    ):
        build_metadata(data)
        score.stats(True)
        actual, selected = pipeline(
            data, capture=capture_target(data, name + "-actual")
        )
        route_stats = score.stats()
        repeated, repeated_selected = pipeline(
            data, capture=capture_target(data, name + "-repeated")
        )
    if name == "compact":
        plan = getattr(data.metadata, score._PACKED_ATTRIBUTE, None)
        expected = (
            0
            if plan is None
            else data.spec["groups"]
            * sum(
                end - begin >= 256 and any(n == plan.group_rows for _, n, _ in spans)
                for begin, end, spans in plan.host
            )
        )
        if route_stats.get("compact_chunks", 0) != expected:
            raise AssertionError("grouped chunk hits incomplete: " + str(route_stats))
    if name.startswith("mqa") and route_stats.get("mqa_chunks", 0) != len(actual):
        raise AssertionError("MQA did not cover every group/chunk: " + str(route_stats))
    checks, failures = [], []

    def checked_metric(actual, expected, group, chunk, comparison):
        try:
            return error_metrics(torch, actual, expected)
        except ScoreNumericalError as exc:
            failures.append(
                {
                    "group": group,
                    "chunk": chunk,
                    "comparison": comparison,
                    "error": str(exc),
                    "metrics": exc.metrics,
                }
            )
            return exc.metrics

    chunk_count = len(data.metadata.step5_runtime[4])
    for group in range(data.spec["groups"]):
        begin = 0
        for index in range(chunk_count):
            slot = group * chunk_count + index
            result, ref = actual[slot], baseline[0][slot]
            metric = checked_metric(result, ref, group, index, "packed_baseline")
            torch.testing.assert_close(repeated[slot], result, atol=0, rtol=0)
            assert torch.equal(selected[slot], repeated_selected[slot]), (
                "same-route selector is nondeterministic"
            )
            ids = sorted(
                {
                    round(
                        i
                        * (result.shape[0] - 1)
                        / max(1, min(sample_rows, result.shape[0]) - 1)
                    )
                    for i in range(min(sample_rows, result.shape[0]))
                }
            )
            oracle = oracle_rows(data, group, [begin + row for row in ids])
            oracle_metric = checked_metric(
                result[ids], oracle, group, index, "fp32_oracle"
            )
            ref_oracle_metric = checked_metric(
                ref[ids], oracle, group, index, "packed_fp32_oracle"
            )
            currents = [
                pos // 8 for pos in data.host_positions[begin : begin + result.shape[0]]
            ]
            selection = compare_selection(
                torch,
                baseline[1][slot],
                selected[slot],
                ref,
                result,
                data.spec["topk"],
                currents,
                lambda row, group=group, begin=begin: oracle_rows(
                    data, group, [begin + row]
                )[0],
            )
            checks.append(
                {
                    "group": group,
                    "chunk": index,
                    "packed_baseline": metric,
                    "fp32_oracle": oracle_metric,
                    "packed_fp32_oracle": ref_oracle_metric,
                    "oracle_rows": ids,
                    "selection": selection,
                }
            )
            begin += result.shape[0]
    return {
        "status": "failed" if failures else "passed",
        "checks": checks,
        "failures": failures,
        "route_stats": route_stats,
        "deterministic_scores_and_selection": True,
        "storage": str(data.query.dtype),
        "selector_provider_calls": provider_calls,
    }


def measure(torch, call, reset=lambda: None, groups=5, iterations=5):
    """Synchronized wall time excludes reset; Events wrap only the entry, and the formal pass stays uninstrumented."""
    reject_instrumentation()
    if groups <= 0 or iterations <= 0:
        raise ValueError("groups/iterations must be positive")
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    device_samples, wall_samples = [], []
    for _ in range(groups):
        wall, durations = [], []
        for start, end in events:
            reset()
            torch.cuda.synchronize()
            began = time.perf_counter()
            start.record()
            call()
            end.record()
            end.synchronize()
            wall.append((time.perf_counter() - began) * 1e6)
            durations.append(start.elapsed_time(end) * 1000)
        wall_samples.append(statistics.median(wall))
        device_samples.append(statistics.median(durations))
    median = statistics.median(device_samples)
    return {
        "median_device_us": median,
        "median_sync_wall_us": statistics.median(wall_samples),
        "device_group_us": device_samples,
        "sync_wall_group_us": wall_samples,
        "range_pct": (max(device_samples) - min(device_samples)) / median * 100,
        "groups": groups,
        "iterations": iterations,
        "reset_oracle_jit_included": False,
        "wall_includes_events_and_sync": True,
    }


def diagnose_score(data, name, directory, config=None):
    """A standalone profiler pass records real launches/allocations; not used for formal performance metrics."""
    from pathlib import Path

    from tests.support.step5_attention_testing import trace_summary

    torch = data.torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / (name + ".trace.json")
    if trace.exists():
        raise FileExistsError(trace)
    with route(name, config):
        score.stats(True)
        torch.cuda.reset_peak_memory_stats()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
        ) as prof:
            with torch.profiler.record_function(
                "step5_stage:score_pipeline_with_metadata"
            ):
                pipeline(data, rebuild=True)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace))
        launches = trace_summary(trace)
        runtime = data.metadata.step5_runtime
        tensors = [runtime[0], runtime[1]]
        tensors.extend(
            t for meta in runtime[4] for t in meta if hasattr(t, "untyped_storage")
        )
        if runtime[8] is not None:
            tensors.extend(
                t for pair in runtime[8] for t in pair if hasattr(t, "untyped_storage")
            )
        plan = getattr(data.metadata, score._PLAN_ATTRIBUTE, None)
        if plan is not None:
            tensors.append(plan.tiles)
        packed_plan = getattr(data.metadata, score._PACKED_ATTRIBUTE, None)
        if name == "compact" and packed_plan is not None:
            tensors.append(packed_plan.spans)
        storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in tensors
        }
        return {
            "route_stats": score.stats(),
            "launches": launches,
            "metadata_unique_storage_bytes": sum(storages.values()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "profiler_positive_self_device_alloc_bytes": sum(
                max(0, getattr(event, "self_device_memory_usage", 0))
                for event in prof.key_averages()
            ),
            "timing_scope": "standalone diagnostics including metadata; not evidence of performance gains",
        }


def score_ab(
    data, groups=5, iterations=5, diagnostic_dir=None, requested=None, config=None
):
    """A/B and B/A; multiple rounds report the metadata-inclusive primary metric and the plan-reuse steady state."""
    results = []
    routes = supported_routes(
        "fp8" if str(data.query.dtype).endswith("float8_e4m3fn") else "fp16", requested
    )
    for order in (routes, tuple(reversed(routes))):
        measured = {}
        for name in order:
            with route(name, config):
                build_metadata(data)
                for _ in range(3):
                    pipeline(data, rebuild=True)
                data.torch.cuda.synchronize()
                inclusive = measure(
                    data.torch,
                    lambda: pipeline(data, rebuild=True),
                    groups=groups,
                    iterations=iterations,
                )
                steady = measure(
                    data.torch,
                    lambda: pipeline(data),
                    groups=groups,
                    iterations=iterations,
                )
                score.stats(True)
                data.torch.cuda.reset_peak_memory_stats()
                pipeline(data)
                data.torch.cuda.synchronize()
                diagnostics = {
                    "route_stats": score.stats(),
                    "peak_allocated_bytes": data.torch.cuda.max_memory_allocated(),
                    "packing_calls": data.spec["groups"],
                    "packed_buffers": 0
                    if name.startswith("mqa")
                    else data.spec["groups"],
                    "metadata_chunks": len(data.metadata.step5_runtime[4]),
                    "paired_metadata_retained": data.spec["paired"],
                }
            measured[name] = {
                "score_pipeline_with_metadata": inclusive,
                "steady_state": steady,
                "diagnostics": diagnostics,
            }
        results.append({"order": list(order), "routes": measured})
    diagnostics = (
        {name: diagnose_score(data, name, diagnostic_dir, config) for name in routes}
        if diagnostic_dir
        else {}
    )
    return {
        "rounds": results,
        "profiler_diagnostics": diagnostics,
        "scope": "host grouping/metadata upload/paired/packing/dispatch/all group and chunk scores/select; excludes summary updates and attention",
        "plan_cost_frequency": "once per step, not counted again per layer",
    }


def packed_cpu_oracle(
    torch, query, weights, key, scales, owners, lengths, regions, pages, heads, next_n
):
    """Interprets quantized values against the same packed pages; does not emulate unknown device MMA rounding."""
    cpu_owners = owners.cpu()
    keys = key.cpu().float().reshape(-1, pages * 32, 256)[cpu_owners, :regions]
    cpu_query = query.cpu().float()[:, 0, :heads]
    dot = torch.einsum("rhd,rkd->rhk", cpu_query, keys).clamp_min(0)
    result = (dot * weights.cpu()[:, 0, :heads, None]).sum(1)
    result *= scales.cpu().reshape(-1, pages * 32)[cpu_owners, :regions]
    result.masked_fill_(
        torch.arange(regions)[None] >= lengths.cpu().repeat_interleave(next_n)[:, None],
        -float("inf"),
    )
    return result


def packed_mma_probe(torch, selected_query, selected_key):
    """Standalone dot probe at failed points; copies diagnostic inputs only, never modifies the scorer or acceptance oracle."""
    import hashlib

    from vllm.triton_utils import tl, triton

    @triton.jit
    def probe(Q, K, OUT, HEADS: tl.constexpr, BLOCK: tl.constexpr, HALF: tl.constexpr):
        point = tl.program_id(0)
        part = tl.program_id(1)
        h = tl.arange(0, 16)
        d = part * BLOCK + tl.arange(0, BLOCK)
        n = tl.arange(0, 16)
        q = tl.load(
            Q.to(tl.pointer_type(tl.uint8))
            + point * HEADS * 256
            + h[:, None] * 256
            + d[None, :],
            h[:, None] < HEADS,
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        k = tl.load(
            K.to(tl.pointer_type(tl.uint8)) + point * 256 + d[:, None] + n[None, :] * 0,
        ).to(tl.float8e4nv, bitcast=True)
        if HALF:
            q = q.to(tl.float16)
            k = k.to(tl.float16)
        dot = tl.dot(q, k, out_dtype=tl.float32, max_num_imprecise_acc=0)
        tl.store(
            OUT
            + point * (256 // BLOCK) * 16 * 16
            + part * 16 * 16
            + h[:, None] * 16
            + n[None, :],
            dot,
        )

    points, heads = selected_query.shape[:2]
    query = selected_query.to(torch.float8_e4m3fn).to("cuda").contiguous()
    key = selected_key.to(torch.float8_e4m3fn).to("cuda").contiguous()
    torch.testing.assert_close(query.cpu().double(), selected_query, atol=0, rtol=0)
    torch.testing.assert_close(key.cpu().double(), selected_key, atol=0, rtol=0)
    outputs, compiled_code = {}, {}
    products = selected_query * selected_key[:, None]
    exact = products.sum(-1)
    for block, half in (
        (32, False),
        (64, False),
        (128, False),
        (256, False),
        (256, True),
    ):
        label = ("fp16" if half else "fp8") + "_k" + str(block)
        out = torch.empty(
            (points, 256 // block, 16, 16), device="cuda", dtype=torch.float32
        )
        compiled = probe[(points, 256 // block)](
            query, key, out, heads, block, half, num_warps=4
        )
        partials = out.cpu()[:, :, :heads, 0].double()
        exact_partials = products.reshape(points, heads, -1, block).sum(-1)
        outputs[label] = {
            "partial_dot": partials.tolist(),
            "sum_dot": partials.sum(1).tolist(),
            "max_partial_abs_error_fp64": float(
                (partials.transpose(1, 2) - exact_partials).abs().max()
            ),
            "max_sum_abs_error_fp64": float((partials.sum(1) - exact).abs().max()),
            "max_cpu_fp32_sum_abs_error_fp64": float(
                (partials.float().sum(1).double() - exact).abs().max()
            ),
        }
        assembly = compiled.asm.get("llir", "")
        compiled_code[label] = {
            "llir_sha256": hashlib.sha256(assembly.encode()).hexdigest(),
            "mma_instructions": [
                line.strip() for line in assembly.splitlines() if "mma" in line.lower()
            ][:16],
        }
    return {
        "outputs": outputs,
        "compiled_code": compiled_code,
        "scope": "standalone failed-point probe with partial sums aggregated on CPU; not full scorer acceptance",
    }


def packed_dot_model_summary(points):
    """Distinguishes fitting resident from approaching the oracle; a failed-point subset cannot serve as a numerical gate."""
    models = {}
    for point in points:
        oracle = point["fp32_oracle_score"]
        fp64 = point["models"]["fp64"]["relu_dot"]
        resident = point["resident_relu_dot"]
        for name, prediction in point["models"].items():
            metric = models.setdefault(
                name,
                {
                    "max_relu_dot_abs_error_fp64": 0.0,
                    "max_relu_dot_abs_error_resident": 0.0,
                    "max_score_abs_error_fp32_oracle": 0.0,
                    "scores_outside_original_tolerance": 0,
                },
            )
            for reference, field in (
                (fp64, "max_relu_dot_abs_error_fp64"),
                (resident, "max_relu_dot_abs_error_resident"),
            ):
                delta = max(
                    abs(a - b)
                    for a, b in zip(prediction["relu_dot"], reference, strict=True)
                )
                metric[field] = max(metric[field], delta)
            delta = abs(prediction["score"] - oracle)
            field = "max_score_abs_error_fp32_oracle"
            metric[field] = max(metric[field], delta)
            metric["scores_outside_original_tolerance"] += int(
                delta > 1e-4 + 1e-4 * abs(oracle)
            )
    return {
        "sampled_failed_points": len(points),
        "models": models,
        "scope": "failed-point subset and diagnostic reduction, not the full scorer or a numerical gate",
    }


def packed_dot_diagnostic(
    torch,
    kernels,
    query,
    weights,
    key,
    packed,
    scales,
    owners,
    table,
    lengths,
    regions,
    pages,
    heads,
    next_n,
    actual,
    expected,
):
    """Diagnoses per-head dots and product rounding models at failed points only; never changes the acceptance oracle."""
    actual = actual.cpu()
    finite = torch.isfinite(expected)
    outside = finite & ((actual - expected).abs() > 1e-4 + 1e-4 * expected.abs())
    locations = outside.nonzero()[:16]
    if not locations.numel():
        return {"failed_points": [], "scope": "diagnostics, not a numerical gate"}
    cpu_query = query.cpu().float()[:, 0, :heads]
    cpu_key = key.cpu().float().reshape(-1, pages * 32, 256)
    cpu_weights = weights.cpu()[:, 0, :heads]
    cpu_scales = scales.cpu().reshape(-1, pages * 32)
    cpu_owners = owners.cpu()
    rows, columns = locations.unbind(1)
    selected_query = cpu_query[rows].double()
    selected_key = cpu_key[cpu_owners[rows], columns].double()
    products = selected_query * selected_key[:, None]
    exact = products.sum(-1)
    probe = packed_mma_probe(torch, selected_query, selected_key)
    half_products = products.half().double()
    models = {
        "fp64": exact,
        "fp32_products": products.float().sum(-1).double(),
        "fp16_products": half_products.sum(-1),
        "fp16_product_ftz": torch.where(
            half_products.abs() < 2**-14, 0.0, half_products
        ).sum(-1),
        "fp8_input_ftz": (
            torch.where(selected_query.abs() < 2**-6, 0.0, selected_query)
            * torch.where(
                selected_key[:, None].abs() < 2**-6, 0.0, selected_key[:, None]
            )
        ).sum(-1),
    }
    for block in (16, 32, 64, 128):
        partials = products.reshape(*products.shape[:2], -1, block).sum(-1)
        accum = torch.zeros_like(exact)
        for part in partials.unbind(-1):
            accum = (accum + part).half().double()
        models["fp16_accum_k" + str(block)] = accum
        models["fp16_partials_k" + str(block)] = partials.half().double().sum(-1)
    for label, output in probe["outputs"].items():
        models["probe_" + label] = torch.tensor(output["sum_dot"], dtype=torch.float64)
    resident_heads = []
    effective_query = query[:, 0, :heads].unflatten(0, (-1, next_n))
    for head in range(heads):
        one_head = torch.zeros_like(weights[:, 0, :heads])
        one_head[:, head] = 1.0
        result = kernels.paged_mqa_logits_sm90(
            effective_query, packed, one_head, table, lengths, regions
        )
        resident_heads.append(result.cpu()[rows, columns])
    observed = torch.stack(resident_heads, 1).double()
    selected_scales = cpu_scales[cpu_owners[rows], columns].double()
    observed /= selected_scales[:, None]
    points = []
    for index, (row, column) in enumerate(locations.tolist()):
        predictions = {}
        for name, dot in models.items():
            predictions[name] = {
                "relu_dot": dot[index].clamp_min(0).tolist(),
                "max_relu_dot_error": float(
                    (dot[index].clamp_min(0) - observed[index]).abs().max()
                ),
                "score": float(
                    (dot[index].clamp_min(0) * cpu_weights[row].double()).sum()
                    * selected_scales[index]
                ),
            }
        points.append(
            {
                "row": row,
                "region": column,
                "actual_score": float(actual[row, column]),
                "fp32_oracle_score": float(expected[row, column]),
                "resident_relu_dot": observed[index].tolist(),
                "models": predictions,
            }
        )
    return {
        "failed_points": points,
        "mma_probe": probe,
        "model_summary": packed_dot_model_summary(points),
        "scope": "diagnostic models, not a replacement for the original CPU FP32 oracle or fixed tolerance",
    }


def packed_kernel_configs(heads, tune=False, overrides=None):
    """Overrides defaults or filters the offline sweep; rejects invalid parameters before loading device dependencies."""
    from itertools import product

    if type(heads) is not int or heads not in (4, 8, 16):
        raise ValueError("unsupported grouped heads")
    knobs = {
        "group_rows": (4, 8, 16),
        "tile": (32, 64, 128),
        "grid_cap": (8, 16, 32),
        "warps": (2, 4),
    }
    overrides = dict(overrides or {})
    for name, value in overrides.items():
        if name not in knobs or type(value) is not int or value not in knobs[name]:
            raise ValueError("unsupported grouped score configuration: " + name)
    if not tune:
        return [dict(group_rows=64 // heads, tile=32, grid_cap=32, warps=2) | overrides]
    return [
        dict(zip(knobs, values, strict=True))
        for values in product(
            *(
                knobs[name] if name not in overrides else (overrides[name],)
                for name in knobs
            )
        )
    ]


def packed_generated_code(
    kernels,
    query,
    weights,
    packed,
    lengths,
    table,
    spans,
    regions,
    pages,
    heads,
    next_n,
    config,
    reference,
    actual,
    directory,
):
    """A standalone diagnostic pass exports the actual launcher's generated code; not a timing or accuracy conclusion."""
    import hashlib
    from pathlib import Path

    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    effective_query = query[:, 0, :heads].unflatten(0, (-1, next_n))
    effective_weights = weights[:, 0, :heads]
    reference, actual = torch.empty_like(reference), torch.empty_like(actual)
    tile = kernels._DEFAULT_TILE
    rows = query.shape[0]
    workers = kernels._resident_workers(rows, math.ceil(regions / tile))
    resident = kernels._resident_scores_sm90[(rows * workers,)](
        effective_query,
        packed,
        effective_weights,
        lengths,
        table,
        reference,
        *effective_query.stride()[:3],
        *effective_weights.stride(),
        packed.stride(0),
        *table.stride(),
        lengths.stride(0),
        regions,
        next_n,
        heads,
        32,
        tile,
        workers,
        num_warps=kernels._DEFAULT_WARPS,
    )
    grid = min(math.ceil(regions / config["tile"]), config["grid_cap"])
    grouped = kernels._grouped_scores_sm89[(spans.shape[0], grid)](
        query,
        packed,
        weights,
        lengths,
        spans,
        actual,
        query.stride(0),
        query.stride(2),
        weights.stride(0),
        weights.stride(2),
        packed.stride(0),
        lengths.stride(0),
        regions,
        pages,
        heads,
        next_n,
        config["group_rows"],
        config["tile"],
        grid,
        num_warps=config["warps"],
    )
    outputs = {}
    for name, compiled in (("resident", resident), ("grouped", grouped)):
        stages = {}
        for stage, assembly in compiled.asm.items():
            if not isinstance(assembly, str):
                continue
            path = directory / (name + "." + stage)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(assembly)
            stages[stage] = {
                "path": str(path),
                "sha256": hashlib.sha256(assembly.encode()).hexdigest(),
                "dot_instructions": [
                    line.strip()
                    for line in assembly.splitlines()
                    if any(word in line.lower() for word in ("mma", "dot", "wmma"))
                ][:32],
            }
        outputs[name] = stages
    return {
        "stages": outputs,
        "scope": "generated-code diagnostics, not performance or numerical acceptance",
    }


def packed_kernel_matrix(
    extension_root=None,
    selector="per_row",
    tune=False,
    groups=5,
    iterations=5,
    diagnostic_dir=None,
    *,
    overrides=None,
):
    """Low-level non-unit scales/tail groups/negative weights/updated K/graph; independent of the production eligibility gate."""
    from itertools import product

    from tests.support.step5_attention_testing import require_runtime

    configs = {
        heads: packed_kernel_configs(heads, tune, overrides) for heads in (4, 8, 16)
    }
    torch, module = require_runtime("flashinfer", extension_root)
    from vllm_sail.models.step5.ops import sparse_attention_opt as kernels

    if torch.cuda.get_device_capability(0) != (8, 9):
        raise ScoreDependencyBlocked("grouped scorer requires SM89")
    adapter = module._shared_adapter(torch.device("cuda:0"))
    adapter._topk_backend = selector
    torch.manual_seed(128)
    checks = []
    patterns = ("random",) if tune else ("random", "ties", "near_ties")
    for (heads, next_n), pattern in product(((4, 2), (8, 1), (16, 1)), patterns):
        sizes, regions, pages = (18, 0, 46), 641, 24
        rows, physical = sum(sizes), len(sizes) * pages
        starts = [0, 18, 18, rows]
        owners = torch.tensor([0] * 18 + [2] * 46, dtype=torch.int64, device="cuda")
        # The group slice keeps a non-contiguous row stride; paired padded heads are non-zero.
        query = (torch.randn((rows, 3, max(heads, 8), 256), device="cuda") * 0.125).to(
            torch.float8_e4m3fn
        )[:, 1:2]
        weights = (torch.randn((rows, 3, max(heads, 8)), device="cuda") * 0.2)[:, 1:2]
        packed = torch.empty((physical, 32 * 260), dtype=torch.uint8, device="cuda")
        key = (torch.randn((physical, 32, 256), device="cuda") * 0.125).to(
            torch.float8_e4m3fn
        )
        if pattern != "random":
            query.copy_(
                torch.full(query.shape, 0.125, device="cuda").to(torch.float8_e4m3fn)
            )
            key.copy_(
                torch.full(key.shape, 0.125, device="cuda").to(torch.float8_e4m3fn)
            )
            weights.fill_(0.2)
            weights[..., 0] = -0.1
        packed[:, :8192].copy_(key.view(torch.uint8).reshape(physical, -1))
        scales = packed[:, 8192:].view(torch.float32)
        scales.copy_(torch.rand_like(scales) + 0.25)
        if pattern != "random":
            scales.fill_(0.75)
            if pattern == "near_ties":
                scales[:, 1::2] += 1e-5
        packed = packed.view(physical, 32, 1, 260)
        lengths = torch.tensor(
            [0, 1, 31, 32, 33, 640, 641] * ((rows // next_n + 6) // 7),
            device="cuda",
            dtype=torch.int32,
        )[: rows // next_n].contiguous()
        table = (
            owners[::next_n, None] * pages + torch.arange(pages, device="cuda")
        ).int()
        effective_query = query[:, 0, :heads].unflatten(0, (-1, next_n))
        effective_weights = weights[:, 0, :heads]
        resident = partial(
            kernels.paged_mqa_logits_sm90,
            effective_query,
            packed,
            effective_weights,
            table,
            lengths,
            regions,
        )
        oracle = partial(
            packed_cpu_oracle,
            torch,
            query,
            weights,
            key,
            scales,
            owners,
            lengths,
            regions,
            pages,
            heads,
            next_n,
        )
        reference, expected = resident(), oracle()

        def recorded_metric(actual, expected):
            try:
                return {"status": "passed", **error_metrics(torch, actual, expected)}
            except ScoreNumericalError as exc:
                return {"status": "failed", "error": str(exc), **exc.metrics}

        resident_oracle = recorded_metric(reference.cpu(), expected)
        timings = []
        for config in configs[heads]:
            host = score.packed_spans(starts, [rows], config["group_rows"])[0][2]
            spans = torch.tensor(host, dtype=torch.int32, device="cuda")
            candidate = partial(
                kernels.grouped_paged_logits,
                query,
                weights,
                packed,
                lengths,
                spans,
                regions,
                pages,
                heads,
                next_n,
                **config,
            )
            actual = candidate()
            comparison = recorded_metric(actual, reference)
            oracle_metric = recorded_metric(actual.cpu(), expected)
            torch.testing.assert_close(candidate(), actual, atol=0, rtol=0)
            # The graph captures score only, without metadata upload; the same addresses are reused after K updates.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = candidate()
            graph.replay()
            torch.testing.assert_close(replayed, actual, atol=0, rtol=0)
            saved_key = key.clone()
            key.copy_((-key.float()).to(torch.float8_e4m3fn))
            packed.view(physical, -1)[:, :8192].copy_(
                key.view(torch.uint8).reshape(physical, -1)
            )
            graph.replay()
            updated = recorded_metric(replayed, resident())
            updated_oracle = recorded_metric(replayed.cpu(), oracle())
            key.copy_(saved_key)
            packed.view(physical, -1)[:, :8192].copy_(
                key.view(torch.uint8).reshape(physical, -1)
            )
            graph.replay()
            torch.testing.assert_close(replayed, actual, atol=0, rtol=0)
            numerical_valid = all(
                metric["status"] == "passed"
                for metric in (
                    comparison,
                    oracle_metric,
                    resident_oracle,
                    updated,
                    updated_oracle,
                )
            )
            if tune:
                if not numerical_valid:
                    raise ScoreNumericalError(
                        "fixed numerical gate failed before tuning", oracle_metric
                    )
                timings.append(
                    {
                        "config": config,
                        "score_only": measure(
                            torch, candidate, groups=groups, iterations=iterations
                        ),
                    }
                )
            else:
                currents = (lengths.repeat_interleave(next_n) - 1).to(torch.int32)
                with selector_evidence(adapter, selector) as provider_calls:
                    selected_ref = adapter.select_regions(
                        reference, currents, 16, topk_lengths=currents, sort=False
                    )
                    selected = adapter.select_regions(
                        actual, currents, 16, topk_lengths=currents, sort=False
                    )
                try:
                    selection = compare_selection(
                        torch,
                        selected_ref.cpu(),
                        selected.cpu(),
                        reference.cpu(),
                        actual.cpu(),
                        16,
                        currents.tolist(),
                        lambda row, expected=expected: expected[row],
                    )
                except AssertionError as exc:
                    selection = {"status": "failed", "error": str(exc)}
                    numerical_valid = False
                checks.append(
                    {
                        "heads": heads,
                        "next_n": next_n,
                        "pattern": pattern,
                        "config": config,
                        "status": "passed" if numerical_valid else "failed",
                        "resident": comparison,
                        "fp32_oracle": oracle_metric,
                        "resident_fp32_oracle": resident_oracle,
                        "query_gpu_cpu_cast_max_abs": float(
                            (query.float().cpu() - query.cpu().float()).abs().max()
                        ),
                        "key_gpu_cpu_cast_max_abs": float(
                            (key.float().cpu() - key.cpu().float()).abs().max()
                        ),
                        "updated_k": updated,
                        "updated_k_fp32_oracle": updated_oracle,
                        "graph_replay": True,
                        "selection": selection,
                        "selector_provider_calls": provider_calls,
                    }
                )
                if oracle_metric["status"] == "failed":
                    checks[-1]["dot_diagnostic"] = packed_dot_diagnostic(
                        torch,
                        kernels,
                        query,
                        weights,
                        key,
                        packed,
                        scales,
                        owners,
                        table,
                        lengths,
                        regions,
                        pages,
                        heads,
                        next_n,
                        actual,
                        expected,
                    )
                if diagnostic_dir is not None and heads == 8 and pattern == "random":
                    from pathlib import Path

                    from tests.support.step5_attention_testing import trace_summary

                    directory = Path(diagnostic_dir)
                    directory.mkdir(parents=True, exist_ok=False)
                    trace = directory / "grouped-selector.trace.json"
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        with torch.profiler.record_function(
                            "step5_stage:grouped_score"
                        ):
                            trace_scores = candidate()
                        with torch.profiler.record_function("step5_stage:selector"):
                            adapter.select_regions(
                                trace_scores,
                                currents,
                                16,
                                topk_lengths=currents,
                                sort=False,
                            )
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(trace))
                    checks[-1]["kernel_trace"] = trace_summary(trace)
                    checks[-1]["generated_code"] = packed_generated_code(
                        kernels,
                        query,
                        weights,
                        packed,
                        lengths,
                        table,
                        spans,
                        regions,
                        pages,
                        heads,
                        next_n,
                        config,
                        reference,
                        actual,
                        directory / "compiled",
                    )
        if tune:
            best = min(timings, key=lambda r: r["score_only"]["median_sync_wall_us"])
            # Offline tuning of the old kernel's tile/warps/worker; never changes the original launcher or defaults.
            resident_timings = []
            for tile, warps, workers in product((32, 64, 128), (2, 4), (1, 2, 4)):
                out = torch.empty_like(reference)

                def tuned_resident(
                    effective_query=effective_query,
                    packed=packed,
                    effective_weights=effective_weights,
                    lengths=lengths,
                    table=table,
                    out=out,
                    rows=rows,
                    workers=workers,
                    regions=regions,
                    next_n=next_n,
                    heads=heads,
                    tile=tile,
                    warps=warps,
                ):
                    kernels._resident_scores_sm90[(rows * workers,)](
                        effective_query,
                        packed,
                        effective_weights,
                        lengths,
                        table,
                        out,
                        effective_query.stride(0),
                        effective_query.stride(1),
                        effective_query.stride(2),
                        effective_weights.stride(0),
                        effective_weights.stride(1),
                        packed.stride(0),
                        table.stride(0),
                        table.stride(1),
                        lengths.stride(0),
                        regions,
                        next_n,
                        heads,
                        32,
                        tile,
                        workers,
                        num_warps=warps,
                    )
                    return out

                error_metrics(torch, tuned_resident(), reference)
                resident_timings.append(
                    {
                        "tile": tile,
                        "warps": warps,
                        "workers": workers,
                        "score_only": measure(
                            torch, tuned_resident, groups=groups, iterations=iterations
                        ),
                    }
                )
            checks.append(
                {
                    "heads": heads,
                    "next_n": next_n,
                    "compact_configs": timings,
                    "best": best,
                    "resident_default": measure(
                        torch, resident, groups=groups, iterations=iterations
                    ),
                    "resident_tuned": resident_timings,
                    "shape": [rows, regions],
                    "representative_production_shape": False,
                }
            )
    if checks:
        from step5_attention_harness import provenance

        checks[0]["provenance"] = provenance(SimpleNamespace(module=module))
    return checks


def diagnostic_graph_ab(torch, calls, groups=5, iterations=32):
    """Same-address warm-cache graph replay; allocation, JIT, checks, and Python submission are all excluded from kernel time."""
    reject_instrumentation()
    if groups < 2 or iterations <= 0 or not calls:
        raise ValueError(
            "diagnostic graph timing requires at least two rounds, positive iterations, and non-empty calls"
        )
    graphs = {}
    for name, call in calls.items():
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                call()
        graphs[name] = graph
        graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in calls}
    orders = []
    events = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for index in range(groups):
        order = tuple(calls) if index % 2 == 0 else tuple(reversed(calls))
        orders.append(list(order))
        for name in order:
            graphs[name].replay()
            torch.cuda.synchronize()
            events[0].record()
            graphs[name].replay()
            events[1].record()
            events[1].synchronize()
            samples[name].append(events[0].elapsed_time(events[1]) * 1000 / iterations)
    return {
        "orders": orders,
        "groups": groups,
        "iterations": iterations,
        "cache_policy": "same-address repetition inside the graph, warm-cache steady state; not a cold HBM measurement",
        "routes": {
            name: {
                "median_device_us": statistics.median(values),
                "device_group_us": values,
                "range_pct": (max(values) - min(values))
                / statistics.median(values)
                * 100,
            }
            for name, values in samples.items()
        },
    }


def diagnostic_bench_summary(defaults, tuned):
    """Microbenchmark gains never lift the accuracy or full-forward gates."""

    def gains(result):
        arms = result["routes"]
        old, new = arms["resident"], arms["compact"]
        paired = [
            a / b
            for a, b in zip(old["device_group_us"], new["device_group_us"], strict=True)
        ]
        return {
            "speedup": old["median_device_us"] / new["median_device_us"],
            "min_round_speedup": min(paired),
            "stable_gain": min(paired) >= 1.05
            and max(old["range_pct"], new["range_pct"]) <= 10,
        }

    default, best = gains(defaults), gains(tuned)
    return {
        "default_gain": default,
        "tuned_gain": best,
        "microbenchmark_gain": default["stable_gain"] and best["stable_gain"],
        "performance_valid": False,
        "production_dispatch_eligible": False,
    }


def packed_diagnostic_bench(
    extension_root=None,
    groups=6,
    iterations=32,
    oracle_samples=26,
    large=False,
    overrides=None,
):
    """Standalone FP8 performance investigation: equivalence with resident first, oracle failures retained as-is."""
    from itertools import product

    from tests.support.step5_attention_testing import require_runtime

    configs = {
        heads: packed_kernel_configs(heads, overrides=overrides)[0]
        for heads in (4, 8, 16)
    }
    if groups < 2 or iterations <= 0 or oracle_samples <= 0:
        raise ValueError("invalid diagnostic timing parameters")
    torch, module = require_runtime("flashinfer", extension_root)
    from vllm_sail.models.step5.ops import sparse_attention_opt as kernels

    if torch.cuda.get_device_capability(0) != (8, 9):
        raise ScoreDependencyBlocked("grouped scorer requires SM89")
    shapes = [("tail_empty", [18, 0, 46], 641), ("prefill512", [256, 256], 1024)]
    if large:
        shapes += [
            ("prefill8192", [631, 631] + [630] * 11, 1024),
            ("context64k_chunk4096", [4096], 8192),
        ]
    results = []
    for (label, sizes, regions), heads in product(shapes, (4, 8, 16)):
        torch.manual_seed(128)
        next_n = 2 if heads == 4 else 1
        if next_n == 2 and any(size % 2 for size in sizes):
            sizes = [630] * 12 + [632]
        rows, pages = sum(sizes), math.ceil(regions / 32)
        physical = len(sizes) * pages
        starts = [0]
        for size in sizes:
            starts.append(starts[-1] + size)
        host_owners = [owner for owner, size in enumerate(sizes) for _ in range(size)]
        owners = torch.tensor(host_owners, dtype=torch.int64, device="cuda")
        query = (torch.randn((rows, 1, heads, 256), device="cuda") * 0.125).to(
            torch.float8_e4m3fn
        )
        weights = torch.randn((rows, 1, heads), device="cuda") * 0.2
        key = (torch.randn((physical, 32, 256), device="cuda") * 0.125).to(
            torch.float8_e4m3fn
        )
        packed = torch.empty((physical, 32 * 260), device="cuda", dtype=torch.uint8)
        packed[:, :8192].copy_(key.view(torch.uint8).reshape(physical, -1))
        scales = packed[:, 8192:].view(torch.float32)
        scales.copy_(torch.rand_like(scales) + 0.25)
        packed = packed.view(physical, 32, 1, 260)
        if label == "tail_empty":
            live = [0, 1, 31, 32, 33, 640, 641]
            host_lengths = [live[i % len(live)] for i in range(rows // next_n)]
        else:
            # Within a request, causal lengths near a full context are generated from query positions.
            live_rows = [
                max(0, regions - math.ceil((size - i - 1) / 8))
                for size in sizes
                for i in range(size)
            ]
            host_lengths = live_rows[next_n - 1 :: next_n]
        lengths = torch.tensor(host_lengths, device="cuda", dtype=torch.int32)
        table = (
            owners[::next_n, None] * pages + torch.arange(pages, device="cuda")
        ).int()
        effective_query = query[:, 0].unflatten(0, (-1, next_n))
        effective_weights = weights[:, 0]
        output = torch.empty((rows, regions), device="cuda", dtype=torch.float32)
        actual = torch.empty_like(output)

        def resident_call(
            config,
            rows=rows,
            effective_query=effective_query,
            packed=packed,
            effective_weights=effective_weights,
            lengths=lengths,
            table=table,
            output=output,
            regions=regions,
            next_n=next_n,
            heads=heads,
        ):
            def launch():
                kernels._resident_scores_sm90[(rows * config["workers"],)](
                    effective_query,
                    packed,
                    effective_weights,
                    lengths,
                    table,
                    output,
                    *effective_query.stride()[:3],
                    *effective_weights.stride(),
                    packed.stride(0),
                    *table.stride(),
                    lengths.stride(0),
                    regions,
                    next_n,
                    heads,
                    32,
                    config["tile"],
                    config["workers"],
                    num_warps=config["warps"],
                )
                return output

            return launch

        def compact_call(
            config,
            starts=starts,
            rows=rows,
            query=query,
            packed=packed,
            weights=weights,
            lengths=lengths,
            actual=actual,
            regions=regions,
            pages=pages,
            heads=heads,
            next_n=next_n,
        ):
            host = score.packed_spans(starts, [rows], config["group_rows"])[0][2]
            spans = torch.tensor(host, device="cuda", dtype=torch.int32)
            grid = min(math.ceil(regions / config["tile"]), config["grid_cap"])

            def launch():
                kernels._grouped_scores_sm89[(spans.shape[0], grid)](
                    query,
                    packed,
                    weights,
                    lengths,
                    spans,
                    actual,
                    query.stride(0),
                    query.stride(2),
                    weights.stride(0),
                    weights.stride(2),
                    packed.stride(0),
                    lengths.stride(0),
                    regions,
                    pages,
                    heads,
                    next_n,
                    config["group_rows"],
                    config["tile"],
                    grid,
                    num_warps=config["warps"],
                )
                return actual

            return launch

        old_default = dict(
            tile=kernels._DEFAULT_TILE,
            warps=kernels._DEFAULT_WARPS,
            workers=kernels._resident_workers(
                rows, math.ceil(regions / kernels._DEFAULT_TILE)
            ),
        )
        old, new = resident_call(old_default), compact_call(configs[heads])
        old()
        new()
        equivalence = error_metrics(torch, actual, output)
        saved = actual.clone()
        new()
        torch.testing.assert_close(actual, saved, atol=0, rtol=0)
        del saved
        pair_ids = sorted(
            {
                round(
                    i
                    * (rows // next_n - 1)
                    / max(1, min(oracle_samples, rows // next_n) - 1)
                )
                for i in range(min(oracle_samples, rows // next_n))
            }
        )
        ids = [pair * next_n + offset for pair in pair_ids for offset in range(next_n)]
        sampled = packed_cpu_oracle(
            torch,
            query[ids],
            weights[ids],
            key,
            scales,
            owners[ids],
            lengths[pair_ids],
            regions,
            pages,
            heads,
            next_n,
        )
        oracle_metrics = {}
        for name, scores in (("resident", output), ("compact", actual)):
            try:
                oracle_metrics[name] = {
                    "status": "passed",
                    **error_metrics(torch, scores[ids].cpu(), sampled),
                }
            except ScoreNumericalError as exc:
                oracle_metrics[name] = {"status": "failed", **exc.metrics}
        old_sweep, new_sweep = [], []
        for tile, warps, workers in product((32, 64, 128), (2, 4), (1, 2, 4)):
            config = dict(tile=tile, warps=warps, workers=workers)
            call = resident_call(config)
            call()
            error_metrics(torch, output, actual)
            timing = diagnostic_graph_ab(torch, {"resident": call}, 2, iterations)
            old_sweep.append({"config": config, **timing["routes"]["resident"]})
        coarse = [
            dict(group_rows=group_rows, tile=tile, grid_cap=32, warps=warps)
            for group_rows, tile, warps in product(
                (max(4, 32 // heads), 64 // heads), (32, 64, 128), (2, 4)
            )
        ]
        if configs[heads] not in coarse:
            coarse.append(configs[heads])
        for config in coarse:
            call = compact_call(config)
            call()
            error_metrics(torch, actual, output)
            timing = diagnostic_graph_ab(torch, {"compact": call}, 2, iterations)
            new_sweep.append({"config": config, **timing["routes"]["compact"]})
        old_best = min(old_sweep, key=lambda row: row["median_device_us"])
        new_best = min(new_sweep, key=lambda row: row["median_device_us"])
        tuned = diagnostic_graph_ab(
            torch,
            {
                "resident": resident_call(old_best["config"]),
                "compact": compact_call(new_best["config"]),
            },
            groups,
            iterations,
        )
        # Re-check the defaults after the sweep to avoid thermal-state differences between the sweep and the main comparison.
        old()
        new()
        error_metrics(torch, actual, output)
        defaults = diagnostic_graph_ab(
            torch, {"resident": old, "compact": new}, groups, iterations
        )
        work = sum(host_lengths) * next_n * heads * 256 * 2
        result = {
            "name": label,
            "shape": [rows, regions],
            "heads": heads,
            "next_n": next_n,
            "query_lens": sizes,
            "oracle_rows": ids,
            "fp32_oracle": oracle_metrics,
            "resident_equivalence": equivalence,
            "deterministic": True,
            "default_configs": {"resident": old_default, "compact": configs[heads]},
            "best_configs": {
                "resident": old_best["config"],
                "compact": new_best["config"],
            },
            "defaults": defaults,
            "both_tuned": tuned,
            "resident_sweep": old_sweep,
            "compact_sweep": new_sweep,
            "work_flops": work,
            "output_bytes_per_arm": rows * regions * 4,
            "scope": "standalone score kernels on identical packed K; no packing/select/metadata/forward",
            **diagnostic_bench_summary(defaults, tuned),
        }
        results.append(result)
        print(
            f"diagnostic done {label} H{heads}: "
            f"default {result['default_gain']['speedup']:.3f}x, "
            f"both tuned {result['tuned_gain']['speedup']:.3f}x",
            flush=True,
        )
    from step5_attention_harness import provenance

    return {
        "cases": results,
        "provenance": provenance(SimpleNamespace(module=module)),
        "performance_valid": False,
        "numerical_valid": False,
        "scope": "diagnostic microbenchmark, never lifts the original FP32 oracle or production dispatch gates",
    }


def score_only_ab(data, routes, groups=5, iterations=5, config=None):
    """Prepares identical packed K and metadata up front; isolates score dispatch/output allocation, excluding pack/select."""
    result = []
    for order in (routes, tuple(reversed(routes))):
        measured = {}
        for name in order:
            if name.startswith("mqa"):
                continue
            with route(name, config):
                runtime = build_metadata(data)
                calls = []
                for group in range(data.spec["groups"]):
                    q, w = (
                        data.query[:, group : group + 1],
                        data.weights[:, group : group + 1],
                    )
                    packed, pages = data.adapter.pack_summaries(
                        data.mean[:, group], data.table, data.spec["regions"]
                    )

                    def call(q=q, w=w, packed=packed, pages=pages, runtime=runtime):
                        with score.bind_packed(
                            data.metadata, runtime, data.adapter, q, w, True
                        ):
                            begin = 0
                            for i, meta in enumerate(runtime[4]):
                                end = begin + meta.current.numel()
                                data.adapter.logits(
                                    q[begin:end],
                                    w[begin:end],
                                    packed,
                                    pages,
                                    runtime[0][begin:end],
                                    meta.lengths,
                                    data.spec["regions"],
                                    block_table=meta.block_table,
                                    pairs=runtime[8][i]
                                    if runtime[8] is not None
                                    else None,
                                    paired_heads=4 if data.spec["paired"] else 0,
                                )
                                begin = end

                    calls.append(call)

                def all_groups(calls=tuple(calls)):
                    for call in calls:
                        call()

                for _ in range(3):
                    all_groups()
                measured[name] = measure(
                    data.torch, all_groups, groups=groups, iterations=iterations
                )
        result.append({"order": list(order), "routes": measured})
    return {
        "rounds": result,
        "scope": "pre-packed K, all group/chunk scores including dispatch and output allocation; no pack/select",
    }


def eligible(score_rounds, forward_rounds, name):
    """Advises only shapes validated by both the score primary metric and the full-forward gate."""

    def median(rows, route_name, field):
        return statistics.median(
            row["routes"][route_name][field]["median_sync_wall_us"] for row in rows
        )

    baseline = "resident" if name == "compact" else "reference"
    score_gain = median(
        score_rounds, baseline, "score_pipeline_with_metadata"
    ) / median(score_rounds, name, "score_pipeline_with_metadata")
    forward_ratio = median(forward_rounds, name, "full_forward") / median(
        forward_rounds, baseline, "full_forward"
    )
    return {
        "eligible": score_gain >= 1.05 and forward_ratio <= 1.05,
        "score_wall_speedup": score_gain,
        "forward_wall_ratio": forward_ratio,
        "baseline": baseline,
    }
