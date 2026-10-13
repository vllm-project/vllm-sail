# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared timing and recording harness for the perf-regression suite.

The measurement protocol is the one fixed by CI plan PART 6.1: every case runs
:data:`WARMUP_ITERS` untimed iterations followed by :data:`TIMED_ITERS` timed
iterations, each bracketed by a ``torch.cuda.Event`` pair (PPU disguises itself
as CUDA, so the event API is available); the authoritative metric is the
**median** device time and the tail is the nearest-rank p90.

Two entry points:

* :func:`bench_kernel` - device-event timing of one zero-arg callable.  The
  callable is invoked with no arguments and must enqueue exactly the work
  being measured; operand construction and quantisation happen outside the
  timed region.
* :class:`PerfHarness` - binds a :class:`~tests.e2e.perf._baseline_store.BaselineStore`
  to the recording flow.  Each ``record_*`` call archives one metric and, in
  comparison mode, enforces the WARN/FAIL thresholds: ``FAIL`` raises
  immediately with the row's numbers, ``WARN`` only emits a pytest warning so
  the nightly lane stays green on acceptable drift (plan: "alert but do not
  block on fluctuation"; the hard 25% band is the blocking one).

Importing this module requires torch; it is imported lazily by the perf test
modules *after* their device guard chain, so collection on a CPU-only host
never reaches it (``--collect-only`` stays green in the PR gate).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import torch

from tests.e2e.perf._baseline_store import (
    FAIL_THRESHOLD,
    WARN_THRESHOLD,
    BaselineStore,
    MetricSample,
)

#: Untimed iterations before measurement (plan PART 6.1).
WARMUP_ITERS = 5
#: Timed iterations behind the median/p90 (plan PART 6.1).
TIMED_ITERS = 20


def summarize(durations_us: Sequence[float]) -> tuple[float, float]:
    """Return ``(median_us, p90_us)`` over per-iteration device times."""
    ordered = sorted(float(value) for value in durations_us)
    count = len(ordered)
    if count == 0:
        raise ValueError("no timed iterations to summarize")
    middle = count // 2
    median = (
        ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    )
    index = int(0.9 * (count - 1) + 0.5)
    return median, ordered[min(index, count - 1)]


def bench_kernel(
    call: Callable[[], Any],
    *,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
) -> tuple[float, float]:
    """Time ``call`` with CUDA events; returns ``(median_us, p90_us)``.

    One event pair per iteration keeps launch overhead out of the median and
    makes the p90 a real tail rather than an artifact of a single bracket
    around the whole loop.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for index in range(iters):
        starts[index].record()
        call()
        stops[index].record()
    torch.cuda.synchronize()

    durations = [
        starts[index].elapsed_time(stops[index]) * 1000.0 for index in range(iters)
    ]
    return summarize(durations)


def wall_time_ms(call: Callable[[], Any]) -> float:
    """Host wall-clock milliseconds of ``call`` (model-scale measurements)."""
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1000.0


class PerfHarness:
    """Records metrics into a per-chip baseline store and enforces thresholds."""

    def __init__(self, store: BaselineStore):
        self.store = store

    # -- recording -------------------------------------------------------------

    def record(
        self,
        key: str,
        value: float,
        direction: str,
        *,
        unit: str = "",
        p90: Optional[float] = None,  # noqa: UP045
        samples: int = 1,
        config: Optional[Mapping[str, Any]] = None,  # noqa: UP045
    ) -> None:
        """Archive one metric, then enforce the regression policy."""
        row = self.store.record(
            MetricSample(
                key=key,
                value=float(value),
                direction=direction,
                unit=unit,
                p90=p90,
                samples=samples,
                config=config,
            )
        )
        if row.status == "FAIL":
            raise AssertionError(
                f"perf regression FAIL on {key!r} ({self.store.chip}/{self.store.suite}): "
                f"current={row.current:.6g}{(' ' + row.unit) if row.unit else ''} "
                f"baseline={row.baseline:.6g} regression={row.regression:.1%} "
                f"> {FAIL_THRESHOLD:.0%} (warn band {WARN_THRESHOLD:.0%})"
            )
        if row.status == "WARN":
            warnings.warn(
                f"perf regression WARN on {key!r}: current={row.current:.6g} "
                f"baseline={row.baseline:.6g} regression={row.regression:.1%} "
                f"> {WARN_THRESHOLD:.0%}",
                stacklevel=2,
            )

    def record_kernel(
        self,
        key_prefix: str,
        call: Callable[[], Any],
        *,
        flops: Optional[float] = None,  # noqa: UP045
        config: Optional[Mapping[str, Any]] = None,  # noqa: UP045
    ) -> tuple[float, float]:
        """Benchmark one kernel case and record latency (+ optional TFLOPS).

        ``key_prefix`` is the case identity (e.g.
        ``"deepgemm.bf16_dense.m1024-n2048-k2048"``); two metrics are stored
        under it: ``<prefix>.latency_us`` (lower is better) and, when the case
        declares its arithmetic work, ``<prefix>.tflops`` (higher is better).
        """
        median_us, p90_us = bench_kernel(call)
        self.record(
            f"{key_prefix}.latency_us",
            median_us,
            "lower_better",
            unit="us",
            p90=p90_us,
            samples=TIMED_ITERS,
            config=config,
        )
        if flops is not None and median_us > 0.0:
            self.record(
                f"{key_prefix}.tflops",
                flops / (median_us * 1e-6) / 1e12,
                "higher_better",
                unit="tflops",
                samples=TIMED_ITERS,
                config=config,
            )
        return median_us, p90_us


def gemm_flops(m: int, n: int, k: int, *, mac_ops: float = 2.0) -> float:
    """FLOP count of an ``[M, K] x [N, K]^T`` GEMM.

    ``mac_ops`` is the FLOPs credited per multiply-accumulate: 2 for bf16 /
    int8 / fp8 tensor-core MACs and 1 for 4-bit paths (int4/mxfp4), whose
    dequant-fused MAC is conventionally counted once.
    """
    return mac_ops * float(m) * float(n) * float(k)


def attention_flops(
    seq_q: int, seq_k: int, heads: int, head_dim: int, *, causal: bool
) -> float:
    """FLOP count of one batch-1 attention (QK^T and PV, forward only)."""
    total = 4.0 * float(seq_q) * float(seq_k) * float(heads) * float(head_dim)
    if causal and seq_q == seq_k:
        total /= 2.0
    return total
