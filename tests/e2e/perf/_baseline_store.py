# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-chip performance baseline storage and regression comparison.

Baseline layout (CI plan PART 6.1)::

    tests/e2e/perf/baselines/<chip>/<suite>.json

``chip`` is ``810e`` / ``890p`` and ``suite`` is ``kernel`` or ``model``.  One
JSON document holds every metric of that suite for that chip; the two chips are
never cross-compared because they take different quantization and TP paths.

Document schema (``SCHEMA_VERSION = 1``)::

    {
      "schema_version": 1,
      "chip": "810e",
      "suite": "kernel",
      "metrics": {
        "<metric key>": {
          "direction": "higher_better" | "lower_better",
          "unit": "us" | "tflops" | ...,
          "status": "pending" | "confirmed",
          "median": 123.4,          # authoritative comparison value
          "p90": 130.0,             # informational tail
          "samples": 20,            # timed iterations behind the median
          "history": [123.4, ...],  # last N run medians (rolling refresh)
          "captured": {"commit": "...", "date": "...", "config": {...}}
        }
      }
    }

Regression policy (PART 6.1 thresholds, centralised here):

* regression fraction ``r`` is ``0.0`` when the metric improved;
* ``r > FAIL_THRESHOLD (25%)`` -> ``FAIL`` (the CI job must fail);
* ``WARN_THRESHOLD (10%) < r <= FAIL_THRESHOLD`` -> ``WARN`` (job summary only);
* a metric whose baseline ``status`` is ``"pending"`` (written by the first run
  on a chip, never reviewed) reports ``PENDING`` and never fails.

Baseline promotion flow:

1. First run on a chip: no file -> results are written with
   ``status="pending"`` and every metric reports ``PENDING`` (not a failure).
2. A human reviews the pending numbers (CI artifact), copies the JSON into the
   repo and commits it, flipping ``status`` to ``"confirmed"``.
3. ``--update-baseline`` (weekly workflow / dispatch input) appends the current
   median to ``history`` and recomputes ``median`` / ``p90`` as the rolling
   median of the last :data:`REFRESH_WINDOW` run medians, so a single noisy
   run never redefines the baseline.

This module is pure standard library: no torch, no vLLM, no device.  It stays
importable (and unit-testable) on a CPU-only host, which is what keeps
``pytest --collect-only`` green in the PR gate.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Optional

SCHEMA_VERSION = 1

#: Relative regression that still passes silently.
WARN_THRESHOLD = 0.10
#: Relative regression that fails the CI job.
FAIL_THRESHOLD = 0.25
#: Run medians kept for the rolling ``--update-baseline`` refresh (PART 6.1:
#: "median of the last 7 nightly runs").
REFRESH_WINDOW = 7
#: Run medians kept in ``history`` for trend inspection (superset of refresh).
HISTORY_LIMIT = 30

HIGHER_BETTER = "higher_better"
LOWER_BETTER = "lower_better"

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_PENDING = "PENDING"

_VALID_DIRECTIONS = (HIGHER_BETTER, LOWER_BETTER)


class MetricSample(NamedTuple):
    """One measured metric handed to :meth:`BaselineStore.record`."""

    key: str
    value: float
    direction: str
    unit: str = ""
    #: Tail latency of the timed iterations; ``None`` when not meaningful
    #: (e.g. a single wall-clock model measurement).
    p90: Optional[float] = None  # noqa: UP045  (py3.9 runtime: keep Optional)
    #: Number of timed iterations behind ``value``.
    samples: int = 1
    #: Fixed shape/config of the case, archived for traceability.
    config: Optional[Mapping[str, Any]] = None  # noqa: UP045


class ComparisonRow(NamedTuple):
    """One line of the post-run comparison table."""

    key: str
    unit: str
    direction: str
    current: float
    baseline: Optional[float]  # noqa: UP045
    #: Signed relative change of the current value vs the baseline
    #: (``0.1`` = +10%), ``None`` while the baseline is pending.
    delta_pct: Optional[float]  # noqa: UP045
    #: Regression fraction in the *bad* direction, clamped at ``0.0``.
    regression: float
    status: str


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in values)
    count = len(ordered)
    if count == 0:
        raise ValueError("cannot take the median of an empty sequence")
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _p90(values: Sequence[float]) -> float:
    """Nearest-rank p90; small samples fall back to the maximum."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("cannot take the p90 of an empty sequence")
    index = int(0.9 * (len(ordered) - 1) + 0.5)
    return ordered[min(index, len(ordered) - 1)]


def git_commit(root: Optional[Path] = None) -> str:  # noqa: UP045
    """Best-effort short commit hash of the checkout; ``"unknown"`` off-git."""
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root) if root is not None else None,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return "unknown"
    if probe.returncode != 0:
        return "unknown"
    return probe.stdout.strip() or "unknown"


def regression_fraction(current: float, baseline: float, direction: str) -> float:
    """Fraction by which ``current`` regressed against ``baseline``.

    ``0.0`` means "no regression" (equal or improved); ``0.25`` means the
    metric moved 25% into the bad direction.  A non-positive baseline makes a
    relative band meaningless, so the metric is treated as unregressed and the
    caller is expected to keep such baselines out of the store.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of {_VALID_DIRECTIONS}"
        )
    if baseline <= 0.0:
        return 0.0
    if direction == HIGHER_BETTER:
        fraction = (baseline - float(current)) / baseline
    else:
        fraction = (float(current) - baseline) / baseline
    return max(fraction, 0.0)


def classify(regression: float) -> str:
    """Map a regression fraction onto PASS / WARN / FAIL."""
    if regression > FAIL_THRESHOLD:
        return STATUS_FAIL
    if regression > WARN_THRESHOLD:
        return STATUS_WARN
    return STATUS_PASS


def compare(
    current: float, baseline: float, direction: str
) -> tuple[str, float, float]:
    """Return ``(status, regression_fraction, signed_delta_pct)``."""
    regression = regression_fraction(current, baseline, direction)
    if baseline > 0.0:
        delta_pct = (float(current) - baseline) / baseline
    else:
        delta_pct = 0.0
    return classify(regression), regression, delta_pct


class BaselineStore:
    """Reads, records into and writes one ``<chip>/<suite>.json`` document."""

    def __init__(self, root: Path, chip: str, suite: str, update: bool = False):
        if not chip:
            raise ValueError("chip key must not be empty")
        self.root = Path(root)
        self.chip = chip
        self.suite = suite
        #: ``True`` under ``--update-baseline``: refresh instead of compare.
        self.update = update
        self.path = self.root / chip / f"{suite}.json"
        self._doc: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "chip": chip,
            "suite": suite,
            "metrics": {},
        }
        self._existed = False
        if self.path.exists():
            self._doc = self._parse(self.path.read_text())
            self._existed = True
        #: Rows produced by :meth:`record` during this run, in record order.
        self.rows: list[ComparisonRow] = []
        #: Metric keys whose baseline is pending human review.
        self.pending: list[str] = []

    # -- loading ------------------------------------------------------------

    def _parse(self, text: str) -> dict[str, Any]:
        doc = json.loads(text)
        if not isinstance(doc, dict) or "metrics" not in doc:
            raise ValueError(f"{self.path} is not a baseline document")
        chip = doc.get("chip", self.chip)
        if chip != self.chip:
            raise ValueError(
                f"baseline chip mismatch: {self.path} records chip={chip!r} but "
                f"the current device is {self.chip!r}; per-chip baselines must "
                "never be cross-compared"
            )
        doc.setdefault("schema_version", SCHEMA_VERSION)
        doc.setdefault("suite", self.suite)
        return doc

    @property
    def existed(self) -> bool:
        """Whether a baseline document was present when the store opened."""
        return self._existed

    def metrics(self) -> dict[str, Any]:
        return self._doc["metrics"]

    # -- capture metadata -----------------------------------------------------

    def _capture(self, sample: MetricSample) -> dict[str, Any]:
        return {
            "commit": os.environ.get("VLLM_SAIL_PERF_COMMIT")
            or git_commit(self.root.parent.parent.parent),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": dict(sample.config or {}),
        }

    # -- write ----------------------------------------------------------------

    def save(self) -> Path:
        """Atomically rewrite the baseline document; returns its path."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(tmp), str(self.path))
        self._existed = True
        return self.path

    # -- record ---------------------------------------------------------------

    def _refresh(self, entry: dict[str, Any], sample: MetricSample) -> None:
        """``--update-baseline``: roll the median over the recent history."""
        history = [float(v) for v in entry.get("history", [])]
        history.append(sample.value)
        entry["history"] = history[-HISTORY_LIMIT:]
        window = entry["history"][-REFRESH_WINDOW:]
        entry["median"] = _median(window)
        entry["p90"] = sample.p90 if sample.p90 is not None else _p90(window)
        entry["samples"] = sample.samples
        entry["status"] = "confirmed"
        entry["captured"] = self._capture(sample)

    def _create(self, sample: MetricSample) -> dict[str, Any]:
        return {
            "direction": sample.direction,
            "unit": sample.unit,
            "status": "pending",
            "median": sample.value,
            "p90": sample.p90 if sample.p90 is not None else sample.value,
            "samples": sample.samples,
            "history": [sample.value],
            "captured": self._capture(sample),
        }

    def record(self, sample: MetricSample) -> ComparisonRow:
        """Compare one metric against its baseline (or establish it).

        Comparison mode appends the run median to ``history`` and rewrites the
        document, so the weekly ``--update-baseline`` refresh sees the rolling
        window the plan asks for.  Nothing here raises on regression; the
        caller decides what a ``FAIL`` row means (the perf harness asserts and
        the CI script renders the table).
        """
        if sample.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"metric {sample.key!r}: unknown direction {sample.direction!r}"
            )
        metrics = self._doc["metrics"]
        entry = metrics.get(sample.key)

        if entry is None:
            metrics[sample.key] = self._create(sample)
            self.pending.append(sample.key)
            row = ComparisonRow(
                key=sample.key,
                unit=sample.unit,
                direction=sample.direction,
                current=sample.value,
                baseline=None,
                delta_pct=None,
                regression=0.0,
                status=STATUS_PENDING,
            )
            self.rows.append(row)
            return row

        baseline_median = float(entry["median"])
        if self.update:
            self._refresh(entry, sample)
            row = ComparisonRow(
                key=sample.key,
                unit=sample.unit,
                direction=sample.direction,
                current=sample.value,
                baseline=baseline_median,
                delta_pct=None,
                regression=0.0,
                status=STATUS_PASS,
            )
            self.rows.append(row)
            return row

        status, regression, delta_pct = compare(
            sample.value, baseline_median, sample.direction
        )
        if entry.get("status") != "confirmed":
            # Pending baseline: still surface the delta, never fail on it.
            status = STATUS_PENDING
            self.pending.append(sample.key)
        # Roll the run median into history for the weekly refresh, keeping the
        # authoritative median untouched in comparison mode.
        history = [float(v) for v in entry.get("history", [])]
        history.append(sample.value)
        entry["history"] = history[-HISTORY_LIMIT:]
        row = ComparisonRow(
            key=sample.key,
            unit=sample.unit,
            direction=sample.direction,
            current=sample.value,
            baseline=baseline_median,
            delta_pct=delta_pct,
            regression=regression,
            status=status,
        )
        self.rows.append(row)
        return row

    # -- reporting --------------------------------------------------------------

    def failures(self) -> list[ComparisonRow]:
        return [row for row in self.rows if row.status == STATUS_FAIL]

    def warnings(self) -> list[ComparisonRow]:
        return [row for row in self.rows if row.status == STATUS_WARN]

    def report_json(self) -> dict[str, Any]:
        """Machine-readable run report (CI artifact ``perf_report.json``)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "chip": self.chip,
            "suite": self.suite,
            "update": self.update,
            "baseline_path": str(self.path),
            "thresholds": {"warn": WARN_THRESHOLD, "fail": FAIL_THRESHOLD},
            "rows": [
                {
                    "key": row.key,
                    "unit": row.unit,
                    "direction": row.direction,
                    "current": row.current,
                    "baseline": row.baseline,
                    "delta_pct": row.delta_pct,
                    "regression": row.regression,
                    "status": row.status,
                }
                for row in self.rows
            ],
            "pending": list(self.pending),
        }

    def report_markdown(self) -> str:
        """Human-readable comparison table (CI artifact / job summary)."""
        lines = [
            f"# perf regression report - {self.chip} / {self.suite}",
            "",
            f"baseline: `{self.path}`  ",
            f"thresholds: WARN > {WARN_THRESHOLD:.0%}, FAIL > {FAIL_THRESHOLD:.0%} "
            "relative regression  ",
            f"mode: {'update (--update-baseline)' if self.update else 'compare'}",
            "",
            "| metric | current | baseline | delta | status |",
            "| --- | ---: | ---: | ---: | :--: |",
        ]
        for row in self.rows:
            baseline = "-" if row.baseline is None else f"{row.baseline:.4g}"
            delta = "-" if row.delta_pct is None else f"{row.delta_pct:+.1%}"
            unit = f" {row.unit}" if row.unit else ""
            lines.append(
                f"| `{row.key}` | {row.current:.4g}{unit} | {baseline} | {delta} "
                f"| {row.status} |"
            )
        if self.pending:
            lines.extend(
                [
                    "",
                    f"pending-review baselines (first run writes, never fails): "
                    f"{len(self.pending)} metric(s)",
                ]
            )
        failures = self.failures()
        if failures:
            lines.extend(["", f"FAILED metrics: {len(failures)}"])
            lines.extend(
                f"- `{row.key}` regressed {row.regression:.1%}" for row in failures
            )
        lines.append("")
        return "\n".join(lines)

    def emit(self, report_dir: Path) -> tuple[Path, Path]:
        """Write ``perf_report.json`` / ``perf_report.md`` into ``report_dir``."""
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / "perf_report.json"
        md_path = report_dir / "perf_report.md"
        json_path.write_text(
            json.dumps(self.report_json(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(self.report_markdown(), encoding="utf-8")
        return json_path, md_path
