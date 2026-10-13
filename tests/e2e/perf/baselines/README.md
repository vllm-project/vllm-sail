# Performance baselines

Per-chip performance baselines for the regression suites in `tests/e2e/perf`.

## Layout

```
baselines/<chip>/<suite>.json
```

* `<chip>` — `810e` (ZW-810E, capability `(8, 0)`) or `890p` (ZW-890P,
  capability `(8, 9)`). The two chips take different quantization and tensor
  -parallel paths, so their baselines are archived separately and **never**
  cross-compared (the store raises on a chip mismatch).
* `<suite>` — `kernel` (`test_kernel_perf.py`) or `model`
  (`test_model_perf.py`).

One JSON document holds every metric of that suite for that chip; see
`tests/e2e/perf/_baseline_store.py` for the authoritative schema
(`SCHEMA_VERSION = 1`).

## Why this directory ships empty

Baseline numbers are only meaningful when captured on the real PPU hardware the
nightly lane runs on, so **no `*.json` is committed here**. `.gitkeep` keeps the
directory in the tree; the JSON files are produced by the first nightly run on
each chip and promoted by a human (see below). A `.gitignore` next to them
enforces that: a generated baseline never shows up as an uncommitted change and
can never be swept into a commit by `git add -A`.

## Promotion flow

1. **First run on a chip** — no baseline file exists. The store writes the
   measured medians with `status="pending"` and every metric reports `PENDING`
   (the lane stays green; a pending baseline never fails).
2. **Human review** — the nightly job uploads the pending baseline plus the
   comparison report (`perf_report.json` / `perf_report.md`) as CI artifacts. A
   reviewer sanity-checks the numbers against the hardware spec, then copies the
   JSON into `baselines/<chip>/<suite>.json`, flips `status` to `"confirmed"`
   and commits it with `git add -f` (baselines are **never** auto-committed —
   the `-f` is the deliberate human act).
3. **Steady state** — every run compares against the confirmed baseline:
   * relative regression `> 10%` → `WARN` (job summary only),
   * relative regression `> 25%` → `FAIL` (the lane fails),
   * a still-`pending` metric keeps reporting `PENDING`.

## Refreshing a baseline

Pass `--update-baseline` (nightly dispatch input `update_baseline`, or the
weekly refresh lane). Instead of asserting, the store appends the current run
median to `history` and recomputes `median` / `p90` as the **rolling median of
the last 7 run medians** (`REFRESH_WINDOW`), so a single noisy run never
redefines the baseline. Refreshed entries are marked `confirmed`.
