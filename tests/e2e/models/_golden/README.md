# Golden baselines — accuracy model E2E (schema v2)

Per-chip evalscope GSM8K accuracy baselines for the model-precision suite
(`tests/e2e/models`). Baselines are **captured on real PPU hardware**, never
hand-written.

## What changed (schema v1 → v2)

Schema v1 stored per-token logprob goldens; schema v2 stores a single accuracy
score from evalscope GSM8K evaluation. A score is judged against its own
checkpoint's golden: the run passes when it lands inside
**`golden ± 0.02`**. The former `score >= 0.9` threshold is no longer a gate —
it only applies to a checkpoint that has no golden yet, so a first run cannot
be reported as "created" while scoring far below anything acceptable.

## File naming

```
accuracy_<model_key>_<chip>.json                   formal baseline
accuracy_<model_key>_<chip>.pending-review.json    unreviewed first-run output
```

`chip` is derived from the detected device capability (`(8,0) → 810e`,
`(8,9) → 890p`).

## Cases vs. goldens

One checkpoint may be exercised by several **cases** in
`scripts/ci/ppu_e2e_models.json`, each with its own `vllm serve` argv
(`server_cmd`) and therefore its own run configuration. Two identifiers keep
the two concepts apart:

| Field | Meaning | Used for |
|-------|---------|----------|
| `key` | catalog case label | job name, artifact name, serve log |
| `model_key` | evaluator model-config key | **golden identity** |

Goldens are keyed on `model_key`, **not** on the case: cases over the same
weights share one baseline, so a second case cannot quietly establish a
second, laxer reference for the same checkpoint. Adding a case therefore needs
no new golden file — only a `model_key` and an optional `server_cmd`.

A golden created for a checkpoint whose common failure mode is a *high*
score is still bounded on the upper side: the band is symmetric, so drift in
either direction fails.

### Adding an accuracy case

```jsonc
{
  "key": "deepseek-v4-mxfp4-eager",      // unique; becomes the job label
  "category": "accuracy",
  "model_key": "deepseek_v4_mxfp4",      // must exist in MODEL_CONFIGS
  "model_id": "T-HEAD/...",              // resolved against the offline cache
  "served_name": "deepseek-v4-mxfp4",
  "board": "ZW-M890P",
  "tp": 8,                               // PPU count the container gets
  "extra_serve_flags": "--trust-remote-code",
  // null / omitted: use the evaluator's built-in argv.  A non-empty array
  // replaces that argv wholesale, so include every flag this case needs.
  "server_cmd": [
    "vllm", "serve", "{model}",
    "--served-model-name", "{served_name}",
    "--host", "127.0.0.1", "--port", "{port}",
    "--tensor-parallel-size", "8",
    "--enforce-eager"
  ]
}
```

`{model}` expands to the resolved checkpoint, `{port}` to `VLLM_PORT`, and
`{served_name}` to the per-run nonce (`<served_name>-run-<hex>`), so the
readiness probe and evalscope keep addressing the process this case started.
Unknown braces are left untouched. Mirror these keys in `MODEL_CONFIGS` of
`scripts/eval/ppu_gsm8k_eval.py` if the case needs a new `model_key`.

## Checkpoint resolution

The checkpoint paths in scripts and `scripts/ci/ppu_e2e_models.json` may be
recorded under either `/nas_aisw/datasets` or `/ppusw/datasets`. Both mount the
same NFS export. `resolve_checkpoint` tries the recorded path first and then
each mirrored root (`_offline_runner.CHECKPOINT_ROOT_MIRRORS`).

## Schema (`schema_version: 2`)

```jsonc
{
  "schema_version": 2,
  "model": "deepseek_v4_int8",        // model key
  "chip": "810e",                      // must equal the running device chip
  "checkpoint": "/ppusw/datasets/...", // resolved weight path
  "checkpoint_tail": "v4/DeepSeek-V4-Flash-0731-W8A8-INT8",  // identity
  "benchmark": "gsm8k",
  "evalscope_version": "1.14.1",
  "dataset": {
    "dataset_id": "/ppusw/datasets/vision/EvalScope/data/gsm8k/v2/gsm8k",
    "few_shot_num": 4,
    "few_shot_random": false,
    "subset_list": ["main"],
    "num_expected": 1319
  },
  "serving": {
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.90,
    "seed": 0
  },
  "score": 0.9742,                     // evalscope accuracy (0.0–1.0)
  "score_num": 1319,                   // number of evaluated samples
  "tolerance": {
    "mode": "band",                    // symmetric band around score
    "abs": 0.02                        // ±2 points
  },
  "measured_at": "2025-07-15T06:00:00+00:00"
}
```

## Comparison rules

1. **Identity gate**: `chip` must match; `checkpoint_tail` (when present) must
   match — prevents cross-chip or cross-checkpoint comparison.
2. **Golden band gate** (when a formal golden exists), in order:
   a. **Completeness**: `num == dataset.num_expected`.
   b. **Band**: `golden.score - abs <= score <= golden.score + abs`, i.e.
      `golden ± 0.02` by default. Two points absorbs run-to-run noise on PPU
      and is still narrow enough that a real regression cannot hide; the band
      is symmetric so an abnormally high golden is caught as well.
3. **Threshold fallback** (only when no golden exists): `score >= 0.9`, so a
   first run cannot be reported as acceptable while scoring far below it.
   Configurable via `--threshold`.

Records still carrying the legacy `mode: lower_bound` tolerance keep their
one-sided bound and hard floor, so an older baseline continues to gate exactly
as it did.

## Nightly CI

`ppu-accuracy-nightly.yaml` runs every accuracy **case** once a night against
the wheels of the newest successful PPU CI build. The checkout inside the
runtime container is read-only, so `scripts/ci/ppu_e2e_accuracy.sh` copies
every `accuracy_*.json` from this directory into the writable results tree
before evaluating (`GOLDEN_SEED_DIR`, default
`/source/tests/e2e/models/_golden`). Consequences:

* A checkpoint with a committed golden here **is** gated on the band.
* A checkpoint without one reports `created`, uploads
  `accuracy_<model>_<chip>.pending-review.json` inside the run's
  `ppu-accuracy-<case_id>-*` artifact, and only enforces `score >= threshold`.

Several cases sharing one checkpoint therefore produce one golden candidate
per case but gate against the single baseline for that checkpoint; import the
first pending file you accept and the sibling cases will gate against it on
the next run.

Promotion stays a human step: review the pending artifact, commit it here as
`accuracy_<model>_<chip>.json`, and the next run gates against it.

`chip` is whatever device the nightly runner exposes, not the catalog `board`
field: the lane runs on the self-hosted `8-ppu-m890p` runner, so
`deepseek_v4_int8` is currently captured against the ZW-M890P int8 path.

## Capturing / refreshing

Run the eval script or test on the target PPU machine with `--update-golden`:

```bash
python scripts/eval/ppu_gsm8k_eval.py \
    --model-key deepseek_v4_int8 --update-golden
```

Add `--case-id <key> --server-cmd '[ ... ]'` to reproduce a specific catalog
case; both only affect the serve log and the argv, never the golden identity.

Or via pytest:

```bash
pytest tests/e2e/models/test_model_accuracy.py::test_deepseek_v4_int8_accuracy \
    --update-golden
```

## Pending-review → formal promotion

A `.pending-review.json` file never participates in comparison — an unreviewed
first run cannot silently become the baseline. Promote it by either:

1. Review the pending file, then copy it:
   `cp accuracy_<model>_<chip>.pending-review.json accuracy_<model>_<chip>.json`
2. Re-run with `--update-golden`, which writes the formal name directly and
   deletes the pending sibling.

## Model configurations

| Model key | Chip | Checkpoint |
|-----------|------|-----------|
| `deepseek_v4_int8` | 810e | `DeepSeek-V4-Flash-0731-W8A8-INT8` |
| `deepseek_v4_mxfp4` | 890p | `DeepSeek-V4-Flash-0731-mxfp4-fp8channel` |
| `minimax-m3-fp8` | 890p | `MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken` |
| `qwen3.8-flash-next` | 890p | `Qwen3.8-Flash-Next` |
