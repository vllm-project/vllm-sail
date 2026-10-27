# Reverify an accepted checkpoint

Run `scripts/verify_checkpoint.py` only with an independently reviewed, frozen
external policy and its SHA256. Keep policy files and raw run records outside
this repository. The command is read-only and does not create acceptance:

```bash
python scripts/verify_checkpoint.py \
  --root /absolute/run-artifacts \
  --policy /absolute/checkpoint-policy.json \
  --policy-sha256 <reviewed-policy-sha256>
```

This consumes the normalized receipts from the run's reviewed executor. Adapt
producers explicitly when upstream or receipt contracts change. A missing receipt
or unsupported schema blocks verification; do not fill gaps with fabricated
booleans, inferred passes or a weaker policy.

The policy schema is version `1` and declares:

- `commits`: full `vllm` and `vllm-sail` commit SHAs.
- `artifacts`: canonical relative paths mapped to SHA256 for every selected
  receipt, raw XML/log, reviewed input and producer. Referenced receipts must be
  in this map. Model audit records additionally bind their complete raw files.
- `predecessors`: accepted record paths; `reviews`: records containing passed
  `spec` and `standards` results with no unresolved findings.
- `producer_bindings`: `{receipt, field, producer}` records binding receipt
  producer SHA fields to the actual reviewed producer files.
- `xml`: named `{file, sha256, passed, skipped}` records. `skip_reasons` maps
  installed-mode reasons to counts; `excluded_cases` maps exact device test IDs
  to reasons. Unexpected exclusions fail. Preserve failures in separate attempts.
- `cpu`: all four versions, each with a `proof` path and an `xml` key. Proofs must
  bind the exact commits, counts, XML and absent Torch/vLLM environments. The
  acceptance summary must reference those same proofs and raw artifacts.
- `source_proofs`: `{file, status, expected}` records for exact-source and native
  generation proof. Bind their producer and raw log/XML hashes in the artifact
  set; a WIP result is not a clean-commit result.
- `metadata_review`: exact-pin runtime fingerprint review with both axes passed and no unresolved targets. `source_log` and `generation_log` bind raw source/generation logs to their typed proof records. Source and controller reviews must bind the same exact commit pair. All source/control/metadata reviews must match the accepted receipt.
- `postproof`, `wheel_manifest`, `installed_files`, `acceptance`: receipt paths.
  The final byte maps must include both installed packages, all four SAIL native
  modules and the exact declared upstream Rust/native members.
- `stacks`: per-environment SDK and Torch SHA256 identities. `rust`: expected
  exact-source artifact contract fields, including the declared member map and
  absent obsolete bridges. Discover that contract from the pinned source rather
  than assuming one historical layout.
- `postproof_artifacts`: remote receipt-relative keys mapped to local artifacts.
- `xfail_causes`: `file` and exact `cases`; each case requires an `exception` and
  `traceback_contains` list. Preserve new raw causes at every checkpoint.
- `models`: every agreed profile, each with `audit`, `request_budget`, required
  `positive_metrics`, minimum `evidence_counts` and `exact_evidence_counts`.
  Require complete natural-stop responses and SSE, zero pre/post shutdown ERROR,
  correct configured paths, memory/capture and matching postproof raw-file hashes.
  Require a passed normalized result equal to its exact raw result.json, including
  the metrics digest; cleanup success cannot override an explicit failed state.

Review expected test sets/counts, profiles, exclusions, identities and producer
contracts independently. Deriving expectations from a failed run or dropping a
required model would invalidate the acceptance policy even if its JSON hashes.
The validator verifies evidence consistency; it cannot replace semantic review of
complete model answers or establish trust in an unreviewed executor.

Each agreed model requires one raw `result.json`, `invocation.json`,
`models.json`, `stream.json`, `serve.log`, `metrics.prom`, `runner-process.txt`,
`owned-process-group.txt` and every `request-0.json` through `request-4.json`.
Require unique basenames in both the manual audit and runtime postproof,
identical hashes across them, and an explicit frozen-policy artifact binding
for each local raw file. Optional extra raw artifacts remain bound and preserved.
A hash-consistent subset of these files is incomplete evidence. This inventory
check supplements manual answer and log review; it does not perform that review.
