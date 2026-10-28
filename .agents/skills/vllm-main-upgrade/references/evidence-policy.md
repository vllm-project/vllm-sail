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

The policy schema is version `2`. Version 1 is rejected because it cannot
establish the model and schema-parser identity bindings below. Update the
reviewed executor and collect missing evidence before freezing a new policy;
changing a version number or relabeling historical receipts is insufficient.
On Python 3.10 the verifier uses `tomli` from `requirements/dev.txt` to read the
generated schema resource; later Python versions use `tomllib`.

The policy declares:

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
  to reasons. Device XML also requires `hardware_inapplicable_cases`, a map of
  exact IDs to their same exclusion reasons. Its IDs and `xfail_causes.cases`
  must be disjoint and together equal all actual excluded device IDs. Use an
  empty map when there are no hardware exclusions. Unexpected exclusions fail.
  Preserve failures in separate attempts.
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
  `traceback_contains` list. The cause receipt requires `commits` and
  `device_xml_sha256` matching the selected device XML. Preserve new raw causes
  at every checkpoint, including an empty bound cause receipt when none apply.
- `models`: every agreed profile, each with `audit`, `request_budget`,
  `request_indices` (a nonempty ordered list of unique nonnegative integers), required
  `positive_metrics`, minimum `evidence_counts` and `exact_evidence_counts`.
  Require complete natural-stop responses and SSE, zero pre/post shutdown ERROR,
  correct configured paths, memory/capture and matching postproof raw-file hashes.
  Require a passed normalized result equal to its exact raw result.json, including
  the metrics digest; cleanup success cannot override an explicit failed state.
- `schema_parse`: `{receipt, resource, producer, log, environment}`. All four
  file paths must be selected artifacts. `resource` is a byte-for-byte copy of
  the installed final SAIL wheel's `vllm_sail/native/data/excluded_ops.toml`;
  `environment` names the compatible-Torch entry in `stacks`. See below for
  the required receipt fields.

Review expected test sets/counts, profiles, exclusions, identities and producer
contracts independently. Deriving expectations from a failed run or dropping a
required model would invalidate the acceptance policy even if its JSON hashes.
The validator verifies evidence consistency; it cannot replace semantic review of
complete model answers or establish trust in an unreviewed executor.

Each agreed model requires one raw `result.json`, `invocation.json`,
`models.json`, `stream.json`, `serve.log`, `metrics.prom`, `runner-process.txt`,
`owned-process-group.txt` and `request-<index>.json` for every policy-declared
request index. The manual audit's answer indices must match that exact order;
the natural-stop count must equal its length plus one for SSE. For example,
`request_indices: [0]` requires one ordinary answer and one SSE answer; ten
ordinary requests declare ten indices. Request cardinality is agreed before
execution rather than inferred from the files that happened to survive.
Require unique basenames in both the manual audit and runtime postproof,
identical hashes across them, and an explicit frozen-policy artifact binding
for each local raw file. Optional extra raw artifacts remain bound and preserved.
A hash-consistent subset of these files is incomplete evidence. This inventory
check supplements manual answer and log review; it does not perform that review.

## Model execution identities

Every model's raw `invocation.json`, raw `result.json`, complete-answer receipt,
manual audit and per-model entry in `postproof.models` requires:

```json
{
  "commits": {"vllm": "<full-sha>", "vllm-sail": "<full-sha>"},
  "wheel_sha256": {"vllm": "<wheel-sha256>", "vllm-sail": "<wheel-sha256>"}
}
```

These values must equal the policy commit pair and the hashes in the final
wheel manifest. Record them while executing against that installed wheel pair.
The verifier checks cross-receipt consistency; it does not independently prove
that an unreviewed producer recorded the running packages honestly.

## Real-Torch schema parsing receipt

The reviewed producer must invoke `torch._C.parse_schema` on each `op.schema`
from the copied installed resource, in resource order. Its receipt requires:

- `status: "all-excluded-op-schemas-parsed"`, `parser: "torch._C.parse_schema"`,
  `exit_code: 0`, and `failed: []`.
- The common `commits` and `wheel_sha256` execution identities above.
- `environment` matching the policy, and `torch_sha256` matching that stack.
- `resource_sha256`, `producer_sha256` and `log_sha256` matching their raw
  policy-bound artifacts.
- `schema_count` derived from the resource and `parsed_schemas`, the complete
  ordered list of `namespace + "::" + schema` strings actually parsed. Counts
  alone cannot prove that all distinct schemas were attempted.

The resource hash must also match its installed-file map entry. Both final
postproof and acceptance require `schema_parse_receipt_sha256` binding this
exact receipt. Missing parsing evidence cannot be replaced by native imports,
synthetic fixtures or TOML round trips. Preserve parser failures as failed
attempts; they cannot qualify a checkpoint.
