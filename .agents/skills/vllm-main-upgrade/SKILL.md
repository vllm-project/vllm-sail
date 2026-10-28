---
name: vllm-main-upgrade
description: Update the upstream vLLM pin in vLLM SAIL through sampled commits while preserving tested checkpoint history and reviewable pull requests.
---

# Upgrade the vLLM main pin

Use this for upstream pin migrations in vLLM SAIL. Read the repository's contribution, patch, kernel and verification guides for the interfaces being changed.

## Freeze the run

Resolve the starting SAIL commit, release endpoints and target main SHA once. Keep plans and run records outside the repository. Use [sample_commits.py](scripts/sample_commits.py) with explicit start, release and frozen target refs to write a new immutable sampling plan. Sample the frozen main first-parent history using the chosen calendar timezone and intervals. Include release tag commits as mandatory endpoints even when they belong to a release branch. Later upstream commits belong to a subsequent run.

Bind authorized source roots, node, existing build/test environments and model profiles, including each profile's ordinary request indices and one SSE request. Capture the SAIL SDK and both Torch identities. Preserve dirty, untracked and needed ignored content plus live work before source changes. Continue within already authorized bounds without repeating approvals; additional nodes, replacement environments and publication need their own scope authorization.

## Advance one checkpoint

Do not advance the formal pin until its predecessor is accepted. Independent investigation or private drafts may proceed while hardware work runs; label their evidence accordingly.

1. Inventory changed copied bodies, delegated providers, signatures, native inputs and separately cached by-value consumers. Review real source bodies; an unchanged signature or successful import does not prove compatibility.
2. Prefer registration and platform interfaces. Preserve PPU modification regions when refreshing copied upstream bodies. Regenerate native outputs, patch inventory, interface locks and reviewed drift baselines with maintained tools.
3. Add behavioral regressions at changed call boundaries using real upstream helpers where feasible. If an existing fixture becomes stale, retain its semantic assertions and preserve the failing evidence.
4. Where metadata prerequisites allow it, run the installed UT selector against the preliminary wheel before freezing the final candidate. Always set the exact checked-out VLLM_SOURCE_ROOT, including for selectors that deselect upstream_source: unmarked inventory tests also need it. If generated inventory is a prerequisite, record the deferred gate explicitly. Immediately after metadata adoption, run the complete installed UT selector against the WIP tree before freezing or committing it; a deferred gate is never a pass. Build the Rust outputs declared by that exact pin through its official source build entry. Verify the full SHA in the frontend version, reject undeclared stale modules and use separate per-attempt build/packaging directories. Build SAIL with prepared SAIL Torch; install without dependency replacement in the separate compatible Torch environment.
5. Review captured runtime provider bodies before updating fingerprints. After adopting real generated metadata, regenerate the interface lock and run the complete WIP CPU/source/generation gates again. Assert every patch declaration agrees with the supported release range. Then freeze and independently review the complete tree, save a clean signed checkpoint commit and rerun its CPU/source/generation gates. Build its clean SAIL wheel and perform all gates in [acceptance.md](references/acceptance.md).
6. Accept only after the composite evidence check succeeds. Reverify the accepted raw evidence with [verify_checkpoint.py](scripts/verify_checkpoint.py) and a frozen external schema-2 policy; read [evidence-policy.md](references/evidence-policy.md) for its input contract. Collect model commit/wheel identities at execution time and the complete real-Torch schema parsing receipt before final postproof. Preserve failed attempts and supplements separately. Record acceptance before preparing the next checkpoint.

Capture conditional aliases in both cold and preloaded consumer processes when the generated target set changes. Resolve the captured callable's actual provider, including inherited methods, compilation decorators and unavailable-backend fallbacks, before assigning a source fingerprint. Read the provider guidance in [acceptance.md](references/acceptance.md).

Recheck release-branch API ownership at mandatory off-main endpoints; do not assume main and the release tag provide the same helper. Audit real-environment fixtures as well as production adapters: a bare-CPU or source-only pass can skip a fixture that imports the real provider. Exercise that provider in the compatible Torch environment, preserve a focused failing case and retain all behavioral assertions when adapting the fixture. At a release-range transition update patch declarations, coarse compatibility policy, tests and maintained user guides together. The full-SHA pin remains authoritative.

For source relocations, represent deleted tracked files separately from present and new files. Preserve their original Git bytes, file modes and the patch before cleanup; [owned_source.py](scripts/owned_source.py) supplies these schema-2 captures. [source_transition.py](scripts/source_transition.py) verifies backup bytes and modes before restoring an exact reviewed snapshot. Recapture older snapshots that lack modes before cleanup. Prove target-added ignored paths cannot collide before using Git no-overwrite-ignore for the final switch. Do not rebuild a generated directory over unowned files. Distinguish intentional older-layout fixtures from current-corpus assertions.

## Package and publish

Keep upstream follow-up candidates separate from plugin fixes. Preserve each accepted pin as a checkoutable commit. Publish only to owned topic branches and create the requested PR sequence; do not merge without authorization. State exact checkpoint ranges and validation actually completed.

Bind model requests, metrics and shutdown logs to the same final wheel pair. Review both pre-shutdown and post-shutdown ERROR counts; a runner exit of zero does not waive either. Distinguish abort ordering from draining and retain genuine failure reporting in regressions.

After a completed milestone, reconcile this workflow with observed failures and validate reusable scripts. Keep machine paths, run logs and attempt receipts outside the project skill.
