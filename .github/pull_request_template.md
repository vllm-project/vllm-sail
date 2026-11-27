### What this PR does / why we need it?

<!-- Describe the problem and the change. Link related issues if applicable. -->

### Does this PR introduce *any* user-facing change?

<!-- Describe changes to behavior, APIs or configuration, or write None. -->

### How was this patch tested?

<!-- List test commands and results. State what was not run and why. -->

### Model Recipe Sign-off (required when modifying `tests/ppu/recipes/`, `tests/ppu/policy/`, or model E2E cases)

- [ ] **Preserved capabilities & intentional command diff**: Required serving capabilities (topology/parallelism, quantization/KV dtype, tool-call & reasoning parsers, speculative decoding) are preserved, and every `base -> head` `argv` / `env` / `tier` change in the CI `Resolved Command Diff` summary is intentional.
- [ ] **Behavioral assertion beyond HTTP 200**: Verified non-empty generation (`content` or `reasoning_content`), complete SSE `[DONE]` framing, and/or bounded GSM8K golden accuracy — not bare server startup or HTTP 200 health checks.
- [ ] **Legacy deduplication & verification links**: Updated `replaces_legacy_cases` (if retiring a legacy case), passed `python3 scripts/ci/ppu_recipe.py validate --all` and `render-matrix --check`, and linked the CI / PPU verification evidence or tracking issue.
