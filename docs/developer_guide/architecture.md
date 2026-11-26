# Architecture

vLLM SAIL provides the PPU implementation behind vLLM's inference interfaces.
The engine, scheduler and serving API remain owned by upstream vLLM. The plugin
owns PPU platform configuration, kernel selection, native extensions and the
model adaptations that require device-specific behavior.

## SAIL software stack

| Layer | Responsibility |
| --- | --- |
| vLLM | Model execution, scheduling, batching and serving |
| vLLM SAIL | Platform hooks, registrations, PPU operators and model integration |
| Compute libraries | PPU DeepGEMM, ACEXT, FlashAttention, Triton and workload-specific libraries |
| Native build | HGGC kernels compiled with `hgcc` through cmake-hgcc |
| SDK and device | SAIL PyTorch integration, PPU runtime, driver and accelerator |

The Python integration preserves the upstream interfaces required by SAIL
PyTorch. Do not rename tensor device identifiers, dispatch keys or operation
namespaces as part of a native-build change. Use `current_platform.is_ppu()`
when behavior is specific to PPU; broader inherited platform predicates do not
uniquely identify it.

## Plugin lifecycle

Both entry points use the allowlist name `ppu`, declared in
[`pyproject.toml`](../../pyproject.toml):

1. **Platform discovery:** `vllm_sail.register()` prepares early import adapters
   and returns the PPU platform class path. Keep this hook lightweight.
2. **General registration:** `vllm_sail.register_out_of_tree()` checks vLLM
   compatibility, installs patches, registers operators and kernels, and
   registers model implementations.
3. **Execution:** platform hooks and selected operators load the required PPU
   libraries and run the workload.

vLLM calls general plugins in the server, engine and worker processes. Each hook
must be safe to call repeatedly within a process. Import caching alone does not
provide this guarantee: registration has explicit guards, and incomplete work
must not be marked successful.

Provider replacement must also account for consumers that imported a callable
by value. When a patch changes a provider, test the final consumer path and any
already-loaded aliases, including optional modules.

## Source layout

| Path | Responsibility |
| --- | --- |
| [`vllm_sail/platform.py`](../../vllm_sail/platform.py) | PPU platform hooks and backend selection |
| [`vllm_sail/registry`](../../vllm_sail/registry) | Kernel registration, quantization configs and MoE backend integration |
| [`vllm_sail/attention`](../../vllm_sail/attention) | Attention adapters, FlashAttention shims and SDK integration |
| [`vllm_sail/model_executor`](../../vllm_sail/model_executor) | PPU layers, online quantization, kernels and tuning data |
| [`vllm_sail/models`](../../vllm_sail/models) | PPU model implementations, model-specific ops and registrations |
| [`vllm_sail/native`](../../vllm_sail/native) | Toolchain detection, source manifests, extension loading, Triton op fallbacks and capability checks |
| [`vllm_sail/patch`](../../vllm_sail/patch) | Reviewed runtime adaptations to upstream behavior |
| [`vllm_sail/utils`](../../vllm_sail/utils) | SDK wrappers (`acext`, `deep_gemm`, `deep_select`), MegaMoE helpers and DeepGEMM tuning data |
| [`vllm_sail/profiling`](../../vllm_sail/profiling) | Optional NVTX instrumentation and config-gated layerwise NVTX hooks |
| [`vllm_sail/benchmarks`](../../vllm_sail/benchmarks) | Profiling-aware serving benchmark wrapper |
| [`csrc/plugin`](../../csrc/plugin) | Authored HGGC kernels, headers, host bindings and their build manifest |
| [`csrc/upstream`](../../csrc/upstream) | Generated upstream HGGC kernels and their input manifest |
| [`tools`](../../tools) | Kernel generation, patch maintenance, env-doc generation and online-quantization validation |
| [`scripts`](../../scripts) | Local wheel builds, CI orchestration, model sanity and GSM8K evaluation runners |
| [`tests/ut`](../../tests/ut) / [`tests/e2e`](../../tests/e2e) | CPU merge gate / device correctness, accuracy and performance tests |
| [`tests/ppu`](../../tests/ppu) / [`tests/upstream_reuse`](../../tests/upstream_reuse) | Recipe-driven PPU CI policies / curated upstream vLLM test reuse |
| [`tests/tensor`](../../tests/tensor) / [`tests/integration`](../../tests/integration) | PyTorch numerical tests and mocked-device loader integration tests |

## Integration rules

- Prefer a registration API, then a platform hook, then an attention backend
  extension. Add a runtime patch only when those interfaces cannot express the
  required behavior. Follow the [patch guide](../../vllm_sail/patch/README.md).
- Keep version compatibility checks in `vllm_sail/compat.py` and patch selection,
  outside kernel hot paths.
- Declare runtime environment variables in `vllm_sail/envs.py`. Read them lazily;
  distinguish an explicit backend choice from an unset default.
- Defer optional SDK imports until a capability query or the selected execution
  path needs them. Preserve actionable import and ABI errors.
- Keep native build decisions separate from runtime registration. See
  [HGGC kernels](kernels.md) for source ownership and build configuration.

## Optional profiling

[`vllm_sail/profiling`](../../vllm_sail/profiling) owns optional instrumentation.
NVTX range instrumentation installs only when `VLLM_SAIL_NVTX_PROFILE` is enabled
and the required profiling libraries are available. Separately,
[`vllm_sail/profiling/layerwise.py`](../../vllm_sail/profiling/layerwise.py)
installs config-gated hooks for `--enable-layerwise-nvtx-tracing` for the V2
runner's target and draft/MTP models without requiring `VLLM_SAIL_NVTX_PROFILE` or PyPI
`nvtx`. Core inference and the CPU test environment must remain usable without
profiling dependencies. Keep instrumentation scoped to the lifecycle and
operation it measures.
