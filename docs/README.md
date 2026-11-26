# vLLM SAIL Documentation

vLLM SAIL connects vLLM to the SAIL software stack on PPU accelerators.
Start with installation and quickstart, then use the configuration and developer
guides for your workload.

## Getting started

1. [Install vLLM SAIL](getting_started/installation.md): prepare the SDK and build
   the native plugin.
2. [Run inference](getting_started/quickstart.md): start a server or use the
   Python API.
3. [Verify your environment](user_guide/verification_guide.md): check extension
   loading, kernel correctness and model execution.

## User guides

- [Supported features](user_guide/supported_features.md): backends, precision
  formats and model integration constraints.
- [Configuration](user_guide/configuration.md): runtime backend selection and
  PPU-specific settings.
- [Environment variables](user_guide/environment_variables.md): generated
  reference of all runtime variables, defaults, choices and aliases.
- [Online quantization](user_guide/online_quantization.md): loading-time
  `fp8-to-int8`, `bf16-to-int8` and `fp8-to-bf16` conversion and recipes.

## Developer guides

- [Contributing](../CONTRIBUTING.md): local setup, tests, PPU CI commands and
  code review.
- [Architecture](developer_guide/architecture.md): module responsibilities and
  plugin lifecycle.
- [HGGC kernels](developer_guide/kernels.md): source manifests, generation and
  native extension builds.
- [Runtime patch maintenance](../vllm_sail/patch/README.md): extension points,
  patch metadata and upstream drift checks.
- [PPU Recipe CI Matrix](../tests/ppu/README.md): generated PR `fast_sanity`
  and Nightly accuracy matrix.
- [Kernel benchmarks](../benchmarks/README.md): device measurements, MoE tuning
  and profiling-aware serving benchmarks.

[Return to the project overview](../README.md).
