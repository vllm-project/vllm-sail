# Kernel Benchmarks

These tools measure PPU kernels and generate workload-specific tuning data.
Prepare the [SAIL environment](../docs/getting_started/installation.md) before
running them. They require a PPU device and are separate from the CPU unit tests.

| Tool | Purpose |
| --- | --- |
| [`kernels/benchmark_ep_scatters.py`](kernels/benchmark_ep_scatters.py) | Expert-parallel scatter correctness and latency checks |
| [Fused MoE tuning](kernels/fusedmoe_tuning_tools/README.md) | Compile candidate Triton configurations, profile them and select configurations for a workload |
| [`vllm_sail/benchmarks/serve.py`](../vllm_sail/benchmarks/serve.py) | Profiling-aware serving benchmark (`python -m vllm_sail.benchmarks.serve`, supporting `--skip-first-concurrency` with SDK `model_prof`) |

Record the PPU card, SAIL SDK and PyTorch versions, vLLM revision, tensor shapes
and precision with benchmark results. Validate numerical correctness before
using a tuning result in a deployment. Keep machine-specific profiles and raw
measurement data outside version control; review reusable tuning configurations
before adding them to the package.
