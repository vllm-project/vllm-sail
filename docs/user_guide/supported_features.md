# Supported Features

This page describes implemented PPU integration paths and their constraints.
An implementation entry is not a qualification of every device, SDK version or
checkpoint. Use the [verification guide](verification_guide.md) to establish
support for a specific deployment.

## Platform and compatibility

| Component | Scope |
| --- | --- |
| Hardware | Zhenwu 810E and Zhenwu M890 AI chips |
| Native targets | PPU 1.0 (`ppu_10`) and PPU 1.5 (`ppu_15`) |
| Bundled tuning data | Configurations include `PPU-ZW810E` (Zhenwu 810E) and `ZW-M890P` (Zhenwu M890); coverage varies by shape and backend |
| vLLM | `>=0.31.0,<0.32.0`; generated native sources pinned to v0.31.0 |
| Python | 3.10–3.13 for the plugin and CPU test matrix; SDK wheel availability determines runtime combinations |

A tuned configuration describes a measured kernel shape. It does not establish
full-model compatibility. A missing configuration may use backend defaults and
need workload-specific tuning.

## Compute and attention

| Area | Implementation | Requirements and limits |
| --- | --- | --- |
| Dense linear layers | PPU DeepGEMM, ACEXT and Triton paths | Selection depends on precision, tensor layout, device and installed libraries |
| Mixture of experts | PPU backends for unquantized and selected quantized experts | Backend support varies with expert shape and quantization format |
| FlashAttention | PPU FlashAttention 2 and 3 integration | Matching SDK libraries; version selection follows availability and model constraints |
| MLA and sparse attention | PPU FlashMLA integration and sparse indexer | Matching PPU libraries; FP8 dense FlashMLA decode is unavailable |
| Gated delta networks | PLA and Triton paths for GDN | PLA availability and supported shapes determine dispatch |
| Kimi delta attention | PLA and Triton paths for KDA | PLA prefill retains final recurrent state; intermediate prefill checkpoints are disabled. Backend and sequence-shape constraints apply |
| Native operations | HGGC activation, normalization, quantization, routing, cache, sampling and model-specific kernels | Only the selected source corpus is compiled; each changed kernel needs device correctness tests |
| Distributed MoE | DeepEP integration, including low-latency receive handling | Compatible PPU communication libraries and a validated multi-device setup |

## Precision and quantization

| Format | Implemented scope | Constraints |
| --- | --- | --- |
| BF16 / FP16 | Dense, attention and unquantized MoE paths | Individual backends may accept only a subset; BF16 dense DeepGEMM is opt-in on PPU 1.5 |
| INT8 | Quantized dense and MoE compute | Scaling mode and weight layout must match the selected backend |
| FP8 | Dense, MoE and selected cache paths | Device-specific representation and scaling requirements; not every attention path accepts FP8 |
| MXFP8 | Dense weight dequantization to BF16 using upstream emulation | FP8 E4M3 values with E8M0 scales for each 32 elements along K; execution uses BF16/FP16 linear operations |
| MXFP4 | PPU DeepGEMM dense and MoE paths | Requires the matching library and checkpoint layout |
| INT4 W4A16 | Selected mixed-precision MoE paths | The compressed-tensors path accepts symmetric group-size-32 weights without zero points, expert bias or activation ordering |

Checkpoint format support is specific to the implementation. Weight repacking
alone does not establish execution support. Native Marlin execution remains
unavailable; do not infer general GPTQ or AWQ coverage from packing utilities.

### MXFP8 dense emulation

PPU selects `PPUEmulationMxfp8LinearKernel` before CUDA-only MXFP8 kernels.
It inherits upstream's loading and execution methods and is also available
through `--linear-backend emulation`. Non-PPU devices retain upstream selection.
This path applies to MXFP8 dense layers independently of model architecture;
MoE backend selection is separate.

For a weight of shape `[N, K]`, K must be divisible by 32. Runtime scales have
shape `[N, K / 32]`. ModelOpt expands checkpoint row-block scales, such as
`[32, 32]`, to one scale row per weight row while preserving the E8M0 bytes.
The emulation path does not impose FP8 blockwise K128 constraints or pad K.

Upstream's `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=1` is the default. It replaces
the FP8 weight with BF16 once after loading; subsequent linear calls use that
weight directly, cast to the activation dtype when needed. Dense-weight storage
doubles compared with the one-byte FP8 values, and the scale tensor is retained.
With `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=0`, the weight remains FP8 and each
linear call dequantizes it to a temporary BF16 tensor. Both settings use the
original E8M0 scale (`2 ** (scale_byte - 127)`) and execute the emulation path.

On a prepared PPU environment, verify both settings with
`python -m pytest tests/e2e/test_mxfp8_emulation.py -q`. These numerical tests
cover checkpoint scale expansion and K64/K576 output correctness; CPU/source
tests do not establish device correctness or full-model support.

## Model integration

| Model family or path | PPU integration |
| --- | --- |
| Dense and MoE models | Upstream vLLM architectures using the applicable PPU operators and backends |
| DeepSeek V4 | Model, MTP and DSpark registrations, with PPU attention, routing and cache paths |
| DeepSeek V4.1 | Upstream CUDA target and DSpark models with PPU FlashMLA, delayed mHC and mixed-precision checkpoint overrides; see the format constraints below |
| MiniMax M3 | Optional model registrations, quantization mapping and native attention-related operations |
| Qwen hybrid models | GDN attention and selected quantization and speculative-decoding adaptations |
| Kimi K3 | KDA attention and native MLA/cache operations |
| Mamba / Jamba | HGGC selective-scan source integration |

Model registration and kernel availability are separate checks. Record the exact
checkpoint revision, quantization, context length, parallelism and SDK when
reporting a successful model run. Newly compiled model-specific operations need
numerical and model-level validation on each target device.

### DeepSeek V4.1 checkpoint formats

The PPU path reuses upstream V4.1 model execution, shared KV metadata and DSpark
loading. Select `--moe-backend ppu_deep_gemm` and set `VLLM_USE_DEEP_GEMM=1` for
MXFP4 routed experts. Dense precision follows the checkpoint:

| Checkpoint | Dense layers | Engram embedding |
| --- | --- | --- |
| FP8 with `weight_block_size: [32, 32]` and `expert_dtype: fp4` | Upstream MXFP8 loading and BF16 emulation; this increases resident dense-weight memory by default | Upstream FP8 values with E8M0 block scales |
| MXFP4 with an explicit `fp8_channelwise_layers` list | Listed layers use PPU FP8 kernels with FP32 per-output-channel weight scales and dynamic per-token activation scales; ignored and unlisted dense layers stay unquantized | A listed `engram.embed` uses FP8 values and one FP32 scale per row |

The mixed-format override accepts `expert_dtype: mxfp4` and normalizes the
resolved target and draft configs to upstream's `fp4` spelling. It does not
rewrite checkpoint files or requantize weights. Per-layer precision lists must
include all shards of a fused projection; partial matches are rejected.

For host-resident Engram tables use `--engram-config '{"cpu_offload":true}'`.
Channelwise Engram supports the upstream sharding, UVA lookup and prefetch path;
its `dp_shared_memory` mode is not implemented. DSpark adaptive verification
requires full CUDA graphs in upstream vLLM, so disable adaptive verification
when isolating failures with `--enforce-eager`.

These format adaptations do not establish checkpoint accuracy or full-model
qualification. Requantizing block-scaled FP8 into per-channel FP8 is lossy;
validate generated outputs and task accuracy for the exported checkpoint.

## Selecting a backend

See [Configuration](configuration.md) for runtime choices and defaults. Build
coverage is defined by the [native source manifests](../developer_guide/kernels.md#source-ownership);
an excluded operation fails with a capability diagnostic rather than being
enabled by an environment variable.
