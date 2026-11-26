# Supported Features

This page describes implemented PPU integration paths and their constraints.
An implementation entry is not a qualification of every device, SDK version or
checkpoint. Use the [verification guide](verification_guide.md) to establish
support for a specific deployment.

## Platform and compatibility

| Component | Scope |
| --- | --- |
| Hardware | Zhenwu 810E and Zhenwu M890 AI chips |
| Native targets | Zhenwu 810E (`ppu_10`, compute capability 8.0) and Zhenwu M890 (`ppu_15`, compute capability 8.9) |
| Bundled tuning data | Configurations include `PPU-ZW810` and `PPU-ZW810E` (Zhenwu 810E) and `ZW-M890P` (Zhenwu M890); coverage varies by shape and backend |
| vLLM | Verified upstream ref in [verified pin](../../.github/vllm-main-verified.commit) and [kernel manifest](../../csrc/upstream/manifest.toml), built with `VLLM_TARGET_DEVICE=empty` |
| Python | 3.10–3.13 for the plugin and CPU test matrix; SDK wheel availability determines runtime combinations |

A tuned configuration describes a measured kernel shape. It does not establish
full-model compatibility. A missing configuration may use backend defaults and
need workload-specific tuning.

## Compute and attention

| Area | Implementation | Requirements and limits |
| --- | --- | --- |
| Dense linear layers | PPU DeepGEMM, ACEXT and Triton paths | Selection depends on precision, tensor layout, device and installed libraries |
| Mixture of experts | PPU DeepGEMM (including MegaMoE), ACEXT and Triton backends for unquantized and selected quantized experts | Backend support varies with expert shape and quantization format; MegaMoE (`--moe-backend deep_gemm_mega_moe`) targets MXFP4 experts on Zhenwu M890 |
| FlashAttention | PPU FlashAttention 2 and 3 integration (including FA3 QSA for Qwen4Exp) | Matching SDK libraries; version selection follows availability and model constraints |
| MLA and sparse attention | PPU FlashMLA integration (dense, FP8 dense decode, sparse, and opt-in DeepSeek V4.1 fused norm-rope sparse attention) and sparse indexers | Matching PPU libraries (`flash_mla`, `deep_gemm`, `deep_select`); opt-in fused V4.1 path requires a compatible `flash_mla` wheel and `VLLM_SAIL_PPU_KV_MXFP8=1` |
| Model-specific sparse attention | MiniMax M3 / M3.1 SAIL MSA (`msa_for_sail`), Step-5 DSA, GLM-5-Next kpool, and Qwen4Exp QSA | Library, chip-capability and shape gates apply; see [Configuration](configuration.md) |
| Gated delta networks | PLA, native HGGC fused MTP decode and Triton paths for GDN | `VLLM_GDN_DECODE_KERNEL`, PLA availability and supported shapes determine dispatch |
| Kimi delta attention | PLA and Triton paths for KDA | PLA prefill retains final recurrent state; intermediate prefill checkpoints are disabled. Backend and sequence-shape constraints apply |
| Native operations | HGGC activation, normalization, quantization, routing, cache, sampling and model-specific kernels, plus Triton-backed registrations (such as `merge_attn_states`) | Only the selected source corpus is compiled; each changed kernel needs device correctness tests |
| Distributed MoE | DeepEP integration, including low-latency receive handling | Compatible PPU communication libraries and a validated multi-device setup |

## Precision and quantization

| Format | Implemented scope | Constraints |
| --- | --- | --- |
| BF16 / FP16 | Dense, attention and unquantized MoE paths | Individual backends may accept only a subset; BF16 dense DeepGEMM is opt-in on Zhenwu M890 |
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

Loading-time `fp8-to-int8`, `bf16-to-int8`, and `fp8-to-bf16` paths are described in
[Online quantization](online_quantization.md). They accept E4M3 128x128 block FP8
or unquantized BF16 checkpoints and reuse existing Dense/MoE backends. Model
recipes select target precision; fallback uses checkpoint policy for FP8 and
the bundled default recipe for BF16. Device and model qualification is required.

| Model family or path | PPU integration |
| --- | --- |
| Dense and MoE models | Upstream vLLM architectures using the applicable PPU operators and backends |
| DeepSeek V4 | Model, MTP and DSpark registrations, with PPU attention, routing and cache paths |
| DeepSeek V4.1 | Target and DSpark adaptation with PPU FlashMLA, delayed mHC and Zhenwu M890 Mega mHC, MegaMoE, candidate-only `SparseMQAIndexer`, opt-in fused norm-rope sparse attention, and mixed-precision quantization |
| GLM-5 / GLM-5-Next / GLM-5.3 | `GlmMoeDsaForCausalLM` registration, FP8/INT8 indexers, and kpool sparse attention (INT8 kpool on Zhenwu 810E, FP8 on Zhenwu M890) |
| MiniMax M3 / M3.1-Flash | Optional model registrations with `SupportsQuant`, SAIL MSA sparse attention (`msa_for_sail`), quantization/channelwise mappings, DSpark attention and native fused QK-norm/RoPE/KV-insert operations |
| Qwen hybrid and Qwen4Exp models | GDN attention, Qwen4Exp QSA (FA3 QSA kernel and Zhenwu M890 DeepGEMM MXFP4 compressed-key scoring), PLE quantization mapping and speculative-decoding adaptations |
| Step-5 | Step-5 DSA attention backend (native E4M3 on Zhenwu M890, software E4M3 on Zhenwu 810E, optional FP16 indexer storage) and opt-in native HGGC prefill and radix selector |
| Kimi K3 | KDA attention, native MLA/cache operations, and dense EAGLE3/DSpark MLA drafts retargeted to native FP8 `FLASHMLA` |
| Mamba / Jamba | HGGC selective-scan source integration |

Model registration and kernel availability are separate checks. Record the exact
checkpoint revision, quantization, context length, parallelism and SDK when
reporting a successful model run. Newly compiled model-specific operations need
numerical and model-level validation on each target device.

## Selecting a backend

See [Configuration](configuration.md) for runtime choices and defaults. Build
coverage is defined by the [native source manifests](../developer_guide/kernels.md#source-ownership);
an excluded operation fails with a capability diagnostic rather than being
enabled by an environment variable.

The GLM-5.3-Flash INT8 kpool path on Zhenwu 810E requires decode context
parallel size 1. It rejects larger sizes until distributed cache mapping and
global top-k merging are implemented. Zhenwu M890 keeps the upstream FP8 path.

The INT8 kpool path rejects `VLLM_SAIL_KPOOL_DECODE_LOGITS=paged` because the
paged kernel does not produce complete logits on PPU. Use the default
`non_paged` backend.
