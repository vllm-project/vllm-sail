# Configuration

Set environment variables before starting vLLM so the server and worker processes
inherit the same configuration. Runtime defaults and metadata are defined in
[`vllm_sail/envs.py`](../../vllm_sail/envs.py); see the generated
[Runtime environment variables](environment_variables.md) reference for the full
table of variables, defaults, choices and aliases.

Every `VLLM_SAIL_*` runtime variable accepts the corresponding `VLLM_PPU_*`
name for compatibility. The SAIL name takes precedence whenever it is set,
including `0` or an empty string. An empty boolean is false; an empty backend
selection uses its default. Boolean truthy values are `true`, `1`, `yes` and
`on`, case-insensitively. `is_set()` recognizes either name, and `snapshot()`
reports effective values under SAIL names. `VLLM_DEEPEPLL_RECV_HOOK`,
`VLLM_GDN_DECODE_KERNEL`, and `VLLM_STEP5_INDEXER_FP8_AS_FP16` keep their
shared or upstream names and have no `VLLM_PPU_*` aliases. `VLLM_SAIL_NVTX_PROFILE`
and `VLLM_SAIL_DISABLE_MEGA_MHC` additionally accept `SAIL_NVTX_PROFILE` and
`SAIL_DISABLE_MEGA_MHC` after their `VLLM_PPU_*` aliases.

## Plugin discovery

vLLM discovers the installed plugin through package entry points. If you use
vLLM's plugin allowlist, include `ppu`:

```bash
export VLLM_PLUGINS=ppu
```

The same name enables platform discovery and the general hook that installs
registrations and model integrations. Include any other plugins needed by your
application in the allowlist.

## Backend selection

See [Backend selection](environment_variables.md#backend-selection) for the full
list of runtime backend toggles (`VLLM_SAIL_MOE_BACKEND`,
`VLLM_SAIL_DENSE_BACKEND`, `VLLM_GDN_DECODE_KERNEL`, `VLLM_SAIL_QSA_KERNEL`,
DeepGEMM/PLA/DeepSelect/DeepEP switches and fused quantization flags).

For example, to request the MoE DeepGEMM backend:

```bash
VLLM_SAIL_MOE_BACKEND=deepgemm vllm serve /path/to/model
```

Leave backend variables unset for normal capability-based selection.
An explicit choice still requires the corresponding library, device support and
compatible tensor layout. Avoid setting overlapping backend controls while
diagnosing a selection issue.

INT8 dense `deepgemm` is used for BF16 output without bias. Layers with bias
use ACEXT when selected and available, otherwise Triton fuses bias in its GEMM.
Symmetric INT8 dense layers fall back to Triton when neither DeepGEMM nor ACEXT
is selected; `VLLM_SAIL_DENSE_BACKEND=triton` selects that path directly. Triton
input quantization is controlled separately by `VLLM_SAIL_USE_TRITON_INT8_QUANT`.
Asymmetric INT8 activation quantization still requires the unsupported CUTLASS
zero-point path.

On PPU, GDN decode `auto` prefers PLA when the legacy `VLLM_SAIL_USE_PLA`
switch permits it and the model uses supported head dimensions and FP32 SSM
state. Otherwise it tries the registered native fused kernel, then Triton.
An explicit `pla`, `cuda` or `triton` choice overrides that legacy switch for
GDN decode only. Prefill and KDA retain their existing selection.

PLA decode requires `--mamba-ssm-cache-dtype float32` and K=V=128; the resolver
does not change an existing cache dtype. Missing dependencies or unsupported
static configurations fail early for explicit choices. Native `cuda` requires
a real CUDA dispatch implementation, not just an operation schema or an
unsupported-op stub. Its fused MTP kernel applies only to eligible batches;
other batches use the existing Triton path. The new HGGC port still requires
native build and device correctness verification before deployment.

On PPU, Qwen4Exp QSA (query-sparse attention) `auto` selects FA3's QSA kernel
when every gate passes and otherwise keeps the upstream Triton kernel, logging
the reason once per distinct decision. The gates are: a BF16 main KV cache
(FA3's QSA path rejects FP8 outright), head dimension 256, at most 32 query
heads per KV head, and a `flash_attn_3` build whose QSA entry point carries the
`qsa_allow_aiu` argument and passes a numerical self-test. `cuda` and `triton`
pin either side for A/B measurement; an explicit `cuda` that fails a gate raises
at layer construction rather than falling back silently. Selection is per layer,
so a model whose layers disagree on KV-cache dtype resolves each independently.

Note that FA3's hand-tuned QSA kernels require one KV head per rank under the
interleaved KV-cache layout this pin allocates. Per-rank KV heads are
`max(1, num_key_value_heads // tp_size)`, so with this model's two KV heads any
even tensor-parallel size of 2 or more replicates them down to one per rank and
reaches a tuned kernel, while tensor-parallel size 1 keeps two and runs FA3's
QSA-aware generic kernel. Correctness is unaffected. Use `triton` to compare
against the Triton kernel on your own shape.

## DeepSeek V4.1

The main integration retains upstream V4.1 target/DSpark execution, shared KV
and index metadata, and adds PPU attention, delayed mHC, channelwise dense/Engram
quantization and checkpoint-name mappings. These source adaptations still need
device, model-load and inference verification; see the
[verification guide](verification_guide.md).

### MegaMoE

On Zhenwu M890, select MegaMoE with expert parallelism:

```bash
vllm serve /path/to/deepseek-v41 \
  --enable-expert-parallel --moe-backend deep_gemm_mega_moe
```

The mode selects the activation precision for MXFP4 routed expert weights.
All three modes return BF16 outputs. Install a
[DeepGEMM-for-sail](https://github.com/t-head/DeepGEMM-for-sail) build exposing
the selected kernel and its weight transform, where applicable.

| Mode | `VLLM_SAIL_MEGA_MOE_MMA_TYPE` | Activation | Expert dimensions |
| --- | --- | --- | --- |
| W4A4 | `fp4xfp4` | SwiGLU, SiTU | Hidden ≥ 2048 and divisible by 512. Intermediate divisible by 64. |
| W4A8 (default) | `fp8xfp4` | SwiGLU, SiTU | Hidden divisible by 256. Intermediate divisible by 128. |
| W4A16 | `bf16xfp4` | SiTU | Hidden divisible by 256. Intermediate divisible by 128. |

DeepSeek V4/V4.1 uses SwiGLU and can select W4A4 or W4A8. Kimi K3's native
latent MoE uses SiTU and requires an eligible MXFP4 checkpoint. The platform
gate checks hardware and DeepGEMM APIs without a model-name allow-list and
accepts EPLB configuration; the expert implementation checks the actual geometry,
activation and EPLB compatibility. Models without `moe_intermediate_size` defer
the shape check to expert construction. W4A8 still requires EPLB to be disabled
because rebalancing can change the inverse weight scale captured by the kernel.
W4A16 refreshes its transformed weight banks each step when EPLB is active.
Restart the server to replace checkpoint weights
with W4A8 or W4A16, whose transformed banks are bound by CUDA graphs. Shared
experts run as a separate MLP.

Upstream DeepSeek uses GateLinear with fused top-k for small local batches.
Larger local batches require a DeepGEMM build exposing `bf16_mega_gate`.

`VLLM_SAIL_MOE_BACKEND=deepgemm` continues to select the ordinary PPU DeepGEMM
expert backend. Use the CLI selection above for MegaMoE.

Indexer precision is independent of the expert mode. On M890, DeepSeek V4
and V4.1 default to MXFP4 indexer Q/K with BF16 scoring weights. GLM-5 and
GLM-5-Next retain their FP8 indexers with FP32 scoring weights; GLM-5-Next
uses the legacy FP8 DeepGEMM API and pads its query heads to at least 32.
On Zhenwu 810E, the GLM indexer uses INT8. Selecting MXFP4 indexer cache
explicitly is supported only for DeepSeek V4/V4.1 on M890. These format
rules do not enable MegaMoE for GLM's ordinary FusedMoE path.

The integration retains upstream's `SparseAttnIndexer` for dense
scoring, including candidate masking. V4/V4.1 query producers use the selected
operator's `weights_dtype` contract. The separate candidate-only
`SparseMQAIndexer` path can be enabled for DeepSeek V4.1 on M890 with
`--attention-config '{"indexer_sparse_logits":true,"indexer_kv_dtype":"mxfp4"}'`.
It requires the SAIL DeepGEMM sparse APIs, the `deep_select==1.0.0` wheel,
`VLLM_SAIL_USE_DEEPSELECT=1` (default), and top-k backend `auto` or
`deep_select`. The candidate-source layer still uses dense scoring; later
indexers score only its candidate blocks and return request-local token indices.

This path requires 32 indexer heads, head dimension 128, candidate blocks of
8/16/32 positions, and a power-of-two candidate count yielding 4–4096 sparse
blocks divisible by four. Top-k must fit the candidate width and be at most
4096; upstream DeepSelect row-alignment constraints still apply. Context
parallelism and non-MXFP4 indexer caches are rejected. Missing or disabled
dependencies fail at startup instead of silently selecting dense scoring.
Keep the option disabled to use the dense indexer. PPU inference correctness
and performance for the candidate-only path still require device validation.

Prefill chunking retains upstream's chunker and BF16 budget conversion for the
sparse backend. The PPU sparse builder includes SAIL's physical row padding in
that budget; the dense path keeps the conservative upstream FP32 budget.

### Mega mHC

On Zhenwu M890 (`ZW-M890P`, capability 8.9), DeepSeek V4.1 automatically uses
the PPU DeepGEMM Mega mHC kernel for `hidden_size=5120`, `hc_mult=4` when a
DeepGEMM build exposing `mega_mhc` is installed. Set
`VLLM_SAIL_DISABLE_MEGA_MHC=1` (also accepts `VLLM_PPU_DISABLE_MEGA_MHC` and
`SAIL_DISABLE_MEGA_MHC`) to disable the PPU Mega mHC gate and fall back to the
upstream TileLang mHC path for A/B bisection.

### Fused norm-rope sparse attention

An opt-in PPU fused norm-rope sparse-attention path is available for DeepSeek
V4.1 when using a compatible `flash_mla` wheel. Enable it with
`VLLM_SAIL_FUSED_NORM_ROPE_ATTN=1` and `VLLM_SAIL_PPU_KV_MXFP8=1` (528-byte
MXFP8 KV record, read once per process); optional Zhenwu M890 extensions are
`VLLM_SAIL_PPU_KV_NVFP4=1` (lossy 288-byte NVFP4 compressed KV record when
`sliding_window % 8 == 0`) and `VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE=1` (exact
per-token FP32 output scales). See
[DeepSeek V4.1 variables](environment_variables.md#deepseek-v41) for exact
defaults and aliases.

When any capability, wheel, or topology gate is closed, the selector falls back
to the baseline attention class and logs the closing gate.

## Model-specific sparse attention

### GLM-5-Next kpool

On Zhenwu 810E, the INT8 kpool indexer defaults to
`VLLM_SAIL_KPOOL_DECODE_LOGITS=non_paged` (`paged` is rejected on PPU) and
guards its dense decode fallback with `VLLM_SAIL_KPOOL_MAX_DENSE_POOLS`. See
[GLM-5-Next kpool variables](environment_variables.md#glm-5-next-kpool).

### MiniMax M3 SAIL MSA

MiniMax M3 / M3.1-Flash selects SAIL MSA sparse attention (`msa_for_sail`) by
default on PPU (`VLLM_SAIL_MINIMAX_M3_MSA=1`) when the wheel, architecture, and
model-shape gates pass (`block_size=128`, `head_dim=128`,
`sparse_index_dim=128`, `sparse_topk_blocks=16`, `sparse_score_type="max"`,
matching KV/index head counts, and context within the SAIL tile limit). On
Zhenwu M890 with `--kv-cache-dtype fp8`, `auto` indexer KV dtype defaults to
FP8 with BF16 top-k logits; on Zhenwu 810E or with a BF16 KV cache, SAIL MSA
uses BF16 index/attend with FP32 logits. Selecting SAIL MSA downgrades full
CUDA graphs to `PIECEWISE`. Use `VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_TRITON=1` or
`VLLM_SAIL_MINIMAX_M3_MSA_ATTEND_TRITON=1` to bisect either half against
Triton; see
[MiniMax M3 SAIL MSA variables](environment_variables.md#minimax-m3-sail-msa).

### Step-5 DSA

By default on PPU, Step-5 DSA runs upstream's Triton SM90 indexer scorer
(native E4M3 on Zhenwu M890, software E4M3 on Zhenwu 810E) and FlashInfer
prefill attention, with SAIL DeepSelect used for `auto` / `deep_select` region
selection on Zhenwu M890 when `VLLM_SAIL_USE_DEEPSELECT=1`. Opt-in switches
cover the resident-block Triton scorer (`VLLM_SAIL_STEP5_DSA_SCORES`), native
HGGC radix selector (`VLLM_SAIL_STEP5_DSA_SELECTOR_PPU`), native HGGC BF16
paged prefill/decode (`VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND=native`,
`VLLM_SAIL_STEP5_DSA_PREFILL_PPU`,
`VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION`), and FP16 indexer storage
(`VLLM_STEP5_INDEXER_FP8_AS_FP16`, set before startup). See
[Step-5 DSA variables](environment_variables.md#step-5-dsa).

## Compatibility controls

For FP8/BF16 conversion and recipe selection, see [Online quantization](online_quantization.md).
`VLLM_SAIL_FP8_LOAD_POLICY=native|bf16|int8` applies only when no explicit
quantization argument was supplied. Leaving it unset preserves native behavior;
the guide describes legacy opt-in variable precedence and source-format limits.
Legacy MoE and GDN switches (`VLLM_SAIL_DISABLE_MOE_WNA16_CUDA`,
`VLLM_SAIL_FORCE_MOE_WNA16_CUDA`, `VLLM_SAIL_ENABLE_MOE_MARLIN`,
`VLLM_SAIL_FUSED_GDN_DECODE`) are listed in
[Online quantization and compatibility variables](environment_variables.md#online-quantization-and-compatibility).

## Profiling

Enable opt-in NVTX instrumentation with `VLLM_SAIL_NVTX_PROFILE=1` (also
accepts `VLLM_PPU_NVTX_PROFILE` and `SAIL_NVTX_PROFILE`), along with optional
label detail flags (`VLLM_SAIL_NVTX_DUMP_DICT`, `VLLM_SAIL_NVTX_DUMP_TOPK`,
`VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN`); see
[Profiling variables](environment_variables.md#profiling).
The profiling integration requires the optional `nvtx` package; install it with
`python -m pip install --no-build-isolation -e '.[profiling]'` from the repository
root. The SAIL SDK's optional `model_prof` package adds iteration hooks, and
`python -m vllm_sail.benchmarks.serve` provides a profiling-aware serving
benchmark wrapper (`--skip-first-concurrency`).

Separately, `--enable-layerwise-nvtx-tracing` is extended on PPU for the V2
runner's target and draft/MTP models with shared-module deduplication. It does not
require `VLLM_SAIL_NVTX_PROFILE` or the optional `nvtx` package; use
`--enforce-eager` to observe per-layer markers without compilation or CUDA
graph replay bypassing module hooks.

## Version metadata

`vllm_sail.compat.check_vllm_compatibility()` checks the installed vLLM build
against the `upstream_ref` recorded in the packaged kernel manifest
(`csrc/upstream/manifest.toml`). By default, a mismatch logs a warning and
continues so developers can validate neighboring upstream commits via API and
functional tests. Set `VLLM_SAIL_STRICT_COMPAT=1` to reject a mismatch with a
`RuntimeError`. CI passes `strict=True`.

Commit-pinned installations match the full 40-character source SHA reported by
the installed vLLM package. Release-tag manifests match the major and minor
version. For those manifests, `VLLM_VERSION` can state the actual release version
when a source build reports an ambiguous SCM prefix.

## Inspect the active environment

```bash
python -c "from vllm_sail import collect_env; collect_env()"
```

The report includes versions, device information, SDK package availability and
the PPU environment-variable snapshot. Package presence is a discovery check;
successful loading and execution require the
[verification steps](verification_guide.md).

`PPU_SDK` is also read at runtime and defaults to `/usr/local/PPU_SDK`.

Build-time settings such as `PPU_SDK`, `HGCC`, `CMAKE_HGCC_DIR` and
`PYTORCH_SAIL_ARCH` are documented in [HGGC kernels](../developer_guide/kernels.md#build-settings).
