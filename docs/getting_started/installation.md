# Installation

vLLM SAIL runs on Linux hosts with a PPU accelerator and a compatible SAIL SDK.
For documentation and unit-test development on a machine without PPU hardware,
follow [Contributing](../../CONTRIBUTING.md#development-environment).

## 1. Prepare the SAIL software stack

Use an SDK environment with the following components installed together:

| Component | Requirement |
| --- | --- |
| SAIL SDK | Driver, HGGC headers, runtime libraries and `hgcc` compiler |
| SAIL PyTorch | A PPU build with `.ppu_compat/compatible_wrapper.h` |
| Triton | The [Triton for SAIL](https://github.com/t-head/triton-for-sail) build matching your SDK |
| Compute libraries | PPU builds of `acext`, `deep_gemm`, `flash_attn` and `flash_attn_3`, as needed by the selected backend |
| Model-specific libraries | PPU FlashMLA, PLA, TileLang or DeepEP when required by the workload |
| Python | 3.10–3.13, with matching SDK wheels |

Obtain the SDK and its Python packages through your PPU software distribution.
Use a matched set of libraries for the active SAIL PyTorch build. The plugin's
Python requirements do not install the SDK.

Activate that environment, then set paths for your installation:

```bash
export PPU_SDK=/path/to/ppu-sdk
export HGCC="$PPU_SDK/bin/hgcc"
export PYTORCH_SAIL_ARCH='ppu_15;ppu_10'
unset VLLM_SAIL_HG_ARCH
unset VLLM_SAIL_SKIP_EXT
```

CMake first uses an explicit `CMAKE_HGCC_DIR`, configured module path or
installed package. When none is available, it downloads the pinned
[cmake-hgcc](https://github.com/t-head/cmake-hgcc) revision and reuses it from
the build directory. For an offline build or a local mirror, set
`CMAKE_HGCC_DIR` to an existing checkout. Ensure the SDK runtime libraries are
on the system loader's search path, following your SDK's environment setup.

`PYTORCH_SAIL_ARCH` selects the device images included in the build. Use
`ppu_10`, `ppu_15`, or the quoted list above. The default includes both targets;
quoting prevents the shell from interpreting the semicolon as a command boundary.

## 2. Install matching vLLM

### Build both wheels from the checkout's verified vLLM commit

After activating the SAIL SDK and SAIL PyTorch environment from step 1, run
from your vLLM SAIL checkout on Linux (Bash 4+, Python 3.10–3.13):

```bash
BUILD_JOBS=8 bash scripts/build_wheels.sh
# Or select an empty output directory:
bash scripts/build_wheels.sh --output-dir /path/to/new-wheelhouse
```

The script reads `.github/vllm-main-verified.commit` from this checkout and
fetches that exact upstream commit into a temporary directory. It builds vLLM
with `VLLM_TARGET_DEVICE=empty` and `VLLM_REQUIRE_RUST_FRONTEND=1`, explicitly
compiles the Rust frontend and parser, installs the resulting vLLM wheel with
`--no-deps`, then builds the current vLLM SAIL sources. Both builds use the
active SAIL PyTorch without build isolation. No device is needed to compile.

Use a dedicated build environment: the script installs Python build requirements
and replaces its installed vLLM. It constrains SDK Python packages to their
installed versions, uses your current pip index configuration and verifies the
SDK/PyTorch stack after building. Runtime-only dependencies and the final
vllm-sail wheel are not installed. Rustup is installed if absent and downloads
the toolchain specified by upstream `rust-toolchain.toml`; the default Rust
mirrors are `rsproxy.cn`, overridable with `RUSTUP_DIST_SERVER` and
`RUSTUP_UPDATE_ROOT`. Network access is required for source, toolchain and
dependency downloads. `VLLM_REPOSITORY` may point to a local upstream Git mirror.

The default output is a fresh `artifacts/local-<run>/` directory containing:

- `wheels/vllm/*.whl` and `wheels/vllm-sail/*.whl`.
- `wheels/wheel-manifest.json` with source commits, wheel versions and SHA256 hashes.
- `build-manifest.txt` and `logs/`, including the explicit Rust build log.

The build rejects missing Rust artifacts and missing native SAIL extensions.
Output directories must be empty. Existing source checkouts are not reset or
cleaned; the temporary upstream checkout is removed on exit. The SAIL build
uses the current checkout, including local source edits. Wheel validation is
packaging evidence; use step 4 and the verification guide for runtime checks.
See `bash scripts/build_wheels.sh --help` for SDK, architecture, mirror and
parallelism controls. When `BUILD_JOBS` is omitted, parallelism is capped by
available memory and at most 16 jobs.

### Install a separately supplied vLLM build

Install a vLLM build with `VLLM_TARGET_DEVICE=empty` in the same environment.
This lets the PPU plugin provide the native operations used by the engine.
The plugin accepts `>=0.31.0,<0.32.0`; its generated upstream kernel sources use
**v0.31.0** as the reproducible source baseline.

Use a matching wheel supplied with your PPU environment, or build that upstream
release from source. For a source build, first prepare vLLM's build and runtime
dependencies using the SDK's package constraints. Then run from the vLLM checkout:

```bash
python tools/use_existing_torch.py
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation --no-deps .
```

The upstream helper removes dependency pins that would replace the active torch
installation. `--no-deps` assumes the runtime dependencies are already installed.
Keep the actual vLLM revision with your environment records.

## 3. Build and install the plugin

Clone the repository, then install its build dependencies in the same environment:

```bash
git clone https://github.com/t-head/vllm-sail.git
cd vllm-sail
python -m pip install -r requirements/build.txt -r requirements/ppu.txt
```

Check that the build will include native extensions:

```bash
python - <<'PY'
from vllm_sail.native.toolchain import decide, probe

decision = decide(probe())
print(decision.reason)
if not decision.build:
    raise SystemExit(1)
PY
```

Resolve any prerequisite reported by the check before continuing:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

Build isolation is disabled so the build can use the active SAIL PyTorch and
SDK. Native builds compile the authored plugin HGGC sources and committed
upstream corpus; installation requires neither sailify nor an upstream source
checkout.

For a distributable wheel and build controls, see [HGGC kernels](../developer_guide/kernels.md).

## 4. Check the installation

```bash
python -c "from vllm_sail import collect_env; collect_env()"
python - <<'PY'
import importlib
import torch
import vllm_sail
from vllm_sail.native.extensions import import_kernels

print('Plugin:', vllm_sail.__file__)
for name in ('_upstream_C', '_upstream_moe_C', '_C', '_moe_C'):
    extension = importlib.import_module(f'vllm_sail.{name}')
    print(name, extension.__file__)
print('Native preflight:', import_kernels(strict=True))
PY
```

The four libraries must load from the intended installation. Strict preflight
also checks representative operation registrations. These checks do not execute
a model; continue with the [quickstart](quickstart.md) and
[device verification](../user_guide/verification_guide.md).

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Build fails before CMake configuration | Check SAIL PyTorch, its compatibility header, `hgcc` and CMake. Native extensions are required by default. |
| Build unexpectedly produces a `py3-none-any` wheel | Unset `VLLM_SAIL_SKIP_EXT`; it is the only Python-only build path. |
| CMake cannot fetch the HG language modules | Restore network access or set `CMAKE_HGCC_DIR` to an existing cmake-hgcc checkout. |
| Architectures disagree | Set `PYTORCH_SAIL_ARCH`; unset `VLLM_SAIL_HG_ARCH` or make both settings use the same targets. |
| A shared library reports an undefined symbol | Match the SDK library to the active SAIL PyTorch and runtime, then restart the process. |
| Native extensions are missing after installing a wheel | Check `vllm_sail.__file__`. A checkout can shadow the installed package; use an editable native build there or run outside it. |
| Plugin is not selected | If `VLLM_PLUGINS` is set, include `ppu`; see [Configuration](../user_guide/configuration.md). |
| vLLM version is rejected | Use the supported release range. For a development build with ambiguous metadata, see [Configuration](../user_guide/configuration.md#version-metadata). |
