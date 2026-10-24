#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts}"
VLLM_WHEEL_DIR="${ARTIFACT_DIR}/wheels/vllm"
SAIL_WHEEL_DIR="${ARTIFACT_DIR}/wheels/vllm-sail"
LOG_DIR="${ARTIFACT_DIR}/logs"
MANIFEST="${ARTIFACT_DIR}/build-manifest.txt"

VLLM_REPOSITORY="${VLLM_REPOSITORY:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-$(tr -d '[:space:]' < "${ROOT_DIR}/.github/vllm-main-verified.commit")}"
PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH:-ppu_15;ppu_10}"
# Building the native PPU kernels is memory-heavy (~2 GiB per compile job). The
# The CI build environment exposes many cores but a smaller memory limit, so an unbounded
# MAX_JOBS makes the OOM killer terminate the pod (exit 137). Derive a safe
# default from the cgroup memory limit (falling back to MemTotal) and apply a
# conservative ceiling. Override explicitly with BUILD_JOBS.
default_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
mem_bytes=0
if [[ -r /sys/fs/cgroup/memory.max ]]; then
    read -r mem_bytes </sys/fs/cgroup/memory.max || mem_bytes=0
    [[ "${mem_bytes}" == "max" ]] && mem_bytes=0
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    read -r mem_bytes </sys/fs/cgroup/memory/memory.limit_in_bytes || mem_bytes=0
fi
if [[ ! "${mem_bytes}" =~ ^[0-9]+$ ]] || ((mem_bytes <= 0 || mem_bytes > 1099511627776)); then
    mem_bytes="$(awk '/MemTotal/{print $2*1024}' /proc/meminfo 2>/dev/null || printf '0')"
fi
if [[ "${mem_bytes}" =~ ^[0-9]+$ ]] && ((mem_bytes > 0)); then
    mem_jobs=$((mem_bytes / (2 * 1024 * 1024 * 1024)))
    ((mem_jobs < 1)) && mem_jobs=1
    ((default_jobs > mem_jobs)) && default_jobs="${mem_jobs}"
fi
((default_jobs > 16)) && default_jobs=16
BUILD_JOBS="${BUILD_JOBS:-${default_jobs}}"

mkdir -p "${VLLM_WHEEL_DIR}" "${SAIL_WHEEL_DIR}" "${LOG_DIR}"
: >"${MANIFEST}"
exec > >(tee -a "${LOG_DIR}/build.log") 2>&1

WORK_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "${WORK_DIR}"
}
on_error() {
    local status=$?
    trap - ERR
    printf 'status=failed\nexit_code=%s\n' "${status}" >>"${MANIFEST}"
    exit "${status}"
}
trap cleanup EXIT
trap on_error ERR

fail() {
    printf '%s\n' "$1" >&2
    return 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

[[ "$(uname -s)" == "Linux" ]] || fail "native wheel builds require Linux"
for command_name in git python tee awk find; do
    require_command "${command_name}"
done
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.12" ]] || fail "the selected image requires Python 3.12"
{
    printf 'status=running\n'
    printf 'vllm_ref=%s\n' "${VLLM_REF}"
    printf 'pytorch_sail_arch=%s\n' "${PYTORCH_SAIL_ARCH}"
    printf 'build_image=%s\n' "${VLLM_SAIL_BUILD_IMAGE:-}"
} >>"${MANIFEST}"

[[ -r /usr/local/PPU_SDK/envsetup.sh ]] || fail "the image must contain the SAIL SDK"
# shellcheck disable=SC1091
source /usr/local/PPU_SDK/envsetup.sh
export PPU_SDK=/usr/local/PPU_SDK
export HGCC="${PPU_SDK}/bin/hgcc"
export PYTORCH_SAIL_ARCH
unset TORCH_CUDA_ARCH_LIST
unset VLLM_SAIL_HG_ARCH
unset VLLM_SAIL_SKIP_EXT
# Derive package versions from the selected source, never the image's overrides.
unset VLLM_VERSION VLLM_VERSION_OVERRIDE
unset SETUPTOOLS_SCM_PRETEND_VERSION SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM
unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST
unset VLLM_PRECOMPILED_WHEEL_LOCATION
# envsetup.sh repoints pip at an internal mirror (art.eng.t-head.cn) that is
# unreachable from the CI build environment; force the public Aliyun PyPI mirror after the
# SDK env is sourced. Override with SAIL_PIP_INDEX_URL.
export PIP_INDEX_URL="${SAIL_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_EXTRA_INDEX_URL=""
export PIP_CONSTRAINT="${WORK_DIR}/image-constraints.txt"

command -v hgcc
"${HGCC}" --version 2>&1 | tee "${LOG_DIR}/hgcc-version.log"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" capture \
    --sdk "${PPU_SDK}" --output "${LOG_DIR}/image-stack.json" \
    --constraints "${PIP_CONSTRAINT}" --require-ppu-headers | tee -a "${MANIFEST}"
python - <<'PY' 2>&1 | tee "${LOG_DIR}/torch-env.log"
from pathlib import Path
import torch

print(torch.__version__)
print(torch.__file__)
if not (Path(torch.__file__).parent / ".ppu_compat/compatible_wrapper.h").is_file():
    raise SystemExit("prepared SAIL PyTorch must contain PPU headers")
PY

VLLM_SRC="${WORK_DIR}/vllm"
git init "${VLLM_SRC}"
git -C "${VLLM_SRC}" remote add origin "${VLLM_REPOSITORY}"
# setuptools-scm needs history and release tags, including for commit/branch refs.
# A depth-1 checkout can otherwise produce an unverifiable 0.1.dev1 version.
for attempt in 1 2 3; do
    if git -C "${VLLM_SRC}" fetch --tags origin "${VLLM_REF}"; then
        break
    fi
    if ((attempt == 3)); then
        fail "failed to fetch vLLM after ${attempt} attempts"
    fi
    sleep "$((2 ** (attempt - 1)))"
done
git -C "${VLLM_SRC}" checkout --detach FETCH_HEAD
(
    cd "${VLLM_SRC}"
    export RUSTUP_DIST_SERVER="${RUSTUP_DIST_SERVER:-https://rsproxy.cn}"
    export RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://rsproxy.cn/rustup}"
    if ! command -v rustup >/dev/null 2>&1; then
        curl -k --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --profile minimal --default-toolchain none
    fi
    if [[ -r "${CARGO_HOME:-$HOME/.cargo}/env" ]]; then
        # shellcheck disable=SC1091
        source "${CARGO_HOME:-$HOME/.cargo}/env"
    fi
    # Fetch Cargo Git dependencies through the Git CLI to honor job-local Git configuration.
    export CARGO_NET_GIT_FETCH_WITH_CLI=true
    # Follow the Rust toolchain pinned by the checked-out vLLM source.
    RUSTUP_TOOLCHAIN="$(python - <<'PY'
import tomllib
from pathlib import Path

print(tomllib.loads(Path("rust-toolchain.toml").read_text())["toolchain"]["channel"])
PY
    )"
    export RUSTUP_TOOLCHAIN
    rustup toolchain install "${RUSTUP_TOOLCHAIN}" --profile minimal
    rustc --version | tee "${LOG_DIR}/rustc-version.log"
    cargo --version | tee "${LOG_DIR}/cargo-version.log"
    python tools/use_existing_torch.py
    python -m pip install -r requirements/build/cuda.txt
    export VLLM_TARGET_DEVICE=empty VLLM_REQUIRE_RUST_FRONTEND=1
    export MAX_JOBS="${BUILD_JOBS}" CARGO_BUILD_JOBS="${BUILD_JOBS}"
    # Compile explicitly: pip hides successful backend output by default, and
    # a wheel's presence alone does not demonstrate a local Cargo build.
    python tools/build_rust.py --release \
        2>&1 | tee "${LOG_DIR}/vllm-rust-build.log"
    python - <<'RUST_ARTIFACTS'
import runpy
from pathlib import Path

binary = Path("vllm/vllm-rs")
declared = runpy.run_path("tools/build_rust.py")["rust_py_extension_module_names"]()
extensions = []
for module in declared:
    paths = list(Path("vllm").glob(f"{module}.*.so"))
    if len(paths) != 1:
        raise SystemExit(f"required local Rust extension was not built: {module}")
    extensions.extend(paths)
if not binary.is_file() or binary.stat().st_size == 0:
    raise SystemExit("required local Rust artifacts were not built")
if set(Path("vllm").glob("_rust_*.so")) != set(extensions):
    raise SystemExit("unexpected local Rust extensions outside upstream declarations")
for artifact in (binary, *extensions):
    if artifact.stat().st_size == 0:
        raise SystemExit(f"empty Rust artifact: {artifact}")
    print(f"Local Rust artifact: {artifact} ({artifact.stat().st_size} bytes)")
RUST_ARTIFACTS
    # Upstream setup.py packages these locally built artifacts without rebuilding.
    python -m pip wheel --verbose --no-build-isolation --no-deps \
        --wheel-dir "${VLLM_WHEEL_DIR}" .
) 2>&1 | tee "${LOG_DIR}/vllm-build.log"

mapfile -t vllm_wheels < <(
    find "${VLLM_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print
)
((${#vllm_wheels[@]} == 1)) || \
    fail "expected exactly one vLLM wheel, found ${#vllm_wheels[@]}"
python -m pip install --force-reinstall --no-deps "${vllm_wheels[0]}"

# The CUDA-free build image intentionally does not carry runtime-only PPU
# packages such as tokenspeed-mla. They are protected by the image constraints
# and supplied by the separate CUDA-compatible runtime image, so install only
# the dependencies needed to compile the plugin here.
python -m pip install -r "${ROOT_DIR}/requirements/build.txt"
(
    cd "${ROOT_DIR}"
    python - <<'PY'
from vllm_sail.native.toolchain import decide, probe
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility()
decision = decide(probe())
print(decision.reason)
if not decision.build:
    raise SystemExit(1)
PY
) 2>&1 | tee "${LOG_DIR}/native-preflight.log"
(
    cd "${ROOT_DIR}"
    VERBOSE=1 MAX_JOBS="${BUILD_JOBS}" \
        python setup.py bdist_wheel --dist-dir "${SAIL_WHEEL_DIR}"
) 2>&1 | tee "${LOG_DIR}/vllm-sail-build.log"

python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" create \
    "${ARTIFACT_DIR}/wheels" --vllm-source "${VLLM_SRC}" --sail-source "${ROOT_DIR}"

mapfile -t sail_wheels < <(
    find "${SAIL_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print
)
((${#sail_wheels[@]} == 1)) || \
    fail "expected exactly one vLLM SAIL wheel, found ${#sail_wheels[@]}"
[[ "$(basename "${sail_wheels[0]}")" != *-none-any.whl ]] || \
    fail "native build unexpectedly produced a pure-Python wheel"
[[ -s "${vllm_wheels[0]}" ]] || fail "vLLM wheel is empty"
[[ -s "${sail_wheels[0]}" ]] || fail "vLLM SAIL wheel is empty"

python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk "${PPU_SDK}" --output "${LOG_DIR}/image-stack.json" \
    --require-ppu-headers | tee -a "${MANIFEST}"

{
    printf 'status=succeeded\n'
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'hgcc=%s\n' "$(hgcc --version 2>&1 | head -n 1)"
    printf 'rustc=%s\n' "$(cat "${LOG_DIR}/rustc-version.log")"
    printf 'cargo=%s\n' "$(cat "${LOG_DIR}/cargo-version.log")"
    printf 'vllm_commit=%s\n' "$(git -C "${VLLM_SRC}" rev-parse HEAD)"
    printf 'vllm_wheel=%s\n' "$(basename "${vllm_wheels[0]}")"
    printf 'vllm_sail_commit=%s\n' "$(git -c "safe.directory=${ROOT_DIR}" -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'vllm_sail_wheel=%s\n' "$(basename "${sail_wheels[0]}")"
} >>"${MANIFEST}"
