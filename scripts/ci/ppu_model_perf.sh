#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${WHEELS_DIR:?WHEELS_DIR is required}"
: "${RESULTS_DIR:?RESULTS_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${MODEL_REVISION:?MODEL_REVISION must identify the immutable checkpoint}"
: "${RUNTIME_IMAGE_ID:?RUNTIME_IMAGE_ID is required}"
: "${EXPECTED_VLLM_COMMIT:?EXPECTED_VLLM_COMMIT is required}"
: "${EXPECTED_SAIL_COMMIT:?EXPECTED_SAIL_COMMIT is required}"
mkdir -p "${RESULTS_DIR}"
exec > >(tee "${RESULTS_DIR}/model-perf.log") 2>&1
unset VLLM_VERSION PYTHONPATH GH_TOKEN GITHUB_TOKEN
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0 VLLM_PLUGINS=ppu
export VLLM_SAIL_PERF_COMMIT="${EXPECTED_SAIL_COMMIT}"
export EXPECTED_VLLM_COMMIT EXPECTED_SAIL_COMMIT RUNTIME_IMAGE_ID RESULTS_DIR
cd /tmp

python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}" \
    --vllm-commit "${EXPECTED_VLLM_COMMIT}" --sail-commit "${EXPECTED_SAIL_COMMIT}"
export PIP_CONSTRAINT="${RESULTS_DIR}/image-constraints.txt"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" capture \
    --sdk /usr/local/PPU_SDK --output "${RESULTS_DIR}/image-stack.json" \
    --constraints "${PIP_CONSTRAINT}"
verify_stack() {
    code=$?
    trap - EXIT
    if ! python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
        --sdk /usr/local/PPU_SDK --output "${RESULTS_DIR}/image-stack.json"; then
        code=1
    fi
    exit "${code}"
}
trap verify_stack EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mapfile -t vllm_wheels < <(find "${WHEELS_DIR}" -type f -name 'vllm-*.whl' -print | sort)
mapfile -t sail_wheels < <(find "${WHEELS_DIR}" -type f -name 'vllm_sail-*.whl' -print | sort)
((${#vllm_wheels[@]} == 1 && ${#sail_wheels[@]} == 1)) || {
    echo "Expected exactly one vLLM wheel and one SAIL wheel" >&2
    exit 2
}
python -m pip install --no-index --no-deps --force-reinstall \
    "${vllm_wheels[0]}" "${sail_wheels[0]}"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output "${RESULTS_DIR}/image-stack.json"

python - <<'PY'
import json
import os
from pathlib import Path

import torch
import vllm
import vllm_sail
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), current_platform
assert torch.cuda.is_available(), "model performance requires real PPU devices"
assert torch.cuda.device_count() == 8, "DP8+EP performance requires eight devices"
devices = []
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = tuple(torch.cuda.get_device_capability(index))
    assert capability == (8, 9) and "890" in name.upper(), (name, capability)
    devices.append({"name": name, "capability": capability})
imports = {"torch": torch.__file__, "vllm": vllm.__file__, "vllm_sail": vllm_sail.__file__}
assert all(not Path(path).resolve().is_relative_to("/source") for path in imports.values()), imports
identity = {
    "tested_sail_commit": os.environ["EXPECTED_SAIL_COMMIT"],
    "upstream_commit": os.environ["EXPECTED_VLLM_COMMIT"],
    "runtime_image_id": os.environ["RUNTIME_IMAGE_ID"],
    "devices": devices,
    "imports": imports,
    "vllm_version": vllm.__version__,
    "sail_version": vllm_sail.__version__,
    "torch_version": str(torch.__version__),
    "810e": "not tested: runner unavailable",
}
(Path(os.environ["RESULTS_DIR"]) / "runtime-identity.json").write_text(
    json.dumps(identity, indent=2) + "\n"
)
print(json.dumps(identity))
PY

python "${ROOT_DIR}/scripts/ci/ppu_forward_perf.py" \
    --model "${MODEL_PATH}" --model-revision "${MODEL_REVISION}" \
    --results-dir "${RESULTS_DIR}" --golden-root "${BASELINE_ROOT}"
