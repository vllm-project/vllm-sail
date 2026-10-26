#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resolve the runner-configured CUDA-compatible PPU runtime to an immutable ID.
set -Eeuo pipefail

: "${PPU_CI_RUNTIME_IMAGE:?Runner must configure PPU_CI_RUNTIME_IMAGE}"

if ! image_id=$(docker image inspect "${PPU_CI_RUNTIME_IMAGE}" --format '{{.Id}}' 2>/dev/null); then
    private_dir=$(mktemp -d "${HOME:?}/.vllm-sail-ppu-pull.XXXXXX")
    chmod 0700 "${private_dir}"
    pull_log="${private_dir}/pull.log"
    if ! docker pull "${PPU_CI_RUNTIME_IMAGE}" >"${pull_log}" 2>&1; then
        echo "Failed to pull the runner-configured PPU runtime image" >&2
        exit 2
    fi
    image_id=$(docker image inspect "${PPU_CI_RUNTIME_IMAGE}" --format '{{.Id}}' 2>>"${pull_log}") || {
        echo "Failed to inspect the runner-configured PPU runtime image" >&2
        exit 2
    }
fi
[[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo "Docker did not resolve an immutable PPU runtime image ID" >&2
    exit 2
}
printf '%s\n' "${image_id}"
