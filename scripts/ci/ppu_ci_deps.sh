# SPDX-License-Identifier: Apache-2.0
# Use the image's dependency set when pip can satisfy the request offline.
ppu_ci_install_if_missing() {
    if python -m pip install --dry-run --no-index "$@" >/dev/null 2>&1; then
        return
    fi
    python -m pip install "$@"
}
