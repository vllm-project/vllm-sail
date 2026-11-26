# SPDX-License-Identifier: Apache-2.0
# Use the image's dependency set when pip can satisfy the request offline.
ppu_ci_install_if_missing() {
    if python -m pip install --dry-run --no-index "$@" >/dev/null 2>&1; then
        return
    fi
    python -m pip install "$@"
}

ppu_ci_install_wheel_dependency() {
    local package=$1 extra=$2
    shift 2
    local wheel_requirements spec
    wheel_requirements="$(python - "$package" "$extra" <<'PY'
import sys
from importlib.metadata import requires

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

package, extra = sys.argv[1:]
expected_name = canonicalize_name(package)
requirements = []
for spec in requires("vllm-sail") or ():
    requirement = Requirement(spec)
    if canonicalize_name(requirement.name) != expected_name:
        continue
    if requirement.marker and not requirement.marker.evaluate({"extra": extra}):
        continue
    if requirement.url:
        raise SystemExit(f"Expected an index requirement for {package} in vllm-sail metadata")
    # The marker already matched this interpreter and the requested extra.
    requirement.marker = None
    requirements.append(str(requirement))

if not requirements:
    raise SystemExit(f"Missing active {package} requirement in installed vllm-sail metadata")
print(*requirements, sep="\n")
PY
    )" || return
    local requirements=()
    while IFS= read -r spec; do
        requirements+=("$spec")
    done <<< "$wheel_requirements"
    ppu_ci_install_if_missing "$@" "${requirements[@]}"
}
