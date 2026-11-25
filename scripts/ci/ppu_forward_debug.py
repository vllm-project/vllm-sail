# SPDX-License-Identifier: Apache-2.0
"""Report fixed diagnostic markers for the Forward replay."""

import json
import os
import re
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppu_recipe_diagnostics import LOG_TAIL_LIMIT, _read_file, failure_reason

EXCEPTION = re.compile(
    r"^(?:\[rank\d+\]:\s*)?(?P<kind>torch(?:\.cuda)?\.OutOfMemoryError|"
    r"RuntimeError|ValueError|AssertionError|FileNotFoundError|ImportError|"
    r"ModuleNotFoundError|MemoryError|subprocess\.CalledProcessError):",
    re.MULTILINE,
)
FRAME = re.compile(
    r'File "[^"\n]*/(vllm/benchmarks/forward/(?:main|capture|batch|routing|kv_init)\.py|'
    r'ppu_recipe_forward_perf\.py)", line ([0-9]{1,5})'
)
CASES = re.compile(r"^\[forward\] (prefill_(?:4k|16k|64k|100k)):", re.MULTILINE)


def diagnostics(artifacts, result_dir):
    runtime = _read_file(artifacts / "runtime.log", LOG_TAIL_LIMIT, tail=True)
    try:
        has_results = stat.S_ISDIR(result_dir.lstat().st_mode)
    except OSError:
        has_results = False
    prefill = (
        _read_file(result_dir / "prefill.log", LOG_TAIL_LIMIT, tail=True)
        if has_results
        else ""
    )
    combined = runtime + "\n" + prefill
    completed = CASES.findall(prefill)
    return {
        "phase": failure_reason(artifacts, "forward_perf"),
        "last_completed_case": completed[-1] if completed else None,
        "allocator_retries": prefill.count("memory allocation failed with OOM"),
        "terminal_oom": bool(
            re.search(
                r"(?:OutOfMemoryError|RuntimeError):[^\n]*out of memory", combined, re.I
            )
        ),
        "exceptions": sorted(set(EXCEPTION.findall(combined))),
        "forward_frames": list(dict.fromkeys(FRAME.findall(combined)))[-8:],
        "exit_codes": sorted(
            {
                int(value)
                for value in re.findall(
                    r"(?:exitcode\s*:\s*|exit status )(-?[0-9]{1,3})", combined
                )
            }
        ),
    }


def main():
    current = Path(os.environ["PRIVATE_DIR"])
    print(json.dumps({"replay": diagnostics(current, current / "results")}))
    root = os.environ.get("PPU_CI_PRIVATE_RESULTS_ROOT")
    if not root:
        return
    retained = Path(root) / "vllm-sail/45536457776-1/forward_perf/deepseek-v4.1-flash"
    if _read_file(retained / "recipe-phase.txt", 64):
        print(json.dumps({"original_run": diagnostics(retained, retained)}))


if __name__ == "__main__":
    main()
