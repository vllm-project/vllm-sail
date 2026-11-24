# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_DIR = ROOT / "scripts/ci"
POLICY = ROOT / "tests/ppu/policy/models/deepseek-v4.1-flash.yaml"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(CI_DIR))
    monkeypatch.syspath_prepend(str(ROOT / "tests/e2e/perf"))
    import ppu_recipe
    import ppu_recipe_forward_perf

    return ppu_recipe, ppu_recipe_forward_perf


def test_recipe_resolves_two_forward_commands(tmp_path, modules):
    recipe, runner = modules
    policy = recipe._load_yaml(POLICY)
    resolved = recipe.resolve_recipe(
        ROOT / policy["recipe"], policy_path=POLICY, tier="forward_perf"
    )
    commands = runner.commands(
        resolved, policy["tiers"]["forward_perf"]["groups"], tmp_path, tmp_path
    )
    assert list(commands) == ["prefill", "decode"]
    for cmd in commands.values():
        assert "--nproc-per-node=8" in cmd
        assert cmd[cmd.index("--load-format") + 1] == "dummy"
        assert "--no-enable-prefix-caching" in cmd
        assert "--tool-call-parser" not in cmd
        assert "--engram-config" in cmd
        assert cmd[cmd.index("--model") + 1] == str(tmp_path)
    assert "--max-model-len=102528" in commands["prefill"]
    assert "prefill_100k=102400" in commands["prefill"]
    assert "--max-num-seqs=256" in commands["decode"]
    assert "decode_256=256*1@4097" in commands["decode"]


def test_result_validation_and_comparison(tmp_path, modules):
    _, runner = modules
    expected = {"prefill_4k": [{"query_len": 4096}]}
    path = tmp_path / "prefill.json"
    rank = {
        "finite": True,
        "routing": "zipf",
        "latency_us": {
            "timer": "graph_events_rank_max",
            "median": 100,
            "count": 100,
            "samples_us": [100] * 100,
        },
    }
    runner.write_json(
        path,
        [
            {
                "case": "prefill_4k",
                "requests": expected["prefill_4k"],
                "median_us": 100,
                "ranks": [rank] * 8,
            }
        ],
    )
    metrics = runner.read_measurements(path, expected, 8)
    with pytest.raises(ValueError, match="rank count"):
        runner.read_measurements(path, expected, 4)
    baseline = {"schema_version": 1, "config": {"devices": 8}, "metrics": metrics}
    current = {**baseline, "metrics": {"prefill_4k": {"median_us": 140, "cv": 0}}}
    assert runner.assess(baseline, baseline)["status"] == "pass"
    assert runner.assess(current, baseline)["failure_reason"] == "forward_regression"
    assert (
        runner.assess({**baseline, "config": {}}, baseline)["failure_reason"]
        == "forward_config_mismatch"
    )


@pytest.mark.parametrize(
    "models, scheduled, enabled, forward",
    [
        ("forward:deepseek-v4.1-flash", False, False, True),
        ("", True, False, False),
        ("", True, True, True),
    ],
)
def test_nightly_selection(tmp_path, models, scheduled, enabled, forward):
    flow = yaml.safe_load(
        (ROOT / ".github/workflows/ppu-accuracy-nightly.yaml").read_text()
    )
    step = next(s for s in flow["jobs"]["select"]["steps"] if s.get("id") == "select")
    assert step["env"]["SCHEDULED_FORWARD_PERF_ENABLED"] == "false"
    script = step["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    output = tmp_path / "output"
    env = {
        **os.environ,
        "TARGET_SHA": "b" * 40,
        "GITHUB_OUTPUT": str(output),
        "MODELS": models,
        "SKIP_WHEEL_BUILD": "false",
        "RECIPE_PUBLIC_ENABLED": "true",
        "SCHEDULED_RECIPE_ENABLED": str(scheduled).lower(),
        "SCHEDULED_FORWARD_PERF_ENABLED": str(enabled).lower(),
    }
    subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, env=env, check=True)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert (values.get("run_forward") == "true") is forward
    if models:
        assert not values.get("matrix") and not values.get("run_recipe")
    else:
        assert json.loads(values["recipe_matrix"])["include"]
