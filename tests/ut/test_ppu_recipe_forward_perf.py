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
BASELINE = ROOT / "tests/e2e/perf/baselines/890p/deepseek-v4.1-flash-forward.json"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(CI_DIR))
    monkeypatch.syspath_prepend(str(ROOT / "tests/e2e/perf"))
    import ppu_recipe
    import ppu_recipe_forward_perf

    return ppu_recipe, ppu_recipe_forward_perf


def test_recipe_resolves_only_the_tp_prefill_command(tmp_path, modules):
    recipe, runner = modules
    policy = recipe._load_yaml(POLICY)
    resolved = recipe.resolve_recipe(
        ROOT / policy["recipe"], policy_path=POLICY, tier="forward_perf"
    )
    commands = runner.commands(
        resolved, policy["tiers"]["forward_perf"]["groups"], tmp_path, tmp_path
    )
    assert list(commands) == ["prefill"]
    for cmd in commands.values():
        assert "--nproc-per-node=8" in cmd
        assert cmd[cmd.index("--load-format") + 1] == "dummy"
        assert "--no-enable-prefix-caching" in cmd
        assert "--tool-call-parser" not in cmd
        assert "--engram-config" in cmd
        assert cmd[cmd.index("--model") + 1] == str(tmp_path)
    assert "--max-model-len=102528" in commands["prefill"]
    assert "--kv-cache-memory-bytes=17179869184" in commands["prefill"]
    assert "prefill_100k=102400" in commands["prefill"]


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
    assert runner.assess(baseline, baseline)["status"] == "pass"
    assert runner.assess({**baseline, "config": {}}, baseline) == {
        "status": "fail",
        "failure_reason": "forward_config_mismatch",
        "cases": [],
        "config_mismatch": ["devices"],
    }


@pytest.mark.parametrize(
    "median_us, cv, case_status, reason",
    [
        (99, 0, "PASS", "not_applicable"),
        (104.99, 0, "PASS", "not_applicable"),
        (105, 0, "PASS", "not_applicable"),
        (105.00001, 0, "FAIL", "forward_regression"),
        (106, 0.10, "FAIL", "forward_regression"),
        (104, 0.16, "UNSTABLE", "forward_measurement_unstable"),
        (106, 0.16, "FAIL", "forward_regression"),
    ],
)
def test_forward_regression_threshold(modules, median_us, cv, case_status, reason):
    _, runner = modules
    baseline = {
        "schema_version": 1,
        "config": {},
        "metrics": {"prefill_4k": {"median_us": 100, "cv": 0}},
    }
    current = {
        **baseline,
        "metrics": {"prefill_4k": {"median_us": median_us, "cv": cv}},
    }
    actual = runner.assess(current, baseline)
    assert actual["cases"][0]["status"] == case_status
    assert actual["failure_reason"] == reason
    assert actual["status"] == ("pass" if case_status == "PASS" else "fail")


def test_one_regressed_case_is_reported_to_nightly(tmp_path, monkeypatch, modules):
    _, runner = modules
    from ppu_notify_nightly import collect

    baseline = json.loads(BASELINE.read_text())
    baseline_us = baseline["metrics"]["prefill_16k"]["median_us"]
    current = {
        **baseline,
        "metrics": {
            **baseline["metrics"],
            "prefill_16k": {"median_us": baseline_us * 1.06, "cv": 0},
        },
    }
    report = runner.assess(current, baseline)
    assert report["status"] == "fail"
    assert sum(case["status"] == "FAIL" for case in report["cases"]) == 1

    private = tmp_path / "private"
    (private / "results").mkdir(parents=True)
    runner.write_json(private / "results/summary.json", report)
    public = tmp_path / "ppu-recipe-forward_perf-deepseek-v4.1-flash-123-1"
    public.mkdir()
    summary = tmp_path / "step-summary.md"
    for key, value in {
        "PRIVATE_DIR": str(private),
        "PUBLIC_DIR": str(public),
        "GITHUB_STEP_SUMMARY": str(summary),
        "RECIPE_OUTCOME": "failure",
        "RETENTION_OUTCOME": "success",
        "GITHUB_REPOSITORY": "example/framework",
        "GITHUB_RUN_ID": "123",
        "NOTIFY_SELECT_RESULT": "success",
        "NOTIFY_BUILD_RESULT": "success",
        "NOTIFY_RECIPE_RESULT": "success",
        "NOTIFY_LEGACY_RESULT": "skipped",
        "NOTIFY_RECIPE_MATRIX": '{"include":[{"key":"atlas"}]}',
        "NOTIFY_LEGACY_MATRIX": "",
        "NOTIFY_FORWARD_SELECTED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    assert runner.publish() == 0
    assert "| +6.00% | FAIL |" in summary.read_text()

    recipe_job = "Recipe model sanity & micro-E2E / Model sanity & micro-E2E atlas"
    forward_job = "Recipe forward performance / Forward performance deepseek-v4.1-flash"
    artifact = tmp_path / "ppu-recipe-model_sanity-atlas-123-1"
    artifact.mkdir()
    runner.write_json(
        artifact / "summary.json",
        {
            "model_key": "atlas",
            "status": "pass",
            "failure_reason": "not_applicable",
            "gsm8k": {"gate": "pass", "score": 0.9, "golden_score": 0.9},
            "micro_e2e": "pass",
        },
    )
    jobs = {recipe_job: ("success", 1), forward_job: ("failure", 1)}
    monkeypatch.setattr(
        "ppu_notify_nightly.run_jobs",
        lambda expected: ({name: jobs[name] for name in expected}, 0),
    )
    rows = collect(tmp_path)["results"]
    assert [(row["model"], row["status"]) for row in rows] == [
        ("atlas", "pass"),
        ("deepseek-v4.1-flash-forward", "fail"),
    ]
    monkeypatch.setenv("NOTIFY_FORWARD_SELECTED", "")
    assert collect(tmp_path)["results"] == rows[:1]


@pytest.mark.parametrize("has_result", [False, True])
def test_isolated_publisher_reports_failure_without_private_details(
    tmp_path, has_result
):
    private = tmp_path / "private"
    (private / "results").mkdir(parents=True)
    (private / "recipe-phase.txt").write_text("model_cache_check\n")
    (private / "runtime.log").write_text("private-runtime-error\n")
    if has_result:
        (private / "results/summary.json").write_text(
            json.dumps(
                {
                    "status": "fail",
                    "failure_reason": "forward_config_mismatch",
                    "cases": [],
                    "config_mismatch": ["torch_sha256", "private-detail"],
                }
            )
        )

    public = tmp_path / "public"
    public.mkdir()
    summary = tmp_path / "step-summary.md"
    result = subprocess.run(
        [sys.executable, "-IB", str(CI_DIR / "ppu_recipe_forward_perf.py"), "publish"],
        env={
            **os.environ,
            "PRIVATE_DIR": str(private),
            "PUBLIC_DIR": str(public),
            "GITHUB_STEP_SUMMARY": str(summary),
            "RECIPE_OUTCOME": "failure",
            "RETENTION_OUTCOME": "success",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    expected = {
        "status": "fail",
        "failure_reason": "model_cache_check_failed",
        "cases": [],
    }
    if has_result:
        expected["failure_reason"] = "forward_config_mismatch"
        expected["config_mismatch"] = ["torch_sha256"]
    assert json.loads((public / "summary.json").read_text()) == expected
    assert expected["failure_reason"] in summary.read_text()
    for output in (result.stdout, result.stderr, summary.read_text()):
        assert "private-runtime-error" not in output
        assert "private-detail" not in output


@pytest.mark.parametrize(
    "models, scheduled, enabled, skip_build, forward",
    [
        ("forward:deepseek-v4.1-flash", False, False, False, True),
        ("", True, False, False, False),
        ("", True, True, False, True),
        ("recipe:deepseek-v4.1-flash", False, False, False, False),
        ("deepseek-v4-int8", False, False, False, False),
        ("", False, False, True, False),
    ],
)
def test_nightly_selection(tmp_path, models, scheduled, enabled, skip_build, forward):
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
        "SKIP_WHEEL_BUILD": str(skip_build).lower(),
        "RECIPE_PUBLIC_ENABLED": "true",
        "SCHEDULED_RECIPE_ENABLED": str(scheduled).lower(),
        "SCHEDULED_FORWARD_PERF_ENABLED": str(enabled).lower(),
    }
    subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, env=env, check=True)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert (values.get("run_forward") == "true") is forward
    if models.startswith("forward:"):
        assert not values.get("matrix") and not values.get("run_recipe")
    elif models.startswith("recipe:"):
        assert not values["matrix"]
        assert values["recipe_models"] == "deepseek-v4.1-flash"
        assert (
            json.loads(values["recipe_matrix"])["include"][0]["key"]
            == "deepseek-v4.1-flash"
        )
    elif models:
        assert json.loads(values["matrix"]) == {"include": [{"model": models}]}
        assert not values.get("recipe_matrix")
    else:
        assert json.loads(values["recipe_matrix"])["include"]
