# SPDX-License-Identifier: Apache-2.0
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_DIR = ROOT / "scripts/ci"
POLICY = ROOT / "tests/ppu/policy/models/deepseek-v4.1-flash.yaml"
BASELINE = ROOT / "tests/e2e/perf/baselines/890p/deepseek-v4.1-flash-light-perf.json"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(CI_DIR))
    monkeypatch.syspath_prepend(str(ROOT / "tests/e2e/perf"))
    import ppu_recipe
    import ppu_recipe_light_perf

    return ppu_recipe, ppu_recipe_light_perf


def test_recipe_resolves_only_the_tp_prefill_command(tmp_path, modules):
    recipe, runner = modules
    policy = recipe._load_yaml(POLICY)
    resolved = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-B",
                str(CI_DIR / "ppu_recipe.py"),
                "resolve",
                "--recipe",
                str(ROOT / policy["recipe"]),
                "--policy",
                str(POLICY),
                "--tier",
                "light_perf",
            ],
            text=True,
        )
    )
    commands = runner.commands(
        resolved, policy["tiers"]["light_perf"]["groups"], tmp_path, tmp_path
    )
    assert list(commands) == ["prefill"]
    for cmd in commands.values():
        assert "--nproc-per-node=8" in cmd
        assert "--standalone" in cmd
        assert "--local-addr=127.0.0.1" in cmd[: cmd.index(runner.VLLM_CLI_MODULE)]
        assert "--shutdown-timeout=20" in cmd[: cmd.index(runner.VLLM_CLI_MODULE)]
        assert "--exclude-sampler" in cmd[cmd.index(runner.VLLM_CLI_MODULE) + 1 :]
        assert cmd[cmd.index("--load-format") + 1] == "dummy"
        assert "--no-enable-prefix-caching" in cmd
        assert "--tool-call-parser" not in cmd
        assert "--engram-config" in cmd
        assert cmd[cmd.index("--model") + 1] == str(tmp_path)
    assert "--max-model-len=102528" in commands["prefill"]
    assert "--kv-cache-memory-bytes=17179869184" in commands["prefill"]
    assert "prefill_100k=102400" in commands["prefill"]
    baseline = json.loads(BASELINE.read_text())
    assert resolved["config_digest"] == baseline["config"]["recipe_config_digest"]
    assert policy["tiers"]["light_perf"]["groups"] == baseline["config"]["groups"]
    assert {
        name: runner.benchmark_args(cmd, resolved["hf_model_id"])
        for name, cmd in commands.items()
    } == baseline["config"]["benchmark_args"]


def test_policy_validation_rejects_non_tp_light_perf_recipe(tmp_path, modules):
    recipe, _ = modules
    policy = recipe._load_yaml(POLICY)
    doc = recipe._load_yaml(ROOT / policy["recipe"])
    flags = doc["model"]["base_args"]
    flags[flags.index("--tensor-parallel-size") + 1] = "1"
    model_recipe = tmp_path / "prefill.yaml"
    model_recipe.write_text(yaml.safe_dump(doc))
    policy["recipe"] = model_recipe.name
    with pytest.raises(recipe.ValidationError, match="requires a TP recipe"):
        recipe.validate_policy_structure(
            policy, context="light_perf", base_dir=tmp_path
        )


def test_snapshot_provenance_supports_commit_and_local_cache_ref(tmp_path, modules):
    recipe, runner = modules
    snapshots = tmp_path / "snapshots"
    alias = "local-ci-20261119"
    revision = "a" * 40
    local = runner.snapshot_provenance(snapshots / alias)
    assert local == {
        "checkpoint_source": "hf_snapshot",
        "checkpoint_revision": None,
        "checkpoint_cache_ref": alias,
    }
    recipe.redact_and_verify_public_payload(local)
    assert runner.snapshot_provenance(snapshots / revision) == {
        "checkpoint_source": "hf_snapshot",
        "checkpoint_revision": revision,
    }
    with pytest.raises(
        ValueError, match="Invalid offline checkpoint snapshot reference"
    ):
        runner.snapshot_provenance(snapshots / "invalid ref")


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
        "failure_reason": "light_perf_config_mismatch",
        "cases": [],
        "config_mismatch": ["devices"],
    }


@pytest.mark.parametrize(
    "median_us, cv, case_status, reason",
    [
        (99, 0, "PASS", "not_applicable"),
        (104.99, 0, "PASS", "not_applicable"),
        (105, 0, "PASS", "not_applicable"),
        (105.00001, 0, "FAIL", "light_perf_regression"),
        (106, 0.10, "FAIL", "light_perf_regression"),
        (104, 0.16, "UNSTABLE", "light_perf_measurement_unstable"),
        (106, 0.16, "FAIL", "light_perf_regression"),
    ],
)
def test_light_perf_regression_threshold(modules, median_us, cv, case_status, reason):
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
    from ppu_notify_summary import JobExecution, make_payload

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
    public = tmp_path / "ppu-recipe-light_perf-deepseek-v4.1-flash-123-1"
    public.mkdir()
    summary = tmp_path / "step-summary.md"
    for key, value in {
        "PRIVATE_DIR": str(private),
        "PUBLIC_DIR": str(public),
        "GITHUB_STEP_SUMMARY": str(summary),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
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
        "NOTIFY_LIGHT_PERF_SELECTED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    assert runner.publish() == 0
    assert (tmp_path / "outputs").read_text() == "status=fail\n"
    assert "| +6.00% | FAIL |" in summary.read_text()

    recipe_job = "Recipe model sanity & micro-E2E / Model sanity & micro-E2E atlas"
    light_perf_job = (
        "Lightweight Performance Regression / Light perf deepseek-v4.1-flash"
    )
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
    jobs = {
        recipe_job: JobExecution("success", 1, 201, 60),
        light_perf_job: JobExecution("failure", 1, 202, 720),
    }
    monkeypatch.setattr(
        "ppu_notify_nightly.run_jobs",
        lambda expected: ({name: jobs[name] for name in expected}, 0),
    )
    report = collect(tmp_path)
    rows = report["results"]
    assert [(row["model"], row["status"]) for row in rows] == [
        ("atlas", "pass"),
        ("deepseek-v4.1-flash-light-perf", "fail"),
    ]
    markdown = make_payload(report, {}, 0, version="main")["markdown"]
    assert markdown["title"] == "[vllm-sail | main] PPU Nightly 晨报"
    assert (
        "`deepseek-v4.1-flash-light-perf` · lightweight performance regression"
        in markdown["text"]
    )
    assert "actions/runs/123/job/202" in markdown["text"]
    monkeypatch.setenv("NOTIFY_LIGHT_PERF_SELECTED", "")
    assert collect(tmp_path)["results"] == rows[:1]


@pytest.mark.parametrize(
    "results, recipe_outcome, retention_outcome, reason",
    [
        ("missing", "failure", "success", "model_cache_check_failed"),
        ("mismatch", "failure", "success", "light_perf_config_mismatch"),
        ("malformed", "success", "success", "light_perf_measurement_failed"),
        ("empty", "success", "success", "light_perf_measurement_failed"),
        ("pass", "success", "failure", "private_retention_failed"),
        ("pass", "success", "success", "not_applicable"),
        ("timeout", "failure", "success", "light_perf_measurement_timeout"),
    ],
)
def test_isolated_publisher_emits_verdict_without_private_details(
    tmp_path, results, recipe_outcome, retention_outcome, reason
):
    private = tmp_path / "private"
    (private / "results").mkdir(parents=True)
    phase = "model_cache_check" if results == "missing" else "light_perf_measurement"
    (private / "recipe-phase.txt").write_text(phase + "\n")
    (private / "runtime.log").write_text("private-runtime-error\n")
    evidence = private / "results/summary.json"
    row = {
        "case": "prefill_4k",
        "status": "PASS",
        "median_us": 100,
        "baseline_us": 100,
        "delta": 0,
    }
    if results in ("mismatch", "timeout"):
        evidence.write_text(
            json.dumps(
                {
                    "status": "fail",
                    "failure_reason": reason,
                    "cases": [],
                    "config_mismatch": ["torch_sha256", "private-detail"],
                }
            )
        )
    elif results == "malformed":
        evidence.write_text("{-private-detail")
    elif results in ("empty", "pass"):
        evidence.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "failure_reason": "not_applicable",
                    "cases": [row] if results == "pass" else [],
                }
            )
        )

    public = tmp_path / "public"
    public.mkdir()
    summary = tmp_path / "step-summary.md"
    result = subprocess.run(
        [sys.executable, "-IB", str(CI_DIR / "ppu_recipe_light_perf.py"), "publish"],
        env={
            **os.environ,
            "PRIVATE_DIR": str(private),
            "PUBLIC_DIR": str(public),
            "GITHUB_STEP_SUMMARY": str(summary),
            "GITHUB_OUTPUT": str(tmp_path / "outputs"),
            "RECIPE_OUTCOME": recipe_outcome,
            "RETENTION_OUTCOME": retention_outcome,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    expected = {
        "status": "pass" if reason == "not_applicable" else "fail",
        "failure_reason": reason,
        "cases": [row] if results == "pass" else [],
    }
    if results == "mismatch":
        expected["config_mismatch"] = ["torch_sha256"]
    assert json.loads((public / "summary.json").read_text()) == expected
    assert (tmp_path / "outputs").read_text() == f"status={expected['status']}\n"
    assert expected["failure_reason"] in summary.read_text()
    for output in (result.stdout, result.stderr, summary.read_text()):
        assert "private-runtime-error" not in output
        assert "private-detail" not in output


@pytest.mark.parametrize(
    "reason, phase",
    [
        ("wheel_verification_failed", "runtime_setup"),
        ("runtime_dependencies_failed", "runtime_setup"),
        ("platform_check_failed", "runtime_setup"),
        ("recipe_validation_failed", "runtime_setup"),
        ("runtime_setup_failed", "runtime_setup"),
        ("model_cache_check_failed", "model_cache_check"),
        ("private_retention_failed", "private_retention"),
        ("light_perf_config_mismatch", "runtime_setup"),
        ("light_perf_measurement_timeout", "lightweight performance regression"),
        ("light_perf_regression", "lightweight performance regression"),
        ("unexpected", "unknown"),
    ],
)
def test_nightly_attributes_light_perf_failures(tmp_path, modules, reason, phase):
    _, runner = modules
    from ppu_notify_nightly import light_perf_evidence

    artifact = tmp_path / "ppu-recipe-light_perf-deepseek-v4.1-flash-123-1"
    artifact.mkdir()
    runner.write_json(
        artifact / "summary.json",
        {"status": "fail", "failure_reason": reason, "cases": []},
    )
    assert light_perf_evidence(tmp_path, "123", 1) == ("fail", phase, None, None)


@pytest.mark.parametrize("launch_signal", [None, signal.SIGTERM, signal.SIGINT])
def test_light_perf_timeout_or_cancellation_stops_launcher(
    tmp_path, monkeypatch, modules, launch_signal
):
    _, runner = modules
    launch = subprocess.Popen
    processes = []

    def track(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        proc = launch(*args, **kwargs)
        processes.append(proc)
        if launch_signal is not None:
            os.kill(os.getpid(), launch_signal)
        return proc

    monkeypatch.setattr(subprocess, "Popen", track)
    monkeypatch.setattr(runner, "BENCHMARK_TIMEOUT_S", 0.1)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    expected = subprocess.TimeoutExpired if launch_signal is None else SystemExit
    with (tmp_path / "log").open("w") as log, pytest.raises(expected) as error:
        runner.run_benchmark(argv, os.environ, log)
    if launch_signal is not None:
        assert error.value.code == 128 + launch_signal
    assert len(processes) == 1
    assert processes[0].returncode in (-signal.SIGTERM, -signal.SIGKILL)
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())


@pytest.mark.parametrize(
    "models, scheduled, enabled, skip_build, light_perf",
    [
        ("light_perf:deepseek-v4.1-flash", False, False, False, True),
        ("", True, False, False, False),
        ("", True, True, False, True),
        ("recipe:deepseek-v4.1-flash", False, False, False, False),
        ("deepseek-v4-int8", False, False, False, False),
        ("", False, False, True, False),
    ],
)
def test_nightly_selection(
    tmp_path, models, scheduled, enabled, skip_build, light_perf
):
    flow = yaml.safe_load(
        (ROOT / ".github/workflows/ppu-accuracy-nightly.yaml").read_text()
    )
    step = next(s for s in flow["jobs"]["select"]["steps"] if s.get("id") == "select")
    assert step["env"]["SCHEDULED_LIGHT_PERF_ENABLED"] == "false"
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
        "SCHEDULED_LIGHT_PERF_ENABLED": str(enabled).lower(),
    }
    subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, env=env, check=True)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert (values.get("run_light_perf") == "true") is light_perf
    if models.startswith("light_perf:"):
        assert not values.get("matrix") and not values.get("run_recipe")
        assert values["full_run"] == "false"
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


@pytest.mark.parametrize("scheduled", [False, True])
def test_nightly_rejects_light_perf_when_target_has_no_policy_tier(tmp_path, scheduled):
    target = tmp_path / "target"
    policy = yaml.safe_load(POLICY.read_text())
    # Simulate an earlier checkout that contains the serving recipe and its fast tier.
    policy["tiers"] = {"fast_sanity": policy["tiers"]["fast_sanity"]}
    policy.pop("replaces_legacy_cases", None)
    for path in (
        CI_DIR / "ppu_recipe.py",
        CI_DIR / "ppu_e2e_select.py",
        ROOT / policy["recipe"],
    ):
        copy = target / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    target_policy = target / POLICY.relative_to(ROOT)
    target_policy.parent.mkdir(parents=True)
    target_policy.write_text(yaml.safe_dump(policy))

    flow = yaml.safe_load(
        (ROOT / ".github/workflows/ppu-accuracy-nightly.yaml").read_text()
    )
    step = next(
        step for step in flow["jobs"]["select"]["steps"] if step.get("id") == "select"
    )
    script = step["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    output = tmp_path / "outputs"
    selection = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=target,
        env={
            **os.environ,
            "TARGET_SHA": "b" * 40,
            "GITHUB_OUTPUT": str(output),
            "MODELS": "" if scheduled else "light_perf:deepseek-v4.1-flash",
            "SKIP_WHEEL_BUILD": "false",
            "RECIPE_PUBLIC_ENABLED": "true",
            "SCHEDULED_RECIPE_ENABLED": str(scheduled).lower(),
            "SCHEDULED_LIGHT_PERF_ENABLED": str(scheduled).lower(),
        },
        capture_output=True,
        text=True,
    )
    assert selection.returncode != 0
    assert (
        "Light perf policy is unavailable for the selected target_ref"
        in selection.stderr
    )
    assert "has no tier 'light_perf'" in selection.stderr
    assert not output.exists()
