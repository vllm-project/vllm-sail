# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ppu_forward_perf", ROOT / "scripts/ci/ppu_forward_perf.py"
)
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)
CONFIG = json.loads(perf.CONFIG.read_text())


def output(median=1000.0):
    results = []
    for name, count in perf.cases(CONFIG).items():
        request = {
            "query_len": 1,
            "kv_len": 4096,
            "num_draft_tokens": 0,
            "chunked": False,
        }
        rank = {
            "num_tokens": count,
            "routing": "real",
            "timing_scope": "target_only",
            "finite": True,
            "latency_us": {
                "timer": "graph_events_rank_max",
                "median": median,
                "p90": median,
                "count": 100,
                "samples_us": [median] * 100,
                "sample_spread_pct": 0.1,
            },
        }
        results.append(
            {
                "case": name,
                "requests": [request.copy() for _ in range(count)],
                "median_us": median,
                "ranks": [copy.deepcopy(rank) for _ in range(8)],
            }
        )
    return results


def identity():
    return {
        "model_revision": "reviewed-model-revision",
        "model_config_sha256": "config-digest",
        "runtime_image_id": "image-digest",
        "image_stack": {"sdk_sha256": "sdk-digest", "torch_version": "2.13"},
        "tested_sail_commit": "candidate-sha",
        "upstream_commit": "upstream-sha",
    }


def test_global_batch_is_split_across_dp_ranks(tmp_path):
    cmd = perf.command(CONFIG, Path("/model with spaces"), tmp_path / "forward.json")
    shapes = [cmd[index + 1] for index, arg in enumerate(cmd) if arg == "--case"]
    assert shapes == ["decode_b32_k4096=4*1@4096", "decode_b64_k4096=8*1@4096"]
    assert cmd[cmd.index("--model") + 1] == "/model with spaces"
    assert "--nproc-per-node=8" in cmd
    assert cmd[cmd.index("--data-parallel-size") + 1] == "8"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
    assert "--enable-expert-parallel" in cmd
    assert "--exclude-drafter" in cmd
    assert cmd[cmd.index("--load-format") + 1] == "auto"
    assert cmd[cmd.index("--routing") + 1] == "real"
    assert cmd[cmd.index("--kv-len-jitter") + 1] == "0"
    speculative = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert speculative == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "probabilistic",
    }
    assert "--host" not in cmd and "--port" not in cmd
    assert "--tool-call-parser" not in cmd and "--reasoning-parser" not in cmd


def test_complete_cross_rank_results_are_accepted():
    results = output()
    assert perf.validate_results(results, CONFIG) is results


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows[1].update(case=rows[0]["case"]),
        lambda rows: rows[0]["requests"].pop(),
        lambda rows: rows[0]["requests"][0].update(kv_len=4097),
        lambda rows: rows[0]["ranks"].pop(),
        lambda rows: rows[0].update(median_us=0),
        lambda rows: rows[0].update(median_us=float("nan")),
        lambda rows: rows[0].update(median_us=True),
        lambda rows: rows[0]["ranks"][0].update(finite=False),
        lambda rows: rows[0]["ranks"][0].update(timing_scope="target_and_drafter"),
        lambda rows: rows[0]["ranks"][0].update(routing="zipf"),
        lambda rows: rows[0]["ranks"][0]["latency_us"].update(timer="graph_events"),
        lambda rows: rows[0]["ranks"][0]["latency_us"].update(count=99),
        lambda rows: rows[0]["ranks"][0]["latency_us"]["samples_us"].pop(),
        lambda rows: rows[0]["ranks"][0]["latency_us"]["samples_us"].__setitem__(0, -1),
        lambda rows: rows[0]["ranks"][0]["latency_us"].update(median=1100),
        lambda rows: rows[0]["ranks"][0]["latency_us"].update(p90=float("inf")),
        lambda rows: rows[0]["ranks"][0]["latency_us"].update(sample_spread_pct=0.6),
        lambda rows: rows[0]["ranks"][0]["latency_us"].update(
            sample_spread_pct=float("nan")
        ),
    ],
)
def test_invalid_or_unstable_output_cannot_pass(mutation):
    rows = output()
    mutation(rows)
    with pytest.raises(ValueError):
        perf.validate_results(rows, CONFIG)


def confirmed_golden(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    perf.report(output(), CONFIG, identity(), tmp_path / "missing", first)
    root = tmp_path / "golden"
    (root / "890p").mkdir(parents=True)
    doc = json.loads((first / "baselines/890p/model.json").read_text())
    for metric in doc["metrics"].values():
        metric["status"] = "confirmed"
    (root / "890p/model.json").write_text(json.dumps(doc))
    return root


def test_first_run_writes_pending_candidate_without_fabricated_golden(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    golden = tmp_path / "golden"
    assert perf.report(output(), CONFIG, identity(), golden, results) == 0
    assert not golden.exists()
    doc = json.loads((results / "baselines/890p/model.json").read_text())
    assert len(doc["metrics"]) == 2
    assert all(metric["status"] == "pending" for metric in doc["metrics"].values())
    summary = json.loads((results / "perf-summary.json").read_text())
    assert [row["status"] for row in summary["cases"]] == ["PENDING", "PENDING"]
    assert [row["global_batch_size"] for row in summary["cases"]] == [32, 64]
    assert [row["shape_tokens_per_second"] for row in summary["cases"]] == [
        32000,
        64000,
    ]
    assert (results / "perf-summary.html").is_file()


@pytest.mark.parametrize("median,expected", [(1000, 0), (1150, 0), (1400, 1)])
def test_comparison_preserves_reviewed_golden_and_saves_current_candidate(
    tmp_path, median, expected
):
    golden = confirmed_golden(tmp_path)
    path = golden / "890p/model.json"
    before = path.read_bytes()
    results = tmp_path / "results"
    results.mkdir()
    assert perf.report(output(median), CONFIG, identity(), golden, results) == expected
    assert path.read_bytes() == before
    candidate = json.loads((results / "baselines/890p/model.json").read_text())
    assert all(metric["median"] == median for metric in candidate["metrics"].values())
    assert all(
        metric["status"] == "pending" for metric in candidate["metrics"].values()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_revision", "different-model"),
        ("runtime_image_id", "different-image"),
        ("model_config_sha256", "different-config"),
        ("image_stack", {"torch_version": "different-torch"}),
    ],
)
def test_incompatible_configuration_requires_baseline_review(tmp_path, field, value):
    golden = confirmed_golden(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    changed = identity() | {field: value}
    with pytest.raises(ValueError, match="golden configuration differs"):
        perf.report(output(), CONFIG, changed, golden, results)


def test_tested_commit_is_not_part_of_the_comparison_fingerprint(tmp_path):
    golden = confirmed_golden(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    changed = identity() | {
        "tested_sail_commit": "new-sail",
        "upstream_commit": "new-upstream",
    }
    assert perf.report(output(), CONFIG, changed, golden, results) == 0


def test_candidate_cannot_overwrite_golden(tmp_path):
    with pytest.raises(ValueError, match="must not overwrite"):
        perf.report(output(), CONFIG, identity(), tmp_path / "baselines", tmp_path)


def test_wrapper_mounts_checkpoint_and_keeps_cache_and_golden_contracts():
    text = (ROOT / "scripts/ci/ppu_runtime_container.sh").read_text()
    branch = text.split("\n    model-perf)\n", 1)[1].split("\n        ;;", 1)[0]
    assert '"${PPU_CI_DEVICE_COUNT}" == "8"' in branch
    assert 'mount_snapshot_if_external "${model_snapshot}"' in branch
    assert '--env HF_MODULES_CACHE="${hf_modules_cache}"' in branch
    assert '--env MODEL_REVISION="${model_revision}"' in branch
    assert "Explicit MODEL_PATH requires an immutable MODEL_REVISION" in branch
    assert '--env RUNTIME_IMAGE_ID="${image_id}"' in branch
    assert "--env BASELINE_ROOT=/source/tests/e2e/perf/baselines" in branch
    assert "/source/scripts/ci/ppu_model_perf.sh" in branch


def test_runner_preserves_installed_stack_and_invokes_forward_without_server():
    text = (ROOT / "scripts/ci/ppu_model_perf.sh").read_text()
    assert 'ppu_wheel_manifest.py" verify' in text
    assert "--no-index --no-deps --force-reinstall" in text
    assert 'ppu_image_stack.py" capture' in text
    assert "trap verify_stack EXIT" in text
    assert "torch.cuda.device_count() == 8" in text
    assert "capability == (8, 9)" in text
    assert 'ppu_forward_perf.py"' in text
    assert "vllm serve" not in text
    assert "requirements/dev.txt" not in text


def test_reusing_results_directory_refreshes_pending_candidate(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    golden = tmp_path / "golden"
    perf.report(output(), CONFIG, identity(), golden, results)
    perf.report(output(2000), CONFIG, identity(), golden, results)
    candidate = results / "baselines/890p/model.json"
    doc = json.loads(candidate.read_text())
    assert all(metric["median"] == 2000 for metric in doc["metrics"].values())
    summary = json.loads((results / "perf-summary.json").read_text())
    assert summary["candidate_path"] == str(candidate)
    assert str(candidate) in (results / "perf-summary.md").read_text()


@pytest.mark.parametrize("failure", [None, "invalid-output", "process-error"])
def test_cli_runs_forward_and_propagates_invalid_execution(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setenv("VLLM_SAIL_PERF_COMMIT", "test-sail-sha")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    results = tmp_path / "results"
    results.mkdir()
    (results / "runtime-identity.json").write_text(json.dumps(identity()))
    (results / "image-stack.json").write_text("{}")
    monkeypatch.setattr(
        perf.sys,
        "argv",
        [
            "ppu_forward_perf.py",
            "--model",
            str(model),
            "--model-revision",
            "revision",
            "--results-dir",
            str(results),
            "--golden-root",
            str(tmp_path / "golden"),
        ],
    )

    def run(argv, *, cwd, check):
        assert cwd == "/tmp" and check is True
        assert argv == perf.command(CONFIG, model, results / "forward.json")
        if failure == "process-error":
            raise subprocess.CalledProcessError(1, argv)
        rows = output()
        if failure == "invalid-output":
            rows.pop()
        (results / "forward.json").write_text(json.dumps(rows))

    monkeypatch.setattr(perf.subprocess, "run", run)
    if failure == "process-error":
        with pytest.raises(subprocess.CalledProcessError):
            perf.main()
    elif failure == "invalid-output":
        with pytest.raises(ValueError):
            perf.main()
    else:
        assert perf.main() == 0
        assert (results / "perf-summary.json").is_file()
    assert (results / "benchmark-command.json").is_file()
    assert (results / "benchmark-identity.json").is_file()
    assert not (tmp_path / "golden").exists()


def test_snapshot_path_and_revision_use_one_cache_reference(tmp_path):
    cache = tmp_path / "hub/models--T-HEAD--model/refs"
    cache.mkdir(parents=True)
    (cache / "main").write_text("old-revision")
    text = (ROOT / "scripts/ci/ppu_runtime_container.sh").read_text()
    branch = text.split("\n    model-perf)\n", 1)[1].split("\n        ;;", 1)[0]
    resolution = branch.split("\n        else\n", 1)[1].split("\n        fi\n", 1)[0]
    script = r"""
realpath() {
    printf '%s' new-revision > "$hf_home/hub/models--${MODEL_ID//\//--}/refs/main"
    printf '%s' "$1"
}
"""
    script += resolution + '\nprintf "%s\\n%s\\n" "$model_revision" "$model_snapshot"'
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={"hf_home": str(tmp_path), "MODEL_ID": "T-HEAD/model"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "old-revision",
        str(tmp_path / "hub/models--T-HEAD--model/snapshots/old-revision"),
    ]
    assert (cache / "main").read_text() == "new-revision"
