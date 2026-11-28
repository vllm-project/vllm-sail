# SPDX-License-Identifier: Apache-2.0
"""Behavioral acceptance tests for the vLLM-SAIL Post-Merge Three-Gears CI (§2.6).

Covers all six first-release acceptance gates without requiring GPU/PPU hardware:
  Gate 1: Healthy verified-pin SHA end-to-end compatibility & dual-SHA summary
  Gate 2: Candidate 40-char SHA pass-through & SAIL native .so reuse without
          mutating csrc/upstream/manifest.toml
  Gate 3: Anti-false-green fail-closed UNVERIFIED (non-zero exit) on wheel
          mismatch or 100%-skipped pin-UT
  Gate 4: Breaking SHA red + control-first (last_green..red] bisection +
          plugin_interface_break vs sail_internal_patch_drift routing +
          incremental vs known-soft patch drift + Tier-A kernel checks
  Gate 5: Intentional failure negative test proving zero leaks across Console,
          Artifact JSON, GITHUB_STEP_SUMMARY, and Issue + fail-closed file wipe
  Gate 6: Dual git ls-remote polling (refs/heads/main + refs/tags/v*), PPU
          sliding-window coalescing, PR-CI priority yield, daily pin bump, and
          new vX.Y.0rc* routing
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from vllm_sail import envs

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIED_PIN = (REPO_ROOT / ".github" / "vllm-main-verified.commit").read_text().strip()
RELEASE_TAG = (REPO_ROOT / ".github" / "vllm-release-tag.commit").read_text().strip()


@pytest.fixture
def compat_mod(monkeypatch: pytest.MonkeyPatch):
    if "vllm" not in sys.modules:
        vllm_pkg = types.ModuleType("vllm")
        vllm_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    if "vllm.logger" not in sys.modules:
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = logging.getLogger
        monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.delitem(sys.modules, "vllm_sail.compat", raising=False)
    mod = importlib.import_module("vllm_sail.compat")
    yield mod
    sys.modules.pop("vllm_sail.compat", None)


def _load_script_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


post_merge = _load_script_module(
    "ppu_upstream_post_merge", "scripts/ci/ppu_upstream_post_merge.py"
)
wheel_manifest = _load_script_module(
    "ppu_wheel_manifest_under_test", "scripts/ci/ppu_wheel_manifest.py"
)
ops_runner = _load_script_module(
    "ppu_ops_runner_under_test", "scripts/ci/ppu_ops_runner.py"
)


def _sha(n: int) -> str:
    return f"{n:040x}"


def _write_donor_wheels(
    wheels_dir: Path,
    *,
    vllm_commit: str,
    sail_source: Path = REPO_ROOT,
    build_image: str = "sha256:" + "a" * 64,
    pytorch_sail_arch: str = "ppu_15;ppu_10",
) -> dict:
    from tests.ut.test_ppu_wheel_manifest import make_wheels

    wheels_dir.mkdir(parents=True, exist_ok=True)
    make_wheels(wheels_dir)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/ci/ppu_wheel_manifest.py"),
            "create",
            str(wheels_dir),
            "--sail-source",
            str(sail_source),
            "--vllm-commit",
            vllm_commit,
            "--build-image",
            build_image,
            "--pytorch-sail-arch",
            pytorch_sail_arch,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((wheels_dir / "wheel-manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Gate 1: Healthy Verified-Pin SHA End-to-End Baseline
# ---------------------------------------------------------------------------


def test_gate1_healthy_verified_sha_end_to_end(
    tmp_path, monkeypatch, caplog, compat_mod
):
    sail_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.delenv("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", raising=False)
    monkeypatch.delenv("VLLM_PPU_CANDIDATE_UPSTREAM_COMMIT", raising=False)
    monkeypatch.setattr(
        compat_mod,
        "installed_vllm_version",
        lambda: f"0.33.0rc1.dev12+g{VERIFIED_PIN}.empty",
    )
    with caplog.at_level(logging.WARNING):
        compat_mod.check_vllm_compatibility(force=True)
    assert "candidate mode" not in caplog.text.lower()

    wheels_dir = tmp_path / "wheels"
    manifest = _write_donor_wheels(wheels_dir, vllm_commit=VERIFIED_PIN)
    verified = wheel_manifest.verify(
        wheels_dir, vllm_commit=VERIFIED_PIN, sail_commit=sail_head
    )
    assert verified["commits"] == {"vllm": VERIFIED_PIN, "vllm-sail": sail_head}
    assert manifest["commits"]["vllm"] == VERIFIED_PIN

    pub_dir = tmp_path / "public"
    summary_md = tmp_path / "step_summary.md"
    published = post_merge.publish_sanitized_channels(
        gear="probe",
        status="pass",
        sail_sha=sail_head,
        upstream_sha=VERIFIED_PIN,
        verified_pin=VERIFIED_PIN,
        build_image="download.t-head.cn/docker_release/pytorch@sha256:" + "1" * 64,
        runtime_image="download.t-head.cn/docker_release/vllm@sha256:" + "2" * 64,
        cpu_status="pass",
        ppu_status="pass",
        ppu_skip_reason="none",
        public_dir=pub_dir,
        step_summary_path=summary_md,
    )
    assert published["public_summary"]["status"] == "pass"
    assert published["public_summary"]["commits"] == {
        "vllm": VERIFIED_PIN,
        "vllm_sail": sail_head,
        "verified_pin": VERIFIED_PIN,
    }
    assert VERIFIED_PIN in summary_md.read_text(encoding="utf-8")
    assert sail_head in summary_md.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Gate 2: Candidate SHA Full-Chain Pass-Through & Native .so Reuse
# ---------------------------------------------------------------------------


def test_gate2_candidate_sha_passthrough_and_sail_native_so_reuse_without_mutating_manifest(
    tmp_path, monkeypatch, caplog, compat_mod
):
    manifest_toml = REPO_ROOT / "csrc" / "upstream" / "manifest.toml"
    manifest_sha_before = hashlib.sha256(manifest_toml.read_bytes()).hexdigest()
    candidate_sha = "a" * 40
    assert candidate_sha != VERIFIED_PIN

    # 1. Exact 40-char candidate SHA allowed with warning; manifest.toml untouched.
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.setenv("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", candidate_sha)
    assert envs.VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT == candidate_sha
    monkeypatch.setattr(
        compat_mod,
        "installed_vllm_version",
        lambda: f"0.33.1.dev12+g{candidate_sha}.empty",
    )
    with caplog.at_level(logging.WARNING):
        compat_mod.check_vllm_compatibility(force=True)
    assert "candidate" in caplog.text.lower()
    assert hashlib.sha256(manifest_toml.read_bytes()).hexdigest() == manifest_sha_before

    # Exact clean RC tag ("0.34.0rc1" without +g<sha>) also passes when vllm._version.__commit_id__ matches.
    vllm_ver_mod = types.ModuleType("vllm._version")
    vllm_ver_mod.__commit_id__ = f"g{candidate_sha}"
    monkeypatch.setitem(sys.modules, "vllm._version", vllm_ver_mod)
    monkeypatch.setattr(compat_mod, "installed_vllm_version", lambda: "0.34.0rc1+empty")
    compat_mod.check_vllm_compatibility(force=True)
    monkeypatch.delitem(sys.modules, "vllm._version", raising=False)

    # Reject wildcard, short SHA, or mismatched installed vLLM SHA.
    for bad in ("*", candidate_sha[:12], "A" * 40):
        monkeypatch.setenv("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", bad)
        with pytest.raises(ValueError, match="40-character lowercase hex"):
            _ = envs.VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT

    monkeypatch.setenv("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", candidate_sha)
    monkeypatch.setattr(
        compat_mod,
        "installed_vllm_version",
        lambda: f"0.33.1.dev12+g{'b' * 40}.empty",
    )
    with pytest.raises(RuntimeError, match="candidate"):
        compat_mod.check_vllm_compatibility(force=True)

    # 2. Execute build_ppu_wheels.sh's prepare_native_reuse function in real bash:
    #    donor wheel has vllm=VERIFIED_PIN, current build targets VLLM_REF=candidate_sha.
    #    Must set NATIVE_REUSE=0 (rebuild vLLM wheel) AND SAIL_NATIVE_REUSE=1 (reuse .so).
    donor_dir = tmp_path / "donor-wheels"
    build_img = "sha256:" + "3" * 64
    _write_donor_wheels(
        donor_dir,
        vllm_commit=VERIFIED_PIN,
        sail_source=REPO_ROOT,
        build_image=build_img,
        pytorch_sail_arch="ppu_15;ppu_10",
    )

    build_script = (REPO_ROOT / "scripts/ci/build_ppu_wheels.sh").read_text(
        encoding="utf-8"
    )
    start_idx = build_script.index("fingerprint_args=(")
    end_idx = build_script.index('if [[ "${VLLM_REF}" =~ ^[0-9a-f]{40}$ ]]')
    extracted_fn = build_script[start_idx:end_idx]

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    manifest_file = tmp_path / "build-manifest.txt"
    manifest_file.touch()
    vllm_wheel_out = tmp_path / "out-vllm-wheels"
    vllm_wheel_out.mkdir()
    harness = tmp_path / "run_reuse.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f'ROOT_DIR="{REPO_ROOT}"\n'
        f'WORK_DIR="{work_dir}"\n'
        f'MANIFEST="{manifest_file}"\n'
        f'VLLM_WHEEL_DIR="{vllm_wheel_out}"\n'
        f'build_image="{build_img}"\n'
        'vllm_fingerprint=""\n'
        'PYTORCH_SAIL_ARCH="ppu_15;ppu_10"\n'
        f'VLLM_REF="{candidate_sha}"\n'
        f'CANDIDATE_VLLM_COMMIT="{candidate_sha}"\n'
        "NATIVE_REUSE=0\n"
        "SAIL_NATIVE_REUSE=0\n"
        'PREBUILT_EXT_DIR=""\n'
        f"{extracted_fn}\n"
        f'prepare_native_reuse "{donor_dir}"\n'
        'printf "NATIVE_REUSE=%s\\nSAIL_NATIVE_REUSE=%s\\nPREBUILT_EXT_DIR=%s\\n" "$NATIVE_REUSE" "$SAIL_NATIVE_REUSE" "$PREBUILT_EXT_DIR"\n',
        encoding="utf-8",
    )
    reuse_env = dict(os.environ)
    reuse_env["PATH"] = f"{Path(sys.executable).parent}:{reuse_env.get('PATH', '')}"
    proc = subprocess.run(
        ["bash", str(harness)],
        env=reuse_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "NATIVE_REUSE=0" in proc.stdout
    assert "SAIL_NATIVE_REUSE=1" in proc.stdout
    ext_line = next(
        line.split("=", 1)[1]
        for line in proc.stdout.splitlines()
        if line.startswith("PREBUILT_EXT_DIR=")
    )
    prebuilt_dir = Path(ext_line)
    for mod_name in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C"):
        assert (prebuilt_dir / f"{mod_name}.abi3.so").is_file()
    assert hashlib.sha256(manifest_toml.read_bytes()).hexdigest() == manifest_sha_before

    # 3. Verify ppu_ops_runner.clean_environment injects candidate SHA only when
    #    EXPECTED_VLLM_COMMIT != verified_pin.
    isolated = tmp_path / "isolated-ops"
    isolated.mkdir()
    cand_env = ops_runner.clean_environment(
        {"EXPECTED_VLLM_COMMIT": candidate_sha}, REPO_ROOT, isolated
    )
    assert cand_env["VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT"] == candidate_sha
    pin_env = ops_runner.clean_environment(
        {"EXPECTED_VLLM_COMMIT": VERIFIED_PIN}, REPO_ROOT, isolated
    )
    assert "VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT" not in pin_env


# ---------------------------------------------------------------------------
# Gate 3: Anti-False-Green (Mismatch & 100%-Skipped Must Exit Non-Zero)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("require_pr_artifact", ["false", "true"])
def test_gate3_anti_false_green_wheel_mismatch_fails_closed(
    tmp_path, require_pr_artifact
):
    sail_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    wheels_dir = tmp_path / "wheels"
    _write_donor_wheels(wheels_dir, vllm_commit=VERIFIED_PIN)
    mismatched_expected = "c" * 40

    base_env = dict(os.environ)
    base_env["PATH"] = f"{Path(sys.executable).parent}:{base_env.get('PATH', '')}"
    base_env.update(
        {
            "WHEELS_DIR": str(wheels_dir),
            "UT_RESULTS_DIR": str(tmp_path / f"ut-out-{require_pr_artifact}"),
            "REQUIRE_PR_ARTIFACT": require_pr_artifact,
            "EXPECTED_VLLM_COMMIT": mismatched_expected,
            "EXPECTED_SAIL_COMMIT": sail_head,
            "VLLM_SOURCE_ROOT": str(tmp_path),
            "MODEL_ID": "Qwen/Qwen3.8-27B",
            "HF_HOME": str(tmp_path),
        }
    )

    pin_res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/ci/ppu_pin_ut.sh")],
        env=base_env,
        capture_output=True,
        text=True,
    )
    assert pin_res.returncode != 0
    assert "PPU PIN UT UNVERIFIED" in (pin_res.stdout + pin_res.stderr)
    assert "PPU PIN UT SKIP" not in (pin_res.stdout + pin_res.stderr)

    smoke_res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/ci/ppu_e2e_smoke.sh")],
        env=base_env,
        capture_output=True,
        text=True,
    )
    assert smoke_res.returncode != 0
    assert "PPU E2E SMOKE UNVERIFIED" in (smoke_res.stdout + smoke_res.stderr)


def test_gate3_pin_ut_all_skipped_junit_fails_closed(tmp_path):
    """Verify the JUnit zero-executed-tests guard in ppu_pin_ut.sh exits non-zero."""
    pin_script = (REPO_ROOT / "scripts/ci/ppu_pin_ut.sh").read_text(encoding="utf-8")
    snippet = pin_script[pin_script.index('python - "${UT_RESULTS_DIR}/pin-ut.xml"') :]
    xml_path = tmp_path / "pin-ut.xml"
    xml_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite name="pytest" tests="5" skipped="5" failures="0" errors="0"/></testsuites>\n',
        encoding="utf-8",
    )
    guard_sh = tmp_path / "check_xml.sh"
    guard_sh.write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
        f'UT_RESULTS_DIR="{tmp_path}"\n'
        f"{snippet}\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    res = subprocess.run(
        ["bash", str(guard_sh)], env=env, capture_output=True, text=True
    )
    assert res.returncode != 0
    assert "PPU PIN UT UNVERIFIED: zero executed tests" in res.stderr


# ---------------------------------------------------------------------------
# Gate 4: Breaking SHA Red + Bisection + Dual-Track Triage + Kernel Check
# ---------------------------------------------------------------------------


def test_gate4_bisection_control_first_and_dual_track_triage(tmp_path):
    # 1. Build a real 30-commit linear upstream history in a temp git repository.
    vllm_repo = tmp_path / "vllm-bisect-repo"
    vllm_repo.mkdir()
    subprocess.run(["git", "-C", str(vllm_repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(vllm_repo), "config", "user.name", "Test Bot"], check=True
    )
    subprocess.run(
        ["git", "-C", str(vllm_repo), "config", "user.email", "bot@example.com"],
        check=True,
    )

    shas: list[str] = []
    for idx in range(31):  # index 0 = last_green_sha, indices 1..30 = 30 window commits
        (vllm_repo / "step.txt").write_text(f"step={idx}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(vllm_repo), "add", "step.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(vllm_repo), "commit", "-q", "-m", f"commit {idx}"],
            check=True,
        )
        sha = subprocess.check_output(
            ["git", "-C", str(vllm_repo), "rev-parse", "HEAD"], text=True
        ).strip()
        shas.append(sha)

    last_green = shas[0]
    window_commits = shas[1:]  # 30 commits
    culprit_idx = 18  # 19th commit in window (shas[19])
    culprit_sha = window_commits[culprit_idx]
    bad_set = set(window_commits[culprit_idx:])

    # Run CLI `ppu_upstream_post_merge.py bisect` end-to-end!
    bad_file = tmp_path / "bad_shas.txt"
    bad_file.write_text("\n".join(sorted(bad_set)) + "\n", encoding="utf-8")
    bisect_out = tmp_path / "bisect-result.json"
    probe_cmd = (
        f'if grep -qx "$CANDIDATE_SHA" "{bad_file}"; then '
        'printf \'{"status":"fail","stage":"plugin_registration","target_signature":"vllm.plugins.load_general_plugins","is_public_plugin_interface":true}\\n\' > "$PROBE_RESULT_JSON"; exit 1; '
        "else "
        'printf \'{"status":"pass","stage":"none","target_signature":"none"}\\n\' > "$PROBE_RESULT_JSON"; exit 0; '
        "fi"
    )
    rc = post_merge.main(
        [
            "bisect",
            "--vllm-repo",
            str(vllm_repo),
            "--last-green-sha",
            last_green,
            "--red-sha",
            window_commits[-1],
            "--probe-cmd",
            probe_cmd,
            "--output-json",
            str(bisect_out),
        ]
    )
    assert rc == 0
    bisect_data = json.loads(bisect_out.read_text(encoding="utf-8"))
    assert bisect_data["bisect_status"] == "converged"
    assert bisect_data["first_failing_sha"] == culprit_sha
    assert bisect_data["last_good_sha"] == window_commits[culprit_idx - 1]
    assert bisect_data["history"][0] == {
        "role": "baseline_control",
        "sha": last_green,
        "stage": "none",
        "status": "pass",
    }
    assert bisect_data["probe_steps_on_range"] <= 5

    # 2. Control failure on last_green_sha -> aborts immediately as infra_error!
    ctrl_fail = post_merge.bisect_first_failing_sha(
        last_green_sha=last_green,
        red_sha=window_commits[-1],
        commits=window_commits,
        probe_fn=lambda _sha: post_merge.ProbeOutcome(
            status="fail", stage="ppu_probe_smoke", target_signature="runner_oom"
        ),
        verify_baseline_first=True,
    )
    assert ctrl_fail["bisect_status"] == "baseline_control_failed"
    assert ctrl_fail["steps_executed"] == 1
    assert ctrl_fail["probe_steps_on_range"] == 0

    # 3. Dual-track classification with non-PPU hard gate:
    #    (a) WITHOUT verified non-PPU repro evidence, even a public plugin registration
    #        failure MUST downgrade to sail_internal_patch_drift!
    pub_unverified_dir = tmp_path / "pub-plugin-unverified"
    unverified_pub = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=_sha(99),
        upstream_sha=window_commits[-1],
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=culprit_sha,
        last_good_sha=window_commits[culprit_idx - 1],
        bisect_steps=bisect_data["steps_executed"],
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="plugin_registration",
            target_signature="vllm.plugins.load_general_plugins",
            is_public_plugin_interface=True,
        ),
        run_url="https://github.com/vllm-project/vllm-sail/actions/runs/12345",
        public_dir=pub_unverified_dir,
    )
    assert (
        unverified_pub["public_summary"]["triage"]["classification"]
        == "sail_internal_patch_drift"
    )
    assert unverified_pub["public_summary"]["triage"]["non_ppu_repro_verified"] is False
    assert unverified_pub["issue_payload"]["should_file_sail_issue"] is True
    assert unverified_pub["issue_payload"]["should_file_upstream_issue"] is False
    assert unverified_pub["issue_payload"]["upstream_reproducer_markdown"] == ""

    #    (b) WITH verified non-PPU repro evidence (last_good exit 0, first_failing exit != 0),
    #        classifies as plugin_interface_break and renders fenced draft with inline stub plugin.
    verified_repro = post_merge.NonPpuReproEvidence(
        mode="stub_plugin_cpu",
        repro_kind="load_general_plugins",
        last_good_sha=window_commits[culprit_idx - 1],
        first_failing_sha=culprit_sha,
        last_good_exit_code=0,
        first_failing_exit_code=1,
        verified=True,
        last_good_phase="stub_check",
        first_failing_phase="stub_check",
        last_good_check_executed=True,
        first_failing_check_executed=True,
        infra_error=False,
        failure_source="stub_plugin_check",
    )
    pub_plugin_dir = tmp_path / "pub-plugin"
    plugin_pub = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=_sha(99),
        upstream_sha=window_commits[-1],
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=culprit_sha,
        last_good_sha=window_commits[culprit_idx - 1],
        bisect_steps=bisect_data["steps_executed"],
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="plugin_registration",
            target_signature="vllm.plugins.load_general_plugins",
            is_public_plugin_interface=True,
            non_ppu_repro=verified_repro,
        ),
        first_failing_author="vllm-maintainer",
        run_url="https://github.com/vllm-project/vllm-sail/actions/runs/12345",
        public_dir=pub_plugin_dir,
    )
    assert (
        plugin_pub["public_summary"]["triage"]["classification"]
        == "plugin_interface_break"
    )
    assert plugin_pub["public_summary"]["triage"]["non_ppu_repro_verified"] is True
    assert (
        plugin_pub["public_summary"]["triage"]["repro_kind"] == "load_general_plugins"
    )
    assert (
        plugin_pub["public_summary"]["bisect"]["first_failing_author"]
        == "vllm-maintainer"
    )
    assert plugin_pub["issue_payload"]["should_file_sail_issue"] is False
    assert plugin_pub["issue_payload"]["should_file_upstream_issue"] is False
    repro = plugin_pub["issue_payload"]["upstream_reproducer_markdown"]
    assert "````markdown\n" in repro
    assert "VLLM_TARGET_DEVICE=empty" in repro
    assert "vllm_stub_plugin" in repro
    assert culprit_sha in repro
    assert window_commits[culprit_idx - 1] in repro
    assert "(cc @vllm-maintainer)" in repro
    outside_fence = repro.split("````markdown\n", 1)[0]
    assert "@vllm-maintainer" not in outside_fence

    pub_drift_dir = tmp_path / "pub-drift"
    drift_pub = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=_sha(99),
        upstream_sha=window_commits[-1],
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=culprit_sha,
        last_good_sha=window_commits[culprit_idx - 1],
        bisect_steps=5,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="patch_drift",
            target_signature="vllm.v1.worker.gpu_model_runner.GPUModelRunner",
            is_public_plugin_interface=False,
        ),
        public_dir=pub_drift_dir,
    )
    assert (
        drift_pub["public_summary"]["triage"]["classification"]
        == "sail_internal_patch_drift"
    )
    assert drift_pub["issue_payload"]["should_file_sail_issue"] is True
    assert (
        "[Post-Merge][sail_internal_patch_drift]["
        in drift_pub["issue_payload"]["title"]
    )
    assert drift_pub["issue_payload"]["upstream_reproducer_markdown"] == ""


def test_gate4_tier_a_kernel_manifest_compatibility_detects_missing_file_and_schema_drift(
    tmp_path,
):
    manifest_data = post_merge._load_toml(REPO_ROOT / "csrc/upstream/manifest.toml")
    fake_vllm = tmp_path / "vllm-src"
    for entry in manifest_data["entry"]:
        if entry.get("tier") == "a":
            target = fake_vllm / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("// stub upstream kernel\n", encoding="utf-8")

    # Copy vendored torch_bindings.cpp so ops.def schemas initially match 100%.
    for up_rel, sail_rel in (
        ("csrc/torch_bindings.cpp", "csrc/upstream/torch_bindings.cpp"),
        ("csrc/moe/torch_bindings.cpp", "csrc/upstream/moe/torch_bindings.cpp"),
    ):
        dst = fake_vllm / up_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / sail_rel, dst)

    assert post_merge.check_kernel_manifest_compatibility(fake_vllm, REPO_ROOT) == []

    # Mutate an ops.def signature in upstream csrc/torch_bindings.cpp -> detected!
    tb = fake_vllm / "csrc/torch_bindings.cpp"
    orig_tb = tb.read_text(encoding="utf-8")
    tb.write_text(
        orig_tb.replace(
            "rms_norm(Tensor! result,", "rms_norm(Tensor! result, int broken_arg,"
        ),
        encoding="utf-8",
    )
    schema_problems = post_merge.check_kernel_manifest_compatibility(
        fake_vllm, REPO_ROOT
    )
    assert "changed_op_schema:rms_norm" in schema_problems

    # Remove a Tier-A .cu source file -> detected!
    tb.write_text(orig_tb, encoding="utf-8")
    first_tier_a = next(
        e["path"] for e in manifest_data["entry"] if e.get("tier") == "a"
    )
    (fake_vllm / first_tier_a).unlink()
    missing_problems = post_merge.check_kernel_manifest_compatibility(
        fake_vllm, REPO_ROOT
    )
    assert f"missing_tier_a_file:{first_tier_a}" in missing_problems


# ---------------------------------------------------------------------------
# Gate 5: Intentional Failure Three-Channel Zero-Leak & Fail-Closed Wipe
# ---------------------------------------------------------------------------


def test_gate5_intentional_failure_three_channel_zero_leak_and_fail_closed_wipe(
    tmp_path,
):
    pub_dir = tmp_path / "public-neg"
    step_summary = tmp_path / "github_step_summary.md"

    build_digest = "sha256:" + "d" * 64
    runtime_digest = "sha256:" + "e" * 64
    tainted_build = f"reg.eng.t-head.cn/internal/sdk@{build_digest}"
    tainted_runtime = f"art.alibaba-inc.com/ppu/vllm@{runtime_digest}"
    tainted_sig = "/nas_aisw/models/Step-5-Preview from 10.23.45.67. token=ghp_1234567890abcdefghij123456 user=wb-998877"
    tainted_url = "https://ci.alibaba-inc.com/mnt/wl_nas/devops/run/192.168.1.8"

    res = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=_sha(42),
        upstream_sha=_sha(43),
        verified_pin=VERIFIED_PIN,
        build_image=tainted_build,
        runtime_image=tainted_runtime,
        cpu_status="pass",
        ppu_status="fail",
        ppu_skip_reason="none",
        first_failing_sha=_sha(43),
        last_good_sha=VERIFIED_PIN,
        bisect_steps=4,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="ppu_probe_smoke",
            target_signature=tainted_sig,
            is_public_plugin_interface=False,
        ),
        run_url=tainted_url,
        public_dir=pub_dir,
        step_summary_path=step_summary,
    )

    channels = [
        res["console"],
        (pub_dir / "public-summary.json").read_text(encoding="utf-8"),
        (pub_dir / "issue-payload.json").read_text(encoding="utf-8"),
        step_summary.read_text(encoding="utf-8"),
    ]
    forbidden_substrings = (
        "eng.t-head.cn",
        "alibaba-inc.com",
        "/nas_aisw",
        "/mnt/wl_nas",
        "10.23.45.67",
        "192.168.1.8",
        "ghp_",
        "Step-5-Preview",
        "wb-998877",
    )
    for text in channels:
        post_merge.assert_zero_sensitive_leaks(text)
        for bad in forbidden_substrings:
            assert bad.lower() not in text.lower()

    assert res["public_summary"]["images"] == {
        "build_digest": build_digest,
        "runtime_digest": runtime_digest,
    }
    assert res["public_summary"]["run_url"] is None

    # Direct fail-closed wipe test: forced leak raises RedactionViolationError AND deletes staging files!
    for sample in (
        "host=/nas_aisw/models/Qwen",
        "mirror=https://art.eng.t-head.cn/pypi",
        "bare_domain=eng.t-head.cn",
        "bare_corp=alibaba-inc.com",
        "node ip 10.23.45.67.",
        "secret ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
        "model Step-5-Preview",
        "bad\x00nul",
    ):
        staged = tmp_path / "staged-leak.json"
        staged.write_text("stale", encoding="utf-8")
        with pytest.raises(post_merge.RedactionViolationError):
            post_merge.assert_zero_sensitive_leaks(sample, staging_paths=[staged])
        assert not staged.exists()

    # Exact Round-1 CLI counterexample 1: bare internal domain on `publish` CLI
    cli_pub_dir = tmp_path / "cli-bare-domain"
    cli_summary = tmp_path / "cli-summary.md"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(42),
                "--upstream-sha",
                _sha(43),
                "--failure-stage",
                "plugin_registration",
                "--target-signature",
                "eng.t-head.cn",
                "--public-plugin-break",
                "--public-dir",
                str(cli_pub_dir),
                "--step-summary",
                str(cli_summary),
            ]
        )
        == 0
    )
    for out_file in (
        cli_pub_dir / "public-summary.json",
        cli_pub_dir / "issue-payload.json",
        cli_summary,
    ):
        txt = out_file.read_text(encoding="utf-8")
        post_merge.assert_zero_sensitive_leaks(txt)
        assert "t-head.cn" not in txt.lower()

    # Exact Round-1 CLI counterexample 2: tainted cursor field in `route` CLI fails closed without leaking
    ls_file = tmp_path / "ls.txt"
    ls_file.write_text(f"{_sha(44)}\trefs/heads/main\n", encoding="utf-8")
    bad_cursor = tmp_path / "bad-cursor.json"
    bad_cursor.write_text(
        json.dumps({"last_ppu_green_sha": "/nas_aisw/review/Step-5-Preview"}),
        encoding="utf-8",
    )
    route_proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/ci/ppu_upstream_post_merge.py"),
            "route",
            "--ls-remote-file",
            str(ls_file),
            "--cursor-file",
            str(bad_cursor),
        ],
        capture_output=True,
        text=True,
    )
    assert route_proc.returncode != 0
    assert "/nas_aisw" not in (route_proc.stdout + route_proc.stderr)
    assert "Step-5-Preview" not in (route_proc.stdout + route_proc.stderr)


# ---------------------------------------------------------------------------
# Gate 6: Dual git ls-remote Polling, Sliding Window & RC Qualification Routing
# ---------------------------------------------------------------------------


def test_gate6_ls_remote_dual_polling_sliding_window_and_rc_routing(tmp_path):
    sha_main_1 = _sha(101)
    sha_main_2 = _sha(102)
    sha_tag_obj = _sha(200)
    sha_v033_commit = _sha(201)
    sha_v033_rc0 = _sha(202)
    sha_v034_rc0 = _sha(300)
    sha_v034_rc1 = _sha(301)

    ls_text = "\n".join(
        [
            f"{sha_main_1}\trefs/heads/main",
            f"{sha_tag_obj}\trefs/tags/v0.33.0",
            f"{sha_v033_commit}\trefs/tags/v0.33.0^{{}}",
            f"{sha_v033_rc0}\trefs/tags/v0.33.0rc0",
        ]
    )
    parsed = post_merge.parse_ls_remote(ls_text)
    assert parsed["main_sha"] == sha_main_1
    # Peeled commit SHA must override annotated tag object SHA!
    assert parsed["tags"]["v0.33.0"] == sha_v033_commit

    # 1. Docs/ROCm-only upstream changes skip PPU smoke WITHOUT advancing last_ppu_tested_sha.
    d1 = post_merge.route_three_gears(
        ls_remote_data=parsed,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        changed_paths_since_ppu=[
            "docs/design.md",
            "csrc/rocm/attention.cu",
            "README.md",
        ],
    )
    assert d1["selected_gear"] == "probe"
    assert d1["run_cpu"] is True
    assert d1["run_ppu"] is False
    assert d1["ppu_skip_reason"] == "allowlisted_non_ppu_changes"

    cursor1 = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_main_1,
        cpu_status="pass",
        ppu_status="skipped",
        ppu_skip_reason="allowlisted_non_ppu_changes",
    )
    assert cursor1["last_cpu_checked_sha"] == sha_main_1
    assert cursor1["last_ppu_tested_sha"] == VERIFIED_PIN

    # 2. Next upstream commit touches vllm/config/model.py -> cumulative diff forces PPU smoke;
    #    if SAIL PR CI is active, probe yields PPU slot without advancing last_ppu_tested_sha!
    parsed2 = {**parsed, "main_sha": sha_main_2}
    d_defer = post_merge.route_three_gears(
        ls_remote_data=parsed2,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor1,
        changed_paths_since_ppu=["docs/design.md", "vllm/config/model.py"],
        active_pr_runs=2,
    )
    assert d_defer["run_ppu"] is False
    assert d_defer["ppu_skip_reason"] == "deferred_for_sail_pr_ci"

    # Once PR CI finishes (active_pr_runs=0), PPU smoke runs on sha_main_2!
    d_run = post_merge.route_three_gears(
        ls_remote_data=parsed2,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor1,
        changed_paths_since_ppu=["docs/design.md", "vllm/compilation/backends.py"],
        active_pr_runs=0,
    )
    assert d_run["run_ppu"] is True
    assert d_run["ppu_skip_reason"] == "none"

    cursor2 = post_merge.advance_cursor(
        cursor1,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_main_2,
        cpu_status="pass",
        ppu_status="pass",
    )
    assert cursor2["last_ppu_tested_sha"] == sha_main_2
    assert cursor2["last_ppu_green_sha"] == sha_main_2

    # 3. New v0.34.0rc0 & v0.34.0rc1 tags appear -> auto-routes to newest untested v0.34.0rc1!
    parsed_rc = {
        "main_sha": sha_main_2,
        "tags": {
            **parsed["tags"],
            "v0.34.0rc0": sha_v034_rc0,
            "v0.34.0rc1": sha_v034_rc1,
        },
    }
    d_rc = post_merge.route_three_gears(
        ls_remote_data=parsed_rc,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor2,
    )
    assert d_rc["selected_gear"] == "rc_qualify"
    assert d_rc["rc_tag"] == "v0.34.0rc1"
    assert d_rc["target_sha"] == sha_v034_rc1
    assert d_rc["run_cpu"] is True and d_rc["run_ppu"] is True

    cursor3 = post_merge.advance_cursor(
        cursor2,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_v034_rc1,
        rc_tag="v0.34.0rc1",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert "v0.34.0rc1" in cursor3["tested_rc_tags"]
    # Verify last_probe_record survived the interleaved rc_qualify run for daily_pin_bump!
    d_bump = post_merge.route_three_gears(
        ls_remote_data=parsed_rc,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.DAILY_PIN_CRON,
        cursor=cursor3,
    )
    assert d_bump["selected_gear"] == "daily_pin_bump"
    assert d_bump["target_sha"] == sha_main_2

    fake_repo = tmp_path / "sail-pin-bump"
    (fake_repo / ".github").mkdir(parents=True)
    (fake_repo / "csrc" / "upstream").mkdir(parents=True)
    pin_file = fake_repo / ".github" / "vllm-main-verified.commit"
    manifest_copy = fake_repo / "csrc" / "upstream" / "manifest.toml"
    pin_file.write_text(VERIFIED_PIN + "\n", encoding="utf-8")
    shutil.copyfile(REPO_ROOT / "csrc" / "upstream" / "manifest.toml", manifest_copy)
    bump_meta = post_merge.prepare_pin_bump(repo_root=fake_repo, target_sha=sha_main_2)
    assert bump_meta["changed"] == "true"
    assert pin_file.read_text(encoding="utf-8") == sha_main_2 + "\n"
    from vllm_sail.native import manifest as kernel_manifest_mod

    assert kernel_manifest_mod.load(manifest_copy).upstream_ref == sha_main_2


def test_gate6_red_sha_repoll_docs_successor_and_rc_skipped_never_turn_green():
    sha_red = _sha(501)
    sha_docs_after_red = _sha(502)

    # 1. CPU pass + PPU fail on sha_red -> re-polling sha_red must re-run PPU (never pass/noop)!
    cursor_red = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_red,
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    repoll = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_red, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor_red,
    )
    assert repoll["run_ppu"] is True
    assert repoll["ppu_skip_reason"] == "none"

    # 2. Docs-only successor on top of red PPU SHA must NOT skip PPU and must NOT be promoted!
    after_red = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_docs_after_red, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor_red,
        changed_paths_since_ppu=["docs/design.md"],
    )
    assert after_red["run_ppu"] is True
    assert (
        post_merge.select_pin_bump_candidate(
            [
                {
                    "gear": "probe",
                    "upstream_sha": sha_docs_after_red,
                    "cpu_status": "pass",
                    "ppu_status": "skipped",
                    "ppu_skip_reason": "allowlisted_non_ppu_changes",
                    "last_ppu_tested_sha": sha_red,
                    "last_ppu_green_sha": VERIFIED_PIN,
                }
            ],
            verified_pin=VERIFIED_PIN,
        )
        is None
    )

    # 2.5 Exact Round-2 residual case: SHA A passes PPU first, then force-rerun on A fails PPU.
    #     Neither re-polling A nor docs-only successor B may skip PPU or be promoted!
    sha_a = _sha(503)
    sha_b_docs = _sha(504)
    cursor_a_green = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_a,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cursor_a_force_red = post_merge.advance_cursor(
        cursor_a_green,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_a,
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    assert cursor_a_force_red["last_ppu_status"] == "fail"
    repoll_a = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_a, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor_a_force_red,
    )
    assert repoll_a["run_ppu"] is True
    assert repoll_a["ppu_skip_reason"] == "none"

    route_b = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_b_docs, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor_a_force_red,
        changed_paths_since_ppu=["docs/design.md"],
    )
    assert route_b["run_ppu"] is True
    assert route_b["ppu_skip_reason"] == "none"

    # Even if a partial cursor dict omits top-level last_ppu_status, last_probe_record.ppu_status=="fail" still blocks skip!
    legacy_shape = {
        k: v
        for k, v in cursor_a_force_red.items()
        if k not in ("last_cpu_status", "last_ppu_status")
    }
    legacy_shape["last_probe_record"] = {
        k: v
        for k, v in cursor_a_force_red["last_probe_record"].items()
        if k != "last_ppu_result_status"
    }
    route_b_legacy = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_b_docs, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=legacy_shape,
        changed_paths_since_ppu=["docs/design.md"],
    )
    assert route_b_legacy["run_ppu"] is True
    assert (
        post_merge.select_pin_bump_candidate(
            [
                {
                    "gear": "probe",
                    "upstream_sha": sha_b_docs,
                    "cpu_status": "pass",
                    "ppu_status": "skipped",
                    "ppu_skip_reason": "allowlisted_non_ppu_changes",
                    "last_ppu_tested_sha": sha_a,
                    "last_ppu_green_sha": sha_a,
                    "last_ppu_result_status": "fail",
                }
            ],
            verified_pin=VERIFIED_PIN,
        )
        is None
    )

    # 3. Workflow outcome reducer: skipped RC gate or failed ops-select fails closed!
    rc_skipped = post_merge.evaluate_workflow_outcome(
        selected_gear="rc_qualify",
        target_sha=sha_red,
        run_cpu=True,
        run_ppu=True,
        ppu_skip_reason="none",
        build_result="success",
        cpu_result="success",
        rc_pin_result="success",
        rc_fast_result="success",
        rc_model_result="skipped",
    )
    assert rc_skipped["status"] == "fail"
    assert rc_skipped["failure_stage"] == "rc_model_sanity"

    rc_ops_fail = post_merge.evaluate_workflow_outcome(
        selected_gear="rc_qualify",
        target_sha=sha_red,
        run_cpu=True,
        run_ppu=True,
        ppu_skip_reason="none",
        ops_image="sha256:" + "f" * 64,
        build_result="success",
        cpu_result="success",
        rc_pin_result="success",
        rc_fast_result="success",
        rc_model_result="success",
        rc_ops_select_result="failure",
        rc_ops_result="skipped",
    )
    assert rc_ops_fail["status"] == "fail"
    assert rc_ops_fail["failure_stage"] == "rc_ops_tests"


def test_gate6_upstream_canary_workflow_three_gears_contract():
    wf_path = REPO_ROOT / ".github/workflows/upstream-canary.yaml"
    wf_raw = wf_path.read_text(encoding="utf-8")
    wf = yaml.safe_load(wf_raw)
    triggers = wf.get(True) or wf.get("on")

    # No workflow_call, no top-level cache-mode, no statuses:write, no legacy 60k raw log paste.
    assert "workflow_call" not in triggers
    assert "cache-mode" not in wf
    assert "slice(0, 60000)" not in wf_raw
    assert "_ppu-upstream-reuse.yaml" not in wf_raw
    assert wf["permissions"] == {"contents": "read", "actions": "read"}

    crons = [entry["cron"] for entry in triggers["schedule"]]
    assert post_merge.PROBE_CRON in crons
    assert post_merge.DAILY_PIN_CRON in crons

    jobs = wf["jobs"]
    for job in jobs.values():
        assert "statuses" not in job.get("permissions", {})

    assert "pytorch@sha256:" in jobs["cpu-fast-check"]["container"]["image"]
    assert jobs["ppu-probe-smoke"]["runs-on"] == "8-ppu-m890p"
    assert "!cancelled()" in jobs["ppu-probe-smoke"]["if"]
    assert jobs["ppu-probe-smoke"]["concurrency"] == {
        "group": "upstream-post-merge-ppu-slot",
        "cancel-in-progress": False,
    }
    assert wf["concurrency"] == {
        "group": "upstream-post-merge-state",
        "cancel-in-progress": False,
    }
    assert jobs["build"]["with"]["diagnostics_mode"] == "public"
    assert jobs["build"]["with"]["resolve_native_donors"] is True
    assert "ppu-build-and-test.yaml" not in wf_raw
    for rc_job in ("rc-pin-ut", "rc-fast-sanity", "rc-model-sanity", "rc-ops-tests"):
        assert "rc_qualify" in jobs[rc_job]["if"]
    route_run = next(s["run"] for s in jobs["route"]["steps"] if s.get("id") == "route")
    assert (
        'git ls-remote https://github.com/vllm-project/vllm.git refs/heads/main "refs/heads/releases/v*" "refs/tags/v*"'
        in route_run
    )


# ---------------------------------------------------------------------------
# PR #374 Review Regression Suite (All 12 Counterexamples + Controls)
# ---------------------------------------------------------------------------


def test_review_r1_container_build_candidate_export_works_under_posix_sh_and_bash():
    candidate = "b" * 40
    assert candidate != VERIFIED_PIN
    shells = ["sh", "bash"]
    if shutil.which("dash"):
        shells.append("dash")

    for wf_name in ("_ppu-build.yaml", "ppu-ops-tests.yaml"):
        jobs = yaml.safe_load((REPO_ROOT / ".github/workflows" / wf_name).read_text())[
            "jobs"
        ]
        job = next(
            j
            for j in jobs.values()
            if any(
                "export VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT" in s.get("run", "")
                for s in j.get("steps", [])
            )
        )
        step = next(
            s
            for s in job["steps"]
            if "export VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT" in s.get("run", "")
        )
        assert step.get("shell") == "bash"
        snippet = step["run"].replace(
            "bash scripts/ci/build_ppu_wheels.sh",
            'printf "%s" "${VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT:-unset}"',
        )
        for sh in shells:
            proc = subprocess.run(
                [sh, "-e", "-c", snippet],
                cwd=REPO_ROOT,
                env={**os.environ, "VLLM_REF": candidate},
                text=True,
                capture_output=True,
                check=True,
            )
            assert proc.stdout == candidate
            pin_proc = subprocess.run(
                [sh, "-e", "-c", snippet],
                cwd=REPO_ROOT,
                env={**os.environ, "VLLM_REF": VERIFIED_PIN},
                text=True,
                capture_output=True,
                check=True,
            )
            assert pin_proc.stdout == "unset"


def test_review_r2_real_patch_baseline_opaque_fingerprints_pass_redaction_and_diff(
    tmp_path,
):
    from types import SimpleNamespace

    from tools import check_patch_drift

    def _to_records(source_by_target: dict[str, str | None]):
        return [
            SimpleNamespace(target=k, original_source=v)
            for k, v in source_by_target.items()
        ]

    real_baseline = json.loads(
        (REPO_ROOT / "tools/patch_baseline.json").read_text(encoding="utf-8")
    )
    assert len(real_baseline) >= 300
    # Raw baseline contains confidential/long symbols that trip public redaction.
    with pytest.raises(post_merge.RedactionViolationError):
        post_merge.assert_zero_sensitive_leaks(
            json.dumps({"patch_fingerprints": real_baseline})
        )

    # Opaque mapping serializes cleanly through both redaction and cursor sanitization,
    # while still detecting incremental drift on any of the 376 targets.
    base_sources = {
        k: (None if v == "additive" else f"body:{k}:{v}")
        for k, v in real_baseline.items()
    }
    recorded = check_patch_drift._fingerprints(_to_records(base_sources))
    drift_clean = check_patch_drift.evaluate_drift(_to_records(base_sources), recorded)
    opaque_map = drift_clean["opaque_fingerprints"]
    assert len(opaque_map) == len(real_baseline)
    cpu_payload = json.dumps({"patch_fingerprints": opaque_map}, sort_keys=True)
    post_merge.assert_zero_sensitive_leaks(cpu_payload)

    cursor = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=_sha(601),
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
        patch_fingerprints=opaque_map,
    )
    cursor_file = tmp_path / "cursor.json"
    cursor_text = json.dumps(cursor, indent=2, sort_keys=True)
    post_merge.assert_zero_sensitive_leaks(cursor_text)
    cursor_file.write_text(cursor_text, encoding="utf-8")

    # Simulate a known soft drift recorded in cursor.json followed by an incremental drift on a step5 target!
    step5_key = next(k for k in real_baseline if "step5" in k.lower())
    prev_sources = {**base_sources, step5_key: "soft-drifted-body-v1"}
    prev_opaque = check_patch_drift.evaluate_drift(_to_records(prev_sources), recorded)[
        "opaque_fingerprints"
    ]
    cursor_soft = post_merge.advance_cursor(
        cursor,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=_sha(602),
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
        patch_fingerprints=prev_opaque,
    )
    prev_file = tmp_path / "prev-cursor.json"
    prev_file.write_text(json.dumps(cursor_soft), encoding="utf-8")
    loaded_prev = check_patch_drift._read_mapping_file(prev_file, "previous baseline")

    same_soft = check_patch_drift.evaluate_drift(
        _to_records(prev_sources), recorded, previous=loaded_prev
    )
    assert same_soft["known_soft_drifted"] == [step5_key]
    assert same_soft["incremental_drifted"] == []

    new_drift = check_patch_drift.evaluate_drift(
        _to_records({**base_sources, step5_key: "incremental-drifted-body-v2"}),
        recorded,
        previous=loaded_prev,
    )
    assert new_drift["incremental_drifted"] == [step5_key]
    safe_sig = post_merge._sanitize_patch_target_signature(step5_key)
    assert safe_sig.startswith("opaque_target:t_")
    post_merge.assert_zero_sensitive_leaks(safe_sig)


def test_review_r3_r4_r9_r10_promotion_eligibility_deferred_merge_and_cursor_discipline(
    tmp_path,
):
    sha_b = _sha(701)
    sha_c = _sha(702)

    def _cycle(state=None, gear="probe", target=sha_b, permute=None, **kw):
        decision = post_merge.route_three_gears(
            ls_remote_data={"main_sha": target, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear=gear,
            cursor=state,
            **kw,
        )
        args = {
            "selected_gear": decision["selected_gear"],
            "target_sha": decision["target_sha"],
            "run_cpu": decision["run_cpu"],
            "run_ppu": decision["run_ppu"],
            "ppu_skip_reason": decision["ppu_skip_reason"],
        }
        jobs = {
            "build_result": "success"
            if args["run_cpu"] or args["run_ppu"]
            else "skipped",
            "cpu_result": "success" if args["run_cpu"] else "skipped",
            "ppu_smoke_result": "success"
            if args["run_ppu"] and args["selected_gear"] == "probe"
            else "skipped",
        }
        outcome = post_merge.evaluate_workflow_outcome(
            **args, **(jobs | (permute or {}))
        )
        next_state = post_merge.advance_cursor(
            state,
            verified_pin=VERIFIED_PIN,
            selected_gear=args["selected_gear"],
            target_sha=args["target_sha"],
            rc_tag=decision["rc_tag"],
            status=outcome["status"],
            cpu_status=outcome["cpu_status"],
            ppu_status=outcome["ppu_status"],
            ppu_skip_reason=outcome["ppu_skip_reason"],
        )
        return decision, outcome, next_state

    # R3: Prior passing probe on B followed by forced wheel-build failure invalidates B!
    _, _, prior = _cycle()
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=prior,
        )["target_sha"]
        == sha_b
    )
    _, rerun, failed_cursor = _cycle(
        prior,
        force=True,
        permute={
            "build_result": "failure",
            "cpu_result": "skipped",
            "ppu_smoke_result": "skipped",
        },
    )
    assert rerun["status"] == "fail"
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=failed_cursor,
        )["target_sha"]
        == ""
    )
    next_probe_after_build_fail = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_b, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        requested_gear="probe",
        cursor=failed_cursor,
    )
    assert next_probe_after_build_fail["run_cpu"] is True
    assert next_probe_after_build_fail["ppu_skip_reason"] != "already_tested"

    # R4: Manual daily_pin_bump override rejects unprobed SHAs and accepts verified SHAs.
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=None,
            upstream_ref_override=sha_c,
        )["target_sha"]
        == ""
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=prior,
            upstream_ref_override=sha_b,
        )["target_sha"]
        == sha_b
    )

    # R9: Split 2-run probe (CPU pass + PPU deferred, then PPU pass) promotes B!
    _, out1, st1 = _cycle(active_pr_runs=1)
    assert out1["status"] == "deferred"
    d2, out2, st2 = _cycle(st1)
    assert d2["run_cpu"] is False and d2["run_ppu"] is True
    assert out2["status"] == "pass"
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=st2,
        )["target_sha"]
        == sha_b
    )

    # R10: Read-only daily_pin_bump and noop runs never write cursor.json in `publish`.
    pub_daily = tmp_path / "pub-daily"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "daily_pin_bump",
                "--status",
                "pass",
                "--sail-sha",
                _sha(709),
                "--upstream-sha",
                sha_b,
                "--public-dir",
                str(pub_daily),
            ]
        )
        == 0
    )
    assert not (pub_daily / "cursor.json").exists()


def test_review_r5_rc_red_and_green_never_starve_or_pollute_main_probe():
    sha_rc = _sha(720)
    sha_main = _sha(721)
    tags = {"v0.34.0rc1": sha_rc}

    # 1. Failed RC runs once on auto schedule; next auto poll probes advancing main!
    d_rc = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_main, "tags": tags},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        requested_gear="auto",
    )
    assert d_rc["selected_gear"] == "rc_qualify" and d_rc["target_sha"] == sha_rc
    red_rc_cursor = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc,
        rc_tag="v0.34.0rc1",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    after_red_rc = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_main, "tags": tags},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        requested_gear="auto",
        cursor=red_rc_cursor,
    )
    assert after_red_rc["selected_gear"] == "probe"
    assert after_red_rc["target_sha"] == sha_main
    assert after_red_rc["last_green_sha"] == VERIFIED_PIN

    # 2. Passing RC records RC tag without overwriting main's last_ppu_green_sha!
    green_rc_cursor = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc,
        rc_tag="v0.34.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert green_rc_cursor["last_ppu_green_sha"] == VERIFIED_PIN
    assert green_rc_cursor["last_cpu_green_sha"] == VERIFIED_PIN
    after_green_rc = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_main, "tags": tags},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        requested_gear="auto",
        cursor=green_rc_cursor,
    )
    assert after_green_rc["selected_gear"] == "probe"
    assert after_green_rc["last_green_sha"] == VERIFIED_PIN


def test_review_r6_r12_shallow_bisect_completeness_deepening_and_precise_publish_stage(
    tmp_path,
):
    src = tmp_path / "upstream-full"
    shallow_no_origin = tmp_path / "shallow-disconnected"
    shallow_with_origin = tmp_path / "shallow-origin"
    for d in (src, shallow_no_origin, shallow_with_origin):
        d.mkdir()
        subprocess.run(["git", "-C", str(d), "init", "-q"], check=True)

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.org",
        "GIT_COMMITTER_EMAIL": "test@example.org",
    }
    commits: list[str] = []
    for i in range(6):
        (src / "state").write_text(f"{i}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(src), "add", "state"], check=True)
        subprocess.run(
            ["git", "-C", str(src), "commit", "-qm", str(i)],
            check=True,
            env=git_env,
        )
        commits.append(
            subprocess.check_output(
                ["git", "-C", str(src), "rev-parse", "HEAD"], text=True
            ).strip()
        )

    url = src.as_uri()
    # Disconnected shallow clone (no `origin` remote): must report unresolved_history with NO false culprit!
    subprocess.run(
        [
            "git",
            "-C",
            str(shallow_no_origin),
            "fetch",
            "-q",
            "--depth=2",
            url,
            commits[-1],
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(shallow_no_origin),
            "fetch",
            "-q",
            "--depth=1",
            url,
            commits[0],
        ],
        check=True,
    )
    out_disc = tmp_path / "disc.json"
    assert (
        post_merge.main(
            [
                "bisect",
                "--vllm-repo",
                str(shallow_no_origin),
                "--last-green-sha",
                commits[0],
                "--red-sha",
                commits[-1],
                "--probe-cmd",
                f'test "$CANDIDATE_SHA" = {commits[0]}',
                "--output-json",
                str(out_disc),
            ]
        )
        == 0
    )
    disc_data = json.loads(out_disc.read_text(encoding="utf-8"))
    assert disc_data["bisect_status"] == "unresolved_history"
    assert disc_data["first_failing_sha"] == ""

    # Shallow clone WITH `origin` remote: automatically deepens until connected and finds exact commit 1!
    subprocess.run(
        ["git", "-C", str(shallow_with_origin), "remote", "add", "origin", url],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(shallow_with_origin),
            "fetch",
            "-q",
            "--depth=2",
            "origin",
            commits[-1],
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(shallow_with_origin),
            "fetch",
            "-q",
            "--depth=1",
            "origin",
            commits[0],
        ],
        check=True,
    )
    out_deep = tmp_path / "deep.json"
    assert (
        post_merge.main(
            [
                "bisect",
                "--vllm-repo",
                str(shallow_with_origin),
                "--last-green-sha",
                commits[0],
                "--red-sha",
                commits[-1],
                "--probe-cmd",
                f'test "$CANDIDATE_SHA" = {commits[0]}',
                "--output-json",
                str(out_deep),
            ]
        )
        == 0
    )
    deep_data = json.loads(out_deep.read_text(encoding="utf-8"))
    assert deep_data["bisect_status"] == "converged"
    assert deep_data["first_failing_sha"] == commits[1]

    # R6 + R12 in `publish`: unbisected failures have first_failing_sha=None; measured cpu stage wins!
    no_bisect_dir = tmp_path / "pub-no-bisect"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(730),
                "--upstream-sha",
                commits[-1],
                "--failure-stage",
                "wheel_build",
                "--public-dir",
                str(no_bisect_dir),
            ]
        )
        == 0
    )
    no_bisect_sum = json.loads(
        (no_bisect_dir / "public-summary.json").read_text(encoding="utf-8")
    )
    assert no_bisect_sum["bisect"]["first_failing_sha"] is None
    assert no_bisect_sum["triage"]["stage"] == "wheel_build"

    cpu_res_file = tmp_path / "cpu-res.json"
    cpu_res_file.write_text(
        json.dumps(
            {
                "stage": "patch_drift",
                "status": "fail",
                "target_signature": "vllm.config.Example",
            }
        ),
        encoding="utf-8",
    )
    detail_dir = tmp_path / "pub-cpu-detail"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(730),
                "--upstream-sha",
                commits[-1],
                "--failure-stage",
                "plugin_registration",
                "--cpu-result-json",
                str(cpu_res_file),
                "--public-dir",
                str(detail_dir),
            ]
        )
        == 0
    )
    detail_sum = json.loads(
        (detail_dir / "public-summary.json").read_text(encoding="utf-8")
    )
    assert detail_sum["triage"]["stage"] == "patch_drift"
    assert detail_sum["triage"]["classification"] == "sail_internal_patch_drift"


def test_review_r7_r8_moe_sum_schema_parity_and_sanitized_build_diagnostics(tmp_path):
    # R7: Every single compiled Tier-A op is extracted from vendored bindings, including `m.def("moe_sum...")`!
    manifest_data = post_merge._load_toml(REPO_ROOT / "csrc/upstream/manifest.toml")
    fake_vllm = tmp_path / "vllm-tier-a"
    for entry in manifest_data["entry"]:
        if entry.get("tier") == "a":
            target = fake_vllm / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("// stub\n", encoding="utf-8")
    for up_rel, sail_rel in (
        ("csrc/torch_bindings.cpp", "csrc/upstream/torch_bindings.cpp"),
        ("csrc/moe/torch_bindings.cpp", "csrc/upstream/moe/torch_bindings.cpp"),
    ):
        dst = fake_vllm / up_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / sail_rel, dst)

    assert post_merge.check_kernel_manifest_compatibility(fake_vllm, REPO_ROOT) == []
    moe_tb = fake_vllm / "csrc/moe/torch_bindings.cpp"
    orig_moe = moe_tb.read_text(encoding="utf-8")
    assert "moe_sum(Tensor input," in orig_moe
    moe_tb.write_text(
        orig_moe.replace(
            "moe_sum(Tensor input,", "moe_sum(int breaking_arg, Tensor input,"
        ),
        encoding="utf-8",
    )
    moe_problems = post_merge.check_kernel_manifest_compatibility(fake_vllm, REPO_ROOT)
    assert "changed_op_schema:moe_sum" in moe_problems

    # R8: `sanitize-build-manifest` strips internal registry/host paths and passes redaction gate.
    raw_manifest = tmp_path / "build-manifest.txt"
    raw_manifest.write_text(
        "\n".join(
            [
                "status=running",
                f"vllm_ref={VERIFIED_PIN}",
                "pytorch_sail_arch=ppu_15;ppu_10",
                "build_image=download.t-head.cn/docker_release/pytorch@sha256:"
                + "9" * 64,
                "reuse_mode=sail-native",
                "reuse_source=/home/runner/work/vllm-sail/artifacts/native-reuse",
                "hgcc=hgcc (T-Head PPU) /ppusw/sdk",
                "status=succeeded",
                f"vllm_commit={VERIFIED_PIN}",
                f"vllm_sail_commit={_sha(740)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        post_merge.main(["sanitize-build-manifest", "--manifest", str(raw_manifest)])
        == 0
    )
    cleaned = raw_manifest.read_text(encoding="utf-8")
    post_merge.assert_zero_sensitive_leaks(cleaned)
    assert "t-head.cn" not in cleaned
    assert "/home/runner" not in cleaned
    assert "/ppusw" not in cleaned
    assert f"build_image_digest=sha256:{'9' * 64}" in cleaned
    assert "status=succeeded" in cleaned


def test_review_r2_sub1_no_pin_downgrade_fail_closed_manifest_and_rc_ops_digest(
    tmp_path, monkeypatch
):
    from scripts.ci import ppu_ops_select

    sha_a = VERIFIED_PIN
    sha_b = _sha(801)
    sha_c = _sha(802)

    # 1. Counterexample 1: B then C pass probe -> C merges as current verified_pin ->
    #    daily_pin_bump (both auto and manual override=B) must NOT downgrade C -> B!
    cur1 = post_merge.advance_cursor(
        None,
        verified_pin=sha_a,
        selected_gear="probe",
        target_sha=sha_b,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur2 = post_merge.advance_cursor(
        cur1,
        verified_pin=sha_a,
        selected_gear="probe",
        target_sha=sha_c,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    # Before C merges (pin == A): selects C (and manual override=B selects forward B).
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {}},
            verified_pin=sha_a,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur2,
        )["target_sha"]
        == sha_c
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {}},
            verified_pin=sha_a,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur2,
            upstream_ref_override=sha_b,
        )["target_sha"]
        == sha_b
    )
    # After C merges (verified_pin == C): auto AND manual override=B MUST return ""!
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {}},
            verified_pin=sha_c,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur2,
        )["target_sha"]
        == ""
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {}},
            verified_pin=sha_c,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur2,
            upstream_ref_override=sha_b,
        )["target_sha"]
        == ""
    )

    # Also verify prepare_pin_bump rejects non-descendant target_sha when --vllm-repo is provided.
    vllm_git = tmp_path / "vllm-ancestry"
    vllm_git.mkdir()
    subprocess.run(["git", "-C", str(vllm_git), "init", "-q"], check=True)
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.org",
        "GIT_COMMITTER_EMAIL": "test@example.org",
    }
    ancestry_shas: list[str] = []
    for idx in range(2):
        (vllm_git / "f.txt").write_text(f"{idx}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(vllm_git), "add", "f.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(vllm_git), "commit", "-qm", str(idx)],
            check=True,
            env=git_env,
        )
        ancestry_shas.append(
            subprocess.check_output(
                ["git", "-C", str(vllm_git), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    fake_sail = tmp_path / "sail-downgrade"
    (fake_sail / ".github").mkdir(parents=True)
    (fake_sail / "csrc" / "upstream").mkdir(parents=True)
    pin_f = fake_sail / ".github" / "vllm-main-verified.commit"
    man_f = fake_sail / "csrc" / "upstream" / "manifest.toml"
    shutil.copyfile(REPO_ROOT / "csrc/upstream/manifest.toml", man_f)
    # Current pin = newer commit ancestry_shas[1]; attempting to bump to older ancestry_shas[0] fails!
    pin_f.write_text(ancestry_shas[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="first-parent descendant"):
        post_merge.prepare_pin_bump(
            repo_root=fake_sail,
            target_sha=ancestry_shas[0],
            vllm_repo=vllm_git,
        )
    # Forward bump ancestry_shas[0] -> ancestry_shas[1] succeeds!
    pin_f.write_text(ancestry_shas[0] + "\n", encoding="utf-8")
    ok_bump = post_merge.prepare_pin_bump(
        repo_root=fake_sail,
        target_sha=ancestry_shas[1],
        vllm_repo=vllm_git,
    )
    assert ok_bump["changed"] == "true"

    # 2. Counterexample 2: Fault injection before rewrite in sanitize_build_manifest_file
    #    wipes raw manifest (fail-closed), and _ppu-build.yaml gates public upload on sanitize success.
    fault_manifest = tmp_path / "raw-fault-manifest.txt"
    fault_manifest.write_text(
        "status=succeeded\nbuild_image=download.t-head.cn/docker_release/pytorch@sha256:"
        + "a" * 64
        + "\n",
        encoding="utf-8",
    )
    orig_read_text = Path.read_text

    def _boom(self: Path, *a, **kw):
        if self == fault_manifest:
            raise OSError("simulated read failure before rewrite")
        return orig_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _boom)
    with pytest.raises(OSError, match="simulated read failure"):
        post_merge.sanitize_build_manifest_file(fault_manifest)
    monkeypatch.setattr(Path, "read_text", orig_read_text)
    assert not fault_manifest.exists()

    build_wf = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/_ppu-build.yaml").read_text(encoding="utf-8")
    )["jobs"]["build"]["steps"]
    sanitize_step = next(
        s for s in build_wf if s.get("name") == "Sanitize public build diagnostics"
    )
    upload_diag = next(
        s for s in build_wf if s.get("name") == "Upload build diagnostics"
    )
    upload_public_diag = next(
        s
        for s in build_wf
        if s.get("name") == "Upload sanitized public build diagnostics"
    )
    assert sanitize_step.get("id") == "sanitize_diagnostics"
    assert "inputs.diagnostics_mode != 'public'" in upload_diag["if"]
    assert r"\n" not in upload_diag["with"]["path"]
    assert "${{" not in upload_diag["with"]["path"]
    assert [
        line.strip()
        for line in upload_diag["with"]["path"].splitlines()
        if line.strip()
    ] == ["artifacts/logs/**", "artifacts/build-manifest.txt"]
    assert "inputs.diagnostics_mode == 'public'" in upload_public_diag["if"]
    assert "steps.sanitize_diagnostics.outcome == 'success'" in upload_public_diag["if"]
    assert upload_public_diag["with"]["path"].strip() == "artifacts/build-manifest.txt"

    # 3. Counterexample 3: Both raw and sanitized build-manifest.txt are accepted by
    #    ppu_ops_select.build_fingerprint (used by both rc-ops-select and ppu_ops_runner)!
    sail_sha = _sha(810)
    vllm_sha = _sha(811)
    img_digest = "7" * 64
    full_build_image = f"download.t-head.cn/docker_release/pytorch@sha256:{img_digest}"
    wheel_man = {"commits": {"vllm": vllm_sha, "vllm-sail": sail_sha}}

    raw_diag_file = tmp_path / "diag-raw.txt"
    san_diag_file = tmp_path / "diag-sanitized.txt"
    raw_diag_file.write_text(
        "\n".join(
            [
                "status=succeeded",
                f"vllm_commit={vllm_sha}",
                f"vllm_sail_commit={sail_sha}",
                f"build_image={full_build_image}",
                "pytorch_sail_arch=ppu_15;ppu_10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(raw_diag_file, san_diag_file)
    post_merge.sanitize_build_manifest_file(san_diag_file)
    post_merge.assert_zero_sensitive_leaks(san_diag_file.read_text(encoding="utf-8"))

    fp_raw = ppu_ops_select.build_fingerprint(
        ppu_ops_select.read_diagnostics(raw_diag_file),
        wheel_man,
        tested_sha=sail_sha,
        expected_vllm_commit=vllm_sha,
        build_image=full_build_image,
    )
    fp_san = ppu_ops_select.build_fingerprint(
        ppu_ops_select.read_diagnostics(san_diag_file),
        wheel_man,
        tested_sha=sail_sha,
        expected_vllm_commit=vllm_sha,
        build_image=full_build_image,
    )
    assert fp_san == fp_raw
    # Wrong digest in sanitized manifest must still fail closed!
    wrong_image = f"download.t-head.cn/docker_release/pytorch@sha256:{'8' * 64}"
    with pytest.raises(ValueError, match="Observed build image mismatch"):
        ppu_ops_select.build_fingerprint(
            ppu_ops_select.read_diagnostics(san_diag_file),
            wheel_man,
            tested_sha=sail_sha,
            expected_vllm_commit=vllm_sha,
            build_image=wrong_image,
        )


# ---------------------------------------------------------------------------
# 2026-11-24 Maintainer Refinement: Non-PPU Repro Gate, Stub Plugins & @author
# ---------------------------------------------------------------------------


def _make_mock_vllm_git_repo(repo_dir: Path) -> dict[str, str]:
    """Create a real multi-commit git repo with a minimal stdlib vLLM plugin surface."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo_dir), "init", "-q"], check=True)
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Upstream Maintainer",
        "GIT_COMMITTER_NAME": "Upstream Maintainer",
        "GIT_AUTHOR_EMAIL": "12345+upstream-dev@users.noreply.github.com",
        "GIT_COMMITTER_EMAIL": "12345+upstream-dev@users.noreply.github.com",
    }

    def _write_healthy_tree() -> None:
        files = {
            "vllm/__init__.py": (
                '__version__ = "0.34.0.dev0"\n'
                "from vllm.model_executor.models.registry import ModelRegistry\n"
                '__all__ = ["ModelRegistry", "__version__"]\n'
            ),
            "vllm/plugins/__init__.py": (
                "import importlib.metadata\n"
                "import os\n\n"
                "def load_plugins_by_group(group: str) -> dict:\n"
                '    raw = os.environ.get("VLLM_PLUGINS")\n'
                '    allowed = None if raw is None else {p.strip() for p in raw.split(",") if p.strip()}\n'
                "    out = {}\n"
                "    for ep in importlib.metadata.entry_points(group=group):\n"
                "        if allowed is not None and ep.name not in allowed:\n"
                "            continue\n"
                "        out[ep.name] = ep.load()\n"
                "    return out\n\n"
                "def load_general_plugins() -> None:\n"
                '    for fn in load_plugins_by_group("vllm.general_plugins").values():\n'
                "        fn()\n"
            ),
            "vllm/platforms/__init__.py": (
                "import importlib\n"
                "from vllm.platforms.interface import Platform\n"
                "from vllm.plugins import load_plugins_by_group\n\n"
                "def _resolve_platform() -> Platform:\n"
                '    plugins = load_plugins_by_group("vllm.platform_plugins")\n'
                "    for fn in plugins.values():\n"
                "        qual = fn()\n"
                '        mod_name, attr_name = qual.rsplit(".", 1)\n'
                "        cls = getattr(importlib.import_module(mod_name), attr_name)\n"
                "        return cls()\n"
                "    return Platform()\n\n"
                "current_platform = _resolve_platform()\n"
            ),
            "vllm/platforms/interface.py": (
                "from enum import Enum\n\n"
                "class PlatformEnum(Enum):\n"
                '    CUDA = "cuda"\n'
                '    OOT = "oot"\n\n'
                "class Platform:\n"
                "    _enum = PlatformEnum.OOT\n"
                '    device_name = "unspecified"\n'
                '    device_type = "cpu"\n\n'
                "    def is_out_of_tree(self) -> bool:\n"
                "        return self._enum == PlatformEnum.OOT\n"
            ),
            "vllm/model_executor/__init__.py": "",
            "vllm/model_executor/custom_op.py": (
                "op_registry = {}\n"
                "op_registry_oot = {}\n\n\n"
                "class CustomOp:\n"
                "    def __new__(cls, *args, **kwargs):\n"
                "        op_name = cls.__name__\n"
                "        target = op_registry_oot.get(op_name, cls)\n"
                "        return super().__new__(target)\n\n"
                "    @classmethod\n"
                "    def register(cls, name: str, dynamic_arg_dims=None):\n"
                "        def _dec(sub):\n"
                "            sub.name = name\n"
                "            op_registry[name] = sub\n"
                "            return sub\n"
                "        return _dec\n\n"
                "    @classmethod\n"
                "    def register_oot(cls, _decorated_op_cls=None, name: str | None = None):\n"
                "        def _dec(sub):\n"
                "            reg_name = name if name is not None else cls.__name__\n"
                "            sub.name = reg_name\n"
                "            op_registry_oot[reg_name] = sub\n"
                "            return sub\n"
                "        return _dec if _decorated_op_cls is None else _dec(_decorated_op_cls)\n"
            ),
            "vllm/model_executor/models/__init__.py": "",
            "vllm/model_executor/models/registry.py": (
                "class _Registry:\n"
                "    def __init__(self) -> None:\n"
                "        self.models = {}\n\n"
                "    def register_model(self, arch: str, target: str) -> None:\n"
                "        self.models[arch] = target\n\n"
                "    def get_supported_archs(self) -> set[str]:\n"
                "        return set(self.models)\n\n"
                "ModelRegistry = _Registry()\n"
            ),
        }
        for rel, content in files.items():
            p = repo_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def _commit(msg: str) -> str:
        subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-qm", msg],
            check=True,
            env=git_env,
        )
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()

    _write_healthy_tree()
    good_sha = _commit("healthy baseline")
    shas = {"good": good_sha}

    # Second healthy commit (negative control: must NOT verify as a regression).
    (repo_dir / "README.md").write_text("docs touch\n", encoding="utf-8")
    shas["good_2"] = _commit("docs only change")

    # Create independent regression commits branching off `good_sha` for each of the 5 kinds.
    mutations = {
        "cpu_empty_build": (
            "vllm/__init__.py",
            'raise RuntimeError("VLLM_TARGET_DEVICE=empty build broken")\n',
        ),
        "platform_plugin": (
            "vllm/platforms/interface.py",
            "from enum import Enum\n"
            "class PlatformEnum(Enum):\n"
            '    CUDA = "cuda"\n'
            "class Platform:\n"
            "    pass\n",
        ),
        "register_oot": (
            "vllm/model_executor/custom_op.py",
            "op_registry = {}\n"
            "op_registry_oot = {}\n\n\n"
            "class CustomOp:\n"
            "    def __new__(cls, *args, **kwargs):\n"
            "        return super().__new__(cls)\n\n"
            "    @classmethod\n"
            "    def register(cls, name: str, dynamic_arg_dims=None):\n"
            "        def _dec(sub):\n"
            "            sub.name = name\n"
            "            op_registry[name] = sub\n"
            "            return sub\n"
            "        return _dec\n",
        ),
        "model_registry": (
            "vllm/model_executor/models/registry.py",
            "class _Registry:\n"
            "    def __init__(self) -> None:\n"
            "        self.models = {}\n"
            "    def get_supported_archs(self) -> set[str]:\n"
            "        return set(self.models)\n"
            "ModelRegistry = _Registry()\n",
        ),
        "load_general_plugins": (
            "vllm/plugins/__init__.py",
            "def load_plugins_by_group(group: str) -> dict:\n"
            "    return {}\n"
            "def load_general_plugins() -> None:\n"
            "    return None\n",
        ),
    }
    for kind, (rel_path, broken_body) in mutations.items():
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "-q", "--detach", good_sha],
            check=True,
        )
        (repo_dir / rel_path).write_text(broken_body, encoding="utf-8")
        shas[kind] = _commit(f"break {kind}")

    return shas


@pytest.mark.parametrize(
    ("stage", "target_signature", "expected_kind", "expected_mode"),
    [
        (
            "wheel_build",
            "VLLM_TARGET_DEVICE=empty",
            "cpu_empty_build",
            "cpu_empty_build",
        ),
        (
            "plugin_registration",
            "vllm.platforms.Platform",
            "platform_plugin",
            "stub_plugin_cpu",
        ),
        (
            "plugin_registration",
            "CustomOp.register_oot",
            "register_oot",
            "stub_plugin_cpu",
        ),
        (
            "plugin_registration",
            "ModelRegistry.register_model",
            "model_registry",
            "stub_plugin_cpu",
        ),
        (
            "plugin_registration",
            "vllm.plugins.load_general_plugins",
            "load_general_plugins",
            "stub_plugin_cpu",
        ),
    ],
)
def test_non_ppu_repro_hard_gate_real_stub_plugins_all_five_kinds(
    tmp_path, stage, target_signature, expected_kind, expected_mode
):
    """Execute real inline stub plugin / CPU checks on a 2-commit git repo for all 5 kinds."""
    repo = tmp_path / "mock-vllm"
    shas = _make_mock_vllm_git_repo(repo)
    repro_out = tmp_path / f"repro-{expected_kind}.json"

    # Real subprocess probe checking out $CANDIDATE_SHA and executing materialized
    # $STUB_SITE_DIR + $STUB_CHECK_SCRIPT with zero vllm-sail and zero PPU hardware.
    probe_cmd = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'PYTHONPATH="$STUB_SITE_DIR:$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )
    rc = post_merge.main(
        [
            "verify-non-ppu-repro",
            "--vllm-repo",
            str(repo),
            "--last-good-sha",
            shas["good"],
            "--first-failing-sha",
            shas[expected_kind],
            "--stage",
            stage,
            "--target-signature",
            target_signature,
            "--probe-cmd",
            probe_cmd,
            "--output-json",
            str(repro_out),
        ]
    )
    assert rc == 0
    repro_data = json.loads(repro_out.read_text(encoding="utf-8"))
    post_merge.assert_zero_sensitive_leaks(repro_out.read_text(encoding="utf-8"))
    expected_phase = (
        "build_import" if expected_mode == "cpu_empty_build" else "stub_check"
    )
    expected_source = (
        "upstream_import_check"
        if expected_mode == "cpu_empty_build"
        else "stub_plugin_check"
    )
    expected_repro = {
        "failure_source": expected_source,
        "first_failing_check_executed": True,
        "first_failing_exit_code": 1,
        "first_failing_phase": expected_phase,
        "first_failing_sha": shas[expected_kind],
        "infra_error": False,
        "last_good_check_executed": True,
        "last_good_exit_code": 0,
        "last_good_phase": expected_phase,
        "last_good_sha": shas["good"],
        "mode": expected_mode,
        "repro_kind": expected_kind,
        "verified": True,
    }
    if expected_mode == "cpu_empty_build":
        expected_repro["observed_signature"] = "VLLM_TARGET_DEVICE=empty"
    assert repro_data == expected_repro

    # Feed verified receipt + commit author metadata into `publish` CLI end-to-end.
    commit_json = tmp_path / f"commit-{expected_kind}.json"
    commit_json.write_text(
        json.dumps(
            {
                "sha": shas[expected_kind],
                "author": {"login": "kevin-luu", "type": "User"},
            }
        ),
        encoding="utf-8",
    )
    pub_dir = tmp_path / f"pub-{expected_kind}"
    summary_path = tmp_path / f"summary-{expected_kind}.md"
    pub_rc = post_merge.main(
        [
            "publish",
            "--gear",
            "probe",
            "--status",
            "fail",
            "--sail-sha",
            _sha(901),
            "--upstream-sha",
            shas[expected_kind],
            "--cpu-status",
            "fail",
            "--ppu-status",
            "skipped",
            "--ppu-skip-reason",
            "cpu_fast_check_failed",
            "--first-failing-sha",
            shas[expected_kind],
            "--last-good-sha",
            shas["good"],
            "--failure-stage",
            stage,
            "--target-signature",
            target_signature,
            "--public-plugin-break",
            "--non-ppu-repro-json",
            str(repro_out),
            "--upstream-commit-json",
            str(commit_json),
            "--public-dir",
            str(pub_dir),
            "--step-summary",
            str(summary_path),
        ]
    )
    assert pub_rc == 0
    summary_json = json.loads(
        (pub_dir / "public-summary.json").read_text(encoding="utf-8")
    )
    issue_json = json.loads(
        (pub_dir / "issue-payload.json").read_text(encoding="utf-8")
    )
    assert summary_json["triage"]["classification"] == "plugin_interface_break"
    assert summary_json["triage"]["non_ppu_repro_verified"] is True
    assert summary_json["triage"]["non_ppu_repro_mode"] == expected_mode
    assert summary_json["triage"]["repro_kind"] == expected_kind
    assert summary_json["bisect"]["first_failing_author"] == "kevin-luu"
    assert issue_json["should_file_sail_issue"] is False
    assert issue_json["should_file_upstream_issue"] is False
    assert issue_json["first_failing_author"] == "kevin-luu"

    repro_md = issue_json["upstream_reproducer_markdown"]
    assert "````markdown\n" in repro_md
    assert "(cc @kevin-luu)" in repro_md
    assert f"(`{expected_kind}`)" in repro_md
    # Extract the embedded bash script and verify syntax with `bash -n`.
    bash_block = repro_md.split("```bash\n", 1)[1].split("\n```", 1)[0]
    script_file = tmp_path / f"repro-{expected_kind}.sh"
    script_file.write_text(bash_block + "\n", encoding="utf-8")
    syntax = subprocess.run(
        ["bash", "-n", str(script_file)], capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    post_merge.assert_zero_sensitive_leaks(repro_md)
    post_merge.assert_zero_sensitive_leaks(summary_path.read_text(encoding="utf-8"))


def test_non_ppu_repro_hard_gate_negative_controls_ppu_only_and_tampered_evidence(
    tmp_path,
):
    repo = tmp_path / "mock-vllm-neg"
    shas = _make_mock_vllm_git_repo(repo)
    probe_cmd = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'PYTHONPATH="$STUB_SITE_DIR:$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )

    # 1. Real negative control A: both last_good and candidate exit 0 -> verified=False!
    no_repro = post_merge.verify_non_ppu_reproducer(
        vllm_repo=repo,
        last_good_sha=shas["good"],
        first_failing_sha=shas["good_2"],
        stage="plugin_registration",
        target_signature="vllm.plugins.load_general_plugins",
        probe_cmd=probe_cmd,
    )
    assert no_repro.last_good_exit_code == 0
    assert no_repro.first_failing_exit_code == 0
    assert no_repro.verified is False

    # 2. Real negative control B: last_good itself fails (control broken) -> verified=False!
    broken_ctrl = post_merge.verify_non_ppu_reproducer(
        vllm_repo=repo,
        last_good_sha=shas["register_oot"],
        first_failing_sha=shas["model_registry"],
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
        probe_cmd=probe_cmd,
    )
    assert broken_ctrl.last_good_exit_code != 0
    assert broken_ctrl.verified is False

    # 3. Every PPU hardware stage (ppu_probe_smoke / rc_*) WITHOUT CPU/stub repro MUST
    #    route to vllm-sail internal tracking with zero upstream payload!
    for ppu_stage in sorted(post_merge.PPU_HARDWARE_STAGES):
        out_dir = tmp_path / f"ppu-only-{ppu_stage}"
        res = post_merge.publish_sanitized_channels(
            gear="probe" if ppu_stage == "ppu_probe_smoke" else "rc_qualify",
            status="fail",
            sail_sha=_sha(910),
            upstream_sha=shas["register_oot"],
            verified_pin=VERIFIED_PIN,
            cpu_status="pass",
            ppu_status="fail",
            first_failing_sha=shas["register_oot"],
            last_good_sha=shas["good"],
            outcome=post_merge.ProbeOutcome(
                status="fail",
                stage=ppu_stage,
                target_signature="vllm.plugins.load_general_plugins",
                is_public_plugin_interface=True,
                non_ppu_repro=None,
            ),
            first_failing_author="kevin-luu",
            public_dir=out_dir,
        )
        assert (
            res["public_summary"]["triage"]["classification"]
            == "sail_internal_patch_drift"
        )
        assert res["public_summary"]["triage"]["non_ppu_repro_verified"] is False
        assert res["public_summary"]["bisect"]["first_failing_author"] is None
        assert res["issue_payload"]["should_file_sail_issue"] is True
        assert res["issue_payload"]["should_file_upstream_issue"] is False
        assert res["issue_payload"]["upstream_reproducer_markdown"] == ""
        assert "@kevin-luu" not in res["markdown"]

    # 4. Every SAIL-internal stage blocks plugin_interface_break EVEN WITH valid repro evidence!
    valid_stub = post_merge.NonPpuReproEvidence(
        mode="stub_plugin_cpu",
        repro_kind="load_general_plugins",
        last_good_sha=shas["good"],
        first_failing_sha=shas["load_general_plugins"],
        last_good_exit_code=0,
        first_failing_exit_code=1,
        verified=True,
        last_good_phase="stub_check",
        first_failing_phase="stub_check",
        last_good_check_executed=True,
        first_failing_check_executed=True,
        infra_error=False,
        failure_source="stub_plugin_check",
    )
    for internal_stage in sorted(post_merge.SAIL_INTERNAL_ONLY_STAGES):
        triage = post_merge.classify_failure(
            post_merge.ProbeOutcome(
                status="fail",
                stage=internal_stage,
                target_signature="vllm.plugins.load_general_plugins",
                is_public_plugin_interface=True,
                non_ppu_repro=valid_stub,
            ),
            last_good_sha=shas["good"],
            first_failing_sha=shas["load_general_plugins"],
            require_shas=True,
        )
        assert triage["classification"] == "sail_internal_patch_drift"
        assert triage["non_ppu_repro_verified"] is False

    # 5. Tampered / bool-trapped / stale / mismatched-stage evidence MUST fail closed!
    bad_evidences = [
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["load_general_plugins"],
            1,
            1,
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["load_general_plugins"],
            0,
            0,
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["load_general_plugins"],
            False,  # type: ignore[arg-type]
            True,  # type: ignore[arg-type]
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["good"],
            0,
            1,
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["register_oot"],
            0,
            1,
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),  # mismatched first_failing_sha
        post_merge.NonPpuReproEvidence(
            "cpu_empty_build",
            "cpu_empty_build",
            shas["good"],
            shas["load_general_plugins"],
            0,
            1,
            True,
            "build_import",
            "build_install",
            True,
            False,
            False,
            "upstream_build_source",
        ),  # wrong mode for plugin_registration
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["load_general_plugins"],
            0,
            1,
            False,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
    ]
    for idx, ev in enumerate(bad_evidences):
        triage = post_merge.classify_failure(
            post_merge.ProbeOutcome(
                status="fail",
                stage="plugin_registration",
                target_signature="vllm.plugins.load_general_plugins",
                is_public_plugin_interface=True,
                non_ppu_repro=ev,
            ),
            last_good_sha=shas["good"],
            first_failing_sha=shas["load_general_plugins"],
            require_shas=True,
        )
        assert triage["classification"] == "sail_internal_patch_drift", (
            f"case {idx} leaked"
        )
        assert triage["non_ppu_repro_verified"] is False

    # 6. Forged JSON (`verified: true` with `first_failing_exit_code: 0`) & corrupted JSON on `publish` CLI
    forged_json = tmp_path / "forged-repro.json"
    forged_json.write_text(
        json.dumps(
            {
                "mode": "stub_plugin_cpu",
                "repro_kind": "load_general_plugins",
                "last_good_sha": shas["good"],
                "first_failing_sha": shas["load_general_plugins"],
                "last_good_exit_code": 0,
                "first_failing_exit_code": 0,
                "last_good_phase": "stub_check",
                "first_failing_phase": "stub_check",
                "last_good_check_executed": True,
                "first_failing_check_executed": True,
                "infra_error": False,
                "failure_source": "stub_plugin_check",
                "verified": True,
            }
        ),
        encoding="utf-8",
    )
    corrupt_json = tmp_path / "corrupt-repro.json"
    corrupt_json.write_text("{not valid json", encoding="utf-8")
    for label, path in (("forged", forged_json), ("corrupt", corrupt_json)):
        pub_dir = tmp_path / f"pub-{label}"
        assert (
            post_merge.main(
                [
                    "publish",
                    "--gear",
                    "probe",
                    "--status",
                    "fail",
                    "--sail-sha",
                    _sha(911),
                    "--upstream-sha",
                    shas["load_general_plugins"],
                    "--first-failing-sha",
                    shas["load_general_plugins"],
                    "--last-good-sha",
                    shas["good"],
                    "--failure-stage",
                    "plugin_registration",
                    "--target-signature",
                    "vllm.plugins.load_general_plugins",
                    "--public-plugin-break",
                    "--non-ppu-repro-json",
                    str(path),
                    "--public-dir",
                    str(pub_dir),
                ]
            )
            == 0
        )
        pub_sum = json.loads(
            (pub_dir / "public-summary.json").read_text(encoding="utf-8")
        )
        pub_iss = json.loads(
            (pub_dir / "issue-payload.json").read_text(encoding="utf-8")
        )
        assert pub_sum["triage"]["classification"] == "sail_internal_patch_drift"
        assert pub_iss["should_file_sail_issue"] is True
        assert pub_iss["upstream_reproducer_markdown"] == ""


def test_upstream_reproducer_author_resolution_fenced_template_and_zero_bare_ping(
    tmp_path, monkeypatch
):
    bad_sha = _sha(920)
    good_sha = _sha(919)
    sail_sha = _sha(921)

    # 1. Handle sanitization & commit metadata parsing matrix.
    assert post_merge.sanitize_github_login("youkaichao") == "youkaichao"
    assert post_merge.sanitize_github_login("kevin-luu") == "kevin-luu"
    assert post_merge.sanitize_github_login("A1") == "A1"
    for invalid in (
        "",
        None,
        "@alice",
        "-leading",
        "trailing-",
        "double--hyphen",
        "dependabot[bot]",
        "vllm-project/maintainers",
        "alice\ncc @bob",
        "a" * 40,
        "wb-998877",
        "step-5-preview",
    ):
        assert post_merge.sanitize_github_login(invalid) == "", f"accepted {invalid!r}"

    # Valid user commit metadata
    assert (
        post_merge.parse_commit_author_login(
            {"sha": bad_sha, "author": {"login": "youkaichao", "type": "User"}},
            expected_sha=bad_sha,
        )
        == "youkaichao"
    )
    # Bot user (even with noreply email) degrades to ""
    assert (
        post_merge.parse_commit_author_login(
            {
                "sha": bad_sha,
                "author": {"login": "dependabot", "type": "Bot"},
                "commit": {
                    "author": {"email": "49699333+dependabot@users.noreply.github.com"}
                },
            },
            expected_sha=bad_sha,
        )
        == ""
    )
    # Null author + GitHub noreply email resolves login cleanly
    assert (
        post_merge.parse_commit_author_login(
            {
                "sha": bad_sha,
                "author": None,
                "commit": {
                    "author": {"email": "987654+kevin-luu@users.noreply.github.com"}
                },
            },
            expected_sha=bad_sha,
        )
        == "kevin-luu"
    )
    # Null author + corporate/private email degrades to "" (zero email leak)
    assert (
        post_merge.parse_commit_author_login(
            {
                "sha": bad_sha,
                "author": None,
                "commit": {"author": {"email": "dev@alibaba-inc.com"}},
            },
            expected_sha=bad_sha,
        )
        == ""
    )
    # Mismatched commit SHA degrades to ""
    assert (
        post_merge.parse_commit_author_login(
            {"sha": good_sha, "author": {"login": "youkaichao", "type": "User"}},
            expected_sha=bad_sha,
        )
        == ""
    )

    # 2. Remote lookup helper (mocked urlopen): valid 200 vs network failure + repo allowlist.
    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self, _n: int = -1) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        post_merge.urllib.request,
        "urlopen",
        lambda req, timeout=10: _FakeResp(
            json.dumps(
                {"sha": bad_sha, "author": {"login": "youkaichao", "type": "User"}}
            ).encode("utf-8")
        ),
    )
    assert (
        post_merge.resolve_upstream_commit_author(bad_sha, fetch_remote=True)
        == "youkaichao"
    )
    # Non-allowlisted repo is rejected fail-closed.
    assert (
        post_merge.resolve_upstream_commit_author(
            bad_sha, repo="internal/repo", fetch_remote=True
        )
        == ""
    )

    def _net_fail(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(post_merge.urllib.request, "urlopen", _net_fail)
    assert post_merge.resolve_upstream_commit_author(bad_sha, fetch_remote=True) == ""

    # 3. Fenced markdown template invariant: `@author` ONLY inside ````markdown fence, ZERO bare pings!
    repro_ev = post_merge.NonPpuReproEvidence(
        mode="stub_plugin_cpu",
        repro_kind="platform_plugin",
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        last_good_exit_code=0,
        first_failing_exit_code=1,
        verified=True,
        last_good_phase="stub_check",
        first_failing_phase="stub_check",
        last_good_check_executed=True,
        first_failing_check_executed=True,
        infra_error=False,
        failure_source="stub_plugin_check",
    )
    with_author = post_merge.render_upstream_reproducer(
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        sail_sha=sail_sha,
        stage="plugin_registration",
        target_signature="vllm.platforms.Platform",
        repro_kind="platform_plugin",
        first_failing_author="youkaichao",
        non_ppu_repro=repro_ev,
        run_url="https://github.com/vllm-project/vllm-sail/actions/runs/55555",
    )
    parts = with_author.split("````markdown\n", 1)
    assert len(parts) == 2
    before_fence = parts[0]
    inside_and_after = parts[1].split("\n````", 1)
    assert len(inside_and_after) == 2
    inside_fence, after_fence = inside_and_after
    assert "@" not in before_fence
    assert "@" not in after_fence
    assert f"`{bad_sha}` (cc @youkaichao)" in inside_fence
    assert "vllm.platform_plugins" in inside_fence
    assert "StubPlatform" in inside_fence
    assert f"`{good_sha}` -> `0`; `{bad_sha}` -> `1`" in inside_fence

    # Missing/sensitive author degrades cleanly to SHA-only with zero `@` mentions.
    without_author = post_merge.render_upstream_reproducer(
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        sail_sha=sail_sha,
        stage="plugin_registration",
        target_signature="ModelRegistry.register_model",
        repro_kind="model_registry",
        first_failing_author="wb-998877",
        non_ppu_repro=None,
    )
    assert "@" not in without_author
    assert bad_sha in without_author
    assert "StubOOTForCausalLM" in without_author
    post_merge.assert_zero_sensitive_leaks(without_author)

    # Verify full GITHUB_STEP_SUMMARY output also has zero bare pings outside fenced block!
    summary_file = tmp_path / "step_summary.md"
    post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=sail_sha,
        upstream_sha=bad_sha,
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="plugin_registration",
            target_signature="vllm.platforms.Platform",
            is_public_plugin_interface=True,
            non_ppu_repro=repro_ev,
        ),
        first_failing_author="youkaichao",
        public_dir=tmp_path / "pub-fenced",
        step_summary_path=summary_file,
    )
    full_md = summary_file.read_text(encoding="utf-8")
    md_before, md_rest = full_md.split("````markdown\n", 1)
    md_inside, md_after = md_rest.split("\n````", 1)
    assert "@" not in md_before
    assert "@" not in md_after
    assert "@youkaichao" in md_inside


def test_verify_non_ppu_reproducer_builtin_runner_workflow_and_fingerprint_compat(
    tmp_path, monkeypatch
):
    # 1. Built-in runner path (`probe_cmd=""`) uses git worktree, VLLM_TARGET_DEVICE=empty,
    #    VLLM_PLUGINS=stub, cleans up worktrees, and fails closed on TimeoutExpired.
    good_sha = _sha(930)
    bad_sha = _sha(931)
    calls: list[tuple[list[str], dict[str, str]]] = []
    removed_worktrees: list[str] = []

    def _fake_run(cmd, **kwargs):
        cmd_list = [str(c) for c in cmd]
        calls.append((cmd_list, dict(kwargs.get("env") or {})))
        if (
            cmd_list[:4] == ["git", "-C", str(tmp_path), "worktree"]
            and cmd_list[4] == "add"
        ):
            wt_path = Path(cmd_list[7])
            wt_path.mkdir(parents=True, exist_ok=True)
            (wt_path / "tools").mkdir(exist_ok=True)
            (wt_path / "tools" / "use_existing_torch.py").write_text(
                "", encoding="utf-8"
            )
            return subprocess.CompletedProcess(cmd_list, 0, "", "")
        if (
            cmd_list[:4] == ["git", "-C", str(tmp_path), "worktree"]
            and cmd_list[4] == "remove"
        ):
            removed_worktrees.append(cmd_list[6])
            return subprocess.CompletedProcess(cmd_list, 0, "", "")
        if "repro_check.py" in " ".join(cmd_list):
            sha = (kwargs.get("env") or {}).get("CANDIDATE_SHA")
            return subprocess.CompletedProcess(
                cmd_list, 0 if sha == good_sha else 1, "", ""
            )
        return subprocess.CompletedProcess(cmd_list, 0, "", "")

    monkeypatch.setattr(post_merge.subprocess, "run", _fake_run)
    ev = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
    )
    assert ev.verified is True
    assert ev.mode == "stub_plugin_cpu"
    assert ev.repro_kind == "register_oot"
    assert ev.last_good_exit_code == 0
    assert ev.first_failing_exit_code == 1
    assert len(removed_worktrees) == 2
    pip_calls = [c for c in calls if "pip" in c[0] and "install" in c[0]]
    assert len(pip_calls) == 2
    for _, env_used in pip_calls:
        assert env_used["VLLM_TARGET_DEVICE"] == "empty"
        assert env_used["CUDA_VISIBLE_DEVICES"] == ""
        assert env_used["VLLM_PLUGINS"] == "stub"

    # Timeout during first_failing_sha -> verified=False (never false-positive regression!)
    def _timeout_on_bad(cmd, **kwargs):
        cmd_list = [str(c) for c in cmd]
        if (
            "repro_check.py" in " ".join(cmd_list)
            and (kwargs.get("env") or {}).get("CANDIDATE_SHA") == bad_sha
        ):
            raise subprocess.TimeoutExpired(cmd_list, 10)
        return _fake_run(cmd, **kwargs)

    monkeypatch.setattr(post_merge.subprocess, "run", _timeout_on_bad)
    ev_to = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
    )
    assert ev_to.verified is False
    assert ev_to.first_failing_exit_code == 124

    # 2. Verify upstream-canary.yaml wires verify-non-ppu-repro, --non-ppu-repro-json,
    #    --fetch-upstream-author, and explicitly forbids cross-repo upstream issue creation.
    wf_raw = (REPO_ROOT / ".github/workflows/upstream-canary.yaml").read_text(
        encoding="utf-8"
    )
    assert "verify-non-ppu-repro" in wf_raw
    assert "/tmp/post-merge-cpu/non-ppu-repro.json" in wf_raw
    assert "--non-ppu-repro-json /tmp/post-merge-meta/non-ppu-repro.json" in wf_raw
    assert "--fetch-upstream-author" in wf_raw
    assert "payload.should_file_upstream_issue" in wf_raw

    # 3. Verify ppu_native_fingerprint_compat.json matches current _ppu-build.yaml /
    #    build_ppu_wheels.sh / ppu_wheel_manifest.py after merging main.
    compat = json.loads(
        (REPO_ROOT / "scripts/ci/ppu_native_fingerprint_compat.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(compat) == set(wheel_manifest.FINGERPRINT_COMPAT_PATHS)
    for rel_path, entry in compat.items():
        actual_sha = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert entry["current"] == actual_sha, (
            f"ppu_native_fingerprint_compat.json stale for {rel_path}"
        )


def test_review_r1_p1_install_infra_and_unexecuted_stub_cannot_verify_or_classify_upstream(
    tmp_path, monkeypatch
):
    """Exact Round-1 P1 counterexamples + normal controls across producer -> evidence -> consumer."""
    good_sha = _sha(940)
    bad_sha = _sha(941)
    sail_sha = _sha(942)

    def _make_fake_run(
        *,
        bad_pip_rc: int = 0,
        bad_pip_stderr: str = "",
        bad_check_rc: int = 1,
        bad_check_stderr: str = "",
    ):
        counters = {"bad_stub_executions": 0}

        def _fake_run(cmd, **kwargs):
            cmd_list = [str(c) for c in cmd]
            env = kwargs.get("env") or {}
            sha = env.get("CANDIDATE_SHA")
            if (
                cmd_list[:4] == ["git", "-C", str(tmp_path), "worktree"]
                and cmd_list[4] == "add"
            ):
                wt_path = Path(cmd_list[7])
                wt_path.mkdir(parents=True, exist_ok=True)
                return subprocess.CompletedProcess(cmd_list, 0, "", "")
            if (
                cmd_list[:4] == ["git", "-C", str(tmp_path), "worktree"]
                and cmd_list[4] == "remove"
            ):
                return subprocess.CompletedProcess(cmd_list, 0, "", "")
            if "pip" in cmd_list and "install" in cmd_list:
                if sha == bad_sha:
                    return subprocess.CompletedProcess(
                        cmd_list, bad_pip_rc, "", bad_pip_stderr
                    )
                return subprocess.CompletedProcess(cmd_list, 0, "", "")
            if "repro_check.py" in " ".join(cmd_list):
                if sha == bad_sha:
                    counters["bad_stub_executions"] += 1
                    return subprocess.CompletedProcess(
                        cmd_list, bad_check_rc, "", bad_check_stderr
                    )
                return subprocess.CompletedProcess(cmd_list, 0, "", "")
            return subprocess.CompletedProcess(cmd_list, 0, "", "")

        return _fake_run, counters

    # Counterexample 1 (exact Round-1 P1): stub_plugin_cpu where bad pip hits ENOSPC;
    # stub never executes -> MUST be unverified, infra_error=True, sail_internal_patch_drift, no upstream draft!
    fake_enospc, cnt1 = _make_fake_run(
        bad_pip_rc=1,
        bad_pip_stderr="ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device",
    )
    monkeypatch.setattr(post_merge.subprocess, "run", fake_enospc)
    ev_enospc = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="vllm.plugins.load_general_plugins",
    )
    assert cnt1["bad_stub_executions"] == 0
    assert ev_enospc.verified is False
    assert ev_enospc.infra_error is True
    assert ev_enospc.first_failing_check_executed is False
    assert ev_enospc.first_failing_phase == "infra_error"
    pub_enospc = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=sail_sha,
        upstream_sha=bad_sha,
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="plugin_registration",
            target_signature="vllm.plugins.load_general_plugins",
            is_public_plugin_interface=True,
            non_ppu_repro=ev_enospc,
        ),
        first_failing_author="kevin-luu",
        public_dir=tmp_path / "pub-p1-enospc",
    )
    assert (
        pub_enospc["public_summary"]["triage"]["classification"]
        == "sail_internal_patch_drift"
    )
    assert pub_enospc["issue_payload"]["upstream_reproducer_markdown"] == ""
    assert pub_enospc["issue_payload"]["should_file_sail_issue"] is True

    # Counterexample 2: stub_plugin_cpu where bad pip fails with a non-infra build error
    # (stub check STILL never executed) -> MUST remain verified=False!
    fake_bad_pip, cnt2 = _make_fake_run(
        bad_pip_rc=1,
        bad_pip_stderr="SyntaxError: invalid syntax in setup.py",
    )
    monkeypatch.setattr(post_merge.subprocess, "run", fake_bad_pip)
    ev_no_stub = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
    )
    assert cnt2["bad_stub_executions"] == 0
    assert ev_no_stub.verified is False
    assert ev_no_stub.first_failing_phase == "install"
    assert ev_no_stub.first_failing_check_executed is False

    # Counterexample 3: cpu_empty_build where bad pip fails with ENOSPC / network / SIGKILL /
    # Errno 122 (EDQUOT) / Errno 5 (EIO) / Errno 116 (ESTALE) / pip host RuntimeError /
    # 0-line metadata wrapper / opaque error -> infra_error, verified=False!
    for err_rc, err_msg in (
        (1, "OSError: [Errno 28] No space left on device"),
        (1, "OSError: [Errno 122] Disk quota exceeded"),
        (1, "OSError: [Errno 5] Input/output error"),
        (
            1,
            "Building wheel for vllm (pyproject.toml) ... error\n"
            "OSError: [Errno 122] Disk quota exceeded",
        ),
        (1, "OSError: [Errno 116] Stale file handle"),
        (
            1,
            "ERROR: Could not install packages due to an OSError: [Errno 2] host temp missing",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3/site-packages/pip/_vendor/rich/live.py", line 113, in start\n'
            '  File "/usr/lib/python3.12/threading.py", line 992, in start\n'
            "RuntimeError: can't start new thread",
        ),
        (
            1,
            "× Preparing metadata (pyproject.toml) did not run successfully.\n"
            "│ exit code: 1\n"
            "╰─> [0 lines of output]\n"
            "note: This error originates from a subprocess, and is likely not a problem with pip.\n"
            "error: metadata-generation-failed",
        ),
        (1, "RuntimeError: runner worker pool exhausted"),
        (
            1,
            "RuntimeError: runner worker pool exhausted\n"
            "ERROR: Could not build wheels for vllm, which is required to install pyproject.toml-based projects",
        ),
        (
            1,
            "RuntimeError: runner crashed (env VLLM_TARGET_DEVICE=empty in setup.py)\n"
            "ERROR: Could not build wheels for vllm, which is required to install pyproject.toml-based projects",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 20, in <module>\n'
            "    t.start()\n"
            '  File "/usr/lib/python3.11/threading.py", line 975, in start\n'
            "    _start_new_thread()\n"
            "RuntimeError: host thread pool exhausted",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 30, in run\n'
            '    subprocess.check_call(["cmake", ".."])\n'
            '  File "/usr/lib/python3.11/subprocess.py", line 1950, in _execute_child\n'
            "    raise child_exception_type(errno_num, err_msg, err_filename)\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'cmake'",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.11/urllib/request.py", line 100, in urlopen\n'
            '    raise ValueError("bad proxy config")\n'
            "ValueError: bad proxy config\n\n"
            "The above exception was the direct cause of the following exception:\n\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 50, in <module>\n'
            '    raise RuntimeError("wheel fetch failed")\n'
            "RuntimeError: wheel fetch failed",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 1, in <module>\n'
            "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported\n"
            "Exception: runner worker pool exhausted",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "<string>", line 1, in <module>\n'
            "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 374, in prepare_metadata_for_build_wheel\n'
            "    self.run_setup()\n"
            "RuntimeError: corrupted setuptools installation",
        ),
        (1, "ConnectionError: Temporary failure in name resolution"),
        (
            1,
            "",
        ),  # opaque pip failure without any upstream build-backend/source error signal
        (137, "Killed"),
    ):
        fake_build_infra, _ = _make_fake_run(bad_pip_rc=err_rc, bad_pip_stderr=err_msg)
        monkeypatch.setattr(post_merge.subprocess, "run", fake_build_infra)
        ev_build_infra = post_merge.verify_non_ppu_reproducer(
            vllm_repo=tmp_path,
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
        )
        assert ev_build_infra.verified is False, f"leaked infra stderr={err_msg!r}"
        assert ev_build_infra.infra_error is True, f"not infra stderr={err_msg!r}"
        assert ev_build_infra.first_failing_phase == "infra_error"
        assert ev_build_infra.failure_source == "infra_or_unknown"
        assert (
            post_merge.classify_failure(
                post_merge.ProbeOutcome(
                    status="fail",
                    stage="wheel_build",
                    target_signature="VLLM_TARGET_DEVICE=empty",
                    is_public_plugin_interface=True,
                    non_ppu_repro=ev_build_infra,
                ),
                last_good_sha=good_sha,
                first_failing_sha=bad_sha,
                require_shas=True,
            )["classification"]
            == "sail_internal_patch_drift"
        )

    # Normal controls 1–4: cpu_empty_build where bad pip fails with genuine structurally
    # attributed upstream build/packaging errors (setup.py RuntimeError, setup.py missing
    # requirements/common.txt [Errno 2], PEP 517 _in_process.py + setup.py + pip wheel summary,
    # and setup.py -> CMake Error at CMakeLists.txt) -> verified=True!
    for build_err_msg in (
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 18, in <module>\n'
        '    raise RuntimeError("VLLM_TARGET_DEVICE=empty unsupported in setup.py")\n'
        "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported in setup.py",
        "Preparing metadata (pyproject.toml) ... error\n"
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 42, in get_requirements\n'
        '    with open("requirements/common.txt") as f:\n'
        "FileNotFoundError: [Errno 2] No such file or directory: 'requirements/common.txt'",
        "  error: subprocess-exited-with-error\n"
        "  × Preparing metadata (pyproject.toml) did not run successfully.\n"
        "  │ exit code: 1\n"
        "  ╰─> [10 lines of output]\n"
        "      Traceback (most recent call last):\n"
        '        File "/opt/py/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>\n'
        "          main()\n"
        '        File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 12, in <module>\n'
        '          raise RuntimeError("VLLM_TARGET_DEVICE=empty broken")\n'
        "      RuntimeError: VLLM_TARGET_DEVICE=empty broken\n"
        "      [end of output]\n"
        "  note: This error originates from a subprocess, and is likely not a problem with pip.\n"
        "error: metadata-generation-failed\n"
        "ERROR: Could not build wheels for vllm, which is required to install pyproject.toml-based projects",
        "CMake Error at CMakeLists.txt:42 (message):\n"
        "  Unsupported target device empty\n"
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 210, in build_extensions\n'
        '    subprocess.check_call(["cmake", ".."])\n'
        '  File "/usr/lib/python3.11/subprocess.py", line 413, in check_call\n'
        "    raise CalledProcessError(retcode, cmd)\n"
        "subprocess.CalledProcessError: Command '['cmake', '..']' returned non-zero exit status 1.",
        # Real setuptools 80.9.0 build_meta.run_setup -> exec(code, locals()) `<string>` frames:
        "  error: subprocess-exited-with-error\n"
        "  × Preparing metadata (pyproject.toml) did not run successfully.\n"
        "  │ exit code: 1\n"
        "  ╰─> [15 lines of output]\n"
        "      Traceback (most recent call last):\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>\n'
        "          main()\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 374, in prepare_metadata_for_build_wheel\n'
        "          self.run_setup()\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
        "          exec(code, locals())\n"
        '        File "<string>", line 1, in <module>\n'
        "      RuntimeError: VLLM_TARGET_DEVICE=empty unsupported in setup.py\n"
        "      [end of output]\n"
        "error: metadata-generation-failed",
        "      Traceback (most recent call last):\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
        "          exec(code, locals())\n"
        '        File "<string>", line 1, in <module>\n'
        "      FileNotFoundError: [Errno 2] No such file or directory: 'requirements/common.txt'",
        "      Traceback (most recent call last):\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
        "          exec(code, locals())\n"
        '        File "<string>", line 1\n'
        "          def broken(:\n"
        "                     ^\n"
        "      SyntaxError: invalid syntax",
    ):
        fake_real_build_fail, _ = _make_fake_run(
            bad_pip_rc=1,
            bad_pip_stderr=build_err_msg,
        )
        monkeypatch.setattr(post_merge.subprocess, "run", fake_real_build_fail)
        ev_real_build = post_merge.verify_non_ppu_reproducer(
            vllm_repo=tmp_path,
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
        )
        assert ev_real_build.verified is True, (
            f"rejected genuine build err={build_err_msg!r}"
        )
        assert ev_real_build.infra_error is False
        assert ev_real_build.last_good_phase == "build_import"
        assert ev_real_build.last_good_check_executed is True
        assert ev_real_build.first_failing_phase == "build_install"
        assert ev_real_build.failure_source == "upstream_build_source"
        assert (
            post_merge.classify_failure(
                post_merge.ProbeOutcome(
                    status="fail",
                    stage="wheel_build",
                    target_signature="VLLM_TARGET_DEVICE=empty",
                    is_public_plugin_interface=True,
                    non_ppu_repro=ev_real_build,
                ),
                last_good_sha=good_sha,
                first_failing_sha=bad_sha,
                require_shas=True,
            )["classification"]
            == "plugin_interface_break"
        )

    # Counterexample 4: forged serialized JSON claiming verified=True with invalid phase,
    # invalid/mismatched failure_source, or missing required fields -> fails closed!
    forged_payloads = [
        {
            "mode": "stub_plugin_cpu",
            "repro_kind": "load_general_plugins",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "last_good_phase": "stub_check",
            "first_failing_phase": "install",
            "last_good_check_executed": True,
            "first_failing_check_executed": False,
            "infra_error": False,
            "failure_source": "stub_plugin_check",
            "verified": True,
        },
        {
            "mode": "cpu_empty_build",
            "repro_kind": "cpu_empty_build",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "last_good_phase": "build_import",
            "first_failing_phase": "infra_error",
            "last_good_check_executed": True,
            "first_failing_check_executed": False,
            "infra_error": True,
            "failure_source": "infra_or_unknown",
            "verified": True,
        },
        # Forged build_install with failure_source="infra_or_unknown" must fail closed!
        {
            "mode": "cpu_empty_build",
            "repro_kind": "cpu_empty_build",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "last_good_phase": "build_import",
            "first_failing_phase": "build_install",
            "last_good_check_executed": True,
            "first_failing_check_executed": False,
            "infra_error": False,
            "failure_source": "infra_or_unknown",
            "verified": True,
        },
        # Legacy JSON missing failure_source must also fail closed!
        {
            "mode": "cpu_empty_build",
            "repro_kind": "cpu_empty_build",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "last_good_phase": "build_import",
            "first_failing_phase": "build_install",
            "last_good_check_executed": True,
            "first_failing_check_executed": False,
            "infra_error": False,
            "verified": True,
        },
        # Legacy JSON missing execution phase / check_executed fields must also fail closed!
        {
            "mode": "stub_plugin_cpu",
            "repro_kind": "load_general_plugins",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "verified": True,
        },
    ]
    for idx, payload in enumerate(forged_payloads):
        parsed = post_merge.parse_non_ppu_repro_evidence(payload)
        assert parsed is None or parsed.verified is False, f"forged case {idx} leaked"
        triage = post_merge.classify_failure(
            post_merge.ProbeOutcome(
                status="fail",
                stage="plugin_registration"
                if payload["mode"] == "stub_plugin_cpu"
                else "wheel_build",
                target_signature="vllm.plugins.load_general_plugins"
                if payload["mode"] == "stub_plugin_cpu"
                else "VLLM_TARGET_DEVICE=empty",
                is_public_plugin_interface=True,
                non_ppu_repro=parsed,
            ),
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            require_shas=True,
        )
        assert triage["classification"] == "sail_internal_patch_drift", (
            f"forged case {idx} classified upstream"
        )

    # Counterexample 5: real subprocess `probe_cmd` in stub_plugin_cpu mode that fails
    # before invoking `$STUB_CHECK_SCRIPT` -> mechanical marker proves check did NOT execute!
    monkeypatch.undo()
    ev_probe_no_exec = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="vllm.plugins.load_general_plugins",
        probe_cmd=(
            f'if [ "$CANDIDATE_SHA" = "{good_sha}" ]; then '
            "python3 -c \"import pathlib, os; pathlib.Path(os.environ['STUB_CHECK_SCRIPT']).exists()\" && exit 0; "
            "else echo 'pip failed before stub' >&2; exit 1; fi"
        ),
    )
    assert ev_probe_no_exec.verified is False
    assert ev_probe_no_exec.last_good_check_executed is False
    assert ev_probe_no_exec.first_failing_check_executed is False

    # Counterexample 6: real subprocess `probe_cmd` in cpu_empty_build mode where
    # last_good executes check_script cleanly, while first_failing fails inside pip/host
    # (rich/live.py thread RuntimeError, zero-line metadata wrapper, or BackendUnavailable)
    # -> producer -> JSON -> gate -> publisher must all fail closed (zero upstream draft)!
    good_vllm_dir = tmp_path / "fake_good_vllm"
    (good_vllm_dir / "vllm").mkdir(parents=True, exist_ok=True)
    (good_vllm_dir / "vllm" / "__init__.py").write_text(
        "__version__ = '0.0.1'\n", encoding="utf-8"
    )
    probe_infra_snippets = [
        (
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.11/site-packages/pip/_vendor/rich/live.py", line 108, in start\n'
            '  File "/usr/lib/python3.11/threading.py", line 975, in start\n'
            "RuntimeError: can't start new thread"
        ),
        (
            "  × Preparing metadata (pyproject.toml) did not run successfully.\n"
            "  │ exit code: 1\n"
            "  ╰─> [0 lines of output]\n"
            "error: metadata-generation-failed"
        ),
        (
            "ERROR: Exception:\n"
            '  File "/opt/py/lib/python3.13/site-packages/pip/_internal/operations/build/metadata.py", line 34\n'
            '  File "/opt/py/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 402\n'
            "pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'"
        ),
        (
            "RuntimeError: runner worker pool exhausted\n"
            "ERROR: Could not build wheels for vllm, which is required to install pyproject.toml-based projects"
        ),
        (
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 1, in <module>\n'
            "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported\n"
            "Exception: runner worker pool exhausted"
        ),
        (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 1, in <module>\n'
            "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported"
        ),
    ]
    sail_sha = "c" * 40
    for idx, err_blob in enumerate(probe_infra_snippets):
        err_file = tmp_path / f"probe_err_{idx}.txt"
        err_file.write_text(err_blob + "\n", encoding="utf-8")
        ev_probe_infra = post_merge.verify_non_ppu_reproducer(
            vllm_repo=tmp_path,
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
            probe_cmd=(
                f'if [ "$CANDIDATE_SHA" = "{good_sha}" ]; then '
                f'PYTHONPATH="{good_vllm_dir}" python3 "$STUB_CHECK_SCRIPT"; '
                f'else cat "{err_file}" >&2; exit 1; fi'
            ),
        )
        assert ev_probe_infra.verified is False, f"probe infra case {idx} verified"
        assert ev_probe_infra.infra_error is True, f"probe infra case {idx} infra"
        assert ev_probe_infra.first_failing_phase == "infra_error"
        assert ev_probe_infra.failure_source == "infra_or_unknown"
        # Round-trip through JSON -> gate -> publisher
        roundtrip = post_merge.parse_non_ppu_repro_evidence(
            ev_probe_infra.to_public_dict()
        )
        assert roundtrip is not None and roundtrip.verified is False
        pub = post_merge.publish_sanitized_channels(
            gear="probe",
            status="fail",
            sail_sha=sail_sha,
            upstream_sha=bad_sha,
            verified_pin=good_sha,
            first_failing_sha=bad_sha,
            last_good_sha=good_sha,
            outcome=post_merge.ProbeOutcome(
                status="fail",
                stage="wheel_build",
                target_signature="VLLM_TARGET_DEVICE=empty",
                is_public_plugin_interface=True,
                non_ppu_repro=roundtrip,
            ),
            public_dir=tmp_path / f"pub_infra_{idx}",
        )
        assert (
            pub["public_summary"]["triage"]["classification"]
            == "sail_internal_patch_drift"
        )
        assert pub["issue_payload"]["upstream_reproducer_markdown"] == ""

    # Positive probe_cmd control: genuine PEP 517 _in_process.py + setup.py error -> verified!
    pep517_err_file = tmp_path / "probe_pep517_real.txt"
    pep517_err_file.write_text(
        "  × Preparing metadata (pyproject.toml) did not run successfully.\n"
        "  │ exit code: 1\n"
        "  ╰─> [8 lines of output]\n"
        "      Traceback (most recent call last):\n"
        '        File "/opt/py/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>\n'
        '        File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 10, in <module>\n'
        "      RuntimeError: VLLM_TARGET_DEVICE=empty unsupported\n"
        "      [end of output]\n"
        "error: metadata-generation-failed\n",
        encoding="utf-8",
    )
    ev_probe_real = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="wheel_build",
        target_signature="VLLM_TARGET_DEVICE=empty",
        probe_cmd=(
            f'if [ "$CANDIDATE_SHA" = "{good_sha}" ]; then '
            f'PYTHONPATH="{good_vllm_dir}" python3 "$STUB_CHECK_SCRIPT"; '
            f'else cat "{pep517_err_file}" >&2; exit 1; fi'
        ),
    )
    assert ev_probe_real.verified is True
    assert ev_probe_real.infra_error is False
    assert ev_probe_real.first_failing_phase == "build_install"
    assert ev_probe_real.failure_source == "upstream_build_source"
    roundtrip_real = post_merge.parse_non_ppu_repro_evidence(
        ev_probe_real.to_public_dict()
    )
    assert roundtrip_real is not None and roundtrip_real.verified is True
    pub_real = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=sail_sha,
        upstream_sha=bad_sha,
        verified_pin=good_sha,
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
            is_public_plugin_interface=True,
            non_ppu_repro=roundtrip_real,
        ),
        public_dir=tmp_path / "pub_real_pep517",
    )
    assert (
        pub_real["public_summary"]["triage"]["classification"]
        == "plugin_interface_break"
    )
    assert pub_real["issue_payload"]["upstream_reproducer_markdown"] != ""

    # Positive probe_cmd controls for real setuptools 80.9.0 `build_meta.run_setup` -> `<string>`
    # (RuntimeError, missing requirements/common.txt FileNotFoundError, and SyntaxError)
    for st_idx, st_stderr in enumerate(
        (
            "      Traceback (most recent call last):\n"
            '        File "/opt/homebrew/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>\n'
            '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
            "          exec(code, locals())\n"
            '        File "<string>", line 1, in <module>\n'
            "      RuntimeError: VLLM_TARGET_DEVICE=empty unsupported in setup.py\n",
            "      Traceback (most recent call last):\n"
            '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
            "          exec(code, locals())\n"
            '        File "<string>", line 1, in <module>\n'
            "      FileNotFoundError: [Errno 2] No such file or directory: 'requirements/common.txt'\n",
            "      Traceback (most recent call last):\n"
            '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
            "          exec(code, locals())\n"
            '        File "<string>", line 1\n'
            "          def broken(:\n"
            "                     ^\n"
            "      SyntaxError: invalid syntax\n",
        )
    ):
        st_file = tmp_path / f"probe_setuptools_{st_idx}.txt"
        st_file.write_text(st_stderr, encoding="utf-8")
        ev_st = post_merge.verify_non_ppu_reproducer(
            vllm_repo=tmp_path,
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
            probe_cmd=(
                f'if [ "$CANDIDATE_SHA" = "{good_sha}" ]; then '
                f'PYTHONPATH="{good_vllm_dir}" python3 "$STUB_CHECK_SCRIPT"; '
                f'else cat "{st_file}" >&2; exit 1; fi'
            ),
        )
        assert ev_st.verified is True, f"setuptools case {st_idx} rejected"
        assert ev_st.infra_error is False
        assert ev_st.first_failing_phase == "build_install"
        assert ev_st.failure_source == "upstream_build_source"
        rt_st = post_merge.parse_non_ppu_repro_evidence(ev_st.to_public_dict())
        assert rt_st is not None and rt_st.verified is True
        pub_st = post_merge.publish_sanitized_channels(
            gear="probe",
            status="fail",
            sail_sha=sail_sha,
            upstream_sha=bad_sha,
            verified_pin=good_sha,
            first_failing_sha=bad_sha,
            last_good_sha=good_sha,
            outcome=post_merge.ProbeOutcome(
                status="fail",
                stage="wheel_build",
                target_signature="VLLM_TARGET_DEVICE=empty",
                is_public_plugin_interface=True,
                non_ppu_repro=rt_st,
            ),
            public_dir=tmp_path / f"pub_setuptools_{st_idx}",
        )
        assert (
            pub_st["public_summary"]["triage"]["classification"]
            == "plugin_interface_break"
        )
        assert pub_st["issue_payload"]["upstream_reproducer_markdown"] != ""


def test_review_pr374_four_inline_comments_end_to_end(tmp_path, monkeypatch):
    """Regression coverage for PR #374 review comments (3 P1s + 1 P2)."""
    import traceback

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }

    # --- 1. [P1] Rename from runtime path (`vllm/v1/worker/`) into `examples/` must NOT skip PPU ---
    wf_raw = (REPO_ROOT / ".github/workflows/upstream-canary.yaml").read_text(
        encoding="utf-8"
    )
    assert "diff --no-renames --name-only" in wf_raw

    ren_repo = tmp_path / "rename-repo"
    ren_repo.mkdir()
    subprocess.run(["git", "-C", str(ren_repo), "init", "-q"], check=True)
    worker_file = ren_repo / "vllm" / "v1" / "worker" / "gpu_probe.py"
    worker_file.parent.mkdir(parents=True, exist_ok=True)
    worker_file.write_text("def run_worker():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(ren_repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(ren_repo), "commit", "-qm", "base"],
        check=True,
        env=git_env,
    )
    c_base = subprocess.check_output(
        ["git", "-C", str(ren_repo), "rev-parse", "HEAD"], text=True
    ).strip()

    dst_file = ren_repo / "examples" / "gpu_probe.py"
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(ren_repo),
            "mv",
            "vllm/v1/worker/gpu_probe.py",
            "examples/gpu_probe.py",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(ren_repo), "commit", "-qm", "move worker to examples"],
        check=True,
        env=git_env,
    )
    c_moved = subprocess.check_output(
        ["git", "-C", str(ren_repo), "rev-parse", "HEAD"], text=True
    ).strip()

    no_rename_lines = subprocess.check_output(
        [
            "git",
            "-C",
            str(ren_repo),
            "diff",
            "--no-renames",
            "--name-only",
            c_base,
            c_moved,
        ],
        text=True,
    ).splitlines()
    assert "vllm/v1/worker/gpu_probe.py" in no_rename_lines
    assert "examples/gpu_probe.py" in no_rename_lines
    assert post_merge.should_skip_ppu_for_paths(no_rename_lines) is False

    # Defense-in-depth: even `--name-status` (`R100\tvllm/v1/worker/...\texamples/...`) requires PPU!
    name_status_lines = subprocess.check_output(
        ["git", "-C", str(ren_repo), "diff", "--name-status", c_base, c_moved],
        text=True,
    ).splitlines()
    assert post_merge.should_skip_ppu_for_paths(name_status_lines) is False

    dec_moved = post_merge.route_three_gears(
        ls_remote_data={"main_sha": c_moved, "tags": {}},
        verified_pin=c_base,
        release_baseline_tag="v0.33.0",
        requested_gear="probe",
        cursor=None,
        changed_paths_since_ppu=no_rename_lines,
        changed_files_base_sha=c_base,
    )
    assert dec_moved["run_ppu"] is True
    assert dec_moved["ppu_skip_reason"] == "none"

    # CPU-only pass without PPU execution on a renamed runtime file must NOT qualify for daily_pin_bump!
    cur_cpu_only = post_merge.advance_cursor(
        None,
        verified_pin=c_base,
        selected_gear="probe",
        target_sha=c_moved,
        status="deferred",
        cpu_status="pass",
        ppu_status="skipped",
        ppu_skip_reason="deferred_for_sail_pr_ci",
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": c_moved, "tags": {}},
            verified_pin=c_base,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur_cpu_only,
        )["target_sha"]
        == ""
    )
    # Normal control: rename strictly between two allowlisted paths still skips PPU.
    assert (
        post_merge.should_skip_ppu_for_paths(["R100\tdocs/old.md\texamples/demo.py"])
        is True
    )

    # --- 2. [P1] Initial wheel-build failure runs source-only control-first triage & bisection ---
    assert "source-build-triage" in wf_raw
    assert "if: needs.build.result == 'success'" in wf_raw

    # 2a. Untriaged/skipped fallback (exact replay from review): must be infra_error, NOT file SAIL issue!
    untriaged_out = post_merge.evaluate_workflow_outcome(
        selected_gear="probe",
        target_sha=c_moved,
        run_cpu=True,
        run_ppu=True,
        ppu_skip_reason="none",
        build_result="failure",
        cpu_result="skipped",
        ppu_smoke_result="skipped",
    )
    untriaged_dir = tmp_path / "pub-untriaged-build"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                untriaged_out["status"],
                "--sail-sha",
                _sha(950),
                "--upstream-sha",
                c_moved,
                "--cpu-status",
                untriaged_out["cpu_status"],
                "--ppu-status",
                untriaged_out["ppu_status"],
                "--ppu-skip-reason",
                untriaged_out["ppu_skip_reason"],
                "--failure-stage",
                untriaged_out["failure_stage"],
                "--public-dir",
                str(untriaged_dir),
            ]
        )
        == 0
    )
    untriaged_sum = json.loads(
        (untriaged_dir / "public-summary.json").read_text(encoding="utf-8")
    )
    untriaged_iss = json.loads(
        (untriaged_dir / "issue-payload.json").read_text(encoding="utf-8")
    )
    assert untriaged_sum["status"] == "infra_error"
    assert untriaged_sum["triage"]["classification"] == "infra_error"
    assert untriaged_iss["should_file_sail_issue"] is False

    # Build a 4-commit upstream repo where commits[2] breaks VLLM_TARGET_DEVICE=empty setup.py
    src_repo = tmp_path / "src-bisect-repo"
    src_repo.mkdir()
    subprocess.run(["git", "-C", str(src_repo), "init", "-q"], check=True)
    (src_repo / "vllm").mkdir()
    (src_repo / "vllm" / "__init__.py").write_text(
        "__version__ = '0.1'\n", encoding="utf-8"
    )
    (src_repo / "setup.py").write_text("# ok\n", encoding="utf-8")
    src_shas: list[str] = []
    for idx in range(5):
        (src_repo / "README.md").write_text(f"step {idx}\n", encoding="utf-8")
        if idx == 3:
            (src_repo / "setup.py").write_text(
                'raise RuntimeError("VLLM_TARGET_DEVICE=empty broken")\n',
                encoding="utf-8",
            )
        subprocess.run(["git", "-C", str(src_repo), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(src_repo), "commit", "-qm", f"c{idx}"],
            check=True,
            env=git_env,
        )
        src_shas.append(
            subprocess.check_output(
                ["git", "-C", str(src_repo), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    pin_file = tmp_path / "verified.commit"
    pin_file.write_text(src_shas[0] + "\n", encoding="utf-8")

    # 2b. Control-first failure -> baseline_control_failed -> infra_error, zero issues!
    ctrl_fail_dir = tmp_path / "sbt-ctrl-fail"
    assert (
        post_merge.main(
            [
                "source-build-triage",
                "--vllm-repo",
                str(src_repo),
                "--sail-root",
                str(REPO_ROOT),
                "--target-sha",
                src_shas[4],
                "--last-green-sha",
                src_shas[0],
                "--verified-pin-file",
                str(pin_file),
                "--probe-cmd",
                "echo 'Connection reset by peer' >&2; exit 1",
                "--output-dir",
                str(ctrl_fail_dir),
            ]
        )
        == 0
    )
    pub_ctrl_fail = tmp_path / "pub-sbt-ctrl-fail"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(951),
                "--upstream-sha",
                src_shas[4],
                "--verified-pin-file",
                str(pin_file),
                "--failure-stage",
                "wheel_build",
                "--cpu-result-json",
                str(ctrl_fail_dir / "cpu-result.json"),
                "--bisect-result-json",
                str(ctrl_fail_dir / "bisect-result.json"),
                "--public-dir",
                str(pub_ctrl_fail),
            ]
        )
        == 0
    )
    cf_sum = json.loads(
        (pub_ctrl_fail / "public-summary.json").read_text(encoding="utf-8")
    )
    cf_iss = json.loads(
        (pub_ctrl_fail / "issue-payload.json").read_text(encoding="utf-8")
    )
    assert cf_sum["status"] == "infra_error"
    assert cf_sum["triage"]["classification"] == "infra_error"
    assert cf_iss["should_file_sail_issue"] is False

    # 2c. Genuine upstream VLLM_TARGET_DEVICE=empty build break at src_shas[3] -> converges & verifies!
    sbt_probe_cmd = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'"{sys.executable}" "$wt/setup.py"\n'
        f'PYTHONPATH="$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )
    real_sbt_dir = tmp_path / "sbt-converged"
    assert (
        post_merge.main(
            [
                "source-build-triage",
                "--vllm-repo",
                str(src_repo),
                "--sail-root",
                str(REPO_ROOT),
                "--target-sha",
                src_shas[4],
                "--last-green-sha",
                src_shas[0],
                "--verified-pin-file",
                str(pin_file),
                "--probe-cmd",
                sbt_probe_cmd,
                "--output-dir",
                str(real_sbt_dir),
            ]
        )
        == 0
    )
    pub_sbt_ok = tmp_path / "pub-sbt-converged"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(952),
                "--upstream-sha",
                src_shas[4],
                "--verified-pin-file",
                str(pin_file),
                "--failure-stage",
                "wheel_build",
                "--first-failing-author",
                "kevin-luu",
                "--cpu-result-json",
                str(real_sbt_dir / "cpu-result.json"),
                "--bisect-result-json",
                str(real_sbt_dir / "bisect-result.json"),
                "--non-ppu-repro-json",
                str(real_sbt_dir / "non-ppu-repro.json"),
                "--public-dir",
                str(pub_sbt_ok),
            ]
        )
        == 0
    )
    ok_sum = json.loads(
        (pub_sbt_ok / "public-summary.json").read_text(encoding="utf-8")
    )
    ok_iss = json.loads((pub_sbt_ok / "issue-payload.json").read_text(encoding="utf-8"))
    assert ok_sum["status"] == "fail"
    assert ok_sum["bisect"]["first_failing_sha"] == src_shas[3]
    assert ok_sum["bisect"]["last_good_sha"] == src_shas[2]
    assert ok_sum["bisect"]["steps"] > 0
    assert ok_sum["triage"]["classification"] == "plugin_interface_break"
    assert ok_sum["triage"]["non_ppu_repro_verified"] is True
    assert ok_iss["should_file_sail_issue"] is False
    assert "(cc @kevin-luu)" in ok_iss["upstream_reproducer_markdown"]

    # --- 3. [P1] CustomOp.register_oot against module-level `op_registry_oot` (no class attr!) ---
    inline_repo = tmp_path / "custom-op-repo"
    inline_shas = _make_mock_vllm_git_repo(inline_repo)
    custom_op_probe = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'PYTHONPATH="$STUB_SITE_DIR:$wt" "{sys.executable}" -c '
        "\"from vllm.model_executor.custom_op import CustomOp; assert not hasattr(CustomOp, 'op_registry_oot')\"\n"
        f'PYTHONPATH="$STUB_SITE_DIR:$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )
    ev_reg_oot = post_merge.verify_non_ppu_reproducer(
        vllm_repo=inline_repo,
        last_good_sha=inline_shas["good"],
        first_failing_sha=inline_shas["register_oot"],
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
        probe_cmd=custom_op_probe,
    )
    assert ev_reg_oot.verified is True
    assert ev_reg_oot.last_good_exit_code == 0
    assert ev_reg_oot.first_failing_exit_code != 0
    assert ev_reg_oot.failure_source == "stub_plugin_check"

    # --- 4. [P2] Mixed tree-formatted `ExceptionGroup` alongside valid setup.py traceback fails closed ---
    if sys.version_info >= (3, 11):
        try:
            raise ExceptionGroup(  # noqa: F821
                "runner worker pool exhausted", [Exception("runner worker failed")]
            )
        except BaseException as eg:
            eg_text = "".join(traceback.format_exception(eg)).rstrip()
    else:
        eg_text = (
            "  + Exception Group Traceback (most recent call last):\n"
            '  |   File "<stdin>", line 1, in <module>\n'
            "  | ExceptionGroup: runner worker pool exhausted (1 sub-exception)\n"
            "  +-+---------------- 1 ----------------\n"
            "    | Exception: runner worker failed\n"
            "    +------------------------------------"
        )

    mixed_stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 12, in <module>\n'
        '    raise RuntimeError("VLLM_TARGET_DEVICE=empty unsupported")\n'
        "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported\n" + eg_text
    )
    phase_eg, src_eg, infra_eg = post_merge._classify_install_failure(
        1, mixed_stderr, "", mode="cpu_empty_build"
    )
    assert (phase_eg, src_eg, infra_eg) == ("infra_error", "infra_or_unknown", True)

    good_vllm_dir = tmp_path / "good_vllm_eg"
    (good_vllm_dir / "vllm").mkdir(parents=True, exist_ok=True)
    (good_vllm_dir / "vllm" / "__init__.py").write_text(
        "__version__ = '0.1'\n", encoding="utf-8"
    )
    eg_err_file = tmp_path / "mixed_eg.txt"
    eg_err_file.write_text(mixed_stderr + "\n", encoding="utf-8")
    ev_eg = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=_sha(960),
        first_failing_sha=_sha(961),
        stage="wheel_build",
        target_signature="VLLM_TARGET_DEVICE=empty",
        probe_cmd=(
            f'if [ "$CANDIDATE_SHA" = "{_sha(960)}" ]; then '
            f'PYTHONPATH="{good_vllm_dir}" "{sys.executable}" "$STUB_CHECK_SCRIPT"; '
            f'else cat "{eg_err_file}" >&2; exit 1; fi'
        ),
    )
    assert ev_eg.verified is False
    assert ev_eg.infra_error is True
    assert ev_eg.failure_source == "infra_or_unknown"


def test_review_pr374_rereview_mandatory_rust_and_target_matched_bisect(
    tmp_path: Path,
) -> None:
    """Regression test for review #6035578476 (2 P1s: r4677055119 & r4677055123)."""
    # Lightweight python wrapper implementing `-m pip install ... --target <site> <src>`
    # without requiring external `setuptools` in bare CI/host Python environments.
    py_shim = tmp_path / "py_pip_shim.py"
    py_shim.write_text(
        f"#!{sys.executable}\n"
        "import os, shutil, subprocess, sys\n"
        "from pathlib import Path\n"
        "argv = sys.argv[1:]\n"
        'if len(argv) >= 3 and argv[0] == "-m" and argv[1] == "pip" and argv[2] == "install":\n'
        '    t_idx = argv.index("--target")\n'
        "    site_dir = Path(argv[t_idx + 1])\n"
        "    src_dir = Path(argv[-1])\n"
        '    setup_py = src_dir / "setup.py"\n'
        "    if setup_py.is_file():\n"
        f'        r = subprocess.run(["{sys.executable}", str(setup_py)], cwd=str(src_dir))\n'
        "        if r.returncode != 0:\n"
        "            raise SystemExit(r.returncode)\n"
        "    site_dir.mkdir(parents=True, exist_ok=True)\n"
        '    dst = site_dir / "vllm"\n'
        "    if dst.exists():\n"
        "        shutil.rmtree(dst)\n"
        '    pkg_src = src_dir / "vllm"\n'
        '    skip_rs = os.environ.get("TEST_OMIT_INSTALLED_VLLM_RS") == "1"\n'
        "    shutil.copytree(\n"
        "        pkg_src,\n"
        "        dst,\n"
        '        ignore=(lambda _d, names: ["vllm-rs"] if skip_rs and "vllm-rs" in names else []),\n'
        "    )\n"
        "    raise SystemExit(0)\n"
        f'os.execv("{sys.executable}", ["{sys.executable}", *argv])\n',
        encoding="utf-8",
    )
    py_shim.chmod(0o755)

    def _init_sail_fixture(sail_dir: Path) -> None:
        for rel in (
            "csrc/upstream/torch_bindings.cpp",
            "csrc/upstream/moe/torch_bindings.cpp",
        ):
            p = sail_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("// healthy bindings\n", encoding="utf-8")
        mf = sail_dir / "csrc/upstream/manifest.toml"
        mf.write_text(
            'compiled_tiers = ["a"]\n[[entry]]\ntier = "a"\npath = "csrc/kernel.cu"\nops = []\n',
            encoding="utf-8",
        )

    def _init_upstream_repo(repo_dir: Path):
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.email", "ci@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.name", "CI"], check=True
        )
        for rel in (
            "csrc/torch_bindings.cpp",
            "csrc/moe/torch_bindings.cpp",
            "csrc/kernel.cu",
        ):
            p = repo_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("// healthy\n", encoding="utf-8")
        (repo_dir / "vllm").mkdir(parents=True, exist_ok=True)
        (repo_dir / "vllm/__init__.py").write_text(
            "__version__ = '1.0'\n", encoding="utf-8"
        )

        def _commit(msg: str) -> str:
            subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True)
            subprocess.run(
                ["git", "-C", str(repo_dir), "commit", "-qm", msg], check=True
            )
            return subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
            ).strip()

        return _commit

    sail_dir = tmp_path / "sail"
    _init_sail_fixture(sail_dir)

    # --- 1a. [P1 r4677055119] Optional-by-default native/Rust flag (`VLLM_REQUIRE_RUST_FRONTEND=1`) ---
    opt_repo = tmp_path / "opt-repo"
    commit_opt = _init_upstream_repo(opt_repo)
    (opt_repo / "native.c").write_text("int ok(void) { return 0; }\n", encoding="utf-8")
    (opt_repo / "setup.py").write_text(
        "import os, pathlib, sys\n"
        "req = os.getenv('VLLM_REQUIRE_RUST_FRONTEND', '').strip().lower() in ('1', 'true', 'yes', 'on')\n"
        "src = pathlib.Path('native.c').read_text(encoding='utf-8')\n"
        "if '#error' in src and req:\n"
        "    sys.stderr.write('native.c:1:2: error: native probe compile regression\\n')\n"
        "    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    opt_good = commit_opt("control")
    (opt_repo / "native.c").write_text(
        "#error native probe compile regression\n", encoding="utf-8"
    )
    opt_bad = commit_opt("break mandatory native build")

    opt_out = tmp_path / "opt-out"
    res_opt = post_merge.run_source_only_build_triage(
        vllm_repo=opt_repo,
        sail_root=sail_dir,
        target_sha=opt_bad,
        verified_pin=opt_good,
        last_green_sha=opt_good,
        output_dir=opt_out,
        python_exe=str(py_shim),
    )
    assert res_opt["bisect"]["bisect_status"] == "converged"
    assert res_opt["bisect"]["first_failing_sha"] == opt_bad
    assert res_opt["bisect"]["last_good_sha"] == opt_good
    assert res_opt["non_ppu_repro"]["verified"] is True

    opt_pub = tmp_path / "opt-pub"
    opt_pin = tmp_path / "opt.pin"
    opt_pin.write_text(opt_good + "\n", encoding="utf-8")
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(970),
                "--upstream-sha",
                opt_bad,
                "--verified-pin-file",
                str(opt_pin),
                "--failure-stage",
                "wheel_build",
                "--cpu-result-json",
                str(opt_out / "cpu-result.json"),
                "--bisect-result-json",
                str(opt_out / "bisect-result.json"),
                "--non-ppu-repro-json",
                str(opt_out / "non-ppu-repro.json"),
                "--public-dir",
                str(opt_pub),
            ]
        )
        == 0
    )
    opt_sum = json.loads((opt_pub / "public-summary.json").read_text(encoding="utf-8"))
    opt_iss = json.loads((opt_pub / "issue-payload.json").read_text(encoding="utf-8"))
    assert opt_sum["triage"]["classification"] == "plugin_interface_break"
    assert opt_iss["should_file_sail_issue"] is False
    assert "VLLM_REQUIRE_RUST_FRONTEND=1" in opt_iss["upstream_reproducer_markdown"]
    assert "tools/build_rust.py --release" in opt_iss["upstream_reproducer_markdown"]

    # --- 1b & 1c. [P1 r4677055119] `tools/build_rust.py --release` rustc error & missing `vllm-rs` artifact ---
    rust_repo = tmp_path / "rust-repo"
    commit_rust = _init_upstream_repo(rust_repo)
    (rust_repo / "setup.py").write_text("", encoding="utf-8")
    (rust_repo / "tools").mkdir(parents=True, exist_ok=True)
    (rust_repo / "tools/build_rust.py").write_text(
        "from pathlib import Path\n"
        "def rust_py_extension_module_names():\n"
        "    return ['_rust_tool_parser']\n"
        "if __name__ == '__main__':\n"
        "    vdir = Path(__file__).resolve().parents[1] / 'vllm'\n"
        "    (vdir / 'vllm-rs').write_bytes(b'#!/bin/sh\\necho vllm-rs\\n')\n"
        "    (vdir / '_rust_tool_parser.abi3.so').write_bytes(b'ELF')\n",
        encoding="utf-8",
    )
    rust_good = commit_rust("control with vllm-rs and _rust_tool_parser")

    # 1b: rustc syntax error (without [E0425] code) in `rust/src/cmd/src/main.rs`
    (rust_repo / "tools/build_rust.py").write_text(
        "import sys\n"
        "def rust_py_extension_module_names():\n"
        "    return ['_rust_tool_parser']\n"
        "if __name__ == '__main__':\n"
        "    sys.stderr.write('error: expected `;`, found `}`\\n  --> rust/src/cmd/src/main.rs:19:5\\n')\n"
        "    raise SystemExit(101)\n",
        encoding="utf-8",
    )
    rust_compile_bad = commit_rust("break rust/src/cmd/src/main.rs")
    rc_out = tmp_path / "rust-compile-out"
    res_rc = post_merge.run_source_only_build_triage(
        vllm_repo=rust_repo,
        sail_root=sail_dir,
        target_sha=rust_compile_bad,
        verified_pin=rust_good,
        last_green_sha=rust_good,
        output_dir=rc_out,
        python_exe=str(py_shim),
    )
    assert res_rc["bisect"]["bisect_status"] == "converged"
    assert res_rc["bisect"]["first_failing_sha"] == rust_compile_bad
    assert res_rc["bisect"]["target_signature"] == "rust_frontend_build"
    assert res_rc["non_ppu_repro"]["verified"] is True

    # 1c: `tools/build_rust.py --release` exits 0 without producing `vllm/vllm-rs`
    subprocess.run(
        ["git", "-C", str(rust_repo), "checkout", "-q", rust_good], check=True
    )
    (rust_repo / "tools/build_rust.py").write_text(
        "def rust_py_extension_module_names():\n"
        "    return []\n"
        "if __name__ == '__main__':\n"
        "    pass\n",
        encoding="utf-8",
    )
    rust_missing_bin = commit_rust("silently drop vllm-rs artifact")
    rm_out = tmp_path / "rust-missing-out"
    res_rm = post_merge.run_source_only_build_triage(
        vllm_repo=rust_repo,
        sail_root=sail_dir,
        target_sha=rust_missing_bin,
        verified_pin=rust_good,
        last_green_sha=rust_good,
        output_dir=rm_out,
        python_exe=str(py_shim),
    )
    assert res_rm["bisect"]["bisect_status"] == "converged"
    assert res_rm["bisect"]["first_failing_sha"] == rust_missing_bin
    assert res_rm["non_ppu_repro"]["verified"] is True
    assert res_rm["non_ppu_repro"]["first_failing_phase"] == "build_import"
    assert res_rm["non_ppu_repro"]["failure_source"] == "upstream_import_check"

    # 1d: Cargo network / missing toolchain errors classify as `infra_error`, never upstream!
    for infra_err in (
        "error: failed to download from `https://rsproxy.cn/crates.io-index`",
        "error: can't find Rust compiler",
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm/tools/build_rust.py", line 13, in <module>\n'
        "    from setuptools_rust import RustExtension\n"
        "ModuleNotFoundError: No module named 'setuptools_rust'",
    ):
        ph, fsrc, is_inf = post_merge._classify_install_failure(
            1, infra_err, "", mode="cpu_empty_build"
        )
        assert (ph, fsrc, is_inf) == ("infra_error", "infra_or_unknown", True)

    # --- 2a. [P1 r4677055123] Mixed Tier-A drift (still builds wheel) + later `setup.py` break ---
    mix_repo = tmp_path / "mix-repo"
    commit_mix = _init_upstream_repo(mix_repo)
    (mix_repo / "setup.py").write_text("", encoding="utf-8")
    mix_good = commit_mix("control")
    (mix_repo / "csrc/kernel.cu").unlink()
    mix_drift = commit_mix("move Tier-A kernel file; Python wheel still builds")
    (mix_repo / "setup.py").write_text(
        'raise RuntimeError("invalid upstream build configuration")\n',
        encoding="utf-8",
    )
    mix_bad = commit_mix("break setup.py packaging")

    mix_out = tmp_path / "mix-out"
    res_mix = post_merge.run_source_only_build_triage(
        vllm_repo=mix_repo,
        sail_root=sail_dir,
        target_sha=mix_bad,
        verified_pin=mix_good,
        last_green_sha=mix_good,
        output_dir=mix_out,
        python_exe=str(py_shim),
    )
    assert res_mix["bisect"]["bisect_status"] == "converged"
    assert res_mix["bisect"]["first_failing_sha"] == mix_bad
    assert res_mix["bisect"]["last_good_sha"] == mix_drift
    assert res_mix["non_ppu_repro"]["verified"] is True

    mix_pub = tmp_path / "mix-pub"
    mix_pin = tmp_path / "mix.pin"
    mix_pin.write_text(mix_good + "\n", encoding="utf-8")
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(971),
                "--upstream-sha",
                mix_bad,
                "--verified-pin-file",
                str(mix_pin),
                "--failure-stage",
                "wheel_build",
                "--cpu-result-json",
                str(mix_out / "cpu-result.json"),
                "--bisect-result-json",
                str(mix_out / "bisect-result.json"),
                "--non-ppu-repro-json",
                str(mix_out / "non-ppu-repro.json"),
                "--public-dir",
                str(mix_pub),
            ]
        )
        == 0
    )
    mix_sum = json.loads((mix_pub / "public-summary.json").read_text(encoding="utf-8"))
    mix_iss = json.loads((mix_pub / "issue-payload.json").read_text(encoding="utf-8"))
    assert mix_sum["triage"]["classification"] == "plugin_interface_break"
    assert mix_sum["bisect"]["first_failing_sha"] == mix_bad
    assert mix_iss["should_file_sail_issue"] is False
    assert bool(mix_iss["upstream_reproducer_markdown"]) is True

    # --- 2b. [P1 r4677055123] Independent Tier-A `kernel_port_check` triage when wheel builds at target ---
    kern_out = tmp_path / "kern-only-out"
    res_kern = post_merge.run_source_only_build_triage(
        vllm_repo=mix_repo,
        sail_root=sail_dir,
        target_sha=mix_drift,
        verified_pin=mix_good,
        last_green_sha=mix_good,
        output_dir=kern_out,
        python_exe=str(py_shim),
    )
    assert res_kern["bisect"]["bisect_status"] == "converged"
    assert res_kern["bisect"]["stage"] == "kernel_port_check"
    assert res_kern["bisect"]["first_failing_sha"] == mix_drift
    assert not (kern_out / "non-ppu-repro.json").exists()

    # --- 2c. [R1 B1] A(pass) -> B(import fail, wheel builds) -> C(Rust compile fail):
    #         never replace C's target failure with B's earlier import failure! ---
    b1_repo = tmp_path / "b1-repo"
    commit_b1 = _init_upstream_repo(b1_repo)
    (b1_repo / "setup.py").write_text("", encoding="utf-8")
    (b1_repo / "tools").mkdir(parents=True, exist_ok=True)
    (b1_repo / "tools/build_rust.py").write_text(
        "from pathlib import Path\n"
        "def rust_py_extension_module_names():\n"
        "    return []\n"
        "if __name__ == '__main__':\n"
        "    (Path(__file__).resolve().parents[1] / 'vllm' / 'vllm-rs').write_bytes(b'#!/bin/sh\\n')\n",
        encoding="utf-8",
    )
    sha_a = commit_b1("A: healthy")
    (b1_repo / "vllm/__init__.py").write_text(
        'raise RuntimeError("earlier import regression")\n', encoding="utf-8"
    )
    sha_b = commit_b1("B: wheel builds, import fails")
    (b1_repo / "tools/build_rust.py").write_text(
        "import sys\n"
        "def rust_py_extension_module_names():\n"
        "    return []\n"
        "if __name__ == '__main__':\n"
        "    sys.stderr.write('error: expected `;`\\n  --> rust/src/cmd/src/main.rs:19:5\\n')\n"
        "    raise SystemExit(101)\n",
        encoding="utf-8",
    )
    sha_c = commit_b1("C: Rust compile error on top of B")

    b1_out = tmp_path / "b1-out"
    res_b1 = post_merge.run_source_only_build_triage(
        vllm_repo=b1_repo,
        sail_root=sail_dir,
        target_sha=sha_c,
        verified_pin=sha_a,
        last_green_sha=sha_a,
        output_dir=b1_out,
        python_exe=str(py_shim),
    )
    assert res_b1["bisect"]["bisect_status"] == "unverified_target_boundary"
    assert res_b1["bisect"]["first_failing_sha"] == ""
    assert res_b1["bisect"]["first_failing_sha"] != sha_b
    assert res_b1["bisect"]["stage"] == "wheel_build"
    assert res_b1["bisect"]["target_signature"] == "rust_frontend_build"
    assert res_b1["non_ppu_repro"]["verified"] is False

    b1_pub = tmp_path / "b1-pub"
    b1_pin = tmp_path / "b1.pin"
    b1_pin.write_text(sha_a + "\n", encoding="utf-8")
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(972),
                "--upstream-sha",
                sha_c,
                "--verified-pin-file",
                str(b1_pin),
                "--failure-stage",
                "wheel_build",
                "--cpu-result-json",
                str(b1_out / "cpu-result.json"),
                "--bisect-result-json",
                str(b1_out / "bisect-result.json"),
                "--non-ppu-repro-json",
                str(b1_out / "non-ppu-repro.json"),
                "--public-dir",
                str(b1_pub),
            ]
        )
        == 0
    )
    b1_sum = json.loads((b1_pub / "public-summary.json").read_text(encoding="utf-8"))
    b1_iss = json.loads((b1_pub / "issue-payload.json").read_text(encoding="utf-8"))
    assert b1_sum["bisect"]["first_failing_sha"] is None
    assert b1_sum["triage"]["target_signature"] == "rust_frontend_build"
    assert b1_sum["triage"]["non_ppu_repro_verified"] is False
    assert b1_iss["should_file_sail_issue"] is False
    assert b1_iss["upstream_reproducer_markdown"] == ""

    # --- 2d. [R2 B1-sig] Same phase (`build_install`) with different signature on re-verification
    #         (both primary boundary and `boundary_recheck`, plus `classify_failure` consumer):
    #         must reject as `unverified_target_boundary` (`verified=False`)! ---
    sig_counter = tmp_path / "sig_counter.txt"
    sig_counter.write_text("0\n", encoding="utf-8")
    # 5-commit non-monotonic history: [c1(fail), c2(fail), c3(pass), c4(fail=target)]
    # Binary search on [c1, c2, c3, c4] picks mid=1 (c2 fail) -> mid=0 (c1 fail) -> chosen_bad=c1 != target(c4),
    # exercising BOTH primary verification on (c0, c1) AND `boundary_recheck` on (c3, c4)!
    nm_repo = tmp_path / "nm-sig-repo"
    commit_nm = _init_upstream_repo(nm_repo)
    (nm_repo / "setup.py").write_text("", encoding="utf-8")
    nm_c0 = commit_nm("c0: control")
    nm_shas = [nm_c0]
    for idx in range(1, 5):
        (nm_repo / "README.md").write_text(f"nm {idx}\n", encoding="utf-8")
        nm_shas.append(commit_nm(f"c{idx}"))

    # During triage probes (`PROBE_RESULT_JSON` without `REPRO_KIND`), c1/c2/c4 fail as `rust_frontend_build`
    # and c0/c3 pass. During `verify_non_ppu_reproducer` (`REPRO_KIND=cpu_empty_build`), c1 and c4 fail in the
    # SAME phase (`build_install`) with `VLLM_TARGET_DEVICE=empty` (setup.py RuntimeError).
    nm_probe_cmd = (
        "set -euo pipefail\n"
        f'if [ "$CANDIDATE_SHA" = "{nm_shas[0]}" ] || [ "$CANDIDATE_SHA" = "{nm_shas[3]}" ]; then\n'
        f'  PYTHONPATH="{nm_repo}" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "${REPRO_KIND:-}" = "cpu_empty_build" ]; then\n'
        "  cat >&2 <<'EOF'\n"
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm/setup.py", line 4, in <module>\n'
        '    raise RuntimeError("different build_install signature")\n'
        "RuntimeError: different build_install signature\n"
        "EOF\n"
        "  exit 1\n"
        "fi\n"
        "cat >&2 <<'EOF'\n"
        "error[E0425]: cannot find value `x`\n"
        "  --> rust/src/cmd/src/main.rs:9:5\n"
        "EOF\n"
        "exit 101\n"
    )
    nm_out = tmp_path / "nm-sig-out"
    res_nm = post_merge.run_source_only_build_triage(
        vllm_repo=nm_repo,
        sail_root=sail_dir,
        target_sha=nm_shas[4],
        verified_pin=nm_shas[0],
        last_green_sha=nm_shas[0],
        output_dir=nm_out,
        probe_cmd=nm_probe_cmd,
    )
    assert res_nm["bisect"]["bisect_status"] == "unverified_target_boundary"
    assert res_nm["bisect"]["first_failing_sha"] == ""
    assert res_nm["bisect"]["target_signature"] == "rust_frontend_build"
    assert res_nm["non_ppu_repro"]["verified"] is False
    assert res_nm["non_ppu_repro"]["observed_signature"] == "VLLM_TARGET_DEVICE=empty"
    assert any(h["role"] == "boundary_recheck" for h in res_nm["bisect"]["history"])

    # Consumer `classify_failure` also rejects mismatched canonical wheel_build `observed_signature`
    mismatched_ev = post_merge.NonPpuReproEvidence(
        mode="cpu_empty_build",
        repro_kind="cpu_empty_build",
        last_good_sha=nm_shas[3],
        first_failing_sha=nm_shas[4],
        last_good_exit_code=0,
        first_failing_exit_code=1,
        verified=True,
        last_good_phase="build_import",
        first_failing_phase="build_install",
        last_good_check_executed=True,
        first_failing_check_executed=False,
        infra_error=False,
        failure_source="upstream_build_source",
        observed_signature="VLLM_TARGET_DEVICE=empty",
    )
    mismatch_triage = post_merge.classify_failure(
        post_merge.ProbeOutcome(
            status="fail",
            stage="wheel_build",
            target_signature="rust_frontend_build",
            is_public_plugin_interface=True,
            non_ppu_repro=mismatched_ev,
        ),
        last_good_sha=nm_shas[3],
        first_failing_sha=nm_shas[4],
        require_shas=True,
    )
    assert mismatch_triage["classification"] == "sail_internal_patch_drift"
    assert mismatch_triage["non_ppu_repro_verified"] is False

    # Reverse canonical direction (`target=VLLM_TARGET_DEVICE=empty`, `observed=rust_frontend_build`)
    # through JSON serialization, `parse_non_ppu_repro_evidence`, and full `publish` CLI must ALSO reject!
    rev_ev = post_merge.NonPpuReproEvidence(
        mode="cpu_empty_build",
        repro_kind="cpu_empty_build",
        last_good_sha=nm_shas[3],
        first_failing_sha=nm_shas[4],
        last_good_exit_code=0,
        first_failing_exit_code=101,
        verified=True,
        last_good_phase="build_import",
        first_failing_phase="build_install",
        last_good_check_executed=True,
        first_failing_check_executed=False,
        infra_error=False,
        failure_source="upstream_build_source",
        observed_signature="rust_frontend_build",
    )
    rev_json = tmp_path / "rev-repro.json"
    rev_json.write_text(
        json.dumps(rev_ev.to_public_dict(), indent=2) + "\n", encoding="utf-8"
    )
    rev_pub = tmp_path / "rev-pub"
    rev_pin = tmp_path / "rev.pin"
    rev_pin.write_text(nm_shas[0] + "\n", encoding="utf-8")
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(973),
                "--upstream-sha",
                nm_shas[4],
                "--verified-pin-file",
                str(rev_pin),
                "--first-failing-sha",
                nm_shas[4],
                "--last-good-sha",
                nm_shas[3],
                "--failure-stage",
                "wheel_build",
                "--target-signature",
                "VLLM_TARGET_DEVICE=empty",
                "--public-plugin-break",
                "--non-ppu-repro-json",
                str(rev_json),
                "--public-dir",
                str(rev_pub),
            ]
        )
        == 0
    )
    rev_sum = json.loads((rev_pub / "public-summary.json").read_text(encoding="utf-8"))
    rev_iss = json.loads((rev_pub / "issue-payload.json").read_text(encoding="utf-8"))
    assert rev_sum["triage"]["target_signature"] == "VLLM_TARGET_DEVICE=empty"
    assert rev_sum["triage"]["classification"] == "sail_internal_patch_drift"
    assert rev_sum["triage"]["non_ppu_repro_verified"] is False
    assert rev_iss["upstream_reproducer_markdown"] == ""

    # Malformed/sensitive `observed_signature` must fail closed (`None`), never impersonating `VLLM_TARGET_DEVICE=empty`!
    forged_sig_dict = rev_ev.to_public_dict()
    forged_sig_dict["observed_signature"] = "/Users/leak/secret"
    assert post_merge.parse_non_ppu_repro_evidence(forged_sig_dict) is None


# ---------------------------------------------------------------------------
# Active RC Window `refs/heads/releases/v*` Tracking & Deduplication Suite
# ---------------------------------------------------------------------------


def test_rc_release_branch_ls_remote_filtering_and_real_v034_final_tag_cutoff():
    sha_main = _sha(980)
    sha_exp = _sha(981)
    sha_patch_br = _sha(982)
    sha_v034_tag_obj = "0c57b4195da4a6b69a7725908e428053e51e51b8"
    sha_v034_commit = "62d4bc3c7e33b1b6c906e233e6af2d3506c482ca"
    sha_v034_rc3 = "9950ce36a2e1f4307e99bb4972202c1618b4787e"
    sha_v035_br = _sha(985)

    # 1. Strict RELEASE_BRANCH_RE filtering in parse_ls_remote.
    raw_ls = "\n".join(
        [
            f"{sha_main}\trefs/heads/main",
            f"{sha_exp}\trefs/heads/releases/v0.20.1-python-from-source",
            f"{sha_patch_br}\trefs/heads/releases/v0.34.1",
            f"{sha_exp}\trefs/heads/releases/v0.35.0-experimental",
            f"{sha_exp}\trefs/heads/releases/v0.35.0rc0",
            f"{sha_v035_br}\trefs/heads/releases/v0.35.0",
            f"{sha_v034_commit}\trefs/heads/releases/v0.34.0",
            f"{sha_v034_tag_obj}\trefs/tags/v0.34.0",
            f"{sha_v034_commit}\trefs/tags/v0.34.0^{{}}",
            f"{sha_v034_rc3}\trefs/tags/v0.34.0rc3",
            f"{sha_v034_commit}\trefs/tags/v0.34.0rc4",
        ]
    )
    parsed = post_merge.parse_ls_remote(raw_ls)
    assert parsed["main_sha"] == sha_main
    assert parsed["release_branches"] == {
        "releases/v0.34.0": sha_v034_commit,
        "releases/v0.35.0": sha_v035_br,
    }

    # 2. Real upstream v0.34.0 snapshot (final v0.34.0 tag already exists, RELEASE_TAG == v0.31.0):
    #    neither releases/v0.34.0 nor historical v0.34.0rc* may trigger rc_qualify!
    v034_only = {
        "main_sha": sha_main,
        "tags": parsed["tags"],
        "release_branches": {"releases/v0.34.0": sha_v034_commit},
    }
    d_auto_v034 = post_merge.route_three_gears(
        ls_remote_data=v034_only,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag=RELEASE_TAG,
        requested_gear="auto",
    )
    assert d_auto_v034["selected_gear"] == "probe"
    assert d_auto_v034["target_sha"] == sha_main
    assert d_auto_v034["rc_tag"] == ""

    d_rc_v034 = post_merge.route_three_gears(
        ls_remote_data=v034_only,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag=RELEASE_TAG,
        requested_gear="rc_qualify",
    )
    assert d_rc_v034["selected_gear"] == "rc_qualify"
    assert d_rc_v034["target_sha"] == ""
    assert d_rc_v034["rc_tag"] == ""
    assert d_rc_v034["run_cpu"] is False and d_rc_v034["run_ppu"] is False
    assert (
        post_merge.evaluate_workflow_outcome(
            selected_gear=d_rc_v034["selected_gear"],
            target_sha=d_rc_v034["target_sha"],
            run_cpu=d_rc_v034["run_cpu"],
            run_ppu=d_rc_v034["run_ppu"],
            ppu_skip_reason=d_rc_v034["ppu_skip_reason"],
        )["status"]
        == "noop"
    )

    # Even if post-GA commits land on releases/v0.34.0, presence of v0.34.0 tag keeps RC polling closed.
    v034_post_ga = {
        "main_sha": sha_main,
        "tags": parsed["tags"],
        "release_branches": {"releases/v0.34.0": _sha(986)},
    }
    assert (
        post_merge.select_newest_rc_target(
            v034_post_ga["tags"],
            v034_post_ga["release_branches"],
            release_baseline_tag=RELEASE_TAG,
        )
        is None
    )

    # Meanwhile active-window releases/v0.35.0 (no v0.35.0 final tag yet) IS selected!
    d_active_v035 = post_merge.route_three_gears(
        ls_remote_data=parsed,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag=RELEASE_TAG,
        requested_gear="auto",
    )
    assert d_active_v035["selected_gear"] == "rc_qualify"
    assert d_active_v035["rc_tag"] == "releases/v0.35.0"
    assert d_active_v035["target_sha"] == sha_v035_br

    # Once final tag v0.35.0 is pushed, releases/v0.35.0 window closes immediately.
    closed_v035 = {
        **parsed,
        "tags": {**parsed["tags"], "v0.35.0": _sha(987)},
        "release_branches": {
            **parsed["release_branches"],
            "releases/v0.35.0": _sha(988),
        },
    }
    d_closed_v035 = post_merge.route_three_gears(
        ls_remote_data=closed_v035,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag=RELEASE_TAG,
        requested_gear="auto",
    )
    assert d_closed_v035["selected_gear"] == "probe"
    assert d_closed_v035["target_sha"] == sha_main


def test_rc_release_branch_priority_same_sha_dedup_cherrypick_bisect_and_cursor_isolation(
    tmp_path,
):
    sha_main = _sha(990)
    sha_main_green = _sha(991)
    sha_rc0 = _sha(992)
    sha_cp1 = _sha(993)
    sha_cp2_bad = _sha(994)
    sha_cp3_bad = _sha(995)
    sha_rc1 = _sha(996)
    sha_cp4 = _sha(997)

    # Seed main cursor with healthy probe history and patch fingerprints.
    fp_key = "t_" + "a" * 24
    fp_val = "b" * 64
    base_cursor = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_main_green,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
        patch_fingerprints={fp_key: fp_val},
    )

    # 1. Both untested v0.35.0rc0 AND releases/v0.35.0 exist -> named v0.35.0rc0 tag wins priority!
    ls_rc0 = {
        "main_sha": sha_main,
        "tags": {"v0.35.0rc0": sha_rc0},
        "release_branches": {"releases/v0.35.0": sha_rc0},
    }
    d1 = post_merge.route_three_gears(
        ls_remote_data=ls_rc0,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=base_cursor,
    )
    assert d1["selected_gear"] == "rc_qualify"
    assert d1["rc_tag"] == "v0.35.0rc0"
    assert d1["target_sha"] == sha_rc0
    assert d1["last_green_sha"] == sha_main_green

    # If v0.35.0rc0 fails first, non-force poll with releases/v0.35.0 still at sha_rc0 must NOT re-run sha_rc0!
    cur_rc0_fail = post_merge.advance_cursor(
        base_cursor,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc0,
        rc_tag="v0.35.0rc0",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    d_after_rc0_fail = post_merge.route_three_gears(
        ls_remote_data=ls_rc0,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_rc0_fail,
    )
    assert d_after_rc0_fail["selected_gear"] == "probe"
    assert d_after_rc0_fail["target_sha"] == sha_main

    # Now v0.35.0rc0 passes: same-SHA releases/v0.35.0 MUST deduplicate (noop on rc_qualify, probe on auto)!
    cur_rc0_pass = post_merge.advance_cursor(
        base_cursor,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc0,
        rc_tag="v0.35.0rc0",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    d_same_sha_auto = post_merge.route_three_gears(
        ls_remote_data=ls_rc0,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_rc0_pass,
    )
    assert d_same_sha_auto["selected_gear"] == "probe"
    assert d_same_sha_auto["target_sha"] == sha_main

    d_same_sha_rc = post_merge.route_three_gears(
        ls_remote_data=ls_rc0,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="rc_qualify",
        cursor=cur_rc0_pass,
    )
    assert d_same_sha_rc["target_sha"] == ""
    assert d_same_sha_rc["rc_tag"] == ""
    assert d_same_sha_rc["run_cpu"] is False and d_same_sha_rc["run_ppu"] is False

    # 2. Cherry-pick sha_cp1 lands on releases/v0.35.0 -> triggers rc_qualify with last_green_sha == sha_rc0!
    ls_cp1 = {
        "main_sha": sha_main,
        "tags": {"v0.35.0rc0": sha_rc0},
        "release_branches": {"releases/v0.35.0": sha_cp1},
    }
    d_cp1 = post_merge.route_three_gears(
        ls_remote_data=ls_cp1,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_rc0_pass,
    )
    assert d_cp1["selected_gear"] == "rc_qualify"
    assert d_cp1["rc_tag"] == "releases/v0.35.0"
    assert d_cp1["target_sha"] == sha_cp1
    assert d_cp1["last_green_sha"] == sha_rc0
    assert d_cp1["run_cpu"] is True and d_cp1["run_ppu"] is True

    # sha_cp1 passes via full publish CLI -> public-summary.json, Step Summary, and cursor.json record releases/v0.35.0.
    cursor_in = tmp_path / "cursor-in.json"
    cursor_in.write_text(json.dumps(cur_rc0_pass, indent=2) + "\n", encoding="utf-8")
    pub_cp1 = tmp_path / "pub-cp1"
    summary_cp1 = tmp_path / "summary-cp1.md"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "rc_qualify",
                "--status",
                "pass",
                "--sail-sha",
                _sha(998),
                "--upstream-sha",
                sha_cp1,
                "--rc-tag",
                "releases/v0.35.0",
                "--cpu-status",
                "pass",
                "--ppu-status",
                "pass",
                "--cursor-in",
                str(cursor_in),
                "--public-dir",
                str(pub_cp1),
                "--step-summary",
                str(summary_cp1),
            ]
        )
        == 0
    )
    pub_sum = json.loads((pub_cp1 / "public-summary.json").read_text(encoding="utf-8"))
    assert pub_sum["rc_tag"] == "releases/v0.35.0"
    assert "(`releases/v0.35.0`)" in summary_cp1.read_text(encoding="utf-8")
    cur_cp1_pass = json.loads((pub_cp1 / "cursor.json").read_text(encoding="utf-8"))
    assert cur_cp1_pass["schema"] == 1
    assert cur_cp1_pass["rc_branch_states"]["releases/v0.35.0"] == {
        "sha": sha_cp1,
        "status": "pass",
        "last_green_sha": sha_cp1,
        "green_history": [sha_cp1, sha_rc0],
    }

    # Re-polling sha_cp1 now deduplicates cleanly!
    assert (
        post_merge.route_three_gears(
            ls_remote_data=ls_cp1,
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.34.0",
            requested_gear="rc_qualify",
            cursor=cur_cp1_pass,
        )["target_sha"]
        == ""
    )

    # 3. Two consecutive cherry-pick failures (sha_cp2_bad, sha_cp3_bad) preserve last_green_sha == sha_cp1
    #    and NEVER mutate main-branch cursors, patch_fingerprints, or probe_history!
    ls_cp2 = {**ls_cp1, "release_branches": {"releases/v0.35.0": sha_cp2_bad}}
    d_cp2 = post_merge.route_three_gears(
        ls_remote_data=ls_cp2,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_cp1_pass,
    )
    assert d_cp2["selected_gear"] == "rc_qualify"
    assert d_cp2["last_green_sha"] == sha_cp1

    cur_cp2_fail = post_merge.advance_cursor(
        cur_cp1_pass,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_cp2_bad,
        rc_tag="releases/v0.35.0",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    assert cur_cp2_fail["rc_branch_states"]["releases/v0.35.0"] == {
        "sha": sha_cp2_bad,
        "status": "fail",
        "last_green_sha": sha_cp1,
        "green_history": [sha_cp1, sha_rc0],
        "revoked_shas": [sha_cp2_bad],
    }

    ls_cp3 = {**ls_cp1, "release_branches": {"releases/v0.35.0": sha_cp3_bad}}
    d_cp3 = post_merge.route_three_gears(
        ls_remote_data=ls_cp3,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_cp2_fail,
    )
    assert d_cp3["selected_gear"] == "rc_qualify"
    assert d_cp3["last_green_sha"] == sha_cp1

    cur_cp3_fail = post_merge.advance_cursor(
        cur_cp2_fail,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_cp3_bad,
        rc_tag="refs/heads/releases/v0.35.0",
        status="infra_error",
        cpu_status="not_run",
        ppu_status="not_run",
    )
    assert cur_cp3_fail["rc_branch_states"]["releases/v0.35.0"] == {
        "sha": sha_cp3_bad,
        "status": "infra_error",
        "last_green_sha": sha_cp1,
        "green_history": [sha_cp1, sha_rc0],
        "revoked_shas": [sha_cp3_bad, sha_cp2_bad],
    }
    for main_key in (
        "last_cpu_checked_sha",
        "last_cpu_green_sha",
        "last_cpu_status",
        "last_ppu_tested_sha",
        "last_ppu_green_sha",
        "last_ppu_status",
        "patch_fingerprints",
        "probe_history",
        "last_probe_record",
    ):
        assert cur_cp3_fail[main_key] == base_cursor[main_key]

    # 4. Newer v0.35.0rc1 tag passes -> advances branch last_green_sha to sha_rc1;
    #    manual re-run of older v0.35.0rc0 does NOT regress branch last_green_sha!
    cur_rc1_pass = post_merge.advance_cursor(
        cur_cp3_fail,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc1,
        rc_tag="v0.35.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert (
        cur_rc1_pass["rc_branch_states"]["releases/v0.35.0"]["last_green_sha"]
        == sha_rc1
    )
    cur_manual_rc0 = post_merge.advance_cursor(
        cur_rc1_pass,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc0,
        rc_tag="v0.35.0rc0",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert (
        cur_manual_rc0["rc_branch_states"]["releases/v0.35.0"]["last_green_sha"]
        == sha_rc1
    )
    ls_cp4 = {
        "main_sha": sha_main,
        "tags": {"v0.35.0rc0": sha_rc0, "v0.35.0rc1": sha_rc1},
        "release_branches": {"releases/v0.35.0": sha_cp4},
    }
    d_cp4 = post_merge.route_three_gears(
        ls_remote_data=ls_cp4,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_manual_rc0,
    )
    assert d_cp4["selected_gear"] == "rc_qualify"
    assert d_cp4["rc_tag"] == "releases/v0.35.0"
    assert d_cp4["last_green_sha"] == sha_rc1


def test_rc_legacy_cursor_upgrade_and_manual_override_variants(tmp_path):
    sha_main = _sha(1001)
    sha_rc0 = _sha(1002)
    sha_cp1 = _sha(1003)

    # 1. Legacy schema-1 cursor without `rc_branch_states` (and even without `rc_tag_states` entry) upgrades cleanly.
    legacy_cursor = {
        "schema": 1,
        "last_cpu_checked_sha": VERIFIED_PIN,
        "last_cpu_green_sha": VERIFIED_PIN,
        "last_cpu_status": "pass",
        "last_ppu_tested_sha": VERIFIED_PIN,
        "last_ppu_green_sha": VERIFIED_PIN,
        "last_ppu_status": "pass",
        "tested_rc_tags": ["v0.35.0rc0"],
        "rc_tag_states": {},
        "patch_fingerprints": {},
        "probe_history": [],
    }
    sanitized = post_merge._sanitize_cursor(legacy_cursor, verified_pin=VERIFIED_PIN)
    assert sanitized["schema"] == 1
    assert sanitized["rc_branch_states"] == {}

    # Legacy name-only `tested_rc_tags` without an immutable SHA binding in `rc_tag_states`
    # must re-qualify the RC tag to establish an immutable `(tag -> sha)` receipt, and
    # cannot supply an unverified live tag SHA as `last_green_sha`!
    ls_same = {
        "main_sha": sha_main,
        "tags": {"v0.35.0rc0": sha_rc0},
        "release_branches": {"releases/v0.35.0": sha_rc0},
    }
    d_legacy_dedup = post_merge.route_three_gears(
        ls_remote_data=ls_same,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=legacy_cursor,
    )
    assert d_legacy_dedup["selected_gear"] == "rc_qualify"
    assert d_legacy_dedup["rc_tag"] == "v0.35.0rc0"
    assert d_legacy_dedup["target_sha"] == sha_rc0

    # Once SHA-bound in `rc_tag_states`, deduplication and `last_green_sha` resolution work normally:
    bound_rc0_cursor = post_merge.advance_cursor(
        legacy_cursor,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc0,
        rc_tag="v0.35.0rc0",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data=ls_same,
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.34.0",
            requested_gear="auto",
            cursor=bound_rc0_cursor,
        )["selected_gear"]
        == "probe"
    )
    ls_next = {
        "main_sha": sha_main,
        "tags": {"v0.35.0rc0": sha_rc0},
        "release_branches": {"releases/v0.35.0": sha_cp1},
    }
    d_legacy_next = post_merge.route_three_gears(
        ls_remote_data=ls_next,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=bound_rc0_cursor,
    )
    assert d_legacy_next["selected_gear"] == "rc_qualify"
    assert d_legacy_next["rc_tag"] == "releases/v0.35.0"
    assert d_legacy_next["last_green_sha"] == sha_rc0

    # 2. Manual overrides via `route` CLI: `releases/v0.35.0`, `refs/heads/releases/v0.35.0`, `v0.35.0rc0`, 40-char SHA,
    #    plus `--last-green-sha` override.
    ls_file = tmp_path / "ls-remote.txt"
    ls_file.write_text(
        f"{sha_main}\trefs/heads/main\n"
        f"{sha_cp1}\trefs/heads/releases/v0.35.0\n"
        f"{sha_rc0}\trefs/tags/v0.35.0rc0\n",
        encoding="utf-8",
    )
    cur_file = tmp_path / "cursor.json"
    cur_file.write_text(json.dumps(bound_rc0_cursor), encoding="utf-8")
    rel_file = tmp_path / "release-tag.txt"
    rel_file.write_text("v0.34.0\n", encoding="utf-8")

    for ref_arg, override, expected_control in (
        ("releases/v0.35.0", VERIFIED_PIN, VERIFIED_PIN),
        ("refs/heads/releases/v0.35.0", _sha(1004), sha_rc0),
    ):
        gh_out = tmp_path / f"gh-{ref_arg.replace('/', '_')}.env"
        assert (
            post_merge.main(
                [
                    "route",
                    "--ls-remote-file",
                    str(ls_file),
                    "--release-tag-file",
                    str(rel_file),
                    "--gear",
                    "auto",
                    "--upstream-ref",
                    ref_arg,
                    "--last-green-sha",
                    override,
                    "--cursor-file",
                    str(cur_file),
                    "--github-output",
                    str(gh_out),
                ]
            )
            == 0
        )
        kv = dict(
            line.split("=", 1)
            for line in gh_out.read_text(encoding="utf-8").strip().splitlines()
        )
        assert kv["selected_gear"] == "rc_qualify"
        assert kv["rc_tag"] == "releases/v0.35.0"
        assert kv["target_sha"] == sha_cp1
        assert kv["last_green_sha"] == expected_control

    # Non-existent release branch override fails closed.
    with pytest.raises(ValueError, match="RC release branch"):
        post_merge.route_three_gears(
            ls_remote_data=ls_next,
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.34.0",
            requested_gear="rc_qualify",
            upstream_ref_override="releases/v0.99.0",
        )

    # Malformed rc_branch_states keys/statuses fail closed.
    for bad_branch_states in (
        {"releases/v0.20.1-python-from-source": {"sha": sha_cp1, "status": "pass"}},
        {"releases/v0.35.0": {"sha": sha_cp1, "status": "unknown"}},
        {
            "releases/v0.35.0": {
                "sha": sha_cp1,
                "status": "fail",
                "last_green_sha": "bad",
            }
        },
        {
            "releases/v0.35.0": {
                "sha": sha_cp1,
                "status": "pass",
                "green_history": ["not-a-sha"],
            }
        },
    ):
        with pytest.raises(ValueError):
            post_merge._sanitize_cursor(
                {**legacy_cursor, "rc_branch_states": bad_branch_states},
                verified_pin=VERIFIED_PIN,
            )


def test_review_r1_rc_green_baseline_retry_promotion_and_rerun_revocation_bisect():
    sha_main = _sha(1010)
    sha_main_green = _sha(1011)
    sha_a = _sha(1012)
    sha_b = _sha(1013)
    sha_c = _sha(1014)

    base_cur = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_main_green,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )

    # Trigger A: release branch A pass -> rc1 B fail -> force rc1 B pass -> release branch C
    # Must promote B as last_green_sha once rc1 B passes on retry!
    c_a_pass = post_merge.advance_cursor(
        base_cur,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_a,
        rc_tag="releases/v0.35.0",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    c_rc1_fail = post_merge.advance_cursor(
        c_a_pass,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    c_rc1_retry_pass = post_merge.advance_cursor(
        c_rc1_fail,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    route_after_retry = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_main,
            "tags": {"v0.35.0rc1": sha_b},
            "release_branches": {"releases/v0.35.0": sha_c},
        },
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=c_rc1_retry_pass,
    )
    assert route_after_retry["selected_gear"] == "rc_qualify"
    assert route_after_retry["rc_tag"] == "releases/v0.35.0"
    assert route_after_retry["target_sha"] == sha_c
    assert route_after_retry["last_green_sha"] == sha_b

    # Trigger B: release branch A pass -> rc1 B pass -> manual rc1 B fail -> release branch C
    # Must revoke B from branch green baseline and roll back to A so bisection converges to B!
    c_rc1_pass_first = post_merge.advance_cursor(
        c_a_pass,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert (
        c_rc1_pass_first["rc_branch_states"]["releases/v0.35.0"]["last_green_sha"]
        == sha_b
    )
    c_rc1_rerun_fail = post_merge.advance_cursor(
        c_rc1_pass_first,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    assert (
        c_rc1_rerun_fail["rc_branch_states"]["releases/v0.35.0"]["last_green_sha"]
        == sha_a
    )
    route_after_revoke = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_main,
            "tags": {"v0.35.0rc1": sha_b},
            "release_branches": {"releases/v0.35.0": sha_c},
        },
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=c_rc1_rerun_fail,
    )
    assert route_after_revoke["selected_gear"] == "rc_qualify"
    assert route_after_revoke["last_green_sha"] == sha_a

    # Feed routed `last_green_sha` directly into `bisect_first_failing_sha` consumer (A pass, B/C fail).
    bisect_res = post_merge.bisect_first_failing_sha(
        last_green_sha=route_after_revoke["last_green_sha"],
        red_sha=route_after_revoke["target_sha"],
        commits=[sha_b, sha_c],
        probe_fn=lambda sha: (
            post_merge.ProbeOutcome(status="pass")
            if sha == sha_a
            else post_merge.ProbeOutcome(
                status="fail",
                stage="plugin_registration",
                target_signature="ModelRegistry.register_model",
                is_public_plugin_interface=True,
            )
        ),
        verify_baseline_first=True,
    )
    assert bisect_res["bisect_status"] == "converged"
    assert bisect_res["first_failing_sha"] == sha_b
    assert bisect_res["last_good_sha"] == sha_a

    # Sibling Trigger C (same-SHA branch or cross tag/branch re-run failure also revokes B and preserves main cursor).
    c_branch_b_fail = post_merge.advance_cursor(
        c_rc1_pass_first,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="releases/v0.35.0",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    assert (
        c_branch_b_fail["rc_branch_states"]["releases/v0.35.0"]["last_green_sha"]
        == sha_a
    )
    assert c_branch_b_fail["rc_tag_states"]["v0.35.0rc1"]["status"] == "fail"
    assert "v0.35.0rc1" not in c_branch_b_fail["tested_rc_tags"]
    assert c_branch_b_fail["last_ppu_green_sha"] == sha_main_green

    # If the ONLY green SHA on the branch (`sha_a`) is itself revoked on re-run, `last_green_sha` falls back to main's green SHA!
    c_only_a_revoked = post_merge.advance_cursor(
        c_a_pass,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_a,
        rc_tag="releases/v0.35.0",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    assert (
        "last_green_sha" not in c_only_a_revoked["rc_branch_states"]["releases/v0.35.0"]
    )
    route_after_full_revoke = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_main,
            "tags": {},
            "release_branches": {"releases/v0.35.0": sha_c},
        },
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=c_only_a_revoked,
    )
    assert route_after_full_revoke["last_green_sha"] == sha_main_green


def test_review_r2_sub1_symmetric_same_sha_multi_tag_and_cross_branch_recovery():
    sha_main = _sha(1020)
    sha_main_green = _sha(1021)
    sha_a = _sha(1022)
    sha_b = _sha(1023)
    sha_c = _sha(1024)
    sha_d = _sha(1025)

    base_cur = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_main_green,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_a = post_merge.advance_cursor(
        base_cur,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_a,
        rc_tag="releases/v0.35.0",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )

    # 1. R2 Counterexample 1:
    #    branch(A) pass -> rc1(B) pass -> rc2(B) pass -> rc1(B) fail -> rc1(B) retry pass -> route branch(C)
    cur_rc1_p = post_merge.advance_cursor(
        cur_a,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_rc2_p = post_merge.advance_cursor(
        cur_rc1_p,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc2",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_rc1_f = post_merge.advance_cursor(
        cur_rc2_p,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    assert cur_rc1_f["rc_tag_states"]["v0.35.0rc2"]["status"] == "fail"
    assert cur_rc1_f["rc_branch_states"]["releases/v0.35.0"]["last_green_sha"] == sha_a

    cur_rc1_retry_p = post_merge.advance_cursor(
        cur_rc1_f,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert cur_rc1_retry_p["rc_tag_states"]["v0.35.0rc1"]["status"] == "pass"
    assert cur_rc1_retry_p["rc_tag_states"]["v0.35.0rc2"]["status"] == "pass"
    assert "v0.35.0rc1" in cur_rc1_retry_p["tested_rc_tags"]
    assert "v0.35.0rc2" in cur_rc1_retry_p["tested_rc_tags"]
    assert (
        cur_rc1_retry_p["rc_branch_states"]["releases/v0.35.0"]["last_green_sha"]
        == sha_b
    )

    route_multi_tag = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_main,
            "tags": {"v0.35.0rc1": sha_b, "v0.35.0rc2": sha_b},
            "release_branches": {"releases/v0.35.0": sha_c},
        },
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_rc1_retry_p,
    )
    assert route_multi_tag["selected_gear"] == "rc_qualify"
    assert route_multi_tag["last_green_sha"] == sha_b

    # 2. R2 Counterexample 2:
    #    branch(A) pass -> rc1(B) fail -> branch(B) pass -> branch(D) fail -> route branch(C)
    cur_tag_fail = post_merge.advance_cursor(
        cur_a,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    cur_br_recover = post_merge.advance_cursor(
        cur_tag_fail,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="releases/v0.35.0",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert cur_br_recover["rc_tag_states"]["v0.35.0rc1"]["status"] == "pass"
    assert "v0.35.0rc1" in cur_br_recover["tested_rc_tags"]
    cur_br_d_fail = post_merge.advance_cursor(
        cur_br_recover,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_d,
        rc_tag="releases/v0.35.0",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    route_after_cross = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_main,
            "tags": {"v0.35.0rc1": sha_b},
            "release_branches": {"releases/v0.35.0": sha_c},
        },
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_br_d_fail,
    )
    assert route_after_cross["last_green_sha"] == sha_b
    assert cur_br_d_fail["last_ppu_green_sha"] == sha_main_green

    # 3. Hand-crafted conflicting raw cursor (`rc1(B)=fail` alongside `last_green_sha=B, green_history=[B, A]`):
    #    both `_sanitize_cursor` and `route_three_gears` fail closed consistently to `A`!
    conflicting_raw = {
        **cur_a,
        "rc_tag_states": {"v0.35.0rc1": {"sha": sha_b, "status": "fail"}},
        "rc_branch_states": {
            "releases/v0.35.0": {
                "sha": sha_d,
                "status": "fail",
                "last_green_sha": sha_b,
                "green_history": [sha_b, sha_a],
            }
        },
    }
    clean_conflicting = post_merge._sanitize_cursor(
        conflicting_raw, verified_pin=VERIFIED_PIN
    )
    assert clean_conflicting["rc_branch_states"]["releases/v0.35.0"] == {
        "sha": sha_d,
        "status": "fail",
        "last_green_sha": sha_a,
        "green_history": [sha_a],
        "revoked_shas": [sha_d, sha_b],
    }
    route_conflicting = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_main,
            "tags": {"v0.35.0rc1": sha_b},
            "release_branches": {"releases/v0.35.0": sha_c},
        },
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=conflicting_raw,
    )
    assert route_conflicting["last_green_sha"] == sha_a

    # 4. SUB1 R1 Trigger 1: conflicting `rc1(B)=pass` + `branch(B)=fail/infra_error`, followed by
    #    `advance_cursor(branch, C, fail)` and then `route(D)`: B must NEVER resurrect without passing retry!
    for bad_st in ("fail", "infra_error"):
        conflict_t1 = {
            **base_cur,
            "tested_rc_tags": ["v0.35.0rc1"],
            "rc_tag_states": {"v0.35.0rc1": {"sha": sha_b, "status": "pass"}},
            "rc_branch_states": {
                "releases/v0.35.0": {
                    "sha": sha_b,
                    "status": bad_st,
                    "last_green_sha": sha_b,
                    "green_history": [sha_b],
                }
            },
        }
        after_c_fail = post_merge.advance_cursor(
            conflict_t1,
            verified_pin=VERIFIED_PIN,
            selected_gear="rc_qualify",
            target_sha=sha_c,
            rc_tag="releases/v0.35.0",
            status="fail",
            cpu_status="pass",
            ppu_status="fail",
        )
        assert after_c_fail["rc_tag_states"]["v0.35.0rc1"]["status"] == bad_st
        assert "v0.35.0rc1" not in after_c_fail["tested_rc_tags"]
        route_d = post_merge.route_three_gears(
            ls_remote_data={
                "main_sha": sha_main,
                "tags": {"v0.35.0rc1": sha_b},
                "release_branches": {"releases/v0.35.0": sha_d},
            },
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.34.0",
            requested_gear="auto",
            cursor=after_c_fail,
        )
        assert route_d["last_green_sha"] == sha_main_green
        bisect_t1 = post_merge.bisect_first_failing_sha(
            last_green_sha=route_d["last_green_sha"],
            red_sha=route_d["target_sha"],
            commits=[sha_b, sha_c, sha_d],
            probe_fn=lambda sha: (
                post_merge.ProbeOutcome(status="pass")
                if sha == sha_main_green
                else post_merge.ProbeOutcome(
                    status="fail",
                    stage="plugin_registration",
                    target_signature="ModelRegistry.register_model",
                    is_public_plugin_interface=True,
                )
            ),
            verify_baseline_first=True,
        )
        assert bisect_t1["bisect_status"] == "converged"
        assert bisect_t1["first_failing_sha"] == sha_b

    # Same Trigger 1 with legacy `tested_rc_tags=["v0.35.0rc1"]` and empty `rc_tag_states={}`:
    # `revoked_shas` still prevents B from resurrecting after branch moves to C!
    legacy_conflict_t1 = {
        **base_cur,
        "tested_rc_tags": ["v0.35.0rc1"],
        "rc_tag_states": {},
        "rc_branch_states": {"releases/v0.35.0": {"sha": sha_b, "status": "fail"}},
    }
    legacy_after_c_fail = post_merge.advance_cursor(
        legacy_conflict_t1,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_c,
        rc_tag="releases/v0.35.0",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    route_legacy_d = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_main,
            "tags": {"v0.35.0rc1": sha_b},
            "release_branches": {"releases/v0.35.0": sha_d},
        },
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=legacy_after_c_fail,
    )
    assert route_legacy_d["last_green_sha"] == sha_main_green

    # 5. SUB1 R1 Trigger 2: conflicting `rc1(B)=pass`, `rc2(B)=fail`, `branch={sha:D, status:fail, green_history:[B,A]}`
    #    followed by genuine passing retry `advance_cursor(rc1, B, pass)` -> MUST promote B ahead of A!
    conflict_t2 = {
        **cur_a,
        "tested_rc_tags": ["v0.35.0rc1"],
        "rc_tag_states": {
            "v0.35.0rc1": {"sha": sha_b, "status": "pass"},
            "v0.35.0rc2": {"sha": sha_b, "status": "fail"},
        },
        "rc_branch_states": {
            "releases/v0.35.0": {
                "sha": sha_d,
                "status": "fail",
                "last_green_sha": sha_b,
                "green_history": [sha_b, sha_a],
            }
        },
    }
    after_rc1_genuine_retry = post_merge.advance_cursor(
        conflict_t2,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert after_rc1_genuine_retry["rc_branch_states"]["releases/v0.35.0"][
        "green_history"
    ] == [sha_b, sha_a]
    assert (
        after_rc1_genuine_retry["rc_branch_states"]["releases/v0.35.0"][
            "last_green_sha"
        ]
        == sha_b
    )
    route_after_t2 = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_main,
            "tags": {"v0.35.0rc1": sha_b, "v0.35.0rc2": sha_b},
            "release_branches": {"releases/v0.35.0": sha_c},
        },
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=after_rc1_genuine_retry,
    )
    assert route_after_t2["last_green_sha"] == sha_b


def test_review_6057527910_partial_rerun_cursor_exact_sha_rc_ancestry_and_infra_verdict(
    tmp_path: Path,
) -> None:
    """Regression suite for PR #374 re-review #6057527910 (2 P1s + 2 P2s + controls)."""
    repo = tmp_path / "upstream"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)

    def _git(*args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=RC Review",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    (repo / "vllm").mkdir()
    (repo / "vllm" / "__init__.py").write_text(
        "__version__ = '0.35.0'\n", encoding="utf-8"
    )
    (repo / "setup.py").write_text("# ok\n", encoding="utf-8")
    shas: list[str] = []
    for label in ("pin", "main-green", "A", "B", "C"):
        (repo / "version").write_text(label + "\n", encoding="utf-8")
        if label == "A":
            (repo / "setup.py").write_text(
                'raise RuntimeError("VLLM_TARGET_DEVICE=empty broken")\n',
                encoding="utf-8",
            )
        _git("add", "-A")
        _git("commit", "-qm", label)
        shas.append(_git("rev-parse", "HEAD"))
    pin, main_green, sha_a, sha_b, sha_c = shas

    pin_file = tmp_path / "pin"
    pin_file.write_text(pin + "\n", encoding="utf-8")
    baseline_tag = tmp_path / "release"
    baseline_tag.write_text("v0.34.0\n", encoding="utf-8")
    refs_file = tmp_path / "ls-remote.txt"
    branch = "releases/v0.35.0"
    rc0, rc1 = "v0.35.0rc0", "v0.35.0rc1"
    sail_sha = _sha(991)

    _git("branch", "main-green", main_green)
    _git("checkout", "-q", "main-green")
    _git("update-ref", "refs/heads/main", main_green)

    def _snapshot(tip: str, tags: dict[str, str]) -> None:
        _git("update-ref", f"refs/heads/{branch}", tip)
        for tag, sha in tags.items():
            _git("update-ref", f"refs/tags/{tag}", sha)
        out = subprocess.run(
            [
                "git",
                "ls-remote",
                str(repo),
                "refs/heads/main",
                "refs/heads/releases/v*",
                "refs/tags/v*",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        refs_file.write_text(out + "\n", encoding="utf-8")

    def _cli_route(
        cursor_path: Path | None,
        gear: str = "auto",
        override: str = "",
        vllm_repo_arg: Path | None = None,
    ) -> dict[str, str]:
        out_env = tmp_path / "gh-out.txt"
        out_env.unlink(missing_ok=True)
        argv = [
            "route",
            "--ls-remote-file",
            str(refs_file),
            "--verified-pin-file",
            str(pin_file),
            "--release-tag-file",
            str(baseline_tag),
            "--gear",
            gear,
            "--upstream-ref",
            override,
            "--github-output",
            str(out_env),
        ]
        if cursor_path is not None:
            argv += ["--cursor-file", str(cursor_path)]
        if vllm_repo_arg is not None:
            argv += ["--vllm-repo", str(vllm_repo_arg)]
        assert post_merge.main(argv) == 0
        parsed: dict[str, str] = {}
        for line in out_env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            parsed[k] = v
        return parsed

    def _cli_publish(
        label: str,
        cursor_path: Path | None,
        gear: str,
        sha: str,
        rc_ref: str = "",
        status: str = "pass",
        cpu_status: str = "pass",
        ppu_status: str | None = None,
        failure_stage: str = "none",
    ) -> Path:
        pub_dir = tmp_path / label
        eff_ppu = status if ppu_status is None else ppu_status
        argv = [
            "publish",
            "--gear",
            gear,
            "--upstream-sha",
            sha,
            "--verified-pin-file",
            str(pin_file),
            "--sail-sha",
            sail_sha,
            "--public-dir",
            str(pub_dir),
            "--status",
            status,
            "--cpu-status",
            cpu_status,
            "--ppu-status",
            eff_ppu,
            "--rc-tag",
            rc_ref,
        ]
        if failure_stage != "none":
            argv += ["--failure-stage", failure_stage]
        elif status == "fail":
            argv += ["--failure-stage", "rc_model_sanity"]
        if cursor_path is not None:
            argv += ["--cursor-in", str(cursor_path)]
        assert post_merge.main(argv) == 0
        return pub_dir / "cursor.json"

    # --- 1. [P1 r4694815962] Exact-SHA RC retry updates matching release line & fails closed on ambiguity ---
    main_cursor = _cli_publish("main-cursor", None, "probe", main_green)
    a_pass = _cli_publish("br-a-pass", main_cursor, "rc_qualify", sha_a, branch)
    b_pass = _cli_publish("br-b-pass", a_pass, "rc_qualify", sha_b, branch)
    _snapshot(sha_b, {})

    for label, ref in (("exact-sha", sha_b), ("named-branch-control", branch)):
        dec = _cli_route(b_pass, "rc_qualify", ref)
        assert dec["target_sha"] == sha_b
        assert dec["rc_tag"] == branch
        cur = _cli_publish(
            f"pub-{label}", b_pass, dec["selected_gear"], sha_b, dec["rc_tag"], "fail"
        )
        persisted = json.loads(cur.read_text(encoding="utf-8"))["rc_branch_states"][
            branch
        ]
        assert persisted["status"] == "fail"
        _snapshot(sha_c, {})
        following = _cli_route(cur)
        assert following["last_green_sha"] == sha_a
        _snapshot(sha_b, {})

    # Exact-SHA retry after branch tip already moved to C in ls-remote (matched via cursor history):
    _snapshot(sha_c, {})
    dec_hist_sha = _cli_route(b_pass, "rc_qualify", sha_b)
    assert dec_hist_sha["rc_tag"] == branch
    # Direct publish with empty --rc-tag also resolves unique release line in cursor:
    direct_empty_rc = _cli_publish(
        "pub-direct-empty-rc", b_pass, "rc_qualify", sha_b, "", "fail"
    )
    assert (
        json.loads(direct_empty_rc.read_text(encoding="utf-8"))["rc_branch_states"][
            branch
        ]["last_green_sha"]
        == sha_a
    )
    # Ambiguous exact SHA across two active release lines fails closed:
    with pytest.raises(ValueError, match="ambiguous"):
        post_merge.route_three_gears(
            ls_remote_data={
                "main_sha": main_green,
                "tags": {},
                "release_branches": {
                    "releases/v0.35.0": sha_b,
                    "releases/v0.36.0": sha_b,
                },
            },
            verified_pin=pin,
            release_baseline_tag="v0.34.0",
            requested_gear="rc_qualify",
            upstream_ref_override=sha_b,
            cursor=json.loads(b_pass.read_text(encoding="utf-8")),
        )

    # --- 2. [P1 r4694815957] Partial retry (`Re-run failed jobs`) preserves route cursor & upstream artifacts ---
    wf = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/upstream-canary.yaml").read_text(
            encoding="utf-8"
        )
    )["jobs"]
    restore_step = next(
        s
        for s in wf["route"]["steps"]
        if s.get("name") == "Restore prior cursor artifact when available"
    )
    assert "continue-on-error" not in restore_step
    assert "has_route_cursor" in wf["route"]["outputs"]
    assert "route_artifact_name" in wf["route"]["outputs"]
    assert "fallback_green_shas" in wf["route"]["outputs"]
    assert "has_cpu_artifact" in wf["cpu-fast-check"]["outputs"]
    assert "cpu_artifact_name" in wf["cpu-fast-check"]["outputs"]
    assert "has_ppu_artifact" in wf["ppu-probe-smoke"]["outputs"]
    assert "ppu_artifact_name" in wf["ppu-probe-smoke"]["outputs"]
    assert "selection_artifact" in wf["rc-ops-select"]["outputs"]
    assert (
        wf["rc-ops-tests"]["with"]["selection_artifact"]
        == "${{ needs.rc-ops-select.outputs.selection_artifact }}"
    )

    pub_downloads = [
        s
        for s in wf["publish-and-triage"]["steps"]
        if s.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert len(pub_downloads) >= 3
    for dl in pub_downloads:
        assert "continue-on-error" not in dl
        assert "pattern" not in dl["with"]
        assert "github.run_attempt" not in dl["with"].get("name", "")

    route_dl = next(
        dl
        for dl in pub_downloads
        if dl["with"].get("name") == "${{ needs.route.outputs.route_artifact_name }}"
    )
    # Simulate attempt 1 producing `upstream-post-merge-route-42-1` and attempt 2 reusing `needs.route.outputs`:
    available_artifacts = {"upstream-post-merge-route-42-1"}
    route_output_from_attempt_1 = "upstream-post-merge-route-42-1"
    for attempt in (1, 2):
        requested_name = route_dl["with"]["name"].replace(
            "${{ needs.route.outputs.route_artifact_name }}",
            route_output_from_attempt_1,
        )
        assert requested_name in available_artifacts
        cur_attempt = _cli_publish(
            f"attempt-{attempt}", b_pass, "rc_qualify", sha_c, branch
        )
        loaded = json.loads(cur_attempt.read_text(encoding="utf-8"))
        assert loaded["last_ppu_green_sha"] == main_green
        assert loaded["rc_branch_states"][branch]["green_history"] == [
            sha_c,
            sha_b,
            sha_a,
        ]
        assert len(loaded["probe_history"]) == 1

    # --- 3. [P2 r4694815973] Verified first-parent ancestor selected for RC backlog & post-cut divergent main ---
    _snapshot(sha_b, {rc0: sha_a, rc1: sha_b})
    first_poll = _cli_route(main_cursor)
    assert first_poll["rc_tag"] == rc1 and first_poll["target_sha"] == sha_b
    cur_after_rc1 = _cli_publish(
        "newer-tag-passed", main_cursor, "rc_qualify", sha_b, rc1
    )
    second_poll = _cli_route(cur_after_rc1)
    assert second_poll["rc_tag"] == rc0 and second_poll["target_sha"] == sha_a
    # Even without --vllm-repo, rc0 must NOT pick newer sibling/descendant rc1(B) as its control:
    assert second_poll["last_green_sha"] == main_green

    bisect_backlog = tmp_path / "bisect-backlog.json"
    assert (
        post_merge.main(
            [
                "bisect",
                "--vllm-repo",
                str(repo),
                "--last-green-sha",
                second_poll["last_green_sha"],
                "--red-sha",
                sha_a,
                "--probe-cmd",
                "true",
                "--output-json",
                str(bisect_backlog),
            ]
        )
        == 0
    )
    backlog_res = json.loads(bisect_backlog.read_text(encoding="utf-8"))
    assert backlog_res["bisect_status"] == "converged"
    assert backlog_res["first_failing_sha"] == sha_a

    # Divergent main after release branch cut: `pin -> main-green -> newer_main` vs `pin -> main-green -> A`
    _git("update-ref", "-d", f"refs/tags/{rc0}")
    _git("update-ref", "-d", f"refs/tags/{rc1}")
    (repo / "version").write_text("main after release cut\n", encoding="utf-8")
    _git("add", "version")
    _git("commit", "-qm", "main after release cut")
    newer_main = _git("rev-parse", "HEAD")
    _git("update-ref", "refs/heads/main", newer_main)
    advanced_main = _cli_publish("main-after-cut", main_cursor, "probe", newer_main)
    _snapshot(sha_a, {})

    # (a) Repo-aware route (used by upstream-canary.yaml) picks verified ancestor `main_green` directly,
    #     including on an empty shallow clone with `origin` configured (matching `state/vllm-diff`):
    routed_with_repo = _cli_route(advanced_main, vllm_repo_arg=repo)
    assert routed_with_repo["target_sha"] == sha_a
    assert routed_with_repo["last_green_sha"] == main_green

    empty_diff_repo = tmp_path / "vllm-diff-sim"
    subprocess.run(["git", "init", "-q", str(empty_diff_repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(empty_diff_repo),
            "remote",
            "add",
            "origin",
            repo.as_uri(),
        ],
        check=True,
    )
    routed_shallow_origin = _cli_route(advanced_main, vllm_repo_arg=empty_diff_repo)
    assert routed_shallow_origin["target_sha"] == sha_a
    assert routed_shallow_origin["last_green_sha"] == main_green

    # (b) Offline route without --vllm-repo emits fallback_green_shas, and bisect with --cursor-file /
    #     --fallback-green-shas walks older verified controls to `main_green` and converges to `sha_a`:
    routed_offline = _cli_route(advanced_main)
    assert main_green in routed_offline["fallback_green_shas"].split(",")
    bisect_divergent = tmp_path / "bisect-divergent.json"
    assert (
        post_merge.main(
            [
                "bisect",
                "--vllm-repo",
                str(repo),
                "--last-green-sha",
                routed_offline["last_green_sha"],
                "--fallback-green-shas",
                routed_offline["fallback_green_shas"],
                "--cursor-file",
                str(advanced_main),
                "--verified-pin-file",
                str(pin_file),
                "--red-sha",
                sha_a,
                "--probe-cmd",
                "true",
                "--output-json",
                str(bisect_divergent),
            ]
        )
        == 0
    )
    div_res = json.loads(bisect_divergent.read_text(encoding="utf-8"))
    assert div_res["bisect_status"] == "converged"
    assert div_res["first_failing_sha"] == sha_a
    assert div_res["last_good_sha"] == main_green

    # Also verify source-build-triage resolves verified ancestor `main_green` when given divergent `newer_main`:
    sbt_div_dir = tmp_path / "sbt-divergent"
    sbt_probe_cmd = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'"{sys.executable}" "$wt/setup.py"\n'
        f'PYTHONPATH="$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )
    sbt_div = post_merge.run_source_only_build_triage(
        vllm_repo=repo,
        sail_root=REPO_ROOT,
        target_sha=sha_a,
        verified_pin=pin,
        last_green_sha=routed_offline["last_green_sha"],
        fallback_green_shas=routed_offline["fallback_green_shas"].split(","),
        cursor=json.loads(advanced_main.read_text(encoding="utf-8")),
        rc_tag=branch,
        output_dir=sbt_div_dir,
        probe_cmd=sbt_probe_cmd,
    )
    assert sbt_div["bisect"]["bisect_status"] == "converged"
    assert sbt_div["bisect"]["first_failing_sha"] == sha_a
    assert sbt_div["bisect"]["last_good_sha"] == main_green

    # (c) Unverified fork-point control: if release branch forked from an unverified commit `unver_mid`
    #     (`pin -> unver_mid -> newer_main2` and `pin -> unver_mid -> rc_cut`), only verified `pin` is chosen!
    _git("checkout", "-q", "--detach", pin)
    (repo / "version").write_text("unverified-mid\n", encoding="utf-8")
    _git("add", "version")
    _git("commit", "-qm", "unverified-mid")
    unver_mid = _git("rev-parse", "HEAD")
    (repo / "version").write_text("newer-main-2\n", encoding="utf-8")
    _git("add", "version")
    _git("commit", "-qm", "newer-main-2")
    newer_main2 = _git("rev-parse", "HEAD")
    _git("checkout", "-q", "--detach", unver_mid)
    (repo / "version").write_text("rc-from-unver\n", encoding="utf-8")
    _git("add", "version")
    _git("commit", "-qm", "rc-from-unver")
    rc_from_unver = _git("rev-parse", "HEAD")
    _git("update-ref", "refs/heads/main", newer_main2)
    cur_only_pin_and_nm2 = _cli_publish("only-nm2", None, "probe", newer_main2)
    _snapshot(rc_from_unver, {})
    routed_unver = _cli_route(cur_only_pin_and_nm2, vllm_repo_arg=repo)
    assert routed_unver["last_green_sha"] == pin

    # --- 4. [P2 r4694815979] Explicit `infra_error` verdict survives cursor reload & unrelated main probes ---
    _snapshot(sha_b, {})
    infra_cursor = _cli_publish(
        "infra-rc",
        a_pass,
        "rc_qualify",
        sha_b,
        branch,
        status="fail",
        cpu_status="not_run",
        ppu_status="not_run",
        failure_stage="wheel_build",
    )
    infra_summary = json.loads(
        (tmp_path / "infra-rc" / "public-summary.json").read_text(encoding="utf-8")
    )
    assert infra_summary["status"] == "infra_error"
    assert (
        json.loads(infra_cursor.read_text(encoding="utf-8"))["rc_branch_states"][
            branch
        ]["status"]
        == "infra_error"
    )
    after_unrelated_main = _cli_publish(
        "unrelated-main", infra_cursor, "probe", newer_main
    )
    after_data = json.loads(after_unrelated_main.read_text(encoding="utf-8"))
    assert after_data["rc_branch_states"][branch]["status"] == "infra_error"
    assert sha_b in after_data["rc_branch_states"][branch]["revoked_shas"]
    assert after_data["rc_branch_states"][branch]["last_green_sha"] == sha_a

    # Same preservation when an RC tag also has `infra_error` while `fail` still wins on true conflict:
    tag_infra_cur = post_merge._sanitize_cursor(
        {
            **after_data,
            "rc_tag_states": {rc1: {"sha": sha_b, "status": "infra_error"}},
        },
        verified_pin=pin,
    )
    assert tag_infra_cur["rc_tag_states"][rc1]["status"] == "infra_error"
    assert tag_infra_cur["rc_branch_states"][branch]["status"] == "infra_error"
    tag_fail_conflict = post_merge._sanitize_cursor(
        {
            **after_data,
            "rc_tag_states": {rc1: {"sha": sha_b, "status": "fail"}},
        },
        verified_pin=pin,
    )
    assert tag_fail_conflict["rc_tag_states"][rc1]["status"] == "fail"
    assert tag_fail_conflict["rc_branch_states"][branch]["status"] == "fail"

    # --- 5. [R1 P1-1] Ambiguous empty --rc-tag publish fails closed for pass, fail, and infra_error ---
    ambig_cursor_path = tmp_path / "ambig-cursor.json"
    ambig_cursor_path.write_text(
        json.dumps(
            {
                **json.loads(b_pass.read_text(encoding="utf-8")),
                "rc_branch_states": {
                    "releases/v0.35.0": {
                        "sha": sha_b,
                        "status": "pass",
                        "last_green_sha": sha_b,
                        "green_history": [sha_b, sha_a],
                    },
                    "releases/v0.36.0": {
                        "sha": sha_b,
                        "status": "pass",
                        "last_green_sha": sha_b,
                        "green_history": [sha_b],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for ambig_st, ambig_stage in (
        ("pass", "none"),
        ("fail", "rc_model_sanity"),
        ("fail", "wheel_build"),
    ):
        ambig_pub_dir = tmp_path / f"ambig-pub-{ambig_st}-{ambig_stage}"
        with pytest.raises(ValueError, match="ambiguous"):
            post_merge.main(
                [
                    "publish",
                    "--gear",
                    "rc_qualify",
                    "--upstream-sha",
                    sha_b,
                    "--verified-pin-file",
                    str(pin_file),
                    "--sail-sha",
                    sail_sha,
                    "--public-dir",
                    str(ambig_pub_dir),
                    "--status",
                    ambig_st,
                    "--cpu-status",
                    "not_run" if ambig_stage == "wheel_build" else "pass",
                    "--ppu-status",
                    "not_run" if ambig_stage == "wheel_build" else ambig_st,
                    "--failure-stage",
                    ambig_stage,
                    "--rc-tag",
                    "",
                    "--cursor-in",
                    str(ambig_cursor_path),
                ]
            )
        assert not (ambig_pub_dir / "cursor.json").exists()
        assert not (ambig_pub_dir / "public-summary.json").exists()

    # --- 6. [R1 P1-2] Revoked RC SHA that previously passed on main never re-enters control chain ---
    _git("update-ref", "refs/heads/main", sha_b)
    main_b_pass = _cli_publish("p12-main-b-pass", None, "probe", sha_b)
    rc_b_revoked = _cli_publish(
        "p12-rc-b-revoked", main_b_pass, "rc_qualify", sha_b, branch, status="fail"
    )
    _snapshot(sha_c, {})
    for routed_p12 in (
        _cli_route(rc_b_revoked),
        _cli_route(rc_b_revoked, vllm_repo_arg=repo),
    ):
        assert routed_p12["target_sha"] == sha_c
        assert routed_p12["last_green_sha"] == pin
        assert sha_b not in routed_p12["fallback_green_shas"].split(",")

    # Downstream bisect & source-build-triage skip revoked `sha_b` even if passed as --last-green-sha:
    p12_bisect_json = tmp_path / "p12-bisect.json"
    p12_probe_cmd = (
        f'test "$CANDIDATE_SHA" = "{pin}" || test "$CANDIDATE_SHA" = "{main_green}"'
    )
    assert (
        post_merge.main(
            [
                "bisect",
                "--vllm-repo",
                str(repo),
                "--last-green-sha",
                sha_b,
                "--cursor-file",
                str(rc_b_revoked),
                "--verified-pin-file",
                str(pin_file),
                "--red-sha",
                sha_c,
                "--probe-cmd",
                p12_probe_cmd,
                "--output-json",
                str(p12_bisect_json),
            ]
        )
        == 0
    )
    p12_bisect_res = json.loads(p12_bisect_json.read_text(encoding="utf-8"))
    assert p12_bisect_res["bisect_status"] == "converged"
    assert p12_bisect_res["last_good_sha"] == main_green
    assert p12_bisect_res["first_failing_sha"] == sha_a
    assert all(
        not (step["role"] == "baseline_control" and step["sha"] == sha_b)
        for step in p12_bisect_res["history"]
    )

    p12_sbt_dir = tmp_path / "p12-sbt"
    p12_sbt = post_merge.run_source_only_build_triage(
        vllm_repo=repo,
        sail_root=REPO_ROOT,
        target_sha=sha_c,
        verified_pin=pin,
        last_green_sha=sha_b,
        cursor=json.loads(rc_b_revoked.read_text(encoding="utf-8")),
        rc_tag=branch,
        output_dir=p12_sbt_dir,
        probe_cmd=sbt_probe_cmd,
    )
    assert p12_sbt["bisect"]["bisect_status"] == "converged"
    assert p12_sbt["bisect"]["last_good_sha"] == main_green
    assert p12_sbt["bisect"]["first_failing_sha"] == sha_a

    # Cross-release-line exclusion: `sha_b` revoked on `releases/v0.35.0` must also be excluded
    # when a newer release line (`releases/v0.36.0`) qualifies `sha_c` via route/bisect/source-build-triage:
    branch_036 = "releases/v0.36.0"
    _git("update-ref", f"refs/heads/{branch_036}", sha_c)
    ls_out_036 = subprocess.run(
        [
            "git",
            "ls-remote",
            str(repo),
            "refs/heads/main",
            "refs/heads/releases/v*",
            "refs/tags/v*",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    refs_file.write_text(ls_out_036 + "\n", encoding="utf-8")
    for routed_036 in (
        _cli_route(rc_b_revoked),
        _cli_route(rc_b_revoked, vllm_repo_arg=repo),
    ):
        assert routed_036["rc_tag"] == branch_036
        assert routed_036["target_sha"] == sha_c
        assert routed_036["last_green_sha"] == pin
        assert sha_b not in routed_036["fallback_green_shas"].split(",")

    p12_cross_bisect = tmp_path / "p12-cross-bisect.json"
    assert (
        post_merge.main(
            [
                "bisect",
                "--vllm-repo",
                str(repo),
                "--last-green-sha",
                sha_b,
                "--rc-tag",
                branch_036,
                "--cursor-file",
                str(rc_b_revoked),
                "--verified-pin-file",
                str(pin_file),
                "--red-sha",
                sha_c,
                "--probe-cmd",
                p12_probe_cmd,
                "--output-json",
                str(p12_cross_bisect),
            ]
        )
        == 0
    )
    cross_bisect_res = json.loads(p12_cross_bisect.read_text(encoding="utf-8"))
    assert cross_bisect_res["bisect_status"] == "converged"
    assert cross_bisect_res["last_good_sha"] == main_green
    assert cross_bisect_res["first_failing_sha"] == sha_a

    p12_cross_sbt = post_merge.run_source_only_build_triage(
        vllm_repo=repo,
        sail_root=REPO_ROOT,
        target_sha=sha_c,
        verified_pin=pin,
        last_green_sha=sha_b,
        cursor=json.loads(rc_b_revoked.read_text(encoding="utf-8")),
        rc_tag=branch_036,
        output_dir=tmp_path / "p12-cross-sbt",
        probe_cmd=sbt_probe_cmd,
    )
    assert p12_cross_sbt["bisect"]["bisect_status"] == "converged"
    assert p12_cross_sbt["bisect"]["last_good_sha"] == main_green
    assert p12_cross_sbt["bisect"]["first_failing_sha"] == sha_a

    # Manual `--last-green-sha` override on `route` (same-line, cross-line, and probe/daily_pin_bump)
    # also filters through `_collect_excluded_control_shas` so revoked `sha_b` never re-enters `route`:
    for gear_arg, ref_arg in (
        ("rc_qualify", branch),
        ("rc_qualify", branch_036),
        ("probe", sha_c),
        ("daily_pin_bump", ""),
    ):
        out_ovr = tmp_path / f"gh-ovr-{gear_arg}.txt"
        out_ovr.unlink(missing_ok=True)
        assert (
            post_merge.main(
                [
                    "route",
                    "--ls-remote-file",
                    str(refs_file),
                    "--verified-pin-file",
                    str(pin_file),
                    "--release-tag-file",
                    str(baseline_tag),
                    "--gear",
                    gear_arg,
                    "--upstream-ref",
                    ref_arg,
                    "--last-green-sha",
                    sha_b,
                    "--cursor-file",
                    str(rc_b_revoked),
                    "--github-output",
                    str(out_ovr),
                ]
            )
            == 0
        )
        ovr_kv = {
            k: v
            for k, _, v in (
                ln.partition("=")
                for ln in out_ovr.read_text(encoding="utf-8").splitlines()
            )
        }
        assert ovr_kv["last_green_sha"] != sha_b
        assert sha_b not in ovr_kv["fallback_green_shas"].split(",")


def test_review_6066390398_cursor_discovery_late_rerun_and_durable_branch_revocation(
    tmp_path: Path,
):
    """End-to-end regression suite for GitHub review #6066390398 (2 P1s + 1 P2)."""
    repo = tmp_path / "upstream"
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )

    def _git(*args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=RC Review",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    hashes: list[str] = []
    for label in ("pin", "main-green", "A", "B", "C"):
        (repo / "version").write_text(f"{label}\n", encoding="utf-8")
        _git("add", "version")
        _git("commit", "-qm", label)
        hashes.append(_git("rev-parse", "HEAD"))
    pin_sha, main_green, sha_a, sha_b, sha_c = hashes
    pin_file = tmp_path / "pin"
    pin_file.write_text(f"{pin_sha}\n", encoding="utf-8")
    baseline_tag = tmp_path / "release"
    baseline_tag.write_text("v0.34.0\n", encoding="utf-8")
    refs_file = tmp_path / "ls-remote.txt"
    branch = "releases/v0.35.0"
    sail_sha = "f" * 40

    _git("branch", "main-green", main_green)
    _git("checkout", "-q", "main-green")
    _git("update-ref", "refs/heads/main", main_green)

    def _snapshot(tip: str, tags: dict[str, str]) -> None:
        _git("update-ref", f"refs/heads/{branch}", tip)
        for t_name, t_sha in tags.items():
            _git("update-ref", f"refs/tags/{t_name}", t_sha)
        ls_out = subprocess.run(
            [
                "git",
                "ls-remote",
                str(repo),
                "refs/heads/main",
                "refs/heads/releases/v*",
                "refs/tags/v*",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        refs_file.write_text(f"{ls_out}\n", encoding="utf-8")

    def _cli_route(
        cursor: Path | None, gear: str = "auto", override: str = ""
    ) -> dict[str, str]:
        out_env = tmp_path / "route-gh.out"
        out_env.unlink(missing_ok=True)
        argv = [
            "route",
            "--ls-remote-file",
            str(refs_file),
            "--verified-pin-file",
            str(pin_file),
            "--release-tag-file",
            str(baseline_tag),
            "--vllm-repo",
            str(repo),
            "--gear",
            gear,
            "--upstream-ref",
            override,
            "--github-output",
            str(out_env),
        ]
        if cursor is not None:
            argv.extend(["--cursor-file", str(cursor)])
        assert post_merge.main(argv) == 0
        return {
            k: v
            for k, _, v in (
                ln.partition("=")
                for ln in out_env.read_text(encoding="utf-8").splitlines()
            )
        }

    def _cli_publish(
        label: str,
        cursor: Path | None,
        gear: str,
        sha: str,
        rc_ref: str = "",
        status: str = "pass",
    ) -> Path:
        pub_dir = tmp_path / label
        argv = [
            "publish",
            "--gear",
            gear,
            "--upstream-sha",
            sha,
            "--verified-pin-file",
            str(pin_file),
            "--sail-sha",
            sail_sha,
            "--public-dir",
            str(pub_dir),
            "--status",
            status,
            "--cpu-status",
            "pass" if sha else "not_run",
            "--ppu-status",
            status if sha else "not_run",
            "--rc-tag",
            rc_ref,
        ]
        if status == "fail":
            argv.extend(["--failure-stage", "rc_model_sanity"])
        elif status == "infra_error":
            argv.extend(["--failure-stage", "route"])
        if cursor is not None:
            argv.extend(["--cursor-in", str(cursor)])
        assert post_merge.main(argv) == 0
        return pub_dir / "cursor.json"

    main_cursor = _cli_publish("main-cursor", None, "probe", main_green)
    a_pass = _cli_publish("br-a-pass", main_cursor, "rc_qualify", sha_a, branch)
    b_pass = _cli_publish("br-b-pass", a_pass, "rc_qualify", sha_b, branch)
    b_fail = _cli_publish("br-b-rejected", b_pass, "rc_qualify", sha_b, branch, "fail")
    _snapshot(sha_c, {})

    wf = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/upstream-canary.yaml").read_text(
            encoding="utf-8"
        )
    )["jobs"]
    assert (
        wf["route"]["outputs"]["prior_cursor_artifact"]
        == "${{ steps.prior.outputs.prior_cursor_artifact }}"
    )
    cursor_upload = next(
        s
        for s in wf["publish-and-triage"]["steps"]
        if s.get("name") == "Upload persistent post-merge cursor artifact"
    )
    assert "needs.route.result == 'success'" in cursor_upload["if"]
    assert "needs.route.outputs.target_sha != ''" in cursor_upload["if"]

    # Failed route (`upstream-sha=""`) must NOT emit a reset cursor.json:
    failed_route_cursor = _cli_publish(
        "route-failed-pub", None, "probe", "", status="infra_error"
    )
    assert not failed_route_cursor.exists()
    assert (
        json.loads(
            (failed_route_cursor.parent / "public-summary.json").read_text(
                encoding="utf-8"
            )
        )["status"]
        == "infra_error"
    )

    node_bin = shutil.which("node")
    if node_bin:
        prior_script = next(
            s["with"]["script"] for s in wf["route"]["steps"] if s.get("id") == "prior"
        )
        latest_script = next(
            s["with"]["script"]
            for s in wf["publish-and-triage"]["steps"]
            if s.get("id") == "latest_cursor"
        )
        driver = tmp_path / "discovery_driver.cjs"
        driver.write_text(
            """
const fs = require("node:fs");
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const { script, scenario } = JSON.parse(fs.readFileSync(0, "utf8"));
const calls = [];
const outputs = {};
let transientLeft = Number(scenario.transientCount || 0);
const actions = {
  async listWorkflowRuns(input) {
    if (input.workflow_id === "ppu-pr-ci.yaml") return { data: { total_count: 0 } };
    calls.push(["listRuns", input.page || 1, input.per_page]);
    if (scenario.error === "runs") {
      if (transientLeft > 0) {
        transientLeft -= 1;
        throw new Error("503 transient while listing workflow runs");
      }
      if (!scenario.recover) throw new Error("503 while listing workflow runs");
    }
    const offset = ((input.page || 1) - 1) * input.per_page;
    const slice = scenario.runs.slice(offset, offset + input.per_page).map((x) => {
      const item = { id: x.id };
      if (x.created_at) item.created_at = x.created_at;
      if (x.updated_at) item.updated_at = x.updated_at;
      if (typeof x.run_attempt === "number") item.run_attempt = x.run_attempt;
      return item;
    });
    return { data: { total_count: scenario.runs.length, workflow_runs: slice } };
  },
  async listWorkflowRunArtifacts(input) {
    const page = input.page || 1;
    const perPage = input.per_page || 100;
    calls.push(["listArtifacts", input.run_id, page]);
    if (input.run_id === scenario.error) {
      if (transientLeft > 0) {
        transientLeft -= 1;
        throw new Error("503 transient while listing run artifacts");
      }
      if (!scenario.recover) throw new Error("503 while listing run artifacts");
    }
    const run = scenario.runs.find((x) => x.id === input.run_id);
    const allArts = (run && run.artifacts) || [];
    const offset = (page - 1) * perPage;
    return {
      data: {
        total_count: allArts.length,
        artifacts: allArts.slice(offset, offset + perPage),
      },
    };
  },
};
new AsyncFunction("github", "context", "core", "require", script)(
  { rest: { actions } },
  { repo: { owner: "fixture", repo: "fixture" }, runId: scenario.runId || 9999 },
  { setOutput: (k, v) => { outputs[k] = v; } },
  require
).then(() => {
  process.stdout.write(JSON.stringify({ outputs, calls }));
}).catch((e) => {
  console.error(e && e.message ? e.message : String(e));
  process.exitCode = 1;
});
""".lstrip(),
            encoding="utf-8",
        )

        def _run_js(
            script_src: str,
            runs_data: list[dict],
            *,
            error: object = None,
            transient_count: int = 0,
            recover: bool = False,
            run_id: int = 9999,
            extra_env: dict[str, str] | None = None,
        ) -> subprocess.CompletedProcess[str]:
            work = tmp_path / "js-cwd"
            work.mkdir(exist_ok=True)
            return subprocess.run(
                [node_bin, str(driver)],
                input=json.dumps(
                    {
                        "script": script_src,
                        "scenario": {
                            "runs": runs_data,
                            "error": error,
                            "transientCount": transient_count,
                            "recover": recover,
                            "runId": run_id,
                        },
                    }
                ),
                text=True,
                capture_output=True,
                cwd=work,
                env={
                    **{
                        k: v for k, v in os.environ.items() if k != "GITHUB_RUN_ATTEMPT"
                    },
                    "CURSOR_DISCOVERY_RETRY_MS": "2",
                    **(extra_env or {}),
                },
                check=False,
            )

        def _art(run_id: int, attempt: int, art_id: int) -> dict:
            day = min(art_id, 28)
            return {
                "id": art_id,
                "name": f"upstream-post-merge-cursor-{run_id}-{attempt}",
                "expired": False,
                "created_at": f"2026-11-{day:02d}T{art_id % 24:02d}:00:00Z",
            }

        base_runs = [
            {"id": 12, "artifacts": [_art(12, 1, 12)]},
            {"id": 11, "artifacts": [_art(11, 1, 11)]},
        ]

        # [P1 r4702066251] Persistent 503 on artifact listing or run listing MUST fail closed:
        p_art_503 = _run_js(prior_script, base_runs, error=12)
        assert p_art_503.returncode != 0
        assert "503 while listing run artifacts" in p_art_503.stderr

        p_runs_503 = _run_js(prior_script, base_runs, error="runs")
        assert p_runs_503.returncode != 0
        assert "503 while listing workflow runs" in p_runs_503.stderr

        # Transient single/double 503 recovers cleanly via retry:
        p_rec_art = _run_js(
            prior_script, base_runs, error=12, transient_count=2, recover=True
        )
        assert p_rec_art.returncode == 0, p_rec_art.stderr
        assert (
            json.loads(p_rec_art.stdout)["outputs"]["prior_cursor_artifact"]
            == "upstream-post-merge-cursor-12-1"
        )

        # 20 and 125 cursorless runs (spanning 2 pages of per_page=100) still recover run 12:
        for cursorless_n in (20, 125):
            padded = [
                {"id": 2000 - idx, "artifacts": []} for idx in range(cursorless_n)
            ] + base_runs
            p_pad = _run_js(prior_script, padded)
            assert p_pad.returncode == 0, p_pad.stderr
            out_pad = json.loads(p_pad.stdout)["outputs"]
            assert out_pad["prior_canary_run_id"] == "12"
            assert out_pad["prior_cursor_artifact"] == "upstream-post-merge-cursor-12-1"
            dec_pad = _cli_route(b_fail, "rc_qualify", branch)
            assert dec_pad["last_green_sha"] == sha_a

        # >100 non-cursor artifacts in a single run paginate to page 2:
        noise_arts = [
            {
                "id": 500 + idx,
                "name": f"other-artifact-{idx}",
                "expired": False,
                "created_at": "2026-11-12T00:00:00Z",
            }
            for idx in range(110)
        ] + [_art(12, 1, 12)]
        p_art_pages = _run_js(prior_script, [{"id": 12, "artifacts": noise_arts}])
        assert p_art_pages.returncode == 0, p_art_pages.stderr
        assert (
            json.loads(p_art_pages.stdout)["outputs"]["prior_cursor_artifact"]
            == "upstream-post-merge-cursor-12-1"
        )

        # Genuine first run (0 cursors across successful listings) outputs empty strings:
        p_empty = _run_js(prior_script, [{"id": 50, "artifacts": []}])
        assert p_empty.returncode == 0
        assert json.loads(p_empty.stdout)["outputs"] == {
            "prior_canary_run_id": "",
            "prior_cursor_artifact": "",
        }

        # [P1 r4702066258] Late rerun of older workflow run (run 12 attempt 2 after run 13 attempt 1):
        newer_probe = _cli_publish("intervening-probe", b_pass, "probe", sha_a)
        rerun_fail = _cli_publish(
            "older-run-attempt-2", newer_probe, "rc_qualify", sha_b, branch, "fail"
        )
        runs_late = [
            {"id": 13, "artifacts": [_art(13, 1, 13)]},
            {"id": 12, "artifacts": [_art(12, 2, 14), _art(12, 1, 12)]},
        ]
        p_late = _run_js(prior_script, runs_late)
        assert p_late.returncode == 0, p_late.stderr
        assert json.loads(p_late.stdout)["outputs"] == {
            "prior_canary_run_id": "12",
            "prior_cursor_artifact": "upstream-post-merge-cursor-12-2",
        }
        dec_after_late = _cli_route(rerun_fail, "rc_qualify", branch)
        assert dec_after_late["last_green_sha"] == sha_a
        after_c = json.loads(
            _cli_publish(
                "after-late-c", rerun_fail, "rc_qualify", sha_c, branch
            ).read_text(encoding="utf-8")
        )
        assert sha_b in after_c["rc_branch_states"][branch]["revoked_shas"]
        assert after_c["last_ppu_green_sha"] == sha_a

        # Partial failed-job rerun on run 12 (`route` did not re-run; saw `11-1` while run 13 published `13-1`):
        p_superseded = _run_js(
            latest_script,
            [
                {"id": 13, "artifacts": [_art(13, 1, 13)]},
                {"id": 12, "artifacts": [_art(12, 1, 12)]},
                {"id": 11, "artifacts": [_art(11, 1, 11)]},
            ],
            run_id=12,
            extra_env={
                "ROUTE_PRIOR_CURSOR_ARTIFACT": "upstream-post-merge-cursor-11-1"
            },
        )
        assert p_superseded.returncode == 0, p_superseded.stderr
        assert json.loads(p_superseded.stdout)["outputs"] == {
            "has_newer_external_cursor": "true",
            "latest_canary_run_id": "13",
            "latest_cursor_artifact": "upstream-post-merge-cursor-13-1",
        }
        # A rerun whose route already loaded the latest cursor needs no replacement.
        p_same_ext = _run_js(
            latest_script,
            [
                {"id": 12, "artifacts": [_art(12, 1, 12)]},
                {"id": 11, "artifacts": [_art(11, 1, 11)]},
            ],
            run_id=12,
            extra_env={
                "ROUTE_PRIOR_CURSOR_ARTIFACT": "upstream-post-merge-cursor-12-1"
            },
        )
        assert p_same_ext.returncode == 0
        assert (
            json.loads(p_same_ext.stdout)["outputs"]["has_newer_external_cursor"]
            == "false"
        )

    # [P2 r4702066263] Untagged release branch retains live control revocations across >20 and >256 failures:
    _git("checkout", "-q", "--detach", main_green)
    cur_branch = main_cursor
    for idx in range(25):
        if idx == 0:
            tip_sha = main_green
        else:
            _git("commit", "--allow-empty", "-qm", f"release-tip-{idx}")
            tip_sha = _git("rev-parse", "HEAD")
        _snapshot(tip_sha, {})
        dec_tip = _cli_route(cur_branch, "auto")
        assert (
            dec_tip["selected_gear"] == "rc_qualify"
            and dec_tip["target_sha"] == tip_sha
        )
        assert dec_tip["last_green_sha"] == pin_sha
        cur_branch = _cli_publish(
            f"untagged-fail-{idx}",
            cur_branch,
            "rc_qualify",
            tip_sha,
            dec_tip["rc_tag"],
            "fail",
        )
        cur_model = json.loads(cur_branch.read_text(encoding="utf-8"))
        assert cur_model["rc_tag_states"] == {} and cur_model["tested_rc_tags"] == []
        assert main_green in cur_model["rc_branch_states"][branch]["revoked_shas"]

    _git("commit", "--allow-empty", "-qm", "release-tip-after-25")
    tip_26 = _git("rev-parse", "HEAD")
    _snapshot(tip_26, {})
    dec_26 = _cli_route(cur_branch, "auto")
    assert dec_26["selected_gear"] == "rc_qualify"
    assert dec_26["last_green_sha"] == pin_sha

    # Even after >MAX_EPHEMERAL_REVOKED_SHAS additional ephemeral failures, `main_green` stays revoked:
    cur_obj = json.loads(cur_branch.read_text(encoding="utf-8"))
    for k in range(post_merge.MAX_EPHEMERAL_REVOKED_SHAS + 20):
        cur_obj = post_merge.advance_cursor(
            cur_obj,
            verified_pin=pin_sha,
            selected_gear="rc_qualify",
            target_sha=f"{k + 1:040x}",
            rc_tag=branch,
            status="fail",
            cpu_status="pass",
            ppu_status="fail",
        )
    assert main_green in cur_obj["rc_branch_states"][branch]["revoked_shas"]
    assert (
        len(cur_obj["rc_branch_states"][branch]["revoked_shas"])
        == post_merge.MAX_EPHEMERAL_REVOKED_SHAS + 1
    )
    cur_300 = tmp_path / "cur-after-300.json"
    cur_300.write_text(json.dumps(cur_obj), encoding="utf-8")
    assert _cli_route(cur_300, "auto")["last_green_sha"] == pin_sha

    # Passing requalification of `main_green` clears its revocation cleanly:
    cur_requal = _cli_publish(
        "requalify-main-green", cur_300, "rc_qualify", main_green, branch, "pass"
    )
    req_model = json.loads(cur_requal.read_text(encoding="utf-8"))
    assert main_green not in req_model["rc_branch_states"][branch].get(
        "revoked_shas", ()
    )
    assert _cli_route(cur_requal, "auto")["last_green_sha"] == main_green

    # [R1 P1-1] Late / out-of-order `probe` rerun on older SHA `B` after newer `C` already passed
    # must NOT regress `last_ppu_green_sha`, `probe_history` ordering, or `daily_pin_bump` candidate:
    probe_b_fail_dir = tmp_path / "probe-b-fail"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--upstream-sha",
                sha_b,
                "--verified-pin-file",
                str(pin_file),
                "--sail-sha",
                sail_sha,
                "--public-dir",
                str(probe_b_fail_dir),
                "--status",
                "fail",
                "--cpu-status",
                "pass",
                "--ppu-status",
                "fail",
                "--failure-stage",
                "ppu_probe_smoke",
            ]
        )
        == 0
    )
    probe_c_pass = _cli_publish(
        "probe-c-pass", probe_b_fail_dir / "cursor.json", "probe", sha_c, status="pass"
    )
    probe_b_late_pass = _cli_publish(
        "probe-b-late-pass", probe_c_pass, "probe", sha_b, status="pass"
    )
    late_probe_model = json.loads(probe_b_late_pass.read_text(encoding="utf-8"))
    assert late_probe_model["last_ppu_green_sha"] == sha_c
    assert late_probe_model["last_ppu_tested_sha"] == sha_c
    assert late_probe_model["last_cpu_green_sha"] == sha_c
    assert late_probe_model["last_probe_record"]["upstream_sha"] == sha_c
    assert [r["upstream_sha"] for r in late_probe_model["probe_history"]] == [
        sha_c,
        sha_b,
    ]
    assert late_probe_model["probe_history"][1]["ppu_status"] == "pass"
    assert (
        post_merge.select_pin_bump_candidate(
            late_probe_model["probe_history"], verified_pin=pin_sha
        )
        == sha_c
    )

    # Same protection when `12/1` never uploaded a cursor (`B` absent from `probe_history = [C]`)
    # and `publish-and-triage` passes `--superseded-probe-frontier`:
    c_only_cursor = _cli_publish("probe-c-only", None, "probe", sha_c, status="pass")
    sup_b_dir = tmp_path / "probe-b-superseded"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--upstream-sha",
                sha_b,
                "--verified-pin-file",
                str(pin_file),
                "--sail-sha",
                sail_sha,
                "--public-dir",
                str(sup_b_dir),
                "--status",
                "pass",
                "--cpu-status",
                "pass",
                "--ppu-status",
                "pass",
                "--cursor-in",
                str(c_only_cursor),
                "--superseded-probe-frontier",
            ]
        )
        == 0
    )
    sup_b_model = json.loads((sup_b_dir / "cursor.json").read_text(encoding="utf-8"))
    assert sup_b_model["last_ppu_green_sha"] == sha_c
    assert [r["upstream_sha"] for r in sup_b_model["probe_history"]] == [sha_c, sha_b]
    assert (
        post_merge.select_pin_bump_candidate(
            sup_b_model["probe_history"], verified_pin=pin_sha
        )
        == sha_c
    )

    # Docs-only `C` (`ppu_status="skipped"`) inheriting from `B` rolls back cleanly if late `B` fails:
    b_init_pass = _cli_publish("probe-b-init-pass", None, "probe", sha_b, status="pass")
    c_docs_dir = tmp_path / "probe-c-docs"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--upstream-sha",
                sha_c,
                "--verified-pin-file",
                str(pin_file),
                "--sail-sha",
                sail_sha,
                "--public-dir",
                str(c_docs_dir),
                "--status",
                "pass",
                "--cpu-status",
                "pass",
                "--ppu-status",
                "skipped",
                "--ppu-skip-reason",
                "docs_or_tests_only",
                "--cursor-in",
                str(b_init_pass),
            ]
        )
        == 0
    )
    b_late_fail = _cli_publish(
        "probe-b-late-fail",
        c_docs_dir / "cursor.json",
        "probe",
        sha_b,
        status="fail",
    )
    docs_demoted = json.loads(b_late_fail.read_text(encoding="utf-8"))
    assert docs_demoted["last_ppu_green_sha"] == pin_sha
    assert docs_demoted["last_ppu_status"] == "fail"
    assert docs_demoted["probe_history"][0]["upstream_sha"] == sha_c
    assert docs_demoted["probe_history"][0]["last_ppu_result_status"] == "fail"
    assert (
        post_merge.select_pin_bump_candidate(
            docs_demoted["probe_history"], verified_pin=pin_sha
        )
        is None
    )

    # Legacy cursor with `tested_rc_tags=["v0.35.0rc1"]` and `rc_tag_states={}`
    # re-qualifies unmapped `v0.35.0rc1` and retains `sha_b` revocation across >256 (`300`) failures:
    legacy_seed = tmp_path / "legacy-rc-seed.json"
    legacy_seed.write_text(
        json.dumps(
            {
                "schema": 1,
                "last_cpu_checked_sha": pin_sha,
                "last_cpu_green_sha": pin_sha,
                "last_cpu_status": "pass",
                "last_ppu_tested_sha": pin_sha,
                "last_ppu_green_sha": pin_sha,
                "last_ppu_status": "pass",
                "tested_rc_tags": ["v0.35.0rc1"],
                "rc_tag_states": {},
                "rc_branch_states": {},
                "patch_fingerprints": {},
                "probe_history": [],
            }
        ),
        encoding="utf-8",
    )
    _snapshot(sha_b, {"v0.35.0rc1": sha_b})
    assert _cli_route(legacy_seed, "auto")["target_sha"] == sha_b
    leg_cur = _cli_publish(
        "legacy-b-fail", legacy_seed, "rc_qualify", sha_b, branch, "fail"
    )
    leg_obj = json.loads(leg_cur.read_text(encoding="utf-8"))
    for k in range(300):
        leg_obj = post_merge.advance_cursor(
            leg_obj,
            verified_pin=pin_sha,
            selected_gear="rc_qualify",
            target_sha=f"{k + 1000:040x}",
            rc_tag=branch,
            status="fail",
            cpu_status="pass",
            ppu_status="fail",
        )
    leg_300_path = tmp_path / "legacy-after-300.json"
    leg_300_path.write_text(json.dumps(leg_obj), encoding="utf-8")
    _snapshot(sha_c, {})
    leg_dec = _cli_route(leg_300_path, "auto")
    assert leg_dec["selected_gear"] == "rc_qualify"
    assert leg_dec["target_sha"] == sha_c
    assert leg_dec["last_green_sha"] == pin_sha
    assert (
        sha_b
        in json.loads(leg_300_path.read_text(encoding="utf-8"))["rc_branch_states"][
            branch
        ]["revoked_shas"]
    )


def test_cursor_audit_rev6066758288_invariants(tmp_path: Path) -> None:
    """End-to-end regressions for PR #374 cursor state-machine audit (#6066758288)."""
    wf_jobs = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/upstream-canary.yaml").read_text(
            encoding="utf-8"
        )
    )["jobs"]
    discover_js = next(
        s["with"]["script"] for s in wf_jobs["route"]["steps"] if s.get("id") == "prior"
    )
    latest_js = next(
        s["with"]["script"]
        for s in wf_jobs["publish-and-triage"]["steps"]
        if s.get("id") == "latest_cursor"
    )
    daily_steps = wf_jobs["daily-pin-bump"]["steps"]
    assert not any(
        "needs.route.outputs.target_sha" in str(s.get("env", {})) for s in daily_steps
    )
    assert not any(
        "parents: [context.sha]" in str(s.get("with", {})) for s in daily_steps
    )

    node_bin = shutil.which("node")
    if node_bin:
        harness = tmp_path / "js_audit_harness.cjs"
        harness.write_text(
            """
const fs = require("node:fs");
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const outputs = {};
const actions = {
  async listWorkflowRuns(x) {
    if (x.workflow_id === "ppu-pr-ci.yaml") return { data: { total_count: 0, workflow_runs: [] } };
    const r = payload.records.filter(v => v.status === x.status);
    return { data: { total_count: r.length, workflow_runs: r.map(({ artifacts, ...run }) => run) } };
  },
  async listWorkflowRunArtifacts(x) {
    const r = payload.records.find(q => q.id === x.run_id).artifacts;
    return { data: { total_count: r.length, artifacts: r } };
  },
};
if (payload.attemptJobs !== undefined) {
  actions.listJobsForWorkflowRunAttempt = async () => ({ data: { jobs: payload.attemptJobs } });
}
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
(async () => {
  try {
    await new AsyncFunction("github", "context", "core", "require", payload.script)(
      { rest: { actions } },
      { repo: { owner: "o", repo: "r" }, runId: payload.current, runAttempt: payload.runAttempt || 0 },
      { setOutput: (k, v) => { outputs[k] = v; } },
      require
    );
    console.log(JSON.stringify({ ok: true, outputs }));
  } catch (e) {
    console.log(JSON.stringify({ ok: false, outputs, message: e.message }));
  }
})();
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / "state").mkdir(exist_ok=True)

        def _run_js(
            script: str,
            records: list[dict],
            *,
            current: int,
            prior: str = "",
            run_attempt: int = 0,
            attempt_jobs: list[dict] | None = None,
            gear: str = "rc_qualify",
        ) -> dict:
            inp: dict = {
                "script": script,
                "records": records,
                "current": current,
                "runAttempt": run_attempt,
            }
            if attempt_jobs is not None:
                inp["attemptJobs"] = attempt_jobs
            js_env = {k: v for k, v in os.environ.items() if k != "GITHUB_RUN_ATTEMPT"}
            if run_attempt > 0:
                js_env["GITHUB_RUN_ATTEMPT"] = str(run_attempt)
            js_env.update(
                {
                    "CURSOR_DISCOVERY_RETRY_MS": "0",
                    "ROUTE_PRIOR_CURSOR_ARTIFACT": prior,
                    "SELECTED_GEAR": gear,
                    "RUN_CPU": "true",
                    "RUN_PPU": "true",
                }
            )
            proc = subprocess.run(
                [node_bin, str(harness)],
                input=json.dumps(inp),
                cwd=str(tmp_path),
                env=js_env,
                text=True,
                capture_output=True,
                check=True,
            )
            return json.loads(proc.stdout)

        # 1. Full rerun of active run 12 (`status="in_progress"`, `attempt=2`) restores `12-1`:
        in_prog_records = [
            {
                "id": 12,
                "created_at": "2026-11-26T00:12:00Z",
                "updated_at": "2026-11-26T00:12:00Z",
                "run_attempt": 2,
                "status": "in_progress",
                "artifacts": [
                    {
                        "id": 12,
                        "name": "upstream-post-merge-cursor-12-1",
                        "created_at": "2026-11-26T00:12:00Z",
                        "expired": False,
                    }
                ],
            },
            {
                "id": 11,
                "created_at": "2026-11-26T00:11:00Z",
                "updated_at": "2026-11-26T00:11:00Z",
                "run_attempt": 1,
                "status": "completed",
                "artifacts": [
                    {
                        "id": 11,
                        "name": "upstream-post-merge-cursor-11-1",
                        "created_at": "2026-11-26T00:11:00Z",
                        "expired": False,
                    }
                ],
            },
        ]
        own_res = _run_js(discover_js, in_prog_records, current=12, run_attempt=2)
        assert own_res["ok"] is True
        assert (
            own_res["outputs"]["prior_cursor_artifact"]
            == "upstream-post-merge-cursor-12-1"
        )

        # 2. Publisher-only partial retry (`retested_lanes == 0`) after external `13-1` fails closed:
        partial_records = [
            {
                "id": 13,
                "created_at": "2026-11-26T00:13:00Z",
                "updated_at": "2026-11-26T00:13:00Z",
                "run_attempt": 1,
                "status": "completed",
                "artifacts": [
                    {
                        "id": 13,
                        "name": "upstream-post-merge-cursor-13-1",
                        "created_at": "2026-11-26T00:13:00Z",
                        "expired": False,
                    }
                ],
            },
            *in_prog_records,
        ]
        pub_only_res = _run_js(
            latest_js,
            partial_records,
            current=12,
            prior="upstream-post-merge-cursor-11-1",
            run_attempt=2,
            attempt_jobs=[
                {
                    "name": "Publish allowlisted summary & dual-track triage",
                    "status": "in_progress",
                    "conclusion": None,
                }
            ],
        )
        assert pub_only_res["ok"] is False
        assert "retested_lanes=0" in pub_only_res["message"]

        def completed_job(job_key: str, leaf: str = "") -> dict[str, str]:
            return {
                "name": wf_jobs[job_key]["name"] + leaf,
                "status": "completed",
                "conclusion": "success",
            }

        rc_gates = [
            completed_job("build", " / PPU wheel build"),
            completed_job("cpu-fast-check"),
            completed_job("rc-pin-ut", " / main-verified"),
            completed_job("rc-fast-sanity", " / Fast sanity glm-5.3"),
            completed_job("rc-model-sanity", " / Model sanity & micro-E2E glm-5.3"),
        ]
        for attempt_jobs in (rc_gates[-1:], rc_gates[:2], rc_gates[:2] + rc_gates[3:]):
            partial_res = _run_js(
                latest_js,
                partial_records,
                current=12,
                prior="upstream-post-merge-cursor-11-1",
                run_attempt=2,
                attempt_jobs=attempt_jobs,
            )
            assert partial_res["ok"] is False

        # In the same run and attempt all required gates allow adopting the external parent.
        retested_res = _run_js(
            latest_js,
            partial_records,
            current=12,
            prior="upstream-post-merge-cursor-11-1",
            run_attempt=2,
            attempt_jobs=rc_gates,
        )
        assert retested_res["ok"] is True
        assert retested_res["outputs"]["has_newer_external_cursor"] == "true"
        assert (
            retested_res["outputs"]["latest_cursor_artifact"]
            == "upstream-post-merge-cursor-13-1"
        )

        # Attempt 2 already adopted run 13. Retrying just the publisher must
        # retain that parent even if route's old output still points at run 11.
        self_at_three = {
            **in_prog_records[0],
            "run_attempt": 3,
            "artifacts": [
                *in_prog_records[0]["artifacts"],
                {
                    "id": 14,
                    "name": "upstream-post-merge-cursor-12-2",
                    "created_at": "2026-11-26T00:14:00Z",
                    "expired": False,
                },
            ],
        }
        same_run_retry = _run_js(
            latest_js,
            [partial_records[0], self_at_three, in_prog_records[1]],
            current=12,
            prior="upstream-post-merge-cursor-11-1",
            run_attempt=3,
            attempt_jobs=[],
        )
        assert same_run_retry["ok"] is True
        assert same_run_retry["outputs"]["latest_cursor_artifact"] == (
            "upstream-post-merge-cursor-12-2"
        )
        assert same_run_retry["outputs"]["has_newer_external_cursor"] == "true"

    # Build a real Git DAG: pin -> A -> B -> X1 -> C -> D on `main`, plus `side` off `pin`.
    repo = tmp_path / "upstream-dag"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.org",
        "GIT_COMMITTER_EMAIL": "test@example.org",
    }

    def _mk(label: str) -> str:
        (repo / "src").mkdir(exist_ok=True)
        (repo / "docs").mkdir(exist_ok=True)
        if label not in {"C"}:
            (repo / "src" / "code.py").write_text(f"v = {label!r}\n", encoding="utf-8")
        (repo / "docs" / "note.md").write_text(f"{label}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "src", "docs"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-qm", label], check=True, env=git_env
        )
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()

    pin = _mk("pin")
    sha_a = _mk("A")
    sha_b = _mk("B")
    _mk("X1")
    sha_c = _mk("C")
    sha_d = _mk("D")
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--detach", "-q", pin], check=True
    )
    sha_side = _mk("side")
    subprocess.run(
        ["git", "-C", str(repo), "update-ref", "refs/heads/main", sha_d], check=True
    )

    pin_file = tmp_path / "verified.commit"
    pin_file.write_text(pin + "\n", encoding="utf-8")
    rel_file = tmp_path / "release.tag"
    rel_file.write_text("v0.34.0\n", encoding="utf-8")
    branch = "releases/v0.35.0"

    # 3. Retagged RC tag (`v0.35.0rc0` moved from `side` -> `B`) must re-qualify `B`
    #    and never supply unverified `B` as a green control for `v0.35.0rc1(C)`:
    cur_a = post_merge.advance_cursor(
        None,
        verified_pin=pin,
        selected_gear="probe",
        target_sha=sha_a,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_rc0_side = post_merge.advance_cursor(
        cur_a,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_side,
        rc_tag="v0.35.0rc0",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    retag_dec = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_a,
            "tags": {"v0.35.0rc0": sha_b},
            "release_branches": {branch: sha_b},
        },
        verified_pin=pin,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_rc0_side,
        vllm_repo=repo,
    )
    assert retag_dec["selected_gear"] == "rc_qualify"
    assert retag_dec["target_sha"] == sha_b
    retag_rc1_dec = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_a,
            "tags": {"v0.35.0rc0": sha_b, "v0.35.0rc1": sha_c},
            "release_branches": {branch: sha_c},
        },
        verified_pin=pin,
        release_baseline_tag="v0.34.0",
        requested_gear="auto",
        cursor=cur_rc0_side,
        vllm_repo=repo,
    )
    assert retag_rc1_dec["last_green_sha"] == sha_a

    # 4. Nearest strict first-parent ancestor (`C` wins over `B` even after historical rerun of `B`):
    cur_b_rc = post_merge.advance_cursor(
        cur_a,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag=branch,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_c_rc = post_merge.advance_cursor(
        cur_b_rc,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_c,
        rc_tag=branch,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_b_rerun = post_merge.advance_cursor(
        cur_c_rc,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag=branch,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    chosen_ctrl, chosen_rng = post_merge.resolve_verified_bisect_control(
        repo,
        primary_green=sha_b,
        red_sha=sha_d,
        fallback_greens=[sha_c, sha_a, pin],
    )
    assert chosen_ctrl == sha_c
    assert chosen_rng == [sha_d]
    assert cur_b_rerun["rc_branch_states"][branch]["last_green_sha"] == sha_c

    # 5. Shared CPU gate failure in `rc_qualify` invalidates probe reuse, direct pin bump,
    #    and docs-only inheritance (plus `--force` bypasses `allowlisted_non_ppu_changes`):
    cur_main_b = post_merge.advance_cursor(
        cur_a,
        verified_pin=pin,
        selected_gear="probe",
        target_sha=sha_b,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_rc_cpu_fail = post_merge.advance_cursor(
        cur_main_b,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag=branch,
        status="fail",
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
    )
    reprobe_b = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_b,
            "tags": {},
            "release_branches": {branch: sha_b},
        },
        verified_pin=pin,
        release_baseline_tag="v0.34.0",
        requested_gear="probe",
        cursor=cur_rc_cpu_fail,
        vllm_repo=repo,
    )
    assert reprobe_b["run_cpu"] is True and reprobe_b["run_ppu"] is True
    assert (
        post_merge.route_three_gears(
            ls_remote_data={
                "main_sha": sha_b,
                "tags": {},
                "release_branches": {branch: sha_b},
            },
            verified_pin=pin,
            release_baseline_tag="v0.34.0",
            requested_gear="daily_pin_bump",
            cursor=cur_rc_cpu_fail,
            vllm_repo=repo,
        )["target_sha"]
        == sha_a
    )
    docs_on_revoked = post_merge.route_three_gears(
        ls_remote_data={
            "main_sha": sha_c,
            "tags": {},
            "release_branches": {branch: sha_b},
        },
        verified_pin=pin,
        release_baseline_tag="v0.34.0",
        requested_gear="probe",
        cursor=cur_rc_cpu_fail,
        changed_paths_since_ppu=["docs/note.md"],
        changed_files_base_sha=sha_b,
        vllm_repo=repo,
    )
    assert docs_on_revoked["run_ppu"] is True
    forced_docs = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_c, "tags": {}},
        verified_pin=pin,
        release_baseline_tag="v0.34.0",
        requested_gear="probe",
        cursor=cur_main_b,
        changed_paths_since_ppu=["docs/note.md"],
        changed_files_base_sha=sha_b,
        force=True,
        vllm_repo=repo,
    )
    assert forced_docs["run_ppu"] is True

    # 6. `prepare-pin-bump` revalidates cursor + upstream `main` first-parent membership:
    pin_repo = tmp_path / "sail-pin-repo"
    (pin_repo / ".github").mkdir(parents=True)
    (pin_repo / "csrc" / "upstream").mkdir(parents=True)
    (pin_repo / ".github" / "vllm-main-verified.commit").write_text(
        pin + "\n", encoding="utf-8"
    )
    (pin_repo / "csrc" / "upstream" / "manifest.toml").write_text(
        f'upstream_ref = "{pin}"\n', encoding="utf-8"
    )
    revoked_cur_file = tmp_path / "revoked-cur.json"
    revoked_cur_file.write_text(json.dumps(cur_rc_cpu_fail), encoding="utf-8")
    stale_bump_out = tmp_path / "stale-bump.json"
    assert (
        post_merge.main(
            [
                "prepare-pin-bump",
                "--repo-root",
                str(pin_repo),
                "--target-sha",
                sha_b,
                "--cursor-file",
                str(revoked_cur_file),
                "--vllm-repo",
                str(repo),
                "--output-json",
                str(stale_bump_out),
            ]
        )
        == 0
    )
    assert json.loads(stale_bump_out.read_text(encoding="utf-8"))["changed"] == "false"

    cur_off_main = post_merge.advance_cursor(
        cur_a,
        verified_pin=pin,
        selected_gear="probe",
        target_sha=sha_side,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={
                "main_sha": sha_c,
                "tags": {},
                "release_branches": {branch: sha_side},
            },
            verified_pin=pin,
            release_baseline_tag="v0.34.0",
            requested_gear="daily_pin_bump",
            cursor=cur_off_main,
            vllm_repo=repo,
        )["target_sha"]
        == sha_a
    )
    with pytest.raises(ValueError, match="upstream main first-parent"):
        post_merge.prepare_pin_bump(
            repo_root=pin_repo,
            target_sha=sha_side,
            vllm_repo=repo,
        )

    # 7. Unbound exact-SHA RC failure, >256 revocations, corrupt `null` cursor, and empty unresolved `last_good_sha`:
    cur_unbound = post_merge.advance_cursor(
        cur_main_b,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    assert cur_unbound["rc_sha_states"] == {sha_b: "fail"}
    assert (
        post_merge.route_three_gears(
            ls_remote_data={
                "main_sha": sha_a,
                "tags": {},
                "release_branches": {branch: sha_c},
            },
            verified_pin=pin,
            release_baseline_tag="v0.34.0",
            requested_gear="rc_qualify",
            upstream_ref_override=branch,
            cursor=cur_unbound,
            vllm_repo=repo,
        )["last_green_sha"]
        == sha_a
    )

    null_cur = tmp_path / "null-cursor.json"
    null_cur.write_text("null", encoding="utf-8")
    ls_f = tmp_path / "ls-null.txt"
    ls_f.write_text(f"{sha_a}\trefs/heads/main\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        post_merge.main(
            [
                "route",
                "--ls-remote-file",
                str(ls_f),
                "--verified-pin-file",
                str(pin_file),
                "--release-tag-file",
                str(rel_file),
                "--cursor-file",
                str(null_cur),
            ]
        )

    for bad_ctrl in (sha_d, sha_side):
        bad_pin = tmp_path / "bad-pin.commit"
        bad_pin.write_text(bad_ctrl + "\n", encoding="utf-8")
        out_unres = tmp_path / f"unres-{bad_ctrl[:8]}.json"
        assert (
            post_merge.main(
                [
                    "bisect",
                    "--vllm-repo",
                    str(repo),
                    "--last-green-sha",
                    bad_ctrl,
                    "--verified-pin-file",
                    str(bad_pin),
                    "--rc-tag",
                    branch,
                    "--red-sha",
                    sha_d,
                    "--probe-cmd",
                    "true",
                    "--output-json",
                    str(out_unres),
                ]
            )
            == 0
        )
        unres_payload = json.loads(out_unres.read_text(encoding="utf-8"))
        assert unres_payload["bisect_status"] == "unresolved_history"
        assert unres_payload["last_good_sha"] == ""
        assert unres_payload["first_failing_sha"] == ""

        # [R1 B4] Passing unresolved bisect JSON (`last_good_sha=""`) to `publish` must NOT
        # backfill `last_good_sha` with `verified_pin` (especially when `verified_pin == red`!):
        pub_unres_dir = tmp_path / f"pub-unres-{bad_ctrl[:8]}"
        assert (
            post_merge.main(
                [
                    "publish",
                    "--gear",
                    "rc_qualify",
                    "--status",
                    "fail",
                    "--sail-sha",
                    "f" * 40,
                    "--upstream-sha",
                    sha_d,
                    "--verified-pin-file",
                    str(bad_pin),
                    "--rc-tag",
                    branch,
                    "--cpu-status",
                    "pass",
                    "--ppu-status",
                    "fail",
                    "--failure-stage",
                    "rc_model_sanity",
                    "--bisect-result-json",
                    str(out_unres),
                    "--public-dir",
                    str(pub_unres_dir),
                ]
            )
            == 0
        )
        pub_unres_sum = json.loads(
            (pub_unres_dir / "public-summary.json").read_text(encoding="utf-8")
        )
        assert pub_unres_sum["bisect"]["last_good_sha"] is None
        assert pub_unres_sum["bisect"]["first_failing_sha"] is None

    # [R1 B1] Failed RC tag overwritten by a successful retag (`v0.35.0rc1(B)=fail` -> `v0.35.0rc1(C)=pass`)
    # preserves `B`'s revocation across `daily_pin_bump` and `prepare_pin_bump` until `B` is requalified:
    cur_b1_fail = post_merge.advance_cursor(
        cur_main_b,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="v0.35.0rc1",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    cur_b1_retag_c = post_merge.advance_cursor(
        cur_b1_fail,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_c,
        rc_tag="v0.35.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert sha_b in post_merge._collect_excluded_control_shas(cur_b1_retag_c)
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {"v0.35.0rc1": sha_c}},
            verified_pin=pin,
            release_baseline_tag="v0.34.0",
            requested_gear="daily_pin_bump",
            cursor=cur_b1_retag_c,
            vllm_repo=repo,
        )["target_sha"]
        == sha_a
    )
    (pin_repo / ".github" / "vllm-main-verified.commit").write_text(
        pin + "\n", encoding="utf-8"
    )
    (pin_repo / "csrc" / "upstream" / "manifest.toml").write_text(
        f'upstream_ref = "{pin}"\n', encoding="utf-8"
    )
    assert (
        post_merge.prepare_pin_bump(
            repo_root=pin_repo,
            target_sha=sha_b,
            vllm_repo=repo,
            cursor=cur_b1_retag_c,
        )["changed"]
        == "false"
    )
    cur_b1_requal = post_merge.advance_cursor(
        cur_b1_retag_c,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag="",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert sha_b not in post_merge._collect_excluded_control_shas(cur_b1_requal)

    # [R1 B3] `A pass -> B fail -> C pass -> historical B pass` on RC branch preserves `last_green_sha == C`:
    cur_b3_a = post_merge.advance_cursor(
        cur_a,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_a,
        rc_tag=branch,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_b3_bfail = post_merge.advance_cursor(
        cur_b3_a,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag=branch,
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    cur_b3_cpass = post_merge.advance_cursor(
        cur_b3_bfail,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_c,
        rc_tag=branch,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur_b3_bpass = post_merge.advance_cursor(
        cur_b3_cpass,
        verified_pin=pin,
        selected_gear="rc_qualify",
        target_sha=sha_b,
        rc_tag=branch,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert cur_b3_bpass["rc_branch_states"][branch]["last_green_sha"] == sha_c
    assert cur_b3_bpass["rc_branch_states"][branch]["green_history"] == [
        sha_c,
        sha_b,
        sha_a,
    ]

    # [R2 sub1 B3] Multi-step flip-flops, mixed tag/branch sequences, and tip recovery after old green rerun:
    def _rc_step(
        cur_in: dict,
        sha_val: str,
        ref_val: str,
        ok: bool,
    ) -> dict:
        out = post_merge.advance_cursor(
            json.loads(json.dumps(cur_in)),
            verified_pin=pin,
            selected_gear="rc_qualify",
            target_sha=sha_val,
            rc_tag=ref_val,
            status="pass" if ok else "fail",
            cpu_status="pass",
            ppu_status="pass" if ok else "fail",
        )
        return post_merge._sanitize_cursor(
            json.loads(json.dumps(out)), verified_pin=pin
        )

    # (a) Repeated flip-flop: `A pass -> B fail -> C pass -> B pass -> B fail -> B pass` keeps `C`,
    #     and only rolls back to `B` if `C` itself subsequently fails:
    cur_b3_bfail2 = _rc_step(cur_b3_bpass, sha_b, branch, False)
    assert cur_b3_bfail2["rc_branch_states"][branch]["last_green_sha"] == sha_c
    assert cur_b3_bfail2["rc_branch_states"][branch]["green_history"] == [sha_c, sha_a]
    cur_b3_bpass2 = _rc_step(cur_b3_bfail2, sha_b, branch, True)
    assert cur_b3_bpass2["rc_branch_states"][branch]["last_green_sha"] == sha_c
    assert cur_b3_bpass2["rc_branch_states"][branch]["green_history"] == [
        sha_c,
        sha_b,
        sha_a,
    ]
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_d, "release_branches": {branch: sha_d}},
            verified_pin=pin,
            release_baseline_tag="v0.34.0",
            requested_gear="auto",
            cursor=cur_b3_bpass2,
        )["last_green_sha"]
        == sha_c
    )
    cur_b3_cfail = _rc_step(cur_b3_bpass2, sha_c, branch, False)
    assert cur_b3_cfail["rc_branch_states"][branch]["last_green_sha"] == sha_b

    # (b) Mixed branch & tag sequences (in every combination):
    #     1) `A branch pass -> B branch fail -> C tag pass -> B branch pass -> B branch fail -> B branch pass`
    m1 = _rc_step(cur_a, sha_a, branch, True)
    m1 = _rc_step(m1, sha_b, branch, False)
    m1 = _rc_step(m1, sha_c, "v0.35.0rc1", True)
    assert m1["rc_branch_states"][branch]["last_green_sha"] == sha_c
    m1 = _rc_step(m1, sha_b, branch, True)
    assert m1["rc_branch_states"][branch]["last_green_sha"] == sha_c
    m1 = _rc_step(m1, sha_b, branch, False)
    assert m1["rc_branch_states"][branch]["last_green_sha"] == sha_c
    m1 = _rc_step(m1, sha_b, branch, True)
    assert m1["rc_branch_states"][branch]["last_green_sha"] == sha_c

    #     2) `v0.35.0rc0(B) tag fail -> v0.35.0rc1(C) tag pass -> B branch pass` (empty initial branch state)
    m2 = _rc_step(cur_a, sha_b, "v0.35.0rc0", False)
    m2 = _rc_step(m2, sha_c, "v0.35.0rc1", True)
    m2 = _rc_step(m2, sha_b, branch, True)
    assert m2["rc_branch_states"][branch]["last_green_sha"] == sha_c
    assert m2["rc_branch_states"][branch]["green_history"] == [sha_c, sha_b]

    #     3) `A branch pass -> B branch fail -> C branch pass -> v0.35.0rc0(B) tag pass`
    m3 = _rc_step(cur_a, sha_a, branch, True)
    m3 = _rc_step(m3, sha_b, branch, False)
    m3 = _rc_step(m3, sha_c, branch, True)
    m3 = _rc_step(m3, sha_b, "v0.35.0rc0", True)
    assert m3["rc_branch_states"][branch]["last_green_sha"] == sha_c

    #     4) Retag (`v0.35.0rc1(B) fail -> v0.35.0rc1(C) pass`) followed by manual SHA B pass:
    m4 = _rc_step(cur_a, sha_b, "v0.35.0rc1", False)
    m4 = _rc_step(m4, sha_c, "v0.35.0rc1", True)
    m4 = _rc_step(m4, sha_b, "", True)
    assert m4["rc_branch_states"][branch]["last_green_sha"] == sha_c

    # (c) Tip recovery after old green rerun (`A pass -> B fail -> old A rerun pass -> tip B recovery pass`
    #     and `A pass -> B fail -> old A fail -> old A pass -> tip B recovery pass`):
    rec1 = _rc_step(cur_a, sha_a, branch, True)
    rec1 = _rc_step(rec1, sha_b, branch, False)
    rec1 = _rc_step(rec1, sha_a, branch, True)
    assert rec1["rc_branch_states"][branch]["sha"] == sha_b
    assert rec1["rc_branch_states"][branch]["status"] == "fail"
    assert rec1["rc_branch_states"][branch]["last_green_sha"] == sha_a
    rec1 = _rc_step(rec1, sha_b, branch, True)
    assert rec1["rc_branch_states"][branch]["sha"] == sha_b
    assert rec1["rc_branch_states"][branch]["status"] == "pass"
    assert rec1["rc_branch_states"][branch]["last_green_sha"] == sha_b
    assert rec1["rc_branch_states"][branch]["green_history"] == [sha_b, sha_a]

    rec2 = _rc_step(cur_a, sha_a, branch, True)
    rec2 = _rc_step(rec2, sha_b, branch, False)
    rec2 = _rc_step(rec2, sha_a, branch, False)
    rec2 = _rc_step(rec2, sha_a, branch, True)
    rec2 = _rc_step(rec2, sha_b, branch, True)
    assert rec2["rc_branch_states"][branch]["sha"] == sha_b
    assert rec2["rc_branch_states"][branch]["status"] == "pass"
    assert rec2["rc_branch_states"][branch]["last_green_sha"] == sha_b
    assert rec2["rc_branch_states"][branch]["green_history"] == [sha_b, sha_a]

    for bad_line_order in (
        "not-a-dict",
        [],
        False,
        0,
        "",
        {"releases/v0.35.0": [sha_a]},
        {"v0.35.0": "not-a-list"},
        {"v0.35.0": {}},
        {"v0.35.0": False},
        {"v0.35.0": ["bad-sha"]},
    ):
        with pytest.raises(ValueError):
            post_merge._sanitize_cursor(
                {**cur_a, "rc_line_commit_order": bad_line_order},
                verified_pin=pin,
            )

    for bad_root_cursor in ([], "", False, 0):
        with pytest.raises(ValueError):
            post_merge._sanitize_cursor(bad_root_cursor, verified_pin=pin)  # type: ignore[arg-type]

    # (d) [sub1 R1] Partial `rc_line_commit_order` (`[C, B]` with older `A` in `green_history: [C, A]`)
    #     preserves anchored order `[C, B, A]` instead of prepending `A` ahead of `C`:
    partial_order_raw = {
        **cur_a,
        "rc_branch_states": {
            branch: {
                "sha": sha_c,
                "status": "pass",
                "last_green_sha": sha_c,
                "green_history": [sha_c, sha_a],
                "revoked_shas": [sha_b],
                "superseded_shas": [sha_b],
            }
        },
        "rc_line_commit_order": {"v0.35.0": [sha_c, sha_b]},
    }
    partial_sanitized = post_merge._sanitize_cursor(partial_order_raw, verified_pin=pin)
    assert partial_sanitized["rc_line_commit_order"]["v0.35.0"] == [
        sha_c,
        sha_b,
        sha_a,
    ]
    partial_after_b = _rc_step(partial_sanitized, sha_b, branch, True)
    assert partial_after_b["rc_branch_states"][branch]["last_green_sha"] == sha_c
    assert partial_after_b["rc_branch_states"][branch]["green_history"] == [
        sha_c,
        sha_b,
        sha_a,
    ]
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_d, "release_branches": {branch: sha_d}},
            verified_pin=pin,
            release_baseline_tag="v0.34.0",
            requested_gear="auto",
            cursor=partial_after_b,
            vllm_repo=repo,
        )["last_green_sha"]
        == sha_c
    )

    # (e) [sub1 R1] Stale `last_green_sha=A, green_history=[A]` with passing tip `sha=C` and
    #     `rc_line_commit_order=[C, B, A]` canonicalizes `green_history=[C, A]` and routes `C` on real Git DAG:
    stale_green_raw = {
        **cur_a,
        "rc_branch_states": {
            branch: {
                "sha": sha_c,
                "status": "pass",
                "last_green_sha": sha_a,
                "green_history": [sha_a],
                "revoked_shas": [sha_b],
            }
        },
        "rc_line_commit_order": {"v0.35.0": [sha_c, sha_b, sha_a]},
    }
    stale_sanitized = post_merge._sanitize_cursor(stale_green_raw, verified_pin=pin)
    assert stale_sanitized["rc_branch_states"][branch]["last_green_sha"] == sha_c
    assert stale_sanitized["rc_branch_states"][branch]["green_history"] == [
        sha_c,
        sha_a,
    ]
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_d, "release_branches": {branch: sha_d}},
            verified_pin=pin,
            release_baseline_tag="v0.34.0",
            requested_gear="auto",
            cursor=stale_green_raw,
            vllm_repo=repo,
        )["last_green_sha"]
        == sha_c
    )

    # [R1 B2] `daily-pin-bump` JS writer aborts if `heads/main` moved after `prep`:
    if node_bin:
        daily_writer_js = next(
            s["with"]["script"]
            for s in daily_steps
            if s.get("uses", "").startswith("actions/github-script@")
        )
        writer_driver = tmp_path / "b2_writer_driver.cjs"
        writer_driver.write_text(
            """
const fs = require("node:fs");
const p = JSON.parse(fs.readFileSync(0, "utf8"));
let wroteBranch = false;
const api = {
  git: {
    getRef: async (a) => a.ref === "heads/main" ? { data: { object: { sha: p.liveMain } } } : { data: {} },
    getCommit: async (a) => ({ data: { tree: { sha: "tree-" + a.commit_sha } } }),
    createBlob: async (a) => ({ data: { sha: a.content } }),
    createTree: async () => ({ data: { sha: "tree" } }),
    createCommit: async () => ({ data: { sha: "commit" } }),
    updateRef: async () => { wroteBranch = true; return { data: {} }; },
  },
  pulls: { list: async () => ({ data: [{ number: 1 }] }), update: async () => ({ data: {} }) },
};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const read = (path, fmt) => path === "/tmp/pin-bump.json" ? JSON.stringify(p.meta) : fs.readFileSync(path, fmt);
new AsyncFunction("github", "context", "require", p.script)(
  { rest: api },
  { repo: { owner: "o", repo: "r" }, sha: p.prepSail },
  (id) => id === "fs" ? { readFileSync: read } : require(id)
).then(() => {
  console.log(JSON.stringify({ ok: true, wroteBranch }));
}).catch((e) => {
  console.log(JSON.stringify({ ok: false, wroteBranch, message: e.message }));
});
""".strip()
            + "\n",
            encoding="utf-8",
        )
        race_proc = subprocess.run(
            [node_bin, str(writer_driver)],
            input=json.dumps(
                {
                    "script": daily_writer_js,
                    "prepSail": "1" * 40,
                    "liveMain": "2" * 40,
                    "meta": {
                        "changed": "true",
                        "previous_pin": pin,
                        "new_pin": sha_b,
                        "sail_base_sha": "1" * 40,
                        "title": "bump",
                        "body": "body",
                    },
                }
            ),
            cwd=str(pin_repo),
            env={**os.environ, "SAIL_MAIN_SHA": "1" * 40},
            text=True,
            capture_output=True,
            check=True,
        )
        race_out = json.loads(race_proc.stdout)
        assert race_out["ok"] is False
        assert race_out["wroteBranch"] is False
        assert "heads/main advanced" in race_out["message"]
