# SPDX-License-Identifier: Apache-2.0
"""Exercise the evidence verifier's complete path with synthetic local receipts.

These fixtures are not hardware evidence or accepted project checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

VERIFIER = (
    Path(__file__).resolve().parents[2]
    / ".agents/skills/vllm-main-upgrade/scripts/verify_checkpoint.py"
)

MODEL_RAW_FILES = (
    "result.json",
    "invocation.json",
    "models.json",
    "stream.json",
    "serve.log",
    "metrics.prom",
    "runner-process.txt",
    "owned-process-group.txt",
    *(f"request-{index}.json" for index in range(5)),
)


@pytest.fixture(scope="module")
def evidence_api():
    return runpy.run_path(str(VERIFIER))


def synthetic_receipts(root):
    """Build a complete, self-contained policy without actual run records."""
    root.mkdir()
    commits = {"vllm": "a" * 40, "vllm-sail": "b" * 40}

    def digest(name):
        return hashlib.sha256((root / name).read_bytes()).hexdigest()

    def write(name, data):
        (root / name).write_text(json.dumps(data, sort_keys=True) + "\n")
        return data

    def xml(name, skip=None):
        extra = (
            ""
            if skip is None
            else (
                '<testcase classname="fixture" name="excluded"><skipped message="'
                + skip
                + '"/></testcase>'
            )
        )
        (root / name).write_text(
            '<testsuites><testsuite><testcase classname="fixture" name="passed"/>'
            + extra
            + "</testsuite></testsuites>"
        )
        return {
            "file": name,
            "sha256": digest(name),
            "passed": 1,
            "skipped": int(skip is not None),
        }

    policy = {
        "schema": 1,
        "commits": commits,
        "artifacts": {},
        "producer_bindings": [],
        "predecessors": ["previous.json"],
        "reviews": ["source-review.json", "controls-review.json"],
        "source_proofs": [],
        "models": {},
        "xml": {},
        "cpu": {},
        "source_log": "source.log",
        "generation_log": "generation.log",
        "metadata_review": "metadata-review.json",
        "postproof": "post.json",
        "wheel_manifest": "wheels.json",
        "installed_files": "installed.json",
        "postproof_artifacts": {},
        "acceptance": "acceptance.json",
    }
    write("previous.json", {"status": "accepted", "pending": []})
    for name in policy["reviews"]:
        write(
            name,
            {
                "commits": commits,
                "spec": {"status": "passed", "unresolved_findings": []},
                "standards": {"status": "passed", "unresolved_findings": []},
            },
        )
    cpu = {}
    for version in ("3.10", "3.11", "3.12", "3.13"):
        key = "cpu" + version.replace(".", "")
        raw = policy["xml"][key] = xml(key + ".xml")
        cpu[version] = {
            "passed": 1,
            "skipped": 0,
            "xml_sha256": raw["sha256"],
            "environment": {
                "python": version,
                "forbidden_modules": {"torch": False, "vllm": False},
            },
        }
        policy["cpu"][version] = {"proof": "cpu.json", "xml": key}
    write("cpu.json", {"commits": commits, "cpu": cpu})
    policy["xml"]["source"] = xml("source.xml")
    policy["xml"]["installed"] = dict(
        xml("installed.xml", "known mode"), skip_reasons={"known mode": 1}
    )
    policy["xml"]["device"] = dict(
        xml("device.xml", "known xfail"),
        excluded_cases={"fixture::excluded": "known xfail"},
    )
    (root / "source.log").write_text("synthetic source gate\n")
    (root / "generation.log").write_text("synthetic generation gate\n")
    source = write(
        "source.json",
        {
            "status": "clean-exact-source-ut-passed",
            "commits": commits,
            "passed": 1,
            "skipped": 0,
            "xml_sha256": digest("source.xml"),
            "log_sha256": digest("source.log"),
        },
    )
    generation = write(
        "generation.json",
        {
            "status": "passed",
            "commits": commits,
            "log_sha256": digest("generation.log"),
        },
    )
    policy["source_proofs"] = [
        {"file": "source.json", "status": source["status"], "expected": {}},
        {"file": "generation.json", "status": "passed", "expected": {}},
    ]
    write(
        "metadata-review.json",
        {
            "status": "runtime-fingerprints-reviewed-not-runtime-acceptance",
            "vllm_commit": commits["vllm"],
            "unresolved_targets": [],
            "two_axis_review": {
                a: {"status": "passed", "unresolved_findings": []}
                for a in ("spec", "standards")
            },
        },
    )
    wheels = write(
        "wheels.json",
        {
            "commits": commits,
            "wheels": {
                k: {"version": "synthetic", "sha256": str(i) * 64}
                for i, k in enumerate(("vllm", "vllm-sail"), 1)
            },
        },
    )
    entries = {
        "vllm": {"vllm/vllm-rs": "1" * 64},
        "vllm-sail": {
            "vllm_sail/" + m + ".fixture.so": str(i) * 64
            for i, m in enumerate(("_upstream_C", "_upstream_moe_C", "_C", "_moe_C"), 2)
        },
    }
    write("installed.json", entries)
    rust = {
        "status": "official-rust-source-build-verified",
        "vllm_commit": commits["vllm"],
        "rust_members": entries["vllm"],
        "python_extension_module_names": [],
    }
    policy["rust"] = copy.deepcopy(rust)
    policy["stacks"] = {
        name: {"sdk_sha256": "5" * 64, "torch_sha256": str(i) * 64}
        for i, name in enumerate(("build", "test"), 6)
    }
    write(
        "causes.json",
        {
            "status": "verified-known-failures",
            "causes": {
                "fixture::excluded": {
                    "exception": "FixtureError",
                    "raw_traceback": "known resources",
                }
            },
        },
    )
    policy["xfail_causes"] = {
        "file": "causes.json",
        "cases": {
            "fixture::excluded": {
                "exception": "FixtureError",
                "traceback_contains": ["resources"],
            }
        },
    }
    (root / "metrics.prom").write_text("fixture_tokens 6\n")
    result = write(
        "result.json",
        {
            "case": "fixture-model",
            "state": "passed",
            "server_exit_after_cleanup": 0,
            "metrics_sha256": digest("metrics.prom"),
        },
    )
    complete = {
        "case": "fixture-model",
        "max_completion_tokens": 512,
        "needs_supplement": False,
        "reasons": [],
    }
    write("invocation.json", {"command": ["vllm", "serve", "synthetic-model"]})
    write("models.json", {"data": [{"id": "fixture-model"}]})
    write("stream.json", {"request": {}, "chunks": [{"synthetic": True}]})
    (root / "serve.log").write_text("synthetic server log\n")
    (root / "runner-process.txt").write_text("100\n")
    (root / "owned-process-group.txt").write_text("101\n")
    for index in range(5):
        write(
            f"request-{index}.json",
            {
                "request": {
                    "messages": [{"role": "user", "content": "fixture prompt"}]
                },
                "response": {
                    "choices": [
                        {
                            "message": {"content": "synthetic answer"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            },
        )
    files = {n: digest(n) for n in MODEL_RAW_FILES}
    write(
        "model-audit.json",
        {
            "status": "passed-model-receipt-and-configured-path-review",
            "case": "fixture-model",
            "request_budget": 512,
            "reviewed_final_answers": [
                {
                    "index": i,
                    "finish_reason": "stop",
                    "final_answer": "synthetic answer",
                }
                for i in range(5)
            ],
            "sse_final_answer": "synthetic stream answer",
            "shutdown_review": {
                "pre_shutdown_error_count": 0,
                "post_shutdown_error_count": 0,
                "server_exit": 0,
            },
            "finished_counts": {"stop": 6, "length": 0},
            "metric_totals": {"tokens": 6},
            "evidence": {"parallel": {"count": 8}},
            "complete_answer_receipt": complete,
            "files": files,
        },
    )
    policy["models"]["fixture-model"] = {
        "audit": "model-audit.json",
        "request_budget": 512,
        "positive_metrics": ["tokens"],
        "evidence_counts": {"parallel": 1},
        "exact_evidence_counts": {"parallel": 8},
    }
    write(
        "post.json",
        {
            "status": "runtime-postproof-verified",
            "commits": commits,
            "owned_live_service_handles": [],
            "wheels": wheels,
            "stack_verification": {
                k: dict(v, exit=0) for k, v in policy["stacks"].items()
            },
            "source_build_preparation": rust,
            "installed_files_record_sha256": digest("installed.json"),
            "installed_packages": {
                k: {
                    "checked_package_files": len(v),
                    "version": "synthetic",
                    "package_files_sha256": hashlib.sha256(
                        json.dumps(v, sort_keys=True).encode()
                    ).hexdigest(),
                    "native_or_rust": v,
                }
                for k, v in entries.items()
            },
            "artifact_hashes": {"pin-ut.xml": digest("installed.xml")},
            "models": {
                "fixture-model": {
                    "complete_answer_receipt": complete,
                    "result": result,
                    "files": {"models/fixture-model/" + k: v for k, v in files.items()},
                }
            },
        },
    )
    policy["postproof_artifacts"] = {"pin-ut.xml": "installed.xml"}
    accepted = {
        "status": "accepted",
        "pending": [],
        "commits": commits,
        "runtime_postproof": {
            "sha256": digest("post.json"),
            "installed_files_sha256": digest("installed.json"),
            "xfail_causes_sha256": digest("causes.json"),
        },
        "models": {
            "fixture-model": {
                "file": "model-audit.json",
                "sha256": digest("model-audit.json"),
            }
        },
        "previous_acceptances": {"previous.json": digest("previous.json")},
        "cpu": cpu,
        "source": dict(policy["xml"]["source"], receipt_sha256=digest("source.json")),
        "installed_ut": dict(
            policy["xml"]["installed"], mode_exclusions={"known mode": 1}
        ),
        "device_e2e": dict(
            policy["xml"]["device"],
            known_expected_failures=1,
            hardware_inapplicable=0,
            exclusions_are_not_passes=True,
        ),
        "generation": generation,
        "wheels": wheels,
        "source_review": {
            n: digest(n) for n in [*policy["reviews"], policy["metadata_review"]]
        },
    }
    write("acceptance.json", accepted)
    policy["artifacts"] = {p.name: digest(p.name) for p in root.iterdir()}
    return policy, write, digest


def test_complete_synthetic_evidence_reaches_the_end_read_only(tmp_path, evidence_api):
    root = tmp_path / "evidence"
    policy, _, _ = synthetic_receipts(root)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = evidence_api["verify"](root, policy)
    assert result["status"] == "existing-checkpoint-evidence-reverified"
    assert result["commits"] == policy["commits"]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize(
    "corruption,message",
    [
        ("review-identity", "Review commit identity mismatch"),
        ("empty-reviews", "Both source and controller"),
        ("empty-source-proofs", "Exact-source and generation"),
        ("cpu-dependency", "CPU-only dependency contract"),
        ("shutdown-error", "Model shutdown error"),
        ("normalized-model-failed", "Model runner result is failed"),
        ("raw-model-contradiction", "Normalized/raw model result mismatch"),
        ("acceptance-cpu", "Acceptance CPU reference mismatch"),
        ("acceptance-source", "Acceptance test/XML reference mismatch"),
    ],
)
def test_hash_consistent_semantic_corruption_is_rejected(
    tmp_path, evidence_api, corruption, message
):
    root = tmp_path / "evidence"
    policy, write, digest = synthetic_receipts(root)

    def edit(name, change):
        data = json.loads((root / name).read_text())
        change(data)
        write(name, data)
        policy["artifacts"][name] = digest(name)

    if corruption == "review-identity":
        edit("source-review.json", lambda d: d["commits"].update(vllm="c" * 40))
    elif corruption == "empty-reviews":
        policy["reviews"] = []
    elif corruption == "empty-source-proofs":
        policy["source_proofs"] = []
    elif corruption == "cpu-dependency":
        edit(
            "cpu.json",
            lambda d: d["cpu"]["3.10"]["environment"]["forbidden_modules"].update(
                torch=True
            ),
        )
    elif corruption == "shutdown-error":
        edit(
            "model-audit.json",
            lambda d: d["shutdown_review"].update(post_shutdown_error_count=1),
        )
    elif corruption == "normalized-model-failed":
        edit(
            "post.json",
            lambda d: d["models"]["fixture-model"]["result"].update(state="failed"),
        )
    elif corruption == "raw-model-contradiction":
        edit("result.json", lambda d: d.update(state="failed"))
        edit(
            "model-audit.json",
            lambda d: d["files"].update({"result.json": digest("result.json")}),
        )
        audit = json.loads((root / "model-audit.json").read_text())
        post = json.loads((root / "post.json").read_text())
        post["models"]["fixture-model"]["files"]["models/fixture-model/result.json"] = (
            digest("result.json")
        )
        _rebind_model_inventory(root, policy, write, digest, audit, post)
    elif corruption == "acceptance-cpu":
        edit("acceptance.json", lambda d: d["cpu"]["3.10"].update(passed=999))
    else:
        edit("acceptance.json", lambda d: d["source"].update(passed=999))
    with pytest.raises(evidence_api["InvalidEvidence"], match=message):
        evidence_api["verify"](root, policy)


@pytest.mark.parametrize("conflict", ["missing", "symlink", "changed-bytes"])
def test_artifact_conflicts_fail_before_semantic_checks(
    tmp_path, evidence_api, conflict
):
    root = tmp_path / "evidence"
    policy, _, _ = synthetic_receipts(root)
    path = root / "source.log"
    if conflict == "missing":
        path.unlink()
    elif conflict == "symlink":
        outside = tmp_path / "outside.log"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        path.write_text("changed bytes\n")
    with pytest.raises(evidence_api["InvalidEvidence"]):
        evidence_api["verify"](root, policy)


@pytest.mark.parametrize(
    "missing",
    [name for name in MODEL_RAW_FILES if name not in ("result.json", "metrics.prom")]
    + ["all-except-result-and-metrics"],
)
def test_hash_consistent_missing_model_raw_files_are_rejected(
    tmp_path, evidence_api, missing
):
    root = tmp_path / "evidence"
    policy, write, digest = synthetic_receipts(root)
    audit = json.loads((root / "model-audit.json").read_text())
    post = json.loads((root / "post.json").read_text())
    names = (
        [
            name
            for name in MODEL_RAW_FILES
            if name not in ("result.json", "metrics.prom")
        ]
        if missing == "all-except-result-and-metrics"
        else [missing]
    )
    for name in names:
        audit["files"].pop(name)
        post["models"]["fixture-model"]["files"].pop("models/fixture-model/" + name)
        policy["artifacts"].pop(name)
        (root / name).unlink()
    _rebind_model_inventory(root, policy, write, digest, audit, post)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(evidence_api["InvalidEvidence"], match="model raw"):
        evidence_api["verify"](root, policy)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize(
    "conflict", ["postproof-missing", "policy-unbound", "postproof-duplicate"]
)
def test_model_raw_inventory_requires_unique_cross_bound_files(
    tmp_path, evidence_api, conflict
):
    root = tmp_path / "evidence"
    policy, write, digest = synthetic_receipts(root)
    audit = json.loads((root / "model-audit.json").read_text())
    post = json.loads((root / "post.json").read_text())
    files = post["models"]["fixture-model"]["files"]
    if conflict == "postproof-missing":
        files.pop("models/fixture-model/request-4.json")
    elif conflict == "policy-unbound":
        policy["artifacts"].pop("stream.json")
    else:
        files["duplicate/serve.log"] = digest("serve.log")
    _rebind_model_inventory(root, policy, write, digest, audit, post)
    with pytest.raises(evidence_api["InvalidEvidence"], match="model raw"):
        evidence_api["verify"](root, policy)


def _rebind_model_inventory(root, policy, write, digest, audit, post):
    """Keep every summary/hash consistent so omissions reach the semantic gate."""
    write("model-audit.json", audit)
    write("post.json", post)
    accepted = json.loads((root / "acceptance.json").read_text())
    accepted["models"]["fixture-model"]["sha256"] = digest("model-audit.json")
    accepted["runtime_postproof"]["sha256"] = digest("post.json")
    write("acceptance.json", accepted)
    for name in ("model-audit.json", "post.json", "acceptance.json"):
        policy["artifacts"][name] = digest(name)
