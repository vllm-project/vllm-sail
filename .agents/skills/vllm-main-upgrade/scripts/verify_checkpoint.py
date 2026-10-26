# SPDX-License-Identifier: Apache-2.0
"""Recheck a checkpoint's raw evidence against an independently frozen policy.

This reads receipts produced by the run's reviewed executor. It never runs tests,
builds wheels, changes source or infers new exclusions. Successful verification
rechecks evidence; it does not manufacture model-answer review or acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root, name):
    require(isinstance(name, str) and name, "Missing artifact path")
    relative = Path(name)
    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == name,
        "Noncanonical artifact path: " + name,
    )
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "Symlink artifact: " + name)
    require(cursor.is_file(), "Missing artifact: " + name)
    return cursor


def load(root, name):
    return json.loads(artifact(root, name).read_text())


def xml_counts(root, reference):
    path = artifact(root, reference["file"])
    require(digest(path) == reference["sha256"], "XML hash mismatch")
    tree = ET.parse(path)
    require(
        not tree.findall(".//failure") and not tree.findall(".//error"),
        "Failed XML: " + reference["file"],
    )
    cases = tree.findall(".//testcase")
    skipped = [case for case in cases if case.find("skipped") is not None]
    require(
        len(cases) - len(skipped) == reference["passed"]
        and len(skipped) == reference["skipped"],
        "Unexpected XML counts",
    )
    return skipped


def _verify_schema_parser(root, policy, post, bound):
    reference = policy.get("schema_parser")
    require(isinstance(reference, dict), "Schema parser receipt required")
    require(
        {"file", "remote_key", "schemas", "schema_sha256"} <= reference.keys(),
        "Incomplete schema parser reference",
    )
    require(
        type(reference["schemas"]) is int and reference["schemas"] > 0,
        "Invalid schema parser count",
    )
    require(
        isinstance(reference["schema_sha256"], str)
        and re.fullmatch("[a-f0-9]{64}", reference["schema_sha256"]),
        "Invalid schema parser resource hash",
    )
    name, remote = reference["file"], reference["remote_key"]
    require(
        policy["postproof_artifacts"].get(remote) == name,
        "Schema parser postproof mapping mismatch",
    )
    proof = bound(name)
    require(
        proof.get("status") == "all-generated-schemas-real-Torch-parser-passed",
        "Schema parser did not pass",
    )
    require(proof.get("commits") == policy["commits"], "Schema parser commit mismatch")
    require(
        type(proof.get("schemas")) is int and proof["schemas"] == reference["schemas"],
        "Schema parser count mismatch",
    )
    require(
        proof.get("schema_sha256") == reference["schema_sha256"],
        "Schema parser resource mismatch",
    )
    require(
        post["artifact_hashes"].get(remote) == digest(artifact(root, name)),
        "Schema parser raw receipt mismatch",
    )
    require(
        proof.get("wheel_manifest_sha256")
        == digest(artifact(root, policy["wheel_manifest"])),
        "Schema parser wheel mismatch",
    )


def verify(root, policy):
    require(
        root.is_absolute() and root.is_dir() and not root.is_symlink(),
        "Use an absolute evidence root",
    )
    require(policy["schema"] == 1, "Unsupported policy schema")
    commits = policy["commits"]
    require(
        set(commits) == {"vllm", "vllm-sail"}
        and all(re.fullmatch("[a-f0-9]{40}", value) for value in commits.values()),
        "Exact commit pair required",
    )
    # Hash-check the full policy-selected producer/input/raw artifact set first.
    for name, expected in policy["artifacts"].items():
        require(
            re.fullmatch("[a-f0-9]{64}", expected) is not None,
            "Invalid SHA256 for " + name,
        )
        require(digest(artifact(root, name)) == expected, "Changed artifact: " + name)

    def bound(name):
        require(name in policy["artifacts"], "Unbound receipt: " + name)
        return load(root, name)

    for reference in policy["producer_bindings"]:
        require(reference["producer"] in policy["artifacts"], "Unbound producer")
        require(
            bound(reference["receipt"])[reference["field"]]
            == digest(artifact(root, reference["producer"])),
            "Producer identity mismatch",
        )
    for name in policy["predecessors"]:
        require(name in policy["artifacts"], "Unbound predecessor")
        previous = bound(name)
        require(
            previous["status"] == "accepted" and not previous["pending"],
            "Predecessor incomplete",
        )
    require(
        len(policy["reviews"]) >= 2
        and len(set(policy["reviews"])) == len(policy["reviews"]),
        "Both source and controller review records are required",
    )
    require(
        len(policy["source_proofs"]) == 2
        and {proof["status"] for proof in policy["source_proofs"]}
        == {"clean-exact-source-ut-passed", "passed"},
        "Exact-source and generation proof records are required",
    )
    require(bool(policy["models"]), "An agreed model profile set is required")
    for name in policy["reviews"]:
        require(name in policy["artifacts"], "Unbound review")
        review = bound(name)
        require(review["commits"] == commits, "Review commit identity mismatch")
        for axis in ("spec", "standards"):
            require(
                review[axis]["status"] == "passed"
                and not review[axis]["unresolved_findings"],
                "Review incomplete",
            )
    require(
        {"source", "installed", "device"}.issubset(policy["xml"]),
        "Source, installed and device XML records are required",
    )
    require(
        "skip_reasons" in policy["xml"]["installed"]
        and "excluded_cases" in policy["xml"]["device"],
        "Reviewed installed/device exclusions are required",
    )
    for reference in policy["xml"].values():
        require(reference["file"] in policy["artifacts"], "Unbound XML")
        skipped = xml_counts(root, reference)
        if "skip_reasons" in reference:
            counts = {}
            for case in skipped:
                reason = case.find("skipped").get("message")
                counts[reason] = counts.get(reason, 0) + 1
            require(counts == reference["skip_reasons"], "Unexpected mode exclusion")
        if "excluded_cases" in reference:
            exclusions = {
                case.get("classname") + "::" + case.get("name"): case.find(
                    "skipped"
                ).get("message")
                for case in skipped
            }
            require(
                exclusions == reference["excluded_cases"], "Unexpected device exclusion"
            )
    for version, reference in policy["cpu"].items():
        require(version in ("3.10", "3.11", "3.12", "3.13"), "Unexpected CPU version")
        proof = bound(reference["proof"])
        require(proof["commits"] == commits, "CPU source identity mismatch")
        cpu = proof["cpu"][version]
        require(
            cpu["environment"]
            == {
                "python": version,
                "forbidden_modules": {"torch": False, "vllm": False},
            },
            "CPU-only dependency contract violated",
        )
        raw = policy["xml"][reference["xml"]]
        require(
            cpu["xml_sha256"] == raw["sha256"]
            and cpu["passed"] == raw["passed"]
            and cpu["skipped"] == raw["skipped"],
            "CPU proof/XML mismatch",
        )
    require(
        set(policy["cpu"]) == {"3.10", "3.11", "3.12", "3.13"}, "CPU matrix incomplete"
    )
    for reference in policy["source_proofs"]:
        proof = bound(reference["file"])
        require(
            proof["commits"] == commits and proof["status"] == reference["status"],
            "Source proof identity/status mismatch",
        )
        log_name = (
            policy["source_log"]
            if proof["status"] == "clean-exact-source-ut-passed"
            else policy["generation_log"]
        )
        require(
            log_name in policy["artifacts"]
            and proof["log_sha256"] == digest(artifact(root, log_name)),
            "Source/generation raw log mismatch",
        )
        if proof["status"] == "clean-exact-source-ut-passed":
            raw = policy["xml"]["source"]
            require(
                proof["xml_sha256"] == raw["sha256"]
                and proof["passed"] == raw["passed"]
                and proof["skipped"] == raw["skipped"],
                "Source proof/XML mismatch",
            )
        for field, expected in reference["expected"].items():
            require(proof[field] == expected, "Source proof mismatch: " + field)
    metadata = bound(policy["metadata_review"])
    require(
        metadata["status"] == "runtime-fingerprints-reviewed-not-runtime-acceptance"
        and metadata["vllm_commit"] == commits["vllm"]
        and not metadata["unresolved_targets"],
        "Metadata review incomplete",
    )
    for axis in ("spec", "standards"):
        require(
            metadata["two_axis_review"][axis]["status"] == "passed"
            and not metadata["two_axis_review"][axis]["unresolved_findings"],
            "Metadata review axis incomplete",
        )
    post = bound(policy["postproof"])
    require(
        post["status"] == "runtime-postproof-verified" and post["commits"] == commits,
        "Runtime postproof incomplete",
    )
    require(not post["owned_live_service_handles"], "Owned services still running")
    require(
        post["wheels"] == bound(policy["wheel_manifest"])
        and post["wheels"]["commits"] == commits,
        "Final wheel pair mismatch",
    )
    _verify_schema_parser(root, policy, post, bound)
    for container, expected in policy["stacks"].items():
        actual = post["stack_verification"][container]
        require(
            actual["exit"] == 0
            and actual["sdk_sha256"] == expected["sdk_sha256"]
            and actual["torch_sha256"] == expected["torch_sha256"],
            "Fixed stack changed",
        )
    rust = post["source_build_preparation"]
    require(
        rust["status"] == "official-rust-source-build-verified"
        and rust["vllm_commit"] == commits["vllm"],
        "Rust source identity mismatch",
    )
    for field, expected in policy["rust"].items():
        require(rust[field] == expected, "Rust artifact contract mismatch: " + field)
    installed = bound(policy["installed_files"])
    require(
        digest(artifact(root, policy["installed_files"]))
        == post["installed_files_record_sha256"],
        "Installed-file map changed",
    )
    require(
        set(installed) == set(post["installed_packages"]) == {"vllm", "vllm-sail"},
        "Incomplete installed packages",
    )
    for package, entries in installed.items():
        require(bool(entries), "Empty installed package")
        prefix = "vllm/" if package == "vllm" else "vllm_sail/"
        require(
            all(
                name.startswith(prefix)
                and ".." not in Path(name).parts
                and re.fullmatch("[a-f0-9]{64}", value)
                for name, value in entries.items()
            ),
            "Invalid installed-file map",
        )
        actual = post["installed_packages"][package]
        require(
            actual["checked_package_files"] == len(entries)
            and actual["package_files_sha256"]
            == hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
            "Installed package byte digest mismatch",
        )
        require(
            actual["version"] == post["wheels"]["wheels"][package]["version"],
            "Installed version/wheel mismatch",
        )
        native = {
            name: value
            for name, value in entries.items()
            if name.endswith(".so") or name == "vllm/vllm-rs"
        }
        require(actual["native_or_rust"] == native, "Native byte map mismatch")
        if package == "vllm":
            require(native == rust["rust_members"], "Unexpected Rust/native member")
        else:
            for module in ("_upstream_C", "_upstream_moe_C", "_C", "_moe_C"):
                require(
                    sum(
                        Path(name).name.startswith(module + ".")
                        and name.endswith(".so")
                        for name in native
                    )
                    == 1,
                    "Missing or duplicate native module: " + module,
                )
    for remote, local in policy["postproof_artifacts"].items():
        require(
            post["artifact_hashes"][remote] == digest(artifact(root, local)),
            "Postproof artifact mismatch: " + remote,
        )
    causes = bound(policy["xfail_causes"]["file"])
    require(
        causes["status"] == "verified-known-failures"
        and set(causes["causes"]) == set(policy["xfail_causes"]["cases"]),
        "Expected-failure causes incomplete",
    )
    for key, expected in policy["xfail_causes"]["cases"].items():
        actual = causes["causes"][key]
        require(
            actual["exception"] == expected["exception"]
            and all(
                token in actual["raw_traceback"]
                for token in expected["traceback_contains"]
            ),
            "Expected-failure cause changed: " + key,
        )
    require(set(post["models"]) == set(policy["models"]), "Model set incomplete")
    for case, expected in policy["models"].items():
        audit = bound(expected["audit"])
        require(
            audit["status"] == "passed-model-receipt-and-configured-path-review"
            and audit["case"] == case,
            "Manual model review incomplete",
        )
        require(
            audit["request_budget"] == expected["request_budget"],
            "Model budget changed",
        )
        require(
            [answer["index"] for answer in audit["reviewed_final_answers"]]
            == list(range(5))
            and all(
                answer["finish_reason"] == "stop" and answer["final_answer"].strip()
                for answer in audit["reviewed_final_answers"]
            )
            and audit["sse_final_answer"].strip(),
            "Incomplete model answer review",
        )
        shutdown = audit["shutdown_review"]
        require(
            shutdown["pre_shutdown_error_count"]
            == shutdown["post_shutdown_error_count"]
            == 0
            and shutdown["server_exit"] == 0,
            "Model shutdown error",
        )
        require(
            audit["finished_counts"]["stop"] == 6
            and all(
                value == 0
                for key, value in audit["finished_counts"].items()
                if key != "stop"
            ),
            "Unnatural model finish",
        )
        for metric in expected["positive_metrics"]:
            require(
                audit["metric_totals"][metric] > 0,
                "Configured metric absent: " + metric,
            )
        for label, minimum in expected["evidence_counts"].items():
            require(
                audit["evidence"][label]["count"] >= minimum,
                "Configured model path absent: " + label,
            )
        for label, wanted in expected["exact_evidence_counts"].items():
            require(
                audit["evidence"][label]["count"] == wanted,
                "Model evidence count mismatch: " + label,
            )
        complete = audit["complete_answer_receipt"]
        model = post["models"][case]
        require(model["result"]["state"] == "passed", "Model runner result is failed")
        require(
            model["complete_answer_receipt"] == complete
            and not complete["needs_supplement"]
            and not complete["reasons"]
            and complete["case"] == case
            and complete["max_completion_tokens"] == expected["request_budget"]
            and model["result"]["server_exit_after_cleanup"] == 0,
            "Model receipt/postproof mismatch",
        )
        for name, wanted in audit["files"].items():
            require(
                digest(artifact(root, name)) == wanted, "Changed model raw artifact"
            )
        required_raw = (
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
        audit_raw, post_raw = {}, {}
        for inventory, selected in (
            (audit["files"], audit_raw),
            (model["files"], post_raw),
        ):
            for basename in required_raw:
                matches = [name for name in inventory if Path(name).name == basename]
                require(
                    len(matches) == 1,
                    "Incomplete or duplicate model raw inventory: " + basename,
                )
                selected[basename] = matches[0]
        for basename in required_raw:
            name = audit_raw[basename]
            wanted = audit["files"][name]
            require(
                name in policy["artifacts"]
                and policy["artifacts"][name] == wanted
                and model["files"][post_raw[basename]] == wanted,
                "Unbound or mismatched model raw artifact: " + basename,
            )
        result_files = [
            name for name in audit["files"] if Path(name).name == "result.json"
        ]
        metric_files = [
            name for name in audit["files"] if Path(name).name == "metrics.prom"
        ]
        require(
            len(result_files) == len(metric_files) == 1,
            "Unique raw model result and metrics are required",
        )
        raw_result = load(root, result_files[0])
        require(
            model["result"] == raw_result
            and raw_result["case"] == case
            and raw_result["state"] == "passed"
            and raw_result["metrics_sha256"] == audit["files"][metric_files[0]],
            "Normalized/raw model result mismatch",
        )
        for relative, wanted in model["files"].items():
            matches = [
                name
                for name in audit["files"]
                if Path(name).name == Path(relative).name
            ]
            require(
                len(matches) == 1 and audit["files"][matches[0]] == wanted,
                "Model raw artifact/postproof mismatch",
            )
    accepted = bound(policy["acceptance"])
    require(
        accepted["status"] == "accepted"
        and not accepted["pending"]
        and accepted["commits"] == commits,
        "Checkpoint not accepted",
    )
    require(
        accepted["runtime_postproof"]["sha256"]
        == digest(artifact(root, policy["postproof"]))
        and set(accepted["models"]) == set(policy["models"]),
        "Acceptance/runtime mismatch",
    )
    require(
        accepted["previous_acceptances"]
        == {name: policy["artifacts"][name] for name in policy["predecessors"]},
        "Acceptance predecessor references mismatch",
    )
    for version, reference in policy["cpu"].items():
        require(
            accepted["cpu"][version] == bound(reference["proof"])["cpu"][version],
            "Acceptance CPU reference mismatch",
        )
    for key, summary_name in (
        ("source", "source"),
        ("installed", "installed_ut"),
        ("device", "device_e2e"),
    ):
        raw = policy["xml"][key]
        summary = accepted[summary_name]
        require(
            all(
                summary[field] == raw[field]
                for field in ("file", "sha256", "passed", "skipped")
            ),
            "Acceptance test/XML reference mismatch: " + key,
        )
    for reference in policy["source_proofs"]:
        if reference["status"] == "clean-exact-source-ut-passed":
            require(
                accepted["source"]["receipt_sha256"]
                == digest(artifact(root, reference["file"])),
                "Acceptance source proof hash mismatch",
            )
        if reference["status"] == "passed":
            require(
                accepted["generation"] == bound(reference["file"]),
                "Acceptance generation reference mismatch",
            )
    require(
        accepted["installed_ut"]["mode_exclusions"]
        == policy["xml"]["installed"]["skip_reasons"],
        "Acceptance mode exclusions mismatch",
    )
    expected_failures = len(policy["xfail_causes"]["cases"])
    require(
        accepted["device_e2e"]["known_expected_failures"] == expected_failures
        and accepted["device_e2e"]["hardware_inapplicable"]
        == len(policy["xml"]["device"]["excluded_cases"]) - expected_failures
        and accepted["device_e2e"]["exclusions_are_not_passes"] is True,
        "Acceptance device exclusion classification mismatch",
    )
    require(accepted["wheels"] == post["wheels"], "Acceptance wheel pair mismatch")
    require(
        accepted["runtime_postproof"]["installed_files_sha256"]
        == digest(artifact(root, policy["installed_files"]))
        and accepted["runtime_postproof"]["xfail_causes_sha256"]
        == digest(artifact(root, policy["xfail_causes"]["file"])),
        "Acceptance installed/cause reference mismatch",
    )
    require(
        set(accepted["source_review"])
        == set(policy["reviews"]) | {policy["metadata_review"]},
        "Acceptance review set mismatch",
    )
    for name, wanted in accepted["source_review"].items():
        require(
            name in policy["artifacts"] and digest(artifact(root, name)) == wanted,
            "Acceptance review hash mismatch",
        )
    for case, expected in policy["models"].items():
        require(
            accepted["models"][case]["file"] == expected["audit"],
            "Acceptance model audit path mismatch",
        )
        require(
            accepted["models"][case]["sha256"]
            == digest(artifact(root, expected["audit"])),
            "Acceptance/manual model review mismatch",
        )
    return {
        "status": "existing-checkpoint-evidence-reverified",
        "commits": commits,
        "boundary": "Read-only evidence verification; no new tests, model review or acceptance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args()
    require(digest(args.policy) == args.policy_sha256, "Policy identity mismatch")
    result = verify(args.root, json.loads(args.policy.read_text()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
