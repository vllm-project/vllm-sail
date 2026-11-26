# SPDX-License-Identifier: Apache-2.0
"""Select native wheel candidates for Preview and formal PR CI."""

import json
import os
import re
from functools import partial
from typing import NamedTuple
from urllib.parse import quote

from scripts.ci.ppu_ci_command import TITLE, Controller, GitHub, best_effort
from scripts.ci.ppu_wheel_manifest import (
    FINGERPRINT_COMPAT,
    FINGERPRINT_COMPAT_PATHS,
    NATIVE_INPUTS,
    UPSTREAM_BUILD_INPUTS,
)

NIGHTLY_WHEELS = "nightly-wheels.yaml"
UPSTREAM_DOWNLOAD = "Download upstream vLLM wheel donor"
# The build verifies the full fingerprint, including exact legacy orchestration
# revisions. The Git tree prefilter only needs to reject incompatible inputs.
ORCHESTRATION = {*FINGERPRINT_COMPAT_PATHS, FINGERPRINT_COMPAT}


class Donor(NamedTuple):
    run_id: int
    artifact_id: int
    source: str


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)


def native_input(path):
    return (path in NATIVE_INPUTS and path not in ORCHESTRATION) or path.startswith(
        ("cmake/", "csrc/")
    )


def eligible_run(controller, run):
    run_id, attempt = run.get("id"), run.get("run_attempt")
    return (
        type(run_id) is int
        and 0 < run_id < int(controller.env["GITHUB_RUN_ID"])
        and type(attempt) is int
        and attempt > 0
        and run.get("status") == "completed"
        and (run.get("repository") or {}).get("full_name") == controller.repo
        and (run.get("head_repository") or {}).get("full_name") == controller.repo
        and run.get("head_branch") == controller.default_branch
        and valid_sha(run.get("head_sha"))
    )


def built_artifact(controller, run, source, *, historical):
    run_id, attempt = run["id"], run["run_attempt"]
    leaf = "ppu / build / PPU wheel build" if historical else "build / PPU wheel build"
    build = next(
        (
            job
            for job in controller.pages(
                f"/actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs", limit=1
            )
            if job.get("name") == leaf and job.get("conclusion") == "success"
        ),
        None,
    )
    if not build:
        return None
    steps = build.get("steps") or []
    # Historical donors must have compiled SAIL. Its upstream-only step cannot
    # supply SAIL extensions. Scheduled Nightly donors must have built both wheels.
    downloads = [
        step
        for step in steps
        if step.get("name", "").startswith("Download native wheel donor")
    ]
    if not downloads or any(step.get("conclusion") != "skipped" for step in downloads):
        return None
    if not historical and any(
        step.get("name") == UPSTREAM_DOWNLOAD and step.get("conclusion") != "skipped"
        for step in steps
    ):
        return None
    required = {
        "Verify checked-out revision",
        "Build or repackage wheels",
        "Upload wheels",
    }
    completed = {
        step["name"]: step for step in steps if step.get("conclusion") == "success"
    }
    if not required <= completed.keys():
        return None
    upload = completed["Upload wheels"]
    started, finished = upload.get("started_at"), upload.get("completed_at")
    if not started or not finished:
        return None
    for artifact in controller.pages(
        f"/actions/runs/{run_id}/artifacts", "artifacts", limit=1
    ):
        created, ident = artifact.get("created_at"), artifact.get("id")
        if (
            artifact.get("name") == f"ppu-wheels-{run_id}"
            and artifact.get("expired") is False
            and created
            and started <= created <= finished
            and type(ident) is int
            and ident > 0
        ):
            # Pin the immutable artifact ID as well as the run. A rerun may
            # overwrite the public artifact name after selection.
            return Donor(run_id, ident, source)
    return None


def trusted_formal_source(controller, pr_number, source):
    # Only the REST default branch is trusted. In Preview, GITHUB_SHA points
    # at a PR merge commit, so resolve the default branch independently.
    if on_default_branch(controller, source):
        return True
    pr = controller.api("GET", f"/pulls/{pr_number}")
    merge = pr.get("merge_commit_sha")
    base = pr.get("base") or {}
    if (
        pr.get("merged") is not True
        or not valid_sha(merge)
        or base.get("ref") != controller.default_branch
        or (base.get("repo") or {}).get("full_name") != controller.repo
        or not on_default_branch(controller, merge)
    ):
        return False
    # Squash/rebase merges can change the commit ID. Require the same full tree
    # as the verified main commit, including Python and CI inputs.
    return controller.trees_match(source, merge, lambda _path: True)


def on_default_branch(controller, source):
    branch = quote(controller.default_branch, safe="")
    comparison = controller.api("GET", f"/compare/{source}...{branch}")
    return (
        comparison.get("status") in {"ahead", "identical"}
        and comparison.get("merge_base_commit", {}).get("sha") == source
    )


def upstream_inputs_match(controller, source, candidate):
    paths = (".github/vllm-main-verified.commit", *UPSTREAM_BUILD_INPUTS)
    source_tree = controller.tree_entries(source)
    candidate_tree = controller.tree_entries(candidate)
    return all(
        path in source_tree and source_tree[path] == candidate_tree.get(path)
        for path in paths
    )


def nightly_donors(controller, candidate):
    runs = controller.pages(
        f"/actions/workflows/{NIGHTLY_WHEELS}/runs",
        "workflow_runs",
        limit=1,
        page_size=5,
        branch=controller.default_branch,
        event="schedule",
        status="completed",
    )
    upstream_donor = None
    for run in runs:
        if (
            not eligible_run(controller, run)
            or run.get("event") != "schedule"
            or run.get("path") != f".github/workflows/{NIGHTLY_WHEELS}"
        ):
            continue
        source = run["head_sha"]
        native_match = best_effort(
            partial(controller.trees_match, source, candidate, native_input)
        )
        if native_match or (
            upstream_donor is None
            and best_effort(
                partial(upstream_inputs_match, controller, source, candidate)
            )
        ):
            donor = best_effort(
                partial(built_artifact, controller, run, source, historical=False)
            )
            if donor:
                if native_match:
                    return donor, None
                upstream_donor = donor
    return None, upstream_donor


def historical_donors(controller, candidate):
    def eligible_donor(run):
        pr, source, *_ = TITLE.fullmatch(run["display_title"]).groups()
        if not controller.trees_match(source, candidate, native_input):
            return None
        donor = built_artifact(controller, run, source, historical=True)
        if donor and trusted_formal_source(controller, pr, source):
            return donor
        return None

    donors = []
    for run in controller.find_runs(limit=1, page_size=20):
        if not eligible_run(controller, run) or run.get("conclusion") != "success":
            continue
        donor = best_effort(partial(eligible_donor, run))
        if donor:
            donors.append(donor)
            if len(donors) == 2:
                break
    return donors


def select_donors(controller, candidate):
    if not valid_sha(candidate):
        raise ValueError("Invalid candidate commit")
    nightly, upstream = best_effort(lambda: nightly_donors(controller, candidate)) or (
        None,
        None,
    )
    with open(controller.env["GITHUB_OUTPUT"], "a", buffering=1) as output:

        def write_donor(slot, donor):
            output.write(
                f"run_{slot}={donor.run_id}\nartifact_{slot}={donor.artifact_id}\n"
                f"source_{slot}={donor.source}\n"
            )
            print(f"Wheel donor {slot}: {donor.run_id}", flush=True)

        if nightly:
            write_donor(1, nightly)
        historical = best_effort(lambda: historical_donors(controller, candidate)) or []
        for slot, donor in enumerate(historical, 2 if nightly else 1):
            write_donor(slot, donor)
        if upstream:
            write_donor("upstream", upstream)
        if not nightly and not historical and not upstream:
            print("Native wheel donor candidates: none")


def main():
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    github = GitHub(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_API_URL"])
    select_donors(Controller(github, event), os.environ["CANDIDATE_SHA"])


if __name__ == "__main__":
    main()
