# SPDX-License-Identifier: Apache-2.0
"""Discover PPU operator tests, apply routing overrides, and select cumulative Git diffs using only the standard library."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOARDS = {"ppu10": "OAM-810E", "ppu15": "ZW-M890P"}
ID_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)*"
IMAGE_PATTERN = r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected):
    require(
        isinstance(value, dict) and set(value) == set(expected),
        f"Invalid fields: {expected}",
    )


def strings(value, *, nonempty=False):
    require(isinstance(value, list), "Expected an array")
    require(all(isinstance(x, str) and x for x in value), "Expected nonempty strings")
    require(len(value) == len(set(value)), "Array contains duplicates")
    require(not nonempty or bool(value), "Array must not be empty")
    return value


def safe_path(value, *, pattern=False):
    require(isinstance(value, str) and bool(value), "Path must not be empty")
    require(not value.startswith(("/", "-")), "Path must be relative to the repository")
    require(
        not any(c in value for c in ("\\", "\0", ":", "\n", "\r")),
        "Invalid path character",
    )
    require(
        all(p not in ("", ".", "..") for p in value.rstrip("/").split("/")),
        "Invalid path component",
    )
    require(
        pattern or not any(c in value for c in "*?[]"),
        "Exact paths must not contain glob patterns",
    )
    return value


def sha(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
        "Expected an immutable 40-character SHA",
    )
    return value


def digest_image(value):
    require(
        isinstance(value, str) and re.fullmatch(IMAGE_PATTERN, value),
        "Image must use a complete digest reference",
    )
    return value


def decimal(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value),
        "Expected a positive integer identity string",
    )
    return value


def matches(path, rule):
    return (
        path in rule["paths"]
        or any(path.startswith(p) for p in rule["prefixes"])
        or any(fnmatch.fnmatchcase(path, p) for p in rule["globs"])
    )


def validate_profiles(profiles):
    keys(profiles, BOARDS)
    for board, name in BOARDS.items():
        item = profiles[board]
        keys(item, ("runner_group", "node_selector", "nproc_per_node", "group_count"))
        require(item["runner_group"] == "8-ppu-m890p", "Unknown community PPU runner")
        require(item["node_selector"] == {"board-type": name}, "Invalid board routing")
        for key in ("nproc_per_node", "group_count"):
            require(
                type(item[key]) is int and item[key] == 1,
                "Only one device and one group are currently supported",
            )


def validate_ut_isolation(repo):
    repo = Path(repo)
    paths = set()
    for directory in ("vllm_sail", "tests/e2e", "tests/support"):
        paths.update((repo / directory).rglob("*.py"))
    paths.update(
        repo / p
        for p in ("tests/conftest.py", "tests/__init__.py")
        if (repo / p).is_file()
    )
    for path in sorted(paths):
        package = ".".join(path.relative_to(repo).parent.parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            n.body[0].value
            for n in ast.walk(tree)
            if isinstance(
                n, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            )
            and ast.get_docstring(n, clean=False) is not None
        }
        for node in ast.walk(tree):
            if node in docstrings:
                continue
            references = []
            if isinstance(node, ast.Import):
                references.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                try:
                    module = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                except (ImportError, ValueError) as error:
                    raise ValueError(f"Unable to resolve import: {path}") from error
                references += [module] + [module + "." + a.name for a in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                references.append(node.value.replace("/", "."))
            require(
                not any("tests.ut" in ref for ref in references),
                f"Production and device code must not depend on CPU UT: {path}",
            )


def discover_catalog(config, repo):
    """Add unlisted test files without importing device packages.

    JSON entries retain board/dependency overrides and stable existing IDs.
    New files default to both boards and the common runtime dependencies.
    A file can declare a literal PPU_CI_MIN_DEVICES for resource requirements
    beyond the current single-device runner; exclusions are reported explicitly.
    """
    repo = Path(repo).resolve()
    catalog = config["test_catalog"]
    require(isinstance(catalog, list), "Test catalog must be an array")
    listed = {}
    for item in catalog:
        require(isinstance(item, dict), "Test configuration must be an object")
        listed[safe_path(item.get("test"))] = item
    discovered = set()
    for path in sorted((repo / "tests/e2e").rglob("test_*.py")):
        require(
            path.resolve().is_relative_to(repo) and path.is_file(),
            "Test path escapes the repository",
        )
        relative = path.relative_to(repo).as_posix()
        devices = 1
        declarations = []
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if any(
                isinstance(t, ast.Name) and t.id == "PPU_CI_MIN_DEVICES"
                for t in targets
            ):
                require(
                    isinstance(node.value, ast.Constant)
                    and type(node.value.value) is int
                    and node.value.value > 0,
                    f"{relative}: PPU_CI_MIN_DEVICES must be a positive integer literal",
                )
                declarations.append(node.value.value)
        require(
            len(declarations) <= 1,
            f"{relative}: Duplicate device requirement declaration",
        )
        if declarations:
            devices = declarations[0]
        discovered.add(relative)
        item = listed.get(relative)
        if item is None:
            name = re.sub(r"[^a-z0-9]+", "-", relative[10:-3].lower()).strip("-")
            item = {
                "id": f"auto-{name}-{checksum(relative.encode())[:8]}",
                "test": relative,
                "boards": sorted(config["execution_profiles"]),
                "requires": ["torch", "triton", "vllm", "vllm_sail"],
                "board_overrides": {},
            }
            catalog.append(item)
        # Keep overrides and source IDs even when no current profile can run them.
        # Resource metadata is derived from source, never persisted to the JSON.
        if devices > 1:
            item["min_devices"] = devices
        else:
            item.pop("min_devices", None)
        if not runnable_boards(config, item):
            print(
                f"PPU CI excludes {relative}: requires {devices} devices; "
                "current execution profiles cannot run it",
                file=sys.stderr,
            )
    return discovered


def runnable_boards(config, item):
    return sorted(
        board
        for board, profile in config["execution_profiles"].items()
        if board in item["boards"]
        and profile["nproc_per_node"] >= item.get("min_devices", 1)
    )


def validate_config(config, repo):
    repo = Path(repo).resolve()
    keys(config, ("schema", "selection_rules", "test_catalog", "execution_profiles"))
    require(type(config["schema"]) is int and config["schema"] == 1, "Unknown schema")
    validate_profiles(config["execution_profiles"])
    discovered = discover_catalog(config, repo)
    catalog = config["test_catalog"]
    require(isinstance(catalog, list) and bool(catalog), "Missing test catalog")
    ids, tests = set(), set()
    for item in catalog:
        expected_keys = {"id", "test", "boards", "requires", "board_overrides"}
        if "min_devices" in item:
            expected_keys.add("min_devices")
        keys(item, expected_keys)
        require(
            isinstance(item["id"], str) and re.fullmatch(ID_PATTERN, item["id"]),
            "Invalid test ID",
        )
        require(item["id"] not in ids, "Duplicate test ID")
        ids.add(item["id"])
        path = safe_path(item["test"])
        require(
            path.startswith("tests/e2e/") and path not in tests,
            "Duplicate or invalid test path",
        )
        tests.add(path)
        actual = (repo / path).resolve()
        require(
            actual.is_relative_to(repo) and actual.is_file(),
            "Test path does not exist or escapes the repository",
        )
        boards = set(strings(item["boards"], nonempty=True))
        require(boards <= BOARDS.keys(), "Unknown supported board")
        requires = strings(item["requires"], nonempty=True)
        require(requires == sorted(requires), "Dependencies must be sorted")
        require(
            all(re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", x) for x in requires),
            "Invalid import name",
        )
        required = {"torch", "triton", "vllm", "vllm_sail"}
        if item["id"] in {"gdn-decode", "gdn-prefill", "kda"}:
            required.add("pla")
        require(required <= set(requires), "Missing required dependencies")
        functions = {
            n.name
            for n in ast.walk(ast.parse(actual.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name.startswith("test_")
        }
        require(bool(functions), "Test file contains no test functions")
        overrides = item["board_overrides"]
        require(isinstance(overrides, dict), "Invalid board overrides")
        for function, supported in overrides.items():
            require(
                function in functions, "Override references an unknown test function"
            )
            require(
                set(strings(supported, nonempty=True)) <= boards,
                "Override includes boards unsupported by the file",
            )
        for board in boards:
            require(
                any(board in overrides.get(f, boards) for f in functions),
                "Board has no applicable test functions",
            )
    catalog_directories = {(repo / path).parent for path in tests}
    discovered = {
        path.relative_to(repo).as_posix()
        for directory in catalog_directories
        for path in directory.glob("test_*.py")
    }
    require(
        discovered == tests, f"Test catalog is incomplete: {sorted(discovered ^ tests)}"
    )
    rules = config["selection_rules"]
    keys(rules, ("full", "no_effect", "source_to_tests"))
    for category in ("full", "no_effect"):
        rule = rules[category]
        keys(rule, ("paths", "prefixes", "globs"))
        for kind, values in rule.items():
            for path in strings(values):
                safe_path(path, pattern=(kind == "globs"))
                if kind == "paths":
                    require(
                        (repo / path).is_file()
                        and (repo / path).resolve().is_relative_to(repo),
                        "Exact rule path does not exist or escapes the repository",
                    )
                if kind == "prefixes":
                    require(path.endswith("/"), "Directory prefix must end with /")
                if category == "no_effect":
                    # This bounds the no-effect policy; it is not another test-selection mapping.
                    allowed = (
                        (
                            kind == "paths"
                            and (
                                path
                                in {
                                    "README.md",
                                    "CONTRIBUTING.md",
                                    "AGENTS.md",
                                    "LICENSE",
                                }
                                or (path.startswith("docs/") and path.endswith(".md"))
                            )
                        )
                        or (kind == "prefixes" and path == "tests/ut/")
                        or (kind == "globs" and path == "docs/*.md")
                    )
                    require(allowed, "Unapproved no-effect rule")
    rule_ids = set()
    require(
        isinstance(rules["source_to_tests"], list),
        "Source-to-test rules must be an array",
    )
    for rule in rules["source_to_tests"]:
        keys(rule, ("id", "paths", "test_ids"))
        require(
            isinstance(rule["id"], str) and re.fullmatch(ID_PATTERN, rule["id"]),
            "Invalid rule ID",
        )
        require(rule["id"] not in rule_ids, "Duplicate rule ID")
        rule_ids.add(rule["id"])
        require(
            set(strings(rule["test_ids"], nonempty=True)) <= ids,
            "Rule references an unknown test",
        )
        for path in strings(rule["paths"], nonempty=True):
            safe_path(path)
            require(
                (repo / path).is_file()
                and (repo / path).resolve().is_relative_to(repo),
                "Source path does not exist or escapes the repository",
            )
            require(
                not matches(path, rules["full"]),
                "Source-to-test rule is shadowed by a full-selection rule",
            )
    validate_ut_isolation(repo)
    routed = route_tests(config, sorted(ids))
    pairs = [(test, g["board"]) for g in routed["groups"] for test in g["test_ids"]]
    expected = {(t["id"], b) for t in catalog for b in runnable_boards(config, t)}
    require(
        len(pairs) == len(set(pairs)) and set(pairs) == expected,
        "Incomplete or duplicate routing",
    )


def select_tests(config, paths, *, diff_complete, force_full, environment_matches):
    paths = sorted(set(safe_path(path) for path in paths))
    catalog = config["test_catalog"]
    rules = config["selection_rules"]
    selected, reasons, matched, unmapped = set(), set(), set(), set()
    full = force_full or not diff_complete or not environment_matches
    if force_full:
        reasons.add("forced_full")
    if not diff_complete:
        reasons.add("incomplete_diff")
    if not environment_matches:
        reasons.add("environment_changed")
    by_file = {t["test"]: t["id"] for t in catalog}
    for path in paths:
        if matches(path, rules["full"]):
            full = True
            reasons.add(
                "native_change" if path.startswith("csrc/") else "global_dependency"
            )
            continue
        found = False
        if path in by_file:
            selected.add(by_file[path])
            reasons.add("test_changed")
            found = True
        for rule in rules["source_to_tests"]:
            if path in rule["paths"]:
                selected.update(rule["test_ids"])
                matched.add(rule["id"])
                reasons.add("mapped_source")
                found = True
        if not found and not matches(path, rules["no_effect"]):
            full = True
            unmapped.add(path)
            reasons.add("unmapped_path")
    if full:
        selected = {t["id"] for t in catalog}
    runnable = {t["id"] for t in catalog if runnable_boards(config, t)}
    if selected - runnable:
        reasons.add("insufficient_devices")
    selected &= runnable
    return {
        "mode": "full" if full else "subset" if selected else "none",
        "test_ids": sorted(selected),
        "files": sorted(t["test"] for t in catalog if t["id"] in selected),
        "reasons": sorted(reasons or {"no_device_changes"}),
        "matched_rule_ids": sorted(matched),
        "unmapped_files": sorted(unmapped),
    }


def route_tests(config, test_ids):
    validate_profiles(config["execution_profiles"])
    catalog = {t["id"]: t for t in config["test_catalog"]}
    require(set(test_ids) <= catalog.keys(), "Unknown test ID")
    groups, matrix = [], []
    for test_id in test_ids:
        require(
            bool(catalog[test_id]["boards"])
            and set(catalog[test_id]["boards"]) <= BOARDS.keys(),
            "No valid route",
        )
    for board in sorted(BOARDS):
        ids = sorted(
            t for t in set(test_ids) if board in runnable_boards(config, catalog[t])
        )
        if ids:
            groups.append(
                {
                    "group_id": board + "-0",
                    "board": board,
                    "test_ids": ids,
                    "files": sorted(catalog[t]["test"] for t in ids),
                }
            )
            matrix.append(
                {"group_id": board + "-0", "board": board, "nproc_per_node": 1}
            )
    return {"groups": groups, "matrix": {"include": matrix}}


def collect_changes(repo, base_sha, head_sha, tested_sha):
    sha(tested_sha)
    for value in (base_sha, head_sha):
        if value is not None:
            sha(value)
    result = {
        "schema": 1,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "tested_sha": tested_sha,
        "complete": False,
        "paths": [],
        "reason": "no_pr_context",
    }
    if base_sha is None or head_sha is None:
        return result

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True
        ).stdout

    try:
        # A shallow clone may find a common ancestor without having the full ancestry graph.
        require(
            git("rev-parse", "--is-shallow-repository").strip() == b"false",
            "shallow_repository",
        )
        base = git("merge-base", base_sha, head_sha).decode().strip()
        sha(base)
        paths = set()
        for old, new in ((base, head_sha), (head_sha, tested_sha)):
            raw = git("diff", "--name-only", "-z", "--no-renames", old, new, "--")
            paths.update(safe_path(p.decode("utf-8")) for p in raw.split(b"\0") if p)
        result.update(complete=True, paths=sorted(paths), reason=None)
    except (subprocess.SubprocessError, OSError, UnicodeError, ValueError):
        result["reason"] = "incomplete_diff"
    return result


def resolve_build_image(environment, override):
    return digest_image(override or (environment.get("build") or {}).get("build_image"))


def validate_environment(environment):
    keys(
        environment,
        ("schema", "qualified", "build", "runtime_image", "profiles", "evidence"),
    )
    require(
        type(environment["schema"]) is int and environment["schema"] == 1,
        "Invalid environment schema",
    )
    require(type(environment["qualified"]) is bool, "qualified must be a boolean")
    build = environment["build"]
    if build is not None:
        validate_fingerprint(build)
    if environment["runtime_image"] is not None:
        digest_image(environment["runtime_image"])
    require(
        isinstance(environment["profiles"], dict),
        "Environment profiles must be an object",
    )
    require(
        environment["profiles"].keys() <= BOARDS.keys(), "Unknown environment board"
    )
    for profile in environment["profiles"].values():
        validate_runtime_profile(profile)
    evidence = environment["evidence"]
    require(isinstance(evidence, list), "Evidence must be an array")
    identities = {board: set() for board in BOARDS}
    for record in evidence:
        keys(record, ("board", "run_id", "run_attempt", "tested_sha", "run_url"))
        require(record["board"] in BOARDS, "Unknown evidence board")
        decimal(record["run_id"])
        decimal(record["run_attempt"])
        sha(record["tested_sha"])
        require(
            isinstance(record["run_url"], str)
            and record["run_url"].startswith("https://"),
            "Missing evidence URL",
        )
        identity = (record["run_id"], record["run_attempt"])
        require(
            identity not in identities[record["board"]],
            "Duplicate qualification evidence",
        )
        identities[record["board"]].add(identity)
    if environment["qualified"]:
        require(
            build is not None and environment["runtime_image"] is not None,
            "Qualified environment is missing its image or build",
        )
        require(
            environment["profiles"].keys() == BOARDS.keys(),
            "Qualified environment must include both boards",
        )
        require(
            all(len(runs) >= 3 for runs in identities.values()),
            "Each board requires at least three independent evidence records",
        )
    return environment["qualified"]


def validate_fingerprint(build):
    keys(
        build,
        (
            "sdk_sha256",
            "torch_sha256",
            "pytorch_sail_arch",
            "build_image",
            "vllm_commit",
        ),
    )
    for name in ("sdk_sha256", "torch_sha256"):
        require(
            isinstance(build[name], str) and re.fullmatch(r"[0-9a-f]{64}", build[name]),
            "Incomplete build hash",
        )
    require(
        build["pytorch_sail_arch"] == ["ppu_10", "ppu_15"],
        "Build architectures must include both targets in sorted order",
    )
    digest_image(build["build_image"])
    sha(build["vllm_commit"])


def validate_runtime_profile(profile):
    keys(
        profile,
        ("capability", "sdk_identity", "driver_identity", "packages", "runtime_env"),
    )
    cap = profile["capability"]
    require(
        isinstance(cap, list)
        and len(cap) == 2
        and all(type(n) is int and n >= 0 for n in cap),
        "Invalid capability",
    )
    for name in ("sdk_identity", "driver_identity"):
        require(
            isinstance(profile[name], str)
            and profile[name].strip()
            and profile[name].lower() != "unknown",
            "Missing runtime identity",
        )
    keys(profile["packages"], ("torch", "triton", "pla"))
    require(
        all(
            isinstance(v, str) and v.strip() and v.lower() != "unknown"
            for v in profile["packages"].values()
        ),
        "Missing version",
    )
    require(
        profile["runtime_env"] == {"VLLM_SAIL_USE_PLA": True}, "PLA must be enabled"
    )


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_diagnostics(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        require(separator and key, "Invalid build diagnostics")
        require(key == "status" or key not in result, "Duplicate build field")
        result[key] = value
    return result


def build_fingerprint(
    diagnostics, wheel_manifest, *, tested_sha, expected_vllm_commit, build_image
):
    sha(tested_sha)
    sha(expected_vllm_commit)
    digest_image(build_image)
    require(diagnostics.get("status") == "succeeded", "Build did not succeed")
    for field, package, expected in (
        ("vllm_commit", "vllm", expected_vllm_commit),
        ("vllm_sail_commit", "vllm-sail", tested_sha),
    ):
        require(
            diagnostics.get(field) == expected,
            f"Build diagnostics identity mismatch: {field}",
        )
        require(
            wheel_manifest.get("commits", {}).get(package) == expected,
            f"Wheel identity mismatch: {package}",
        )
    require(
        diagnostics.get("build_image") == build_image, "Observed build image mismatch"
    )
    result = {"build_image": build_image, "vllm_commit": expected_vllm_commit}
    for key in ("sdk_sha256", "torch_sha256"):
        value = diagnostics.get(key, "")
        result[key] = value if re.fullmatch(r"[0-9a-f]{64}", value) else None
    arches = sorted(
        set(filter(None, diagnostics.get("pytorch_sail_arch", "").split(";")))
    )
    result["pytorch_sail_arch"] = arches or None
    return result


def validate_changes(changes):
    keys(
        changes,
        ("schema", "base_sha", "head_sha", "tested_sha", "complete", "paths", "reason"),
    )
    require(
        type(changes["schema"]) is int and changes["schema"] == 1, "Invalid diff schema"
    )
    sha(changes["tested_sha"])
    for name in ("base_sha", "head_sha"):
        if changes[name] is not None:
            sha(changes[name])
    require(type(changes["complete"]) is bool, "Invalid diff completeness flag")
    for path in strings(changes["paths"]):
        safe_path(path)
    if changes["complete"]:
        require(
            changes["base_sha"] is not None
            and changes["head_sha"] is not None
            and changes["reason"] is None,
            "Complete diff is missing revision identities",
        )
    else:
        require(
            changes["reason"] in ("no_pr_context", "incomplete_diff"),
            "Missing diff failure reason",
        )


def make_selection(
    *,
    config_bytes,
    environment_bytes,
    changes,
    diagnostics,
    wheel_manifest,
    build_run_id,
    build_image,
    runtime_image,
    expected_vllm_commit,
    force_full,
    repo=ROOT,
):
    config, environment = json.loads(config_bytes), json.loads(environment_bytes)
    discover_catalog(config, repo)
    qualified = validate_environment(environment)
    validate_changes(changes)
    decimal(build_run_id)
    digest_image(runtime_image)
    fingerprint = build_fingerprint(
        diagnostics,
        wheel_manifest,
        tested_sha=changes["tested_sha"],
        expected_vllm_commit=expected_vllm_commit,
        build_image=build_image,
    )
    matched = (
        qualified
        and environment["build"] == fingerprint
        and environment["runtime_image"] == runtime_image
    )
    selected = select_tests(
        config,
        changes["paths"],
        diff_complete=changes["complete"],
        force_full=force_full,
        environment_matches=matched,
    )
    if not qualified:
        selected["reasons"] = sorted(
            (set(selected["reasons"]) - {"environment_changed"})
            | {"environment_unqualified"}
        )
    return {
        "schema": 1,
        **selected,
        **route_tests(config, selected["test_ids"]),
        **{k: changes[k] for k in ("base_sha", "head_sha", "tested_sha")},
        "changed_files": sorted(changes["paths"]),
        "build_run_id": build_run_id,
        "config_sha256": checksum(config_bytes),
        "environment_sha256": checksum(environment_bytes),
        "build_fingerprint": fingerprint,
        "runtime_image": runtime_image,
    }


def validate_selection(
    selection,
    config_bytes,
    environment_bytes,
    *,
    tested_sha,
    build_run_id,
    build_image,
    runtime_image,
    expected_vllm_commit,
    repo=ROOT,
):
    keys(
        selection,
        (
            "schema",
            "mode",
            "reasons",
            "matched_rule_ids",
            "changed_files",
            "unmapped_files",
            "test_ids",
            "files",
            "groups",
            "matrix",
            "base_sha",
            "head_sha",
            "tested_sha",
            "build_run_id",
            "config_sha256",
            "environment_sha256",
            "build_fingerprint",
            "runtime_image",
        ),
    )
    require(
        type(selection["schema"]) is int and selection["schema"] == 1,
        "Invalid selection schema",
    )
    require(selection["tested_sha"] == sha(tested_sha), "Tested SHA mismatch")
    require(selection["build_run_id"] == decimal(build_run_id), "Build run mismatch")
    require(
        selection["runtime_image"] == digest_image(runtime_image),
        "Runtime image mismatch",
    )
    require(
        selection["config_sha256"] == checksum(config_bytes),
        "Configuration checksum mismatch",
    )
    require(
        selection["environment_sha256"] == checksum(environment_bytes),
        "Environment checksum mismatch",
    )
    config = json.loads(config_bytes)
    discover_catalog(config, repo)
    environment = json.loads(environment_bytes)
    qualified = validate_environment(environment)
    for name in ("base_sha", "head_sha"):
        if selection[name] is not None:
            sha(selection[name])
    for name in (
        "test_ids",
        "files",
        "reasons",
        "matched_rule_ids",
        "changed_files",
        "unmapped_files",
    ):
        require(
            strings(selection[name]) == sorted(selection[name]), f"Not sorted: {name}"
        )
    for path in (
        selection["files"] + selection["changed_files"] + selection["unmapped_files"]
    ):
        safe_path(path)
    routes = route_tests(config, selection["test_ids"])
    require(
        selection["groups"] == routes["groups"]
        and selection["matrix"] == routes["matrix"],
        "Group or matrix mismatch",
    )
    expected_files = sorted(
        t["test"] for t in config["test_catalog"] if t["id"] in selection["test_ids"]
    )
    require(selection["files"] == expected_files, "Test ID or file mismatch")
    mode = selection["mode"]
    require(mode in ("none", "subset", "full"), "Unknown selection mode")
    require(
        mode == "full" or bool(selection["test_ids"]) == (mode != "none"),
        "Selection emptiness does not match its mode",
    )
    if mode == "full":
        require(
            selection["test_ids"]
            == sorted(
                t["id"] for t in config["test_catalog"] if runnable_boards(config, t)
            ),
            "Incomplete full selection",
        )
    fingerprint = selection["build_fingerprint"]
    keys(
        fingerprint,
        (
            "sdk_sha256",
            "torch_sha256",
            "pytorch_sail_arch",
            "build_image",
            "vllm_commit",
        ),
    )
    require(
        fingerprint["build_image"] == digest_image(build_image), "Build image mismatch"
    )
    require(
        fingerprint["vllm_commit"] == sha(expected_vllm_commit), "Upstream SHA mismatch"
    )
    if mode != "full":
        require(
            qualified
            and fingerprint == environment["build"]
            and runtime_image == environment["runtime_image"],
            "Non-full selection requires a qualified environment",
        )
        require(
            selection["base_sha"] is not None and selection["head_sha"] is not None,
            "Non-full selection is missing PR revision identities",
        )
        expected = select_tests(
            config,
            selection["changed_files"],
            diff_complete=True,
            force_full=False,
            environment_matches=True,
        )
        require(
            all(selection[key] == value for key, value in expected.items()),
            "Non-full selection does not match source rules",
        )
    return routes


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def github_outputs(values):
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            for key, value in values.items():
                text = (
                    value
                    if isinstance(value, str)
                    else json.dumps(value, separators=(",", ":"))
                )
                require(
                    "\n" not in text and "\r" not in text,
                    "Multiline output is not allowed",
                )
                stream.write(f"{key}={text}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("changes", "select", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, default=Path.cwd())
        if name == "changes":
            command.add_argument("--base-sha")
            command.add_argument("--head-sha")
            command.add_argument("--tested-sha", required=True)
        else:
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--environment", type=Path, required=True)
        if name != "validate":
            command.add_argument("--output", type=Path, required=True)
        if name == "select":
            for field in ("changes", "build-manifest", "wheel-manifest"):
                command.add_argument("--" + field, type=Path, required=True)
            for field in (
                "build-run-id",
                "build-image",
                "runtime-image",
                "expected-vllm-commit",
            ):
                command.add_argument("--" + field, required=True)
            command.add_argument("--scope", choices=("auto", "full"), required=True)
    args = parser.parse_args()
    if args.command == "changes":
        result = collect_changes(
            args.repo, args.base_sha, args.head_sha, args.tested_sha
        )
    else:
        config_bytes, environment_bytes = (
            args.config.read_bytes(),
            args.environment.read_bytes(),
        )
        validate_config(json.loads(config_bytes), args.repo)
        qualified = validate_environment(json.loads(environment_bytes))
        if args.command == "validate":
            print(json.dumps({"schema_valid": True, "qualified": qualified}))
            return
        changes = json.loads(args.changes.read_bytes())
        require(
            args.wheel_manifest.name == "wheel-manifest.json",
            "Wheel manifest filename mismatch",
        )
        manifest = load_sibling("ppu_wheel_manifest").verify(
            args.wheel_manifest.parent,
            vllm_commit=sha(args.expected_vllm_commit),
            sail_commit=sha(changes["tested_sha"]),
        )
        result = make_selection(
            config_bytes=config_bytes,
            environment_bytes=environment_bytes,
            changes=changes,
            diagnostics=read_diagnostics(args.build_manifest),
            wheel_manifest=manifest,
            build_run_id=args.build_run_id,
            build_image=args.build_image,
            runtime_image=args.runtime_image,
            expected_vllm_commit=args.expected_vllm_commit,
            force_full=args.scope == "full",
            repo=args.repo,
        )
    write_json(args.output, result)
    if args.command == "select":
        github_outputs(
            {
                "mode": result["mode"],
                "matrix": result["matrix"],
                "has_tests": bool(result["groups"]),
            }
        )


if __name__ == "__main__":
    main()
