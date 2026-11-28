#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Post-merge upstream vLLM compatibility orchestrator for vLLM SAIL.

Implements the three-gear post-merge cadence without upstream repository hooks
or cross-repository write permissions:
  1. ``probe`` (every 4-6h): SDK-container CPU fast check + sliding-window 1-PPU
     unified Recipe sanity smoke, with control-first binary search bisection on
     ``(last_green_sha..red_sha]`` when red;
  2. ``daily_pin_bump`` (1x daily): promotes the latest probe-verified upstream
     commit to ``.github/vllm-main-verified.commit`` via a normal SAIL PR;
  3. ``rc_qualify`` (on new ``vX.Y.0rc*`` tags and active-RC-window
     ``releases/vX.Y.0`` branch tips): runs the full PPU gate suite
     (wheel build + CPU fast check + pin-UT + Fast Sanity + Model Sanity +
     optional ops qualification).

This module uses only the Python 3.10+ standard library (with ``tomli`` fallback
for Python 3.10) so CPU unit tests run without torch, vLLM, or PPU hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM_REPO = "vllm-project/vllm"
DEFAULT_UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"
DEFAULT_VERIFIED_PIN_FILE = REPO_ROOT / ".github" / "vllm-main-verified.commit"
DEFAULT_RELEASE_TAG_FILE = REPO_ROOT / ".github" / "vllm-release-tag.commit"
DEFAULT_KERNEL_MANIFEST = REPO_ROOT / "csrc" / "upstream" / "manifest.toml"
PROBE_CRON = "23 */5 * * *"
DAILY_PIN_CRON = "41 1 * * *"

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
OPAQUE_PATCH_KEY_RE = re.compile(r"^t_[0-9a-f]{24}$")
IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
RC_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.0rc(\d+)$")
RELEASE_BRANCH_RE = re.compile(r"^releases/v(\d+)\.(\d+)\.0$")
RELEASE_LINE_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?")
SAFE_SYMBOL_RE = re.compile(r"^[A-Za-z0-9_./:<>\[\]()!?,* -]{1,160}$")
PUBLIC_GITHUB_URL_RE = re.compile(
    r"^https://github\.com/vllm-project/(?:vllm-sail|vllm)(?:/[A-Za-z0-9_./#?=&-]*)?$"
)
GITHUB_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
GITHUB_NOREPLY_EMAIL_RE = re.compile(
    r"^(?:\d+\+)?([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})@users\.noreply\.github\.com$",
    re.IGNORECASE,
)

VALID_GEARS = ("auto", "probe", "daily_pin_bump", "rc_qualify")
SELECTED_GEARS = ("probe", "daily_pin_bump", "rc_qualify")
VALID_STATUSES = ("pass", "fail", "skipped", "deferred", "noop", "infra_error")
VALID_CLASSIFICATIONS = (
    "none",
    "infra_error",
    "plugin_interface_break",
    "sail_internal_patch_drift",
)
VALID_STAGES = (
    "none",
    "route",
    "wheel_build",
    "compat_check",
    "plugin_registration",
    "patch_contracts",
    "patch_drift",
    "kernel_port_check",
    "ppu_probe_smoke",
    "rc_pin_ut",
    "rc_fast_sanity",
    "rc_model_sanity",
    "rc_ops_tests",
    "baseline_control",
)
PPU_HARDWARE_STAGES = frozenset(
    {
        "ppu_probe_smoke",
        "rc_pin_ut",
        "rc_fast_sanity",
        "rc_model_sanity",
        "rc_ops_tests",
    }
)
SAIL_INTERNAL_ONLY_STAGES = frozenset(
    {
        "compat_check",
        "patch_contracts",
        "patch_drift",
        "kernel_port_check",
    }
)
VALID_NON_PPU_REPRO_MODES = ("cpu_empty_build", "stub_plugin_cpu")
VALID_REPRO_KINDS = (
    "cpu_empty_build",
    "platform_plugin",
    "register_oot",
    "model_registry",
    "load_general_plugins",
)
STUB_PLUGIN_REPRO_KINDS = frozenset(
    {
        "platform_plugin",
        "register_oot",
        "model_registry",
        "load_general_plugins",
    }
)
VALID_REPRO_PHASES = (
    "none",
    "setup",
    "install",
    "build_install",
    "build_import",
    "stub_check",
    "infra_error",
)
VALID_REPRO_FAILURE_SOURCES = (
    "none",
    "upstream_build_source",
    "stub_plugin_check",
    "upstream_import_check",
    "infra_or_unknown",
)
_INFRA_EXIT_CODES = frozenset({124, 126, 127, 137})
_INSTALL_INFRA_ERROR_RE = re.compile(
    r"(?i)(?:"
    r"could not install packages due to an oserror"
    r"|\boserror\b|\bioerror\b|\bblockingioerror\b|\btimeouterror\b"
    r"|\burllib\.error\b|\burlerror\b|\bhttperror\b"
    r"|\[errno\s+(?!2\b)\d+\]|\berrno\s+(?!2\b)\d+\b"
    r"|no space left on device|disk quota exceeded|input/output error|stale file handle"
    r"|cannot allocate memory|\bmemoryerror\b"
    r"|read-only file system|too many open files|broken pipe"
    r"|permissionerror|permission denied|bus error|segmentation fault"
    r"|can't start new thread|resource temporarily unavailable"
    r"|\[0 lines of output\]"
    r"|temporary failure in name resolution|name or service not known|nodename nor servname"
    r"|connectionerror|connectionreseterror|connectionrefusederror"
    r"|readtimeouterror|connecttimeouterror|newconnectionerror|maxretryerror|timed out"
    r"|sslerror|proxyerror|failed to establish a new connection|urlopen error"
    r"|could not find a version that satisfies"
    r"|no module named ['\"]?pip['\"]?|command not found"
    r"|\bkilled\b|\bsigkill\b"
    r")"
)
_PIP_INTERNAL_TRACEBACK_RE = re.compile(
    r"(?i)(?:"
    r"site-packages[/\\]pip[/\\](?!_vendor[/\\]pyproject_hooks[/\\]_in_process[/\\])"
    r"|pip[/\\]_internal[/\\]"
    r"|pip[/\\]_vendor[/\\](?!pyproject_hooks[/\\]_in_process[/\\])"
    r")"
)
_PIP_WRAPPER_LINE_RE = re.compile(
    r"(?i)^\s*(?:"
    r"[×x✗•│╰├┌─>\s]*(?:"
    r"preparing metadata|getting requirements to build|building wheel for"
    r"|installing build dependencies|running setup\.py|encountered error while"
    r"|exit code:|\[?\d+\s+lines? of output\]?|\[end of output\]"
    r"|see above for output\.?|vllm\s*$"
    r")"
    r"|(?:(?:processing|obtaining|using cached|running command|failed to build)\s+"
    r"|(?:looking in indexes|created temporary directory|installing collected packages|building wheels for collected packages):"
    r"|\[notice\]\s+)"
    r"|note:\s+this (?:error originates from a subprocess|is an issue with the package mentioned above)"
    r"|error:\s+(?:metadata-generation-failed|subprocess-exited-with-error|failed building wheel|legacy-install-failure|could not build wheels\b|failed to build\b|directory\b)"
    r"|hint:\s+"
    r")"
)
_NON_UPSTREAM_PATH_PART_RE = re.compile(
    r"(?i)(?:^|/)(?:"
    r"site-packages|dist-packages|\.?venv|conda|envs"
    r"|lib(?:64)?/(?:python|pypy)\d[\d.]*"
    r"|usr/(?:local/)?lib|library/frameworks/python\.framework"
    r"|pip|setuptools|pkg_resources|wheel|distutils"
    r"|proc|sys|dev|etc"
    r")(?:/|$)"
)
_UPSTREAM_PY_FRAME_RE = re.compile(
    r"(?:^|/)(?:setup\.py|(?:vllm|cmake|csrc|tools)/[A-Za-z0-9_./-]+\.py)$"
)
_SETUPTOOLS_RUN_SETUP_FILE_RE = re.compile(
    r"(?i)(?:^|/)setuptools/(?:build_meta|sandbox)\.py$"
)
_SETUPTOOLS_PYPROJECT_FILE_RE = re.compile(
    r"(?i)(?:^|/)setuptools/config/pyprojecttoml\.py$"
)
_STDLIB_PATHLIB_FRAME_RE = re.compile(r"(?:^|/)pathlib(?:/[A-Za-z0-9_.-]+)?\.py$")
_STDLIB_SUBPROCESS_FRAME_RE = re.compile(r"(?:^|/)subprocess\.py$")
_TRACEBACK_HEADER_RE = re.compile(r"^(\s*)Traceback \(most recent call last\):\s*$")
_TRACEBACK_FILE_RE = re.compile(r'^(\s*)File "([^"]+)", line \d+(?:, in ([^\s]+))?\s*$')
_PYTHON_EXC_LINE_RE = re.compile(
    r"^(\s*)(?:"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)+([A-Z][A-Za-z0-9_]*)"
    r"|((?:[A-Z][A-Za-z0-9_]*)?(?:Error|Exception|ExceptionGroup|Warning|Interrupt|Exit)|Stop(?:Async)?Iteration)"
    r")(?::\s*(.*))?$"
)
_EXCEPTION_GROUP_LINE_RE = re.compile(
    r"^\s*(?:"
    r"\+\s*Exception Group Traceback \(most recent call last\):"
    r"|\+(?:-\+)?-{8,}\s+\d+\s+-{8,}\s*$"
    r"|(?:[|├─└]+\s*)+Traceback \(most recent call last\):"
    r"|(?:[|├─└+]+[\s-]*)?(?:[A-Za-z_][A-Za-z0-9_.]*)?ExceptionGroup(?::|\s*\()"
    r"|(?:[|├─└]+\s+)+(?:(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Z][A-Za-z0-9_]*"
    r"|(?:[A-Z][A-Za-z0-9_]*)?(?:Error|Exception|ExceptionGroup|Warning|Interrupt|Exit)|Stop(?:Async)?Iteration)"
    r"(?::\s+\S|\s*$)"
    r")"
)
_ALLOWED_BUILD_EXC_NAMES = frozenset(
    {
        "SyntaxError",
        "IndentationError",
        "TabError",
        "ImportError",
        "ModuleNotFoundError",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "AttributeError",
        "KeyError",
        "AssertionError",
        "NameError",
        "NotImplementedError",
        "FileNotFoundError",
        "TOMLDecodeError",
        "DistutilsError",
        "SetuptoolsError",
        "InvalidConfigError",
    }
)
_REPO_FILENOENT_TARGET_RE = re.compile(
    r"""['"]?/?(?:[A-Za-z0-9_.-]+/)*(?:requirements/[A-Za-z0-9_.-]+\.(?:txt|in)|setup\.py|pyproject\.toml|CMakeLists\.txt|README\.md|version\.txt|(?:vllm|cmake|csrc)/[A-Za-z0-9_./-]+)['"]?\s*$"""
)
_CMAKE_REPO_ERROR_RE = re.compile(
    r"^\s*CMake Error at (?!/usr/|/opt/|/lib/|/Library/|/System/|/Applications/)(?:[A-Za-z0-9_./-]+/)?(?:CMakeLists\.txt|(?:cmake|csrc|vllm)/[A-Za-z0-9_./-]+):\d+"
)
_NATIVE_COMPILE_REPO_ERROR_RE = re.compile(
    r"^\s*(?!/usr/|/opt/|/lib/|/Library/|/System/|/Applications/)"
    r"(?!.*(?:^|/)(?:site-packages|dist-packages|pip-build-env|\.(?:cargo|rustup))(?:/|$))"
    r"(?:(?!/|[A-Za-z]:)[A-Za-z0-9_./-]+"
    r"|(?:[A-Za-z0-9_./-]+/)?(?:csrc|vllm|rust|pip-req-build-[A-Za-z0-9_]+|wt-[A-Za-z0-9_-]+|vllm-[0-9a-f]{12})/[A-Za-z0-9_./-]+)"
    r"\.(?:c|cc|cpp|cxx|h|hpp|cu):\d+(?::\d+)?:\s*(?:fatal )?error:"
)
_RUSTC_ERROR_CODE_RE = re.compile(r"(?m)^\s*error(?:\[E\d+\])?:")
_RUSTC_REPO_SPAN_RE = re.compile(
    r"(?m)^\s*-->\s+(?!/usr/|/opt/|/cargo/|/Library/|/System/|/Applications/)"
    r"(?!.*(?:^|/)\.(?:cargo|rustup)/)"
    r"(?:(?!/|[A-Za-z]:)[A-Za-z0-9_./-]+"
    r"|(?:[A-Za-z0-9_./-]+/)?(?:rust|src|csrc|vllm)/[A-Za-z0-9_./-]+)"
    r"\.rs:\d+:\d+"
)
_BUILD_ENV_MISSING_MOD_RE = re.compile(
    r"""(?i)\bno module named ['"]?(?:setuptools(?:_rust|_scm)?|wheel|pip|tomli|packaging|cmake|ninja|torch|cython)(?:\.[A-Za-z0-9_]+)*['"]?"""
)
VALID_PPU_SKIP_REASONS = (
    "none",
    "already_tested",
    "allowlisted_non_ppu_changes",
    "deferred_for_sail_pr_ci",
    "cpu_fast_check_failed",
    "not_applicable",
)

# Conservative skip allowlist for PPU smoke coalescing: only pure docs, examples,
# benchmarks, upstream CI metadata, or non-CUDA/non-PPU backend-exclusive files.
SAFE_PPU_SKIP_PATTERNS: tuple[str, ...] = (
    "*.md",
    "docs/*",
    "docs/**",
    "benchmarks/*",
    "benchmarks/**",
    "examples/*",
    "examples/**",
    ".buildkite/*",
    ".buildkite/**",
    ".github/*",
    ".github/**",
    "csrc/rocm/*",
    "csrc/rocm/**",
    "csrc/cpu/*",
    "csrc/cpu/**",
    "csrc/xpu/*",
    "csrc/xpu/**",
    "vllm/platforms/rocm.py",
    "vllm/platforms/tpu.py",
    "vllm/platforms/xpu.py",
    "vllm/platforms/cpu.py",
    "vllm/v1/worker/tpu_*",
    "vllm/v1/worker/xpu_*",
    "vllm/v1/worker/cpu_*",
    "vllm/v1/attention/backends/rocm_*",
    "vllm/v1/attention/backends/pallas*",
    "docker/Dockerfile.rocm*",
    "docker/Dockerfile.tpu*",
    "docker/Dockerfile.xpu*",
    "docker/Dockerfile.cpu*",
    "requirements/rocm*.txt",
    "requirements/tpu*.txt",
    "requirements/xpu*.txt",
    "requirements/cpu*.txt",
)

PUBLIC_PLUGIN_SYMBOLS: tuple[str, ...] = (
    "CustomOp.register_oot",
    "vllm.model_executor.custom_op",
    "register_oot",
    "ModelRegistry.register_model",
    "vllm.model_executor.models.registry",
    "ModelRegistry",
    "vllm.platforms.Platform",
    "vllm.platforms.interface",
    "vllm.platform_plugins",
    "vllm.plugins.load_general_plugins",
    "load_general_plugins",
    "vllm.plugins",
    "AttentionBackend",
    "ServeSubcommand",
    "VLLM_TARGET_DEVICE=empty",
)

_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "internal_domain",
        re.compile(
            r"(?i)(?<![a-z0-9-])(?:[a-z0-9-]+\.)*(?:alibaba-inc\.com|t-head\.cn|alipay\.net|aliyun-inc\.com)(?![a-z0-9-])"
        ),
    ),
    (
        "private_ipv4",
        re.compile(
            r"(?<![0-9.])(?:"
            r"10(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}"
            r"|192\.168(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}"
            r"|172\.(?:1[6-9]|2\d|3[01])(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}"
            r"|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}"
            r"|169\.254(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}"
            r")(?=\.(?:\s|$)|[^0-9.]|$)"
        ),
    ),
    (
        "host_or_nas_path",
        re.compile(
            r"(?:^|[\s'\"=(`\[])(?:"
            r"/nas_aisw(?:/|\b)"
            r"|/mnt/(?:wl_nas|[A-Za-z0-9_.-]+)"
            r"|/wl_nas(?:/|\b)"
            r"|/ppusw(?:/|\b)"
            r"|/Users/[A-Za-z0-9_.-]+"
            r"|/home/[A-Za-z0-9_.-]+"
            r"|/root/"
            r"|/private/(?:tmp|var)\b"
            r"|/var/folders/"
            r"|/tmp/[A-Za-z0-9_.-]+"
            r")"
        ),
    ),
    (
        "credential_or_token",
        re.compile(
            r"(?:\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b"
            r"|\bgithub_pat_[A-Za-z0-9_]{20,}\b"
            r"|\bAKIA[0-9A-Z]{16}\b"
            r"|\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"
            r"|\bQODER_(?:PAT|PERSONAL_ACCESS_TOKEN)\b)"
        ),
    ),
    (
        "confidential_model",
        re.compile(r"(?i)\bstep[-_]?5(?:[-_]?preview)?\b"),
    ),
    (
        "staff_id",
        re.compile(r"(?i)\b(?:wb-[a-z0-9]{4,}|emp[_-]?id\s*[:=]\s*\w+)\b"),
    ),
)


class RedactionViolationError(ValueError):
    """Raised when public output fails mechanical zero-leak validation."""


@dataclass(frozen=True)
class NonPpuReproEvidence:
    """Verified hardware-free (CPU / inline stub plugin) reproduction receipt."""

    mode: str  # "cpu_empty_build" | "stub_plugin_cpu"
    repro_kind: str  # one of VALID_REPRO_KINDS
    last_good_sha: str
    first_failing_sha: str
    last_good_exit_code: int
    first_failing_exit_code: int
    verified: bool = False
    last_good_phase: str = "none"
    first_failing_phase: str = "none"
    last_good_check_executed: bool = False
    first_failing_check_executed: bool = False
    infra_error: bool = False
    failure_source: str = "none"
    observed_signature: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mode": self.mode,
            "repro_kind": self.repro_kind,
            "last_good_sha": self.last_good_sha,
            "first_failing_sha": self.first_failing_sha,
            "last_good_exit_code": self.last_good_exit_code,
            "first_failing_exit_code": self.first_failing_exit_code,
            "last_good_phase": self.last_good_phase,
            "first_failing_phase": self.first_failing_phase,
            "last_good_check_executed": bool(self.last_good_check_executed),
            "first_failing_check_executed": bool(self.first_failing_check_executed),
            "infra_error": bool(self.infra_error),
            "failure_source": self.failure_source,
            "verified": bool(self.verified),
        }
        if self.observed_signature:
            safe_obs = _sanitize_symbol(self.observed_signature, "")
            if safe_obs:
                out["observed_signature"] = safe_obs
        return out


@dataclass(frozen=True)
class ProbeOutcome:
    status: str  # "pass" | "fail" | "infra_error"
    stage: str = "none"
    target_signature: str = ""
    is_public_plugin_interface: bool = False
    detail_code: str = ""
    non_ppu_repro: NonPpuReproEvidence | None = None


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _require_sha1(value: str, label: str = "commit") -> str:
    cleaned = value.strip()
    if not SHA1_RE.fullmatch(cleaned):
        raise ValueError(f"{label} must be a 40-character lowercase hex SHA")
    return cleaned


def extract_image_digest(image_ref: str) -> str:
    """Strip any registry/repository prefix and return only ``sha256:<64hex>``."""
    if not image_ref:
        return "not_recorded"
    match = IMAGE_DIGEST_RE.search(image_ref.strip())
    return match.group(0) if match else "not_recorded"


def assert_zero_sensitive_leaks(
    text: str, *, staging_paths: Sequence[Path] = ()
) -> None:
    """Fail closed if ``text`` contains control chars or sensitive patterns."""

    def _wipe() -> None:
        for path in staging_paths:
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    for ch in text:
        code = ord(ch)
        if code == 0 or 0xD800 <= code <= 0xDFFF:
            _wipe()
            raise RedactionViolationError(
                "public payload contains forbidden NUL or UTF-16 surrogate codepoint"
            )

    for category, pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            _wipe()
            raise RedactionViolationError(
                f"public payload blocked by redaction gate: {category}"
            )


def opaque_target_id(target: str) -> str:
    """Return a deterministic opaque identifier ``t_<24hex>`` for a patch target."""
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:24]
    return f"t_{digest}"


CANONICAL_WHEEL_BUILD_SIGNATURES = frozenset(
    {
        "VLLM_TARGET_DEVICE=empty",
        "rust_frontend_build",
        "vllm_empty_device_build",
    }
)


def _normalize_wheel_build_signature(raw_sig: str) -> str:
    if raw_sig == "vllm_empty_device_build":
        return "VLLM_TARGET_DEVICE=empty"
    return raw_sig


def _sanitize_symbol(value: str, fallback: str = "unspecified") -> str:
    cleaned = value.strip()
    if not cleaned:
        return fallback
    if cleaned in CANONICAL_WHEEL_BUILD_SIGNATURES:
        return cleaned
    # Strip any accidental leading slash paths before checking symbol syntax.
    if cleaned.startswith("/") or ".." in cleaned:
        return fallback
    if not SAFE_SYMBOL_RE.fullmatch(cleaned):
        return fallback
    try:
        assert_zero_sensitive_leaks(cleaned)
    except RedactionViolationError:
        return fallback
    return cleaned


def _sanitize_patch_target_signature(
    value: str, fallback: str = "patch_baseline"
) -> str:
    cleaned = value.strip()
    if not cleaned:
        return fallback
    direct = _sanitize_symbol(cleaned, "")
    if direct:
        return direct
    return f"opaque_target:{opaque_target_id(cleaned)}"


# ---------------------------------------------------------------------------
# 1. Polling (refs/heads/main + refs/heads/releases/v* + refs/tags/v*) & Router
# ---------------------------------------------------------------------------


def _normalize_rc_ref(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("refs/heads/"):
        cand = cleaned.removeprefix("refs/heads/")
        if RELEASE_BRANCH_RE.fullmatch(cand):
            return cand
    elif cleaned.startswith("refs/tags/"):
        cand = cleaned.removeprefix("refs/tags/")
        if RC_TAG_RE.fullmatch(cand):
            return cand
    return cleaned


def _is_valid_rc_ref(value: str) -> bool:
    return bool(RC_TAG_RE.fullmatch(value) or RELEASE_BRANCH_RE.fullmatch(value))


def parse_ls_remote(output: str) -> dict[str, Any]:
    """Parse ``git ls-remote`` output, preferring peeled ``^{}`` commit SHAs."""
    main_sha: str | None = None
    raw_tags: dict[str, str] = {}
    peeled_tags: dict[str, str] = {}
    release_branches: dict[str, str] = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, ref = parts
        if not SHA1_RE.fullmatch(sha):
            continue
        if ref == "refs/heads/main":
            main_sha = sha
        elif ref.startswith("refs/heads/"):
            branch_part = ref.removeprefix("refs/heads/")
            if RELEASE_BRANCH_RE.fullmatch(branch_part):
                release_branches[branch_part] = sha
        elif ref.startswith("refs/tags/"):
            tag_part = ref.removeprefix("refs/tags/")
            if tag_part.endswith("^{}"):
                peeled_tags[tag_part[:-3]] = sha
            else:
                raw_tags[tag_part] = sha

    tags = {**raw_tags, **peeled_tags}
    return {
        "main_sha": main_sha,
        "tags": dict(sorted(tags.items())),
        "release_branches": dict(sorted(release_branches.items())),
    }


def select_newest_rc_tag(
    tags: dict[str, str],
    *,
    release_baseline_tag: str,
    tested_rc_tags: Sequence[str] = (),
    rc_tag_states: dict[str, dict[str, str]] | None = None,
    rc_branch_states: dict[str, dict[str, Any]] | None = None,
    rc_sha_states: dict[str, str] | None = None,
    rc_line_revoked_shas: dict[str, list[str]] | None = None,
    include_failed: bool = False,
) -> tuple[str, str] | None:
    """Return ``(rc_tag, commit_sha)`` for the newest eligible ``vX.Y.0rc*``."""
    base_match = RELEASE_LINE_RE.match(release_baseline_tag.strip())
    if not base_match:
        raise ValueError(f"invalid release baseline tag: {release_baseline_tag!r}")
    base_line = (int(base_match.group(1)), int(base_match.group(2)))
    _ = tested_rc_tags
    states = rc_tag_states or {}
    br_states = rc_branch_states or {}
    sha_states = rc_sha_states or {}
    line_revoked = rc_line_revoked_shas or {}

    candidates: list[tuple[tuple[int, int, int], str, str]] = []
    for tag, sha in tags.items():
        match = RC_TAG_RE.fullmatch(tag)
        if not match or not SHA1_RE.fullmatch(sha):
            continue
        major, minor, rc_num = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if (major, minor) <= base_line or f"v{major}.{minor}.0" in tags:
            continue
        prior = states.get(tag)
        if isinstance(prior, dict) and prior.get("sha") == sha:
            if prior.get("status") == "pass" or not include_failed:
                continue
        if not include_failed:
            if sha in sha_states and sha_states[sha] != "pass":
                continue
            if sha in (line_revoked.get(f"v{major}.{minor}.0") or ()):
                continue
            prior_br = br_states.get(f"releases/v{major}.{minor}.0")
            if isinstance(prior_br, dict):
                br_rev = prior_br.get("revoked_shas") or ()
                if sha in br_rev or (
                    prior_br.get("sha") == sha and prior_br.get("status") != "pass"
                ):
                    continue
        candidates.append(((major, minor, rc_num), tag, sha))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, best_tag, best_sha = candidates[-1]
    return best_tag, best_sha


def select_newest_rc_target(
    tags: dict[str, str],
    release_branches: dict[str, str] | None = None,
    *,
    release_baseline_tag: str,
    tested_rc_tags: Sequence[str] = (),
    rc_tag_states: dict[str, dict[str, str]] | None = None,
    rc_branch_states: dict[str, dict[str, str]] | None = None,
    rc_sha_states: dict[str, str] | None = None,
    rc_line_revoked_shas: dict[str, list[str]] | None = None,
    include_failed: bool = False,
) -> tuple[str, str] | None:
    """Return ``(rc_ref, commit_sha)`` for the newest active-window RC tag or branch.

    Named ``vX.Y.0rcN`` tags take priority over ``releases/vX.Y.0`` branch tips.
    A ``releases/vX.Y.0`` branch is eligible only when ``(X, Y) > base_line``,
    the final ``vX.Y.0`` tag does not yet exist in ``tags``, and its ``tip_sha``
    has not already passed (or failed/errored without ``include_failed``) in
    ``rc_branch_states`` or any SHA-bound same-line ``rc_tag_states``.
    """
    tag_hit = select_newest_rc_tag(
        tags,
        release_baseline_tag=release_baseline_tag,
        tested_rc_tags=tested_rc_tags,
        rc_tag_states=rc_tag_states,
        rc_branch_states=rc_branch_states,
        rc_sha_states=rc_sha_states,
        rc_line_revoked_shas=rc_line_revoked_shas,
        include_failed=include_failed,
    )
    if tag_hit is not None:
        return tag_hit

    base_match = RELEASE_LINE_RE.match(release_baseline_tag.strip())
    if not base_match:
        raise ValueError(f"invalid release baseline tag: {release_baseline_tag!r}")
    base_line = (int(base_match.group(1)), int(base_match.group(2)))
    branches = release_branches or {}
    tag_states = rc_tag_states or {}
    branch_states = rc_branch_states or {}

    branch_candidates: list[tuple[tuple[int, int], str, str]] = []
    for branch, tip_sha in branches.items():
        b_match = RELEASE_BRANCH_RE.fullmatch(branch)
        if not b_match or not SHA1_RE.fullmatch(tip_sha):
            continue
        major, minor = int(b_match.group(1)), int(b_match.group(2))
        if (major, minor) <= base_line or f"v{major}.{minor}.0" in tags:
            continue

        prior_br = branch_states.get(branch)
        br_revoked_set: set[str] = set()
        if isinstance(prior_br, dict):
            raw_rev = prior_br.get("revoked_shas")
            if isinstance(raw_rev, list):
                br_revoked_set.update(
                    str(x).strip()
                    for x in raw_rev
                    if SHA1_RE.fullmatch(str(x or "").strip())
                )
            prior_hist = prior_br.get("green_history")
            if prior_br.get("last_green_sha") == tip_sha or (
                isinstance(prior_hist, list) and tip_sha in prior_hist
            ):
                continue
            if prior_br.get("sha") == tip_sha and (
                prior_br.get("status") == "pass" or not include_failed
            ):
                continue
            if not include_failed and tip_sha in br_revoked_set:
                continue

        covered_by_tag = False
        for t_name, t_entry in tag_states.items():
            t_match = RC_TAG_RE.fullmatch(t_name)
            if (
                not t_match
                or (int(t_match.group(1)), int(t_match.group(2))) != (major, minor)
                or not isinstance(t_entry, dict)
            ):
                continue
            if t_entry.get("sha") == tip_sha and (
                t_entry.get("status") == "pass" or not include_failed
            ):
                covered_by_tag = True
                break
        if covered_by_tag:
            continue

        branch_candidates.append(((major, minor), branch, tip_sha))

    if not branch_candidates:
        return None
    branch_candidates.sort(key=lambda item: item[0])
    _, best_branch, best_sha = branch_candidates[-1]
    return best_branch, best_sha


def _resolve_unique_rc_ref_for_sha(
    target_sha: str,
    *,
    tags: dict[str, str],
    release_branches: dict[str, str],
    release_baseline_tag: str = "",
    rc_branch_states: dict[str, Any] | None = None,
    rc_tag_states: dict[str, Any] | None = None,
    rc_line_revoked_shas: dict[str, Any] | None = None,
    tested_rc_tags: Sequence[str] = (),
) -> str:
    """Resolve a 40-char RC target SHA to its unique release ref, failing closed on ambiguity."""
    clean_sha = _require_sha1(target_sha, "target_sha")
    base_match = (
        RELEASE_LINE_RE.match(release_baseline_tag.strip())
        if release_baseline_tag.strip()
        else None
    )
    base_line = (
        (int(base_match.group(1)), int(base_match.group(2))) if base_match else None
    )
    br_states = rc_branch_states or {}
    tg_states = rc_tag_states or {}
    line_revs = rc_line_revoked_shas or {}
    tested_set = set(tested_rc_tags)

    tracked_lines: set[tuple[int, int]] = set()
    for br_name in br_states:
        bm = RELEASE_BRANCH_RE.fullmatch(br_name)
        if bm:
            tracked_lines.add((int(bm.group(1)), int(bm.group(2))))
    for tg_name in (*tg_states.keys(), *tested_set):
        tm = RC_TAG_RE.fullmatch(tg_name)
        if tm:
            tracked_lines.add((int(tm.group(1)), int(tm.group(2))))
    for lk_name in line_revs:
        lm = RELEASE_LINE_RE.fullmatch(str(lk_name))
        if lm:
            tracked_lines.add((int(lm.group(1)), int(lm.group(2))))

    def _is_eligible_line(maj: int, minr: int) -> bool:
        if (maj, minr) in tracked_lines:
            return True
        if f"v{maj}.{minr}.0" in tags:
            return False
        return base_line is None or (maj, minr) > base_line

    matched_by_line: dict[tuple[int, int], dict[str, Any]] = {}

    def _slot(line_key: tuple[int, int]) -> dict[str, Any]:
        return matched_by_line.setdefault(line_key, {"branch": "", "tags": []})

    for br_name, br_sha in release_branches.items():
        bm = RELEASE_BRANCH_RE.fullmatch(br_name)
        if bm and br_sha == clean_sha:
            maj, minr = int(bm.group(1)), int(bm.group(2))
            if _is_eligible_line(maj, minr):
                _slot((maj, minr))["branch"] = br_name

    for br_name, br_ent in br_states.items():
        bm = RELEASE_BRANCH_RE.fullmatch(br_name)
        if not bm or not isinstance(br_ent, dict):
            continue
        hist = br_ent.get("green_history")
        rev = br_ent.get("revoked_shas")
        if (
            br_ent.get("sha") == clean_sha
            or br_ent.get("last_green_sha") == clean_sha
            or (isinstance(hist, list) and clean_sha in hist)
            or (isinstance(rev, list) and clean_sha in rev)
        ):
            _slot((int(bm.group(1)), int(bm.group(2))))["branch"] = br_name

    for tg_name, tg_sha in tags.items():
        tm = RC_TAG_RE.fullmatch(tg_name)
        if tm and tg_sha == clean_sha:
            maj, minr, rc_num = (
                int(tm.group(1)),
                int(tm.group(2)),
                int(tm.group(3)),
            )
            if _is_eligible_line(maj, minr):
                _slot((maj, minr))["tags"].append((rc_num, tg_name))

    for tg_name, tg_ent in tg_states.items():
        tm = RC_TAG_RE.fullmatch(tg_name)
        if tm and isinstance(tg_ent, dict) and tg_ent.get("sha") == clean_sha:
            maj, minr, rc_num = (
                int(tm.group(1)),
                int(tm.group(2)),
                int(tm.group(3)),
            )
            _slot((maj, minr))["tags"].append((rc_num, tg_name))

    for lk_name, lk_revs in line_revs.items():
        lm = RELEASE_LINE_RE.fullmatch(str(lk_name))
        if lm and isinstance(lk_revs, list) and clean_sha in lk_revs:
            maj, minr = int(lm.group(1)), int(lm.group(2))
            slot = _slot((maj, minr))
            if not slot["branch"]:
                slot["branch"] = f"releases/v{maj}.{minr}.0"

    if not matched_by_line:
        return ""
    if len(matched_by_line) > 1:
        lines_fmt = ", ".join(
            f"v{maj}.{minr}.0" for maj, minr in sorted(matched_by_line)
        )
        raise ValueError(
            f"upstream_ref {clean_sha} is ambiguous across multiple RC release lines "
            f"({lines_fmt}); pass an explicit vX.Y.0rcN tag or releases/vX.Y.0 branch"
        )
    (maj, minr), info = next(iter(matched_by_line.items()))
    tag_hits: list[tuple[int, str]] = info["tags"]
    if tag_hits:
        tag_hits.sort(key=lambda item: item[0])
        return tag_hits[-1][1]
    return str(info["branch"] or f"releases/v{maj}.{minr}.0")


def _match_rc_line_context(
    state: dict[str, Any] | None,
    *,
    rc_ref: str = "",
    target_sha: str = "",
    tags: dict[str, str] | None = None,
    release_branches: dict[str, str] | None = None,
) -> tuple[tuple[int, int] | None, int | None]:
    """Identify ``(release_line, max_rc_num)`` from ``rc_ref`` or a uniquely matched ``target_sha``."""
    tags_map = tags or {}
    branches_map = release_branches or {}
    rc_branch_states = (state or {}).get("rc_branch_states") or {}
    rc_tag_states = (state or {}).get("rc_tag_states") or {}

    norm_ref = _normalize_rc_ref(rc_ref) if rc_ref else ""
    t_match = RC_TAG_RE.fullmatch(norm_ref) if norm_ref else None
    b_match = (
        RELEASE_BRANCH_RE.fullmatch(norm_ref) if (norm_ref and not t_match) else None
    )
    if t_match:
        return (int(t_match.group(1)), int(t_match.group(2))), int(t_match.group(3))
    if b_match:
        return (int(b_match.group(1)), int(b_match.group(2))), None
    if target_sha and SHA1_RE.fullmatch(target_sha):
        matched_lines: set[tuple[int, int]] = set()
        for br_name, br_sha in branches_map.items():
            bm = RELEASE_BRANCH_RE.fullmatch(br_name)
            if bm and br_sha == target_sha:
                matched_lines.add((int(bm.group(1)), int(bm.group(2))))
        for br_name, br_ent in rc_branch_states.items():
            bm = RELEASE_BRANCH_RE.fullmatch(br_name)
            if bm and isinstance(br_ent, dict):
                hist = br_ent.get("green_history")
                rev = br_ent.get("revoked_shas")
                if (
                    br_ent.get("sha") == target_sha
                    or br_ent.get("last_green_sha") == target_sha
                    or (isinstance(hist, list) and target_sha in hist)
                    or (isinstance(rev, list) and target_sha in rev)
                ):
                    matched_lines.add((int(bm.group(1)), int(bm.group(2))))
        for tg_name, tg_sha in tags_map.items():
            tm = RC_TAG_RE.fullmatch(tg_name)
            if tm and tg_sha == target_sha:
                matched_lines.add((int(tm.group(1)), int(tm.group(2))))
        for tg_name, tg_ent in rc_tag_states.items():
            tm = RC_TAG_RE.fullmatch(tg_name)
            if tm and isinstance(tg_ent, dict) and tg_ent.get("sha") == target_sha:
                matched_lines.add((int(tm.group(1)), int(tm.group(2))))
        if len(matched_lines) == 1:
            return next(iter(matched_lines)), None
    return None, None


MAX_EPHEMERAL_REVOKED_SHAS = 2048


def _protected_control_shas_in_cursor(
    state: dict[str, Any] | None, *, verified_pin: str = ""
) -> set[str]:
    """Return every SHA still referenced as a candidate green control in ``state`` or ``verified_pin``.

    Revocations for these SHAs must never be evicted by subsequent ephemeral branch-tip
    failures; they may only be cleared when that exact SHA passes RC requalification.
    """
    protected: set[str] = set()

    def _keep(raw: Any) -> None:
        s = str(raw or "").strip()
        if SHA1_RE.fullmatch(s):
            protected.add(s)

    _keep(verified_pin)
    if not isinstance(state, dict):
        return protected

    for key in (
        "last_ppu_green_sha",
        "last_cpu_green_sha",
        "last_ppu_tested_sha",
        "last_cpu_checked_sha",
    ):
        _keep(state.get(key))

    raw_last = state.get("last_probe_record")
    if isinstance(raw_last, dict):
        for key in ("upstream_sha", "last_ppu_green_sha", "last_ppu_tested_sha"):
            _keep(raw_last.get(key))

    for rec in state.get("probe_history") or ():
        if isinstance(rec, dict):
            for key in ("upstream_sha", "last_ppu_green_sha", "last_ppu_tested_sha"):
                _keep(rec.get(key))

    for br_ent in (state.get("rc_branch_states") or {}).values():
        if not isinstance(br_ent, dict):
            continue
        _keep(br_ent.get("last_green_sha"))
        if br_ent.get("status") == "pass":
            _keep(br_ent.get("sha"))
        for item in br_ent.get("green_history") or ():
            _keep(item)

    for tg_ent in (state.get("rc_tag_states") or {}).values():
        if isinstance(tg_ent, dict) and tg_ent.get("status") == "pass":
            _keep(tg_ent.get("sha"))

    return protected


def _cap_revoked_shas(
    revoked: Sequence[str],
    protected_shas: set[str],
    *,
    max_ephemeral: int | None = MAX_EPHEMERAL_REVOKED_SHAS,
) -> list[str]:
    """Bound ephemeral failed branch tips while retaining all live control revocations."""
    kept: list[str] = []
    seen: set[str] = set()
    ephemeral_count = 0
    for raw in revoked:
        s = str(raw or "").strip()
        if not SHA1_RE.fullmatch(s) or s in seen:
            continue
        if s in protected_shas or max_ephemeral is None:
            kept.append(s)
            seen.add(s)
        elif ephemeral_count < max_ephemeral:
            kept.append(s)
            seen.add(s)
            ephemeral_count += 1
    return kept


def _collect_excluded_control_shas(
    state: dict[str, Any] | None,
    *,
    rc_ref: str = "",
    target_sha: str = "",
    tags: dict[str, str] | None = None,
    release_branches: dict[str, str] | None = None,
) -> set[str]:
    """Return all SHAs ineligible to act as a verified green control in the given context."""
    tags_map = tags or {}
    rc_branch_states = (state or {}).get("rc_branch_states") or {}
    rc_tag_states = (state or {}).get("rc_tag_states") or {}
    rc_sha_states = (state or {}).get("rc_sha_states") or {}
    rc_line_revoked = (state or {}).get("rc_line_revoked_shas") or {}
    shared_cpu_failed = (state or {}).get("shared_cpu_failed_shas") or ()
    line, max_rc_num = _match_rc_line_context(
        state,
        rc_ref=rc_ref,
        target_sha=target_sha,
        tags=tags,
        release_branches=release_branches,
    )

    excluded: set[str] = set()

    # Fail closed across all RC release lines: any SHA with an unrecovered non-pass
    # RC verdict on any branch, tag, or unbound SHA cannot serve as a verified green control.
    for br_ent in rc_branch_states.values():
        if not isinstance(br_ent, dict):
            continue
        for item in br_ent.get("revoked_shas") or ():
            s = str(item or "").strip()
            if SHA1_RE.fullmatch(s):
                excluded.add(s)
        if br_ent.get("status") != "pass":
            bs_bad = str(br_ent.get("sha") or "").strip()
            if SHA1_RE.fullmatch(bs_bad):
                excluded.add(bs_bad)
    for tg_ent in rc_tag_states.values():
        if not isinstance(tg_ent, dict):
            continue
        ts_bad = str(tg_ent.get("sha") or "").strip()
        if tg_ent.get("status") != "pass" and SHA1_RE.fullmatch(ts_bad):
            excluded.add(ts_bad)
    if isinstance(rc_sha_states, dict):
        for raw_sha, raw_st in rc_sha_states.items():
            s = str(raw_sha or "").strip()
            if raw_st != "pass" and SHA1_RE.fullmatch(s):
                excluded.add(s)
    if isinstance(rc_line_revoked, dict):
        for rev_items in rc_line_revoked.values():
            if isinstance(rev_items, list):
                for raw_rev in rev_items:
                    s = str(raw_rev or "").strip()
                    if SHA1_RE.fullmatch(s):
                        excluded.add(s)
    for raw_shared in shared_cpu_failed:
        s = str(raw_shared or "").strip()
        if SHA1_RE.fullmatch(s):
            excluded.add(s)

    if line is not None and max_rc_num is not None:
        for tg_name, tg_entry in rc_tag_states.items():
            tm = RC_TAG_RE.fullmatch(tg_name)
            if (
                tm
                and (int(tm.group(1)), int(tm.group(2))) == line
                and int(tm.group(3)) >= max_rc_num
                and isinstance(tg_entry, dict)
            ):
                tg_sha = str(tg_entry.get("sha") or "").strip()
                if SHA1_RE.fullmatch(tg_sha):
                    excluded.add(tg_sha)
        for tg_name, tg_sha in tags_map.items():
            tm = RC_TAG_RE.fullmatch(tg_name)
            if (
                tm
                and (int(tm.group(1)), int(tm.group(2))) == line
                and int(tm.group(3)) >= max_rc_num
                and SHA1_RE.fullmatch(str(tg_sha or "").strip())
            ):
                excluded.add(str(tg_sha).strip())

    if isinstance(state, dict):
        cpu_checked = str(state.get("last_cpu_checked_sha") or "").strip()
        cpu_st = str(state.get("last_cpu_status") or "pass").strip()
        ppu_tested = str(state.get("last_ppu_tested_sha") or "").strip()
        ppu_st = str(state.get("last_ppu_status") or "pass").strip()
        if cpu_st != "pass" and SHA1_RE.fullmatch(cpu_checked):
            excluded.add(cpu_checked)
        if ppu_st != "pass" and SHA1_RE.fullmatch(ppu_tested):
            excluded.add(ppu_tested)
        for rec in state.get("probe_history") or ():
            if not isinstance(rec, dict):
                continue
            up_sha = str(rec.get("upstream_sha") or "").strip()
            if not SHA1_RE.fullmatch(up_sha):
                continue
            if rec.get("cpu_status") in ("fail", "infra_error") or rec.get(
                "ppu_status"
            ) in ("fail", "infra_error"):
                if not (
                    up_sha == ppu_tested
                    and ppu_st == "pass"
                    and (up_sha != cpu_checked or cpu_st == "pass")
                ):
                    excluded.add(up_sha)

    return excluded


def _merge_commit_orders(
    base_order: Sequence[str], snap_order: Sequence[str]
) -> list[str]:
    """Merge ``snap_order`` into ``base_order`` preserving all anchored relative positions."""
    clean_base: list[str] = []
    for raw_b in base_order:
        sb = str(raw_b or "").strip()
        if SHA1_RE.fullmatch(sb) and sb not in clean_base:
            clean_base.append(sb)
    clean_snap: list[str] = []
    for raw_s in snap_order:
        ss = str(raw_s or "").strip()
        if SHA1_RE.fullmatch(ss) and ss not in clean_snap:
            clean_snap.append(ss)
    if not clean_base:
        return clean_snap
    merged = list(clean_base)
    for idx, item in enumerate(clean_snap):
        if item in merged:
            continue
        if idx > 0:
            prev_item = clean_snap[idx - 1]
            merged.insert(merged.index(prev_item) + 1, item)
        else:
            next_anchor = next(
                (later for later in clean_snap[1:] if later in merged),
                None,
            )
            if next_anchor is not None:
                merged.insert(merged.index(next_anchor), item)
            else:
                merged.insert(0, item)
    return merged


def _collect_verified_control_shas(
    state: dict[str, Any] | None,
    *,
    verified_pin: str = "",
    rc_ref: str = "",
    target_sha: str = "",
    tags: dict[str, str] | None = None,
    release_branches: dict[str, str] | None = None,
) -> list[str]:
    """Return ordered verified green SHAs (same-line RC greens, main greens, verified pin)."""
    rc_branch_states = (state or {}).get("rc_branch_states") or {}
    rc_tag_states = (state or {}).get("rc_tag_states") or {}
    line, max_rc_num = _match_rc_line_context(
        state,
        rc_ref=rc_ref,
        target_sha=target_sha,
        tags=tags,
        release_branches=release_branches,
    )
    excluded = _collect_excluded_control_shas(
        state,
        rc_ref=rc_ref,
        target_sha=target_sha,
        tags=tags,
        release_branches=release_branches,
    )

    controls: list[str] = []

    def _add(cand_sha: str) -> None:
        s = cand_sha.strip()
        if SHA1_RE.fullmatch(s) and s not in excluded and s not in controls:
            controls.append(s)

    if line is not None:
        major, minor = line
        branch_key = f"releases/v{major}.{minor}.0"
        line_key = f"v{major}.{minor}.0"
        line_candidates: list[str] = []

        def _add_line_cand(cand_sha: str) -> None:
            s = cand_sha.strip()
            if SHA1_RE.fullmatch(s) and s not in excluded and s not in line_candidates:
                line_candidates.append(s)

        br_entry = rc_branch_states.get(branch_key)
        if isinstance(br_entry, dict):
            raw_hist = br_entry.get("green_history")
            if isinstance(raw_hist, list):
                for item in raw_hist:
                    _add_line_cand(str(item or ""))
            br_green = str(br_entry.get("last_green_sha") or "").strip()
            if br_green and br_green not in excluded:
                _add_line_cand(br_green)
            if br_entry.get("status") == "pass":
                _add_line_cand(str(br_entry.get("sha") or ""))

        passing_tags: list[tuple[int, str]] = []
        for tg_name, tg_entry in rc_tag_states.items():
            tm = RC_TAG_RE.fullmatch(tg_name)
            if (
                tm
                and (int(tm.group(1)), int(tm.group(2))) == line
                and isinstance(tg_entry, dict)
                and tg_entry.get("status") == "pass"
            ):
                rc_num = int(tm.group(3))
                tg_sha = str(tg_entry.get("sha") or "").strip()
                if (
                    (max_rc_num is None or rc_num < max_rc_num)
                    and SHA1_RE.fullmatch(tg_sha)
                    and tg_sha not in excluded
                ):
                    passing_tags.append((rc_num, tg_sha))
        passing_tags.sort(key=lambda item: item[0], reverse=True)
        for _, tg_sha in passing_tags:
            _add_line_cand(tg_sha)

        raw_line_orders = (state or {}).get("rc_line_commit_order")
        line_order = (
            raw_line_orders.get(line_key)
            if isinstance(raw_line_orders, dict)
            and isinstance(raw_line_orders.get(line_key), list)
            else None
        )
        if line_order:
            cand_set = set(line_candidates)
            for ord_sha in line_order:
                s_ord = str(ord_sha or "").strip()
                if s_ord in cand_set:
                    _add(s_ord)
        for lc in line_candidates:
            _add(lc)

    if isinstance(state, dict):
        ppu_green = str(state.get("last_ppu_green_sha") or "").strip()
        if ppu_green:
            _add(ppu_green)
        for rec in state.get("probe_history") or ():
            if not isinstance(rec, dict):
                continue
            up_sha = str(rec.get("upstream_sha") or "").strip()
            if rec.get("cpu_status") == "pass" and rec.get("ppu_status") == "pass":
                _add(up_sha)
            rec_green = str(rec.get("last_ppu_green_sha") or "").strip()
            if rec.get("last_ppu_result_status") == "pass":
                _add(rec_green)

    if verified_pin and SHA1_RE.fullmatch(verified_pin.strip()):
        _add(verified_pin.strip())

    return controls


def _resolve_rc_last_green_sha(
    rc_ref: str,
    *,
    target_sha: str,
    tags: dict[str, str],
    release_branches: dict[str, str],
    rc_branch_states: dict[str, dict[str, str]],
    rc_tag_states: dict[str, dict[str, str]],
    tested_rc_tags: Sequence[str],
    fallback_green: str,
    state: dict[str, Any] | None = None,
    verified_pin: str = "",
    vllm_repo: Path | None = None,
) -> str:
    """Return the newest verified non-revoked ancestor RC/main SHA."""
    synthetic_state: dict[str, Any] = (
        dict(state)
        if isinstance(state, dict)
        else {
            "last_ppu_green_sha": fallback_green,
            "last_ppu_tested_sha": fallback_green,
            "last_ppu_status": "pass",
            "rc_branch_states": rc_branch_states,
            "rc_tag_states": rc_tag_states,
            "tested_rc_tags": list(tested_rc_tags),
            "probe_history": [],
        }
    )
    excluded = _collect_excluded_control_shas(
        synthetic_state,
        rc_ref=rc_ref,
        target_sha=target_sha,
        tags=tags,
        release_branches=release_branches,
    )
    candidates = _collect_verified_control_shas(
        synthetic_state,
        verified_pin=verified_pin,
        rc_ref=rc_ref,
        target_sha=target_sha,
        tags=tags,
        release_branches=release_branches,
    )
    if (
        SHA1_RE.fullmatch(fallback_green)
        and fallback_green not in excluded
        and fallback_green not in candidates
    ):
        candidates.append(fallback_green)

    non_target = [cand for cand in candidates if cand != target_sha]
    if vllm_repo is not None and SHA1_RE.fullmatch(target_sha):
        if not non_target:
            return ""
        chosen, rng = resolve_verified_bisect_control(
            vllm_repo,
            primary_green=non_target[0],
            red_sha=target_sha,
            fallback_greens=non_target[1:],
            excluded_shas=excluded,
        )
        return chosen if rng else ""

    if non_target:
        return non_target[0]
    if candidates:
        return candidates[0]
    return ""


def is_safe_ppu_skip_path(path: str) -> bool:
    norm = path.strip().lstrip("./")
    if not norm or ".." in norm.split("/"):
        return False
    base = norm.rsplit("/", 1)[-1]
    if base.endswith(".md"):
        return True
    for pattern in SAFE_PPU_SKIP_PATTERNS:
        if fnmatchcase(norm, pattern):
            return True
    return False


_GIT_DIFF_STATUS_TOKEN_RE = re.compile(r"^[ACDMRTUXB][0-9]{0,3}$")


def parse_changed_paths_lines(raw_lines: Sequence[str] | None) -> list[str]:
    """Expand ``git diff --name-only`` or ``--name-status`` lines into all touched paths.

    For rename/copy lines such as ``R100\\tvllm/v1/worker/foo.py\\texamples/foo.py``,
    both source and destination paths are returned so moving runtime code into an
    allowlisted directory never skips PPU verification.
    """
    if not raw_lines:
        return []
    expanded: list[str] = []
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if "\t" in stripped:
            parts = [col.strip() for col in stripped.split("\t") if col.strip()]
            if parts and _GIT_DIFF_STATUS_TOKEN_RE.fullmatch(parts[0]):
                parts = parts[1:]
            expanded.extend(parts)
        else:
            expanded.append(stripped)
    return expanded


def should_skip_ppu_for_paths(changed_paths: Sequence[str] | None) -> bool:
    """Return True iff ``changed_paths`` is non-empty and 100% allowlisted."""
    expanded = parse_changed_paths_lines(changed_paths)
    if not expanded:
        return False
    return all(is_safe_ppu_skip_path(path) for path in expanded)


def _sanitize_probe_record(
    rec: Any,
    *,
    default_ppu_green: str,
    default_ppu_tested: str,
    default_ppu_status: str,
    label: str = "last_probe_record",
) -> dict[str, str]:
    if not isinstance(rec, dict):
        raise ValueError(f"{label} must be an object")
    up_sha = _require_sha1(str(rec.get("upstream_sha", "")), f"{label}.upstream_sha")
    green_sha = _require_sha1(
        str(rec.get("last_ppu_green_sha", default_ppu_green)),
        f"{label}.last_ppu_green_sha",
    )
    tested_sha = _require_sha1(
        str(rec.get("last_ppu_tested_sha", default_ppu_tested)),
        f"{label}.last_ppu_tested_sha",
    )
    cpu_st = str(rec.get("cpu_status", "not_run"))
    ppu_st = str(rec.get("ppu_status", "not_run"))
    skip_rs = str(rec.get("ppu_skip_reason", "none"))
    rec_ppu_res = str(
        rec.get(
            "last_ppu_result_status",
            "pass"
            if tested_sha == green_sha and default_ppu_status == "pass"
            else "fail",
        )
    )
    if cpu_st not in ("pass", "fail", "not_run", "infra_error"):
        raise ValueError(f"invalid {label}.cpu_status")
    if ppu_st not in ("pass", "fail", "skipped", "not_run", "infra_error"):
        raise ValueError(f"invalid {label}.ppu_status")
    if skip_rs not in VALID_PPU_SKIP_REASONS:
        raise ValueError(f"invalid {label}.ppu_skip_reason")
    if rec_ppu_res not in ("pass", "fail"):
        raise ValueError(f"invalid {label}.last_ppu_result_status")
    return {
        "gear": "probe",
        "upstream_sha": up_sha,
        "cpu_status": cpu_st,
        "ppu_status": ppu_st,
        "ppu_skip_reason": skip_rs,
        "last_ppu_tested_sha": tested_sha,
        "last_ppu_green_sha": green_sha,
        "last_ppu_result_status": rec_ppu_res,
    }


def _sanitize_cursor(
    cursor: dict[str, Any] | None, *, verified_pin: str
) -> dict[str, Any]:
    """Validate and canonicalize cursor state, failing closed on malformed fields."""
    pin = _require_sha1(verified_pin, "verified_pin")
    if cursor is not None and not isinstance(cursor, dict):
        raise ValueError("cursor must be a JSON object")
    if not cursor:
        return {
            "schema": 1,
            "last_cpu_checked_sha": pin,
            "last_cpu_green_sha": pin,
            "last_cpu_status": "pass",
            "last_ppu_tested_sha": pin,
            "last_ppu_green_sha": pin,
            "last_ppu_status": "pass",
            "tested_rc_tags": [],
            "rc_tag_states": {},
            "rc_branch_states": {},
            "patch_fingerprints": {},
            "probe_history": [],
        }

    def _sha_field(key: str) -> str:
        raw = cursor.get(key)
        if raw is None or raw == "":
            return pin
        if not isinstance(raw, str):
            raise ValueError(f"{key} must be a 40-character lowercase hex SHA")
        return _require_sha1(raw, key)

    raw_rcs = cursor.get("tested_rc_tags")
    if raw_rcs is None:
        raw_rcs = []
    if not isinstance(raw_rcs, list):
        raise ValueError("tested_rc_tags must be a list")
    tested_rcs: list[str] = []
    for item in raw_rcs:
        if not isinstance(item, str) or not RC_TAG_RE.fullmatch(item.strip()):
            raise ValueError(f"invalid RC tag in cursor: {item!r}")
        tag_clean = item.strip()
        if tag_clean not in tested_rcs:
            tested_rcs.append(tag_clean)

    raw_rc_states = cursor.get("rc_tag_states")
    if raw_rc_states is None:
        raw_rc_states = {}
    if not isinstance(raw_rc_states, dict):
        raise ValueError("rc_tag_states must be an object")
    clean_rc_states: dict[str, dict[str, str]] = {}
    for raw_tag, raw_entry in raw_rc_states.items():
        if not isinstance(raw_tag, str) or not RC_TAG_RE.fullmatch(raw_tag.strip()):
            raise ValueError(f"invalid RC tag key in rc_tag_states: {raw_tag!r}")
        if not isinstance(raw_entry, dict):
            raise ValueError("rc_tag_states entry must be an object")
        rc_sha = _require_sha1(str(raw_entry.get("sha", "")), "rc_tag_states.sha")
        rc_st = str(raw_entry.get("status", "fail"))
        if rc_st not in ("pass", "fail", "infra_error"):
            raise ValueError(f"invalid rc_tag_states status: {rc_st!r}")
        tag_clean = raw_tag.strip()
        clean_rc_states[tag_clean] = {"sha": rc_sha, "status": rc_st}
        if rc_st == "pass" and tag_clean not in tested_rcs:
            tested_rcs.append(tag_clean)

    raw_branch_states = cursor.get("rc_branch_states")
    if raw_branch_states is None:
        raw_branch_states = {}
    if not isinstance(raw_branch_states, dict):
        raise ValueError("rc_branch_states must be an object")
    clean_branch_states: dict[str, dict[str, Any]] = {}
    for raw_branch, raw_entry in raw_branch_states.items():
        if not isinstance(raw_branch, str) or not RELEASE_BRANCH_RE.fullmatch(
            raw_branch.strip()
        ):
            raise ValueError(
                f"invalid release branch key in rc_branch_states: {raw_branch!r}"
            )
        if not isinstance(raw_entry, dict):
            raise ValueError("rc_branch_states entry must be an object")
        br_sha = _require_sha1(str(raw_entry.get("sha", "")), "rc_branch_states.sha")
        br_st = str(raw_entry.get("status", "fail"))
        if br_st not in ("pass", "fail", "infra_error"):
            raise ValueError(f"invalid rc_branch_states status: {br_st!r}")
        br_clean = raw_branch.strip()
        hist: list[str] = []
        raw_hist_val = raw_entry.get("green_history")
        if raw_hist_val is not None:
            if not isinstance(raw_hist_val, list):
                raise ValueError("rc_branch_states.green_history must be a list")
            for idx, item in enumerate(raw_hist_val):
                if not isinstance(item, str):
                    raise ValueError(
                        f"rc_branch_states.green_history[{idx}] must be a 40-character lowercase hex SHA"
                    )
                s = _require_sha1(item, f"rc_branch_states.green_history[{idx}]")
                if s not in hist:
                    hist.append(s)
        revoked_list: list[str] = []
        raw_rev_val = raw_entry.get("revoked_shas")
        if raw_rev_val is not None:
            if not isinstance(raw_rev_val, list):
                raise ValueError("rc_branch_states.revoked_shas must be a list")
            for idx, item in enumerate(raw_rev_val):
                if not isinstance(item, str):
                    raise ValueError(
                        f"rc_branch_states.revoked_shas[{idx}] must be a 40-character lowercase hex SHA"
                    )
                rs = _require_sha1(item, f"rc_branch_states.revoked_shas[{idx}]")
                if rs not in revoked_list:
                    revoked_list.append(rs)
        superseded_list: list[str] = []
        raw_sup_val = raw_entry.get("superseded_shas")
        if raw_sup_val is not None:
            if not isinstance(raw_sup_val, list):
                raise ValueError("rc_branch_states.superseded_shas must be a list")
            for idx, item in enumerate(raw_sup_val):
                if not isinstance(item, str):
                    raise ValueError(
                        f"rc_branch_states.superseded_shas[{idx}] must be a 40-character lowercase hex SHA"
                    )
                ss = _require_sha1(item, f"rc_branch_states.superseded_shas[{idx}]")
                if ss not in superseded_list:
                    superseded_list.append(ss)
        raw_green = raw_entry.get("last_green_sha")
        if raw_green is not None and raw_green != "":
            if not isinstance(raw_green, str):
                raise ValueError(
                    "rc_branch_states.last_green_sha must be a 40-character lowercase hex SHA"
                )
            green_clean = _require_sha1(raw_green, "rc_branch_states.last_green_sha")
            if green_clean not in hist:
                hist.insert(0, green_clean)
        if br_st == "pass" and br_sha not in hist:
            hist.insert(0, br_sha)
        entry_clean: dict[str, Any] = {"sha": br_sha, "status": br_st}
        if hist:
            entry_clean["last_green_sha"] = hist[0]
            entry_clean["green_history"] = hist
        if revoked_list:
            entry_clean["revoked_shas"] = revoked_list
            kept_sup = [s for s in superseded_list if s in revoked_list]
            if kept_sup:
                entry_clean["superseded_shas"] = kept_sup
        clean_branch_states[br_clean] = entry_clean

    # Enforce cross-entry fail-closed SHA verdict per release line (`fail` > `infra_error` > `pass`).
    all_rc_lines: set[tuple[int, int]] = set()
    for t_name in clean_rc_states:
        tm = RC_TAG_RE.fullmatch(t_name)
        if tm:
            all_rc_lines.add((int(tm.group(1)), int(tm.group(2))))
    for b_name in clean_branch_states:
        bm = RELEASE_BRANCH_RE.fullmatch(b_name)
        if bm:
            all_rc_lines.add((int(bm.group(1)), int(bm.group(2))))

    def _record_bad_sha(
        bad_map: dict[str, str], order_list: list[str], bad_sha: str, st: str
    ) -> None:
        if bad_sha not in order_list:
            order_list.append(bad_sha)
        if st == "fail" or bad_sha not in bad_map:
            bad_map[bad_sha] = "fail" if st == "fail" else "infra_error"

    def _snap_add(target_snap: list[str], cand: str) -> None:
        if SHA1_RE.fullmatch(cand) and cand not in target_snap:
            target_snap.append(cand)

    pre_strip_line_snapshots: dict[str, list[str]] = {}
    for maj, minr in sorted(all_rc_lines):
        b_key = f"releases/v{maj}.{minr}.0"
        l_key = f"v{maj}.{minr}.0"
        b_ent = clean_branch_states.get(b_key)
        snap: list[str] = []
        if isinstance(b_ent, dict):
            sup_pre = set(b_ent.get("superseded_shas") or ())
            if b_ent["status"] != "pass" and b_ent["sha"] not in sup_pre:
                _snap_add(snap, b_ent["sha"])
            raw_gh = list(b_ent.get("green_history") or ())
            if raw_gh:
                _snap_add(snap, raw_gh[0])
            for s_sup in b_ent.get("superseded_shas") or ():
                _snap_add(snap, s_sup)
            for s_rev in b_ent.get("revoked_shas") or ():
                _snap_add(snap, s_rev)
            for s_gh in raw_gh[1:]:
                _snap_add(snap, s_gh)
            _snap_add(snap, b_ent["sha"])
        line_tags_pre: list[tuple[int, str]] = []
        for t_name, t_ent in clean_rc_states.items():
            tm = RC_TAG_RE.fullmatch(t_name)
            if tm and (int(tm.group(1)), int(tm.group(2))) == (maj, minr):
                line_tags_pre.append((int(tm.group(3)), t_ent["sha"]))
        line_tags_pre.sort(key=lambda item: item[0], reverse=True)
        for _, t_sha in line_tags_pre:
            _snap_add(snap, t_sha)
        if snap:
            pre_strip_line_snapshots[l_key] = snap

    for maj, minr in sorted(all_rc_lines):
        b_key = f"releases/v{maj}.{minr}.0"
        b_ent = clean_branch_states.get(b_key)
        bad_by_sha: dict[str, str] = {}
        bad_order: list[str] = []

        if isinstance(b_ent, dict) and b_ent["status"] != "pass":
            _record_bad_sha(bad_by_sha, bad_order, b_ent["sha"], b_ent["status"])
        for t_name, t_ent in clean_rc_states.items():
            tm = RC_TAG_RE.fullmatch(t_name)
            if (
                tm
                and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                and t_ent["status"] != "pass"
            ):
                _record_bad_sha(bad_by_sha, bad_order, t_ent["sha"], t_ent["status"])
        if isinstance(b_ent, dict):
            for rs in b_ent.get("revoked_shas") or ():
                if rs not in bad_order:
                    bad_order.append(rs)
                if rs not in bad_by_sha:
                    bad_by_sha[rs] = "fail"
        if not bad_by_sha:
            continue
        for t_name, t_ent in clean_rc_states.items():
            tm = RC_TAG_RE.fullmatch(t_name)
            if (
                tm
                and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                and t_ent["sha"] in bad_by_sha
            ):
                worst = bad_by_sha[t_ent["sha"]]
                if t_ent["status"] == "pass" or worst == "fail":
                    t_ent["status"] = worst
        if isinstance(b_ent, dict):
            if b_ent["sha"] in bad_by_sha:
                worst_b = bad_by_sha[b_ent["sha"]]
                if b_ent["status"] == "pass" or worst_b == "fail":
                    b_ent["status"] = worst_b
            rem_hist = [
                s for s in (b_ent.get("green_history") or ()) if s not in bad_by_sha
            ][:20]
            b_ent.pop("last_green_sha", None)
            b_ent.pop("green_history", None)
            if rem_hist:
                b_ent["last_green_sha"] = rem_hist[0]
                b_ent["green_history"] = rem_hist
            b_ent["revoked_shas"] = bad_order
            if "superseded_shas" in b_ent:
                rem_sup = [
                    s for s in (b_ent.get("superseded_shas") or ()) if s in bad_by_sha
                ]
                if rem_sup:
                    b_ent["superseded_shas"] = rem_sup
                else:
                    b_ent.pop("superseded_shas", None)

    tested_rcs = [
        t
        for t in tested_rcs
        if t not in clean_rc_states or clean_rc_states[t]["status"] == "pass"
    ]

    raw_rc_sha_states = cursor.get("rc_sha_states")
    if raw_rc_sha_states is None:
        raw_rc_sha_states = {}
    if not isinstance(raw_rc_sha_states, dict):
        raise ValueError("rc_sha_states must be an object")
    clean_rc_sha_states: dict[str, str] = {}
    for raw_k, raw_v in raw_rc_sha_states.items():
        if not isinstance(raw_k, str):
            raise ValueError(
                "rc_sha_states key must be a 40-character lowercase hex SHA"
            )
        sha_k = _require_sha1(raw_k, "rc_sha_states.sha")
        st_v = str(raw_v or "")
        if st_v not in ("fail", "infra_error"):
            raise ValueError(f"invalid rc_sha_states status: {raw_v!r}")
        clean_rc_sha_states[sha_k] = st_v

    raw_rc_line_revs = cursor.get("rc_line_revoked_shas")
    if raw_rc_line_revs is None:
        raw_rc_line_revs = {}
    if not isinstance(raw_rc_line_revs, dict):
        raise ValueError("rc_line_revoked_shas must be an object")
    clean_rc_line_revs: dict[str, list[str]] = {}
    for raw_line_k, raw_line_v in raw_rc_line_revs.items():
        if not isinstance(raw_line_k, str) or not RELEASE_LINE_RE.fullmatch(
            raw_line_k.strip()
        ):
            raise ValueError(
                f"invalid release line key in rc_line_revoked_shas: {raw_line_k!r}"
            )
        if not isinstance(raw_line_v, list):
            raise ValueError("rc_line_revoked_shas entry must be a list")
        lk_clean = raw_line_k.strip()
        lv_clean: list[str] = []
        for idx, item in enumerate(raw_line_v):
            if not isinstance(item, str):
                raise ValueError(
                    f"rc_line_revoked_shas[{lk_clean!r}][{idx}] must be a 40-character lowercase hex SHA"
                )
            s = _require_sha1(item, f"rc_line_revoked_shas[{lk_clean!r}][{idx}]")
            if s not in lv_clean:
                lv_clean.append(s)
        if lv_clean:
            clean_rc_line_revs[lk_clean] = lv_clean

    raw_rc_line_order = cursor.get("rc_line_commit_order")
    if raw_rc_line_order is None:
        raw_rc_line_order = {}
    if not isinstance(raw_rc_line_order, dict):
        raise ValueError("rc_line_commit_order must be an object")
    clean_rc_line_order: dict[str, list[str]] = {}
    for raw_ord_k, raw_ord_v in raw_rc_line_order.items():
        if not isinstance(raw_ord_k, str) or not RELEASE_LINE_RE.fullmatch(
            raw_ord_k.strip()
        ):
            raise ValueError(
                f"invalid release line key in rc_line_commit_order: {raw_ord_k!r}"
            )
        if not isinstance(raw_ord_v, list):
            raise ValueError("rc_line_commit_order entry must be a list")
        ok_clean = raw_ord_k.strip()
        ov_clean: list[str] = []
        for idx, item in enumerate(raw_ord_v):
            if not isinstance(item, str):
                raise ValueError(
                    f"rc_line_commit_order[{ok_clean!r}][{idx}] must be a 40-character lowercase hex SHA"
                )
            s = _require_sha1(item, f"rc_line_commit_order[{ok_clean!r}][{idx}]")
            if s not in ov_clean:
                ov_clean.append(s)
        snap_list = list(pre_strip_line_snapshots.get(ok_clean) or [])
        for rev_s in clean_rc_line_revs.get(ok_clean) or ():
            if rev_s not in snap_list:
                snap_list.append(rev_s)
        merged_line_order = _merge_commit_orders(ov_clean, snap_list)
        if merged_line_order:
            clean_rc_line_order[ok_clean] = merged_line_order

    # Re-derive canonical branch `green_history` and `last_green_sha` from authoritative
    # line commit order + unrevoked passing commits so stale histories cannot hide newer greens.
    for maj, minr in sorted(all_rc_lines):
        b_key = f"releases/v{maj}.{minr}.0"
        l_key = f"v{maj}.{minr}.0"
        b_ent = clean_branch_states.get(b_key)
        if not isinstance(b_ent, dict):
            continue
        line_excluded = set(b_ent.get("revoked_shas") or ()) | set(
            clean_rc_line_revs.get(l_key) or ()
        )
        if b_ent["status"] != "pass":
            line_excluded.add(b_ent["sha"])
        for t_name, t_ent in clean_rc_states.items():
            tm = RC_TAG_RE.fullmatch(t_name)
            if (
                tm
                and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                and t_ent["status"] != "pass"
            ):
                line_excluded.add(t_ent["sha"])
        line_greens: list[str] = []
        for s_g in b_ent.get("green_history") or ():
            if s_g not in line_excluded and s_g not in line_greens:
                line_greens.append(s_g)
        lg_cur = str(b_ent.get("last_green_sha") or "").strip()
        if (
            SHA1_RE.fullmatch(lg_cur)
            and lg_cur not in line_excluded
            and lg_cur not in line_greens
        ):
            line_greens.append(lg_cur)
        if (
            b_ent["status"] == "pass"
            and b_ent["sha"] not in line_excluded
            and b_ent["sha"] not in line_greens
        ):
            line_greens.append(b_ent["sha"])
        if l_key in clean_rc_line_order:
            for t_name, t_ent in clean_rc_states.items():
                tm = RC_TAG_RE.fullmatch(t_name)
                if (
                    tm
                    and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                    and t_ent["status"] == "pass"
                    and t_ent["sha"] not in line_excluded
                    and t_ent["sha"] not in line_greens
                ):
                    line_greens.append(t_ent["sha"])
            eff_ord = clean_rc_line_order[l_key]
            green_set = set(line_greens)
            canon_hist = [s_o for s_o in eff_ord if s_o in green_set]
            for extra_g in line_greens:
                if extra_g not in canon_hist:
                    canon_hist.append(extra_g)
        else:
            canon_hist = line_greens
        canon_hist = canon_hist[:20]
        b_ent.pop("last_green_sha", None)
        b_ent.pop("green_history", None)
        if canon_hist:
            b_ent["last_green_sha"] = canon_hist[0]
            b_ent["green_history"] = canon_hist

    raw_shared_failed = cursor.get("shared_cpu_failed_shas")
    if raw_shared_failed is None:
        raw_shared_failed = []
    if not isinstance(raw_shared_failed, list):
        raise ValueError("shared_cpu_failed_shas must be a list")
    clean_shared_failed: list[str] = []
    for idx, item in enumerate(raw_shared_failed):
        if not isinstance(item, str):
            raise ValueError(
                f"shared_cpu_failed_shas[{idx}] must be a 40-character lowercase hex SHA"
            )
        sf_sha = _require_sha1(item, f"shared_cpu_failed_shas[{idx}]")
        if sf_sha not in clean_shared_failed:
            clean_shared_failed.append(sf_sha)

    raw_fps = cursor.get("patch_fingerprints")
    if raw_fps is None:
        raw_fps = {}
    if not isinstance(raw_fps, dict):
        raise ValueError("patch_fingerprints must be an object")
    clean_fps: dict[str, str] = {}
    for k, v in raw_fps.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("patch_fingerprints entries must be strings")
        assert_zero_sensitive_leaks(k)
        if not OPAQUE_PATCH_KEY_RE.fullmatch(k) and not SAFE_SYMBOL_RE.fullmatch(k):
            raise ValueError(f"invalid patch target key in cursor: {k!r}")
        if v != "additive" and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(f"invalid patch fingerprint hash for {k!r}")
        clean_fps[k] = v

    cpu_checked = _sha_field("last_cpu_checked_sha")
    cpu_green = _sha_field("last_cpu_green_sha")
    ppu_tested = _sha_field("last_ppu_tested_sha")
    ppu_green = _sha_field("last_ppu_green_sha")

    raw_rec = cursor.get("last_probe_record")
    rec_cpu_failed = isinstance(raw_rec, dict) and raw_rec.get("cpu_status") == "fail"
    rec_ppu_failed = isinstance(raw_rec, dict) and raw_rec.get("ppu_status") == "fail"

    raw_cpu_st = cursor.get("last_cpu_status")
    if raw_cpu_st is None:
        last_cpu_st = (
            "pass" if (cpu_checked == cpu_green and not rec_cpu_failed) else "fail"
        )
    elif raw_cpu_st in ("pass", "fail"):
        last_cpu_st = raw_cpu_st
    else:
        raise ValueError(f"invalid last_cpu_status in cursor: {raw_cpu_st!r}")

    raw_ppu_st = cursor.get("last_ppu_status")
    if raw_ppu_st is None:
        last_ppu_st = (
            "pass" if (ppu_tested == ppu_green and not rec_ppu_failed) else "fail"
        )
    elif raw_ppu_st in ("pass", "fail"):
        last_ppu_st = raw_ppu_st
    else:
        raise ValueError(f"invalid last_ppu_status in cursor: {raw_ppu_st!r}")

    sanitized: dict[str, Any] = {
        "schema": 1,
        "last_cpu_checked_sha": cpu_checked,
        "last_cpu_green_sha": cpu_green,
        "last_cpu_status": last_cpu_st,
        "last_ppu_tested_sha": ppu_tested,
        "last_ppu_green_sha": ppu_green,
        "last_ppu_status": last_ppu_st,
        "tested_rc_tags": tested_rcs,
        "rc_tag_states": dict(sorted(clean_rc_states.items())),
        "rc_branch_states": dict(sorted(clean_branch_states.items())),
        "patch_fingerprints": dict(sorted(clean_fps.items())),
        "probe_history": [],
    }

    clean_history: list[dict[str, str]] = []
    seen_shas: set[str] = set()
    if raw_rec is not None:
        clean_last = _sanitize_probe_record(
            raw_rec,
            default_ppu_green=ppu_green,
            default_ppu_tested=ppu_tested,
            default_ppu_status=last_ppu_st,
            label="last_probe_record",
        )
        sanitized["last_probe_record"] = clean_last
        clean_history.append(clean_last)
        seen_shas.add(clean_last["upstream_sha"])

    raw_hist = cursor.get("probe_history")
    if raw_hist is None:
        raw_hist = []
    if not isinstance(raw_hist, list):
        raise ValueError("probe_history must be a list")
    for idx, item in enumerate(raw_hist):
        clean_item = _sanitize_probe_record(
            item,
            default_ppu_green=ppu_green,
            default_ppu_tested=ppu_tested,
            default_ppu_status=last_ppu_st,
            label=f"probe_history[{idx}]",
        )
        if clean_item["upstream_sha"] not in seen_shas:
            clean_history.append(clean_item)
            seen_shas.add(clean_item["upstream_sha"])
        if len(clean_history) >= 20:
            break

    sanitized["probe_history"] = clean_history
    if clean_history and "last_probe_record" not in sanitized:
        sanitized["last_probe_record"] = clean_history[0]
    if clean_rc_sha_states:
        sanitized["rc_sha_states"] = dict(sorted(clean_rc_sha_states.items()))
    if clean_rc_line_revs:
        sanitized["rc_line_revoked_shas"] = dict(sorted(clean_rc_line_revs.items()))
    if clean_rc_line_order:
        sanitized["rc_line_commit_order"] = dict(sorted(clean_rc_line_order.items()))
    if clean_shared_failed:
        sanitized["shared_cpu_failed_shas"] = clean_shared_failed[
            :MAX_EPHEMERAL_REVOKED_SHAS
        ]

    protected_controls = _protected_control_shas_in_cursor(sanitized, verified_pin=pin)
    for b_ent in sanitized["rc_branch_states"].values():
        if isinstance(b_ent, dict) and "revoked_shas" in b_ent:
            capped_rev = _cap_revoked_shas(
                b_ent.get("revoked_shas") or (),
                protected_controls,
            )
            if capped_rev:
                b_ent["revoked_shas"] = capped_rev
            else:
                b_ent.pop("revoked_shas", None)
    if "rc_line_revoked_shas" in sanitized:
        capped_lines: dict[str, list[str]] = {}
        for lk, lv in sanitized["rc_line_revoked_shas"].items():
            capped_lv = _cap_revoked_shas(lv, protected_controls)
            if capped_lv:
                capped_lines[lk] = capped_lv
        if capped_lines:
            sanitized["rc_line_revoked_shas"] = dict(sorted(capped_lines.items()))
        else:
            sanitized.pop("rc_line_revoked_shas", None)
    if "rc_line_commit_order" in sanitized:
        order_protected = set(protected_controls)
        for b_ent in sanitized["rc_branch_states"].values():
            if isinstance(b_ent, dict):
                s_tip = str(b_ent.get("sha") or "").strip()
                if SHA1_RE.fullmatch(s_tip):
                    order_protected.add(s_tip)
                for rs in b_ent.get("revoked_shas") or ():
                    if SHA1_RE.fullmatch(str(rs)):
                        order_protected.add(str(rs))
        for t_ent in sanitized["rc_tag_states"].values():
            if isinstance(t_ent, dict) and SHA1_RE.fullmatch(
                str(t_ent.get("sha") or "")
            ):
                order_protected.add(str(t_ent["sha"]))
        for lv in (sanitized.get("rc_line_revoked_shas") or {}).values():
            for rs in lv:
                if SHA1_RE.fullmatch(str(rs)):
                    order_protected.add(str(rs))
        capped_orders: dict[str, list[str]] = {}
        for ok, ov in sanitized["rc_line_commit_order"].items():
            capped_ov = _cap_revoked_shas(ov, order_protected)
            if capped_ov:
                capped_orders[ok] = capped_ov
        if capped_orders:
            sanitized["rc_line_commit_order"] = dict(sorted(capped_orders.items()))
        else:
            sanitized.pop("rc_line_commit_order", None)

    return sanitized


def _is_on_main_first_parent_lineage(
    vllm_repo: Path | None,
    *,
    previous_pin: str,
    candidate_sha: str,
    main_sha: str = "",
) -> bool:
    """Return True iff ``candidate_sha`` advances from ``previous_pin`` along upstream ``main``."""
    if vllm_repo is None:
        return True
    _ensure_commit_fetched(vllm_repo, candidate_sha, depth=250)
    _ensure_commit_fetched(vllm_repo, previous_pin, depth=64)
    if not resolve_complete_first_parent_range(vllm_repo, previous_pin, candidate_sha):
        return False

    eff_main = main_sha.strip() if SHA1_RE.fullmatch(main_sha.strip()) else ""
    if not eff_main:
        for ref_name in ("refs/remotes/origin/main", "refs/heads/main"):
            rev = subprocess.run(
                [
                    "git",
                    "-C",
                    str(vllm_repo),
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    f"{ref_name}^{{commit}}",
                ],
                capture_output=True,
                text=True,
            )
            resolved = rev.stdout.strip()
            if rev.returncode == 0 and SHA1_RE.fullmatch(resolved):
                eff_main = resolved
                break
    if eff_main:
        _ensure_commit_fetched(vllm_repo, eff_main, depth=250)
        if candidate_sha == eff_main:
            return True
        if not resolve_complete_first_parent_range(vllm_repo, candidate_sha, eff_main):
            return False
    return True


def select_pin_bump_candidate(
    probe_records: Sequence[dict[str, Any]],
    *,
    verified_pin: str,
    excluded_shas: set[str] | frozenset[str] = frozenset(),
    vllm_repo: Path | None = None,
    main_sha: str = "",
) -> str | None:
    """Pick the latest healthy, unrevoked main-line probe SHA newer than verified_pin."""
    pin = _require_sha1(verified_pin, "verified_pin")
    for record in probe_records:
        if not isinstance(record, dict) or record.get("gear") != "probe":
            continue
        sha = str(record.get("upstream_sha", "")).strip()
        if not SHA1_RE.fullmatch(sha):
            continue
        # probe_records is ordered newest-to-oldest; reaching verified_pin means
        # all remaining entries predate the current pin and must never downgrade it.
        if sha == pin:
            return None
        if sha in excluded_shas or record.get("cpu_status") != "pass":
            continue
        ppu_status = record.get("ppu_status")
        skip_reason = record.get("ppu_skip_reason", "none")
        last_ppu_green = str(record.get("last_ppu_green_sha", "")).strip()
        last_ppu_tested = str(record.get("last_ppu_tested_sha", last_ppu_green)).strip()
        last_ppu_res = str(record.get("last_ppu_result_status", "pass")).strip()
        eligible = False
        if ppu_status == "pass":
            eligible = True
        elif (
            ppu_status == "skipped"
            and skip_reason == "allowlisted_non_ppu_changes"
            and SHA1_RE.fullmatch(last_ppu_green)
            and last_ppu_tested == last_ppu_green
            and last_ppu_res == "pass"
            and last_ppu_green not in excluded_shas
            and last_ppu_tested not in excluded_shas
        ):
            eligible = True
        if eligible and _is_on_main_first_parent_lineage(
            vllm_repo,
            previous_pin=pin,
            candidate_sha=sha,
            main_sha=main_sha,
        ):
            return sha
    return None


def route_three_gears(
    *,
    ls_remote_data: dict[str, Any],
    verified_pin: str,
    release_baseline_tag: str,
    requested_gear: str = "auto",
    schedule_cron: str = "",
    upstream_ref_override: str = "",
    last_green_sha_override: str = "",
    cursor: dict[str, Any] | None = None,
    changed_paths_since_ppu: Sequence[str] | None = None,
    changed_files_base_sha: str = "",
    active_pr_runs: int = 0,
    probe_history: Sequence[dict[str, Any]] = (),
    force: bool = False,
    vllm_repo: Path | None = None,
) -> dict[str, Any]:
    """Resolve gear, target SHA, and CPU/PPU execution plan."""
    if requested_gear not in VALID_GEARS:
        raise ValueError(f"unsupported gear: {requested_gear!r}")
    pin = _require_sha1(verified_pin, "verified_pin")
    override_green = (
        _require_sha1(last_green_sha_override, "last_green_sha")
        if last_green_sha_override.strip()
        else ""
    )
    state = _sanitize_cursor(cursor, verified_pin=pin)
    last_cpu_green = state["last_cpu_green_sha"]
    last_ppu_tested = state["last_ppu_tested_sha"]
    last_ppu_green = state["last_ppu_green_sha"]
    tested_rcs = list(state["tested_rc_tags"])
    tags: dict[str, str] = dict(ls_remote_data.get("tags") or {})
    release_branches: dict[str, str] = dict(
        ls_remote_data.get("release_branches") or {}
    )
    main_sha = ls_remote_data.get("main_sha")

    override = _normalize_rc_ref(upstream_ref_override)
    effective_gear = requested_gear
    rc_tag = ""
    target_sha = ""

    if effective_gear == "auto":
        if schedule_cron.strip() == DAILY_PIN_CRON:
            effective_gear = "daily_pin_bump"
        elif override and _is_valid_rc_ref(override):
            effective_gear = "rc_qualify"
        else:
            rc_hit = select_newest_rc_target(
                tags,
                release_branches,
                release_baseline_tag=release_baseline_tag,
                tested_rc_tags=tested_rcs,
                rc_tag_states=state["rc_tag_states"],
                rc_branch_states=state["rc_branch_states"],
                rc_sha_states=state.get("rc_sha_states"),
                rc_line_revoked_shas=state.get("rc_line_revoked_shas"),
                include_failed=force,
            )
            if rc_hit is not None and not override:
                effective_gear = "rc_qualify"
                rc_tag, target_sha = rc_hit
            else:
                effective_gear = "probe"

    if effective_gear == "rc_qualify":
        if not target_sha:
            if override:
                if RC_TAG_RE.fullmatch(override):
                    if override not in tags:
                        raise ValueError(f"RC tag {override!r} not found in upstream")
                    rc_tag = override
                    target_sha = tags[override]
                elif RELEASE_BRANCH_RE.fullmatch(override):
                    if override not in release_branches:
                        raise ValueError(
                            f"RC release branch {override!r} not found in upstream"
                        )
                    rc_tag = override
                    target_sha = release_branches[override]
                else:
                    target_sha = _require_sha1(override, "upstream_ref")
                    rc_tag = _resolve_unique_rc_ref_for_sha(
                        target_sha,
                        tags=tags,
                        release_branches=release_branches,
                        release_baseline_tag=release_baseline_tag,
                        rc_branch_states=state["rc_branch_states"],
                        rc_tag_states=state["rc_tag_states"],
                        rc_line_revoked_shas=state.get("rc_line_revoked_shas"),
                        tested_rc_tags=tested_rcs,
                    )
            else:
                rc_hit = select_newest_rc_target(
                    tags,
                    release_branches,
                    release_baseline_tag=release_baseline_tag,
                    tested_rc_tags=tested_rcs,
                    rc_tag_states=state["rc_tag_states"],
                    rc_branch_states=state["rc_branch_states"],
                    rc_line_revoked_shas=state.get("rc_line_revoked_shas"),
                    include_failed=force,
                )
                if rc_hit is None:
                    noop_controls = _collect_verified_control_shas(
                        state,
                        verified_pin=pin,
                        tags=tags,
                        release_branches=release_branches,
                    )
                    noop_override = (
                        override_green if override_green in noop_controls else ""
                    )
                    return {
                        "selected_gear": "rc_qualify",
                        "target_sha": "",
                        "rc_tag": "",
                        "last_green_sha": noop_override
                        or (noop_controls[0] if noop_controls else ""),
                        "fallback_green_shas": "",
                        "last_ppu_tested_sha": last_ppu_tested,
                        "run_cpu": False,
                        "run_ppu": False,
                        "ppu_skip_reason": "not_applicable",
                    }
                rc_tag, target_sha = rc_hit
        clean_target = _require_sha1(target_sha, "rc_target_sha")
        verified_controls = _collect_verified_control_shas(
            state,
            verified_pin=pin,
            rc_ref=rc_tag,
            target_sha=clean_target,
            tags=tags,
            release_branches=release_branches,
        )
        non_target_controls = [s for s in verified_controls if s != clean_target]
        effective_rc_override = (
            override_green if override_green in non_target_controls else ""
        )
        rc_green = effective_rc_override or _resolve_rc_last_green_sha(
            rc_tag,
            target_sha=clean_target,
            tags=tags,
            release_branches=release_branches,
            rc_branch_states=state["rc_branch_states"],
            rc_tag_states=state["rc_tag_states"],
            tested_rc_tags=tested_rcs,
            fallback_green=last_ppu_green,
            state=state,
            verified_pin=pin,
            vllm_repo=vllm_repo,
        )
        fallback_list = [
            rc_green,
            *(s for s in non_target_controls if s != rc_green),
        ]
        return {
            "selected_gear": "rc_qualify",
            "target_sha": clean_target,
            "rc_tag": rc_tag,
            "last_green_sha": rc_green,
            "fallback_green_shas": ",".join(
                s for s in fallback_list if SHA1_RE.fullmatch(s)
            ),
            "last_ppu_tested_sha": last_ppu_tested,
            "run_cpu": True,
            "run_ppu": True,
            "ppu_skip_reason": "none",
        }

    if effective_gear == "daily_pin_bump":
        bump_excluded = _collect_excluded_control_shas(
            state,
            tags=tags,
            release_branches=release_branches,
        )
        bump_controls = _collect_verified_control_shas(
            state,
            verified_pin=pin,
            tags=tags,
            release_branches=release_branches,
        )
        bump_override = override_green if override_green in bump_controls else ""
        history = list(probe_history)
        if not history:
            history = list(state.get("probe_history") or [])
        if not history and state.get("last_probe_record"):
            history = [state["last_probe_record"]]
        newer_than_pin: list[dict[str, Any]] = []
        for rec in history:
            if not isinstance(rec, dict) or rec.get("gear") != "probe":
                continue
            if str(rec.get("upstream_sha", "")).strip() == pin:
                break
            newer_than_pin.append(rec)
        if override:
            candidate = _require_sha1(override, "upstream_ref")
            if candidate == pin:
                bump_sha = ""
            else:
                matching = [
                    rec
                    for rec in newer_than_pin
                    if str(rec.get("upstream_sha", "")).strip() == candidate
                ]
                bump_sha = (
                    select_pin_bump_candidate(
                        matching,
                        verified_pin=pin,
                        excluded_shas=bump_excluded,
                        vllm_repo=vllm_repo,
                        main_sha=str(main_sha or ""),
                    )
                    or ""
                )
        else:
            bump_sha = (
                select_pin_bump_candidate(
                    newer_than_pin,
                    verified_pin=pin,
                    excluded_shas=bump_excluded,
                    vllm_repo=vllm_repo,
                    main_sha=str(main_sha or ""),
                )
                or ""
            )
        return {
            "selected_gear": "daily_pin_bump",
            "target_sha": bump_sha,
            "rc_tag": "",
            "last_green_sha": bump_override
            or (bump_controls[0] if bump_controls else ""),
            "fallback_green_shas": "",
            "last_ppu_tested_sha": last_ppu_tested,
            "run_cpu": False,
            "run_ppu": False,
            "ppu_skip_reason": "not_applicable",
        }

    # Gear 1: post-merge probe
    if override:
        target_sha = _require_sha1(override, "upstream_ref")
    else:
        if not main_sha:
            raise ValueError("git ls-remote output did not contain refs/heads/main")
        target_sha = _require_sha1(main_sha, "main_sha")

    probe_excluded = _collect_excluded_control_shas(
        state,
        target_sha=target_sha,
        tags=tags,
        release_branches=release_branches,
    )
    probe_controls = _collect_verified_control_shas(
        state,
        verified_pin=pin,
        target_sha=target_sha,
        tags=tags,
        release_branches=release_branches,
    )
    effective_probe_override = (
        override_green if override_green in probe_controls else ""
    )
    if effective_probe_override:
        last_ppu_green = effective_probe_override
        last_ppu_tested = effective_probe_override
        state["last_ppu_status"] = "pass"

    shared_cpu_failed = set(state.get("shared_cpu_failed_shas") or ())
    # Skip checks ONLY when target_sha and the latest CPU/PPU outcomes are GREEN and unrevoked.
    run_cpu = bool(
        force
        or target_sha != last_cpu_green
        or state["last_cpu_status"] != "pass"
        or target_sha in shared_cpu_failed
    )
    ppu_diff_base_ok = (not changed_files_base_sha) or (
        _require_sha1(changed_files_base_sha, "changed_files_base_sha")
        == last_ppu_green
    )
    if (
        not force
        and not run_cpu
        and target_sha == last_cpu_green
        and target_sha == last_ppu_green
        and state["last_cpu_status"] == "pass"
        and state["last_ppu_status"] == "pass"
        and target_sha not in probe_excluded
    ):
        run_ppu = False
        skip_reason = "already_tested"
    elif (
        not force
        and state["last_ppu_status"] == "pass"
        and last_ppu_tested == last_ppu_green
        and last_ppu_green not in probe_excluded
        and ppu_diff_base_ok
        and should_skip_ppu_for_paths(changed_paths_since_ppu)
    ):
        run_ppu = False
        skip_reason = "allowlisted_non_ppu_changes"
    elif active_pr_runs > 0 and not force:
        run_ppu = False
        skip_reason = "deferred_for_sail_pr_ci"
    else:
        run_ppu = True
        skip_reason = "none"

    non_target_probe_controls = [s for s in probe_controls if s != target_sha]
    probe_green = (
        effective_probe_override
        if (effective_probe_override and effective_probe_override != target_sha)
        else (
            non_target_probe_controls[0]
            if non_target_probe_controls
            else (
                effective_probe_override
                or (probe_controls[0] if probe_controls else "")
            )
        )
    )
    probe_fallbacks = [
        probe_green,
        *(s for s in non_target_probe_controls if s != probe_green),
    ]
    return {
        "selected_gear": "probe",
        "target_sha": target_sha,
        "rc_tag": "",
        "last_green_sha": probe_green,
        "fallback_green_shas": ",".join(
            s for s in probe_fallbacks if SHA1_RE.fullmatch(s)
        ),
        "last_ppu_tested_sha": last_ppu_tested,
        "run_cpu": run_cpu,
        "run_ppu": run_ppu,
        "ppu_skip_reason": skip_reason,
    }


def evaluate_workflow_outcome(
    *,
    selected_gear: str,
    target_sha: str,
    run_cpu: bool,
    run_ppu: bool,
    ppu_skip_reason: str,
    ops_image: str = "",
    route_result: str = "success",
    build_result: str = "skipped",
    cpu_result: str = "skipped",
    ppu_smoke_result: str = "skipped",
    rc_pin_result: str = "skipped",
    rc_fast_result: str = "skipped",
    rc_model_result: str = "skipped",
    rc_ops_select_result: str = "skipped",
    rc_ops_result: str = "skipped",
    bump_result: str = "skipped",
) -> dict[str, str]:
    """Deterministically reduce workflow job results without turning skipped/red into pass."""
    safe_skip = ppu_skip_reason if ppu_skip_reason in VALID_PPU_SKIP_REASONS else "none"
    if route_result != "success":
        return {
            "status": "infra_error",
            "cpu_status": "not_run",
            "ppu_status": "not_run",
            "ppu_skip_reason": safe_skip,
            "failure_stage": "route",
        }
    if selected_gear == "daily_pin_bump":
        if not target_sha:
            st = "noop"
        elif bump_result == "success":
            st = "pass"
        else:
            st = "infra_error"
        return {
            "status": st,
            "cpu_status": "not_run",
            "ppu_status": "not_run",
            "ppu_skip_reason": "not_applicable",
            "failure_stage": "route" if st == "infra_error" else "none",
        }
    if not target_sha:
        return {
            "status": "noop",
            "cpu_status": "not_run",
            "ppu_status": "not_run",
            "ppu_skip_reason": safe_skip,
            "failure_stage": "none",
        }
    if selected_gear == "probe":
        if not run_cpu and not run_ppu and safe_skip == "already_tested":
            return {
                "status": "noop",
                "cpu_status": "not_run",
                "ppu_status": "skipped",
                "ppu_skip_reason": "already_tested",
                "failure_stage": "none",
            }
        if (run_cpu or run_ppu) and build_result != "success":
            return {
                "status": "fail",
                "cpu_status": "not_run",
                "ppu_status": "not_run",
                "ppu_skip_reason": safe_skip,
                "failure_stage": "wheel_build",
            }
        cpu_st = (
            ("pass" if cpu_result == "success" else "fail") if run_cpu else "not_run"
        )
        if cpu_st == "fail":
            return {
                "status": "fail",
                "cpu_status": "fail",
                "ppu_status": "skipped",
                "ppu_skip_reason": "cpu_fast_check_failed",
                "failure_stage": "plugin_registration",
            }
        if run_ppu:
            ppu_st = "pass" if ppu_smoke_result == "success" else "fail"
            return {
                "status": "pass" if ppu_st == "pass" else "fail",
                "cpu_status": cpu_st,
                "ppu_status": ppu_st,
                "ppu_skip_reason": "none",
                "failure_stage": "none" if ppu_st == "pass" else "ppu_probe_smoke",
            }
        return {
            "status": "deferred" if safe_skip == "deferred_for_sail_pr_ci" else "pass",
            "cpu_status": cpu_st,
            "ppu_status": "skipped",
            "ppu_skip_reason": safe_skip,
            "failure_stage": "none",
        }

    # Gear 3: rc_qualify requires EVERY mandatory gate (and ops when requested) to be "success".
    if build_result != "success":
        return {
            "status": "fail",
            "cpu_status": "not_run",
            "ppu_status": "not_run",
            "ppu_skip_reason": "none",
            "failure_stage": "wheel_build",
        }
    if cpu_result != "success":
        return {
            "status": "fail",
            "cpu_status": "fail",
            "ppu_status": "skipped",
            "ppu_skip_reason": "cpu_fast_check_failed",
            "failure_stage": "plugin_registration",
        }
    rc_gates: list[tuple[str, str]] = [
        ("rc_pin_ut", rc_pin_result),
        ("rc_fast_sanity", rc_fast_result),
        ("rc_model_sanity", rc_model_result),
    ]
    if ops_image.strip():
        rc_gates.append(("rc_ops_tests", rc_ops_select_result))
        rc_gates.append(("rc_ops_tests", rc_ops_result))
    for stage_name, result_val in rc_gates:
        if result_val != "success":
            return {
                "status": "fail",
                "cpu_status": "pass",
                "ppu_status": "fail",
                "ppu_skip_reason": "none",
                "failure_stage": stage_name,
            }
    return {
        "status": "pass",
        "cpu_status": "pass",
        "ppu_status": "pass",
        "ppu_skip_reason": "none",
        "failure_stage": "none",
    }


def advance_cursor(
    cursor: dict[str, Any] | None,
    *,
    verified_pin: str,
    selected_gear: str,
    target_sha: str,
    rc_tag: str = "",
    status: str = "pass",
    cpu_status: str = "not_run",
    ppu_status: str = "not_run",
    ppu_skip_reason: str = "none",
    patch_fingerprints: dict[str, str] | None = None,
    superseded_probe_frontier: bool = False,
) -> dict[str, Any]:
    """Update persistent cursor without recording hardware pass for skipped SHAs."""
    pin = _require_sha1(verified_pin, "verified_pin")
    updated = _sanitize_cursor(cursor, verified_pin=pin)
    if status == "noop" or not target_sha or not SHA1_RE.fullmatch(target_sha):
        return updated

    if selected_gear == "daily_pin_bump":
        return updated

    if selected_gear == "rc_qualify":
        # RC qualification runs on release-branch commits and must NEVER mutate
        # main-branch CPU/PPU cursors, patch fingerprints, or bisection baselines.
        rc_passed = status == "pass" and cpu_status == "pass" and ppu_status == "pass"
        rc_st = (
            "pass"
            if rc_passed
            else ("infra_error" if status == "infra_error" else "fail")
        )
        rc_clean = _normalize_rc_ref(rc_tag)
        tag_match = RC_TAG_RE.fullmatch(rc_clean) if rc_clean else None
        branch_match = (
            RELEASE_BRANCH_RE.fullmatch(rc_clean)
            if (rc_clean and not tag_match)
            else None
        )

        def _seed_branch_green_history(
            br_key: str, maj: int, minr: int, *, exclude_tag: str = ""
        ) -> list[str]:
            prior = updated["rc_branch_states"].get(br_key) or {}
            prior_rev = {
                str(x).strip()
                for x in (prior.get("revoked_shas") or ())
                if SHA1_RE.fullmatch(str(x or "").strip())
            }
            if prior.get("status") not in (None, "pass"):
                ps_bad = str(prior.get("sha") or "").strip()
                if SHA1_RE.fullmatch(ps_bad):
                    prior_rev.add(ps_bad)
            for t_name, t_entry in updated["rc_tag_states"].items():
                tm = RC_TAG_RE.fullmatch(t_name)
                if (
                    tm
                    and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                    and isinstance(t_entry, dict)
                    and t_entry.get("status") != "pass"
                ):
                    ts_bad = str(t_entry.get("sha") or "").strip()
                    if SHA1_RE.fullmatch(ts_bad):
                        prior_rev.add(ts_bad)
            seeded: list[str] = []
            raw_h = prior.get("green_history")
            if isinstance(raw_h, list):
                for item in raw_h:
                    s = str(item or "").strip()
                    if SHA1_RE.fullmatch(s) and s not in prior_rev and s not in seeded:
                        seeded.append(s)
            lg = str(prior.get("last_green_sha") or "").strip()
            if SHA1_RE.fullmatch(lg) and lg not in prior_rev and lg not in seeded:
                seeded.append(lg)
            if prior.get("status") == "pass":
                ps = str(prior.get("sha") or "").strip()
                if SHA1_RE.fullmatch(ps) and ps not in prior_rev and ps not in seeded:
                    seeded.append(ps)
            tag_greens: list[tuple[int, str]] = []
            for t_name, t_entry in updated["rc_tag_states"].items():
                if t_name == exclude_tag:
                    continue
                tm = RC_TAG_RE.fullmatch(t_name)
                if (
                    tm
                    and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                    and isinstance(t_entry, dict)
                    and t_entry.get("status") == "pass"
                ):
                    ts = str(t_entry.get("sha") or "").strip()
                    if SHA1_RE.fullmatch(ts) and ts not in prior_rev:
                        tag_greens.append((int(tm.group(3)), ts))
            tag_greens.sort(key=lambda item: item[0], reverse=True)
            for _, ts in tag_greens:
                if ts not in seeded:
                    seeded.append(ts)
            return seeded

        def _prior_revoked_list(br_key: str) -> list[str]:
            prior = updated["rc_branch_states"].get(br_key) or {}
            out: list[str] = []
            for item in prior.get("revoked_shas") or ():
                s = str(item or "").strip()
                if SHA1_RE.fullmatch(s) and s not in out:
                    out.append(s)
            if prior.get("status") not in (None, "pass"):
                ps = str(prior.get("sha") or "").strip()
                if SHA1_RE.fullmatch(ps) and ps not in out:
                    out.insert(0, ps)
            return out

        def _revoke_line_sha(maj: int, minr: int, bad_sha: str, bad_st: str) -> None:
            demoted: set[str] = set()
            for t_name, t_entry in updated["rc_tag_states"].items():
                tm = RC_TAG_RE.fullmatch(t_name)
                if (
                    tm
                    and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                    and isinstance(t_entry, dict)
                    and t_entry.get("sha") == bad_sha
                ):
                    if t_entry.get("status") == "pass" or bad_st == "fail":
                        t_entry["status"] = bad_st
                    demoted.add(t_name)
            if demoted:
                updated["tested_rc_tags"] = [
                    t for t in updated["tested_rc_tags"] if t not in demoted
                ]

        def _restore_line_sha(maj: int, minr: int, good_sha: str) -> None:
            for t_name, t_entry in updated["rc_tag_states"].items():
                tm = RC_TAG_RE.fullmatch(t_name)
                if (
                    tm
                    and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                    and isinstance(t_entry, dict)
                    and t_entry.get("sha") == good_sha
                ):
                    t_entry["status"] = "pass"
                    if t_name not in updated["tested_rc_tags"]:
                        updated["tested_rc_tags"].append(t_name)
            br_k = f"releases/v{maj}.{minr}.0"
            if br_k in updated["rc_branch_states"]:
                b_cur = updated["rc_branch_states"][br_k]
                if b_cur.get("sha") == good_sha:
                    b_cur["status"] = "pass"
                rem_rev = [s for s in _prior_revoked_list(br_k) if s != good_sha]
                if rem_rev:
                    b_cur["revoked_shas"] = rem_rev
                else:
                    b_cur.pop("revoked_shas", None)
            line_k = f"v{maj}.{minr}.0"
            if (
                "rc_line_revoked_shas" in updated
                and line_k in updated["rc_line_revoked_shas"]
            ):
                rem_lk = [
                    s for s in updated["rc_line_revoked_shas"][line_k] if s != good_sha
                ]
                if rem_lk:
                    updated["rc_line_revoked_shas"][line_k] = rem_lk
                else:
                    updated["rc_line_revoked_shas"].pop(line_k, None)
                if not updated["rc_line_revoked_shas"]:
                    updated.pop("rc_line_revoked_shas", None)

        if not rc_clean and SHA1_RE.fullmatch(target_sha):
            rc_clean = _resolve_unique_rc_ref_for_sha(
                target_sha,
                tags={},
                release_branches={},
                release_baseline_tag="",
                rc_branch_states=updated["rc_branch_states"],
                rc_tag_states=updated["rc_tag_states"],
                rc_line_revoked_shas=updated.get("rc_line_revoked_shas"),
                tested_rc_tags=updated["tested_rc_tags"],
            )
            tag_match = RC_TAG_RE.fullmatch(rc_clean) if rc_clean else None
            branch_match = (
                RELEASE_BRANCH_RE.fullmatch(rc_clean)
                if (rc_clean and not tag_match)
                else None
            )

        def _update_line_commit_order(
            maj: int,
            minr: int,
            *,
            new_sha: str,
            new_rc_num: int | None = None,
            exclude_tag: str = "",
        ) -> list[str]:
            line_k = f"v{maj}.{minr}.0"
            br_k = f"releases/v{maj}.{minr}.0"
            order_map = dict(updated.get("rc_line_commit_order") or {})
            order: list[str] = [
                s
                for s in (order_map.get(line_k) or ())
                if SHA1_RE.fullmatch(str(s or "").strip())
            ]
            seed_snap: list[str] = []

            def _seed_add(cand: str) -> None:
                c = cand.strip()
                if SHA1_RE.fullmatch(c) and c not in seed_snap:
                    seed_snap.append(c)

            prior_br = updated["rc_branch_states"].get(br_k) or {}
            prior_sup = [
                str(x).strip()
                for x in (prior_br.get("superseded_shas") or ())
                if SHA1_RE.fullmatch(str(x or "").strip())
            ]
            prior_greens = _seed_branch_green_history(
                br_k, maj, minr, exclude_tag=exclude_tag
            )
            if isinstance(prior_br, dict) and prior_br.get("status") not in (
                None,
                "pass",
            ):
                ps_bad = str(prior_br.get("sha") or "").strip()
                if ps_bad and ps_bad not in prior_sup:
                    _seed_add(ps_bad)
            if prior_greens:
                _seed_add(prior_greens[0])
            for s_sup in prior_sup:
                _seed_add(s_sup)
            for s_rev in _prior_revoked_list(br_k):
                _seed_add(s_rev)
            for s_grn in prior_greens[1:]:
                _seed_add(s_grn)
            all_line_tags: list[tuple[int, str]] = []
            for t_name, t_ent in updated["rc_tag_states"].items():
                if t_name == exclude_tag:
                    continue
                tm = RC_TAG_RE.fullmatch(t_name)
                if (
                    tm
                    and (int(tm.group(1)), int(tm.group(2))) == (maj, minr)
                    and isinstance(t_ent, dict)
                ):
                    ts = str(t_ent.get("sha") or "").strip()
                    if SHA1_RE.fullmatch(ts):
                        all_line_tags.append((int(tm.group(3)), ts))
            all_line_tags.sort(key=lambda item: item[0], reverse=True)
            for _, ts in all_line_tags:
                _seed_add(ts)
            for ls in (updated.get("rc_line_revoked_shas") or {}).get(line_k) or ():
                _seed_add(str(ls or ""))
            order = _merge_commit_orders(order, seed_snap)

            if new_sha not in order:
                if new_rc_num is not None:
                    last_higher_idx = -1
                    for idx, existing_sha in enumerate(order):
                        higher = any(
                            rc_n > new_rc_num and ts == existing_sha
                            for rc_n, ts in all_line_tags
                        )
                        if higher:
                            last_higher_idx = idx
                    order.insert(last_higher_idx + 1, new_sha)
                else:
                    order.insert(0, new_sha)

            order_map[line_k] = order
            updated["rc_line_commit_order"] = dict(sorted(order_map.items()))
            return order

        def _ordered_branch_green_history(
            br_k: str,
            maj: int,
            minr: int,
            line_order: Sequence[str],
            *,
            include_sha: str = "",
            exclude_sha: str = "",
        ) -> list[str]:
            greens = set(_seed_branch_green_history(br_k, maj, minr))
            if include_sha and SHA1_RE.fullmatch(include_sha):
                greens.add(include_sha)
            if exclude_sha:
                greens.discard(exclude_sha)
            ordered = [s for s in line_order if s in greens]
            for extra in _seed_branch_green_history(br_k, maj, minr):
                if extra in greens and extra not in ordered:
                    ordered.append(extra)
            return ordered[:20]

        if tag_match:
            major, minor, rc_num = (
                int(tag_match.group(1)),
                int(tag_match.group(2)),
                int(tag_match.group(3)),
            )
            prior_tag = updated["rc_tag_states"].get(rc_clean) or {}
            branch_key = f"releases/v{major}.{minor}.0"
            line_key = f"v{major}.{minor}.0"
            if isinstance(prior_tag, dict):
                old_sha = str(prior_tag.get("sha") or "").strip()
                old_st = str(prior_tag.get("status") or "").strip()
                if (
                    SHA1_RE.fullmatch(old_sha)
                    and old_sha != target_sha
                    and old_st in ("fail", "infra_error")
                ):
                    line_map = dict(updated.get("rc_line_revoked_shas") or {})
                    prev_line_list = list(line_map.get(line_key) or [])
                    line_map[line_key] = [
                        old_sha,
                        *(s for s in prev_line_list if s != old_sha),
                    ]
                    updated["rc_line_revoked_shas"] = dict(sorted(line_map.items()))
                    if branch_key in updated["rc_branch_states"]:
                        br_dis = dict(updated["rc_branch_states"][branch_key])
                        prev_br_rev = _prior_revoked_list(branch_key)
                        br_dis["revoked_shas"] = [
                            old_sha,
                            *(s for s in prev_br_rev if s != old_sha),
                        ]
                        updated["rc_branch_states"][branch_key] = br_dis
            line_order = _update_line_commit_order(
                major,
                minor,
                new_sha=target_sha,
                new_rc_num=rc_num,
                exclude_tag=rc_clean,
            )
            updated["rc_tag_states"][rc_clean] = {
                "sha": target_sha,
                "status": rc_st,
            }
            updated["rc_tag_states"] = dict(sorted(updated["rc_tag_states"].items()))
            if rc_passed:
                if rc_clean not in updated["tested_rc_tags"]:
                    updated["tested_rc_tags"].append(rc_clean)
                _restore_line_sha(major, minor, target_sha)
                if branch_key in updated["rc_branch_states"]:
                    br_cur = dict(updated["rc_branch_states"][branch_key])
                    br_cur.pop("superseded_shas", None)
                    hist = _ordered_branch_green_history(
                        branch_key,
                        major,
                        minor,
                        line_order,
                        include_sha=target_sha,
                    )
                    if hist:
                        br_cur["last_green_sha"] = hist[0]
                        br_cur["green_history"] = hist
                    updated["rc_branch_states"][branch_key] = br_cur
            else:
                if rc_clean in updated["tested_rc_tags"]:
                    updated["tested_rc_tags"] = [
                        t for t in updated["tested_rc_tags"] if t != rc_clean
                    ]
                _revoke_line_sha(major, minor, target_sha, rc_st)
                if branch_key in updated["rc_branch_states"]:
                    prev_rev = _prior_revoked_list(branch_key)
                    br_cur = dict(updated["rc_branch_states"][branch_key])
                    br_cur.pop("superseded_shas", None)
                    if br_cur.get("sha") == target_sha and (
                        br_cur.get("status") == "pass" or rc_st == "fail"
                    ):
                        br_cur["status"] = rc_st
                    br_cur["revoked_shas"] = [
                        target_sha,
                        *(s for s in prev_rev if s != target_sha),
                    ]
                    updated["rc_branch_states"][branch_key] = br_cur
                    hist = _ordered_branch_green_history(
                        branch_key,
                        major,
                        minor,
                        line_order,
                        exclude_sha=target_sha,
                    )
                    br_cur.pop("last_green_sha", None)
                    br_cur.pop("green_history", None)
                    if hist:
                        br_cur["last_green_sha"] = hist[0]
                        br_cur["green_history"] = hist
                    updated["rc_branch_states"][branch_key] = br_cur
        elif branch_match:
            major, minor = int(branch_match.group(1)), int(branch_match.group(2))
            prior_br_cur = updated["rc_branch_states"].get(rc_clean) or {}
            prior_tip_sha = (
                str(prior_br_cur.get("sha") or "").strip()
                if isinstance(prior_br_cur, dict)
                else ""
            )
            prior_tip_st = (
                str(prior_br_cur.get("status") or "pass")
                if isinstance(prior_br_cur, dict)
                else "pass"
            )
            prev_rev = _prior_revoked_list(rc_clean)
            line_order = _update_line_commit_order(
                major,
                minor,
                new_sha=target_sha,
            )
            is_older_than_tip = (
                SHA1_RE.fullmatch(prior_tip_sha) is not None
                and prior_tip_sha != target_sha
                and prior_tip_sha in line_order
                and line_order.index(target_sha) > line_order.index(prior_tip_sha)
            )
            if rc_passed:
                _restore_line_sha(major, minor, target_sha)
                rem_rev = [s for s in prev_rev if s != target_sha]
                hist = _ordered_branch_green_history(
                    rc_clean,
                    major,
                    minor,
                    line_order,
                    include_sha=target_sha,
                )
                keep_tip_sha = prior_tip_sha if is_older_than_tip else target_sha
                keep_tip_st = prior_tip_st if is_older_than_tip else "pass"
                br_entry: dict[str, Any] = {
                    "sha": keep_tip_sha
                    if SHA1_RE.fullmatch(keep_tip_sha)
                    else target_sha,
                    "status": keep_tip_st
                    if keep_tip_st in ("pass", "fail", "infra_error")
                    else "pass",
                    "last_green_sha": hist[0],
                    "green_history": hist,
                }
                if rem_rev:
                    br_entry["revoked_shas"] = rem_rev
            else:
                _revoke_line_sha(major, minor, target_sha, rc_st)
                new_rev = [
                    target_sha,
                    *(s for s in prev_rev if s != target_sha),
                ]
                keep_tip_sha = prior_tip_sha if is_older_than_tip else target_sha
                keep_tip_st = prior_tip_st if is_older_than_tip else rc_st
                br_Seed = dict(prior_br_cur) if isinstance(prior_br_cur, dict) else {}
                br_Seed["revoked_shas"] = new_rev
                updated["rc_branch_states"][rc_clean] = br_Seed
                hist = _ordered_branch_green_history(
                    rc_clean,
                    major,
                    minor,
                    line_order,
                    exclude_sha=target_sha,
                )
                br_entry = {
                    "sha": keep_tip_sha
                    if SHA1_RE.fullmatch(keep_tip_sha)
                    else target_sha,
                    "status": keep_tip_st
                    if keep_tip_st in ("pass", "fail", "infra_error")
                    else rc_st,
                    "revoked_shas": new_rev,
                }
                if hist:
                    br_entry["last_green_sha"] = hist[0]
                    br_entry["green_history"] = hist
            updated["rc_branch_states"][rc_clean] = br_entry
            updated["rc_branch_states"] = dict(
                sorted(updated["rc_branch_states"].items())
            )
        else:
            rc_sha_map = dict(updated.get("rc_sha_states") or {})
            if rc_passed:
                rc_sha_map.pop(target_sha, None)
            else:
                prev_sha_st = rc_sha_map.get(target_sha)
                if prev_sha_st != "fail" or rc_st == "fail":
                    rc_sha_map[target_sha] = rc_st
            if rc_sha_map:
                updated["rc_sha_states"] = dict(sorted(rc_sha_map.items()))
            else:
                updated.pop("rc_sha_states", None)

        if rc_passed and "rc_sha_states" in updated:
            rc_sha_map = dict(updated["rc_sha_states"])
            rc_sha_map.pop(target_sha, None)
            if rc_sha_map:
                updated["rc_sha_states"] = dict(sorted(rc_sha_map.items()))
            else:
                updated.pop("rc_sha_states", None)

        shared_failed = [
            s
            for s in (updated.get("shared_cpu_failed_shas") or ())
            if SHA1_RE.fullmatch(str(s or "").strip())
        ]
        if cpu_status == "pass":
            shared_failed = [s for s in shared_failed if s != target_sha]
        elif not rc_passed:
            shared_failed = [
                target_sha,
                *(s for s in shared_failed if s != target_sha),
            ][:MAX_EPHEMERAL_REVOKED_SHAS]
        if shared_failed:
            updated["shared_cpu_failed_shas"] = shared_failed
        else:
            updated.pop("shared_cpu_failed_shas", None)

        protected_rc_controls = _protected_control_shas_in_cursor(
            updated, verified_pin=pin
        )
        for b_ent in updated["rc_branch_states"].values():
            if isinstance(b_ent, dict) and "revoked_shas" in b_ent:
                capped_rev = _cap_revoked_shas(
                    b_ent.get("revoked_shas") or (),
                    protected_rc_controls,
                )
                if capped_rev:
                    b_ent["revoked_shas"] = capped_rev
                    if "superseded_shas" in b_ent:
                        capped_set = set(capped_rev)
                        kept_s = [
                            s
                            for s in (b_ent.get("superseded_shas") or ())
                            if s in capped_set
                        ]
                        if kept_s:
                            b_ent["superseded_shas"] = kept_s
                        else:
                            b_ent.pop("superseded_shas", None)
                else:
                    b_ent.pop("revoked_shas", None)
                    b_ent.pop("superseded_shas", None)
        if "rc_line_revoked_shas" in updated:
            capped_rc_lines: dict[str, list[str]] = {}
            for lk, lv in updated["rc_line_revoked_shas"].items():
                capped_lv = _cap_revoked_shas(lv or (), protected_rc_controls)
                if capped_lv:
                    capped_rc_lines[lk] = capped_lv
            if capped_rc_lines:
                updated["rc_line_revoked_shas"] = dict(sorted(capped_rc_lines.items()))
            else:
                updated.pop("rc_line_revoked_shas", None)
        return updated

    pre_lane_failure = (
        status in ("fail", "infra_error")
        and cpu_status not in ("pass", "fail")
        and ppu_status not in ("pass", "fail")
    )

    prior_history = [dict(r) for r in (updated.get("probe_history") or [])]
    existing_idx = next(
        (
            idx
            for idx, r in enumerate(prior_history)
            if r.get("upstream_sha") == target_sha
        ),
        None,
    )
    is_historical_probe = selected_gear == "probe" and (
        (existing_idx is not None and existing_idx >= 1)
        or (
            superseded_probe_frontier
            and bool(prior_history)
            and prior_history[0].get("upstream_sha") != target_sha
        )
    )

    def _merge_patch_fingerprints() -> None:
        if cpu_status == "pass" and patch_fingerprints:
            for k, v in patch_fingerprints.items():
                assert_zero_sensitive_leaks(k)
                if (
                    OPAQUE_PATCH_KEY_RE.fullmatch(k) or SAFE_SYMBOL_RE.fullmatch(k)
                ) and (v == "additive" or re.fullmatch(r"[0-9a-f]{64}", v)):
                    updated["patch_fingerprints"][k] = v
            updated["patch_fingerprints"] = dict(
                sorted(updated["patch_fingerprints"].items())
            )

    def _older_ppu_green_from(
        records: Sequence[dict[str, str]],
        *,
        excluded: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        for rec_cand in records:
            up_c = str(rec_cand.get("upstream_sha") or "").strip()
            if (
                rec_cand.get("cpu_status") == "pass"
                and rec_cand.get("ppu_status") == "pass"
                and SHA1_RE.fullmatch(up_c)
                and up_c not in excluded
            ):
                return up_c
            gr_c = str(rec_cand.get("last_ppu_green_sha") or "").strip()
            if (
                rec_cand.get("last_ppu_result_status") == "pass"
                and SHA1_RE.fullmatch(gr_c)
                and gr_c != target_sha
                and gr_c not in excluded
            ):
                return gr_c
        return pin if pin not in excluded else ""

    def _older_cpu_green_from(records: Sequence[dict[str, str]]) -> str:
        for rec_cand in records:
            up_c = str(rec_cand.get("upstream_sha") or "").strip()
            if rec_cand.get("cpu_status") == "pass" and SHA1_RE.fullmatch(up_c):
                return up_c
        return pin

    if is_historical_probe:
        if not (
            pre_lane_failure
            or cpu_status in ("pass", "fail", "infra_error")
            or ppu_status in ("pass", "fail", "infra_error")
        ):
            return updated

        if updated["last_cpu_status"] != "pass":
            _merge_patch_fingerprints()

        prior_b = prior_history[existing_idx] if existing_idx is not None else None
        older_slice = (
            prior_history[existing_idx + 1 :]
            if existing_idx is not None
            else prior_history[1:]
        )

        if pre_lane_failure:
            rec_cpu = "infra_error" if status == "infra_error" else "fail"
        elif (
            cpu_status == "not_run"
            and status in ("pass", "deferred")
            and prior_b is not None
            and prior_b.get("cpu_status") == "pass"
        ):
            rec_cpu = "pass"
        else:
            rec_cpu = cpu_status

        older_green = _older_ppu_green_from(older_slice)
        if ppu_status in ("pass", "fail", "infra_error"):
            b_tested = target_sha
            if ppu_status == "pass" and rec_cpu == "pass":
                b_green = target_sha
                b_res = "pass"
            else:
                b_green = older_green
                b_res = "fail"
        elif prior_b is not None:
            b_tested = str(prior_b.get("last_ppu_tested_sha") or older_green)
            prev_b_green = str(prior_b.get("last_ppu_green_sha") or older_green)
            b_green = older_green if prev_b_green == target_sha else prev_b_green
            b_res = (
                "fail"
                if (pre_lane_failure and b_tested == target_sha)
                else str(prior_b.get("last_ppu_result_status") or "pass")
            )
            if b_tested == target_sha and b_res != "pass":
                b_green = older_green
        else:
            base_rec = older_slice[0] if older_slice else None
            b_tested = (
                str(base_rec.get("last_ppu_tested_sha") or older_green)
                if base_rec is not None
                else older_green
            )
            b_green = older_green
            b_res = (
                str(base_rec.get("last_ppu_result_status") or "pass")
                if base_rec is not None
                else "pass"
            )

        hist_rec = {
            "gear": "probe",
            "upstream_sha": target_sha,
            "cpu_status": rec_cpu,
            "ppu_status": ppu_status,
            "ppu_skip_reason": ppu_skip_reason
            if ppu_skip_reason in VALID_PPU_SKIP_REASONS
            else "none",
            "last_ppu_tested_sha": b_tested,
            "last_ppu_green_sha": b_green,
            "last_ppu_result_status": b_res,
        }

        if existing_idx is not None:
            merged_history = [
                *prior_history[:existing_idx],
                hist_rec,
                *prior_history[existing_idx + 1 :],
            ][:20]
            b_pos = existing_idx
        elif len(prior_history) < 20:
            merged_history = [prior_history[0], hist_rec, *prior_history[1:]][:20]
            b_pos = 1
        else:
            merged_history = prior_history[:20]
            b_pos = None

        if b_pos is not None:
            eff_tested, eff_green, eff_res = b_tested, b_green, b_res
            for idx in range(b_pos - 1, -1, -1):
                rec_up = dict(merged_history[idx])
                up_s = rec_up["upstream_sha"]
                p_st = rec_up.get("ppu_status", "not_run")
                c_st = rec_up.get("cpu_status", "not_run")
                if p_st in ("pass", "fail", "infra_error"):
                    eff_tested = up_s
                    if p_st == "pass" and c_st == "pass":
                        eff_green = up_s
                        eff_res = "pass"
                    else:
                        eff_res = "fail"
                    rec_up["last_ppu_tested_sha"] = eff_tested
                    rec_up["last_ppu_green_sha"] = eff_green
                    rec_up["last_ppu_result_status"] = eff_res
                else:
                    if c_st in ("fail", "infra_error") and eff_tested == up_s:
                        eff_res = "fail"
                    rec_up["last_ppu_tested_sha"] = eff_tested
                    rec_up["last_ppu_green_sha"] = eff_green
                    rec_up["last_ppu_result_status"] = eff_res
                merged_history[idx] = rec_up
        elif ppu_status == "pass" and rec_cpu == "pass":
            # Even when >20 newer records exist, if none of them passed PPU and prior green was `pin`,
            # `target_sha` can still serve as fallback PPU green for failed newer records.
            if updated["last_ppu_green_sha"] == pin and not any(
                r.get("cpu_status") == "pass" and r.get("ppu_status") == "pass"
                for r in merged_history
            ):
                updated["last_ppu_green_sha"] = target_sha

        front = merged_history[0]
        updated["probe_history"] = merged_history
        updated["last_probe_record"] = front
        updated["last_ppu_tested_sha"] = front["last_ppu_tested_sha"]
        updated["last_ppu_green_sha"] = front["last_ppu_green_sha"]
        updated["last_ppu_status"] = front["last_ppu_result_status"]
        if front.get("cpu_status") in ("pass", "fail", "infra_error"):
            updated["last_cpu_checked_sha"] = front["upstream_sha"]
            updated["last_cpu_status"] = (
                "pass" if front["cpu_status"] == "pass" else "fail"
            )
        updated["last_cpu_green_sha"] = _older_cpu_green_from(merged_history)
        return updated

    if cpu_status == "pass" and "shared_cpu_failed_shas" in updated:
        rem_shared = [
            s for s in (updated.get("shared_cpu_failed_shas") or ()) if s != target_sha
        ]
        if rem_shared:
            updated["shared_cpu_failed_shas"] = rem_shared
        else:
            updated.pop("shared_cpu_failed_shas", None)

    older_without_target = [
        r for r in prior_history if r.get("upstream_sha") != target_sha
    ]

    if pre_lane_failure:
        updated["last_cpu_checked_sha"] = target_sha
        updated["last_cpu_status"] = "fail"
        if updated["last_cpu_green_sha"] == target_sha:
            updated["last_cpu_green_sha"] = _older_cpu_green_from(older_without_target)
        if updated["last_ppu_tested_sha"] == target_sha:
            updated["last_ppu_status"] = "fail"
        if updated["last_ppu_green_sha"] == target_sha:
            updated["last_ppu_green_sha"] = _older_ppu_green_from(older_without_target)
    elif cpu_status in ("pass", "fail", "infra_error"):
        updated["last_cpu_checked_sha"] = target_sha
        updated["last_cpu_status"] = "pass" if cpu_status == "pass" else "fail"
        if cpu_status == "pass":
            updated["last_cpu_green_sha"] = target_sha
            _merge_patch_fingerprints()
        elif updated["last_cpu_green_sha"] == target_sha:
            updated["last_cpu_green_sha"] = _older_cpu_green_from(older_without_target)
            if updated["last_ppu_green_sha"] == target_sha:
                updated["last_ppu_green_sha"] = _older_ppu_green_from(
                    older_without_target
                )
                if updated["last_ppu_tested_sha"] == target_sha:
                    updated["last_ppu_status"] = "fail"

    # Hardware cursor advances ONLY when device smoke actually executed!
    if ppu_status in ("pass", "fail", "infra_error"):
        updated["last_ppu_tested_sha"] = target_sha
        updated["last_ppu_status"] = (
            "pass"
            if (ppu_status == "pass" and updated["last_cpu_status"] == "pass")
            else "fail"
        )
        if ppu_status == "pass" and updated["last_cpu_status"] == "pass":
            updated["last_ppu_green_sha"] = target_sha
        elif updated["last_ppu_green_sha"] == target_sha:
            updated["last_ppu_green_sha"] = _older_ppu_green_from(older_without_target)
    else:
        active_excluded = _collect_excluded_control_shas(updated)
        if updated["last_ppu_green_sha"] in active_excluded:
            fallback_p = _older_ppu_green_from(
                older_without_target, excluded=active_excluded
            )
            if SHA1_RE.fullmatch(fallback_p):
                updated["last_ppu_green_sha"] = fallback_p
        if updated["last_ppu_tested_sha"] in active_excluded:
            updated["last_ppu_status"] = "fail"

    if selected_gear == "probe" and (
        pre_lane_failure
        or cpu_status in ("pass", "fail", "infra_error")
        or ppu_status in ("pass", "fail", "infra_error")
    ):
        if pre_lane_failure:
            rec_cpu = "infra_error" if status == "infra_error" else "fail"
        elif (
            cpu_status == "not_run"
            and status in ("pass", "deferred")
            and updated["last_cpu_green_sha"] == target_sha
            and updated["last_cpu_status"] == "pass"
        ):
            rec_cpu = "pass"
        else:
            rec_cpu = cpu_status

        new_rec = {
            "gear": "probe",
            "upstream_sha": target_sha,
            "cpu_status": rec_cpu,
            "ppu_status": ppu_status,
            "ppu_skip_reason": ppu_skip_reason
            if ppu_skip_reason in VALID_PPU_SKIP_REASONS
            else "none",
            "last_ppu_tested_sha": updated["last_ppu_tested_sha"],
            "last_ppu_green_sha": updated["last_ppu_green_sha"],
            "last_ppu_result_status": updated["last_ppu_status"],
        }
        updated["last_probe_record"] = new_rec
        merged_history: list[dict[str, str]] = [new_rec]
        for old_rec in updated.get("probe_history") or []:
            if old_rec.get("upstream_sha") == target_sha:
                continue
            item = dict(old_rec)
            if (
                item.get("ppu_status") == "skipped"
                and item.get("last_ppu_green_sha") == target_sha
                and updated["last_ppu_status"] != "pass"
            ):
                item["last_ppu_tested_sha"] = updated["last_ppu_tested_sha"]
                item["last_ppu_result_status"] = "fail"
            merged_history.append(item)
            if len(merged_history) >= 20:
                break
        updated["probe_history"] = merged_history
    return updated


# ---------------------------------------------------------------------------
# 2. Tier-A Kernel Manifest & Binding Schema Checker (stdlib, no GPU/sailify)
# ---------------------------------------------------------------------------

_OPS_DEF_RE = re.compile(
    r'\b[A-Za-z_][A-Za-z0-9_]*\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\s*[,)]',
    re.DOTALL,
)
_STRING_LIT_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def extract_ops_defs(cpp_text: str) -> dict[str, str]:
    """Extract ``{op_name: normalized_schema}`` from C++ TorchLibrary ``.def(...)`` calls."""
    schemas: dict[str, str] = {}
    for match in _OPS_DEF_RE.finditer(cpp_text):
        concatenated = "".join(_STRING_LIT_RE.findall(match.group(1)))
        normalized = " ".join(concatenated.split())
        if "(" not in normalized or "->" not in normalized:
            continue
        op_name = normalized.split("(", 1)[0].strip()
        if op_name:
            schemas[op_name] = normalized
    return schemas


def check_kernel_manifest_compatibility(
    vllm_src: Path, sail_root: Path = REPO_ROOT
) -> list[str]:
    """Check Tier-A kernel source existence and C++ TorchLibrary schema parity."""
    manifest_path = sail_root / "csrc" / "upstream" / "manifest.toml"
    manifest = _load_toml(manifest_path)
    compiled_tiers = set(manifest.get("compiled_tiers") or ["a"])
    problems: list[str] = []

    compiled_ops: set[str] = set()
    for entry in manifest.get("entry", []):
        if entry.get("tier") not in compiled_tiers:
            continue
        rel_path = entry["path"]
        if not (vllm_src / rel_path).is_file():
            problems.append(f"missing_tier_a_file:{rel_path}")
        compiled_ops.update(entry.get("ops") or [])

    binding_pairs = (
        ("csrc/torch_bindings.cpp", "csrc/upstream/torch_bindings.cpp"),
        ("csrc/moe/torch_bindings.cpp", "csrc/upstream/moe/torch_bindings.cpp"),
    )
    vendored_schemas: dict[str, str] = {}
    upstream_schemas: dict[str, str] = {}
    for up_rel, sail_rel in binding_pairs:
        sail_file = sail_root / sail_rel
        up_file = vllm_src / up_rel
        if sail_file.is_file():
            vendored_schemas.update(
                extract_ops_defs(sail_file.read_text(encoding="utf-8"))
            )
        else:
            problems.append(f"missing_vendored_binding_file:{sail_rel}")
        if not up_file.is_file():
            problems.append(f"missing_binding_file:{up_rel}")
        else:
            upstream_schemas.update(
                extract_ops_defs(up_file.read_text(encoding="utf-8"))
            )

    for op_key in sorted(compiled_ops):
        base_op = op_key.split(".", 1)[0]
        if base_op not in vendored_schemas:
            problems.append(f"unverified_vendored_op_schema:{base_op}")
            continue
        if base_op not in upstream_schemas:
            problems.append(f"missing_op_schema:{base_op}")
        elif upstream_schemas[base_op] != vendored_schemas[base_op]:
            problems.append(f"changed_op_schema:{base_op}")

    return problems


# ---------------------------------------------------------------------------
# 3. Control-First Binary Search Bisection on (last_green_sha..red_sha]
# ---------------------------------------------------------------------------


def bisect_first_failing_sha(
    *,
    last_green_sha: str,
    red_sha: str,
    commits: Sequence[str],
    probe_fn: Callable[[str], ProbeOutcome],
    verify_baseline_first: bool = True,
    initial_red_outcome: ProbeOutcome | None = None,
) -> dict[str, Any]:
    """Locate the exact first failing commit in ``(last_green_sha..red_sha]``.

    ``commits`` must be ordered oldest-to-newest along ``--first-parent`` and end
    with ``red_sha``. ``red_sha`` is already known bad from the window probe, so
    binary search probes ``commits[:-1]`` in at most ``ceil(log2(len(commits)))``
    steps after optional control verification of ``last_green_sha``.
    """
    green = _require_sha1(last_green_sha, "last_green_sha")
    red = _require_sha1(red_sha, "red_sha")
    ordered = [_require_sha1(c, "commit") for c in commits]
    if not ordered or ordered[-1] != red:
        raise ValueError("commits must be non-empty and end with red_sha")

    history: list[dict[str, str]] = []
    if verify_baseline_first:
        baseline_res = probe_fn(green)
        history.append(
            {
                "sha": green,
                "role": "baseline_control",
                "status": baseline_res.status,
                "stage": _sanitize_symbol(baseline_res.stage, "baseline_control"),
            }
        )
        if baseline_res.status != "pass":
            return {
                "bisect_status": "baseline_control_failed",
                "first_failing_sha": "",
                "last_good_sha": "",
                "steps_executed": len(history),
                "probe_steps_on_range": 0,
                "winning_outcome": baseline_res,
                "history": history,
            }

    best_bad_outcome = initial_red_outcome or ProbeOutcome(
        status="fail", stage="ppu_probe_smoke", target_signature="window_head"
    )
    if len(ordered) == 1:
        return {
            "bisect_status": "converged",
            "first_failing_sha": red,
            "last_good_sha": green,
            "steps_executed": len(history),
            "probe_steps_on_range": 0,
            "winning_outcome": best_bad_outcome,
            "history": history,
        }

    lo = 0
    hi = len(ordered) - 1  # ordered[hi] == red is already known bad
    range_probes = 0
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = ordered[mid]
        outcome = probe_fn(candidate)
        range_probes += 1
        history.append(
            {
                "sha": candidate,
                "role": "bisect_probe",
                "status": outcome.status,
                "stage": _sanitize_symbol(outcome.stage, "none"),
            }
        )
        if outcome.status == "infra_error":
            return {
                "bisect_status": "aborted_infra_error",
                "first_failing_sha": "",
                "last_good_sha": ordered[lo - 1] if lo > 0 else green,
                "steps_executed": len(history),
                "probe_steps_on_range": range_probes,
                "winning_outcome": outcome,
                "history": history,
            }
        if outcome.status == "pass":
            lo = mid + 1
        else:
            hi = mid
            best_bad_outcome = outcome

    first_bad = ordered[lo]
    last_good = ordered[lo - 1] if lo > 0 else green
    return {
        "bisect_status": "converged",
        "first_failing_sha": first_bad,
        "last_good_sha": last_good,
        "steps_executed": len(history),
        "probe_steps_on_range": range_probes,
        "winning_outcome": best_bad_outcome,
        "history": history,
    }


# ---------------------------------------------------------------------------
# 4. Dual-Track Root-Cause Routing, Non-PPU Repro Gate & Sanitized Templates
# ---------------------------------------------------------------------------


def sanitize_github_login(value: str | None) -> str:
    """Validate a public GitHub user login; return ``""`` if invalid, bot, or sensitive."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned or not GITHUB_LOGIN_RE.fullmatch(cleaned):
        return ""
    try:
        assert_zero_sensitive_leaks(cleaned)
    except RedactionViolationError:
        return ""
    return cleaned


def parse_commit_author_login(payload: Any, *, expected_sha: str = "") -> str:
    """Extract a validated GitHub login from public ``vllm-project/vllm`` commit JSON."""
    if not isinstance(payload, dict):
        return ""
    if expected_sha:
        sha_in_payload = payload.get("sha")
        if (
            isinstance(sha_in_payload, str)
            and sha_in_payload.strip()
            and sha_in_payload.strip() != expected_sha
        ):
            return ""

    author = payload.get("author")
    if isinstance(author, dict):
        author_type = str(author.get("type", "User"))
        if author_type != "User":
            return ""
        login = sanitize_github_login(author.get("login"))
        if login:
            return login

    commit_obj = payload.get("commit")
    if isinstance(commit_obj, dict):
        commit_author = commit_obj.get("author")
        if isinstance(commit_author, dict):
            email = commit_author.get("email")
            if isinstance(email, str):
                match = GITHUB_NOREPLY_EMAIL_RE.fullmatch(email.strip())
                if match:
                    return sanitize_github_login(match.group(1))
    return ""


def resolve_upstream_commit_author(
    first_failing_sha: str,
    *,
    repo: str = DEFAULT_UPSTREAM_REPO,
    commit_metadata_path: Path | None = None,
    fetch_remote: bool = False,
    github_token: str = "",
    timeout_sec: int = 10,
) -> str:
    """Resolve ``first_failing_sha`` author GitHub login from public metadata.

    Degrades gracefully to ``""`` (SHA-only) when metadata is missing, invalid,
    belongs to a bot, fails redaction, or cannot be fetched.
    """
    sha = _require_sha1(first_failing_sha, "first_failing_sha")
    if repo != DEFAULT_UPSTREAM_REPO:
        return ""

    if commit_metadata_path is not None and commit_metadata_path.is_file():
        try:
            payload = json.loads(commit_metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return parse_commit_author_login(payload, expected_sha=sha)

    if not fetch_remote:
        return ""

    url = f"https://api.github.com/repos/{DEFAULT_UPSTREAM_REPO}/commits/{sha}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-sail-post-merge-ci",
    }
    token = github_token.strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read(256 * 1024).decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except (OSError, ValueError, urllib.error.URLError):
        return ""
    return parse_commit_author_login(payload, expected_sha=sha)


def select_repro_kind(stage: str, target_signature: str = "") -> str:
    """Map ``(stage, target_signature)`` to a hardware-free reproducer kind."""
    sig = target_signature.strip()
    sig_lower = sig.lower()
    if stage == "wheel_build" or sig in {
        "VLLM_TARGET_DEVICE=empty",
        "vllm_empty_device_build",
        "rust_frontend_build",
    }:
        return "cpu_empty_build"
    if (
        "register_oot" in sig_lower
        or "customop" in sig_lower
        or "custom_op" in sig_lower
    ):
        return "register_oot"
    if (
        "modelregistry" in sig_lower
        or "register_model" in sig_lower
        or "models.registry" in sig_lower
    ):
        return "model_registry"
    if (
        "platform" in sig_lower
        or "attentionbackend" in sig_lower
        or "vllm.platforms" in sig_lower
    ):
        return "platform_plugin"
    return "load_general_plugins"


def build_stub_plugin_spec(repro_kind: str) -> dict[str, str]:
    """Return the single-source inline stub plugin files and check script."""
    if repro_kind not in VALID_REPRO_KINDS:
        raise ValueError(f"unsupported repro_kind: {repro_kind!r}")

    if repro_kind == "cpu_empty_build":
        return {
            "mode": "cpu_empty_build",
            "repro_kind": "cpu_empty_build",
            "entry_points": "",
            "plugin_module": "",
            "check_script": (
                "import os\n"
                "from pathlib import Path\n"
                "import runpy\n"
                "import sys\n"
                "import vllm\n\n"
                'assert "vllm_sail" not in sys.modules\n'
                "pkg_dir = Path(vllm.__file__).resolve().parent\n"
                'src_root_raw = os.environ.get("VLLM_SOURCE_ROOT", "").strip()\n'
                "src_root = Path(src_root_raw) if src_root_raw else pkg_dir.parent\n"
                'if not (src_root / "tools" / "build_rust.py").is_file():\n'
                "    src_root = Path.cwd()\n"
                'build_rust_py = src_root / "tools" / "build_rust.py"\n'
                "if build_rust_py.is_file():\n"
                "    mod_ns = runpy.run_path(str(build_rust_py))\n"
                '    ext_fn = mod_ns.get("rust_py_extension_module_names")\n'
                "    declared = list(ext_fn()) if callable(ext_fn) else []\n"
                '    for label, vllm_dir in (("source", src_root / "vllm"), ("installed", pkg_dir)):\n'
                '        binary = vllm_dir / "vllm-rs"\n'
                "        if not binary.is_file() or binary.stat().st_size == 0:\n"
                '            raise SystemExit(f"required local Rust artifacts were not built ({label}): {binary}")\n'
                "        extensions = []\n"
                "        for module in declared:\n"
                '            paths = list(vllm_dir.glob(f"{module}.*.so"))\n'
                "            if len(paths) != 1 or paths[0].stat().st_size == 0:\n"
                '                raise SystemExit(f"required local Rust extension was not built ({label}): {module}")\n'
                "            extensions.extend(paths)\n"
                '        if set(vllm_dir.glob("_rust_*.so")) != set(extensions):\n'
                '            raise SystemExit(f"unexpected local Rust extensions outside upstream declarations ({label})")\n'
                'print("cpu_empty_build_ok", vllm.__version__)\n'
            ),
        }

    if repro_kind == "platform_plugin":
        return {
            "mode": "stub_plugin_cpu",
            "repro_kind": "platform_plugin",
            "entry_points": (
                "[vllm.platform_plugins]\nstub = vllm_stub_plugin:register_platform\n"
            ),
            "plugin_module": (
                "from vllm.platforms.interface import Platform, PlatformEnum\n\n\n"
                "class StubPlatform(Platform):\n"
                "    _enum = PlatformEnum.OOT\n"
                '    device_name = "stub"\n'
                '    device_type = "cpu"\n\n\n'
                "def register_platform() -> str:\n"
                '    return "vllm_stub_plugin.StubPlatform"\n'
            ),
            "check_script": (
                "import importlib\n"
                "import sys\n"
                "import vllm.platforms as vp\n"
                "from vllm.platforms.interface import Platform\n"
                "from vllm.plugins import load_plugins_by_group\n\n"
                'assert "vllm_sail" not in sys.modules\n'
                'plugins = load_plugins_by_group("vllm.platform_plugins")\n'
                'assert "stub" in plugins, f"stub platform plugin missing: {sorted(plugins)}"\n'
                'qualname = plugins["stub"]()\n'
                'mod_name, attr_name = qualname.rsplit(".", 1)\n'
                "plat_cls = getattr(importlib.import_module(mod_name), attr_name)\n"
                "assert issubclass(plat_cls, Platform)\n"
                "plat = plat_cls()\n"
                "assert plat.is_out_of_tree()\n"
                'resolved = getattr(vp, "current_platform", None)\n'
                "if resolved is not None:\n"
                "    assert isinstance(resolved, Platform)\n"
                'print("platform_plugin_ok", plat.device_name)\n'
            ),
        }

    if repro_kind == "register_oot":
        return {
            "mode": "stub_plugin_cpu",
            "repro_kind": "register_oot",
            "entry_points": (
                "[vllm.general_plugins]\nstub = vllm_stub_plugin:register_general\n"
            ),
            "plugin_module": (
                "_REGISTERED = False\n"
                "StubBaseOp = None\n\n\n"
                "def register_general() -> None:\n"
                "    global _REGISTERED, StubBaseOp\n"
                "    if _REGISTERED:\n"
                "        return\n"
                "    from vllm.model_executor.custom_op import CustomOp\n\n"
                '    @CustomOp.register("stub_oot_op")\n'
                "    class _StubBaseOp(CustomOp):\n"
                "        def forward_native(self, x):\n"
                "            return x\n\n"
                '    @CustomOp.register_oot(name="_StubBaseOp")\n'
                "    class _StubOOTOp(_StubBaseOp):\n"
                "        def forward_oot(self, x):\n"
                "            return x\n\n"
                "    StubBaseOp = _StubBaseOp\n"
                "    _REGISTERED = True\n"
            ),
            "check_script": (
                "import sys\n"
                "import vllm.model_executor.custom_op as custom_op_mod\n"
                "from vllm.model_executor.custom_op import CustomOp\n"
                "from vllm.plugins import load_general_plugins\n"
                "import vllm_stub_plugin\n\n"
                'assert "vllm_sail" not in sys.modules\n'
                "load_general_plugins()\n"
                'oot_reg = getattr(custom_op_mod, "op_registry_oot", None)\n'
                "if oot_reg is None:\n"
                '    oot_reg = getattr(CustomOp, "op_registry_oot", None)\n'
                'assert isinstance(oot_reg, dict) and "_StubBaseOp" in oot_reg, (\n'
                '    f"_StubBaseOp missing from op_registry_oot: {oot_reg!r}"\n'
                ")\n"
                "base_cls = vllm_stub_plugin.StubBaseOp\n"
                "assert base_cls is not None\n"
                "dispatched = CustomOp.__new__(base_cls)\n"
                'assert type(dispatched) is oot_reg["_StubBaseOp"], (\n'
                '    f"CustomOp.__new__ did not dispatch to OOT subclass: {type(dispatched)!r}"\n'
                ")\n"
                'print("register_oot_ok")\n'
            ),
        }

    if repro_kind == "model_registry":
        return {
            "mode": "stub_plugin_cpu",
            "repro_kind": "model_registry",
            "entry_points": (
                "[vllm.general_plugins]\nstub = vllm_stub_plugin:register_general\n"
            ),
            "plugin_module": (
                "_REGISTERED = False\n\n\n"
                "class StubOOTForCausalLM:\n"
                "    pass\n\n\n"
                "def register_general() -> None:\n"
                "    global _REGISTERED\n"
                "    if _REGISTERED:\n"
                "        return\n"
                "    from vllm import ModelRegistry\n\n"
                "    ModelRegistry.register_model(\n"
                '        "StubOOTForCausalLM",\n'
                '        "vllm_stub_plugin:StubOOTForCausalLM",\n'
                "    )\n"
                "    _REGISTERED = True\n"
            ),
            "check_script": (
                "import sys\n"
                "from vllm import ModelRegistry\n"
                "from vllm.plugins import load_general_plugins\n\n"
                'assert "vllm_sail" not in sys.modules\n'
                "load_general_plugins()\n"
                "archs = ModelRegistry.get_supported_archs()\n"
                'assert "StubOOTForCausalLM" in archs, "StubOOTForCausalLM not registered in ModelRegistry"\n'
                'print("model_registry_ok")\n'
            ),
        }

    return {
        "mode": "stub_plugin_cpu",
        "repro_kind": "load_general_plugins",
        "entry_points": (
            "[vllm.general_plugins]\nstub = vllm_stub_plugin:register_general\n"
        ),
        "plugin_module": (
            "LOADED = False\n\n\n"
            "def register_general() -> None:\n"
            "    global LOADED\n"
            "    LOADED = True\n"
        ),
        "check_script": (
            "import sys\n"
            "import vllm\n"
            "import vllm_stub_plugin\n"
            "from vllm.plugins import load_general_plugins\n\n"
            'assert "vllm_sail" not in sys.modules\n'
            "load_general_plugins()\n"
            'assert vllm_stub_plugin.LOADED is True, "vllm.general_plugins stub hook was not invoked"\n'
            'print("load_general_plugins_ok", vllm.__version__)\n'
        ),
    }


def materialize_stub_plugin(repro_kind: str, workdir: Path) -> tuple[Path, Path]:
    """Write the inline stub plugin package and check script into ``workdir``."""
    spec = build_stub_plugin_spec(repro_kind)
    stub_site = workdir / "stub_site"
    stub_site.mkdir(parents=True, exist_ok=True)
    if spec["entry_points"] and spec["plugin_module"]:
        pkg_dir = stub_site / "vllm_stub_plugin"
        dist_dir = stub_site / "vllm_stub_plugin-0.0.1.dist-info"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        dist_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text(spec["plugin_module"], encoding="utf-8")
        (dist_dir / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: vllm-stub-plugin\nVersion: 0.0.1\n",
            encoding="utf-8",
        )
        (dist_dir / "entry_points.txt").write_text(
            spec["entry_points"], encoding="utf-8"
        )
    check_script = workdir / "repro_check.py"
    check_script.write_text(spec["check_script"], encoding="utf-8")
    return stub_site, check_script


def _is_infra_process_failure(
    returncode: int, stderr: str = "", stdout: str = ""
) -> bool:
    """Return True when a subprocess exit indicates runner/host/network/pip-internal infra failure."""
    if returncode < 0 or returncode in _INFRA_EXIT_CODES:
        return True
    combined = f"{stderr}\n{stdout}"
    if _INSTALL_INFRA_ERROR_RE.search(combined) is not None:
        return True
    return _PIP_INTERNAL_TRACEBACK_RE.search(combined) is not None


def _strip_pip_wrapper_lines(text: str) -> str:
    kept = [
        line
        for line in text.splitlines()
        if line.strip() and _PIP_WRAPPER_LINE_RE.match(line) is None
    ]
    return "\n".join(kept)


def _is_upstream_py_frame(raw_path: str) -> bool:
    norm = raw_path.replace("\\", "/").strip()
    if not norm or norm.startswith("<"):
        return False
    if _NON_UPSTREAM_PATH_PART_RE.search(norm) is not None:
        return False
    return _UPSTREAM_PY_FRAME_RE.search(norm) is not None


def _is_valid_repo_filenoent(msg: str) -> bool:
    norm = msg.replace("\\", "/").strip()
    if _NON_UPSTREAM_PATH_PART_RE.search(norm) is not None:
        return False
    return _REPO_FILENOENT_TARGET_RE.search(norm) is not None


def _has_native_repo_compile_error(payload: str, lines: list[str]) -> bool:
    if any(
        _CMAKE_REPO_ERROR_RE.match(ln) or _NATIVE_COMPILE_REPO_ERROR_RE.match(ln)
        for ln in lines
    ):
        return True
    return (
        _RUSTC_ERROR_CODE_RE.search(payload) is not None
        and _RUSTC_REPO_SPAN_RE.search(payload) is not None
    )


def _extract_python_exc_info(m: re.Match[str]) -> tuple[int, str, str]:
    indent = len(m.group(1))
    exc_name = m.group(2) or m.group(3) or ""
    exc_msg = m.group(4) or ""
    return indent, exc_name, exc_msg


def _has_upstream_build_source_evidence(payload: str) -> bool:
    """Require structural source-frame or native compiler attribution; never cross-match free text."""
    lines = payload.splitlines()
    if not lines:
        return False
    if any(_EXCEPTION_GROUP_LINE_RE.match(ln) for ln in lines):
        return False

    has_native_diag = _has_native_repo_compile_error(payload, lines)
    tb_headers = [
        (idx, len(m.group(1)))
        for idx, ln in enumerate(lines)
        if (m := _TRACEBACK_HEADER_RE.match(ln)) is not None
    ]
    consumed_lines: set[int] = set()

    if tb_headers:
        for pos, (start_idx, tb_indent) in enumerate(tb_headers):
            next_tb = (
                tb_headers[pos + 1][0] if pos + 1 < len(tb_headers) else len(lines)
            )
            consumed_lines.add(start_idx)
            real_frames: list[str] = []
            in_setuptools_run_setup = False
            exc_info: tuple[int, str, str] | None = None
            for j in range(start_idx + 1, next_tb):
                fm = _TRACEBACK_FILE_RE.match(lines[j])
                if fm is not None:
                    consumed_lines.add(j)
                    norm_fpath = fm.group(2).replace("\\", "/").strip()
                    func_name = fm.group(3) or ""
                    if norm_fpath.startswith("<frozen "):
                        continue
                    if _SETUPTOOLS_RUN_SETUP_FILE_RE.search(
                        norm_fpath
                    ) is not None and func_name in ("run_setup", "_execfile"):
                        in_setuptools_run_setup = True
                        real_frames.append(norm_fpath)
                        continue
                    if norm_fpath == "<string>" and in_setuptools_run_setup:
                        real_frames.append("setup.py")
                        continue
                    real_frames.append(norm_fpath)
                    continue
                em = _PYTHON_EXC_LINE_RE.match(lines[j])
                if em is not None:
                    e_indent, e_name, e_msg = _extract_python_exc_info(em)
                    if e_indent <= tb_indent:
                        exc_info = (e_indent, e_name, e_msg)
                        consumed_lines.add(j)
                        break
                consumed_lines.add(j)

            if exc_info is None or not real_frames:
                return False
            _, exc_name, exc_msg = exc_info
            last_frame = real_frames[-1]
            has_upstream_ancestor = any(_is_upstream_py_frame(f) for f in real_frames)

            if _is_upstream_py_frame(last_frame):
                if exc_name not in _ALLOWED_BUILD_EXC_NAMES:
                    return False
                if (
                    exc_name
                    in (
                        "ModuleNotFoundError",
                        "ImportError",
                    )
                    and _BUILD_ENV_MISSING_MOD_RE.search(exc_msg) is not None
                ):
                    return False
                if exc_name == "FileNotFoundError" and not _is_valid_repo_filenoent(
                    exc_msg
                ):
                    return False
            elif (
                exc_name == "FileNotFoundError"
                and has_upstream_ancestor
                and _STDLIB_PATHLIB_FRAME_RE.search(last_frame) is not None
                and _is_valid_repo_filenoent(exc_msg)
            ):
                pass
            elif (
                has_upstream_ancestor
                and has_native_diag
                and (
                    (
                        exc_name == "CalledProcessError"
                        and _STDLIB_SUBPROCESS_FRAME_RE.search(last_frame) is not None
                    )
                    or exc_name
                    in (
                        "CompileError",
                        "DistutilsExecError",
                        "CCompilerError",
                        "SystemExit",
                    )
                )
            ):
                pass
            elif (
                has_upstream_ancestor
                and _SETUPTOOLS_PYPROJECT_FILE_RE.search(last_frame) is not None
                and exc_name in ("ValueError", "InvalidConfigError", "TOMLDecodeError")
            ):
                pass
            else:
                return False

        for idx, ln in enumerate(lines):
            if idx not in consumed_lines and _PYTHON_EXC_LINE_RE.match(ln):
                return False
        return True

    file_matches = [
        (idx, m)
        for idx, ln in enumerate(lines)
        if (m := _TRACEBACK_FILE_RE.match(ln)) is not None
    ]
    if file_matches:
        if any(not _is_upstream_py_frame(m.group(2)) for _, m in file_matches):
            return False
        last_file_idx, last_fm = file_matches[-1]
        file_indent = len(last_fm.group(1))
        syntax_exc_idx = -1
        for j in range(last_file_idx + 1, min(last_file_idx + 8, len(lines))):
            em = _PYTHON_EXC_LINE_RE.match(lines[j])
            if em is not None:
                e_indent, e_name, _ = _extract_python_exc_info(em)
                if e_indent <= file_indent and e_name in (
                    "SyntaxError",
                    "IndentationError",
                    "TabError",
                ):
                    syntax_exc_idx = j
                    break
        if syntax_exc_idx < 0:
            return False
        for idx, ln in enumerate(lines):
            if (idx < file_matches[0][0] or idx > syntax_exc_idx) and (
                _PYTHON_EXC_LINE_RE.match(ln) is not None
            ):
                return False
        return True

    if any(_PYTHON_EXC_LINE_RE.match(ln) for ln in lines):
        return False
    return has_native_diag


def _classify_install_failure(
    returncode: int, stderr: str, stdout: str, *, mode: str
) -> tuple[str, str, bool]:
    """Classify a non-zero install/build exit into ``(phase, failure_source, is_infra_error)``."""
    if _is_infra_process_failure(returncode, stderr, stdout):
        return "infra_error", "infra_or_unknown", True
    if mode != "cpu_empty_build":
        return "install", "infra_or_unknown", False
    payload = _strip_pip_wrapper_lines(f"{stdout}\n{stderr}")
    if payload and _has_upstream_build_source_evidence(payload):
        return "build_install", "upstream_build_source", False
    return "infra_error", "infra_or_unknown", True


def is_valid_non_ppu_repro(
    evidence: NonPpuReproEvidence | None,
    *,
    stage: str | None = None,
    last_good_sha: str = "",
    first_failing_sha: str = "",
    require_shas: bool = False,
) -> bool:
    """Fail-closed hard gate for hardware-free (CPU / stub plugin) reproduction."""
    if not isinstance(evidence, NonPpuReproEvidence):
        return False
    if evidence.verified is not True:
        return False
    if evidence.infra_error is not False:
        return False
    if evidence.mode not in VALID_NON_PPU_REPRO_MODES:
        return False
    if evidence.repro_kind not in VALID_REPRO_KINDS:
        return False
    if evidence.last_good_phase not in VALID_REPRO_PHASES:
        return False
    if evidence.first_failing_phase not in VALID_REPRO_PHASES:
        return False
    if evidence.failure_source not in VALID_REPRO_FAILURE_SOURCES:
        return False
    if type(evidence.last_good_check_executed) is not bool:
        return False
    if type(evidence.first_failing_check_executed) is not bool:
        return False

    if evidence.mode == "cpu_empty_build":
        if evidence.repro_kind != "cpu_empty_build":
            return False
        if (
            evidence.last_good_phase != "build_import"
            or evidence.last_good_check_executed is not True
        ):
            return False
        if evidence.first_failing_phase == "build_import":
            if (
                evidence.first_failing_check_executed is not True
                or evidence.failure_source != "upstream_import_check"
            ):
                return False
        elif evidence.first_failing_phase == "build_install":
            if (
                evidence.first_failing_check_executed is not False
                or evidence.failure_source != "upstream_build_source"
            ):
                return False
        else:
            return False
    elif evidence.mode == "stub_plugin_cpu":
        if evidence.repro_kind not in STUB_PLUGIN_REPRO_KINDS:
            return False
        if (
            evidence.last_good_phase != "stub_check"
            or evidence.last_good_check_executed is not True
        ):
            return False
        if (
            evidence.first_failing_phase != "stub_check"
            or evidence.first_failing_check_executed is not True
            or evidence.failure_source != "stub_plugin_check"
        ):
            return False
    else:
        return False

    if not isinstance(evidence.last_good_sha, str) or not SHA1_RE.fullmatch(
        evidence.last_good_sha
    ):
        return False
    if not isinstance(evidence.first_failing_sha, str) or not SHA1_RE.fullmatch(
        evidence.first_failing_sha
    ):
        return False
    if evidence.last_good_sha == evidence.first_failing_sha:
        return False
    if type(evidence.last_good_exit_code) is not int or (
        evidence.last_good_exit_code != 0
    ):
        return False
    if (
        type(evidence.first_failing_exit_code) is not int
        or evidence.first_failing_exit_code <= 0
        or evidence.first_failing_exit_code in _INFRA_EXIT_CODES
    ):
        return False

    if require_shas:
        if not SHA1_RE.fullmatch(last_good_sha) or not SHA1_RE.fullmatch(
            first_failing_sha
        ):
            return False
        if (
            evidence.last_good_sha != last_good_sha
            or evidence.first_failing_sha != first_failing_sha
        ):
            return False
    else:
        if last_good_sha and evidence.last_good_sha != last_good_sha:
            return False
        if first_failing_sha and evidence.first_failing_sha != first_failing_sha:
            return False

    if stage is not None:
        if stage in SAIL_INTERNAL_ONLY_STAGES:
            return False
        if stage == "wheel_build" and evidence.mode != "cpu_empty_build":
            return False
        if stage == "plugin_registration" and evidence.mode != "stub_plugin_cpu":
            return False
    return True


def parse_non_ppu_repro_evidence(data: Any) -> NonPpuReproEvidence | None:
    """Strictly parse and re-validate serialized ``NonPpuReproEvidence`` JSON."""
    if not isinstance(data, dict):
        return None
    mode = data.get("mode")
    repro_kind = data.get("repro_kind")
    last_good = data.get("last_good_sha")
    first_bad = data.get("first_failing_sha")
    good_rc = data.get("last_good_exit_code")
    bad_rc = data.get("first_failing_exit_code")
    good_phase = data.get("last_good_phase")
    bad_phase = data.get("first_failing_phase")
    good_exec = data.get("last_good_check_executed")
    bad_exec = data.get("first_failing_check_executed")
    infra_err = data.get("infra_error")
    failure_source = data.get("failure_source")
    raw_verified = data.get("verified")
    raw_obs_sig = data.get("observed_signature", "")
    if raw_obs_sig is None:
        raw_obs_sig = ""
    if (
        not isinstance(mode, str)
        or not isinstance(repro_kind, str)
        or not isinstance(last_good, str)
        or not isinstance(first_bad, str)
        or not isinstance(good_phase, str)
        or not isinstance(bad_phase, str)
        or not isinstance(failure_source, str)
        or not isinstance(raw_obs_sig, str)
        or good_phase not in VALID_REPRO_PHASES
        or bad_phase not in VALID_REPRO_PHASES
        or failure_source not in VALID_REPRO_FAILURE_SOURCES
        or type(good_rc) is not int
        or type(bad_rc) is not int
        or type(good_exec) is not bool
        or type(bad_exec) is not bool
        or type(infra_err) is not bool
        or type(raw_verified) is not bool
    ):
        return None
    if raw_obs_sig.strip():
        safe_obs_sig = _sanitize_symbol(raw_obs_sig, "")
        if not safe_obs_sig:
            return None
    else:
        safe_obs_sig = ""
    candidate = NonPpuReproEvidence(
        mode=mode,
        repro_kind=repro_kind,
        last_good_sha=last_good.strip(),
        first_failing_sha=first_bad.strip(),
        last_good_exit_code=good_rc,
        first_failing_exit_code=bad_rc,
        verified=raw_verified,
        last_good_phase=good_phase,
        first_failing_phase=bad_phase,
        last_good_check_executed=good_exec,
        first_failing_check_executed=bad_exec,
        infra_error=infra_err,
        failure_source=failure_source,
        observed_signature=safe_obs_sig,
    )
    return NonPpuReproEvidence(
        mode=candidate.mode,
        repro_kind=candidate.repro_kind,
        last_good_sha=candidate.last_good_sha,
        first_failing_sha=candidate.first_failing_sha,
        last_good_exit_code=candidate.last_good_exit_code,
        first_failing_exit_code=candidate.first_failing_exit_code,
        verified=is_valid_non_ppu_repro(candidate),
        last_good_phase=candidate.last_good_phase,
        first_failing_phase=candidate.first_failing_phase,
        last_good_check_executed=candidate.last_good_check_executed,
        first_failing_check_executed=candidate.first_failing_check_executed,
        infra_error=candidate.infra_error,
        failure_source=candidate.failure_source,
        observed_signature=candidate.observed_signature,
    )


_CLEAN_BUILD_ENV_DROP_KEYS = frozenset(
    {
        "PYTHONPATH",
        "VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT",
        "VLLM_PPU_CANDIDATE_UPSTREAM_COMMIT",
        "VLLM_USE_PRECOMPILED",
        "VLLM_USE_PRECOMPILED_RUST",
        "VLLM_PRECOMPILED_WHEEL_LOCATION",
        "VLLM_VERSION",
        "VLLM_VERSION_OVERRIDE",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM",
    }
)
_RUST_TOOLCHAIN_CHANNEL_RE = re.compile(
    r"""(?m)^\s*channel\s*=\s*['"]([^'"]+)['"]\s*(?:#.*)?$"""
)


def _read_rust_toolchain_channel(wt_dir: Path) -> str:
    toolchain_file = wt_dir / "rust-toolchain.toml"
    if not toolchain_file.is_file():
        return ""
    try:
        raw = toolchain_file.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]

        parsed = tomllib.loads(raw)
        ch = (parsed.get("toolchain") or {}).get("channel")
        if isinstance(ch, str) and ch.strip():
            return ch.strip()
    except Exception:
        pass
    m = _RUST_TOOLCHAIN_CHANNEL_RE.search(raw)
    return m.group(1).strip() if m else ""


def _execute_vllm_worktree_build(
    *,
    vllm_repo: Path,
    sha: str,
    workdir: Path,
    step_tag: str,
    mode: str,
    base_env: dict[str, str],
    stub_site: Path,
    check_script: Path,
    python_exe: str,
    timeout_sec: int,
) -> tuple[int, str, str, bool, bool, str]:
    """Build & check ``sha`` in a detached worktree, replaying mandatory Rust build in ``cpu_empty_build``.

    Returns ``(exit_code, phase, failure_source, check_executed, is_infra_error, target_signature)``.
    """
    wt_dir = workdir / f"vllm-{step_tag}"
    site_dir = workdir / f"site-{step_tag}"
    site_dir.mkdir(parents=True, exist_ok=True)
    wt_added = False
    try:
        add_proc = subprocess.run(
            [
                "git",
                "-C",
                str(vllm_repo),
                "worktree",
                "add",
                "--quiet",
                "--detach",
                str(wt_dir),
                sha,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if add_proc.returncode != 0:
            return (
                add_proc.returncode,
                "setup",
                "infra_or_unknown",
                False,
                True,
                "VLLM_TARGET_DEVICE=empty",
            )
        wt_added = True

        run_env = dict(base_env)
        run_env["STUB_CHECK_SCRIPT"] = str(check_script)
        run_env["VLLM_SOURCE_ROOT"] = str(wt_dir)

        use_torch = wt_dir / "tools" / "use_existing_torch.py"
        if use_torch.is_file():
            torch_proc = subprocess.run(
                [python_exe, str(use_torch)],
                cwd=str(wt_dir),
                env=run_env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if torch_proc.returncode != 0:
                if mode == "cpu_empty_build":
                    phase, src, is_infra = _classify_install_failure(
                        torch_proc.returncode,
                        torch_proc.stderr,
                        torch_proc.stdout,
                        mode=mode,
                    )
                    return (
                        torch_proc.returncode,
                        phase,
                        src,
                        False,
                        is_infra,
                        "VLLM_TARGET_DEVICE=empty",
                    )
                is_infra = _is_infra_process_failure(
                    torch_proc.returncode, torch_proc.stderr, torch_proc.stdout
                )
                return (
                    torch_proc.returncode,
                    "infra_error" if is_infra else "install",
                    "infra_or_unknown",
                    False,
                    is_infra,
                    "VLLM_TARGET_DEVICE=empty",
                )

        has_build_rust = False
        if mode == "cpu_empty_build":
            build_rust_py = wt_dir / "tools" / "build_rust.py"
            if build_rust_py.is_file():
                has_build_rust = True
                channel = _read_rust_toolchain_channel(wt_dir)
                rustup_bin = shutil.which(
                    "rustup", path=run_env.get("PATH", os.environ.get("PATH"))
                )
                if channel and not run_env.get("RUSTUP_TOOLCHAIN") and rustup_bin:
                    run_env["RUSTUP_TOOLCHAIN"] = channel
                    probe_tc = subprocess.run(
                        [rustup_bin, "which", "rustc", "--toolchain", channel],
                        cwd=str(wt_dir),
                        env=run_env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if probe_tc.returncode != 0:
                        tc_install = subprocess.run(
                            [
                                rustup_bin,
                                "toolchain",
                                "install",
                                channel,
                                "--profile",
                                "minimal",
                            ],
                            cwd=str(wt_dir),
                            env=run_env,
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                        if tc_install.returncode != 0:
                            return (
                                tc_install.returncode,
                                "infra_error",
                                "infra_or_unknown",
                                False,
                                True,
                                "rust_frontend_build",
                            )

                rust_proc = subprocess.run(
                    [python_exe, str(build_rust_py), "--release"],
                    cwd=str(wt_dir),
                    env=run_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if rust_proc.returncode != 0:
                    phase, src, is_infra = _classify_install_failure(
                        rust_proc.returncode,
                        rust_proc.stderr,
                        rust_proc.stdout,
                        mode=mode,
                    )
                    return (
                        rust_proc.returncode,
                        phase,
                        src,
                        False,
                        is_infra,
                        "rust_frontend_build",
                    )

        install_proc = subprocess.run(
            [
                python_exe,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-build-isolation",
                "--no-deps",
                "--upgrade",
                "--target",
                str(site_dir),
                str(wt_dir),
            ],
            cwd=str(wt_dir),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if install_proc.returncode != 0:
            phase, src, is_infra = _classify_install_failure(
                install_proc.returncode,
                install_proc.stderr,
                install_proc.stdout,
                mode=mode,
            )
            combined_out = f"{install_proc.stdout}\n{install_proc.stderr}"
            sig = (
                "rust_frontend_build"
                if _RUSTC_REPO_SPAN_RE.search(combined_out) is not None
                else "VLLM_TARGET_DEVICE=empty"
            )
            return install_proc.returncode, phase, src, False, is_infra, sig

        check_env = dict(run_env)
        py_entries = [str(site_dir)]
        if mode == "stub_plugin_cpu":
            py_entries.insert(0, str(stub_site))
        check_env["PYTHONPATH"] = os.pathsep.join(py_entries)
        check_proc = subprocess.run(
            [python_exe, str(check_script)],
            cwd=str(workdir),
            env=check_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if check_proc.returncode != 0 and _is_infra_process_failure(
            check_proc.returncode, check_proc.stderr, check_proc.stdout
        ):
            return (
                check_proc.returncode,
                "infra_error",
                "infra_or_unknown",
                True,
                True,
                "VLLM_TARGET_DEVICE=empty",
            )
        phase = "stub_check" if mode == "stub_plugin_cpu" else "build_import"
        src = (
            "none"
            if check_proc.returncode == 0
            else (
                "stub_plugin_check"
                if mode == "stub_plugin_cpu"
                else "upstream_import_check"
            )
        )
        chk_out = f"{check_proc.stdout}\n{check_proc.stderr}"
        sig = (
            "rust_frontend_build"
            if (check_proc.returncode != 0 and has_build_rust and "Rust" in chk_out)
            else "VLLM_TARGET_DEVICE=empty"
        )
        return check_proc.returncode, phase, src, True, False, sig
    except subprocess.TimeoutExpired:
        return (
            124,
            "infra_error",
            "infra_or_unknown",
            False,
            True,
            "VLLM_TARGET_DEVICE=empty",
        )
    finally:
        if wt_added:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(vllm_repo),
                    "worktree",
                    "remove",
                    "--force",
                    str(wt_dir),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )


def verify_non_ppu_reproducer(
    *,
    vllm_repo: Path,
    last_good_sha: str,
    first_failing_sha: str,
    stage: str,
    target_signature: str = "",
    repro_kind: str = "",
    probe_cmd: str = "",
    python_exe: str = sys.executable,
    timeout_sec: int = 900,
) -> NonPpuReproEvidence:
    """Run the hardware-free CPU/stub reproducer against ``last_good_sha`` and ``first_failing_sha``."""
    good = _require_sha1(last_good_sha, "last_good_sha")
    bad = _require_sha1(first_failing_sha, "first_failing_sha")
    safe_stage = stage if stage in VALID_STAGES else "plugin_registration"
    kind = repro_kind.strip() or select_repro_kind(safe_stage, target_signature)
    spec = build_stub_plugin_spec(kind)
    mode = spec["mode"]

    if good == bad or safe_stage in SAIL_INTERNAL_ONLY_STAGES:
        return NonPpuReproEvidence(
            mode=mode,
            repro_kind=kind,
            last_good_sha=good,
            first_failing_sha=bad,
            last_good_exit_code=-1,
            first_failing_exit_code=-1,
            verified=False,
            last_good_phase="none",
            first_failing_phase="none",
            last_good_check_executed=False,
            first_failing_check_executed=False,
            infra_error=False,
            failure_source="none",
        )

    def _run_one(
        sha: str, workdir: Path, stub_site: Path, check_script: Path
    ) -> tuple[int, str, str, bool, bool, str]:
        """Return ``(exit_code, phase, failure_source, check_executed, is_infra_error, observed_sig)``."""
        base_env = {
            k: v for k, v in os.environ.items() if k not in _CLEAN_BUILD_ENV_DROP_KEYS
        }
        base_env["VLLM_TARGET_DEVICE"] = "empty"
        if mode == "cpu_empty_build":
            base_env["VLLM_REQUIRE_RUST_FRONTEND"] = "1"
            base_env["CARGO_NET_GIT_FETCH_WITH_CLI"] = "true"
        base_env["CUDA_VISIBLE_DEVICES"] = ""
        base_env["VLLM_PLUGINS"] = "stub" if mode == "stub_plugin_cpu" else ""
        base_env["CANDIDATE_SHA"] = sha
        base_env["VLLM_REPO"] = str(vllm_repo)
        base_env["REPRO_KIND"] = kind
        base_env["STUB_SITE_DIR"] = str(stub_site)

        if probe_cmd.strip():
            marker_path = workdir / f"check-exec-{sha[:12]}.marker"
            marker_path.unlink(missing_ok=True)
            probe_json = workdir / f"repro-probe-{sha[:12]}.json"
            probe_json.unlink(missing_ok=True)
            runner_script = workdir / f"run-check-{sha[:12]}.py"
            runner_script.write_text(
                "import pathlib, runpy\n"
                f"pathlib.Path({str(marker_path)!r}).write_text('executed\\n', encoding='utf-8')\n"
                f"runpy.run_path({str(check_script)!r}, run_name='__main__')\n",
                encoding="utf-8",
            )
            base_env["STUB_CHECK_SCRIPT"] = str(runner_script)
            base_env["PROBE_RESULT_JSON"] = str(probe_json)
            try:
                proc = subprocess.run(
                    ["bash", "-c", probe_cmd],
                    env=base_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return (
                    124,
                    "infra_error",
                    "infra_or_unknown",
                    marker_path.is_file(),
                    True,
                    "",
                )
            executed = marker_path.is_file()
            explicit_sig = ""
            if probe_json.is_file():
                try:
                    pdata = json.loads(probe_json.read_text(encoding="utf-8"))
                    if isinstance(pdata, dict) and isinstance(
                        pdata.get("target_signature"), str
                    ):
                        explicit_sig = pdata["target_signature"].strip()
                except (OSError, ValueError):
                    explicit_sig = ""
                probe_json.unlink(missing_ok=True)
            combined_out = f"{proc.stdout}\n{proc.stderr}"
            inferred_build_sig = explicit_sig or (
                "rust_frontend_build"
                if (
                    _RUSTC_REPO_SPAN_RE.search(combined_out) is not None
                    or (
                        executed
                        and proc.returncode != 0
                        and "required local Rust" in combined_out
                    )
                )
                else "VLLM_TARGET_DEVICE=empty"
            )
            if proc.returncode != 0 and _is_infra_process_failure(
                proc.returncode, proc.stderr, proc.stdout
            ):
                return (
                    proc.returncode,
                    "infra_error",
                    "infra_or_unknown",
                    executed,
                    True,
                    "",
                )
            if mode == "stub_plugin_cpu":
                if executed:
                    src = "none" if proc.returncode == 0 else "stub_plugin_check"
                    return (
                        proc.returncode,
                        "stub_check",
                        src,
                        True,
                        False,
                        explicit_sig if proc.returncode != 0 else "",
                    )
                src = "none" if proc.returncode == 0 else "infra_or_unknown"
                return proc.returncode, "install", src, False, False, ""
            if executed:
                src = "none" if proc.returncode == 0 else "upstream_import_check"
                return (
                    proc.returncode,
                    "build_import",
                    src,
                    True,
                    False,
                    inferred_build_sig if proc.returncode != 0 else "",
                )
            if proc.returncode != 0:
                phase, src, is_infra = _classify_install_failure(
                    proc.returncode, proc.stderr, proc.stdout, mode=mode
                )
                return (
                    proc.returncode,
                    phase,
                    src,
                    False,
                    is_infra,
                    "" if is_infra else inferred_build_sig,
                )
            return 0, "install", "none", False, False, ""

        rc, phase, src, executed, is_infra, obs_sig = _execute_vllm_worktree_build(
            vllm_repo=vllm_repo,
            sha=sha,
            workdir=workdir,
            step_tag=sha[:12],
            mode=mode,
            base_env=base_env,
            stub_site=stub_site,
            check_script=check_script,
            python_exe=python_exe,
            timeout_sec=timeout_sec,
        )
        return (
            rc,
            phase,
            src,
            executed,
            is_infra,
            "" if (rc == 0 or is_infra) else obs_sig,
        )

    with tempfile.TemporaryDirectory(prefix="non-ppu-repro-") as td:
        workdir = Path(td)
        stub_site, check_script = materialize_stub_plugin(kind, workdir)
        good_rc, good_phase, good_src, good_exec, good_infra, _good_sig = _run_one(
            good, workdir, stub_site, check_script
        )
        if good_infra or good_rc != 0 or not good_exec:
            return NonPpuReproEvidence(
                mode=mode,
                repro_kind=kind,
                last_good_sha=good,
                first_failing_sha=bad,
                last_good_exit_code=good_rc,
                first_failing_exit_code=-1,
                verified=False,
                last_good_phase=good_phase,
                first_failing_phase="none",
                last_good_check_executed=good_exec,
                first_failing_check_executed=False,
                infra_error=good_infra,
                failure_source=good_src if good_rc != 0 else "infra_or_unknown",
                observed_signature="",
            )
        bad_rc, bad_phase, bad_src, bad_exec, bad_infra, bad_sig = _run_one(
            bad, workdir, stub_site, check_script
        )

    any_infra = good_infra or bad_infra
    eff_obs_sig = (
        _sanitize_symbol(bad_sig, "")
        if (not any_infra and bad_rc > 0 and bad_sig.strip())
        else ""
    )
    raw_evidence = NonPpuReproEvidence(
        mode=mode,
        repro_kind=kind,
        last_good_sha=good,
        first_failing_sha=bad,
        last_good_exit_code=good_rc,
        first_failing_exit_code=bad_rc,
        verified=(not any_infra and good_rc == 0 and bad_rc > 0),
        last_good_phase=good_phase,
        first_failing_phase=bad_phase,
        last_good_check_executed=good_exec,
        first_failing_check_executed=bad_exec,
        infra_error=any_infra,
        failure_source="infra_or_unknown" if any_infra else bad_src,
        observed_signature=eff_obs_sig,
    )
    return NonPpuReproEvidence(
        mode=raw_evidence.mode,
        repro_kind=raw_evidence.repro_kind,
        last_good_sha=raw_evidence.last_good_sha,
        first_failing_sha=raw_evidence.first_failing_sha,
        last_good_exit_code=raw_evidence.last_good_exit_code,
        first_failing_exit_code=raw_evidence.first_failing_exit_code,
        verified=is_valid_non_ppu_repro(
            raw_evidence,
            stage=safe_stage,
            last_good_sha=good,
            first_failing_sha=bad,
            require_shas=True,
        ),
        last_good_phase=raw_evidence.last_good_phase,
        first_failing_phase=raw_evidence.first_failing_phase,
        last_good_check_executed=raw_evidence.last_good_check_executed,
        first_failing_check_executed=raw_evidence.first_failing_check_executed,
        infra_error=raw_evidence.infra_error,
        failure_source=raw_evidence.failure_source,
        observed_signature=raw_evidence.observed_signature,
    )


def classify_failure(
    outcome: ProbeOutcome,
    *,
    non_ppu_repro: NonPpuReproEvidence | None = None,
    last_good_sha: str = "",
    first_failing_sha: str = "",
    require_shas: bool = False,
) -> dict[str, Any]:
    """Classify a failure with the non-PPU hardware reproducer hard gate."""
    stage = outcome.stage if outcome.stage in VALID_STAGES else "none"
    sig = _sanitize_symbol(outcome.target_signature, "unspecified")

    if outcome.status == "pass":
        return {
            "classification": "none",
            "stage": "none",
            "target_signature": "none",
            "bucket_key": "",
            "non_ppu_repro_verified": False,
            "non_ppu_repro_mode": "",
            "repro_kind": "",
        }

    if outcome.status == "infra_error" or stage == "baseline_control":
        return {
            "classification": "infra_error",
            "stage": stage,
            "target_signature": sig,
            "bucket_key": "",
            "non_ppu_repro_verified": False,
            "non_ppu_repro_mode": "",
            "repro_kind": "",
        }

    if stage in SAIL_INTERNAL_ONLY_STAGES:
        return {
            "classification": "sail_internal_patch_drift",
            "stage": stage,
            "target_signature": sig,
            "bucket_key": f"{stage}:{sig}",
            "non_ppu_repro_verified": False,
            "non_ppu_repro_mode": "",
            "repro_kind": "",
        }

    is_public_candidate = outcome.is_public_plugin_interface
    if not is_public_candidate:
        if stage == "wheel_build" and sig.startswith("vllm."):
            is_public_candidate = True
        elif stage == "wheel_build" and sig in {
            "VLLM_TARGET_DEVICE=empty",
            "vllm_empty_device_build",
            "rust_frontend_build",
        }:
            is_public_candidate = True
        elif (
            stage == "plugin_registration"
            and any(sig.startswith(pub) for pub in PUBLIC_PLUGIN_SYMBOLS)
            and "vllm_sail.patch" not in sig
        ):
            is_public_candidate = True

    if (
        non_ppu_repro is not None
        and outcome.non_ppu_repro is not None
        and non_ppu_repro != outcome.non_ppu_repro
    ):
        effective_repro: NonPpuReproEvidence | None = None
    else:
        effective_repro = (
            non_ppu_repro if non_ppu_repro is not None else outcome.non_ppu_repro
        )

    repro_verified = is_valid_non_ppu_repro(
        effective_repro,
        stage=stage,
        last_good_sha=last_good_sha,
        first_failing_sha=first_failing_sha,
        require_shas=require_shas,
    )
    if (
        repro_verified
        and stage == "wheel_build"
        and effective_repro is not None
        and effective_repro.observed_signature
    ):
        norm_obs = _normalize_wheel_build_signature(effective_repro.observed_signature)
        norm_sig = _normalize_wheel_build_signature(sig)
        if sig == "unspecified":
            repro_verified = False
        elif (
            norm_obs in CANONICAL_WHEEL_BUILD_SIGNATURES
            and norm_sig in CANONICAL_WHEEL_BUILD_SIGNATURES
            and norm_obs != norm_sig
        ):
            repro_verified = False
        elif (
            norm_sig not in CANONICAL_WHEEL_BUILD_SIGNATURES
            and norm_obs not in CANONICAL_WHEEL_BUILD_SIGNATURES
            and norm_obs != norm_sig
        ):
            repro_verified = False

    if is_public_candidate and repro_verified and effective_repro is not None:
        return {
            "classification": "plugin_interface_break",
            "stage": stage,
            "target_signature": sig,
            "bucket_key": f"upstream_plugin_api:{stage}:{sig}",
            "non_ppu_repro_verified": True,
            "non_ppu_repro_mode": effective_repro.mode,
            "repro_kind": effective_repro.repro_kind,
        }

    bucket_key = f"{stage}:{sig}"
    return {
        "classification": "sail_internal_patch_drift",
        "stage": stage,
        "target_signature": sig,
        "bucket_key": bucket_key,
        "non_ppu_repro_verified": False,
        "non_ppu_repro_mode": "",
        "repro_kind": "",
    }


def _render_standalone_repro_script(
    *,
    first_failing_sha: str,
    last_good_sha: str,
    repro_kind: str,
) -> str:
    """Render a standalone, hardware-free bash reproducer script for ``repro_kind``."""
    spec = build_stub_plugin_spec(repro_kind)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# Control (expected exit 0): bash repro.sh {last_good_sha}",
        f"# Regression (expected non-zero): bash repro.sh {first_failing_sha}",
        f'TARGET_REF="${{1:-{first_failing_sha}}}"',
        'WORKDIR="$(mktemp -d)"',
        "trap 'rm -rf \"$WORKDIR\"' EXIT",
        "",
        'git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$WORKDIR/vllm"',
        'cd "$WORKDIR/vllm"',
        'git checkout --quiet --detach "$TARGET_REF"',
        "if [[ -f tools/use_existing_torch.py ]]; then",
        "  python3 tools/use_existing_torch.py",
        "fi",
    ]
    if spec["mode"] == "cpu_empty_build":
        lines.extend(
            [
                "unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST VLLM_PRECOMPILED_WHEEL_LOCATION VLLM_VERSION VLLM_VERSION_OVERRIDE",
                "export VLLM_TARGET_DEVICE=empty VLLM_REQUIRE_RUST_FRONTEND=1 CARGO_NET_GIT_FETCH_WITH_CLI=true",
                "if [[ -f tools/build_rust.py ]]; then",
                "  if [[ -f rust-toolchain.toml ]] && command -v rustup >/dev/null 2>&1; then",
                "    RUSTUP_TOOLCHAIN=\"$(python3 - <<'PY'",
                "import pathlib, re",
                'text = pathlib.Path("rust-toolchain.toml").read_text(encoding="utf-8")',
                'm = re.search(r"(?m)^\\s*channel\\s*=\\s*[\\x22\\x27]([^\\x22\\x27]+)[\\x22\\x27]", text)',
                'print(m.group(1) if m else "")',
                "PY",
                '    )"',
                '    if [[ -n "$RUSTUP_TOOLCHAIN" ]]; then',
                "      export RUSTUP_TOOLCHAIN",
                '      rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal',
                "    fi",
                "  fi",
                "  python3 tools/build_rust.py --release",
                "fi",
                'mkdir -p "$WORKDIR/vllm_site"',
                'python3 -m pip install --no-build-isolation --no-deps --upgrade --target "$WORKDIR/vllm_site" .',
            ]
        )
    else:
        lines.extend(
            [
                'mkdir -p "$WORKDIR/vllm_site"',
                'VLLM_TARGET_DEVICE=empty python3 -m pip install --no-build-isolation --no-deps --upgrade --target "$WORKDIR/vllm_site" .',
            ]
        )
    if spec["mode"] == "stub_plugin_cpu":
        lines.extend(
            [
                "",
                'mkdir -p "$WORKDIR/stub_site/vllm_stub_plugin" "$WORKDIR/stub_site/vllm_stub_plugin-0.0.1.dist-info"',
                "cat <<'EOF' > \"$WORKDIR/stub_site/vllm_stub_plugin-0.0.1.dist-info/METADATA\"",
                "Metadata-Version: 2.1",
                "Name: vllm-stub-plugin",
                "Version: 0.0.1",
                "EOF",
                "cat <<'EOF' > \"$WORKDIR/stub_site/vllm_stub_plugin-0.0.1.dist-info/entry_points.txt\"",
                spec["entry_points"].rstrip("\n"),
                "EOF",
                "cat <<'PY' > \"$WORKDIR/stub_site/vllm_stub_plugin/__init__.py\"",
                spec["plugin_module"].rstrip("\n"),
                "PY",
                "cat <<'PY' > \"$WORKDIR/repro_check.py\"",
                spec["check_script"].rstrip("\n"),
                "PY",
                'PYTHONPATH="$WORKDIR/stub_site:$WORKDIR/vllm_site" VLLM_TARGET_DEVICE=empty VLLM_PLUGINS=stub CUDA_VISIBLE_DEVICES="" python3 "$WORKDIR/repro_check.py"',
            ]
        )
    else:
        lines.extend(
            [
                "",
                "cat <<'PY' > \"$WORKDIR/repro_check.py\"",
                spec["check_script"].rstrip("\n"),
                "PY",
                'VLLM_SOURCE_ROOT="$WORKDIR/vllm" PYTHONPATH="$WORKDIR/vllm_site" VLLM_TARGET_DEVICE=empty VLLM_REQUIRE_RUST_FRONTEND=1 VLLM_PLUGINS="" CUDA_VISIBLE_DEVICES="" python3 "$WORKDIR/repro_check.py"',
            ]
        )
    return "\n".join(lines)


def render_upstream_reproducer(
    *,
    first_failing_sha: str,
    last_good_sha: str,
    sail_sha: str,
    stage: str,
    target_signature: str,
    repro_kind: str = "",
    first_failing_author: str = "",
    non_ppu_repro: NonPpuReproEvidence | None = None,
    run_url: str = "",
) -> str:
    """Render a sanitized, fenced-markdown issue draft for human filing in ``vllm-project/vllm``.

    ``@<author>`` appears strictly inside the outer 4-backtick fenced markdown
    block so CI never emits bare live pings or auto-files cross-repo issues.
    """
    bad = _require_sha1(first_failing_sha, "first_failing_sha")
    good = _require_sha1(last_good_sha, "last_good_sha")
    sail = _require_sha1(sail_sha, "sail_sha")
    safe_stage = stage if stage in VALID_STAGES else "plugin_registration"
    safe_sig = _sanitize_symbol(target_signature, "vllm.plugins")
    safe_url = run_url if PUBLIC_GITHUB_URL_RE.fullmatch(run_url or "") else ""
    safe_author = sanitize_github_login(first_failing_author)

    if repro_kind.strip():
        kind = repro_kind.strip()
        if kind not in VALID_REPRO_KINDS:
            raise ValueError(f"unsupported repro_kind: {kind!r}")
    elif non_ppu_repro is not None and non_ppu_repro.repro_kind in VALID_REPRO_KINDS:
        kind = non_ppu_repro.repro_kind
    else:
        kind = select_repro_kind(safe_stage, safe_sig)

    spec = build_stub_plugin_spec(kind)
    script_text = _render_standalone_repro_script(
        first_failing_sha=bad,
        last_good_sha=good,
        repro_kind=kind,
    )

    commit_line = f"- **First failing vLLM commit:** `{bad}`"
    if safe_author:
        commit_line += f" (cc @{safe_author})"

    draft_lines = [
        f"### Out-of-tree plugin compatibility regression (`{safe_stage}` / `{safe_sig}`)",
        "",
        commit_line,
        f"- **Last passing vLLM commit (control-verified):** `{good}`",
        f"- **Hardware-free reproducer mode:** `{spec['mode']}` (`{kind}`)",
    ]
    if non_ppu_repro is not None and is_valid_non_ppu_repro(
        non_ppu_repro,
        stage=safe_stage,
        last_good_sha=good,
        first_failing_sha=bad,
        require_shas=True,
    ):
        draft_lines.append(
            f"- **Verified non-PPU exit codes:** `{good}` -> `{non_ppu_repro.last_good_exit_code}`; "
            f"`{bad}` -> `{non_ppu_repro.first_failing_exit_code}`"
        )
    draft_lines.extend(
        [
            f"- **Reference out-of-tree plugin (`vllm-project/vllm-sail`) commit:** `{sail}`",
            f"- **Failing stage:** `{safe_stage}`",
            f"- **Affected public interface:** `{safe_sig}`",
        ]
    )
    if safe_url:
        draft_lines.append(f"- **Public CI run:** {safe_url}")
    draft_lines.extend(
        [
            "",
            f"#### Minimal hardware-free reproducer (`VLLM_TARGET_DEVICE=empty`, `{kind}`)",
            "",
            "```bash",
            script_text,
            "```",
        ]
    )

    lines = [
        "### Manual upstream issue draft for `vllm-project/vllm`",
        "",
        "- **Action required:** Review and file manually in `vllm-project/vllm` (CI never auto-files cross-repo issues or emits live mentions).",
        f"- **Reproducer kind:** `{kind}` (`{spec['mode']}`, no PPU hardware required)",
        f"- **First failing vLLM commit:** `{bad}`",
        f"- **Last passing vLLM commit (control-verified):** `{good}`",
        f"- **Affected public interface:** `{safe_sig}`",
        "",
        "````markdown",
        *draft_lines,
        "````",
        "",
    ]
    text = "\n".join(lines)
    assert_zero_sensitive_leaks(text)
    return text


def render_sail_bucket_issue(
    *,
    bucket_key: str,
    first_failing_sha: str,
    last_good_sha: str,
    sail_sha: str,
    image_digest: str,
    stage: str,
    target_signature: str,
    run_url: str = "",
) -> tuple[str, str]:
    """Render ``(title, body)`` for bucketed tracking inside ``vllm-project/vllm-sail``."""
    safe_stage = stage if stage in VALID_STAGES else "patch_drift"
    safe_sig = _sanitize_symbol(target_signature, "unspecified")
    digest_id = hashlib.sha256(bucket_key.encode("utf-8")).hexdigest()[:10]
    title = (
        f"[Post-Merge][sail_internal_patch_drift][{digest_id}] {safe_stage}: {safe_sig}"
    )
    bad = first_failing_sha if SHA1_RE.fullmatch(first_failing_sha) else "unbisected"
    good = last_good_sha if SHA1_RE.fullmatch(last_good_sha) else "unknown"
    sail = _require_sha1(sail_sha, "sail_sha")
    img = extract_image_digest(image_digest)
    safe_url = run_url if PUBLIC_GITHUB_URL_RE.fullmatch(run_url or "") else ""

    lines = [
        "## Post-Merge Internal Patch / Kernel Drift",
        "",
        f"- **Bucket ID:** `{digest_id}`",
        "- **Classification:** `sail_internal_patch_drift` (track and fix in `vllm-sail`; do not file in `vllm-project/vllm`)",
        "- **Non-PPU reproducer gate:** `not_verified` (PPU-only or internal SAIL failure; no upstream report generated)",
        f"- **Failure stage:** `{safe_stage}`",
        f"- **Target signature:** `{safe_sig}`",
        f"- **First failing upstream vLLM SHA:** `{bad}`",
        f"- **Last known-good upstream vLLM SHA:** `{good}`",
        f"- **Tested `vllm-sail` SHA:** `{sail}`",
        f"- **Container image digest:** `{img}`",
    ]
    if safe_url:
        lines.append(f"- **Public Actions run:** {safe_url}")
    body = "\n".join(lines) + "\n"
    assert_zero_sensitive_leaks(title)
    assert_zero_sensitive_leaks(body)
    return title, body


# ---------------------------------------------------------------------------
# 5. Three-Channel Allowlist Publisher (Console + Artifact + Summary + Issue)
# ---------------------------------------------------------------------------


def publish_sanitized_channels(
    *,
    gear: str,
    status: str,
    sail_sha: str,
    upstream_sha: str,
    verified_pin: str,
    build_image: str = "",
    runtime_image: str = "",
    rc_tag: str = "",
    cpu_status: str = "not_run",
    ppu_status: str = "not_run",
    ppu_skip_reason: str = "none",
    first_failing_sha: str = "",
    last_good_sha: str = "",
    bisect_steps: int = 0,
    outcome: ProbeOutcome | None = None,
    non_ppu_repro: NonPpuReproEvidence | None = None,
    first_failing_author: str = "",
    known_soft_drift_count: int = 0,
    run_url: str = "",
    public_dir: Path,
    step_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Construct, mechanically redact-verify, and write public outputs."""
    if gear not in SELECTED_GEARS:
        raise ValueError(f"invalid selected gear: {gear!r}")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    sail = _require_sha1(sail_sha, "sail_sha")
    pin = _require_sha1(verified_pin, "verified_pin")
    up_sha = _require_sha1(upstream_sha, "upstream_sha") if upstream_sha else ""
    norm_rc = _normalize_rc_ref(rc_tag)
    safe_rc = norm_rc if (norm_rc and _is_valid_rc_ref(norm_rc)) else ""
    safe_skip = ppu_skip_reason if ppu_skip_reason in VALID_PPU_SKIP_REASONS else "none"
    safe_url = run_url if PUBLIC_GITHUB_URL_RE.fullmatch(run_url or "") else ""
    build_digest = extract_image_digest(build_image)
    runtime_digest = extract_image_digest(runtime_image)

    safe_first_bad = first_failing_sha if SHA1_RE.fullmatch(first_failing_sha) else ""
    safe_last_good = (
        last_good_sha
        if (
            SHA1_RE.fullmatch(last_good_sha)
            and last_good_sha != up_sha
            and last_good_sha != safe_first_bad
        )
        else ""
    )

    triage = (
        classify_failure(
            outcome,
            non_ppu_repro=non_ppu_repro,
            last_good_sha=safe_last_good,
            first_failing_sha=safe_first_bad,
            require_shas=True,
        )
        if outcome is not None and status in ("fail", "infra_error")
        else {
            "classification": "infra_error" if status == "infra_error" else "none",
            "stage": "none",
            "target_signature": "none",
            "bucket_key": "",
            "non_ppu_repro_verified": False,
            "non_ppu_repro_mode": "",
            "repro_kind": "",
        }
    )

    effective_repro = (
        non_ppu_repro
        if non_ppu_repro is not None
        else (outcome.non_ppu_repro if outcome is not None else None)
    )
    safe_author = ""
    reproducer_md = ""
    issue_title = ""
    issue_body = ""
    if triage["classification"] == "plugin_interface_break" and safe_first_bad:
        safe_author = sanitize_github_login(first_failing_author)
        reproducer_md = render_upstream_reproducer(
            first_failing_sha=safe_first_bad,
            last_good_sha=safe_last_good,
            sail_sha=sail,
            stage=triage["stage"],
            target_signature=triage["target_signature"],
            repro_kind=triage["repro_kind"],
            first_failing_author=safe_author,
            non_ppu_repro=effective_repro,
            run_url=safe_url,
        )
    elif triage["classification"] == "sail_internal_patch_drift":
        issue_title, issue_body = render_sail_bucket_issue(
            bucket_key=triage["bucket_key"],
            first_failing_sha=safe_first_bad,
            last_good_sha=safe_last_good,
            sail_sha=sail,
            image_digest=runtime_digest
            if runtime_digest != "not_recorded"
            else build_digest,
            stage=triage["stage"],
            target_signature=triage["target_signature"],
            run_url=safe_url,
        )

    public_summary: dict[str, Any] = {
        "schema": 1,
        "gear": gear,
        "status": status,
        "commits": {
            "vllm": up_sha,
            "vllm_sail": sail,
            "verified_pin": pin,
        },
        "rc_tag": safe_rc or None,
        "images": {
            "build_digest": build_digest,
            "runtime_digest": runtime_digest,
        },
        "checks": {
            "cpu_fast_check": cpu_status
            if cpu_status in ("pass", "fail", "not_run", "infra_error")
            else "not_run",
            "ppu_smoke": ppu_status
            if ppu_status in ("pass", "fail", "skipped", "not_run", "infra_error")
            else "not_run",
            "ppu_skip_reason": safe_skip,
            "known_soft_drift_count": max(0, min(int(known_soft_drift_count), 10000)),
        },
        "bisect": {
            "first_failing_sha": (safe_first_bad or None) if status == "fail" else None,
            "last_good_sha": (safe_last_good or None) if status == "fail" else None,
            "steps": max(0, min(int(bisect_steps), 64)),
            "first_failing_author": safe_author or None,
        },
        "triage": {
            "classification": triage["classification"],
            "stage": triage["stage"],
            "target_signature": triage["target_signature"],
            "bucket_key": triage["bucket_key"] or None,
            "non_ppu_repro_verified": bool(triage["non_ppu_repro_verified"]),
            "non_ppu_repro_mode": triage["non_ppu_repro_mode"] or None,
            "repro_kind": triage["repro_kind"] or None,
        },
        "run_url": safe_url or None,
    }

    md_lines = [
        f"### Post-Merge Upstream Compatibility (`{gear}`)",
        "",
        f"- **Status:** `{status}`",
        f"- **Upstream `vllm` SHA:** `{up_sha or 'none'}`"
        + (f" (`{safe_rc}`)" if safe_rc else ""),
        f"- **`vllm-sail` SHA:** `{sail}` · **Verified Pin:** `{pin}`",
        f"- **Images (build / runtime):** `{build_digest}` / `{runtime_digest}`",
        f"- **CPU fast check:** `{public_summary['checks']['cpu_fast_check']}` "
        f"(known soft drift: `{public_summary['checks']['known_soft_drift_count']}`)",
        f"- **PPU smoke:** `{public_summary['checks']['ppu_smoke']}` "
        f"(skip reason: `{safe_skip}`)",
    ]
    if status in ("fail", "infra_error"):
        md_lines.extend(
            [
                f"- **Triage classification:** `{triage['classification']}` "
                f"(`{triage['stage']}` / `{triage['target_signature']}`)",
                f"- **Non-PPU reproducer verified:** `{str(bool(triage['non_ppu_repro_verified'])).lower()}`"
                + (f" (`{triage['repro_kind']}`)" if triage["repro_kind"] else ""),
                f"- **First failing SHA:** `{safe_first_bad or 'n/a'}` "
                f"(bisect steps: `{public_summary['bisect']['steps']}`)",
            ]
        )
    if reproducer_md:
        md_lines.extend(["", reproducer_md.rstrip()])
    markdown_text = "\n".join(md_lines) + "\n"

    console_text = (
        f"post_merge_gear={gear} status={status} vllm_sha={up_sha or 'none'} "
        f"sail_sha={sail} classification={triage['classification']} "
        f"stage={triage['stage']} non_ppu_repro={str(bool(triage['non_ppu_repro_verified'])).lower()} "
        f"ppu_skip={safe_skip}"
    )

    public_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = public_dir / "public-summary.json"
    issue_json_path = public_dir / "issue-payload.json"
    staging = [summary_json_path, issue_json_path]
    if step_summary_path is not None:
        staging.append(step_summary_path)

    summary_json_text = (
        json.dumps(public_summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    issue_payload = {
        "should_file_sail_issue": bool(
            triage["classification"] == "sail_internal_patch_drift"
            and issue_title
            and not (triage["stage"] == "wheel_build" and not safe_first_bad)
        ),
        "should_file_upstream_issue": False,
        "non_ppu_repro_verified": bool(triage["non_ppu_repro_verified"]),
        "first_failing_author": safe_author,
        "title": issue_title,
        "body": issue_body,
        "upstream_reproducer_markdown": reproducer_md,
    }
    issue_json_text = (
        json.dumps(issue_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    for channel_text in (
        console_text,
        summary_json_text,
        markdown_text,
        issue_json_text,
    ):
        assert_zero_sensitive_leaks(channel_text, staging_paths=staging)

    summary_json_path.write_text(summary_json_text, encoding="utf-8")
    issue_json_path.write_text(issue_json_text, encoding="utf-8")
    if step_summary_path is not None:
        with step_summary_path.open("a", encoding="utf-8") as handle:
            handle.write(markdown_text)

    return {
        "console": console_text,
        "public_summary": public_summary,
        "markdown": markdown_text,
        "issue_payload": issue_payload,
    }


def prepare_pin_bump(
    *,
    repo_root: Path,
    target_sha: str = "",
    verified_pin_file: Path | None = None,
    manifest_file: Path | None = None,
    vllm_repo: Path | None = None,
    cursor: dict[str, Any] | None = None,
    ls_remote_data: dict[str, Any] | None = None,
    require_cursor: bool = False,
) -> dict[str, str]:
    """Validate and atomically bump ``vllm-main-verified.commit`` and ``manifest.toml``."""
    pin_path = verified_pin_file or (
        repo_root / ".github" / "vllm-main-verified.commit"
    )
    manifest_path = manifest_file or (repo_root / "csrc" / "upstream" / "manifest.toml")
    previous = _require_sha1(pin_path.read_text(encoding="utf-8"), "previous_pin")
    sail_base_sha = ""
    if (repo_root / ".git").exists():
        rev_sail = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--verify",
                "--quiet",
                "HEAD^{commit}",
            ],
            capture_output=True,
            text=True,
        )
        cand_sail = rev_sail.stdout.strip()
        if rev_sail.returncode == 0 and SHA1_RE.fullmatch(cand_sail):
            sail_base_sha = cand_sail

    def _noop() -> dict[str, str]:
        return {
            "changed": "false",
            "previous_pin": previous,
            "new_pin": previous,
            "sail_base_sha": sail_base_sha,
            "title": "",
            "body": "",
        }

    if require_cursor and cursor is None:
        return _noop()

    main_tip = str((ls_remote_data or {}).get("main_sha") or "").strip()
    if cursor is not None:
        state = _sanitize_cursor(cursor, verified_pin=previous)
        bump_excluded = _collect_excluded_control_shas(
            state,
            tags=(ls_remote_data or {}).get("tags"),
            release_branches=(ls_remote_data or {}).get("release_branches"),
        )
        history = list(state.get("probe_history") or [])
        if not history and state.get("last_probe_record"):
            history = [state["last_probe_record"]]
        newer_than_pin: list[dict[str, Any]] = []
        for rec in history:
            if not isinstance(rec, dict) or rec.get("gear") != "probe":
                continue
            if str(rec.get("upstream_sha", "")).strip() == previous:
                break
            newer_than_pin.append(rec)
        if target_sha.strip():
            req_target = _require_sha1(target_sha, "target_sha")
            if req_target == previous:
                return _noop()
            if vllm_repo is not None and resolve_complete_first_parent_range(
                vllm_repo, req_target, previous
            ):
                return _noop()
            matching = [
                rec
                for rec in newer_than_pin
                if str(rec.get("upstream_sha", "")).strip() == req_target
            ]
            verified_cand = select_pin_bump_candidate(
                matching,
                verified_pin=previous,
                excluded_shas=bump_excluded,
                vllm_repo=vllm_repo,
                main_sha=main_tip,
            )
            if verified_cand != req_target:
                return _noop()
            target = verified_cand
        else:
            verified_cand = select_pin_bump_candidate(
                newer_than_pin,
                verified_pin=previous,
                excluded_shas=bump_excluded,
                vllm_repo=vllm_repo,
                main_sha=main_tip,
            )
            if not verified_cand:
                return _noop()
            target = verified_cand
    else:
        if not target_sha.strip():
            return _noop()
        target = _require_sha1(target_sha, "target_sha")
        if target == previous:
            return _noop()

    if vllm_repo is not None:
        forward_range = resolve_complete_first_parent_range(vllm_repo, previous, target)
        if not forward_range:
            raise ValueError(
                "target_sha must be a strict first-parent descendant of previous_pin"
            )
        if not _is_on_main_first_parent_lineage(
            vllm_repo,
            previous_pin=previous,
            candidate_sha=target,
            main_sha=main_tip,
        ):
            raise ValueError(
                "target_sha must belong to upstream main first-parent history"
            )

    if manifest_path.is_file():
        manifest_text = manifest_path.read_text(encoding="utf-8")
        new_manifest_text, count = re.subn(
            r'(?m)^upstream_ref\s*=\s*"[0-9a-f]{40}"\s*$',
            f'upstream_ref = "{target}"',
            manifest_text,
        )
        if count != 1:
            raise ValueError(
                f"expected exactly one 40-hex upstream_ref assignment in {manifest_path}"
            )
        manifest_path.write_text(new_manifest_text, encoding="utf-8")
        if _load_toml(manifest_path).get("upstream_ref") != target:
            raise ValueError(
                "manifest.toml upstream_ref verification failed after bump"
            )

    pin_path.write_text(target + "\n", encoding="utf-8")
    title = f"[CI][Pin] Advance verified upstream vLLM commit to {target[:12]}"
    body = (
        "## Summary\n\n"
        f"- Advance `.github/vllm-main-verified.commit` and `csrc/upstream/manifest.toml` `upstream_ref` from `{previous}` to probe-verified `{target}`.\n"
        "- Validated via post-merge probe (`check_patch_drift` + Tier-A kernel compatibility + PPU smoke); full PR Fast Gate and Nightly Model Sanity apply before/after merge.\n"
    )
    assert_zero_sensitive_leaks(title)
    assert_zero_sensitive_leaks(body)
    return {
        "changed": "true",
        "previous_pin": previous,
        "new_pin": target,
        "sail_base_sha": sail_base_sha,
        "title": title,
        "body": body,
    }


def sanitize_build_manifest_file(
    manifest_path: Path, *, output_path: Path | None = None
) -> str:
    """Rewrite ``build-manifest.txt`` to an allowlisted public key-value set (fail-closed)."""
    dest_path = output_path or manifest_path
    staging = (
        [manifest_path, dest_path] if dest_path != manifest_path else [manifest_path]
    )
    try:
        exists = manifest_path.is_file()
        raw_content = (
            manifest_path.read_text(encoding="utf-8", errors="replace")
            if exists
            else ""
        )
        # Remove raw source before writing dest so an interrupted/failed run
        # never leaves unredacted diagnostics behind.
        if manifest_path.exists() or manifest_path.is_symlink():
            manifest_path.unlink(missing_ok=True)
        if dest_path != manifest_path and (
            dest_path.exists() or dest_path.is_symlink()
        ):
            dest_path.unlink(missing_ok=True)

        if not exists:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            text = "status=missing\n"
            assert_zero_sensitive_leaks(text, staging_paths=staging)
            dest_path.write_text(text, encoding="utf-8")
            return text

        raw_fields: dict[str, str] = {}
        for raw_line in raw_content.splitlines():
            key, sep, val = raw_line.partition("=")
            if not sep or not key:
                continue
            raw_fields[key.strip()] = val.strip()

        out_lines: list[str] = []
        st = raw_fields.get("status", "unknown")
        out_lines.append(
            f"status={st if st in ('running', 'succeeded', 'failed') else 'unknown'}"
        )
        if re.fullmatch(r"[0-9]{1,3}", raw_fields.get("exit_code", "")):
            out_lines.append(f"exit_code={raw_fields['exit_code']}")
        for sha_key in (
            "vllm_ref",
            "candidate_vllm_commit",
            "vllm_commit",
            "vllm_sail_commit",
        ):
            val = raw_fields.get(sha_key, "")
            if SHA1_RE.fullmatch(val):
                out_lines.append(f"{sha_key}={val}")
        for sha256_key in ("sdk_sha256", "torch_sha256"):
            val = raw_fields.get(sha256_key, "")
            if re.fullmatch(r"[0-9a-f]{64}", val):
                out_lines.append(f"{sha256_key}={val}")
        arch = raw_fields.get("pytorch_sail_arch", "")
        if re.fullmatch(r"[A-Za-z0-9_;.-]{1,64}", arch):
            out_lines.append(f"pytorch_sail_arch={arch}")
        reuse = raw_fields.get("reuse_mode", "")
        if reuse in ("native", "sail-native", "vllm-only", "full"):
            out_lines.append(f"reuse_mode={reuse}")
        out_lines.append(
            f"build_image_digest={extract_image_digest(raw_fields.get('build_image', ''))}"
        )

        sanitized_text = "\n".join(out_lines) + "\n"
        assert_zero_sensitive_leaks(sanitized_text, staging_paths=staging)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(sanitized_text, encoding="utf-8")
        return sanitized_text
    except BaseException:
        for path in staging:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def resolve_complete_first_parent_range(
    vllm_repo: Path,
    last_green_sha: str,
    red_sha: str,
    *,
    max_deepen_rounds: int = 8,
    deepen_step: int = 250,
) -> list[str] | None:
    """Return ``(last_green_sha..red_sha]`` iff the unbroken first-parent chain is verified."""
    green = _require_sha1(last_green_sha, "last_green_sha")
    red = _require_sha1(red_sha, "red_sha")
    if green == red:
        return []

    def _walk() -> tuple[list[str] | None, set[str]]:
        has_green = subprocess.run(
            ["git", "-C", str(vllm_repo), "cat-file", "-e", f"{green}^{{commit}}"],
            capture_output=True,
        )
        if has_green.returncode != 0:
            return None, set()
        rev = subprocess.run(
            [
                "git",
                "-C",
                str(vllm_repo),
                "rev-list",
                "--first-parent",
                "--parents",
                red,
            ],
            capture_output=True,
            text=True,
        )
        if rev.returncode != 0:
            return None, set()
        chain: list[str] = []
        visited: set[str] = {red}
        for raw_line in rev.stdout.splitlines():
            parts = raw_line.split()
            if not parts:
                continue
            sha = parts[0]
            if not SHA1_RE.fullmatch(sha):
                return None, set()
            visited.add(sha)
            if sha == green:
                if not chain:
                    return [], visited
                chain.reverse()
                return chain, visited
            chain.append(sha)
            if len(parts) < 2:
                # Hit root commit or .git/shallow graft boundary before green.
                return None, visited
        return None, visited

    def _diverged_above_walk(visited: set[str]) -> bool:
        if not visited:
            return False
        mb_proc = subprocess.run(
            ["git", "-C", str(vllm_repo), "merge-base", green, red],
            capture_output=True,
            text=True,
        )
        if mb_proc.returncode != 0:
            return False
        mb = mb_proc.stdout.strip()
        if not SHA1_RE.fullmatch(mb) or mb == green or mb not in visited:
            return False
        anc_proc = subprocess.run(
            [
                "git",
                "-C",
                str(vllm_repo),
                "merge-base",
                "--is-ancestor",
                mb,
                green,
            ],
            capture_output=True,
        )
        return anc_proc.returncode == 0

    complete, walked = _walk()
    if complete is not None:
        return complete
    if _diverged_above_walk(walked):
        return None

    remote_check = subprocess.run(
        ["git", "-C", str(vllm_repo), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if remote_check.returncode != 0:
        return None

    for _ in range(max_deepen_rounds):
        is_shallow = subprocess.run(
            ["git", "-C", str(vllm_repo), "rev-parse", "--is-shallow-repository"],
            capture_output=True,
            text=True,
        )
        if is_shallow.returncode == 0 and is_shallow.stdout.strip() == "false":
            # Full history is already fetched; green is not a first-parent ancestor of red.
            return _walk()[0]
        fetch_proc = subprocess.run(
            [
                "git",
                "-C",
                str(vllm_repo),
                "fetch",
                "--quiet",
                f"--deepen={deepen_step}",
                "origin",
                red,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if fetch_proc.returncode != 0:
            break
        subprocess.run(
            [
                "git",
                "-C",
                str(vllm_repo),
                "fetch",
                "--quiet",
                "--depth=64",
                "origin",
                green,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        complete, walked = _walk()
        if complete is not None:
            return complete
        if _diverged_above_walk(walked):
            return None

    return _walk()[0]


def _ensure_commit_fetched(vllm_repo: Path, sha: str, *, depth: int = 1) -> bool:
    """Ensure ``sha`` is locally present in ``vllm_repo``, fetching from ``origin`` if configured."""
    if not SHA1_RE.fullmatch(sha) or not vllm_repo.is_dir():
        return False
    has_commit = subprocess.run(
        ["git", "-C", str(vllm_repo), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
    )
    if has_commit.returncode == 0:
        return True
    remote_check = subprocess.run(
        ["git", "-C", str(vllm_repo), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if remote_check.returncode != 0:
        return False
    subprocess.run(
        [
            "git",
            "-C",
            str(vllm_repo),
            "fetch",
            "--quiet",
            f"--depth={depth}",
            "origin",
            sha,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    recheck = subprocess.run(
        ["git", "-C", str(vllm_repo), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
    )
    return recheck.returncode == 0


def resolve_verified_bisect_control(
    vllm_repo: Path,
    *,
    primary_green: str,
    red_sha: str,
    fallback_greens: Sequence[str] = (),
    excluded_shas: set[str] | frozenset[str] = frozenset(),
) -> tuple[str, list[str] | None]:
    """Select the newest non-revoked verified control SHA that is a strict first-parent ancestor of ``red_sha``."""
    primary = (
        _require_sha1(primary_green, "last_green_sha") if primary_green.strip() else ""
    )
    red = _require_sha1(red_sha, "red_sha")
    _ensure_commit_fetched(vllm_repo, red, depth=250)

    ordered: list[str] = []
    for raw in (primary, *fallback_greens):
        cand = str(raw or "").strip()
        if (
            SHA1_RE.fullmatch(cand)
            and cand not in excluded_shas
            and cand not in ordered
        ):
            ordered.append(cand)

    if not ordered:
        return "", None

    ordered_set = set(ordered)
    for cand in ordered:
        if cand == red:
            continue
        _ensure_commit_fetched(vllm_repo, cand, depth=64)
        rng = resolve_complete_first_parent_range(vllm_repo, cand, red)
        if rng:
            for idx in range(len(rng) - 2, -1, -1):
                closer = rng[idx]
                if closer in ordered_set and closer != red:
                    return closer, rng[idx + 1 :]
            return cand, rng

    return "", None


# ---------------------------------------------------------------------------
# 6. SDK Container CPU Fast Check Runner
# ---------------------------------------------------------------------------


def run_sdk_cpu_fast_check(
    *,
    vllm_src: Path,
    sail_root: Path,
    target_sha: str,
    verified_pin: str,
    previous_baseline: Path | None = None,
    output_json: Path,
    python_exe: str = sys.executable,
) -> int:
    """Execute SDK-container CPU compatibility checks against ``vllm_src``."""
    target = _require_sha1(target_sha, "target_sha")
    pin = _require_sha1(verified_pin, "verified_pin")
    env = dict(os.environ)
    env.pop("VLLM_VERSION", None)
    env.pop("VLLM_VERSION_OVERRIDE", None)
    env.pop("VLLM_PPU_CANDIDATE_UPSTREAM_COMMIT", None)
    if target != pin:
        env["VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT"] = target
    else:
        env.pop("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", None)
    env["VLLM_SOURCE_ROOT"] = str(vllm_src.resolve())

    known_soft_count = 0
    patch_fingerprints: dict[str, str] = {}

    def _write_result(outcome: ProbeOutcome) -> int:
        safe_sig = (
            _sanitize_patch_target_signature(outcome.target_signature, "patch_baseline")
            if outcome.stage == "patch_drift"
            else _sanitize_symbol(outcome.target_signature, "none")
        )
        payload = {
            "status": outcome.status,
            "stage": outcome.stage,
            "target_signature": safe_sig,
            "is_public_plugin_interface": outcome.is_public_plugin_interface,
            "known_soft_drift_count": known_soft_count,
            "patch_fingerprints": patch_fingerprints,
        }
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        assert_zero_sensitive_leaks(text, staging_paths=[output_json])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")
        return 0 if outcome.status == "pass" else 1

    # Step 1: check_vllm_compatibility + register() + register_out_of_tree()
    reg_code = (
        "import json\n"
        "import vllm_sail\n"
        "from vllm_sail.compat import check_vllm_compatibility\n"
        "check_vllm_compatibility(force=True)\n"
        "vllm_sail.register()\n"
        "vllm_sail.register_out_of_tree()\n"
    )
    reg = subprocess.run(
        [python_exe, "-c", reg_code],
        cwd="/tmp",
        env={k: v for k, v in env.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    if reg.returncode != 0:
        combined = reg.stderr + "\n" + reg.stdout
        matched_public = next(
            (sym for sym in PUBLIC_PLUGIN_SYMBOLS if sym in combined),
            "",
        )
        is_public = "vllm_sail/patch/" not in combined and bool(matched_public)
        sig = matched_public if is_public else "vllm_sail.register_out_of_tree"
        if not is_public:
            m = re.search(r"(vllm(?:\.[A-Za-z0-9_]+)+)", combined)
            if m:
                sig = m.group(1)
        stage = (
            "compat_check"
            if "candidate mode requires" in combined
            or "requires verified vLLM commit" in combined
            else "plugin_registration"
        )
        return _write_result(
            ProbeOutcome(
                status="fail",
                stage=stage,
                target_signature=sig,
                is_public_plugin_interface=is_public,
            )
        )

    # Step 2: static upstream_source AST contract checks
    contracts = subprocess.run(
        [
            python_exe,
            "-m",
            "pytest",
            "tests/ut/test_patch_contracts.py",
            "-q",
            "-rs",
            "-m",
            "upstream_source",
            "--require-upstream-source",
        ],
        cwd=str(sail_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if contracts.returncode != 0:
        m = re.search(
            r"(vllm(?:\.[A-Za-z0-9_]+)+)", contracts.stdout + "\n" + contracts.stderr
        )
        sig = m.group(1) if m else "patch_contracts"
        return _write_result(
            ProbeOutcome(
                status="fail",
                stage="patch_contracts",
                target_signature=sig,
                is_public_plugin_interface=False,
            )
        )

    # Step 3: check_patch_drift.py (incremental vs known baseline soft drift)
    drift_json = output_json.parent / "patch-drift-eval.json"
    drift_cmd = [
        python_exe,
        str(sail_root / "tools" / "check_patch_drift.py"),
        "--json-output",
        str(drift_json),
    ]
    if previous_baseline is not None and previous_baseline.is_file():
        drift_cmd.extend(
            ["--previous-baseline", str(previous_baseline), "--allow-known-soft-drift"]
        )
    drift = subprocess.run(
        drift_cmd, cwd=str(sail_root), env=env, capture_output=True, text=True
    )
    if drift_json.is_file():
        drift_data = json.loads(drift_json.read_text(encoding="utf-8"))
        known_soft_count = len(drift_data.get("known_soft_drifted") or [])
        raw_opaque = drift_data.get("opaque_fingerprints")
        if isinstance(raw_opaque, dict):
            patch_fingerprints = {str(k): str(v) for k, v in sorted(raw_opaque.items())}
        else:
            patch_fingerprints = {
                opaque_target_id(str(k)): str(v)
                for k, v in sorted((drift_data.get("fingerprints") or {}).items())
            }
        drift_json.unlink(missing_ok=True)
    else:
        drift_data = {}
    if drift.returncode != 0:
        candidates = (
            list(drift_data.get("removed") or [])
            + list(drift_data.get("incremental_drifted") or [])
            + list(drift_data.get("added") or [])
        )
        sig = candidates[0] if candidates else "patch_baseline"
        return _write_result(
            ProbeOutcome(
                status="fail",
                stage="patch_drift",
                target_signature=sig,
                is_public_plugin_interface=False,
            )
        )

    # Step 4: Tier-A kernel manifest & op schema parity (+ sailify port --check if installed)
    kernel_problems = check_kernel_manifest_compatibility(vllm_src, sail_root=sail_root)
    if kernel_problems:
        return _write_result(
            ProbeOutcome(
                status="fail",
                stage="kernel_port_check",
                target_signature=kernel_problems[0],
                is_public_plugin_interface=False,
            )
        )

    has_sailify = (
        subprocess.run(
            [python_exe, "-c", "import sailify"],
            env=env,
            capture_output=True,
        ).returncode
        == 0
    )
    if has_sailify:
        port_check = subprocess.run(
            [
                python_exe,
                str(sail_root / "tools" / "port_upstream_kernels.py"),
                "--vllm-src",
                str(vllm_src),
                "--check",
            ],
            cwd=str(sail_root),
            env=env,
            capture_output=True,
            text=True,
        )
        if port_check.returncode != 0:
            m = re.search(
                r"(csrc/[A-Za-z0-9_./-]+)", port_check.stderr + "\n" + port_check.stdout
            )
            sig = m.group(1) if m else "csrc/upstream"
            return _write_result(
                ProbeOutcome(
                    status="fail",
                    stage="kernel_port_check",
                    target_signature=sig,
                    is_public_plugin_interface=False,
                )
            )

    return _write_result(ProbeOutcome(status="pass"))


def run_source_only_build_triage(
    *,
    vllm_repo: Path,
    sail_root: Path,
    target_sha: str,
    verified_pin: str,
    last_green_sha: str = "",
    fallback_green_shas: Sequence[str] = (),
    cursor: dict[str, Any] | None = None,
    rc_tag: str = "",
    output_dir: Path,
    probe_cmd: str = "",
    python_exe: str = sys.executable,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    """Run source-only, control-first build check, bisection, and non-PPU verification when initial wheel build fails."""
    pin = _require_sha1(verified_pin, "verified_pin")
    target = _require_sha1(target_sha, "target_sha")
    raw_green = last_green_sha.strip()
    primary_control = raw_green if SHA1_RE.fullmatch(raw_green) else pin
    excluded_controls: set[str] = set()
    extra_controls: list[str] = [
        str(s).strip()
        for s in fallback_green_shas
        if SHA1_RE.fullmatch(str(s or "").strip())
    ]
    if cursor is not None:
        sanitized_cur = _sanitize_cursor(cursor, verified_pin=pin)
        excluded_controls = _collect_excluded_control_shas(
            sanitized_cur,
            rc_ref=rc_tag,
            target_sha=target,
        )
        for cand_s in _collect_verified_control_shas(
            sanitized_cur,
            verified_pin=pin,
            rc_ref=rc_tag,
            target_sha=target,
        ):
            if cand_s not in extra_controls:
                extra_controls.append(cand_s)
    elif pin not in extra_controls:
        extra_controls.append(pin)
    control, _ = resolve_verified_bisect_control(
        vllm_repo,
        primary_green=primary_control,
        red_sha=target,
        fallback_greens=extra_controls,
        excluded_shas=excluded_controls,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_json_path = output_dir / "cpu-result.json"
    bisect_json_path = output_dir / "bisect-result.json"
    repro_json_path = output_dir / "non-ppu-repro.json"
    repro_json_path.unlink(missing_ok=True)

    def _write_sanitized(path: Path, data: dict[str, Any]) -> None:
        text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
        assert_zero_sensitive_leaks(text, staging_paths=[path])
        path.write_text(text, encoding="utf-8")

    def _write_cpu_and_bisect(
        *,
        cpu_status: str,
        stage: str,
        target_sig: str,
        is_public: bool,
        bisect_status: str,
        first_failing: str,
        last_good: str,
        steps_executed: int,
        probe_steps_on_range: int,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        safe_stage = stage if stage in VALID_STAGES else "wheel_build"
        safe_sig = _sanitize_symbol(target_sig, "VLLM_TARGET_DEVICE=empty")
        cpu_payload = {
            "status": cpu_status,
            "stage": safe_stage,
            "target_signature": safe_sig,
            "is_public_plugin_interface": bool(is_public),
            "known_soft_drift_count": 0,
            "patch_fingerprints": {},
        }
        bisect_payload = {
            "bisect_status": bisect_status,
            "first_failing_sha": first_failing,
            "last_good_sha": last_good,
            "steps_executed": steps_executed,
            "probe_steps_on_range": probe_steps_on_range,
            "stage": safe_stage,
            "target_signature": safe_sig,
            "is_public_plugin_interface": bool(is_public),
            "history": history,
        }
        _write_sanitized(cpu_json_path, cpu_payload)
        _write_sanitized(bisect_json_path, bisect_payload)
        return {"cpu": cpu_payload, "bisect": bisect_payload}

    with tempfile.TemporaryDirectory(prefix="source-build-triage-") as td:
        workdir = Path(td)
        stub_site, check_script = materialize_stub_plugin("cpu_empty_build", workdir)
        call_counter = 0
        wheel_cache: dict[str, ProbeOutcome] = {}
        kernel_cache: dict[str, ProbeOutcome] = {}

        def _probe_wheel_only(candidate_sha: str) -> ProbeOutcome:
            if candidate_sha in wheel_cache:
                return wheel_cache[candidate_sha]
            nonlocal call_counter
            call_counter += 1
            step_tag = f"{call_counter}-{candidate_sha[:12]}"
            base_env = {
                k: v
                for k, v in os.environ.items()
                if k not in _CLEAN_BUILD_ENV_DROP_KEYS
            }
            base_env["VLLM_TARGET_DEVICE"] = "empty"
            base_env["VLLM_REQUIRE_RUST_FRONTEND"] = "1"
            base_env["CARGO_NET_GIT_FETCH_WITH_CLI"] = "true"
            base_env["CUDA_VISIBLE_DEVICES"] = ""
            base_env["VLLM_PLUGINS"] = ""
            base_env["CANDIDATE_SHA"] = candidate_sha
            base_env["VLLM_REPO"] = str(vllm_repo)
            base_env["SAIL_ROOT"] = str(sail_root)
            base_env["STUB_SITE_DIR"] = str(stub_site)
            base_env["STUB_CHECK_SCRIPT"] = str(check_script)

            if probe_cmd.strip():
                probe_json = workdir / f"probe-{step_tag}.json"
                probe_json.unlink(missing_ok=True)
                marker_path = workdir / f"probe-exec-{step_tag}.marker"
                marker_path.unlink(missing_ok=True)
                runner_script = workdir / f"probe-check-{step_tag}.py"
                runner_script.write_text(
                    "import pathlib, runpy\n"
                    f"pathlib.Path({str(marker_path)!r}).write_text('executed\\n', encoding='utf-8')\n"
                    f"runpy.run_path({str(check_script)!r}, run_name='__main__')\n",
                    encoding="utf-8",
                )
                base_env["STUB_CHECK_SCRIPT"] = str(runner_script)
                base_env["PROBE_RESULT_JSON"] = str(probe_json)
                try:
                    proc = subprocess.run(
                        ["bash", "-c", probe_cmd],
                        env=base_env,
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    out = ProbeOutcome(
                        status="infra_error",
                        stage="wheel_build",
                        target_signature="VLLM_TARGET_DEVICE=empty",
                        detail_code="infra_error",
                    )
                    wheel_cache[candidate_sha] = out
                    return out
                if probe_json.is_file():
                    data = json.loads(probe_json.read_text(encoding="utf-8"))
                    probe_json.unlink(missing_ok=True)
                    raw_st = str(
                        data.get("status")
                        or ("pass" if proc.returncode == 0 else "fail")
                    )
                    st = raw_st if raw_st in ("pass", "fail", "infra_error") else "fail"
                    stg = str(data.get("stage") or "wheel_build")
                    out = ProbeOutcome(
                        status=st,
                        stage=stg,
                        target_signature=str(
                            data.get("target_signature") or "VLLM_TARGET_DEVICE=empty"
                        ),
                        is_public_plugin_interface=bool(
                            data.get("is_public_plugin_interface", stg == "wheel_build")
                        ),
                        detail_code=str(data.get("phase") or stg),
                    )
                    wheel_cache[candidate_sha] = out
                    return out
                executed = marker_path.is_file()
                if proc.returncode == 0:
                    out = ProbeOutcome(status="pass", detail_code="pass")
                    wheel_cache[candidate_sha] = out
                    return out
                combined_probe_out = f"{proc.stdout}\n{proc.stderr}"
                inferred_probe_sig = (
                    "rust_frontend_build"
                    if (
                        _RUSTC_REPO_SPAN_RE.search(combined_probe_out) is not None
                        or (executed and "required local Rust" in combined_probe_out)
                    )
                    else "VLLM_TARGET_DEVICE=empty"
                )
                if _is_infra_process_failure(proc.returncode, proc.stderr, proc.stdout):
                    out = ProbeOutcome(
                        status="infra_error",
                        stage="wheel_build",
                        target_signature=inferred_probe_sig,
                        detail_code="infra_error",
                    )
                    wheel_cache[candidate_sha] = out
                    return out
                if executed:
                    out = ProbeOutcome(
                        status="fail",
                        stage="wheel_build",
                        target_signature=inferred_probe_sig,
                        is_public_plugin_interface=True,
                        detail_code="build_import",
                    )
                    wheel_cache[candidate_sha] = out
                    return out
                phase, _src, is_infra = _classify_install_failure(
                    proc.returncode,
                    proc.stderr,
                    proc.stdout,
                    mode="cpu_empty_build",
                )
                if is_infra:
                    out = ProbeOutcome(
                        status="infra_error",
                        stage="wheel_build",
                        target_signature=inferred_probe_sig,
                        detail_code="infra_error",
                    )
                else:
                    out = ProbeOutcome(
                        status="fail",
                        stage="wheel_build",
                        target_signature=inferred_probe_sig,
                        is_public_plugin_interface=True,
                        detail_code=phase,
                    )
                wheel_cache[candidate_sha] = out
                return out

            rc, phase, _src, _exec, is_infra, sig = _execute_vllm_worktree_build(
                vllm_repo=vllm_repo,
                sha=candidate_sha,
                workdir=workdir,
                step_tag=step_tag,
                mode="cpu_empty_build",
                base_env=base_env,
                stub_site=stub_site,
                check_script=check_script,
                python_exe=python_exe,
                timeout_sec=timeout_sec,
            )
            if rc == 0 and not is_infra:
                out = ProbeOutcome(status="pass", detail_code="pass")
            elif is_infra:
                out = ProbeOutcome(
                    status="infra_error",
                    stage="wheel_build",
                    target_signature=sig,
                    detail_code="infra_error",
                )
            else:
                out = ProbeOutcome(
                    status="fail",
                    stage="wheel_build",
                    target_signature=sig,
                    is_public_plugin_interface=True,
                    detail_code=phase,
                )
            wheel_cache[candidate_sha] = out
            return out

        def _probe_kernel_only(candidate_sha: str) -> ProbeOutcome:
            if candidate_sha in kernel_cache:
                return kernel_cache[candidate_sha]
            nonlocal call_counter
            call_counter += 1
            wt_dir = workdir / f"kern-{call_counter}-{candidate_sha[:12]}"
            wt_added = False
            try:
                add_proc = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(vllm_repo),
                        "worktree",
                        "add",
                        "--quiet",
                        "--detach",
                        str(wt_dir),
                        candidate_sha,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if add_proc.returncode != 0:
                    out = ProbeOutcome(
                        status="infra_error",
                        stage="kernel_port_check",
                        target_signature="csrc/upstream",
                    )
                    kernel_cache[candidate_sha] = out
                    return out
                wt_added = True
                problems = check_kernel_manifest_compatibility(
                    wt_dir, sail_root=sail_root
                )
                if problems:
                    out = ProbeOutcome(
                        status="fail",
                        stage="kernel_port_check",
                        target_signature=problems[0],
                        is_public_plugin_interface=False,
                        detail_code="kernel_port_check",
                    )
                else:
                    out = ProbeOutcome(status="pass", detail_code="pass")
                kernel_cache[candidate_sha] = out
                return out
            except subprocess.TimeoutExpired:
                out = ProbeOutcome(
                    status="infra_error",
                    stage="kernel_port_check",
                    target_signature="csrc/upstream",
                )
                kernel_cache[candidate_sha] = out
                return out
            finally:
                if wt_added:
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(vllm_repo),
                            "worktree",
                            "remove",
                            "--force",
                            str(wt_dir),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

        # 1. Control-first mandatory wheel-build check on `control`.
        if not control:
            return _write_cpu_and_bisect(
                cpu_status="infra_error",
                stage="wheel_build",
                target_sig="VLLM_TARGET_DEVICE=empty",
                is_public=False,
                bisect_status="unresolved_history",
                first_failing="",
                last_good="",
                steps_executed=0,
                probe_steps_on_range=0,
                history=[],
            )
        ctrl_outcome = _probe_wheel_only(control)
        if ctrl_outcome.status != "pass":
            return _write_cpu_and_bisect(
                cpu_status="infra_error",
                stage="wheel_build",
                target_sig=ctrl_outcome.target_signature or "VLLM_TARGET_DEVICE=empty",
                is_public=False,
                bisect_status="baseline_control_failed",
                first_failing="",
                last_good="",
                steps_executed=1,
                probe_steps_on_range=0,
                history=[
                    {
                        "sha": control,
                        "role": "baseline_control",
                        "status": ctrl_outcome.status,
                    }
                ],
            )

        if target == control:
            return _write_cpu_and_bisect(
                cpu_status="infra_error",
                stage="wheel_build",
                target_sig="VLLM_TARGET_DEVICE=empty",
                is_public=False,
                bisect_status="aborted_infra_error",
                first_failing="",
                last_good=control,
                steps_executed=1,
                probe_steps_on_range=0,
                history=[
                    {"sha": control, "role": "baseline_control", "status": "pass"}
                ],
            )

        # 2. Candidate mandatory wheel-build check on `target`.
        red_wheel = _probe_wheel_only(target)
        if red_wheel.status == "infra_error":
            return _write_cpu_and_bisect(
                cpu_status="infra_error",
                stage="wheel_build",
                target_sig=red_wheel.target_signature or "VLLM_TARGET_DEVICE=empty",
                is_public=False,
                bisect_status="aborted_infra_error",
                first_failing="",
                last_good=control,
                steps_executed=2,
                probe_steps_on_range=0,
                history=[
                    {"sha": control, "role": "baseline_control", "status": "pass"},
                    {"sha": target, "role": "target_head", "status": "infra_error"},
                ],
            )

        # 2A. If upstream wheel build itself passed on `target`, check SAIL Tier-A kernel manifest independently.
        if red_wheel.status == "pass" and not probe_cmd.strip():
            ctrl_kern = _probe_kernel_only(control)
            if ctrl_kern.status != "pass":
                return _write_cpu_and_bisect(
                    cpu_status="infra_error",
                    stage="wheel_build",
                    target_sig="VLLM_TARGET_DEVICE=empty",
                    is_public=False,
                    bisect_status="baseline_control_failed",
                    first_failing="",
                    last_good="",
                    steps_executed=3,
                    probe_steps_on_range=0,
                    history=[
                        {
                            "sha": control,
                            "role": "baseline_control",
                            "status": ctrl_kern.status,
                        },
                        {"sha": target, "role": "target_head", "status": "pass"},
                    ],
                )
            red_kern = _probe_kernel_only(target)
            if red_kern.status != "fail":
                return _write_cpu_and_bisect(
                    cpu_status="infra_error",
                    stage="wheel_build",
                    target_sig="VLLM_TARGET_DEVICE=empty",
                    is_public=False,
                    bisect_status="aborted_infra_error",
                    first_failing="",
                    last_good=control,
                    steps_executed=2,
                    probe_steps_on_range=0,
                    history=[
                        {"sha": control, "role": "baseline_control", "status": "pass"},
                        {"sha": target, "role": "target_head", "status": "pass"},
                    ],
                )
            commits = resolve_complete_first_parent_range(vllm_repo, control, target)
            kern_hist = [
                {"sha": control, "role": "baseline_control", "status": "pass"},
                {"sha": target, "role": "target_head", "status": "fail"},
            ]
            if not commits:
                return _write_cpu_and_bisect(
                    cpu_status="fail",
                    stage="kernel_port_check",
                    target_sig=red_kern.target_signature,
                    is_public=False,
                    bisect_status="unresolved_history",
                    first_failing="",
                    last_good="",
                    steps_executed=2,
                    probe_steps_on_range=0,
                    history=kern_hist,
                )
            k_res = bisect_first_failing_sha(
                last_green_sha=control,
                red_sha=target,
                commits=commits,
                probe_fn=_probe_kernel_only,
                verify_baseline_first=False,
                initial_red_outcome=red_kern,
            )
            k_steps = int(k_res["steps_executed"]) + 2
            k_merged = kern_hist + list(k_res["history"])
            if k_res["bisect_status"] != "converged":
                return _write_cpu_and_bisect(
                    cpu_status="infra_error",
                    stage="wheel_build",
                    target_sig="VLLM_TARGET_DEVICE=empty",
                    is_public=False,
                    bisect_status=k_res["bisect_status"],
                    first_failing="",
                    last_good=control,
                    steps_executed=k_steps,
                    probe_steps_on_range=int(k_res["probe_steps_on_range"]),
                    history=k_merged,
                )
            k_win: ProbeOutcome = k_res["winning_outcome"]
            return _write_cpu_and_bisect(
                cpu_status="fail",
                stage="kernel_port_check",
                target_sig=k_win.target_signature,
                is_public=False,
                bisect_status="converged",
                first_failing=k_res["first_failing_sha"],
                last_good=k_res["last_good_sha"],
                steps_executed=k_steps,
                probe_steps_on_range=int(k_res["probe_steps_on_range"]),
                history=k_merged,
            )

        red_outcome = red_wheel
        base_hist = [
            {"sha": control, "role": "baseline_control", "status": "pass"},
            {"sha": target, "role": "target_head", "status": red_outcome.status},
        ]
        if red_outcome.status != "fail":
            return _write_cpu_and_bisect(
                cpu_status="infra_error",
                stage="wheel_build",
                target_sig="VLLM_TARGET_DEVICE=empty",
                is_public=False,
                bisect_status="aborted_infra_error",
                first_failing="",
                last_good=control,
                steps_executed=2,
                probe_steps_on_range=0,
                history=base_hist,
            )

        commits = resolve_complete_first_parent_range(vllm_repo, control, target)
        if not commits:
            return _write_cpu_and_bisect(
                cpu_status="fail",
                stage=red_outcome.stage,
                target_sig=red_outcome.target_signature,
                is_public=red_outcome.is_public_plugin_interface,
                bisect_status="unresolved_history",
                first_failing="",
                last_good="",
                steps_executed=2,
                probe_steps_on_range=0,
                history=base_hist,
            )

        # 2B. Match `target`'s observed wheel-build failure during bisection so earlier
        # non-matching failures (e.g., Tier-A drift or a different phase where wheels still built)
        # cannot shadow the actual target build break.
        def _matches_target_failure(cand: ProbeOutcome) -> bool:
            if cand.status != "fail" or cand.stage != red_outcome.stage:
                return False
            if (
                red_outcome.detail_code
                and cand.detail_code
                and cand.detail_code != red_outcome.detail_code
            ):
                return False
            if (
                red_outcome.target_signature
                and cand.target_signature
                and cand.target_signature != red_outcome.target_signature
            ):
                return False
            return True

        def _probe_matching_target(candidate_sha: str) -> ProbeOutcome:
            raw = _probe_wheel_only(candidate_sha)
            if raw.status == "infra_error":
                return raw
            if _matches_target_failure(raw):
                return raw
            return ProbeOutcome(status="pass", detail_code="pass")

        res = bisect_first_failing_sha(
            last_green_sha=control,
            red_sha=target,
            commits=commits,
            probe_fn=_probe_matching_target,
            verify_baseline_first=False,
            initial_red_outcome=red_outcome,
        )
        total_steps = int(res["steps_executed"]) + 2
        merged_hist = base_hist + list(res["history"])
        if res["bisect_status"] != "converged":
            return _write_cpu_and_bisect(
                cpu_status="infra_error",
                stage="wheel_build",
                target_sig=red_outcome.target_signature or "VLLM_TARGET_DEVICE=empty",
                is_public=False,
                bisect_status=res["bisect_status"],
                first_failing="",
                last_good=control,
                steps_executed=total_steps,
                probe_steps_on_range=int(res["probe_steps_on_range"]),
                history=merged_hist,
            )

        chosen_good = str(res["last_good_sha"])
        chosen_bad = str(res["first_failing_sha"])
        win: ProbeOutcome = res["winning_outcome"]

        # 3. Verify the target failure across `(last_good_sha, first_failing_sha)`.
        # Never substitute an earlier failure of a different phase or signature when the
        # target boundary cannot be verified.
        def _repro_matches_target_outcome(ev: NonPpuReproEvidence) -> bool:
            if not ev.verified or ev.infra_error:
                return False
            if (
                red_outcome.detail_code in ("build_install", "build_import")
                and ev.first_failing_phase != red_outcome.detail_code
            ):
                return False
            expected_sig = _normalize_wheel_build_signature(
                _sanitize_symbol(red_outcome.target_signature, "")
            )
            obs_sig = _normalize_wheel_build_signature(ev.observed_signature)
            if red_outcome.stage == "wheel_build":
                if not expected_sig or not obs_sig or obs_sig != expected_sig:
                    return False
            elif expected_sig and obs_sig and obs_sig != expected_sig:
                return False
            return True

        final_cpu_status = "fail"
        final_bisect_status = "converged"
        repro_dict: dict[str, Any] | None = None
        if win.is_public_plugin_interface or win.stage in (
            "wheel_build",
            "plugin_registration",
        ):
            repro_ev = verify_non_ppu_reproducer(
                vllm_repo=vllm_repo,
                last_good_sha=chosen_good,
                first_failing_sha=chosen_bad,
                stage=red_outcome.stage,
                target_signature=red_outcome.target_signature,
                probe_cmd=probe_cmd,
                python_exe=python_exe,
                timeout_sec=timeout_sec,
            )
            if (
                not _repro_matches_target_outcome(repro_ev)
                and not repro_ev.infra_error
                and chosen_bad != target
                and len(commits) >= 2
            ):
                parent_sha = commits[-2]
                parent_probe = _probe_wheel_only(parent_sha)
                total_steps += 1
                merged_hist.append(
                    {
                        "sha": parent_sha,
                        "role": "boundary_recheck",
                        "status": parent_probe.status,
                    }
                )
                if parent_probe.status == "pass":
                    alt_ev = verify_non_ppu_reproducer(
                        vllm_repo=vllm_repo,
                        last_good_sha=parent_sha,
                        first_failing_sha=target,
                        stage=red_outcome.stage,
                        target_signature=red_outcome.target_signature,
                        probe_cmd=probe_cmd,
                        python_exe=python_exe,
                        timeout_sec=timeout_sec,
                    )
                    if _repro_matches_target_outcome(alt_ev):
                        chosen_good = parent_sha
                        chosen_bad = target
                        win = red_outcome
                        repro_ev = alt_ev

            boundary_ok = _repro_matches_target_outcome(repro_ev)
            if not boundary_ok and repro_ev.verified:
                repro_ev = NonPpuReproEvidence(
                    mode=repro_ev.mode,
                    repro_kind=repro_ev.repro_kind,
                    last_good_sha=repro_ev.last_good_sha,
                    first_failing_sha=repro_ev.first_failing_sha,
                    last_good_exit_code=repro_ev.last_good_exit_code,
                    first_failing_exit_code=repro_ev.first_failing_exit_code,
                    verified=False,
                    last_good_phase=repro_ev.last_good_phase,
                    first_failing_phase=repro_ev.first_failing_phase,
                    last_good_check_executed=repro_ev.last_good_check_executed,
                    first_failing_check_executed=repro_ev.first_failing_check_executed,
                    infra_error=repro_ev.infra_error,
                    failure_source=repro_ev.failure_source,
                    observed_signature=repro_ev.observed_signature,
                )
            repro_dict = repro_ev.to_public_dict()
            _write_sanitized(repro_json_path, repro_dict)
            if repro_ev.infra_error:
                final_cpu_status = "infra_error"
                final_bisect_status = "aborted_infra_error"
            elif not boundary_ok:
                final_bisect_status = "unverified_target_boundary"
                win = red_outcome

        unconverged = final_bisect_status != "converged"
        out = _write_cpu_and_bisect(
            cpu_status=final_cpu_status,
            stage=red_outcome.stage if unconverged else win.stage,
            target_sig=(
                red_outcome.target_signature if unconverged else win.target_signature
            ),
            is_public=(
                red_outcome.is_public_plugin_interface
                if unconverged
                else win.is_public_plugin_interface
            ),
            bisect_status=final_bisect_status,
            first_failing="" if unconverged else chosen_bad,
            last_good=control if unconverged else chosen_good,
            steps_executed=total_steps,
            probe_steps_on_range=int(res["probe_steps_on_range"]),
            history=merged_hist,
        )
        if repro_dict is not None:
            out["non_ppu_repro"] = repro_dict
        return out


# ---------------------------------------------------------------------------
# 7. CLI Entrypoint
# ---------------------------------------------------------------------------


def _load_cursor_file(path: Path | None) -> dict[str, Any] | None:
    """Load an optional cursor JSON file, rejecting corrupt non-object payloads."""
    if path is None or not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"cursor file {path} must contain a JSON object, got {type(raw).__name__}"
        )
    return raw


def _write_github_outputs(path: Path | None, values: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, raw in values.items():
        val = "true" if raw is True else ("false" if raw is False else str(raw))
        if "\n" in val or "\r" in val:
            raise ValueError(f"multiline GitHub output forbidden for {key}")
        lines.append(f"{key}={val}")
    block = "\n".join(lines) + ("\n" if lines else "")
    staging = [path] if path is not None else ()
    assert_zero_sensitive_leaks(block, staging_paths=staging)
    if path is not None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(block)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("route")
    p_route.add_argument("--upstream-url", default=DEFAULT_UPSTREAM_URL)
    p_route.add_argument("--ls-remote-file", type=Path, default=None)
    p_route.add_argument(
        "--verified-pin-file", type=Path, default=DEFAULT_VERIFIED_PIN_FILE
    )
    p_route.add_argument(
        "--release-tag-file", type=Path, default=DEFAULT_RELEASE_TAG_FILE
    )
    p_route.add_argument("--gear", choices=VALID_GEARS, default="auto")
    p_route.add_argument("--schedule-cron", default="")
    p_route.add_argument("--upstream-ref", default="")
    p_route.add_argument("--last-green-sha", default="")
    p_route.add_argument("--cursor-file", type=Path, default=None)
    p_route.add_argument("--changed-files-file", type=Path, default=None)
    p_route.add_argument("--changed-files-base-sha", default="")
    p_route.add_argument("--probe-history-file", type=Path, default=None)
    p_route.add_argument("--active-pr-runs", type=int, default=0)
    p_route.add_argument("--vllm-repo", type=Path, default=None)
    p_route.add_argument("--force", action="store_true")
    p_route.add_argument("--github-output", type=Path, default=None)

    p_cpu = sub.add_parser("cpu-check")
    p_cpu.add_argument("--vllm-src", type=Path, required=True)
    p_cpu.add_argument("--sail-root", type=Path, default=REPO_ROOT)
    p_cpu.add_argument("--target-sha", required=True)
    p_cpu.add_argument(
        "--verified-pin-file", type=Path, default=DEFAULT_VERIFIED_PIN_FILE
    )
    p_cpu.add_argument("--previous-baseline", type=Path, default=None)
    p_cpu.add_argument("--output-json", type=Path, required=True)

    p_sbt = sub.add_parser("source-build-triage")
    p_sbt.add_argument("--vllm-repo", type=Path, required=True)
    p_sbt.add_argument("--sail-root", type=Path, default=REPO_ROOT)
    p_sbt.add_argument("--target-sha", required=True)
    p_sbt.add_argument("--last-green-sha", default="")
    p_sbt.add_argument("--fallback-green-shas", default="")
    p_sbt.add_argument("--cursor-file", type=Path, default=None)
    p_sbt.add_argument("--rc-tag", default="")
    p_sbt.add_argument(
        "--verified-pin-file", type=Path, default=DEFAULT_VERIFIED_PIN_FILE
    )
    p_sbt.add_argument("--probe-cmd", default="")
    p_sbt.add_argument("--timeout-sec", type=int, default=900)
    p_sbt.add_argument("--output-dir", type=Path, required=True)

    p_bisect = sub.add_parser("bisect")
    p_bisect.add_argument("--vllm-repo", type=Path, required=True)
    p_bisect.add_argument("--last-green-sha", required=True)
    p_bisect.add_argument("--fallback-green-shas", default="")
    p_bisect.add_argument("--cursor-file", type=Path, default=None)
    p_bisect.add_argument("--verified-pin-file", type=Path, default=None)
    p_bisect.add_argument("--rc-tag", default="")
    p_bisect.add_argument("--red-sha", required=True)
    p_bisect.add_argument("--probe-cmd", required=True)
    p_bisect.add_argument("--initial-red-json", type=Path, default=None)
    p_bisect.add_argument("--default-stage", default="ppu_probe_smoke")
    p_bisect.add_argument("--output-json", type=Path, required=True)
    p_bisect.add_argument("--github-output", type=Path, default=None)

    p_bump = sub.add_parser("prepare-pin-bump")
    p_bump.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p_bump.add_argument("--target-sha", default="")
    p_bump.add_argument("--vllm-repo", type=Path, default=None)
    p_bump.add_argument("--cursor-file", type=Path, default=None)
    p_bump.add_argument("--ls-remote-file", type=Path, default=None)
    p_bump.add_argument("--allow-unverified-target", action="store_true")
    p_bump.add_argument("--output-json", type=Path, default=None)
    p_bump.add_argument("--github-output", type=Path, default=None)

    p_resolve = sub.add_parser("resolve-outcome")
    p_resolve.add_argument("--gear", choices=SELECTED_GEARS, required=True)
    p_resolve.add_argument("--target-sha", default="")
    p_resolve.add_argument("--run-cpu", default="false")
    p_resolve.add_argument("--run-ppu", default="false")
    p_resolve.add_argument("--ppu-skip-reason", default="none")
    p_resolve.add_argument("--ops-image", default="")
    p_resolve.add_argument("--route-result", default="success")
    p_resolve.add_argument("--build-result", default="skipped")
    p_resolve.add_argument("--cpu-result", default="skipped")
    p_resolve.add_argument("--ppu-smoke-result", default="skipped")
    p_resolve.add_argument("--rc-pin-result", default="skipped")
    p_resolve.add_argument("--rc-fast-result", default="skipped")
    p_resolve.add_argument("--rc-model-result", default="skipped")
    p_resolve.add_argument("--rc-ops-select-result", default="skipped")
    p_resolve.add_argument("--rc-ops-result", default="skipped")
    p_resolve.add_argument("--bump-result", default="skipped")
    p_resolve.add_argument("--github-output", type=Path, default=None)

    p_repro = sub.add_parser("verify-non-ppu-repro")
    p_repro.add_argument("--vllm-repo", type=Path, required=True)
    p_repro.add_argument("--last-good-sha", required=True)
    p_repro.add_argument("--first-failing-sha", required=True)
    p_repro.add_argument("--stage", default="plugin_registration")
    p_repro.add_argument("--target-signature", default="")
    p_repro.add_argument("--repro-kind", default="")
    p_repro.add_argument("--probe-cmd", default="")
    p_repro.add_argument("--timeout-sec", type=int, default=900)
    p_repro.add_argument("--output-json", type=Path, required=True)
    p_repro.add_argument("--github-output", type=Path, default=None)

    p_pub = sub.add_parser("publish")
    p_pub.add_argument("--gear", choices=SELECTED_GEARS, required=True)
    p_pub.add_argument("--status", choices=VALID_STATUSES, required=True)
    p_pub.add_argument("--sail-sha", required=True)
    p_pub.add_argument("--upstream-sha", default="")
    p_pub.add_argument(
        "--verified-pin-file", type=Path, default=DEFAULT_VERIFIED_PIN_FILE
    )
    p_pub.add_argument("--build-image", default="")
    p_pub.add_argument("--runtime-image", default="")
    p_pub.add_argument("--rc-tag", default="")
    p_pub.add_argument("--cpu-status", default="not_run")
    p_pub.add_argument("--ppu-status", default="not_run")
    p_pub.add_argument("--ppu-skip-reason", default="none")
    p_pub.add_argument("--first-failing-sha", default="")
    p_pub.add_argument("--last-good-sha", default="")
    p_pub.add_argument("--bisect-steps", type=int, default=0)
    p_pub.add_argument("--failure-stage", default="none")
    p_pub.add_argument("--target-signature", default="")
    p_pub.add_argument("--public-plugin-break", action="store_true")
    p_pub.add_argument("--known-soft-drift-count", type=int, default=0)
    p_pub.add_argument("--cpu-result-json", type=Path, default=None)
    p_pub.add_argument("--bisect-result-json", type=Path, default=None)
    p_pub.add_argument("--non-ppu-repro-json", type=Path, default=None)
    p_pub.add_argument("--first-failing-author", default="")
    p_pub.add_argument("--upstream-commit-json", type=Path, default=None)
    p_pub.add_argument("--fetch-upstream-author", action="store_true")
    p_pub.add_argument("--run-url", default="")
    p_pub.add_argument("--cursor-in", type=Path, default=None)
    p_pub.add_argument("--superseded-probe-frontier", action="store_true")
    p_pub.add_argument("--public-dir", type=Path, required=True)
    p_pub.add_argument("--step-summary", type=Path, default=None)

    p_bman = sub.add_parser("sanitize-build-manifest")
    p_bman.add_argument("--manifest", type=Path, default=None)
    p_bman.add_argument("--input", dest="input_path", type=Path, default=None)
    p_bman.add_argument("--output", dest="output_path", type=Path, default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "sanitize-build-manifest":
        src = args.input_path or args.manifest
        dst = args.output_path or args.manifest
        if src is None or dst is None:
            raise ValueError(
                "sanitize-build-manifest requires --manifest or both --input and --output"
            )
        sanitize_build_manifest_file(src, output_path=dst)
        return 0

    if args.command == "route":
        if args.ls_remote_file is not None:
            raw_ls = args.ls_remote_file.read_text(encoding="utf-8")
        else:
            raw_ls = subprocess.check_output(
                [
                    "git",
                    "ls-remote",
                    args.upstream_url,
                    "refs/heads/main",
                    "refs/heads/releases/v*",
                    "refs/tags/v*",
                ],
                text=True,
                timeout=60,
            )
        ls_data = parse_ls_remote(raw_ls)
        verified_pin = args.verified_pin_file.read_text(encoding="utf-8").strip()
        release_tag = args.release_tag_file.read_text(encoding="utf-8").strip()
        cursor = _load_cursor_file(args.cursor_file)
        changed = (
            parse_changed_paths_lines(
                args.changed_files_file.read_text(encoding="utf-8").splitlines()
            )
            if args.changed_files_file and args.changed_files_file.is_file()
            else None
        )
        history = (
            json.loads(args.probe_history_file.read_text(encoding="utf-8"))
            if args.probe_history_file and args.probe_history_file.is_file()
            else ()
        )
        decision = route_three_gears(
            ls_remote_data=ls_data,
            verified_pin=verified_pin,
            release_baseline_tag=release_tag,
            requested_gear=args.gear,
            schedule_cron=args.schedule_cron,
            upstream_ref_override=args.upstream_ref,
            last_green_sha_override=args.last_green_sha,
            cursor=cursor,
            changed_paths_since_ppu=changed,
            changed_files_base_sha=args.changed_files_base_sha,
            active_pr_runs=args.active_pr_runs,
            probe_history=history,
            force=args.force,
            vllm_repo=args.vllm_repo,
        )
        decision_text = json.dumps(decision, indent=2, sort_keys=True)
        assert_zero_sensitive_leaks(decision_text)
        _write_github_outputs(args.github_output, decision)
        print(decision_text)
        return 0

    if args.command == "cpu-check":
        verified_pin = args.verified_pin_file.read_text(encoding="utf-8").strip()
        return run_sdk_cpu_fast_check(
            vllm_src=args.vllm_src,
            sail_root=args.sail_root,
            target_sha=args.target_sha,
            verified_pin=verified_pin,
            previous_baseline=args.previous_baseline,
            output_json=args.output_json,
        )

    if args.command == "source-build-triage":
        verified_pin = args.verified_pin_file.read_text(encoding="utf-8").strip()
        sbt_cursor = _load_cursor_file(args.cursor_file)
        sbt_fallbacks = [
            s.strip()
            for s in args.fallback_green_shas.split(",")
            if SHA1_RE.fullmatch(s.strip())
        ]
        res = run_source_only_build_triage(
            vllm_repo=args.vllm_repo,
            sail_root=args.sail_root,
            target_sha=args.target_sha,
            verified_pin=verified_pin,
            last_green_sha=args.last_green_sha,
            fallback_green_shas=sbt_fallbacks,
            cursor=sbt_cursor,
            rc_tag=args.rc_tag,
            output_dir=args.output_dir,
            probe_cmd=args.probe_cmd,
            timeout_sec=args.timeout_sec,
        )
        print(json.dumps(res["bisect"], indent=2, sort_keys=True))
        return 0

    if args.command == "bisect":
        green = (
            _require_sha1(args.last_green_sha, "last_green_sha")
            if args.last_green_sha.strip()
            else ""
        )
        red = _require_sha1(args.red_sha, "red_sha")
        initial_red = None
        if args.initial_red_json and args.initial_red_json.is_file():
            init_data = json.loads(args.initial_red_json.read_text(encoding="utf-8"))
            initial_red = ProbeOutcome(
                status="infra_error"
                if init_data.get("status") == "infra_error"
                else "fail",
                stage=str(init_data.get("stage") or args.default_stage),
                target_signature=str(
                    init_data.get("target_signature") or "window_head"
                ),
                is_public_plugin_interface=bool(
                    init_data.get("is_public_plugin_interface")
                ),
            )

        bisect_fallbacks: list[str] = [
            s.strip()
            for s in args.fallback_green_shas.split(",")
            if SHA1_RE.fullmatch(s.strip())
        ]
        bisect_excluded: set[str] = set()
        bisect_pin = ""
        if args.verified_pin_file and args.verified_pin_file.is_file():
            bisect_pin = args.verified_pin_file.read_text(encoding="utf-8").strip()
        elif (
            args.cursor_file
            and args.cursor_file.is_file()
            and DEFAULT_VERIFIED_PIN_FILE.is_file()
        ):
            bisect_pin = DEFAULT_VERIFIED_PIN_FILE.read_text(encoding="utf-8").strip()
        raw_cur = _load_cursor_file(args.cursor_file)
        if raw_cur is not None:
            cur_pin = (
                bisect_pin
                if SHA1_RE.fullmatch(bisect_pin)
                else str(raw_cur.get("last_ppu_green_sha") or green or red)
            )
            clean_cur = _sanitize_cursor(raw_cur, verified_pin=cur_pin)
            bisect_excluded = _collect_excluded_control_shas(
                clean_cur,
                rc_ref=args.rc_tag,
                target_sha=red,
            )
            for cand_s in _collect_verified_control_shas(
                clean_cur,
                verified_pin=bisect_pin,
                rc_ref=args.rc_tag,
                target_sha=red,
            ):
                if cand_s not in bisect_fallbacks:
                    bisect_fallbacks.append(cand_s)
        elif SHA1_RE.fullmatch(bisect_pin) and bisect_pin not in bisect_fallbacks:
            bisect_fallbacks.append(bisect_pin)
        if (
            not green
            and not bisect_fallbacks
            and raw_cur is None
            and not SHA1_RE.fullmatch(bisect_pin)
        ):
            raise ValueError("last_green_sha must be a 40-character lowercase hex SHA")

        green, commits = (
            resolve_verified_bisect_control(
                args.vllm_repo,
                primary_green=green,
                red_sha=red,
                fallback_greens=bisect_fallbacks,
                excluded_shas=bisect_excluded,
            )
            if (green or bisect_fallbacks)
            else ("", None)
        )
        if not commits:
            fallback_outcome = initial_red or ProbeOutcome(
                status="fail",
                stage=args.default_stage,
                target_signature="window_head",
            )
            serializable = {
                "bisect_status": "unresolved_history",
                "first_failing_sha": "",
                "last_good_sha": green,
                "steps_executed": 0,
                "probe_steps_on_range": 0,
                "stage": fallback_outcome.stage
                if fallback_outcome.stage in VALID_STAGES
                else args.default_stage,
                "target_signature": _sanitize_symbol(
                    fallback_outcome.target_signature, "unspecified"
                ),
                "is_public_plugin_interface": bool(
                    fallback_outcome.is_public_plugin_interface
                ),
                "history": [],
            }
            text = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            assert_zero_sensitive_leaks(text, staging_paths=[args.output_json])
            args.output_json.write_text(text, encoding="utf-8")
            _write_github_outputs(
                args.github_output,
                {
                    "bisect_status": serializable["bisect_status"],
                    "first_failing_sha": serializable["first_failing_sha"],
                    "last_good_sha": serializable["last_good_sha"],
                    "steps_executed": serializable["steps_executed"],
                    "stage": serializable["stage"],
                    "target_signature": serializable["target_signature"],
                    "is_public_plugin_interface": serializable[
                        "is_public_plugin_interface"
                    ],
                },
            )
            print(text, end="")
            return 0

        def _shell_probe(candidate_sha: str) -> ProbeOutcome:
            probe_json = args.output_json.parent / f"probe-{candidate_sha[:12]}.json"
            probe_json.unlink(missing_ok=True)
            child_env = dict(os.environ)
            child_env["CANDIDATE_SHA"] = candidate_sha
            child_env["PROBE_RESULT_JSON"] = str(probe_json)
            proc = subprocess.run(
                ["bash", "-c", args.probe_cmd],
                env=child_env,
                capture_output=True,
                text=True,
            )
            if probe_json.is_file():
                data = json.loads(probe_json.read_text(encoding="utf-8"))
                probe_json.unlink(missing_ok=True)
                raw_status = str(
                    data.get("status") or ("pass" if proc.returncode == 0 else "fail")
                )
                return ProbeOutcome(
                    status=raw_status
                    if raw_status in ("pass", "fail", "infra_error")
                    else "fail",
                    stage=str(data.get("stage") or args.default_stage),
                    target_signature=str(
                        data.get("target_signature") or "bisect_probe"
                    ),
                    is_public_plugin_interface=bool(
                        data.get("is_public_plugin_interface")
                    ),
                )
            if proc.returncode == 0:
                return ProbeOutcome(status="pass")
            return ProbeOutcome(
                status="infra_error" if proc.returncode == 2 else "fail",
                stage=args.default_stage,
                target_signature="bisect_probe",
            )

        res = bisect_first_failing_sha(
            last_green_sha=green,
            red_sha=red,
            commits=commits,
            probe_fn=_shell_probe,
            verify_baseline_first=True,
            initial_red_outcome=initial_red,
        )
        win: ProbeOutcome = res["winning_outcome"]
        serializable = {
            "bisect_status": res["bisect_status"],
            "first_failing_sha": res["first_failing_sha"],
            "last_good_sha": res["last_good_sha"],
            "steps_executed": res["steps_executed"],
            "probe_steps_on_range": res["probe_steps_on_range"],
            "stage": win.stage if win.stage in VALID_STAGES else args.default_stage,
            "target_signature": _sanitize_symbol(win.target_signature, "unspecified"),
            "is_public_plugin_interface": bool(win.is_public_plugin_interface),
            "history": res["history"],
        }
        text = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        assert_zero_sensitive_leaks(text, staging_paths=[args.output_json])
        args.output_json.write_text(text, encoding="utf-8")
        _write_github_outputs(
            args.github_output,
            {
                "bisect_status": serializable["bisect_status"],
                "first_failing_sha": serializable["first_failing_sha"],
                "last_good_sha": serializable["last_good_sha"],
                "steps_executed": serializable["steps_executed"],
                "stage": serializable["stage"],
                "target_signature": serializable["target_signature"],
                "is_public_plugin_interface": serializable[
                    "is_public_plugin_interface"
                ],
            },
        )
        print(text, end="")
        return 0

    if args.command == "prepare-pin-bump":
        bump_cursor = _load_cursor_file(args.cursor_file)
        bump_ls = (
            parse_ls_remote(args.ls_remote_file.read_text(encoding="utf-8"))
            if args.ls_remote_file and args.ls_remote_file.is_file()
            else None
        )
        res = prepare_pin_bump(
            repo_root=args.repo_root,
            target_sha=args.target_sha,
            vllm_repo=args.vllm_repo,
            cursor=bump_cursor,
            ls_remote_data=bump_ls,
            require_cursor=not args.allow_unverified_target,
        )
        res_text = json.dumps(res, indent=2, sort_keys=True) + "\n"
        staging = [args.output_json] if args.output_json is not None else ()
        assert_zero_sensitive_leaks(res_text, staging_paths=staging)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(res_text, encoding="utf-8")
        _write_github_outputs(
            args.github_output,
            {
                "changed": res["changed"],
                "previous_pin": res["previous_pin"],
                "new_pin": res["new_pin"],
                "title": res["title"],
            },
        )
        print(res_text, end="")
        return 0

    if args.command == "resolve-outcome":
        resolved = evaluate_workflow_outcome(
            selected_gear=args.gear,
            target_sha=args.target_sha.strip(),
            run_cpu=args.run_cpu.strip().lower() == "true",
            run_ppu=args.run_ppu.strip().lower() == "true",
            ppu_skip_reason=args.ppu_skip_reason.strip() or "none",
            ops_image=args.ops_image.strip(),
            route_result=args.route_result.strip(),
            build_result=args.build_result.strip(),
            cpu_result=args.cpu_result.strip(),
            ppu_smoke_result=args.ppu_smoke_result.strip(),
            rc_pin_result=args.rc_pin_result.strip(),
            rc_fast_result=args.rc_fast_result.strip(),
            rc_model_result=args.rc_model_result.strip(),
            rc_ops_select_result=args.rc_ops_select_result.strip(),
            rc_ops_result=args.rc_ops_result.strip(),
            bump_result=args.bump_result.strip(),
        )
        resolved_text = json.dumps(resolved, indent=2, sort_keys=True)
        assert_zero_sensitive_leaks(resolved_text)
        _write_github_outputs(args.github_output, resolved)
        print(resolved_text)
        return 0

    if args.command == "verify-non-ppu-repro":
        evidence = verify_non_ppu_reproducer(
            vllm_repo=args.vllm_repo,
            last_good_sha=args.last_good_sha,
            first_failing_sha=args.first_failing_sha,
            stage=args.stage,
            target_signature=args.target_signature,
            repro_kind=args.repro_kind,
            probe_cmd=args.probe_cmd,
            timeout_sec=args.timeout_sec,
        )
        payload = evidence.to_public_dict()
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        assert_zero_sensitive_leaks(text, staging_paths=[args.output_json])
        args.output_json.write_text(text, encoding="utf-8")
        _write_github_outputs(args.github_output, payload)
        print(text, end="")
        return 0

    if args.command == "publish":
        verified_pin = args.verified_pin_file.read_text(encoding="utf-8").strip()
        cpu_meta = (
            json.loads(args.cpu_result_json.read_text(encoding="utf-8"))
            if args.cpu_result_json and args.cpu_result_json.is_file()
            else {}
        )
        bisect_meta = (
            json.loads(args.bisect_result_json.read_text(encoding="utf-8"))
            if args.bisect_result_json and args.bisect_result_json.is_file()
            else {}
        )
        non_ppu_repro: NonPpuReproEvidence | None = None
        if args.non_ppu_repro_json and args.non_ppu_repro_json.is_file():
            try:
                raw_repro = json.loads(
                    args.non_ppu_repro_json.read_text(encoding="utf-8")
                )
                non_ppu_repro = parse_non_ppu_repro_evidence(raw_repro)
            except (OSError, ValueError):
                non_ppu_repro = None

        effective_status = args.status
        if (
            cpu_meta.get("status") == "infra_error"
            or bisect_meta.get("bisect_status")
            in (
                "baseline_control_failed",
                "aborted_infra_error",
            )
            or (
                args.failure_stage == "wheel_build"
                and not cpu_meta
                and not bisect_meta
                and non_ppu_repro is None
                and not args.first_failing_sha
            )
        ):
            effective_status = "infra_error"
        effective_cpu_status = args.cpu_status
        if effective_cpu_status == "not_run" and cpu_meta.get("status") in (
            "pass",
            "fail",
            "infra_error",
        ):
            effective_cpu_status = str(cpu_meta["status"])
        measured_stage = str(
            bisect_meta.get("stage") or cpu_meta.get("stage") or ""
        ).strip()
        if measured_stage in VALID_STAGES and measured_stage != "none":
            failure_stage = measured_stage
        elif args.failure_stage in VALID_STAGES:
            failure_stage = args.failure_stage
        else:
            failure_stage = "none"
        target_sig = args.target_signature or str(
            bisect_meta.get("target_signature")
            or cpu_meta.get("target_signature")
            or ""
        )
        public_break = bool(
            args.public_plugin_break
            or bisect_meta.get("is_public_plugin_interface")
            or cpu_meta.get("is_public_plugin_interface")
        )
        first_bad = args.first_failing_sha or str(
            bisect_meta.get("first_failing_sha") or ""
        )
        last_good = args.last_good_sha or str(bisect_meta.get("last_good_sha") or "")
        bisect_steps = args.bisect_steps or int(bisect_meta.get("steps_executed") or 0)
        soft_drift = args.known_soft_drift_count or int(
            cpu_meta.get("known_soft_drift_count") or 0
        )
        patch_fps = cpu_meta.get("patch_fingerprints")
        if not isinstance(patch_fps, dict):
            patch_fps = None

        outcome = None
        if effective_status in ("fail", "infra_error"):
            outcome = ProbeOutcome(
                status="infra_error" if effective_status == "infra_error" else "fail",
                stage=failure_stage,
                target_signature=target_sig,
                is_public_plugin_interface=public_break,
                non_ppu_repro=non_ppu_repro,
            )

        resolved_author = sanitize_github_login(args.first_failing_author)
        if (
            not resolved_author
            and outcome is not None
            and effective_status == "fail"
            and SHA1_RE.fullmatch(first_bad)
        ):
            eff_good = (
                last_good
                if (
                    SHA1_RE.fullmatch(last_good)
                    and last_good != args.upstream_sha.strip()
                    and last_good != first_bad
                )
                else ""
            )
            pre_triage = classify_failure(
                outcome,
                non_ppu_repro=non_ppu_repro,
                last_good_sha=eff_good,
                first_failing_sha=first_bad,
                require_shas=True,
            )
            if pre_triage["classification"] == "plugin_interface_break":
                resolved_author = resolve_upstream_commit_author(
                    first_bad,
                    commit_metadata_path=args.upstream_commit_json,
                    fetch_remote=args.fetch_upstream_author,
                )

        prev_cursor = _load_cursor_file(args.cursor_in)
        effective_rc_tag = args.rc_tag
        if (
            args.gear == "rc_qualify"
            and effective_status != "noop"
            and not _normalize_rc_ref(effective_rc_tag)
            and SHA1_RE.fullmatch(args.upstream_sha.strip())
            and prev_cursor is not None
        ):
            clean_prev = _sanitize_cursor(prev_cursor, verified_pin=verified_pin)
            effective_rc_tag = _resolve_unique_rc_ref_for_sha(
                args.upstream_sha.strip(),
                tags={},
                release_branches={},
                release_baseline_tag="",
                rc_branch_states=clean_prev["rc_branch_states"],
                rc_tag_states=clean_prev["rc_tag_states"],
                rc_line_revoked_shas=clean_prev.get("rc_line_revoked_shas"),
                tested_rc_tags=clean_prev["tested_rc_tags"],
            )

        published = publish_sanitized_channels(
            gear=args.gear,
            status=effective_status,
            sail_sha=args.sail_sha,
            upstream_sha=args.upstream_sha,
            verified_pin=verified_pin,
            build_image=args.build_image,
            runtime_image=args.runtime_image,
            rc_tag=effective_rc_tag,
            cpu_status=effective_cpu_status,
            ppu_status=args.ppu_status,
            ppu_skip_reason=args.ppu_skip_reason,
            first_failing_sha=first_bad,
            last_good_sha=last_good,
            bisect_steps=bisect_steps,
            outcome=outcome,
            non_ppu_repro=non_ppu_repro,
            first_failing_author=resolved_author,
            known_soft_drift_count=soft_drift,
            run_url=args.run_url,
            public_dir=args.public_dir,
            step_summary_path=args.step_summary,
        )
        cursor_path = args.public_dir / "cursor.json"
        if (
            args.gear in ("probe", "rc_qualify")
            and effective_status != "noop"
            and SHA1_RE.fullmatch(args.upstream_sha.strip())
        ):
            next_cursor = advance_cursor(
                prev_cursor,
                verified_pin=verified_pin,
                selected_gear=args.gear,
                target_sha=args.upstream_sha,
                rc_tag=effective_rc_tag,
                status=effective_status,
                cpu_status=effective_cpu_status,
                ppu_status=args.ppu_status,
                ppu_skip_reason=args.ppu_skip_reason,
                patch_fingerprints=patch_fps,
                superseded_probe_frontier=args.superseded_probe_frontier,
            )
            cursor_text = json.dumps(next_cursor, indent=2, sort_keys=True) + "\n"
            assert_zero_sensitive_leaks(cursor_text, staging_paths=[cursor_path])
            cursor_path.write_text(cursor_text, encoding="utf-8")
        else:
            cursor_path.unlink(missing_ok=True)
        print(published["console"])
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
