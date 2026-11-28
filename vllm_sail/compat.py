# SPDX-License-Identifier: Apache-2.0
"""vLLM compatibility check for the PPU plugin.

Compatibility with upstream vLLM is verified through source-contract, API, and
functional tests rather than hardcoded version ranges. This module provides a
lightweight runtime check against the single source of truth recorded in the
packaged kernel manifest (``csrc/upstream/manifest.toml``'s ``upstream_ref``):

* **Default (warn):** log a warning when the installed vLLM build does not
  match ``upstream_ref`` so developers can test against neighboring commits or
  releases without being blocked by a version gate, while still flagging
  mismatched environments prominently in startup logs.
* **Strict (CI / release verification):** pass ``strict=True`` or set
  ``VLLM_SAIL_STRICT_COMPAT=1`` to raise :class:`RuntimeError` on mismatch.
"""

from __future__ import annotations

import os
import re

from vllm.logger import init_logger

from vllm_sail import envs

logger = init_logger(__name__)

_checked = False
_last_matched = False


def _upstream_ref() -> str:
    """Read the verified upstream ref from the packaged kernel metadata."""
    from vllm_sail.native.manifest import load

    return load().upstream_ref.strip()


def _is_commit_sha(ref: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", ref))


def _candidate_upstream_commit() -> str | None:
    """Read an explicit 40-hex candidate upstream commit override, if set."""
    try:
        return envs.VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def installed_vllm_version() -> str:
    """Read the version reported by the installed vLLM package."""
    try:
        from vllm.version import __version__

        return __version__
    except Exception:  # pragma: no cover - vllm always has a version in practice
        return "unknown"


def _release_line(version: str) -> str | None:
    """Read the major and minor components from a release or SCM version."""
    match = re.match(r"v?([0-9]+)\.([0-9]+)", version.strip())
    if match is None:
        return None
    major, minor = match.groups()
    return f"{major.lstrip('0') or '0'}.{minor.lstrip('0') or '0'}"


def _installed_vllm_commit(version: str) -> str | None:
    """Extract the verified 40-hex commit from ``version`` or ``vllm._version``."""
    match = re.search(r"(?:\+|\.)g([0-9a-f]{40})(?=\.|$)", version)
    if match is not None:
        return match.group(1)
    if re.search(r"(?:\+|\.)g[0-9a-f]+", version) is not None:
        return None
    for mod_name in ("vllm._version", "vllm.version"):
        try:
            mod = __import__(mod_name, fromlist=["__commit_id__"])
            raw = getattr(mod, "__commit_id__", None)
        except Exception:
            continue
        if isinstance(raw, str):
            cid = re.fullmatch(r"g?([0-9a-f]{40})(?:\.d\d+)?", raw.strip())
            if cid is not None:
                return cid.group(1)
    return None


def _mismatch_reason(version: str, ref: str) -> str | None:
    """Return a human-readable mismatch message, or ``None`` when compatible."""
    if _is_commit_sha(ref):
        # Main and release tags may live on different branches. SCM's numeric
        # prefix then describes an older tag, while its full hash still names
        # the source precisely. An abbreviation cannot establish this identity.
        if _installed_vllm_commit(version) == ref:
            return None
        return (
            f"vllm-sail {_plugin_version()} was built against verified vLLM commit "
            f"{ref}, but the installed vLLM version is {version!r}. Build that "
            "exact source with the full SCM commit hash."
        )

    release_line = _release_line(ref)
    if release_line is not None:
        if _release_line(version) == release_line:
            return None
        return (
            f"vllm-sail {_plugin_version()} was built against vLLM {ref} "
            f"(expected {release_line}.x), but the installed vLLM version is "
            f"{version!r}. Install a matching vLLM build or set VLLM_VERSION if "
            "running from a source checkout."
        )

    if version.strip().lstrip("v") == ref.lstrip("v"):
        return None
    return (
        f"vllm-sail {_plugin_version()} was built against vLLM ref {ref!r}, "
        f"but the installed vLLM version is {version!r}."
    )


def check_vllm_compatibility(
    *, force: bool = False, strict: bool | None = None
) -> None:
    """Validate the installed vLLM against the packaged manifest's ``upstream_ref``.

    Called once from :func:`vllm_sail.register_out_of_tree`. By default, a
    mismatch logs a warning and allows startup to proceed; set
    ``VLLM_SAIL_STRICT_COMPAT=1`` or pass ``strict=True`` (used in CI) to raise
    :class:`RuntimeError` instead. When ``VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT``
    is set, candidate verification always fails closed against that exact 40-hex
    commit SHA without mutating the packaged manifest.

    Raises:
        RuntimeError: Strict or candidate mode is enabled and the installed vLLM
            build does not match the required upstream ref.
    """
    global _checked, _last_matched
    candidate_active = envs.is_set("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT")
    effective_strict = (
        True
        if candidate_active
        else (envs.VLLM_SAIL_STRICT_COMPAT if strict is None else strict)
    )
    if _checked and not force and (_last_matched or not effective_strict):
        return

    # Leave the cache invalid if strict validation raises and the caller catches it.
    _checked = False
    _last_matched = False

    ref = _upstream_ref()
    candidate = _candidate_upstream_commit()
    if candidate is not None:
        raw_version = installed_vllm_version()
        has_version_override = (
            os.getenv("VLLM_VERSION") is not None
            or os.getenv("VLLM_VERSION_OVERRIDE") is not None
        )
        commit = None if has_version_override else _installed_vllm_commit(raw_version)
        if commit != candidate:
            shown_version = (
                os.getenv("VLLM_VERSION")
                or os.getenv("VLLM_VERSION_OVERRIDE")
                or raw_version
            )
            raise RuntimeError(
                f"vllm-sail {_plugin_version()} candidate mode requires exact "
                f"vLLM commit {candidate}, but the installed version is "
                f"{shown_version!r} (VLLM_VERSION / VLLM_VERSION_OVERRIDE must "
                "be unset). Build that exact candidate source with the full "
                "SCM commit hash and unset VLLM_VERSION and "
                "VLLM_VERSION_OVERRIDE."
            )
        _checked = True
        _last_matched = True
        logger.warning(
            "vllm-sail %s is running in candidate mode against upstream vLLM "
            "commit %s (verified manifest ref remains %s).",
            _plugin_version(),
            candidate,
            ref,
        )
        return
    # Only release-line checks accept a developer-supplied version. Commit
    # checks must read the identity recorded in the installed build.
    override = (
        (os.getenv("VLLM_VERSION") or "").strip()
        if _release_line(ref) is not None
        else ""
    )
    version = override or installed_vllm_version()
    mismatch = _mismatch_reason(version, ref)
    if mismatch is None:
        _checked = True
        _last_matched = True
        logger.debug(
            "vllm-sail %s validated against vLLM %s (%s)",
            _plugin_version(),
            version,
            ref,
        )
        return

    if effective_strict:
        raise RuntimeError(mismatch)

    logger.warning("%s", mismatch)
    _checked = True
    _last_matched = False


def _plugin_version() -> str:
    from vllm_sail.version import __version__

    return __version__
