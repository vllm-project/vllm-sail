# SPDX-License-Identifier: Apache-2.0
"""Release handoff tests use real wheel archives without SDK/device dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/release_wheels.py"
spec = importlib.util.spec_from_file_location("release_wheels", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
CONFIG = dict(
    zip(release.AXES, ("312", "ubuntu2404", "3", "2.13.0", "2.2.0"), strict=False)
)
COMMIT = "a" * 40
TAG = "v0.31.0rc3.dev1"


@pytest.fixture
def assets(tmp_path, monkeypatch):
    wheels = tmp_path / "wheels"
    for name in ("vllm", "vllm_sail"):
        target = wheels / name
        target.mkdir(parents=True)
        wheel = target / f"{name}-0.31.0rc3.dev1-cp312-cp312-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                f"{name}-0.31.0rc3.dev1.dist-info/METADATA",
                f"Name: {name}\nVersion: 0.31.0rc3.dev1\n",
            )
            if name == "vllm":
                archive.writestr("vllm/vllm-rs", b"executable")
                archive.writestr("vllm/_rust_tool_parser.abi3.so", b"parser")
            else:
                for module in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C"):
                    archive.writestr(f"vllm_sail/{module}.abi3.so", b"extension")
    records = {}
    for name in ("vllm", "vllm-sail"):
        wheel = next(wheels.rglob(name.replace("-", "_") + "-*.whl"))
        records[name] = {
            "file": str(wheel.relative_to(wheels)),
            "version": "0.31.0rc3.dev1",
            "sha256": release.digest(wheel),
        }
    manifest = {
        "schema": 1,
        "commits": {
            "vllm": (ROOT / ".github/vllm-main-verified.commit").read_text().strip(),
            "vllm-sail": COMMIT,
        },
        "wheels": records,
    }
    (wheels / "wheel-manifest.json").write_text(json.dumps(manifest))
    for key, value in {
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_SHA": COMMIT,
        "GITHUB_REPOSITORY": "vllm-project/vllm-sail",
        "RELEASE_TAG": TAG,
        "RELEASE_CONFIGURATION": json.dumps(CONFIG),
    }.items():
        monkeypatch.setenv(key, value)
    output = tmp_path / "assets"
    release.stage(wheels, output)
    monkeypatch.setenv(
        "EXPECTED_MANIFEST_SHA256", release.digest(output / release.MANIFEST)
    )
    return output


def fetch_args(assets):
    return argparse.Namespace(
        repository="vllm-project/vllm-sail",
        tag=TAG,
        commit=COMMIT,
        manifest_sha256=release.digest(assets / release.MANIFEST),
        configuration=json.dumps(CONFIG),
    )


def public_download(assets, urls):
    def download(url, target):
        urls.append(url)
        shutil.copyfile(assets / target.name, target)

    return download


def test_stage_preserves_both_original_versions_and_bytes(assets):
    data = json.loads((assets / release.MANIFEST).read_text())
    assert set(data["wheels"]) == {"vllm", "vllm-sail"}
    assert data["commits"]["vllm-sail"] == COMMIT
    for record in data["wheels"].values():
        assert record["version"] == "0.31.0rc3.dev1"
        assert release.digest(assets / record["file"]) == record["sha256"]


def test_stage_rejects_tampered_build_before_release(assets, tmp_path):
    wheel = next((tmp_path / "wheels").rglob("vllm_sail-*.whl"))
    wheel.write_bytes(b"replaced")
    with pytest.raises(subprocess.CalledProcessError):
        release.stage(tmp_path / "wheels", tmp_path / "invalid-assets")
    assert not (tmp_path / "invalid-assets").exists()


def test_ptg_downloads_only_sail_and_leaves_one_wheel(assets, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in CONFIG.items():
        monkeypatch.setenv(key.upper(), value)
    urls = []
    monkeypatch.setattr(release, "download", public_download(assets, urls))
    release.fetch_ptg(fetch_args(assets))
    assert len(urls) == 2
    assert all(f"/releases/download/{TAG}/" in url for url in urls)
    assert urls[1].split("/")[-1].startswith("vllm_sail-")
    assert [p.name for p in (tmp_path / "dist").iterdir()] == [Path(urls[1]).name]


@pytest.mark.parametrize(
    "fault", ["manifest", "wheel", "configuration", "source", "path"]
)
def test_ptg_rejects_invalid_handoff_before_staging(
    assets, tmp_path, monkeypatch, fault
):
    monkeypatch.chdir(tmp_path)
    for key, value in CONFIG.items():
        monkeypatch.setenv(key.upper(), value)
    args = fetch_args(assets)
    if fault == "manifest":
        args.manifest_sha256 = "0" * 64
    elif fault == "wheel":
        next(assets.glob("vllm_sail-*.whl")).write_bytes(b"replaced")
    elif fault == "configuration":
        monkeypatch.setenv("TORCH_VERSION", "2.10.0")
    elif fault == "source":
        args.commit = "b" * 40
    else:
        path = assets / release.MANIFEST
        data = json.loads(path.read_text())
        data["wheels"]["vllm-sail"]["file"] = "../escape.whl"
        path.write_text(json.dumps(data))
        args.manifest_sha256 = release.digest(path)
    monkeypatch.setattr(release, "download", public_download(assets, []))
    with pytest.raises(ValueError):
        release.fetch_ptg(args)
    assert not (tmp_path / "dist").exists()


def test_existing_different_asset_is_never_overwritten(assets, monkeypatch):
    wheel = next(assets.glob("vllm_sail-*.whl"))
    calls = []

    def api(endpoint):
        if "/git/ref/" in endpoint:
            return {"object": {"type": "commit", "sha": COMMIT}}
        return {"draft": False, "prerelease": True, "assets": [{"name": wheel.name}]}

    def run(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["release", "download"]:
            directory = Path(args[args.index("--dir") + 1])
            (directory / wheel.name).write_bytes(b"existing different build")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.setattr(release, "gh_json", api)
    monkeypatch.setattr(release.subprocess, "run", run)
    with pytest.raises(ValueError, match="existing release asset differs"):
        release.publish(assets)
    assert not any(args[1:3] == ["release", "upload"] for args in calls)


def test_publish_marks_rc_prerelease_and_emits_download_only_command(
    assets, monkeypatch, tmp_path
):
    calls = []
    exists = False

    def api(endpoint):
        if "/git/ref/" in endpoint:
            return {"object": {"type": "commit", "sha": COMMIT}}
        return {"draft": False, "prerelease": True, "assets": []}

    def run(args, **kwargs):
        nonlocal exists
        calls.append(args)
        if args[1] == "api" and not exists:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="HTTP 404")
        if args[1:3] == ["release", "create"]:
            exists = True
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.setattr(release, "gh_json", api)
    monkeypatch.setattr(release.subprocess, "run", run)
    release.publish(assets)
    create = next(args for args in calls if args[1:3] == ["release", "create"])
    assert "--verify-tag" in create and "--prerelease" in create
    assert len([args for args in calls if args[1:3] == ["release", "upload"]]) == 3
    output = (tmp_path / "outputs").read_text()
    assert "fetch-ptg" in output and "--manifest-sha256" in output
    assert "latest" not in output and "build_wheels" not in output


def test_publish_rejects_existing_rc_marked_as_stable(assets, monkeypatch):
    calls = []

    def api(endpoint):
        if "/git/ref/" in endpoint:
            return {"object": {"type": "commit", "sha": COMMIT}}
        return {"draft": False, "prerelease": False, "assets": []}

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.setattr(release, "gh_json", api)
    monkeypatch.setattr(release.subprocess, "run", run)
    with pytest.raises(ValueError, match="prerelease classification"):
        release.publish(assets)
    assert not any(args[1:3] == ["release", "upload"] for args in calls)


def test_plan_rejects_tag_pointing_to_other_source(monkeypatch):
    monkeypatch.setenv(
        "MATRIX_JSON", json.dumps({"include": [{**CONFIG, "image": "sdk"}]})
    )
    monkeypatch.setenv("TAG_VERSION", TAG)
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    monkeypatch.setattr(release, "command", lambda *args: "b" * 40)
    with pytest.raises(ValueError, match="target tag must point"):
        release.plan()


def test_plan_rejects_multi_matrix_before_build(monkeypatch):
    monkeypatch.setenv("MATRIX_JSON", json.dumps({"include": [CONFIG, CONFIG]}))
    with pytest.raises(ValueError, match="exactly one matrix"):
        release.plan()


def test_workflow_keeps_original_ptg_publisher_and_uses_script():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yaml").read_text())
    jobs = workflow["jobs"]
    ptg = jobs["publish-ptg"]
    assert ptg["uses"].endswith("@" + ptg["with"]["template_ref"])
    assert ptg["with"]["package_name"] == "vllm_sail"
    assert ptg["needs"] == "publish-github-release"
    build = jobs["build-wheels"]
    assert any(
        "bash scripts/build_wheels.sh" in step.get("run", "") for step in build["steps"]
    )
    assert jobs["publish-github-release"]["permissions"] == {"contents": "write"}
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request" not in workflow[True]
