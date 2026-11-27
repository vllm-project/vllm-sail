# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tests.upstream_reuse.model_assets import run


def test_asset_probe_shards_cover_each_module_once():
    modules = {
        f"tests/test_{i}.py": [f"test_{j}" for j in range(i)] for i in range(1, 20)
    }
    shards = [run.shard_modules(modules, 3, i) for i in range(3)]
    assert sorted(path for shard in shards for path in shard) == sorted(modules)
    assert shards == [
        run.shard_modules(dict(reversed(list(modules.items()))), 3, i) for i in range(3)
    ]


def test_asset_probe_requires_pinned_manifest():
    pin = (
        (run.HERE.parents[2] / ".github/vllm-main-verified.commit").read_text().strip()
    )
    assert run.load_modules(run.HERE / "manifest.json", pin)
    with pytest.raises(ValueError, match="upstream pin"):
        run.load_modules(run.HERE / "manifest.json", "0" * 40)


@pytest.fixture
def asset_test_tree(tmp_path):
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "conftest.py").write_text(
        "def pytest_addoption(parser):\n"
        '    parser.addoption("--optional", action="store_true")\n',
        encoding="utf-8",
    )
    (test_dir / "test_assets.py").write_text(
        "import pytest\n"
        "class TestModel:\n"
        '    @pytest.mark.parametrize("model", ["cached", "missing"])\n'
        "    def test_load(self, model):\n"
        '        print("cache lookup:", model)\n'
        '        assert model == "cached", "missing model"\n'
        "def test_skip():\n"
        '    pytest.skip("requires another backend")\n'
        '@pytest.mark.xfail(reason="known failure")\n'
        "def test_xpass():\n"
        "    pass\n"
        "def test_unselected():\n"
        '    raise RuntimeError("must be deselected")\n',
        encoding="utf-8",
    )
    return tmp_path


def test_asset_probe_selects_exact_parameter_id(asset_test_tree):
    output = asset_test_tree / "out"
    result = run.run_module(
        asset_test_tree,
        "tests/test_assets.py",
        ["TestModel::test_load[cached]"],
        output,
        30,
    )
    assert result == {"status": "completed", "exit_code": 0}
    collected = json.loads((output / "events.jsonl").read_text().splitlines()[0])
    assert collected["nodeids"] == [
        "tests/test_assets.py::TestModel::test_load[cached]"
    ]


def test_asset_probe_requires_a_pass_for_each_selected_node(asset_test_tree):
    result = run.run_module(
        asset_test_tree,
        "tests/test_assets.py",
        ["TestModel::test_load[cached]", "test_missing", "test_skip", "test_xpass"],
        asset_test_tree / "out",
        30,
    )
    assert result == {
        "status": "not_passed",
        "nodeids": [
            "tests/test_assets.py::test_missing",
            "tests/test_assets.py::test_skip",
            "tests/test_assets.py::test_xpass",
        ],
    }


def test_asset_probe_keeps_parameter_ids_and_reasons(asset_test_tree):
    output = asset_test_tree / "out"
    result = run.run_module(
        asset_test_tree,
        "tests/test_assets.py",
        ["TestModel::test_load[cached]", "TestModel::test_load[missing]", "test_skip"],
        output,
        30,
    )
    assert result == {"status": "completed", "exit_code": 1}
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    collected = next(
        event["nodeids"] for event in events if event["event"] == "collected"
    )
    assert collected == [
        "tests/test_assets.py::TestModel::test_load[cached]",
        "tests/test_assets.py::TestModel::test_load[missing]",
        "tests/test_assets.py::test_skip",
    ]
    reasons = {
        event["outcome"]: event["reason"] for event in events if event.get("reason")
    }
    assert "missing model" in reasons["failed"]
    assert "cache lookup: missing" in reasons["failed"]
    assert "requires another backend" in reasons["skipped"]
