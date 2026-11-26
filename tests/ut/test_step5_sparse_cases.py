# SPDX-License-Identifier: Apache-2.0
"""Protocol, dry-run and lazy instrumentation regressions without torch/vLLM."""

from __future__ import annotations

import ast
import contextlib
import csv
import importlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_sail.step5_sparse_cases import (
    estimate_memory,
    extract_cases,
    make_label,
    parse_label,
    profiled,
    storage_elements,
)

ROOT = Path(__file__).resolve().parents[2]
NVTX_UT = ROOT / "tests/nvtx_based_ut/step5"


class Tensor:
    is_cuda = False
    dtype = "torch.float32"

    def __init__(self, shape=(2, 8), stride=(12, 1), offset=3):
        self.shape, self._stride, self._offset = shape, stride, offset

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self._offset

    def __getattr__(self, name):
        if name in ("item", "cpu", "tolist", "data_ptr"):
            pytest.fail("Instrumentation read tensor contents: " + name)
        raise AttributeError(name)

    def __repr__(self):
        pytest.fail("Instrumentation formatted arbitrary tensor repr")


@pytest.mark.parametrize("op", ["score", "topk", "attention"])
def test_roundtrip(op):
    label = make_label(
        op,
        "real.launch",
        "resident",
        dict(query=Tensor(), weights=None, topk=4, backend="auto", opaque=object()),
        scope="kernel",
        capture=True,
    )
    case = parse_label(label)
    assert case["tensors"]["query"] == dict(
        shape=[2, 8], stride=[12, 1], offset=3, dtype="torch.float32"
    )
    assert case["tensors"]["weights"] is None
    assert case["backend"] == "resident" and case["requested_backend"] == "auto"
    assert case["scalars"] == {"capture": 1, "topk": 4}
    assert estimate_memory(case) == 23 * 4


def test_mixed_txt_dedup_provenance(tmp_path):
    label = make_label(
        "topk", "select", "per_row", dict(scores=Tensor(), current=None, topk=2)
    )
    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    files[0].write_text("[FW_SPARSE] op:step5_dsa,pd_type:decode\n123 " + label + "\n")
    files[1].write_text(label + "\n")
    cases = extract_cases(files)["cases"]
    assert len(cases) == 1 and cases[0]["observations"] == 2
    assert [s["line"] for s in cases[0]["sources"]] == [2, 1]
    assert all(len(s["sha256"]) == 64 for s in cases[0]["sources"])
    assert cases[0]["raw_label"] == label


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("v:1", "v:2"),
        lambda s: s + ",v:1",
        lambda s: s + ",evil:__import__('os')",
        lambda s: s.replace("query:(2, 8)", "query:(-1, 8)"),
        lambda s: s.replace("query_stride:(12, 1)", "query_stride:(1, 1)"),
        lambda s: s.replace("query_stride:(12, 1)", "query_stride:(0, 1)"),
        lambda s: s.replace("query_stride:(12, 1)", "query_stride:(1,)"),
        lambda s: s.replace(
            "query_dtype:torch.float32", "query_dtype:torch.complex128"
        ),
        lambda s: s.replace("query:(2, 8)", "query:None"),
        lambda s: s + ",scale:nan",
        lambda s: s + ",requested_backend:bad/path",
        lambda s: s.replace("capture:0", "capture:2"),
    ],
)
def test_strict_rejection(change, tmp_path):
    label = change(make_label("score", "call", "baseline", dict(query=Tensor())))
    path = tmp_path / "invalid.txt"
    path.write_text("ignored\n" + label + "\n")
    with pytest.raises(ValueError, match=r"invalid.txt:2:"):
        extract_cases([path])


def test_storage_scalar_empty_and_transpose():
    assert storage_elements(dict(shape=[], stride=[], offset=4)) == 5
    assert storage_elements(dict(shape=[0], stride=[1], offset=4)) == 4
    assert storage_elements(dict(shape=[8, 2], stride=[1, 12], offset=3)) == 23
    with pytest.raises(ValueError):
        storage_elements(dict(shape=[2], stride=[-1], offset=0))


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(NVTX_UT))
    spec = importlib.util.spec_from_file_location(
        "sparse_bench_test", NVTX_UT / "benchmark_step5_sparse_nvtx.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def topk_case():
    current = Tensor((2,), (1,), 0)
    current.dtype = "torch.int64"
    case = parse_label(
        make_label(
            "topk", "select", "dispatch", dict(scores=Tensor(), current=current, topk=4)
        )
    )
    case["sources"] = []
    return case


def test_dry_run_requires_workload_and_preserves_layout(bench):
    case = topk_case()
    args = SimpleNamespace(scope="prepared-op", max_memory_mib=2048, backends=None)
    assert bench.dry_case(case, {}, args)["status"] == "unsupported"
    ready = bench.dry_case(case, {"default": {"policy": "capacity-causal"}}, args)
    assert ready["status"] == "ready" and ready["workload"]["synthetic"]
    assert ready["workload"]["current"] == [7, 6]
    assert case["tensors"]["scores"]["stride"] == [12, 1]
    args.max_memory_mib = 1
    assert (
        bench.dry_case(case, {"default": {"policy": "capacity-causal"}}, args)["status"]
        == "unsupported"
    )


@pytest.mark.parametrize("backend", ["per_row", "deep_select", "selector_v2"])
def test_topk_kernel_requires_recorded_current_layout(bench, backend):
    case = topk_case()
    case["scope"] = "kernel"
    case["tensors"]["current"] = None
    args = SimpleNamespace(scope="kernel", max_memory_mib=2048, backends=[backend])
    result = bench.dry_case(
        case,
        {"default": {"policy": "capacity-causal", "current": [7, 6]}},
        args,
    )
    assert result["status"] == "unsupported"
    assert result["reason"] == "Missing or unsupported current"


@pytest.mark.parametrize("op", ["score", "attention"])
def test_strided_cache_budget_counts_storage_once_per_backend(bench, op):
    def tensor(shape, stride, dtype, offset=0):
        item = Tensor(shape, stride, offset)
        item.dtype = dtype
        return item

    values = dict(
        block_table=tensor((16, 16384), (16384, 1), "torch.int32"),
        query_len=4,
    )
    if op == "score":
        values.update(
            query=tensor(
                (64, 1, 8, 256), (2048, 131072, 256, 1), "torch.float8_e4m3fn"
            ),
            mean=tensor((64530, 8, 256), (1179648, 256, 1), "torch.uint8", 1130496),
            max_seq_len=131072,
        )
        backend = "direct_mtp"
    else:
        values.update(
            query=tensor((64, 16, 192), (3072, 192, 1), "torch.bfloat16"),
            kv_cache=tensor(
                (64530, 1, 64, 384), (589824, 384, 384, 1), "torch.bfloat16", 196608
            ),
            indices=tensor((64, 513), (513, 1), "torch.int64"),
        )
        backend = "native"
    case = parse_label(make_label(op, "launch", backend, values, pd_type="decode"))
    case["sources"] = []
    args = SimpleNamespace(
        scope="prepared-op", max_memory_mib=100 * 1024, backends=[backend]
    )
    config = {"default": {"policy": "capacity-causal"}}
    ready = bench.dry_case(case, config, args)
    assert ready["status"] == "ready"
    assert 70 * 2**30 < ready["estimated_peak_bytes"] < 100 * 2**30
    assert ready["memory_components_per_backend"]["recorded_storage_bytes"] == (
        estimate_memory(case)
    )
    if op == "attention":
        assert (
            ready["memory_components_by_backend"]["native"]["gathered_budget_bytes"]
            == 0
        )
    other_backend = "direct_baseline" if op == "score" else "upstream_triton"
    args.backends = [backend, other_backend]
    assert bench.dry_case(case, config, args)["status"] == "unsupported"
    args.backends = [backend]
    args.max_memory_mib = 64 * 1024
    rejected = bench.dry_case(case, config, args)
    assert rejected["status"] == "unsupported"
    assert rejected["estimated_peak_bytes"] == estimate_memory(case)
    assert rejected["estimated_peak_bytes"] > rejected["memory_budget_bytes"]


@pytest.fixture
def large_prefill_budget_case(tmp_path):
    specs = {
        "query": ((8192, 16, 192), (3072, 192, 1), "torch.bfloat16"),
        "kv_cache": ((64530, 1, 64, 384), (589824, 384, 384, 1), "torch.bfloat16"),
        "indices": ((8192, 513), (513, 1), "torch.int64"),
        "block_table": ((14, 16384), (16384, 1), "torch.int32"),
        "output": ((8192, 16, 192), (3072, 192, 1), "torch.bfloat16"),
        "owners": ((8192,), (1,), "torch.int64"),
        "positions": ((8192,), (1,), "torch.int64"),
        "tiles": ((2, 1031), (1031, 1), "torch.int32"),
    }
    values = dict(max_seq_len=8225, query_tile=0, scale=0.07216878364870322)
    for name, (shape, stride, dtype) in specs.items():
        tensor = Tensor(shape, stride, 0)
        tensor.dtype = dtype
        values[name] = tensor
    case = parse_label(
        make_label(
            "attention",
            "step5_dsa_native_prefill",
            "native_20261110",
            values,
            pd_type="prefill",
        )
    )
    case["sources"] = []
    case["topk_tensor"] = "input.pt"
    (tmp_path / "input.pt").touch()
    return case


@pytest.mark.parametrize(
    "scope,backends",
    [
        ("prepared-op", ["native"]),
        ("kernel", ["native"]),
        ("prepared-op", ["flashinfer_paged"]),
        ("prepared-op", ["flashinfer_ragged"]),
        ("prepared-op", ["native", "flashinfer_ragged"]),
        ("prepared-op", ["flashinfer_ragged", "native"]),
    ],
)
def test_prefill_budget_matches_backend_staging(
    bench, tmp_path, large_prefill_budget_case, scope, backends
):
    """Native paged KV must not reserve an all-row gathered cache."""
    case = large_prefill_budget_case
    case["scope"] = "kernel" if scope == "kernel" else "op"
    original = json.loads(json.dumps(case))
    args = SimpleNamespace(
        scope=scope, max_memory_mib=102400, backends=backends, topk_tensor_dir=tmp_path
    )
    result = bench.dry_case(case, {}, args)
    assert case == original
    assert not result["synthetic"]
    assert result["content_validation"] == "deferred-until-torch-load"
    components = result["memory_components_by_backend"]
    gathered = 8192 * 513 * 8 * 384 * 2 * 3
    for backend, parts in components.items():
        assert parts["recorded_storage_bytes"] == estimate_memory(case)
        assert parts["gathered_budget_bytes"] == (
            0 if backend == "native" else gathered
        )
        gib = sum(parts.values()) / 2**30
        assert 83 < gib < 84 if backend == "native" else 155 < gib < 156
    assert result["estimated_peak_bytes"] == sum(
        sum(parts.values()) for parts in components.values()
    )
    if backends == ["native"]:
        assert result["status"] == "ready"
        assert result["memory_components_per_backend"]["gathered_budget_bytes"] == 0
        assert result["estimated_peak_bytes"] < result["memory_budget_bytes"]
    else:
        assert result["status"] == "unsupported"
        assert "exceeds --max-memory-mib" in result["reason"]
        assert result["estimated_peak_bytes"] > result["memory_budget_bytes"]
        assert result["backends"] == backends
        assert result["geometry"]["rows"] == 8192


@pytest.mark.parametrize("budget_mib", [64 * 1024, 80 * 1024])
def test_runner_keeps_rejected_memory_diagnostics(
    bench, tmp_path, large_prefill_budget_case, budget_mib
):
    case = large_prefill_budget_case
    admission = bench.dry_case(
        case,
        {},
        SimpleNamespace(
            scope="prepared-op",
            max_memory_mib=budget_mib,
            backends=["native"],
            topk_tensor_dir=tmp_path,
        ),
    )
    assert admission["status"] == "unsupported"
    tree = ast.parse((NVTX_UT / "run_step5_dsa_nvtx_ut.py").read_text())
    loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [item.id for item in node.target.elts] == ["case", "admission"]
    )
    rejection = next(node for node in loop.body if isinstance(node, ast.If))
    namespace = dict(
        case=case,
        admission=admission,
        output=tmp_path,
        prefix="case",
        op=case["op"],
        scope=case["scope"],
    )
    exec(
        compile(
            ast.Module(body=[rejection.body[0]], type_ignores=[]), "rejection", "exec"
        ),
        namespace,
    )
    entry = namespace["entry"]
    assert entry["results"] == []
    assert entry["estimated_peak_bytes"] > entry["memory_budget_bytes"]
    for name in (
        "estimated_peak_bytes",
        "memory_budget_bytes",
        "recorded_storage_bytes",
        "memory_components_per_backend",
        "memory_components_by_backend",
    ):
        if name in admission:
            assert entry[name] == admission[name]
    if budget_mib == 80 * 1024:
        assert (
            admission["memory_components_by_backend"]["native"]["gathered_budget_bytes"]
            == 0
        )


def test_explicit_workload_validation(bench):
    case = topk_case()
    args = SimpleNamespace(scope="prepared-op", max_memory_mib=2048, backends=None)
    for values in (
        {"policy": "explicit"},
        {"policy": "explicit", "current": [8, 8]},
        {"policy": "explicit", "current": [1.5, 2]},
        {"policy": "capacity-causal", "scale": float("nan")},
    ):
        assert (
            bench.dry_case(case, {"default": values}, args)["status"] == "unsupported"
        )


@pytest.mark.parametrize("output_prefix", ["", "first"])
def test_cli_compare_exports_only_selected_op_and_case(tmp_path, output_prefix):
    current = Tensor((2,), (1,), 0)
    current.dtype = "torch.int64"
    labels = [
        make_label(
            "topk", "select", "per_row", dict(scores=Tensor(), current=current, topk=k)
        )
        for k in (2, 4)
    ]
    labels.append(make_label("score", "score", "baseline", dict(query=Tensor())))
    source = tmp_path / "mixed.caselist"
    source.write_text("\n".join(labels) + "\n")
    selected_id = parse_label(labels[0])["id"]
    config = tmp_path / "workload.json"
    config.write_text(json.dumps({"default": {"policy": "capacity-causal"}}))
    output = tmp_path / "compare"
    if output_prefix:
        output.mkdir()
        (output / "keep.txt").write_text("Preserve existing output.\n")
    run = subprocess.run(
        [
            sys.executable,
            str(NVTX_UT / "benchmark_step5_sparse_nvtx.py"),
            "compare",
            "--nvtx",
            str(source),
            "--op",
            "topk",
            "--case-id",
            selected_id,
            "--backends",
            "per_row",
            "--workload-config",
            str(config),
            "--output-dir",
            str(output),
            "--output-prefix",
            output_prefix,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    filename = f"{output_prefix}-cases.json" if output_prefix else "cases.json"
    cases = json.loads((output / filename).read_text())["cases"]
    assert [(case["op"], case["id"]) for case in cases] == [("topk", selected_id)]
    assert cases[0]["sources"][0]["path"] == str(source)
    if not output_prefix:
        return
    command = list(run.args)
    command[command.index("--case-id") + 1] = parse_label(labels[1])["id"]
    command[command.index("--output-prefix") + 1] = "second"
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    second_cases = json.loads((output / "second-cases.json").read_text())["cases"]
    assert second_cases[0]["id"] == parse_label(labels[1])["id"]
    assert all(path.is_file() for path in output.iterdir())
    assert {path.name for path in output.iterdir()} == {
        "keep.txt",
        "first-cases.json",
        "first-dry-run.json",
        "second-cases.json",
        "second-dry-run.json",
    }
    (output / "reserved-report.json").write_text("Preserve existing report.\n")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    for prefix in ("first", "reserved", "../escape", "/absolute", "a/b"):
        command[command.index("--output-prefix") + 1] = prefix
        rejected = subprocess.run(command, capture_output=True, text=True)
        assert rejected.returncode != 0
        message = (
            "Output file already exists"
            if prefix in ("first", "reserved")
            else "Output prefix"
        )
        assert message in rejected.stderr
        assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_cli_extract_dry_run_and_no_overwrite(tmp_path):
    label = make_label(
        "topk", "select", "per_row", dict(scores=Tensor(), current=None, topk=4)
    )
    source = tmp_path / "log.txt"
    source.write_text(label + "\n")
    script = NVTX_UT / "benchmark_step5_sparse_nvtx.py"
    output = tmp_path / "cases.json"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "extract",
            "--nvtx",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["cases"]
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "extract",
            "--nvtx",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert json.loads(output.read_text())["cases"]


@pytest.mark.parametrize("visible", [None, "1", "1,2", ""])
def test_nvtx_runner_preserves_device_selection_and_defaults_to_zero(visible):
    """The child environment must not replace an explicit device selection."""
    tree = ast.parse((NVTX_UT / "run_step5_dsa_nvtx_ut.py").read_text())
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "env" for t in node.targets)
    )
    stop = next(
        i
        for i, node in enumerate(tree.body[start:], start)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "env"
        and node.value.func.attr == "update"
    )
    inherited = {} if visible is None else {"CUDA_VISIBLE_DEVICES": visible}
    namespace = {"os": SimpleNamespace(environ=inherited)}
    exec(
        compile(
            ast.Module(body=tree.body[start : stop + 1], type_ignores=[]),
            "runner_environment",
            "exec",
        ),
        namespace,
    )
    assert namespace["env"]["CUDA_VISIBLE_DEVICES"] == (
        "0" if visible is None else visible
    )
    assert inherited == ({} if visible is None else {"CUDA_VISIBLE_DEVICES": visible})


@pytest.mark.parametrize("missing_tensor", [False, True])
def test_copied_nvtx_ut_runs_without_the_repository(tmp_path, missing_tensor):
    portable = tmp_path / "copied-ut"
    shutil.copytree(NVTX_UT, portable, ignore=shutil.ignore_patterns("__pycache__"))
    source = tmp_path / "input.caselist"
    label = topk_case()["raw_label"]
    if missing_tensor:
        label = label.replace("topk_tensor:", "topk_tensor:missing.pt")
    source.write_text(label + "\n")
    selected_id = parse_label(label)["id"]
    command = [
        sys.executable,
        "-I",
        str(portable / "run_step5_dsa_nvtx_ut.py"),
        "--nvtx",
        str(source),
        "--op",
        "topk",
        "--backends",
        "per_row",
        "--dry-only",
        "--peak-compute-tflops",
        "2",
        "--peak-compute-dtype",
        "bf16",
        "--peak-hbm-gbps",
        "200",
    ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "output"
    runs = list(output.iterdir())
    assert len(runs) == 1 and all(path.is_file() for path in runs[0].iterdir())
    selected = json.loads((runs[0] / "selection.json").read_text())
    assert selected["selected_case_ids"] == [selected_id]
    assert selected["utilization_peaks"]["compute_dtype"] == "bf16"
    assert selected["utilization_peaks"]["compute_tflops"] == 2
    entry = json.loads((runs[0] / "summary.json").read_text())[0]
    assert entry["status"] == "ready" and entry["synthetic"]
    assert entry["utilization_peaks"] == selected["utilization_peaks"]
    assert "utilization" not in entry
    if missing_tensor:
        assert "missing.pt not found" in result.stderr
        assert entry["topk_tensor_fallback"]["filename"] == "missing.pt"
        assert entry["workload"]["current"] == [7, 6]
    for extra in (["--help"], ["--op", "topk", "--nvtx", str(source)]):
        result = subprocess.run(
            command[:3] + extra, cwd=tmp_path, capture_output=True, text=True
        )
        assert "Supported backends" in result.stdout + result.stderr
        assert result.returncode == (0 if extra == ["--help"] else 2)


def test_portable_oracles_keep_the_existing_correctness_contract():
    copied = ast.parse((NVTX_UT / "step5_sparse_support.py").read_text())
    definitions = {
        node.name: ast.dump(node)
        for node in copied.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }
    for path, names in (
        (
            "tests/support/step5_score_testing.py",
            ("direct_cpu_oracle", "error_metrics"),
        ),
        ("tests/support/step5_attention_testing.py", ("check_selection_oracle",)),
    ):
        original = ast.parse((ROOT / path).read_text())
        for node in original.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                assert definitions[node.name] == ast.dump(node)


@pytest.mark.parametrize("missing_git", [False, True])
def test_portable_report_does_not_require_git(bench, monkeypatch, missing_git):
    def git(*args, **kwargs):
        if missing_git:
            raise FileNotFoundError("git is not installed")
        return subprocess.CompletedProcess(args[0], 128, "", "not a git repository")

    monkeypatch.setattr(bench.subprocess, "run", git)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda _: SimpleNamespace(name="PPU", uuid="test"),
            get_device_capability=lambda _: (8, 9),
        )
    )
    result = bench.provenance(torch)
    assert result["revision"] is None and result["dirty"] is None
    assert result["device"]["capability"] == [8, 9]
    assert "versions" in result and "sources" in result


def test_timer_statistics_and_invalid_samples(bench):
    result = bench.summarize([10.0, 11.0, 9.0, 10.0])
    assert result["median_us"] == 10 and result["range_pct"] == 20
    for samples in ([], [float("nan")], [0.0], [-1.0]):
        with pytest.raises(RuntimeError):
            bench.summarize(samples)


@pytest.fixture
def sparse(monkeypatch):
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.step5_sparse", raising=False)
    module = importlib.import_module("vllm_sail.profiling.step5_sparse")
    package = importlib.import_module("vllm_sail.profiling")
    events = []
    monkeypatch.setattr(
        package,
        "nvtx",
        SimpleNamespace(
            range_push=lambda s: events.append(parse_label(s)),
            range_pop=lambda: events.append("pop"),
        ),
        raising=False,
    )
    return module, events


def test_lazy_gate_phase_nesting_and_exception(sparse, monkeypatch):
    module, events = sparse
    calls = []

    @profiled("score")
    def original(query):
        calls.append(query)
        raise RuntimeError("original device error")

    class BadTensor:
        @property
        def shape(self):
            pytest.fail("Gate-off queried metadata")

    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "0")
    with pytest.raises(RuntimeError, match="original device"):
        original(BadTensor())
    assert not events and len(calls) == 1
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    with module.phase("decode"), pytest.raises(RuntimeError, match="original device"):
        module.trace_call(
            "score", "dispatch", "dispatch", {}, lambda: original(Tensor())
        )
    assert len(events) == 2 and events[0]["pd_type"] == "decode" and events[1] == "pop"
    assert module._RANGES.get() == () and module._PHASE.get() == "unknown"


def test_missing_signature_preserves_callable_error(monkeypatch):
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")

    @profiled("score")
    def run(*args):
        return args

    monkeypatch.setattr(
        "vllm_sail.step5_sparse_cases.inspect.signature",
        lambda f: (_ for _ in ()).throw(ValueError("no signature")),
    )
    assert run(3) == (3,)


def test_install_gate_aliases_idempotency_and_owned_target(sparse, monkeypatch):
    from vllm_sail.patch.utils import PATCH_MARKER, original_of

    module, events = sparse
    provider_name = "vllm.models.step5.nvidia.ops.fake_sparse"
    consumer_name = "vllm.models.step5.nvidia.fake_consumer"
    provider, consumer = (
        types.ModuleType(provider_name),
        types.ModuleType(consumer_name),
    )

    def run(query):
        return query

    provider.run = consumer.run = run
    monkeypatch.setitem(sys.modules, provider_name, provider)
    monkeypatch.setitem(sys.modules, consumer_name, consumer)
    monkeypatch.setattr(
        module,
        "_TARGETS",
        (("ops.fake_sparse", "run", "score", "actual", "op", "decode"),),
    )
    monkeypatch.setattr(module, "_install_topk_factory", lambda: None)
    monkeypatch.setattr(module, "_install_adapter_logits", lambda: None)
    monkeypatch.setattr(module, "_install_launches", lambda: None)
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "0")
    assert not module.install() and provider.run is run
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    assert module.install() and module.install()
    assert (
        provider.run is consumer.run
        and original_of(provider.run, provider_name + ".run") is run
    )
    assert provider.run(Tensor()) is not None and events[0]["backend"] == "actual"
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "0")
    events.clear()
    provider.run(Tensor())
    assert not events

    def owned(query):
        return query

    setattr(owned, PATCH_MARKER, {"owned": run})
    provider.run = owned
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    assert not module.install() and provider.run is owned


def test_flashinfer_hook_never_mislabels_resident(sparse, monkeypatch):
    module, _ = sparse
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")

    def resident():
        pass

    adapter = SimpleNamespace(_logits=resident)
    module.instrument_adapter(adapter)
    assert adapter._logits is resident


def test_topk_observes_resolution_once_and_only_in_step5(sparse, monkeypatch):
    module, events = sparse
    name = "vllm.models.step5.nvidia.ops.sparse_indexer"
    provider = types.ModuleType(name)

    class Runner:
        def __init__(self):
            self.resolutions = 0

        def resolve_backend(self, *args):
            self.resolutions += 1
            return "per_row"

        def __call__(self, scores, *args):
            self.resolve_backend(scores, 4, 2)
            raise RuntimeError("real launch failure")

    runner = Runner()
    provider.get_indexer_topk = lambda backend: runner
    monkeypatch.setitem(sys.modules, name, provider)
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    module._install_topk_factory()
    assert provider.get_indexer_topk("auto") is runner
    with pytest.raises(RuntimeError, match="real launch failure"):
        module.trace_call(
            "topk",
            "select",
            "dispatch",
            {},
            lambda: provider.get_indexer_topk("auto")(Tensor(), None, 1, None, 4, 8),
        )
    kernel = [e for e in events if isinstance(e, dict) and e["scope"] == "kernel"]
    assert len(kernel) == 1 and kernel[0]["backend"] == "per_row"
    assert kernel[0]["requested_backend"] == "auto" and runner.resolutions == 1
    assert events[-2:] == ["pop", "pop"]
    assert module._TOPK_CONTEXT.get() is None and module._RANGES.get() == ()
    events.clear()
    with pytest.raises(RuntimeError):
        runner(Tensor())
    assert not events


def test_observed_paged_launch_mapping_and_gate(sparse, monkeypatch):
    module, events = sparse

    class JIT:
        __name__ = "_paged_scores_sm90"

        def __getitem__(self, grid):
            return lambda *args, **kw: (grid, args)

    observer = module.LaunchObserver(
        JIT(),
        "score",
        "upstream_paged",
        ("query", "packed", "weights", "lengths", "block_table", "output"),
        {"max_seq_len": 15, "next_n": 16, "heads": 17, "page_size": 18, "tile": 19},
        "unknown",
    )
    args = (Tensor(), None, None, None, None, None, *range(6, 15), 64, 2, 8, 32, 32)
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    assert observer[(2, 2)](*args)[1] == args
    assert events[0]["scalars"]["max_seq_len"] == 64
    assert events[0]["scalars"]["heads"] == 8 and events[0]["scalars"]["tile"] == 32
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "0")
    events.clear()
    observer[(2, 2)](*args)
    assert not events


def test_existing_flashinfer_instances_are_instrumented_on_call(sparse, monkeypatch):
    _, events = sparse
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")

    class CompactPagedPrefill:
        def __init__(self):
            self.wrapper = SimpleNamespace(run=lambda query: query)

        def finish(self, query):
            return self.wrapper.run(query)

    CompactPagedPrefill.finish.__qualname__ = "CompactPagedPrefill.finish"
    instance = CompactPagedPrefill()
    CompactPagedPrefill.finish = profiled(
        "attention", backend="flashinfer_paged", pd_type="prefill"
    )(CompactPagedPrefill.finish)
    assert instance.finish(Tensor()) is not None
    assert [e["scope"] for e in events if isinstance(e, dict)] == ["op", "kernel"]
    original = instance.wrapper.run
    instance.finish(Tensor())
    assert instance.wrapper.run is original


def test_byte_launch_mapping_uses_original_tensor_metadata(sparse, monkeypatch):
    _, events = sparse
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    tree = ast.parse((ROOT / "vllm_sail/models/step5/ops/soft_fp8.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ByteFP8Launch"
    )
    method = next(
        n
        for n in node.body
        if isinstance(n, ast.FunctionDef) and n.name == "_submit_paged"
    )
    namespace = {"envs": SimpleNamespace(VLLM_SAIL_NVTX_PROFILE=True)}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "launcher_contract", "exec"
        ),
        namespace,
    )
    source = (Tensor(), None, None, None, None, None, *range(6, 15), 64, 2, 8, 32, 32)
    assert (
        namespace["_submit_paged"](
            SimpleNamespace(__name__="_paged_scores_sm90"),
            lambda *a, **k: 7,
            list(source),
            {},
            source,
            "software_paged",
        )
        == 7
    )
    assert events[0]["backend"] == "software_paged"
    assert (
        events[0]["scalars"]["max_seq_len"] == 64
        and events[0]["scalars"]["next_n"] == 2
    )


def test_dry_run_reports_backend_contracts(bench):
    args = SimpleNamespace(
        scope="kernel",
        max_memory_mib=2048,
        backends=["per_row", "selector_v2", "invalid"],
    )
    ready = bench.dry_case(
        topk_case(), {"default": {"policy": "capacity-causal"}}, args
    )
    assert ready["status"] == "ready"
    assert [item["status"] for item in ready["backend_support"]] == [
        "requires-device-admission",
        "unsupported",
        "unsupported",
    ]


@pytest.mark.parametrize("input_source", ["synthetic", "recorded", "missing"])
@pytest.mark.parametrize("timer_failure", [False, True])
@pytest.mark.parametrize("check_failure", [None, 1, 2, 3, 4, "all"])
def test_graph_timer_batches_device_ops_and_preserves_samples(
    bench, monkeypatch, timer_failure, check_failure, input_source
):
    import step5_sparse_replay as replay

    state = {"capturing": None, "timed": False, "timed_replays": 0, "checks": 0}
    graphs = []

    @contextlib.contextmanager
    def capture(graph):
        assert not state["timed"]
        state["capturing"] = graph
        try:
            yield
        finally:
            state["capturing"] = None

    def call():
        assert not state["timed"]
        if state["capturing"] is not None:
            state["capturing"].nodes += 1

    def check():
        assert not state["timed"]
        state["checks"] += 1
        if check_failure in (state["checks"], "all"):
            error = AssertionError("Numerical mismatch")
            error.metrics = {"max_abs": 0.01}
            raise error
        return {"status": "passed"}

    class Stream:
        def wait_stream(self, other):
            pass

    class Graph:
        def __init__(self):
            self.nodes = 0
            graphs.append(self)

        def replay(self):
            if state["timed"]:
                assert self.nodes == 4
                state["timed_replays"] += 1

    class Event:
        records = 0

        def __init__(self, **kwargs):
            pass

        def record(self):
            Event.records += 1
            if timer_failure and Event.records == 3:
                raise RuntimeError("timer failed after first sample")
            state["timed"] = Event.records % 2 == 1

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 0.04

    cuda = SimpleNamespace(
        Stream=Stream,
        current_stream=Stream,
        stream=lambda stream: contextlib.nullcontext(),
        synchronize=lambda: None,
        CUDAGraph=Graph,
        graph=capture,
        Event=Event,
    )
    arm = replay.Arm(
        "per_row",
        "real.launch",
        {},
        call,
        check,
        {"bytes": 4, "flops": 0},
        "same-boundary",
    )
    case = topk_case()
    if input_source != "synthetic":
        case["topk_tensor"] = "input.pt"
    normalized = {"synthetic": input_source != "recorded"}
    if input_source == "missing":
        normalized = replay.fallback_topk_workload(case, normalized)
    arm.config["normalized_workload"] = normalized
    monkeypatch.setattr(replay, "prepare_arms", lambda *args: ([arm], []))
    dry = {
        "geometry": {},
        "workload": {},
        "backends": ["per_row"],
        "synthetic": input_source == "synthetic",
    }
    args = SimpleNamespace(timer="graph-events", warmup=1, samples=2, iterations=4)
    gate_calls = 0

    def gate():
        nonlocal gate_calls
        assert not state["timed"]
        gate_calls += 1

    result = bench.compare_case(
        case,
        dry,
        args,
        SimpleNamespace(cuda=cuda),
        gate,
    )
    assert gate_calls == (4 if timer_failure else 5)
    assert result["synthetic"] == (input_source != "recorded")
    assert result["arms"][0]["synthetic"] == result["synthetic"]
    if input_source == "missing":
        assert result["topk_tensor_fallback"]["reason"] == "file-not-found"
        assert (
            result["arms"][0]["topk_tensor_fallback"] == result["topk_tensor_fallback"]
        )
    assert len(result["arms"]) == 1 and [graph.nodes for graph in graphs] == [1, 4]
    item = result["arms"][0]
    stages = ["correctness", "graph_correctness", "batch_graph_correctness"]
    if timer_failure:
        assert result["status"] == "failed" and state["timed_replays"] == 1
        assert item["status"] == "execution_failed" and item["samples_us"] == [10.0]
        assert item["failed_round"] == 1 and "speedup" not in item
        assert "timing" not in item and state["checks"] == 3
    else:
        stages.append("post_timing_correctness")
        assert state["timed_replays"] == 2 and state["checks"] == 4
        assert item["timing"]["median_us"] == 10.0
        assert item["timing"]["samples_us"] == [10.0, 10.0]
        assert item["timing"]["time_domain"] == "device"
        assert item["timing"]["op_repeats_per_graph"] == 4
        assert item["timing"]["valid_for_comparison"] == (check_failure is None)
        if check_failure is not None:
            assert result["status"] == "failed"
            assert item["status"] == "correctness_failed"
            assert "speedup" not in item and "estimated_gbps" not in item
            assert "utilization" not in item
        else:
            assert result["status"] == "completed" and item["status"] == "stable"
            assert item["speedup"] == 1.0
            assert item["utilization"]["mfu_pct"] is None
            assert (
                item["utilization"]["mfu_reason"] == "no-floating-point-work-accounting"
            )
    for index, stage in enumerate(stages, 1):
        if check_failure in (index, "all"):
            assert item[stage] == {
                "status": "failed",
                "error": "Numerical mismatch",
                "metrics": {"max_abs": 0.01},
            }
            assert item["metrics"] == {"max_abs": 0.01}
        else:
            assert item[stage]["status"] == "passed"


def test_case_cli_prints_device_timing_for_correctness_failure(capsys):
    tree = ast.parse((NVTX_UT / "run_step5_dsa_nvtx_ut.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "print_case_timing"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "case-cli", "exec"),
        namespace,
    )
    namespace["print_case_timing"](
        {
            "case_id": "failed-score",
            "op": "score",
            "scope": "op",
            "status": "failed",
            "results": [
                {
                    "arms": [
                        {
                            "backend": "direct_resident",
                            "status": "correctness_failed",
                            "timing": {"median_us": 123.4567, "time_domain": "device"},
                        }
                    ]
                }
            ],
        },
        ["direct_resident"],
    )
    assert capsys.readouterr().out.strip() == (
        "case_id=failed-score op=score scope=op backend=direct_resident "
        "median_us=123.457 status=correctness_failed "
        "mfu_pct=N/A logical_bandwidth_utilization_pct=N/A "
        "case_peak_allocated_capacity_pct=N/A device_memory_used_pct=N/A"
    )


@pytest.mark.parametrize("dtype", ["fp32", "tf32", "bf16", "fp16", "fp8"])
def test_case_utilization_uses_explicit_peak_precision_and_decimal_units(bench, dtype):
    peaks = bench.utilization_peaks(
        SimpleNamespace(
            peak_compute_tflops=2.0,
            peak_compute_dtype=dtype,
            peak_hbm_gbps=200.0,
        )
    )
    metrics = bench.case_utilization(
        {"flops": 10**9, "bytes": 10**8},
        {"median_us": 1000.0, "range_pct": 0.0},
        peaks,
    )
    assert metrics["mfu_pct"] == pytest.approx(50)
    assert metrics["logical_bandwidth_utilization_pct"] == pytest.approx(50)
    assert metrics["peaks"]["compute_dtype"] == dtype
    assert metrics["scope"] == "case-effective, not end-to-end model MFU"
    assert metrics["warnings"] == [] and metrics["timing_stable"]


@pytest.mark.parametrize("flops", [0, 10**9])
def test_missing_peaks_and_topk_never_fabricate_mfu(bench, flops):
    metrics = bench.case_utilization(
        {"flops": flops, "bytes": 10**8},
        {"median_us": 1000.0, "range_pct": 0.0},
        bench.utilization_peaks(SimpleNamespace()),
    )
    assert metrics["mfu_pct"] is None
    assert metrics["logical_bandwidth_utilization_pct"] is None
    assert metrics["mfu_reason"] == (
        "compute-peak-not-configured" if flops else "no-floating-point-work-accounting"
    )
    assert metrics["bandwidth_reason"] == "hbm-peak-not-configured"
    json.dumps(metrics, allow_nan=False)


def test_above_peak_utilization_remains_visible_and_warns(bench):
    metrics = bench.case_utilization(
        {"flops": 10**9, "bytes": 10**8},
        {"median_us": 1000.0, "range_pct": 6.0},
        bench.utilization_peaks(
            SimpleNamespace(
                peak_compute_tflops=0.5,
                peak_compute_dtype="bf16",
                peak_hbm_gbps=50,
            )
        ),
    )
    assert metrics["mfu_pct"] == metrics["logical_bandwidth_utilization_pct"] == 200
    assert len(metrics["warnings"]) == 2 and not metrics["timing_stable"]


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_utilization_rejects_unmeasured_or_invalid_latency(bench, value):
    with pytest.raises(ValueError, match="latency"):
        bench.case_utilization(
            {"bytes": 1},
            {"median_us": value},
            bench.utilization_peaks(SimpleNamespace()),
        )


@pytest.mark.parametrize("flag", ["--peak-compute-tflops", "--peak-hbm-gbps"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_peak_cli_rejects_invalid_denominators_before_device_execution(
    bench, flag, value
):
    with pytest.raises(SystemExit) as error:
        bench.parser().parse_args(
            [
                "compare",
                "--nvtx",
                "unused.txt",
                "--output-dir",
                "unused-output",
                flag + "=" + value,
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "option", ["--peak-compute-tflops=2", "--peak-compute-dtype=fp8"]
)
def test_compute_peak_requires_an_explicit_precision_pair(bench, option):
    args = bench.parser().parse_args(
        [
            "compare",
            "--nvtx",
            "unused.txt",
            "--output-dir",
            "unused-output",
            option,
        ]
    )
    with pytest.raises(ValueError, match="together"):
        bench.utilization_peaks(args)


@pytest.mark.parametrize("snapshot", [True, False])
def test_case_memory_distinguishes_allocator_peaks_from_driver_snapshot(
    bench, snapshot
):
    def mem_get_info(device):
        assert device == 0
        if not snapshot:
            raise RuntimeError("driver does not expose memory counters")
        return 3000, 10000

    cuda = SimpleNamespace(
        max_memory_allocated=lambda: 500,
        max_memory_reserved=lambda: 1000,
        mem_get_info=mem_get_info,
    )
    memory = bench.case_memory(SimpleNamespace(cuda=cuda), 2000, 10000)
    assert memory["peak_allocated_capacity_pct"] == 5
    assert memory["peak_reserved_capacity_pct"] == 10
    assert memory["peak_allocator_budget_util_pct"] == 25
    assert memory["device_memory_used_pct"] == (70 if snapshot else None)
    assert memory["device_used_bytes"] == (7000 if snapshot else None)
    assert "all backends" in memory["scope"]
    assert "not a sampled high-water mark" in memory["device_memory_basis"]
    if not snapshot:
        assert "unavailable" in memory["device_memory_reason"]
    json.dumps(memory, allow_nan=False)


@pytest.mark.parametrize("busy_at", [None, 1, 2, 7])
def test_benchmark_cli_writes_case_utilization_and_memory_to_json_csv(
    bench, monkeypatch, tmp_path, capsys, busy_at
):
    import step5_sparse_replay as replay
    import step5_sparse_support as support

    source, output = tmp_path / "input.txt", tmp_path / "result"
    source.write_text(topk_case()["raw_label"] + "\n")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps({"default": {"policy": "capacity-causal"}}))
    cuda = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(total_memory=10000),
        get_device_name=lambda device: "PPU-test",
        set_per_process_memory_fraction=lambda *a: None,
        reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None,
        empty_cache=lambda: None,
        max_memory_allocated=lambda: 500,
        max_memory_reserved=lambda: 1000,
        mem_get_info=lambda device: (3000, 10000),
    )
    monkeypatch.setattr(
        replay, "initialize_runtime", lambda root: SimpleNamespace(cuda=cuda)
    )
    gate_calls = 0

    def idle_gate():
        nonlocal gate_calls
        gate_calls += 1
        if busy_at is not None and gate_calls >= busy_at:
            raise support.DeviceBusyError("busy", {"pid": 123})

    monkeypatch.setattr(support, "idle_gate", idle_gate)
    monkeypatch.setattr(bench, "provenance", lambda torch: {"device": "PPU-test"})

    def compare(case, dry, args, torch, gate):
        for _ in range(5):
            gate()
        timing = {"median_us": 1000.0, "range_pct": 0.0}
        work = {"bytes": 10**8, "flops": 10**9}
        return {
            **dry,
            "status": "completed",
            "arms": [
                {
                    "backend": "per_row",
                    "status": "stable",
                    "timing": timing,
                    "utilization": bench.case_utilization(
                        work, timing, bench.utilization_peaks(args)
                    ),
                }
            ],
        }

    monkeypatch.setattr(bench, "compare_case", compare)
    for name in ("torch", "vllm_sail.profiling.step5_sparse"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert (
        bench.main(
            [
                "compare",
                "--nvtx",
                str(source),
                "--output-dir",
                str(output),
                "--workload-config",
                str(workload),
                "--backends",
                "per_row",
                "--allow-busy-device",
                "--peak-compute-tflops",
                "2",
                "--peak-compute-dtype",
                "bf16",
                "--peak-hbm-gbps",
                "200",
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert gate_calls == 1
    assert "performance_may_be_inaccurate" not in report
    assert "device_busy_warnings" not in report
    assert capsys.readouterr().err.count(
        "performance measurements may be inaccurate"
    ) == int(busy_at == 1)
    case = report["results"][0]
    assert case["arms"][0]["utilization"]["mfu_pct"] == 50
    assert case["memory"]["peak_allocated_capacity_pct"] == 5
    with (output / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["mfu_pct"] == "50.0"
    assert rows[0]["logical_bandwidth_utilization_pct"] == "50.0"
    assert rows[0]["case_device_memory_used_pct"] == "70.0"
    assert (
        json.loads(rows[0]["utilization_json"])["peaks"] == report["utilization_peaks"]
    )
    assert json.loads(rows[0]["memory_json"]) == case["memory"]


@pytest.mark.parametrize("allow_busy", [False, True])
@pytest.mark.parametrize("failure", ["busy", "query", "value"])
def test_startup_device_occupancy_check_only_downgrades_busy(
    bench, monkeypatch, capsys, allow_busy, failure
):
    import step5_sparse_support as support

    error = {
        "busy": support.DeviceBusyError("busy", {"pid": 123}),
        "query": subprocess.CalledProcessError(1, ["nvidia-smi"]),
        "value": ValueError("invalid UUID"),
    }[failure]

    def idle_gate():
        raise error

    monkeypatch.setattr(support, "idle_gate", idle_gate)
    if allow_busy and failure == "busy":
        assert bench.check_device_occupancy(allow_busy) == {"pid": 123}
        assert (
            capsys.readouterr().err.count("performance measurements may be inaccurate")
            == 1
        )
    else:
        with pytest.raises(type(error)) as caught:
            bench.check_device_occupancy(allow_busy)
        assert caught.value is error
        assert not capsys.readouterr().err
    assert (
        not bench.parser()
        .parse_args(["compare", "--nvtx", "input", "--output-dir", "output"])
        .allow_busy_device
    )


@pytest.mark.parametrize("failure", [None, "correctness", "execution"])
def test_nvtx_runner_continues_busy_ut_but_preserves_failures(
    bench, monkeypatch, tmp_path, capsys, failure
):
    import importlib.metadata as metadata
    import runpy

    import step5_sparse_support as support

    source = tmp_path / "input.txt"
    label = topk_case()["raw_label"]
    source.write_text(label + "\n" + label.replace("topk:4", "topk:2") + "\n")
    extension = Path("vllm_sail/_C.so")
    (tmp_path / extension).parent.mkdir()
    (tmp_path / extension).write_bytes(b"")
    distribution = SimpleNamespace(
        files=[extension], locate_file=lambda file: tmp_path / file
    )
    monkeypatch.setattr(metadata, "version", lambda name: "0.33.1")
    monkeypatch.setattr(metadata, "distributions", lambda **kw: [distribution])
    monkeypatch.setitem(sys.modules, "benchmark_step5_sparse_nvtx", bench)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(NVTX_UT / "run_step5_dsa_nvtx_ut.py"),
            "--nvtx",
            str(source),
            "--op",
            "topk",
            "--backends",
            "per_row",
            "--output",
            str(tmp_path / "output"),
        ],
    )
    gate_calls = 0

    def idle_gate():
        nonlocal gate_calls
        gate_calls += 1
        raise support.DeviceBusyError("busy", {"pid": 123})

    monkeypatch.setattr(support, "idle_gate", idle_gate)

    def execute(command, **kwargs):
        assert gate_calls == 1
        args = bench.parser().parse_args(command[2:])
        assert args.allow_busy_device
        status = "completed" if failure is None else "failed"
        bench.save(
            args.output_dir / f"{args.output_prefix}-report.json",
            {
                "results": [
                    {
                        "case_id": args.case_id[0],
                        "status": status,
                        "arms": [
                            {
                                "backend": "per_row",
                                "status": (
                                    "stable" if failure is None else f"{failure}_failed"
                                ),
                            }
                        ],
                    }
                ],
            },
        )
        return SimpleNamespace(returncode=int(failure is not None))

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(NVTX_UT / "run_step5_dsa_nvtx_ut.py"), run_name="__main__")
    assert caught.value.code == int(failure is not None)
    summaries = list((tmp_path / "output").glob("*/summary.json"))
    assert len(summaries) == 1
    entries = json.loads(summaries[0].read_text())
    assert len(entries) == 2 and gate_calls == 1
    for entry in entries:
        assert entry["status"] == ("completed" if failure is None else "failed")
        assert "performance_may_be_inaccurate" not in entry
        assert "device_busy_warnings" not in entry
    text = capsys.readouterr()
    assert text.err.count("performance measurements may be inaccurate") == 1
    assert "performance_may_be_inaccurate" not in text.out


@pytest.mark.parametrize("measured", [False, True])
def test_runner_summary_preserves_and_prints_utilization(
    bench, tmp_path, capsys, measured
):
    tree = ast.parse((NVTX_UT / "run_step5_dsa_nvtx_ut.py").read_text())
    loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [item.id for item in node.target.elts] == ["case", "admission"]
    )
    read_report = next(
        node
        for node in loop.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "exists"
    )
    printer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "print_case_timing"
    )
    utilization = {
        "mfu_pct": 50.0 if measured else None,
        "logical_bandwidth_utilization_pct": 20.0 if measured else None,
    }
    arm = {
        "backend": "per_row",
        "status": "stable",
        "timing": {"median_us": 1000.0},
        "utilization": utilization,
        "work": {"bytes": 1, "flops": 0},
    }
    memory = {"peak_allocated_capacity_pct": 5.0, "device_memory_used_pct": 70.0}
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "case_id": "case",
                        "status": "completed",
                        "memory": memory,
                        "arms": [arm],
                    }
                ]
            }
        )
    )
    entry = {
        "case_id": "case",
        "op": "topk",
        "scope": "op",
        "status": "completed",
        "results": [],
    }
    namespace = {"json": json, "sys": sys, "report_path": path, "entry": entry}
    exec(
        compile(
            ast.Module(body=[read_report, printer], type_ignores=[]),
            "runner-summary",
            "exec",
        ),
        namespace,
    )
    actual = entry["results"][0]
    assert actual["arms"][0]["utilization"] == utilization
    assert actual["arms"][0]["work"] == arm["work"] and actual["memory"] == memory
    namespace["print_case_timing"](entry, ["per_row"])
    text = capsys.readouterr().out
    assert ("mfu_pct=50.000" if measured else "mfu_pct=N/A") in text
    assert "case_peak_allocated_capacity_pct=5.000" in text
    assert "device_memory_used_pct=70.000" in text


def test_derived_workspace_uses_the_same_budget(bench):
    import step5_sparse_replay as replay

    views = replay.Views(
        SimpleNamespace(empty=lambda *a, **k: "allocation"), topk_case(), 16
    )
    dtype = SimpleNamespace(itemsize=4)
    assert views.workspace([4], dtype) == "allocation" and views.allocated == 16
    with pytest.raises(replay.Unsupported, match="workspace"):
        views.workspace([1], dtype)


def test_ragged_actual_inputs_roundtrip():
    case = parse_label(
        make_label(
            "attention",
            "flashinfer.run",
            "flashinfer_ragged",
            dict(q=Tensor(), k=Tensor(), v=Tensor(), out=None),
        )
    )
    assert set(case["tensors"]) == {"query", "key", "value", "output"}


@pytest.mark.parametrize("name", ["paged_kv_cache", "kv_cache"])
def test_paged_actual_key_value_views_roundtrip(name):
    key, value = Tensor(offset=0), Tensor(offset=8)
    case = parse_label(
        make_label(
            "attention",
            "flashinfer.run",
            "flashinfer_paged",
            {"q": Tensor(), name: (key, value), "out": None},
        )
    )
    assert set(case["tensors"]) == {"query", "key", "value", "output"}
    assert case["tensors"]["key"]["offset"] == 0
    assert case["tensors"]["value"]["offset"] == 8


def native_submission_namespace():
    tree = ast.parse((ROOT / "vllm_sail/models/step5/ops/dsa_native.py").read_text())
    submit = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_submit_attention"
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "NativePrefill"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("prepare_chunk", "finish", "_submit_batched")
    ]
    namespace = {}
    exec(
        compile(
            ast.Module(body=[submit, *methods], type_ignores=[]),
            "native_submission",
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("enabled", [False, True])
def test_native_submission_gate_and_exception(sparse, monkeypatch, enabled):
    module, events = sparse
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1" if enabled else "0")
    submit = native_submission_namespace()["_submit_attention"]

    def fail(*args):
        raise RuntimeError("original native failure")

    values = (Tensor(), None, None, None, None, None, None, 0.1, 1, True, None)
    with pytest.raises(RuntimeError, match="original native failure"):
        submit(fail, "decode", *values)
    ranges = [e for e in events if isinstance(e, dict)]
    assert [e["scope"] for e in ranges] == (["op", "kernel"] if enabled else [])
    assert all(
        e["backend"] == "native_20261110" and e["pd_type"] == "decode" for e in ranges
    )
    assert module._RANGES.get() == ()
    if enabled:
        assert events[-2:] == ["pop", "pop"]


def test_native_batch_selection_copy_is_not_attention_submission(sparse, monkeypatch):
    _, events = sparse
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    copies, calls = [], []

    class View(Tensor):
        def __init__(self, shape):
            stride, step = [], 1
            for size in reversed(shape):
                stride.insert(0, step)
                step *= size
            super().__init__(shape, tuple(stride), 0)

        def __getitem__(self, index):
            return self

        def copy_(self, other):
            copies.append(other)

        def data_ptr(self):
            return 128

    namespace = native_submission_namespace()
    namespace.update(
        common_reason=lambda *a: None,
        prefill_reason=lambda *a: None,
        compiled_op=lambda kind: lambda *a: calls.append(a),
        record_submission=lambda kind: None,
    )
    Plan = type(
        "Plan",
        (),
        {
            name: namespace[name]
            for name in ("prepare_chunk", "finish", "_submit_batched")
        },
    )
    q, cache, selected = View((2, 8, 192)), View((4, 1, 64, 384)), View((2, 4))
    table, owners, positions, out, tiles = (
        View((1, 2)),
        View((2,)),
        View((2,)),
        View(q.shape),
        View((1, 3)),
    )
    plan = Plan()
    plan.query_shape, plan.region_count, plan.scale, plan.query_tile = (
        q.shape,
        4,
        0.1,
        1,
    )
    plan.positions, plan.batch_attention, plan._finished, plan._next_start = (
        positions,
        True,
        False,
        0,
    )
    plan._same_view = lambda *a: True
    plan._bound = (out, {0: (2, None)}, 16)
    plan._batch = (q, cache, table, owners, selected, tiles)
    plan.prepare_chunk(0, q, cache, selected, table, owners)
    assert copies == [selected] and not calls and not events
    plan.finish(q, cache, out)
    assert len(calls) == 1 and plan._finished
    ranges = [e for e in events if isinstance(e, dict)]
    assert [e["scope"] for e in ranges] == ["op", "kernel"]
    assert ranges[1]["tensors"]["query"]["shape"] == [2, 8, 192]
    assert events[-2:] == ["pop", "pop"]


@pytest.mark.parametrize("op", ["topk", "attention"])
def test_topk_tensor_protocol_compatibility_and_distinct_files(bench, op):
    label = make_label(op, "select", "dispatch", {"scores": Tensor(), "topk": 4})
    old = label.replace(",topk_tensor:", "")
    assert parse_label(label)["id"] == parse_label(old)["id"]
    assert parse_label(old)["topk_tensor"] == ""
    a, b = [
        parse_label(label.replace("topk_tensor:", "topk_tensor:" + n))
        for n in ("a.pt", "b.pt")
    ]
    assert a["id"] != b["id"]
    local = importlib.import_module("step5_sparse_cases")
    assert local.parse_label(label) == parse_label(label)
    assert local.parse_label(a["raw_label"]) == a
    dtype_label = make_label(
        "topk",
        "deep_select_topk",
        "deep_select",
        {"scores": Tensor(), "output_idx": None, "indices_dtype": "torch.int64"},
    )
    assert parse_label(dtype_label)["scalars"]["output_index_dtype"] == "torch.int64"
    assert local.parse_label(dtype_label) == parse_label(dtype_label)
    for name in ("../a.pt", "/tmp/a.pt", "a.pt,evil:1", "a.txt"):
        with pytest.raises(ValueError):
            parse_label(label.replace("topk_tensor:", "topk_tensor:" + name))


def test_topk_tensor_dry_resolves_file_without_torch_or_synthetic_overrides(
    bench, tmp_path, caplog
):
    case = topk_case()
    case["topk_tensor"] = "input.pt"
    case["sources"] = [{"path": str(tmp_path / "trace.txt")}]
    args = SimpleNamespace(
        scope="prepared-op", max_memory_mib=2048, backends=["per_row"]
    )
    missing = bench.dry_case(case, {}, args)
    assert missing["status"] == "ready" and missing["synthetic"] is True
    assert missing["workload"]["current"] == [7, 6]
    assert missing["topk_tensor_fallback"] == {
        "filename": "input.pt",
        "reason": "file-not-found",
    }
    assert (
        "input.pt not found" in caplog.text
        and "falling back to synthetic" in caplog.text
    )
    assert "content_validation" not in missing
    assert case["topk_tensor"] == "input.pt"
    (tmp_path / "input.pt").write_bytes(b"Content is validated only during replay")
    ready = bench.dry_case(
        case, {"default": {"policy": "explicit", "current": [99]}}, args
    )
    assert ready["status"] == "ready" and ready["synthetic"] is False
    assert "current" not in ready["workload"]
    assert ready["workload"]["policy"] == "tensor-dump"
    assert ready["content_validation"] == "deferred-until-torch-load"
    args.scope = "kernel"
    assert bench.dry_case(case, {}, args)["status"] == "unsupported"


def test_topk_tensor_directory_is_explicit_and_ambiguous_sources_fail(bench, tmp_path):
    replay = importlib.import_module("step5_sparse_replay")
    case = topk_case()
    case["topk_tensor"] = "input.pt"
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    for folder in (first, second):
        (folder / "input.pt").write_bytes(b"dummy")
    case["sources"] = [{"path": str(p / "trace.txt")} for p in (first, second)]
    with pytest.raises(replay.Unsupported, match="expected one existing file"):
        replay.resolve_topk_tensor(case)
    assert replay.resolve_topk_tensor(case, second) == second / "input.pt"
    assert replay.resolve_topk_tensor(case, tmp_path / "missing-directory") is None


@pytest.mark.parametrize("stage", ["dry", "replay", "load"])
def test_missing_topk_tensor_prepares_synthetic_arms_and_keeps_case_identity(
    bench, monkeypatch, tmp_path, caplog, stage
):
    replay = importlib.import_module("step5_sparse_replay")
    case = topk_case()
    case["topk_tensor"] = "missing.pt"
    case["sources"] = [{"path": str(tmp_path / "trace.txt")}]
    original = json.loads(json.dumps(case))
    args = SimpleNamespace(
        scope="prepared-op", max_memory_mib=2048, seed=0, backends=["per_row"]
    )
    config = {
        "default": {"policy": "capacity-causal"},
        "cases": {case["id"]: {"current": [4, 3], "pattern": "ties"}},
    }
    if stage == "dry":
        entry = bench.dry_case(case, config, args)
    else:
        path = tmp_path / "missing.pt"
        path.write_bytes(b"Validating content is not part of dry-run")
        entry = bench.dry_case(case, config, args)
        assert entry["synthetic"] is False
        # Simulate removal without mutating the recorded source or case identity.
        monkeypatch.setattr(
            replay,
            "resolve_topk_tensor",
            lambda *a: None if stage == "replay" else tmp_path / "gone.pt",
        )
    captured = []
    monkeypatch.setattr(
        replay, "Views", lambda torch, case, budget, inputs: captured.append(inputs)
    )

    def make_arm(torch, case, geo, work, backend, views, args):
        assert (
            work["synthetic"]
            and work["current"] == [4, 3]
            and work["pattern"] == "ties"
        )
        assert "topk_tensor_path" not in work and "topk_tensor_sha256" not in work
        return replay.Arm(
            backend,
            "synthetic.selector",
            {},
            lambda: None,
            lambda: {"status": "passed"},
            {},
            "topk-complete-stable",
        )

    monkeypatch.setattr(replay, "_topk", make_arm)
    arms, blocked = replay.prepare_arms(
        SimpleNamespace(manual_seed=lambda _: None),
        case,
        entry["geometry"],
        entry["workload"],
        args.backends,
        args,
    )
    assert not blocked and len(arms) == 1 and captured == [None]
    assert (
        arms[0].config["normalized_workload"]["topk_tensor_fallback"]["reason"]
        == "file-not-found"
    )
    assert case == original
    if stage != "replay":
        assert "falling back to synthetic" in caplog.text


class DumpInput(Tensor):
    device = "cpu"

    def __init__(self, data, shape, dtype):
        stride, step = [], 1
        for size in reversed(shape):
            stride.insert(0, step)
            step *= size
        super().__init__(tuple(shape), tuple(stride), 0)
        self.data, self.dtype = data, dtype

    def reshape(self, *shape):
        return self

    def tolist(self):
        return self.data

    def numel(self):
        return len(self.data)


def test_topk_tensor_loader_validates_inputs_and_layout_without_device(bench, tmp_path):
    replay = importlib.import_module("step5_sparse_replay")
    case = topk_case()
    case["topk_tensor"] = "input.pt"
    path = tmp_path / "input.pt"
    path.write_bytes(b"dump")
    record = {k: case[k] for k in ("impl", "backend", "scope", "pd_type")}
    record.update(
        schema=1,
        scalars={"topk": 4},
        metadata=case["tensors"],
        inputs={
            "scores": DumpInput([[1] * 8] * 2, [2, 8], "torch.float32"),
            "current": DumpInput([3, 1], [2], "torch.int64"),
        },
    )
    calls = []

    def load(stream, **kwargs):
        calls.append(kwargs)
        return record

    torch = SimpleNamespace(Tensor=DumpInput, load=load)
    inputs, digest = replay.load_topk_tensor(torch, case, path)
    assert len(digest) == 64 and inputs is record["inputs"]
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    geo = bench.geometry(case, {})
    work = replay.tensor_workload(case, geo, {"policy": "tensor-dump"}, inputs)
    assert work["current"] == [3, 1] and work["topk_lengths"] == [3, 1]
    assert work["synthetic"] is False
    record["inputs"]["scores"].dtype = "torch.bfloat16"
    with pytest.raises(replay.Unsupported, match="Invalid TopK input tensor"):
        replay.load_topk_tensor(torch, case, path)
    record["inputs"]["scores"].dtype = "torch.float32"
    record["inputs"].pop("current")
    with pytest.raises(replay.Unsupported, match="Missing TopK input tensor"):
        replay.load_topk_tensor(torch, case, path)


def test_recorded_raw_lengths_preserve_next_n_expansion(bench):
    replay = importlib.import_module("step5_sparse_replay")
    case = topk_case()
    case["scalars"]["next_n"] = 2
    inputs = {"lengths": DumpInput([5], [1], "torch.int32")}
    work = replay.tensor_workload(case, bench.geometry(case, {}), {}, inputs)
    assert work["current"] == [4, 5] and work["topk_lengths"] == [4, 5]


def attention_case(kind):
    def tensor(shape, stride, dtype):
        value = Tensor(shape, stride, 0)
        value.dtype = dtype
        return value

    values = dict(
        query=tensor((2, 8, 192), (1536, 192, 1), "torch.bfloat16"),
        kv_cache=tensor((2, 1, 16, 384), (6144, 6144, 384, 1), "torch.bfloat16"),
        block_table=tensor((2, 2), (2, 1), "torch.int32"),
        indices=tensor((2, 2), (2, 1), "torch.int64"),
        positions=tensor((2,), (1,), "torch.int64"),
        scale=0.1,
    )
    name, dtype = (
        ("active", "torch.bool") if kind == "decode" else ("owners", "torch.int64")
    )
    values[name] = tensor((2,), (1,), dtype)
    case = parse_label(
        make_label(
            "attention",
            "step5_dsa_native_" + kind,
            "native_20261110",
            values,
            pd_type=kind,
            topk_tensor="attention-input.pt",
        )
    )
    case["sources"] = []
    return case


@pytest.mark.parametrize("kind", ["prefill", "decode"])
@pytest.mark.parametrize("stage", ["dry", "replay", "load"])
def test_missing_attention_dump_keeps_identity_and_configured_fallback(
    bench, monkeypatch, tmp_path, caplog, kind, stage
):
    replay = importlib.import_module("step5_sparse_replay")
    case = attention_case(kind)
    case["sources"] = [{"path": str(tmp_path / "trace.txt")}]
    original = json.loads(json.dumps(case))
    config = {"default": {"policy": "capacity-causal", "positions": [15, 8]}}
    args = SimpleNamespace(
        scope="prepared-op", max_memory_mib=2048, seed=0, backends=["native"]
    )
    if stage != "dry":
        (tmp_path / case["topk_tensor"]).write_bytes(b"dump")
    entry = bench.dry_case(case, config, args)
    assert entry["status"] == "ready"
    if stage != "dry":
        monkeypatch.setattr(
            replay,
            "resolve_topk_tensor",
            lambda *a: None if stage == "replay" else tmp_path / "gone.pt",
        )
    monkeypatch.setattr(replay, "Views", lambda *a: None)

    def make_arm(torch, case, geo, work, backend, views, args):
        assert work["synthetic"] and work["positions"] == [15, 8]
        return replay.Arm(
            backend,
            "synthetic.attention",
            {},
            lambda: None,
            lambda: {},
            {},
            "attention-fixed-selection",
        )

    monkeypatch.setattr(replay, "_attention", make_arm)
    arms, blocked = replay.prepare_arms(
        SimpleNamespace(manual_seed=lambda _: None),
        case,
        entry["geometry"],
        entry["workload"],
        ["native"],
        args,
    )
    assert not blocked and len(arms) == 1
    assert arms[0].config["normalized_workload"]["topk_tensor_fallback"] == {
        "filename": "attention-input.pt",
        "reason": "file-not-found",
    }
    assert case == original
    if stage != "replay":
        assert "falling back to synthetic" in caplog.text


@pytest.mark.parametrize("kind", ["prefill", "decode"])
def test_attention_loader_requires_actual_indices_and_matching_op(
    bench, tmp_path, kind
):
    replay = importlib.import_module("step5_sparse_replay")
    case = attention_case(kind)
    record = {name: case[name] for name in ("impl", "backend", "scope", "pd_type")}
    record.update(
        schema=1,
        op="attention",
        metadata=case["tensors"],
        scalars={"scale": 0.1},
        inputs={
            name: DumpInput([], item["shape"], item["dtype"])
            for name, item in case["tensors"].items()
        },
    )
    path = tmp_path / "attention.pt"
    path.write_bytes(b"dump")
    torch = SimpleNamespace(Tensor=DumpInput, load=lambda *a, **kw: record)
    inputs, _ = replay.load_topk_tensor(torch, case, path)
    assert "indices" in inputs
    record["op"] = "topk"
    with pytest.raises(replay.Unsupported, match="mismatch: op"):
        replay.load_topk_tensor(torch, case, path)
    record["op"] = "attention"
    record["inputs"].pop("indices")
    with pytest.raises(
        replay.Unsupported, match="Missing Attention input tensor: indices"
    ):
        replay.load_topk_tensor(torch, case, path)
    record["parameters"] = {"prefill_chunks": []}
    with pytest.raises(
        replay.Unsupported, match="prefill_chunks are saved but not replayable"
    ):
        replay.load_topk_tensor(torch, case, path)


def test_attention_workload_uses_recorded_quantization_scales(bench):
    replay = importlib.import_module("step5_sparse_replay")
    case = attention_case("decode")
    case["tensors"]["kv_cache"]["dtype"] = "torch.uint8"
    inputs = {
        "block_table": DumpInput([[1, 0], [0, 1]], [2, 2], "torch.int32"),
        "positions": DumpInput([15, 8], [2], "torch.int64"),
        "active": DumpInput([True, True], [2], "torch.bool"),
        "indices": DumpInput([[0, 1], [-1, 1]], [2, 2], "torch.int64"),
        "k_scale": DumpInput([0.25], [1], "torch.float32"),
        "v_scale": DumpInput([0.5], [1], "torch.float32"),
    }
    work = replay.tensor_workload(
        case, bench.geometry(case, {}), {"k_scale": 99}, inputs
    )
    assert work["k_scale"] == 0.25 and work["v_scale"] == 0.5
    inputs["k_scale"].data = [float("nan")]
    with pytest.raises(replay.Unsupported, match="Invalid recorded k_scale"):
        replay.tensor_workload(case, bench.geometry(case, {}), {}, inputs)


def test_raw_attention_split_and_combine_are_not_replaced_with_native(bench):
    for impl in ("_sparse_decode_split", "_sparse_decode_combine"):
        case = attention_case("decode")
        case.update(impl=impl, scope="kernel")
        args = SimpleNamespace(scope="kernel", max_memory_mib=2048, backends=["native"])
        result = bench.dry_case(case, {}, args)
        assert result["status"] == "unsupported" and "split/combine" in result["reason"]


@pytest.mark.parametrize("missing_tensor", [False, True])
@pytest.mark.parametrize("scope", ["prepared-op", "kernel"])
def test_real_torch_topk_dump_load_and_replay_uses_values_and_layout(
    bench, monkeypatch, tmp_path, scope, missing_tensor
):
    torch = pytest.importorskip("torch")
    replay = importlib.import_module("step5_sparse_replay")
    scores = torch.arange(32, dtype=torch.float32).as_strided((2, 8), (12, 1), 3)
    values = {"scores": scores, "topk": 2}
    if scope == "prepared-op":
        values["current"] = torch.tensor([3, 1], dtype=torch.int64)
    else:
        values.update(
            lengths=torch.tensor([5], dtype=torch.int32),
            next_n=2,
            max_seq_len=8,
            indices=torch.empty((2, 2), dtype=torch.int32),
        )
    label = make_label(
        "topk",
        "select",
        "per_row",
        values,
        scope="op" if scope == "prepared-op" else "kernel",
        topk_tensor="real.pt",
    )
    case = parse_label(label)
    case["sources"] = [{"path": str(tmp_path / "trace.txt")}]
    record = {k: case[k] for k in ("impl", "backend", "scope", "pd_type")}
    record.update(
        schema=1,
        metadata=case["tensors"],
        scalars={k: v for k, v in case["scalars"].items() if k != "capture"},
        inputs={
            k: v.detach().clone()
            for k, v in values.items()
            if isinstance(v, torch.Tensor) and k != "indices"
        },
    )
    if not missing_tensor:
        torch.save(record, tmp_path / "real.pt")

    class CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *a, **kw):
            kw["device"] = "cpu"
            return torch.empty(*a, **kw)

        def tensor(self, *a, **kw):
            kw["device"] = "cpu"
            return torch.tensor(*a, **kw)

        def randn(self, *a, **kw):
            kw["device"] = "cpu"
            return torch.randn(*a, **kw)

        def arange(self, *a, **kw):
            kw["device"] = "cpu"
            return torch.arange(*a, **kw)

    calls = []
    indexer = types.ModuleType("vllm.models.step5.nvidia.ops.sparse_indexer")

    def raw(s, lengths, n, output, k, width):
        calls.append((s.clone(), s.stride(), s.storage_offset(), lengths.tolist(), n))
        ends = (
            lengths.tolist()
            if lengths.numel() == 2
            else [lengths[0].item() - n + 1 + i for i in range(n)]
        )
        output.fill_(-1)
        for row, end in enumerate(ends):
            chosen = torch.topk(s[row, :end], min(k, end)).indices
            output[row, : len(chosen)] = chosen

    def complete(s, current, k, backend, **kwargs):
        calls.append((s.clone(), s.stride(), s.storage_offset(), current.tolist(), 1))
        return torch.tensor(
            [
                [-1] * (k - min(k, c))
                + sorted(torch.topk(s[i, :c], min(k, c)).indices.tolist())
                + [c]
                for i, c in enumerate(current.tolist())
            ],
            dtype=torch.int64,
        )

    indexer.get_indexer_topk = lambda backend: raw
    indexer.select_regions = complete
    for name in (
        "vllm",
        "vllm.models",
        "vllm.models.step5",
        "vllm.models.step5.nvidia",
        "vllm.models.step5.nvidia.ops",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, indexer.__name__, indexer)
    monkeypatch.setattr(replay, "_original", lambda call, target: call)
    args = SimpleNamespace(
        scope=scope, seed=0, max_memory_mib=2048, topk_tensor_dir=None
    )
    geo = bench.geometry(case, {})
    arms, blocked = replay.prepare_arms(
        CpuTorch(), case, geo, {"policy": "tensor-dump"}, ["per_row"], args
    )
    assert not blocked and len(arms) == 1
    arms[0].call()
    assert arms[0].check()["status"] == "passed"
    assert torch.equal(calls[0][0], scores) == (not missing_tensor)
    assert calls[0][1:3] == ((12, 1), 3)
    assert calls[0][3:] == (
        (([7, 6], 1) if scope == "prepared-op" else ([8], 2))
        if missing_tensor
        else (([3, 1], 1) if scope == "prepared-op" else ([5], 2))
    )
    work = arms[0].config["normalized_workload"]
    assert work["synthetic"] is missing_tensor
    if missing_tensor:
        assert work["topk_tensor_fallback"]["filename"] == "real.pt"
        assert "topk_tensor_sha256" not in work
    else:
        assert len(work["topk_tensor_sha256"]) == 64
    (tmp_path / "real.pt").write_bytes(b"corrupt")
    with pytest.raises(replay.Unsupported, match="Cannot load topk_tensor"):
        replay.prepare_arms(
            CpuTorch(), case, geo, {"policy": "tensor-dump"}, ["per_row"], args
        )


@pytest.mark.parametrize("missing_tensor", [False, True])
@pytest.mark.parametrize(
    "kind",
    ["select-all", "zero-capacity", "paged", "raw-selector", "deep-end", "deep-no-end"],
)
def test_real_torch_special_topk_inputs_replay_without_synthetic_content(
    bench, monkeypatch, tmp_path, kind, missing_tensor
):
    torch = pytest.importorskip("torch")
    replay = importlib.import_module("step5_sparse_replay")

    class CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *a, **kw):
            kw["device"] = "cpu"
            return torch.empty(*a, **kw)

        def tensor(self, *a, **kw):
            kw["device"] = "cpu"
            return torch.tensor(*a, **kw)

        def randn(self, *a, **kw):
            kw["device"] = "cpu"
            return torch.randn(*a, **kw)

        def arange(self, *a, **kw):
            kw["device"] = "cpu"
            return torch.arange(*a, **kw)

    for name in (
        "vllm",
        "vllm.models",
        "vllm.models.step5",
        "vllm.models.step5.nvidia",
        "vllm.models.step5.nvidia.ops",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    indexer = types.ModuleType("vllm.models.step5.nvidia.ops.sparse_indexer")
    monkeypatch.setitem(sys.modules, indexer.__name__, indexer)
    monkeypatch.setattr(replay, "_original", lambda call, target: call)
    selector = types.ModuleType("vllm_sail.models.step5.ops.selector_v2")
    monkeypatch.setitem(sys.modules, selector.__name__, selector)
    ops = importlib.import_module("vllm_sail.models.step5.ops")
    monkeypatch.setattr(ops, "selector_v2", selector, raising=False)
    calls = []
    values = {
        "scores": torch.tensor([[0.0, 4.0, 1.0, 3.0, 2.0, 6.0, 5.0, 7.0]]),
        "topk": 2,
    }
    impl, scope, backend = "select_regions", "op", "per_row"
    excluded = {"indices", "workspace", "starts", "pages"}
    if kind in ("select-all", "zero-capacity"):
        k = 2 if kind == "select-all" else 0
        values = {"current": torch.tensor([-1, 0, k], dtype=torch.int64), "capacity": k}
        impl, scope = "select_all_regions", "kernel"

        def select_all(current, capacity):
            calls.append(current.clone())
            return torch.stack(
                [torch.arange(capacity + 1) - (capacity - c) for c in current]
            ).clamp(min=-1)

        indexer.select_all_regions = select_all
    elif kind == "paged":
        values.update(
            current=torch.tensor([3], dtype=torch.int64),
            topk_lengths=torch.tensor([3], dtype=torch.int32),
            owners=torch.tensor([0]),
            block_table=torch.tensor([[5, 6, 7, 8]], dtype=torch.int32),
            indptr=torch.tensor([0, 3], dtype=torch.int32),
            pages=torch.empty(3, dtype=torch.int32),
            block_size=16,
            page_stride=2,
        )
        indexer.PagedPrefillMetadata = lambda *a: SimpleNamespace(
            owners=a[0],
            block_table=a[1],
            indices=a[2],
            indptr=a[3],
            block_size=a[4],
            page_stride=a[5],
        )

        def select(s, current, k, backend, **kw):
            paged = kw["paged_prefill"]
            calls.append((s.clone(), current.clone(), paged.block_table.clone()))
            c = current.item()
            chosen = sorted(torch.topk(s[0, :c], min(k, c)).indices.tolist())
            selected = [-1] * (k - len(chosen)) + chosen + [c]
            pages = [
                paged.block_table[0, i // (paged.block_size // 8)].item()
                * paged.page_stride
                + i % (paged.block_size // 8)
                for i in selected
                if i >= 0
            ]
            paged.indices.copy_(torch.tensor(pages))
            return torch.tensor([selected], dtype=torch.int64)

        indexer.select_regions = select
        values["scores"] = torch.tensor([[4.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    elif kind == "raw-selector":
        values.update(
            scores=values["scores"].reshape(1, 1, 8),
            positions=torch.tensor([24], dtype=torch.int64),
            block_table=torch.tensor([[5, 6]], dtype=torch.int32),
            workspace=torch.empty((1, 1, 16), dtype=torch.int32),
            regions=torch.empty((1, 1, 3), dtype=torch.int32),
            starts=torch.empty((1, 1, 3), dtype=torch.int32),
            region_block=8,
            page_size=16,
            packed=True,
            threads=0,
        )
        impl, scope, backend = "raw_select", "kernel", "selector_v2"
        selector.workspace_capacity = lambda width, k: 16

        def raw(s, positions, table, workspace, regions, starts, **kw):
            calls.append((s.clone(), positions.clone(), table.clone(), kw))
            c = positions.item() // kw["region_block"]
            chosen = sorted(
                torch.topk(s[0, 0, :c], min(kw["topk"], c)).indices.tolist()
            )
            selected = [-1] * (kw["topk"] - len(chosen)) + chosen + [c]
            rpp = kw["page_size"] // kw["region_block"]
            mapped = [
                table[0, i // rpp].item() * rpp + i % rpp if i >= 0 else -1
                for i in selected
            ]
            regions.copy_(torch.tensor([[mapped]], dtype=torch.int32))
            starts.copy_(
                torch.tensor(
                    [[[i * kw["region_block"] if i >= 0 else -1 for i in selected]]],
                    dtype=torch.int32,
                )
            )

        selector.raw_select = raw
        values["scores"] = torch.tensor([[[4.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        excluded.add("regions")
    else:
        impl, scope, backend = (
            "vllm_sail.utils.deep_select.deep_select_topk",
            "kernel",
            "deep_select",
        )
        values.update(
            scores=values["scores"].to(torch.bfloat16).repeat(2, 1),
            indices_dtype=torch.int64,
        )
        if kind == "deep-end":
            values["end"] = torch.tensor([-4, 99], dtype=torch.int64)
        platform = types.ModuleType("vllm.platforms")
        platform.current_platform = SimpleNamespace(is_ppu=lambda: True)
        monkeypatch.setitem(sys.modules, platform.__name__, platform)
        deep = types.ModuleType("vllm_sail.utils.deep_select")
        deep.can_use_deep_select = lambda scores, k: True

        def topk(s, k, *, end, output_idx, indices_dtype):
            calls.append(
                (
                    s.clone(),
                    None if end is None else end.clone(),
                    output_idx,
                    indices_dtype,
                )
            )
            ends = [8, 8] if end is None else end.to(torch.int32).clamp(0, 8).tolist()
            out = torch.full((2, k), -1, dtype=indices_dtype)
            for row, count in enumerate(ends):
                selected = torch.topk(s[row, :count], min(k, count)).indices
                out[row, : len(selected)] = selected
            return out

        deep.deep_select_topk = topk
        monkeypatch.setitem(sys.modules, deep.__name__, deep)

    case = parse_label(
        make_label("topk", impl, backend, values, scope=scope, topk_tensor="special.pt")
    )
    case["sources"] = [{"path": str(tmp_path / "trace.txt")}]
    from vllm_sail.step5_sparse_cases import normalize_values

    record = {key: case[key] for key in ("impl", "backend", "scope", "pd_type")}
    record.update(
        schema=1,
        metadata=case["tensors"],
        scalars={
            key: value for key, value in case["scalars"].items() if key != "capture"
        },
        inputs={
            key: value.clone()
            for key, value in normalize_values(values).items()
            if isinstance(value, torch.Tensor) and key not in excluded
        },
    )
    if not missing_tensor:
        torch.save(record, tmp_path / "special.pt")
    args = SimpleNamespace(
        scope="kernel" if scope == "kernel" else "prepared-op",
        seed=0,
        max_memory_mib=2048,
        topk_tensor_dir=tmp_path,
    )
    work = (
        {"policy": "capacity-causal"}
        if missing_tensor
        else {"policy": "explicit", "pattern": "ties", "current": [99]}
    )
    arms, blocked = replay.prepare_arms(
        CpuTorch(), case, bench.geometry(case, {}), work, [backend], args
    )
    assert not blocked and len(arms) == 1
    arms[0].call()
    assert arms[0].check()["status"] == "passed"
    assert len(calls) == 1
    work = arms[0].config["normalized_workload"]
    assert work["synthetic"] is missing_tensor
    if missing_tensor:
        assert work["topk_tensor_fallback"]["filename"] == "special.pt"
        assert "topk_tensor_sha256" not in work
    if kind.startswith("deep"):
        assert calls[0][2] is None and calls[0][3] == torch.int64
        if kind == "deep-end":
            assert calls[0][1].dtype == torch.int64
            assert calls[0][1].tolist() == ([7, 6] if missing_tensor else [-4, 99])
        else:
            assert calls[0][1] is None
    elif kind in ("select-all", "zero-capacity"):
        assert calls[0].tolist() == (
            work["current"] if missing_tensor else values["current"].tolist()
        )
    else:
        assert torch.equal(calls[0][0], values["scores"]) == (not missing_tensor)
