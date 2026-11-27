# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in NVTX profiling subpackage (Phase-1 decision Q6).

Everything here runs on a bare CPU runner: no vLLM, no ``nvtx`` and no
``model_prof`` are required. The fork's engine/worker/scheduler NVTX hunks
became delegating ``@patch`` modules under ``vllm_sail/profiling/`` that
install nothing unless ``VLLM_PPU_NVTX_PROFILE`` is set, so the tests cover:

* gate off -> ``install()`` patches nothing;
* gate on  -> every patch lands on its (faked) upstream target;
* the shared helpers degrade to no-ops without the optional dependencies.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import json
import logging
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY, original_of

GATE_VARS = ("VLLM_SAIL_NVTX_PROFILE", "VLLM_PPU_NVTX_PROFILE", "SAIL_NVTX_PROFILE")

PROFILING_TARGETS = (
    "vllm.model_executor.layers.linear.ReplicatedLinear.forward",
    "vllm.model_executor.layers.linear.ColumnParallelLinear.forward",
    "vllm.model_executor.layers.linear.RowParallelLinear.forward",
    "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernelModularImpl.apply",
    "vllm.v1.core.sched.scheduler.Scheduler.schedule",
    "vllm.v1.core.sched.scheduler.Scheduler._update_after_schedule",
    "vllm.v1.engine.core.EngineCore.__init__",
    "vllm.v1.engine.core.EngineCore.step",
    "vllm.v1.engine.core.EngineCore.step_with_batch_queue",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner.__init__",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner._sample",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner._model_forward",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_inputs",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model",
)

PROFILING_SUBMODULES = (
    "vllm_sail.profiling.linear",
    "vllm_sail.profiling.modular_moe",
    "vllm_sail.profiling.moe",
    "vllm_sail.profiling.nvtx",
    "vllm_sail.profiling.engine_core",
    "vllm_sail.profiling.model_runner",
    "vllm_sail.profiling.model_runner_v2",
    "vllm_sail.profiling.scheduler",
)


def _sched_output(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "scheduled_new_reqs": [],
        "total_num_scheduled_tokens": 5,
        "num_scheduled_tokens": {"r1": 5},
        "scheduled_spec_decode_tokens": {},
        "finished_req_ids": set(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def cleared_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in GATE_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A minimal fake vLLM tree carrying the three patch targets."""

    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def plain(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    vllm = package("vllm")
    vllm_logger = plain("vllm.logger")
    vllm_logger.init_logger = logging.getLogger
    vllm.logger = vllm_logger

    package("vllm.v1")
    package("vllm.v1.worker")
    package("vllm.v1.worker.gpu")
    package("vllm.v1.engine")
    package("vllm.v1.core")
    package("vllm.v1.core.sched")

    class GPUModelRunner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
            return ("prepare_inputs", scheduler_output, num_scheduled_tokens)

        def _sample(self, logits, spec_decode_metadata):
            return ("sample", logits, spec_decode_metadata)

        def _model_forward(self, *args: Any, **kwargs: Any):
            return ("forward", args, kwargs)

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return ("execute_model", scheduler_output, intermediate_tensors)

    class GPUModelRunnerV2:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def prepare_inputs(
            self, scheduler_output, batch_req_state, batch_desc, num_active_loras
        ):
            return ("prepare_inputs", scheduler_output, batch_req_state)

        def sample(self, hidden_states, input_batch, grammar_output):
            return ("sample", hidden_states, input_batch)

        def execute_model(
            self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kw
        ):
            return ("execute_model", scheduler_output, dummy_run)

    class EngineCore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def step(self):
            return ("step",)

        def step_with_batch_queue(self):
            return ("step_with_batch_queue",)

    class Scheduler:
        def schedule(self, throttle_prefills: bool = False):
            return ("schedule", throttle_prefills)

        def _update_after_schedule(self, scheduler_output):
            self.last_update = scheduler_output

    package("vllm.model_executor")
    package("vllm.model_executor.layers")
    package("vllm.model_executor.layers.fused_moe")
    linear = plain("vllm.model_executor.layers.linear")
    for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
        setattr(linear, name, type(name, (), {"forward": lambda *a: None}))
    modular = plain("vllm.model_executor.layers.fused_moe.modular_kernel")
    modular.FusedMoEKernelModularImpl = type(
        "FusedMoEKernelModularImpl", (), {"apply": lambda *a: None}
    )

    runner_module = plain("vllm.v1.worker.gpu_model_runner")
    runner_module.GPUModelRunner = GPUModelRunner
    runner_v2_module = plain("vllm.v1.worker.gpu.model_runner")
    runner_v2_module.GPUModelRunner = GPUModelRunnerV2
    core_module = plain("vllm.v1.engine.core")
    core_module.EngineCore = EngineCore
    scheduler_module = plain("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = Scheduler

    return SimpleNamespace(
        GPUModelRunner=GPUModelRunner,
        GPUModelRunnerV2=GPUModelRunnerV2,
        EngineCore=EngineCore,
        Scheduler=Scheduler,
        originals={
            "GPUModelRunner.__init__": GPUModelRunner.__init__,
            "GPUModelRunner._prepare_inputs": GPUModelRunner._prepare_inputs,
            "GPUModelRunner._sample": GPUModelRunner._sample,
            "GPUModelRunner._model_forward": GPUModelRunner._model_forward,
            "GPUModelRunner.execute_model": GPUModelRunner.execute_model,
            "gpu.model_runner.GPUModelRunner.prepare_inputs": (
                GPUModelRunnerV2.prepare_inputs
            ),
            "gpu.model_runner.GPUModelRunner.sample": GPUModelRunnerV2.sample,
            "gpu.model_runner.GPUModelRunner.execute_model": (
                GPUModelRunnerV2.execute_model
            ),
            "EngineCore.__init__": EngineCore.__init__,
            "EngineCore.step": EngineCore.step,
            "EngineCore.step_with_batch_queue": EngineCore.step_with_batch_queue,
            "Scheduler.schedule": Scheduler.schedule,
            "Scheduler._update_after_schedule": Scheduler._update_after_schedule,
        },
    )


@pytest.fixture
def nvtx_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Fake `nvtx` package recording mark/annotate usage."""
    calls: dict[str, list] = {"marks": [], "annotated": []}
    stub = types.ModuleType("nvtx")

    def mark(message: str, **kwargs: Any) -> None:
        calls["marks"].append(message)

    def annotate(name: str, **kwargs: Any):
        calls["annotated"].append(name)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                return func(*args, **kw)

            return wrapper

        return decorator

    stub.mark = mark
    stub.annotate = annotate
    monkeypatch.setitem(sys.modules, "nvtx", stub)
    return calls


@pytest.fixture
def profiling_pkg(fake_vllm: SimpleNamespace, monkeypatch: pytest.MonkeyPatch):
    """Import vllm_sail.profiling fresh, with the gate flag reset."""
    module = importlib.import_module("vllm_sail.profiling")
    for name in PROFILING_SUBMODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # `from vllm_sail.profiling import engine_core, ...` resolves via these
    # attributes first; drop them too so install() re-imports the submodules
    # against this test's fakes instead of reusing a stale cached copy.
    for attr in (
        "nvtx",
        "engine_core",
        "model_runner",
        "model_runner_v2",
        "scheduler",
        "linear",
        "modular_moe",
        "moe",
    ):
        monkeypatch.delattr(module, attr, raising=False)
    # Step5 Norm instrumentation is covered by test_step5_norm_profiling; stub it
    # here so the real upstream Step5RMSNorm is not patched into PATCH_REGISTRY.
    norm = types.ModuleType("vllm_sail.profiling.step5_norm")
    norm.install = lambda: False
    monkeypatch.setitem(sys.modules, norm.__name__, norm)
    monkeypatch.setattr(module, "step5_norm", norm, raising=False)
    # Stub step5_sparse so its extra patches don't perturb the registry-count
    # gate when upstream modules are already cached in sys.modules.
    sparse = types.ModuleType("vllm_sail.profiling.step5_sparse")
    sparse.install = lambda: False
    monkeypatch.setitem(sys.modules, sparse.__name__, sparse)
    monkeypatch.setattr(module, "step5_sparse", sparse, raising=False)
    monkeypatch.setattr(module, "_installed", False)
    return module


@pytest.fixture
def installed(
    profiling_pkg,
    nvtx_stub: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    fake_vllm: SimpleNamespace,
):
    """Run install() with the gate on and return the faked targets."""
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    registry_before = len(PATCH_REGISTRY)
    profiling_pkg.install()
    return SimpleNamespace(
        pkg=profiling_pkg,
        fakes=fake_vllm,
        nvtx_calls=nvtx_stub,
        registry_before=registry_before,
    )


def _records_for(targets) -> dict[str, Any]:
    return {r.target: r for r in PATCH_REGISTRY if r.target in targets}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_gate_off_installs_nothing(
    profiling_pkg, fake_vllm: SimpleNamespace, cleared_gate: None
) -> None:
    records_before = _records_for(PROFILING_TARGETS)
    profiling_pkg.install()

    assert profiling_pkg._installed is False
    for name in PROFILING_SUBMODULES:
        assert name not in sys.modules
    for original in fake_vllm.originals.values():
        assert getattr(original, PATCH_MARKER, None) is None
    assert _records_for(PROFILING_TARGETS) == records_before


def test_gate_on_installs_every_patch(installed) -> None:
    assert installed.pkg._installed is True

    records = _records_for(PROFILING_TARGETS)
    assert set(records) == set(PROFILING_TARGETS)
    assert len(PATCH_REGISTRY) == installed.registry_before + len(PROFILING_TARGETS)

    fakes = installed.fakes
    replacements = {
        "GPUModelRunner.__init__": fakes.GPUModelRunner.__init__,
        "GPUModelRunner._prepare_inputs": fakes.GPUModelRunner._prepare_inputs,
        "GPUModelRunner._sample": fakes.GPUModelRunner._sample,
        "GPUModelRunner._model_forward": fakes.GPUModelRunner._model_forward,
        "GPUModelRunner.execute_model": fakes.GPUModelRunner.execute_model,
        "gpu.model_runner.GPUModelRunner.prepare_inputs": (
            fakes.GPUModelRunnerV2.prepare_inputs
        ),
        "gpu.model_runner.GPUModelRunner.sample": fakes.GPUModelRunnerV2.sample,
        "gpu.model_runner.GPUModelRunner.execute_model": (
            fakes.GPUModelRunnerV2.execute_model
        ),
        "EngineCore.__init__": fakes.EngineCore.__init__,
        "EngineCore.step": fakes.EngineCore.step,
        "EngineCore.step_with_batch_queue": fakes.EngineCore.step_with_batch_queue,
        "Scheduler.schedule": fakes.Scheduler.schedule,
        "Scheduler._update_after_schedule": fakes.Scheduler._update_after_schedule,
    }
    for key, replacement in replacements.items():
        assert getattr(replacement, PATCH_MARKER, None) is not None, key
        assert replacement is not fakes.originals[key], key
        target = records[[t for t in records if t.endswith(key)][0]].target
        assert original_of(replacement, target) is fakes.originals[key], key


def test_install_is_idempotent(installed) -> None:
    count = len(PATCH_REGISTRY)
    installed.pkg.install()
    installed.pkg.install()
    assert len(PATCH_REGISTRY) == count


@pytest.mark.parametrize("gate_var", GATE_VARS)
def test_gate_aliases_enable_profiling(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    gate_var: str,
) -> None:
    monkeypatch.setenv(gate_var, "1")
    profiling_pkg.install()
    assert profiling_pkg._installed is True


@pytest.mark.parametrize("value", ["0", ""])
def test_sail_false_prevents_install_with_truthy_legacy_aliases(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    value: str,
) -> None:
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", value)
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setenv("SAIL_NVTX_PROFILE", "1")
    profiling_pkg.install()
    assert profiling_pkg._installed is False


def test_missing_nvtx_package_disables_with_warning(
    profiling_pkg,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setitem(sys.modules, "nvtx", None)
    with caplog.at_level(logging.WARNING):
        profiling_pkg.install()
    assert profiling_pkg._installed is False
    assert any("nvtx" in message for message in caplog.messages)


def test_missing_model_prof_still_installs(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setitem(sys.modules, "model_prof", None)
    with caplog.at_level(logging.INFO):
        profiling_pkg.install()
    assert profiling_pkg._installed is True
    assert any(
        "without `model_prof`" in record.getMessage() for record in caplog.records
    )


def test_drifted_target_degrades_to_warning(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.delitem(sys.modules, "vllm.v1.worker.gpu_model_runner")
    with caplog.at_level(logging.WARNING):
        profiling_pkg.install()
    assert profiling_pkg._installed is False
    assert any("could not be" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# Wrapper behaviour against the faked targets
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_nvtx(installed, monkeypatch: pytest.MonkeyPatch):
    """Swap the helper backends for recorders after install imported them."""
    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    pushes: list[str] = []
    pops: list[int] = []
    iters: list[int] = []
    monkeypatch.setattr(
        prof_nvtx, "_torch_range_push", lambda label: pushes.append(label)
    )
    monkeypatch.setattr(prof_nvtx, "_torch_range_pop", lambda: pops.append(1))
    monkeypatch.setattr(prof_nvtx, "_model_prof_iter", lambda it: iters.append(it))
    return SimpleNamespace(pushes=pushes, pops=pops, iters=iters)


def test_execute_model_wraps_iteration_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunner()
    assert runner.iteration == 0

    output = _sched_output()
    result = runner.execute_model(output)

    assert result == ("execute_model", output, None)
    assert recording_nvtx.pushes == ["[prof_range]: iter 0"]
    assert len(recording_nvtx.pops) == 1
    assert recording_nvtx.iters == [0]
    assert runner.iteration == 1
    assert "total_tokens=5,req_id:num_tokens={'r1': 5}" in installed.nvtx_calls["marks"]

    runner.execute_model(output)
    assert recording_nvtx.pushes[1] == "[prof_range]: iter 1"


def test_sample_wraps_sampler_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunner()
    logits = SimpleNamespace(shape=(4, 8))

    result = runner._sample(logits, None)

    assert result == ("sample", logits, None)
    assert recording_nvtx.pushes == ["[FW_NATIVE] op:sampler,logits:(4, 8)"]
    assert len(recording_nvtx.pops) == 1

    runner._sample(None, "spec")
    assert recording_nvtx.pushes[1] == "[FW_NATIVE] op:sampler,logits:None"


def test_model_forward_range_uses_stashed_scheduler_output(
    installed, recording_nvtx
) -> None:
    runner = installed.fakes.GPUModelRunner()

    assert runner._model_forward(input_ids=1) == ("forward", (), {"input_ids": 1})
    assert recording_nvtx.pushes == []

    runner._ppu_nvtx_prof_scheduler_output = _sched_output(
        num_scheduled_tokens={"r1": 3, "r2": 1},
        scheduled_spec_decode_tokens={"r1": [10]},
    )
    runner._model_forward(input_ids=2)
    assert recording_nvtx.pushes == ["total bs=2, P bs=1"]
    assert len(recording_nvtx.pops) == 1

    # execute_model clears the stash, so later forwards are uninstrumented.
    runner.execute_model(_sched_output())
    assert recording_nvtx.pushes[-1] == "[prof_range]: iter 0"
    count = len(recording_nvtx.pushes)
    runner._model_forward(input_ids=3)
    assert len(recording_nvtx.pushes) == count


def test_prepare_inputs_is_annotated(installed) -> None:
    runner = installed.fakes.GPUModelRunner()
    assert runner._prepare_inputs("so", "nst") == ("prepare_inputs", "so", "nst")
    assert "prepare_inputs" in installed.nvtx_calls["annotated"]


# ---------------------------------------------------------------------------
# Model Runner V2 (the 0.31 default) carries the same ranges
# ---------------------------------------------------------------------------


def test_v2_execute_model_wraps_iteration_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    # No __init__ patch: the counter is created on first use, like the engine
    # core does it after the profiling patches load mid-construction.
    assert not hasattr(runner, "iteration")

    output = _sched_output()
    assert runner.execute_model(output) == ("execute_model", output, False)

    assert recording_nvtx.pushes == ["[prof_range]: iter 0", "total bs=1, P bs=1"]
    assert len(recording_nvtx.pops) == 2
    assert recording_nvtx.iters == [0]
    assert runner.iteration == 1
    assert any(m.startswith("total_tokens=5") for m in installed.nvtx_calls["marks"])

    runner.execute_model(output)
    assert recording_nvtx.pushes[2:] == [
        "[prof_range]: iter 1",
        "total bs=1, P bs=1",
    ]


def test_v2_execute_model_forwards_dummy_runs_unprofiled(
    installed, recording_nvtx
) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    output = _sched_output()

    assert runner.execute_model(output, dummy_run=True) == (
        "execute_model",
        output,
        True,
    )
    assert recording_nvtx.pushes == []
    assert recording_nvtx.iters == []
    # Memory profiling and CUDA-graph capture must not consume iteration numbers.
    assert not hasattr(runner, "iteration")


def test_v2_sample_wraps_sampler_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    hidden_states = SimpleNamespace(shape=(7, 16))

    assert runner.sample(hidden_states, "batch", None) == (
        "sample",
        hidden_states,
        "batch",
    )
    assert recording_nvtx.pushes == ["[FW_NATIVE] op:sampler,hidden_states:(7, 16)"]
    assert len(recording_nvtx.pops) == 1


def test_v2_prepare_inputs_is_annotated(installed) -> None:
    runner = installed.fakes.GPUModelRunnerV2()

    assert runner.prepare_inputs("so", "state", "desc", 0) == (
        "prepare_inputs",
        "so",
        "state",
    )
    assert "prepare_inputs" in installed.nvtx_calls["annotated"]


def test_v2_instruments_compute_logits_once(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    model = SimpleNamespace(compute_logits=lambda hidden_states: hidden_states)
    runner.model = model

    runner.execute_model(_sched_output())
    traced = model.compute_logits
    runner.execute_model(_sched_output())
    assert model.compute_logits is traced, "compute_logits must be wrapped once"

    hidden_states = SimpleNamespace(shape=(2, 4))
    assert model.compute_logits(hidden_states) is hidden_states
    assert recording_nvtx.pushes[-1] == (
        "[FW_GEMM] op:compute_logits,hidden_states:(2, 4)"
    )


def test_engine_core_steps_carry_iteration_ranges(installed, recording_nvtx) -> None:
    core = installed.fakes.EngineCore("arg")
    assert core.iteration == 0
    assert core.init_args == (("arg",), {})

    assert core.step() == ("step",)
    assert core.step_with_batch_queue() == ("step_with_batch_queue",)

    assert recording_nvtx.pushes == [
        "[prof_range]: iter 0",
        "[prof_range]: iter 1",
    ]
    assert len(recording_nvtx.pops) == 2
    assert recording_nvtx.iters == [0, 1]
    assert core.iteration == 2


@pytest.mark.parametrize("step_name", ["step", "step_with_batch_queue"])
def test_engine_core_loads_profiling_during_init(
    profiling_pkg, nvtx_stub, fake_vllm, cleared_gate, monkeypatch, step_name
) -> None:
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")

    def upstream_init(self):
        profiling_pkg.install()
        self.step_fn = getattr(self, step_name)

    monkeypatch.setattr(fake_vllm.EngineCore, "__init__", upstream_init)

    class EngineCoreProc(fake_vllm.EngineCore):
        def __init__(self):
            super().__init__()

    core = EngineCoreProc()
    assert profiling_pkg._installed
    assert not hasattr(core, "iteration")

    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    events = []
    monkeypatch.setattr(
        prof_nvtx, "_model_prof_iter", lambda i: events.append(("iter", i))
    )
    monkeypatch.setattr(
        prof_nvtx, "_torch_range_push", lambda label: events.append(("push", label))
    )
    monkeypatch.setattr(prof_nvtx, "_torch_range_pop", lambda: events.append(("pop",)))

    for iteration in range(2):
        assert core.step_fn() == (step_name,)
        assert core.iteration == iteration + 1

    assert events == [
        ("iter", 0),
        ("push", "[prof_range]: iter 0"),
        ("pop",),
        ("iter", 1),
        ("push", "[prof_range]: iter 1"),
        ("pop",),
    ]


def test_scheduler_schedule_is_annotated_and_update_marks(installed) -> None:
    scheduler = installed.fakes.Scheduler()

    assert scheduler.schedule(True) == ("schedule", True)
    assert "schedule" in installed.nvtx_calls["annotated"]

    output = _sched_output(
        scheduled_new_reqs=[SimpleNamespace(req_id="r1")],
        finished_req_ids={"r9"},
    )
    scheduler._update_after_schedule(output)
    assert scheduler.last_update is output
    marks = installed.nvtx_calls["marks"]
    assert "new_reqs: ['r1']" in marks
    assert "finish_req: {'r9'}" in marks
    assert any(m.startswith("total_tokens=5") for m in marks)


# ---------------------------------------------------------------------------
# Shared helpers without the optional dependencies
# ---------------------------------------------------------------------------


@pytest.fixture(params=[False, True], ids=["bare", "installed-profiler"])
def bare_nvtx_helpers(
    fake_vllm: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """Import profiling/nvtx.py fresh with nvtx/model_prof genuinely absent."""
    if request.param:
        # Deleting sys.modules is insufficient: installed packages can be
        # imported again from disk. Reproduce an identity NVTX decorator.
        (tmp_path / "nvtx.py").write_text(
            "def annotate(name):\n    return lambda func: func\n"
            "def mark(message):\n    pass\n"
        )
        (tmp_path / "model_prof.py").write_text("def prof_iter(i):\n    pass\n")
        monkeypatch.syspath_prepend(str(tmp_path))
    # None blocks import even if the SDK/PyPI package is installed on disk.
    monkeypatch.setitem(sys.modules, "nvtx", None)
    monkeypatch.setitem(sys.modules, "model_prof", None)
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)
    module = importlib.import_module("vllm_sail.profiling.nvtx")
    yield module
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)


def test_helpers_import_cleanly_without_optional_deps(bare_nvtx_helpers) -> None:
    assert bare_nvtx_helpers.has_nvtx is False
    assert bare_nvtx_helpers.has_model_prof is False

    bare_nvtx_helpers.mark("ignored")
    bare_nvtx_helpers.range_push("ignored")
    bare_nvtx_helpers.range_pop()
    bare_nvtx_helpers.prof_iter(3)
    bare_nvtx_helpers.sche_mark(_sched_output())


def test_annotate_fallback_is_identity(bare_nvtx_helpers) -> None:
    @bare_nvtx_helpers.annotate("whatever")
    def documented(x: int) -> int:
        """Keeps its name."""
        return x + 1

    assert documented(1) == 2
    assert documented.__name__ == "documented"
    assert documented.__wrapped__ is not None


def test_range_push_swallows_cpu_torch_errors(bare_nvtx_helpers) -> None:
    # torch is installed in the test venv but has no CUDA NVTX support, so the
    # real torch.cuda.nvtx.range_push raises; a profiler must not propagate it.
    if not bare_nvtx_helpers.has_torch_nvtx:
        pytest.skip("torch.cuda.nvtx not importable here")
    bare_nvtx_helpers.range_push("probe")
    bare_nvtx_helpers.range_pop()


def test_sche_mark_format_with_nvtx(
    fake_vllm: SimpleNamespace,
    nvtx_stub: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)
    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    output = _sched_output(
        scheduled_new_reqs=[SimpleNamespace(req_id="a"), SimpleNamespace(req_id="b")],
        finished_req_ids={"z"},
    )
    prof_nvtx.sche_mark(output)
    assert nvtx_stub["marks"] == [
        "new_reqs: ['a', 'b']",
        "total_tokens=5,req_id:num_tokens={'r1': 5}",
        "finish_req: {'z'}",
    ]


@pytest.fixture
def topk_dump(fake_vllm, monkeypatch, tmp_path):
    from vllm_sail import envs

    copies, ranges = [], []
    original_torch = sys.modules.get("torch")

    class Tensor:
        def __init__(self, data, shape=None, *, readable=True, device="cpu"):
            self.data = list(data)
            self.shape = tuple(shape or (len(data),))
            self.ndim = len(self.shape)
            self.dtype = "torch.float32"
            self.device = device
            self.is_cuda = device == "cuda"
            self.readable = readable

        def stride(self, dim=None):
            strides, step = [], 1
            for size in reversed(self.shape):
                strides.insert(0, step)
                step *= size
            return tuple(strides) if dim is None else strides[dim]

        def storage_offset(self):
            return 0

        def detach(self):
            assert self.readable, "不能读取未初始化的缓冲区"
            return self

        def to(self, *, device, copy):
            assert device == "cpu" and copy is True
            copies.append(self)
            return Tensor(self.data, self.shape)

        def reshape(self, *shape):
            return Tensor(self.data, (len(self.data),))

        def __getitem__(self, index):
            if isinstance(index, tuple):
                rows = range(self.shape[0])[index[0]]
                cols = range(self.shape[1])[index[1]]
                return Tensor(
                    [self.data[r * self.shape[1] + c] for r in rows for c in cols],
                    (len(rows), len(cols)),
                    readable=self.readable,
                    device=self.device,
                )
            return Tensor(self.data[index])

        def numel(self):
            return len(self.data)

        def tolist(self):
            assert self.device == "cpu"
            return list(self.data)

        def element_size(self):
            return 4

        def data_ptr(self):
            return 0

        def copy_(self, other):
            self.data[:] = other.data
            self.readable = True
            return self

        def __repr__(self):
            raise AssertionError("不能直接格式化设备 tensor")

    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.float32, torch.bfloat16 = "torch.float32", "torch.bfloat16"
    torch.int32, torch.int64 = "torch.int32", "torch.int64"
    torch.dtype = str
    torch.cuda = SimpleNamespace(is_current_stream_capturing=lambda: False)

    def serializable(value):
        if isinstance(value, Tensor):
            return SimpleNamespace(
                data=list(value.data), shape=value.shape, dtype=value.dtype
            )
        if isinstance(value, dict):
            return {name: serializable(item) for name, item in value.items()}
        if isinstance(value, list):
            return [serializable(item) for item in value]
        return value

    def save(record, file):
        pickle.dump(serializable(record), file)

    torch.save = save
    profiler = types.ModuleType("model_prof")
    profiler.active = True
    profiler.prof_is_in_profiling = lambda: profiler.active
    monkeypatch.setitem(sys.modules, "model_prof", profiler)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    monkeypatch.setenv("VLLM_SAIL_NVTX_DUMP_TOPK", "1")
    monkeypatch.setenv("VLLM_SAIL_NVTX_DUMP_TOPK_DIR", str(tmp_path / "dumps"))
    nvtx = types.ModuleType("vllm_sail.profiling.nvtx")

    def push(label):
        from vllm_sail.step5_sparse_cases import parse_label

        filename = parse_label(label).get("topk_tensor", "")
        if filename:
            assert (tmp_path / "dumps" / filename).is_file()
        ranges.append(label)

    nvtx.range_push = push
    nvtx.range_pop = lambda: ranges.append("pop")
    monkeypatch.setitem(sys.modules, nvtx.__name__, nvtx)
    pkg = importlib.import_module("vllm_sail.profiling")
    monkeypatch.setattr(pkg, "nvtx", nvtx, raising=False)
    spec = importlib.util.spec_from_file_location(
        "vllm_sail.profiling.step5_sparse",
        Path(envs.__file__).parent / "profiling/step5_sparse.py",
    )

    def load_module():
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        monkeypatch.setattr(pkg, "step5_sparse", module, raising=False)
        spec.loader.exec_module(module)
        return module

    return SimpleNamespace(
        module=load_module(),
        load_module=load_module,
        profiler=profiler,
        torch=torch,
        original_torch=original_torch,
        Tensor=Tensor,
        copies=copies,
        ranges=ranges,
        directory=tmp_path / "dumps",
    )


def _topk_records(directory):
    return [pickle.loads(path.read_bytes()) for path in sorted(directory.glob("*.pt"))]


@pytest.mark.parametrize("gate", ["VLLM_SAIL_NVTX_PROFILE", "VLLM_SAIL_NVTX_DUMP_TOPK"])
@pytest.mark.parametrize("op", ["topk", "attention"])
def test_topk_dump_off_never_reads_data_or_creates_files(
    topk_dump, monkeypatch, gate, op
):
    monkeypatch.setenv(gate, "0")
    monkeypatch.setattr(
        topk_dump.profiler,
        "prof_is_in_profiling",
        lambda: pytest.fail("Profiler queried with gate off"),
    )
    monkeypatch.setattr(
        topk_dump.module,
        "prof_is_in_profiling",
        topk_dump.profiler.prof_is_in_profiling,
    )
    scores = topk_dump.Tensor([1], readable=False)
    sentinel = object()
    assert (
        topk_dump.module.trace_call(
            op, "selector", "per_row", {"scores": scores}, lambda: sentinel
        )
        is sentinel
    )
    assert topk_dump.copies == []
    assert not topk_dump.directory.exists()


@pytest.mark.parametrize("failure", ["absent", "missing-symbol", "import-error"])
@pytest.mark.parametrize("op", ["topk", "attention"])
def test_dump_profiler_import_failure_keeps_plain_nvtx(
    topk_dump, monkeypatch, caplog, failure, op
):
    from vllm_sail.step5_sparse_cases import parse_label

    profiler = None if failure == "absent" else types.ModuleType("model_prof")
    if failure == "import-error":

        def unavailable(name):
            raise RuntimeError("profiler import failed")

        profiler.__getattr__ = unavailable
    monkeypatch.setitem(sys.modules, "model_prof", profiler)
    module = topk_dump.load_module()
    assert module.DUMP_TENSOR is False
    monkeypatch.setattr(
        module, "_log_topk_tensors", lambda *a: pytest.fail("Unexpected tensor preview")
    )
    scores = topk_dump.Tensor([1], readable=False)
    sentinel = object()
    with caplog.at_level(logging.INFO):
        assert (
            module.trace_call(
                op, "selector", "per_row", {"scores": scores}, lambda: sentinel
            )
            is sentinel
        )
    assert not topk_dump.copies and not topk_dump.directory.exists()
    assert parse_label(topk_dump.ranges[0])["topk_tensor"] == ""
    assert topk_dump.ranges[-1] == "pop"
    assert not any("[SAIL_" in message for message in caplog.messages)


@pytest.mark.parametrize("active", [False, "error"])
@pytest.mark.parametrize("directory", [False, True], ids=["print-only", "save"])
@pytest.mark.parametrize("op", ["topk", "attention"])
def test_dump_inactive_profiler_never_reads_data_and_keeps_nvtx(
    topk_dump, monkeypatch, caplog, active, directory, op
):
    from vllm_sail.step5_sparse_cases import parse_label

    def query():
        if active == "error":
            raise RuntimeError("profiling state unavailable")
        return active

    monkeypatch.setattr(topk_dump.module, "prof_is_in_profiling", query)
    monkeypatch.setattr(
        topk_dump.module,
        "_log_topk_tensors",
        lambda *a: pytest.fail("Unexpected tensor preview"),
    )
    if not directory:
        monkeypatch.setenv("VLLM_SAIL_NVTX_DUMP_TOPK_DIR", "")
    scores = topk_dump.Tensor([1], readable=False)
    sentinel = object()
    with caplog.at_level(logging.INFO):
        assert (
            topk_dump.module.trace_call(
                op, "selector", "per_row", {"scores": scores}, lambda: sentinel
            )
            is sentinel
        )
    assert not topk_dump.copies and not topk_dump.directory.exists()
    assert parse_label(topk_dump.ranges[0])["topk_tensor"] == ""
    assert topk_dump.ranges[-1] == "pop"
    assert not any("[SAIL_" in message for message in caplog.messages)
    if active == "error":
        assert "profiling state unavailable" in caplog.text


@pytest.mark.parametrize("op", ["topk", "attention"])
def test_dump_queries_live_profiling_state_each_call(topk_dump, op):
    from vllm_sail.step5_sparse_cases import parse_label

    assert topk_dump.module.DUMP_TENSOR is True
    scores = topk_dump.Tensor([1])
    calls = []
    for active in (False, True, False, True):
        topk_dump.profiler.active = active
        scores.readable = active
        topk_dump.module.trace_call(
            op, "selector", "per_row", {"scores": scores}, lambda: calls.append(1)
        )
        assert bool(parse_label(topk_dump.ranges[-2])["topk_tensor"]) is active
    assert len(calls) == 4 and len(topk_dump.copies) == 2
    assert len(_topk_records(topk_dump.directory)) == 2


@pytest.mark.parametrize("capture", [True, "error"])
@pytest.mark.parametrize("op", ["topk", "attention"])
def test_topk_dump_capture_is_fail_closed(topk_dump, monkeypatch, capture, op):
    def query():
        if capture == "error":
            raise RuntimeError("capture query failed")
        return capture

    monkeypatch.setattr(topk_dump.torch.cuda, "is_current_stream_capturing", query)
    scores = topk_dump.Tensor([1], device="cuda", readable=False)
    assert (
        topk_dump.module.trace_call(
            op, "selector", "deep_select", {"scores": scores}, lambda: scores
        )
        is scores
    )
    assert topk_dump.copies == []
    assert not topk_dump.directory.exists()


@pytest.mark.parametrize("backend", ["per_row", "deep_select", "selector_v2"])
def test_topk_dump_saves_only_independent_inputs_before_nvtx(
    topk_dump, caplog, backend
):
    scores = topk_dump.Tensor(range(40), (2, 20))
    output = topk_dump.Tensor([-9] * 6, (2, 3), readable=False)
    workspace = topk_dump.Tensor([99], readable=False)
    values = {
        "scores": scores,
        "current": topk_dump.Tensor([20, 20]),
        "regions": output,
        "workspace": workspace,
        "topk": 2,
    }

    def select():
        scores.data[0] = 999
        output.data[:] = [19, 18, 20, 19, 18, 20]
        output.readable = True
        return output

    with caplog.at_level(logging.INFO):
        with topk_dump.module.phase("decode"):
            for _ in range(2):
                assert (
                    topk_dump.module.trace_call(
                        "topk", "selector", backend, values, select
                    )
                    is output
                )
    records = _topk_records(topk_dump.directory)
    assert len(records) == 2 and records[0]["id"] != records[1]["id"]
    first = next(r for r in records if r["inputs"]["scores"].data[0] == 0)
    assert "outputs" not in first
    assert output not in topk_dump.copies and workspace not in topk_dump.copies
    assert first["backend"] == backend and first["pd_type"] == "decode"
    assert "workspace" in first["metadata"]
    assert "workspace" not in first["inputs"]
    assert "indices" not in first["inputs"]
    messages = [
        json.loads(m.split("[SAIL_TOPK_DUMP] ", 1)[1])
        for m in caplog.messages
        if m.startswith("[SAIL_TOPK_DUMP] {")
    ]
    sample = messages[0]["tensors"]["scores"]
    assert sample["truncated"] and sample["first_values"] == list(range(32))
    from vllm_sail.step5_sparse_cases import parse_label

    case = parse_label(topk_dump.ranges[0])
    assert case["v"] == "1" and case["topk_tensor"]
    assert (topk_dump.directory / case["topk_tensor"]).exists()


def test_topk_dump_print_only_leaves_tensor_field_empty(topk_dump, monkeypatch, caplog):
    monkeypatch.setenv("VLLM_SAIL_NVTX_DUMP_TOPK_DIR", "")
    scores = topk_dump.Tensor(range(8), (2, 4))
    output = topk_dump.Tensor([1, 0, -9, 2, 1, -9, -9, -9, -9], (3, 3))
    with caplog.at_level(logging.INFO):
        topk_dump.module.trace_call(
            "topk",
            "selector",
            "per_row",
            {"scores": scores, "indices": output, "topk": 2},
            lambda: output,
        )
    assert not topk_dump.directory.exists()
    message = next(m for m in caplog.messages if '"stage": "inputs"' in m)
    result = json.loads(message.split("[SAIL_TOPK_DUMP] ", 1)[1])
    assert set(result["tensors"]) == {"scores"}
    from vllm_sail.step5_sparse_cases import parse_label

    assert parse_label(topk_dump.ranges[0])["topk_tensor"] == ""


@pytest.mark.parametrize("failure", ["input", "save", "selector"])
@pytest.mark.parametrize("op", ["topk", "attention"])
def test_topk_dump_failures_preserve_selector_contract(
    topk_dump, monkeypatch, failure, op
):
    scores = topk_dump.Tensor([1], readable=failure != "input")
    sentinel, error = object(), RuntimeError("selector failure")
    if failure == "save":

        def save(*args):
            raise OSError("disk full")

        monkeypatch.setattr(topk_dump.torch, "save", save)

    def select():
        if failure == "selector":
            raise error
        return sentinel

    if failure == "selector":
        with pytest.raises(RuntimeError) as caught:
            topk_dump.module.trace_call(
                op, "selector", "per_row", {"scores": scores}, select
            )
        assert caught.value is error
        assert len(_topk_records(topk_dump.directory)) == 1
        assert topk_dump.module._RANGES.get() == ()
    else:
        assert (
            topk_dump.module.trace_call(
                op, "selector", "per_row", {"scores": scores}, select
            )
            is sentinel
        )
        from vllm_sail.step5_sparse_cases import parse_label

        assert parse_label(topk_dump.ranges[0])["topk_tensor"] == ""
        assert not list(topk_dump.directory.glob("*.pt"))


@pytest.mark.parametrize("backend", ["per_row", "deep_select"])
@pytest.mark.parametrize("active", [False, True])
def test_topk_factory_dump_observes_actual_backend(
    topk_dump, monkeypatch, backend, active
):
    topk_dump.profiler.active = active
    module = types.ModuleType("vllm.models.step5.nvidia.ops.sparse_indexer")

    class Selector:
        def resolve_backend(self, *args):
            return backend

        def __call__(self, scores, lengths, next_n, indices, topk, max_seq_len):
            self.resolve_backend(scores, topk, scores.shape[0])
            indices.data[:] = [1, 0]
            indices.readable = True

    selector = Selector()
    module.get_indexer_topk = lambda requested: selector
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        topk_dump.module, "_install_one", lambda m, a, r: setattr(module, a, r)
    )
    topk_dump.module._install_topk_factory()
    scores = topk_dump.Tensor([0, 1], (1, 2))
    indices = topk_dump.Tensor([-9, -9], (1, 2), readable=False)
    with topk_dump.module.trace_range(
        "topk", "select_regions", "dispatch", {"scores": scores}
    ):
        assert (
            module.get_indexer_topk("auto")(
                scores, topk_dump.Tensor([2]), 1, indices, 2, 2
            )
            is None
        )
    if not active:
        from vllm_sail.step5_sparse_cases import parse_label

        label = parse_label(topk_dump.ranges[1])
        assert label["backend"] == backend and label["topk_tensor"] == ""
        assert not topk_dump.copies and not topk_dump.directory.exists()
        return
    (record,) = _topk_records(topk_dump.directory)
    assert record["backend"] == backend and record["scope"] == "kernel"
    assert "outputs" not in record and "indices" not in record["inputs"]
    from vllm_sail.step5_sparse_cases import parse_label

    label = parse_label(topk_dump.ranges[1])
    assert label["backend"] == backend and label["topk_tensor"]


@pytest.mark.parametrize("active", [False, True])
def test_sail_deepselect_entry_dumps_only_real_input(topk_dump, monkeypatch, active):
    topk_dump.profiler.active = active
    platform = types.ModuleType("vllm.platforms")
    platform.current_platform = SimpleNamespace(
        is_ppu=lambda: True, has_device_capability=lambda n: True
    )
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    path = Path(__file__).parents[2] / "vllm_sail/utils/deep_select.py"
    spec = importlib.util.spec_from_file_location("_sail_dump_deepselect", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    indices = topk_dump.Tensor([1, 0], (1, 2))
    calls = []

    def topk(logits, k, **kwargs):
        calls.append((logits, k, kwargs))
        return None, indices

    monkeypatch.setattr(
        module,
        "_wheel",
        lambda: SimpleNamespace(get_stride_requirement=lambda: (4, 4), topk=topk),
    )
    monkeypatch.setenv("VLLM_SAIL_USE_DEEPSELECT", "1")
    scores = topk_dump.Tensor([0, 1], (1, 2))
    assert module.deep_select_topk(scores, 2) is indices
    assert len(calls) == 1 and calls[0][0] is scores
    if not active:
        from vllm_sail.step5_sparse_cases import parse_label

        assert parse_label(topk_dump.ranges[0])["topk_tensor"] == ""
        assert not topk_dump.copies and not topk_dump.directory.exists()
        return
    (record,) = _topk_records(topk_dump.directory)
    assert record["backend"] == "deep_select"
    assert record["inputs"]["scores"].data == [0, 1]
    assert "outputs" not in record and indices not in topk_dump.copies


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int64"])
def test_real_torch_input_snapshot_roundtrip(topk_dump, monkeypatch, dtype):
    monkeypatch.delitem(sys.modules, "torch")
    torch = topk_dump.original_torch or pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)
    scores = torch.arange(40, dtype=getattr(torch, dtype)).as_strided(
        (2, 8), (12, 1), 3
    )
    before = scores.clone()
    output = torch.empty((2, 2), dtype=torch.int32)

    def call():
        scores.fill_(999)
        return output

    assert (
        topk_dump.module.trace_call(
            "topk",
            "selector",
            "per_row",
            {"scores": scores, "indices": output, "topk": 2},
            call,
        )
        is output
    )
    from vllm_sail.step5_sparse_cases import parse_label

    case = parse_label(topk_dump.ranges[0])
    record = torch.load(
        topk_dump.directory / case["topk_tensor"], map_location="cpu", weights_only=True
    )
    assert set(record["inputs"]) == {"scores"} and "outputs" not in record
    assert torch.equal(record["inputs"]["scores"], before)
    assert record["metadata"]["scores"]["stride"] == [12, 1]
    assert record["metadata"]["scores"]["offset"] == 3
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "tests/nvtx_based_ut/step5")
    )
    replay = importlib.import_module("step5_sparse_replay")
    inputs, digest = replay.load_topk_tensor(
        torch, case, topk_dump.directory / case["topk_tensor"]
    )
    assert torch.equal(inputs["scores"], before) and len(digest) == 64


def test_topk_dump_includes_paged_inputs_but_never_output_data(topk_dump):
    paged = SimpleNamespace(
        owners=topk_dump.Tensor([0]),
        block_table=topk_dump.Tensor([1, 2], (1, 2)),
        indices=topk_dump.Tensor([9, 9], readable=False),
        indptr=topk_dump.Tensor([0, 2]),
        block_size=64,
        page_stride=8,
    )
    topk_dump.module.trace_call(
        "topk",
        "selector",
        "per_row",
        {"scores": topk_dump.Tensor([0, 1], (1, 2)), "paged_prefill": paged},
        lambda: None,
    )
    (record,) = _topk_records(topk_dump.directory)
    assert set(record["inputs"]) == {"scores", "owners", "block_table", "indptr"}
    assert "pages" in record["metadata"] and paged.indices not in topk_dump.copies


@pytest.mark.parametrize("kind", ["prefill", "decode"])
def test_native_attention_snapshots_inputs_for_both_scopes(topk_dump, kind):
    from tests.ut.test_step5_sparse_cases import native_submission_namespace
    from vllm_sail.step5_sparse_cases import parse_label

    t = topk_dump.Tensor
    output = t([-9], readable=False)
    indices = t([1, 2], (1, 2))
    prefix = (t([1]), t([2]), indices, t([0]))
    args = (
        (*prefix, t([15]), t([1]), output, 0.1, 1, True, None)
        if kind == "decode"
        else (*prefix, t([0]), t([15]), t([0, 1], (2, 1)), output, 2, 0.1, 1)
    )
    calls = []

    def run(*received):
        calls.append(received)
        indices.data[0] = 99
        return output

    submit = native_submission_namespace()["_submit_attention"]
    assert submit(run, kind, *args) is output
    assert len(calls) == 1 and calls[0] == args
    labels = [parse_label(label) for label in topk_dump.ranges if label != "pop"]
    assert [c["scope"] for c in labels] == ["op", "kernel"]
    assert all(c["topk_tensor"].startswith("attention-") for c in labels)
    records = _topk_records(topk_dump.directory)
    assert len(records) == 2
    for record in records:
        assert record["op"] == "attention" and record["pd_type"] == kind
        assert record["inputs"]["indices"].data == [1, 2]
        assert "output" in record["metadata"] and "output" not in record["inputs"]
        assert "outputs" not in record
    assert output not in topk_dump.copies


@pytest.mark.parametrize("combine", [False, True])
def test_attention_split_outputs_and_combine_inputs_have_distinct_roles(
    topk_dump, combine
):
    t = topk_dump.Tensor
    values = dict(
        partial=t([1], readable=combine),
        lse=t([2], readable=combine),
        output=t([9], readable=False),
        indices=t([3]),
    )
    impl = "_sparse_decode_combine" if combine else "_sparse_decode_split"
    topk_dump.module.trace_call(
        "attention",
        impl,
        "upstream_triton",
        values,
        lambda: None,
        scope="kernel",
        pd_type="decode",
    )
    (record,) = _topk_records(topk_dump.directory)
    assert set(record["inputs"]) == (
        {"indices", "partial", "lse"} if combine else {"indices"}
    )
    assert record["inputs"]["indices"].data == [3]
    assert values["output"] not in topk_dump.copies


@pytest.mark.parametrize("paged", [False, True])
def test_attention_bound_run_snapshots_prepared_plan_inputs(topk_dump, paged):
    from vllm_sail.step5_sparse_cases import profiled

    t = topk_dump.Tensor

    class Wrapper:
        def run(self, q, paged_kv_cache, out=None):
            return out

    owner = Wrapper()
    owner._qo_indptr_buf = t([0, 1])
    owner._sm_scale = 0.1
    if paged:
        owner._paged_kv_indptr_buf = t([0, 1])
        owner._paged_kv_indices_buf = t([3])
        owner._paged_kv_last_page_len_buf = t([8])
    else:
        owner._kv_indptr_buf = t([0, 8])
    run = profiled(
        "attention",
        backend="flashinfer_paged" if paged else "flashinfer_ragged",
        scope="kernel",
        pd_type="prefill",
    )(owner.run)
    output = t([9], readable=False)
    assert run(t([1]), (t([2]), t([3])), out=output) is output
    (record,) = _topk_records(topk_dump.directory)
    names = {"query", "key", "value", "qo_indptr", "indptr"}
    if paged:
        names.update(("pages", "last_page_len"))
    assert set(record["inputs"]) == names
    assert record["scalars"]["scale"] == 0.1


@pytest.mark.parametrize("capture", [False, True, "error"])
def test_attention_chunks_snapshot_nested_inputs_and_capture_guard(
    topk_dump, monkeypatch, capture
):
    t = topk_dump.Tensor
    chunk = SimpleNamespace(
        cu_q=t([0, 1]),
        cu_k=t([0, 8]),
        cu_k_device=t([0, 8], device="cuda", readable=capture is False),
        total_tokens=8,
        max_length=8,
    )

    def query():
        if capture == "error":
            raise RuntimeError("capture query failed")
        return capture

    monkeypatch.setattr(topk_dump.torch.cuda, "is_current_stream_capturing", query)
    values = dict(query=t([1]), requests=t([0]), prefill_chunks=[chunk])
    sentinel = object()
    assert (
        topk_dump.module.trace_call(
            "attention",
            "run_attention",
            "flashinfer_ragged",
            values,
            lambda: sentinel,
            pd_type="prefill",
        )
        is sentinel
    )
    if capture is not False:
        assert not topk_dump.copies and not topk_dump.directory.exists()
        return
    (record,) = _topk_records(topk_dump.directory)
    saved = record["parameters"]["prefill_chunks"][0]
    chunk.cu_k_device.data[1] = 99
    assert saved["cu_k_device"]["tensor"].data == [0, 8]
    assert saved["cu_k_device"]["metadata"]["device"] == "cuda"
    assert saved["total_tokens"] == saved["max_length"] == 8
    assert record["inputs"]["owners"].data == [0]


def test_attention_owner_metadata_failure_keeps_computation(topk_dump):
    class Owner:
        @property
        def pages(self):
            raise RuntimeError("plan metadata unavailable")

    sentinel = object()
    assert (
        topk_dump.module.trace_call(
            "attention", "run", "flashinfer_paged", {"self": Owner()}, lambda: sentinel
        )
        is sentinel
    )
    assert not topk_dump.copies and not topk_dump.directory.exists()


@pytest.mark.parametrize("kind", ["prefill", "decode"])
@pytest.mark.parametrize("scope", ["op", "kernel"])
@pytest.mark.parametrize("missing", [False, True])
def test_real_torch_attention_dump_restores_values_layout_and_native_parameters(
    topk_dump,
    monkeypatch,
    kind,
    scope,
    missing,
):
    from tests.ut.test_step5_sparse_cases import native_submission_namespace
    from vllm_sail.step5_sparse_cases import parse_label

    monkeypatch.delitem(sys.modules, "torch")
    torch = topk_dump.original_torch or pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)
    q = torch.arange(3300, dtype=torch.bfloat16).as_strided(
        (2, 8, 192), (1600, 192, 1), 4
    )
    cache = torch.arange(25000, dtype=torch.bfloat16).as_strided(
        (2, 1, 16, 384),
        (12288, 6144, 384, 1),
        128,
    )
    indices = torch.tensor([0, 1, 9, -1, 1], dtype=torch.int64).as_strided(
        (2, 2), (3, 1), 0
    )
    table = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    pos = torch.tensor([15, 8], dtype=torch.int64)
    output = torch.empty((2, 8, 192), dtype=torch.bfloat16)
    prefix = (q, cache, indices, table)
    native_args = (
        (*prefix, pos, torch.tensor([True, True]), output, 0.1, 1, True, None)
        if kind == "decode"
        else (
            *prefix,
            torch.tensor([1, 0]),
            pos,
            torch.tensor([[0], [2]], dtype=torch.int32),
            output,
            2,
            0.1,
            1,
        )
    )
    submit = native_submission_namespace()["_submit_attention"]
    submit(lambda *a: output, kind, *native_args)
    case = next(
        parse_label(label)
        for label in topk_dump.ranges
        if label != "pop" and parse_label(label)["scope"] == scope
    )
    path = topk_dump.directory / case["topk_tensor"]
    record = torch.load(path, map_location="cpu", weights_only=True)
    assert "output" not in record["inputs"] and "outputs" not in record
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "tests/nvtx_based_ut/step5")
    )
    replay = importlib.import_module("step5_sparse_replay")
    spec = importlib.util.spec_from_file_location(
        "attention_bench",
        Path(__file__).parents[2]
        / "tests/nvtx_based_ut/step5/benchmark_step5_sparse_nvtx.py",
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    if missing:
        case["topk_tensor"] = "missing-attention.pt"
    case["sources"] = []

    class CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *a, **kw):
            return torch.empty(*a, **{**kw, "device": "cpu"})

        def tensor(self, *a, **kw):
            return torch.tensor(*a, **{**kw, "device": "cpu"})

        def randn(self, *a, **kw):
            return torch.randn(*a, **{**kw, "device": "cpu"})

    native = types.ModuleType("vllm_sail.models.step5.ops.dsa_native")
    calls = []
    native.decode_reason = native.prefill_reason = lambda *a: None
    native.compiled_op = lambda kind: lambda *a: calls.append(a)
    ops = importlib.import_module("vllm_sail.models.step5.ops")
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(ops, "dsa_native", native, raising=False)
    support = types.ModuleType("step5_sparse_support")
    support.check_selection_oracle = lambda *a, **kw: [0, 1]
    monkeypatch.setitem(sys.modules, support.__name__, support)
    args = SimpleNamespace(
        scope="prepared-op" if scope == "op" else "kernel",
        max_memory_mib=2048,
        seed=0,
        topk_tensor_dir=topk_dump.directory,
    )
    geo = bench.geometry(case, {})
    arms, blocked = replay.prepare_arms(
        CpuTorch(), case, geo, {"policy": "tensor-dump"}, ["native"], args
    )
    assert not blocked and len(arms) == 1
    arms[0].call()
    (received,) = calls
    work = arms[0].config["normalized_workload"]
    assert work["synthetic"] is missing
    if missing:
        assert work["topk_tensor_fallback"]["filename"] == "missing-attention.pt"
        assert "topk_tensor_sha256" not in work
    else:
        assert len(work["topk_tensor_sha256"]) == 64
        for original, restored in zip(native_args[:4], received[:4], strict=True):
            assert torch.equal(original, restored)
            assert original.stride() == restored.stride()
            assert original.storage_offset() == restored.storage_offset()
        assert torch.equal(received[4], native_args[4])
        if kind == "prefill":
            assert torch.equal(received[6], native_args[6])
            assert received[8:] == native_args[8:]
        else:
            assert received[7:] == native_args[7:]
        assert work["positions"] == [15, 8] and work["indices"] == [[0, 1], [-1, 1]]
        record["inputs"].pop("indices")
        torch.save(record, path)
        with pytest.raises(
            replay.Unsupported, match="Missing Attention input tensor: indices"
        ):
            replay.load_topk_tensor(torch, case, path)
