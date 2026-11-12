# SPDX-License-Identifier: Apache-2.0
"""The sanity runner's public command and response contracts need no SDK."""

import concurrent.futures
import importlib.util
import json
import os
import shlex
import signal
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ppu_810e_sanity", ROOT / "scripts/ci/ppu_810e_sanity.py"
)
sanity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sanity)


def completion(**overrides):
    return {
        "model": "demo",
        "choices": [{"message": {"content": "2"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1},
        **overrides,
    }


@pytest.fixture
def external_launcher(monkeypatch):
    """Strict process/OS doubles; retain the real runner and thread pool."""
    stopped = threading.Event()
    signals = []

    class Launcher:
        pid = 987654321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            assert stopped.is_set()
            return self.returncode

    process = Launcher()

    def launch(argv, *, cwd, env, stdout, stderr, start_new_session):
        assert start_new_session
        return process

    def kill_group(pid, signum):
        assert pid == process.pid
        signals.append(signum)
        if process.returncode is None:
            process.returncode = -signum
        stopped.set()

    monkeypatch.setattr(sanity.subprocess, "Popen", launch)
    monkeypatch.setattr(sanity.os, "killpg", kill_group)
    return SimpleNamespace(process=process, stopped=stopped, signals=signals)


def test_exited_launcher_fails_even_when_leftover_workers_answer(
    tmp_path, monkeypatch, external_launcher
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = SimpleNamespace(
        results_dir=tmp_path, port=0, ready_timeout=5, request_timeout=5, max_tokens=16
    )
    health_checks = 0

    def request(url, *, payload=None, timeout=5):
        nonlocal health_checks
        if url.endswith("/health"):
            health_checks += 1
            if health_checks == 2:
                external_launcher.process.returncode = 23
            return None
        if url.endswith("/v1/models"):
            return {"data": [{"id": "demo"}]}
        assert url.endswith("/v1/chat/completions")
        return completion()

    monkeypatch.setattr(sanity, "request_json", request)
    result = sanity.run_preset(
        {"key": "demo", "cards": 2, "args": []}, checkpoint, args, 2
    )
    assert result["status"] == "failed"
    assert result["stage"] == "generation"
    assert result["launcher_exit_code"] == 23
    assert "server exited with code 23" in result["error"]
    assert json.loads((tmp_path / "demo/result.json").read_text()) == result
    assert (tmp_path / "demo/response-1.json").is_file()


@pytest.mark.parametrize("repeat_signal", [False, True])
def test_sigterm_stops_server_before_joining_blocked_burst_and_saves_results(
    tmp_path, monkeypatch, external_launcher, repeat_signal
):
    key = "qwen3.5-35b_bf16_tp2"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    results = tmp_path / "results"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ppu_810e_sanity.py",
            "--models",
            key,
            "--checkpoint",
            f"{key}={checkpoint}",
            "--results-dir",
            str(results),
            "--port",
            "0",
        ],
    )
    monkeypatch.setattr(
        sanity, "runtime_environment", lambda *args: {"devices": [None, None]}
    )
    generations = 0
    unblocked_by_cleanup = []
    waiting_for_result = threading.Event()
    future_result = concurrent.futures.Future.result

    def observed_result(future, timeout=None):
        if threading.current_thread() is threading.main_thread():
            waiting_for_result.set()
        return future_result(future, timeout=timeout)

    monkeypatch.setattr(concurrent.futures.Future, "result", observed_result)

    def request(url, *, payload=None, timeout=5):
        nonlocal generations
        if url.endswith("/health"):
            return None
        if url.endswith("/v1/models"):
            return {"data": [{"id": key}]}
        assert url.endswith("/v1/chat/completions")
        generations += 1
        if generations == 1:
            return completion(model=key)
        # Future.result is reached after executor.submit has registered the
        # worker. This keeps the negative control inside the old join window.
        assert waiting_for_result.wait(timeout=2)
        os.kill(os.getpid(), signal.SIGTERM)
        # The finite wait makes the old pool-context bug fail deterministically
        # instead of hanging the test process indefinitely.
        unblocked_by_cleanup.append(external_launcher.stopped.wait(timeout=2))
        raise OSError("owned server connection closed")

    kill_group = sanity.os.killpg

    def repeated_cancel(pid, signum):
        kill_group(pid, signum)
        if repeat_signal and signum == signal.SIGTERM:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(sanity, "request_json", request)
    monkeypatch.setattr(sanity.os, "killpg", repeated_cancel)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        try:
            code = sanity.main()
        except KeyboardInterrupt:
            # Keep the negative control on the old implementation from
            # aborting pytest before its cleanup-order assertions can run.
            code = 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert code == 130
    assert unblocked_by_cleanup == [True]
    assert external_launcher.signals == [signal.SIGTERM, signal.SIGKILL]
    result = json.loads((results / key / "result.json").read_text())
    assert result["status"] == "cancelled" and result["stage"] == "generation"
    assert "KeyboardInterrupt" in result["error"]
    assert json.loads((results / "summary.json").read_text()) == [result]


def test_completion_requires_a_final_answer_and_generated_tokens():
    sanity.validate_completion(completion(), "demo")
    for response in (
        completion(error={"message": "engine failed"}),
        completion(model="another-server"),
        completion(choices=[]),
        completion(choices=[{"message": {"content": None, "reasoning": "thinking"}}]),
        completion(choices=[{"message": {"content": " "}}]),
        completion(choices=[{"message": {"content": "2"}, "finish_reason": "error"}]),
        completion(
            choices=[{"message": {"content": "thinking"}, "finish_reason": "length"}]
        ),
        completion(usage={"completion_tokens": 0}),
    ):
        with pytest.raises(ValueError):
            sanity.validate_completion(response, "demo")


@pytest.mark.parametrize("temperature", [None, 1.0])
def test_model_temperature_preserves_request_budget_and_validation(
    tmp_path, monkeypatch, external_launcher, temperature
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = SimpleNamespace(
        results_dir=tmp_path,
        port=0,
        ready_timeout=5,
        request_timeout=5,
        max_tokens=4096,
    )
    preset = {"key": "demo", "cards": 2, "args": []}
    if temperature is not None:
        preset["temperature"] = temperature
    payloads = []

    def request(url, *, payload=None, timeout=5):
        if url.endswith("/health"):
            return None
        if url.endswith("/v1/models"):
            return {"data": [{"id": "demo"}]}
        payloads.append(payload)
        return completion()

    monkeypatch.setattr(sanity, "request_json", request)
    result = sanity.run_preset(preset, checkpoint, args, 2)
    assert result["status"] == "passed" and result["requests"] == 2
    assert [p["temperature"] for p in payloads] == [temperature or 0] * 2
    assert all(p["max_completion_tokens"] == 4096 and p["seed"] == 0 for p in payloads)
    assert all(p["model"] == "demo" for p in payloads)


def test_serve_preserves_json_and_checkpoint_arguments():
    presets = json.loads(sanity.CATALOG.read_text())
    assert len(presets) == 10
    for preset in presets:
        argv = sanity.serve_command(preset, Path("/models/space in name"), 9985)
        assert argv[4] == "/models/space in name"
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert argv[argv.index("--served-model-name") + 1] == preset.get(
            "served_model_name", preset["key"]
        )
        assert argv[-len(preset["args"]) :] == preset["args"]
        if "--speculative-config" in argv:
            config = json.loads(argv[argv.index("--speculative-config") + 1])
            assert config == {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_sample_method": "probabilistic",
            }


def test_missing_checkpoint_fails_without_starting_a_server(tmp_path):
    args = type("Options", (), {"results_dir": tmp_path, "port": 9985})()
    preset = {"key": "missing", "cards": 16, "args": []}
    result = sanity.run_preset(preset, tmp_path / "absent", args, available_cards=16)
    assert result["status"] == "failed"
    assert result["stage"] == "preflight"
    assert "config missing" in result["error"]
    assert not (tmp_path / "missing" / "serve.log").exists()


def test_890_cli_preserves_mega_moe_and_fp4_indexer(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sanity", "--device", "890", "--dry-run"])
    assert sanity.main() == 0
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 3
    deepseek = next(argv for argv in commands if "deepseek_v41" in argv)
    assert "--enable-expert-parallel" in deepseek
    assert deepseek[deepseek.index("--moe-backend") + 1] == "deep_gemm_mega_moe"
    assert json.loads(deepseek[deepseek.index("--attention-config") + 1]) == {
        "indexer_sparse_logits": True,
        "indexer_kv_dtype": "mxfp4",
    }
    assert "DeepSeek-V4.1-Flash-1017-MoE-Quant" in deepseek[4]


def test_810e_v41_uses_1017_int8_checkpoint():
    preset = next(
        item
        for item in json.loads(sanity.CATALOG.read_text())
        if item["key"] == "deepseek-v4.1-flash_1017_int8_tp8"
    )
    assert preset["checkpoint"] == (
        "LLM/T-HEAD/v4.1/DeepSeek-V4.1-Flash-1017-Quant-W-INT8-PerChannel-"
        "A-INT8-PerToken-Engram-MXFP8"
    )


@pytest.mark.parametrize("device,capability", [("810e", (8, 0)), ("890", (8, 9))])
def test_runtime_checks_device_family_and_records_selected_catalog(
    monkeypatch, device, capability
):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_capability=lambda index: capability,
        get_device_name=lambda index: "test PPU",
    )
    for name, module in {
        "torch": SimpleNamespace(cuda=cuda),
        "vllm": SimpleNamespace(__file__="/env/site-packages/vllm/__init__.py"),
        "vllm.platforms": SimpleNamespace(
            current_platform=SimpleNamespace(is_ppu=lambda: True)
        ),
        "vllm_sail": SimpleNamespace(
            __file__="/env/site-packages/vllm_sail/__init__.py"
        ),
        "vllm_sail.compat": SimpleNamespace(
            check_vllm_compatibility=lambda **kwargs: None
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sanity.importlib.metadata, "version", lambda name: "test")
    catalog = sanity.CATALOG.with_name(f"ppu_{device}_sanity_models.json")
    environment = sanity.runtime_environment(device, catalog)
    assert environment["devices"][0]["capability"] == capability
    assert environment["device_family"] == device
    assert (
        environment["catalog_sha256"]
        == sanity.hashlib.sha256(catalog.read_bytes()).hexdigest()
    )
    other_device = "890" if device == "810e" else "810e"
    with pytest.raises(RuntimeError, match=f"is not {other_device}"):
        sanity.runtime_environment(other_device, catalog)


def test_insufficient_devices_fail_without_starting_a_server(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = type("Options", (), {"results_dir": tmp_path, "port": 9985})()
    result = sanity.run_preset(
        {"key": "too-few", "cards": 16, "args": []},
        checkpoint,
        args,
        available_cards=8,
    )
    assert result["status"] == "failed"
    assert "requires 16 cards; got 8" in result["error"]


@pytest.mark.parametrize("answer", ["hello", ""])
@pytest.mark.parametrize("use_v41_settings", [False, True])
def test_real_http_lifecycle_records_generation_and_cleans_up(
    tmp_path, monkeypatch, answer, use_v41_settings
):
    """A strict HTTP double stands in for the external vLLM server only."""
    with socket.socket() as free_port:
        free_port.bind(("127.0.0.1", 0))
        port = free_port.getsockname()[1]
    server = tmp_path / "server.py"
    server.write_text("""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

assert os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S") == "20"
assert os.environ.get("VLLM_USE_RUST_FRONTEND") == sys.argv[3]
model_name = sys.argv[4]

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.path in {"/health", "/v1/models"}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": model_name}]}).encode())

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert payload["model"] == model_name and payload["messages"]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({
            "model": model_name,
            "choices": [{"message": {"content": sys.argv[2]}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }).encode())

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""")
    preset = {"key": "demo", "cards": 2, "args": []}
    if use_v41_settings:
        v41 = next(
            item
            for item in json.loads(sanity.CATALOG.read_text())
            if item["key"] == "deepseek-v4.1-flash_1017_int8_tp8"
        )
        preset.update(served_model_name=v41["served_model_name"], env=v41["env"])
    model_name = preset.get("served_model_name", preset["key"])
    rust_frontend = "1" if use_v41_settings else "0"
    monkeypatch.setattr(
        sanity,
        "serve_command",
        lambda *unused: [
            sys.executable,
            str(server),
            str(port),
            answer,
            rust_frontend,
            model_name,
        ],
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = SimpleNamespace(
        results_dir=tmp_path,
        port=port,
        ready_timeout=20,
        request_timeout=5,
        max_tokens=16,
    )
    result = sanity.run_preset(
        preset,
        checkpoint,
        args,
        available_cards=2,
    )
    assert result["status"] == ("passed" if answer else "failed")
    assert result["serve_env"]["VLLM_USE_RUST_FRONTEND"] == rust_frontend
    assert json.loads((tmp_path / "demo" / "result.json").read_text()) == result
    assert (tmp_path / "demo" / "response-0.json").is_file()
    if answer:
        assert result["requests"] == 2
        assert (tmp_path / "demo" / "response-1.json").is_file()
    else:
        assert result["stage"] == "generation"
        assert "empty final answer" in result["error"]
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0
    # Reuse the same port immediately: TIME_WAIT must not be treated as an
    # active server when advancing to the next preset.
    args.results_dir = tmp_path / "repeat"
    args.results_dir.mkdir()
    repeated = sanity.run_preset(
        preset,
        checkpoint,
        args,
        available_cards=2,
    )
    assert repeated["status"] == ("passed" if answer else "failed")
    assert repeated["stage"] == ("complete" if answer else "generation")


def test_listening_port_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        args = SimpleNamespace(results_dir=tmp_path, port=listener.getsockname()[1])
        result = sanity.run_preset(
            {"key": "occupied", "cards": 2, "args": []},
            checkpoint,
            args,
            available_cards=2,
        )
    assert result["status"] == "failed"
    assert result["stage"] == "preflight"
    assert "Address already in use" in result["error"]
    assert not (tmp_path / "occupied" / "serve.log").exists()
