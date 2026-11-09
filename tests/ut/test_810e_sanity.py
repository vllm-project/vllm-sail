# SPDX-License-Identifier: Apache-2.0
"""The sanity runner's public command and response contracts need no SDK."""

import importlib.util
import json
import socket
import sys
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


def test_serve_preserves_json_and_checkpoint_arguments():
    presets = json.loads(sanity.CATALOG.read_text())
    assert len(presets) == 8
    for preset in presets:
        argv = sanity.serve_command(preset, Path("/models/space in name"), 9985)
        assert argv[4] == "/models/space in name"
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert argv[argv.index("--served-model-name") + 1] == preset["key"]
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
def test_real_http_lifecycle_records_generation_and_cleans_up(
    tmp_path, monkeypatch, answer
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

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.path in {"/health", "/v1/models"}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "demo"}]}).encode())

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert payload["model"] == "demo" and payload["messages"]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({
            "model": "demo",
            "choices": [{"message": {"content": sys.argv[2]}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }).encode())

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""")
    monkeypatch.setattr(
        sanity,
        "serve_command",
        lambda *unused: [sys.executable, str(server), str(port), answer],
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
        {"key": "demo", "cards": 2, "args": []},
        checkpoint,
        args,
        available_cards=2,
    )
    assert result["status"] == ("passed" if answer else "failed")
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
        {"key": "demo", "cards": 2, "args": []},
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
