# SPDX-License-Identifier: Apache-2.0
"""Device unit test and standalone reproducer for DSV4.1 indexer Q GEMM.

Run each case in a fresh process: a failed capture invalidates the CUDA context,
and DeepGEMM caches loaded kernels per process. No checkpoint is needed.

Numerical/capture control:
    python tests/e2e/test_dense_fp8_graph.py --tokens 64 --warmup shape

Reproduce the rc1 failure after a profile-size GEMM, without query-size warmup:
    python tests/e2e/test_dense_fp8_graph.py --tokens 64 --warmup profile

The latter deliberately exits nonzero on the affected SDK. It models the missing
query projection during short-context eager warmup, not a supported cold-capture
contract. ``--warmup none`` isolates loading the first kernel during capture.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


def run_case(tokens: int, warmup: str) -> None:
    import deep_gemm
    import torch

    assert torch.cuda.get_device_capability() == (8, 9), "Requires Zhenwu M890"
    # wq_b is replicated, including in TP8: 32 index heads * 128, rank 1280.
    outputs, hidden = 4096, 1280
    generator = torch.Generator().manual_seed(1017)
    x_cpu = torch.randint(-2, 3, (tokens, hidden), generator=generator).float()
    w_cpu = torch.randint(-2, 3, (outputs, hidden), generator=generator).float()
    xs_cpu = torch.arange(tokens).remainder(3).add(1).float()[:, None] / 16
    ws_cpu = torch.arange(outputs).remainder(3).add(1).float()[:, None] / 16
    expected = ((x_cpu * xs_cpu) @ (w_cpu * ws_cpu).T).bfloat16()
    x = x_cpu.to(device="cuda", dtype=torch.float8_e4m3fn)
    weight = w_cpu.to(device="cuda", dtype=torch.float8_e4m3fn)
    xs, ws = xs_cpu.cuda(), ws_cpu.cuda()
    output = torch.empty((tokens, outputs), dtype=torch.bfloat16, device="cuda")

    def gemm():
        deep_gemm.fp8_gemm_nt((x, xs), (weight, ws), output)

    print(f"M={tokens} N={outputs} K={hidden} warmup={warmup}", flush=True)
    if warmup == "profile":
        profile_x = torch.zeros(
            (4096, hidden), dtype=torch.float8_e4m3fn, device="cuda"
        )
        profile_scale = torch.ones((4096, 1), device="cuda")
        profile_out = torch.empty((4096, outputs), dtype=torch.bfloat16, device="cuda")
        deep_gemm.fp8_gemm_nt((profile_x, profile_scale), (weight, ws), profile_out)
        torch.cuda.synchronize()
        torch.testing.assert_close(profile_out, torch.zeros_like(profile_out))
        print("PROFILE PASS", flush=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if warmup == "shape":
            for _ in range(3):
                gemm()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if warmup == "shape":
        torch.testing.assert_close(output.cpu(), expected, rtol=0.02, atol=0.03)
        print("EAGER PASS", flush=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gemm()
    for _ in range(2):
        # Ensure replay writes the output, rather than validating stale eager data.
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0.02, atol=0.03)
    print("CAPTURE REPLAY PASS", flush=True)


@pytest.mark.parametrize("tokens", [1, 64, 128, 256])
def test_dense_fp8_graph_after_query_shape_warmup(tokens):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--tokens", str(tokens)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EAGER PASS" in result.stdout
    assert "CAPTURE REPLAY PASS" in result.stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64, choices=[1, 64, 128, 256])
    parser.add_argument(
        "--warmup", choices=["none", "profile", "shape"], default="shape"
    )
    args = parser.parse_args()
    run_case(args.tokens, args.warmup)
