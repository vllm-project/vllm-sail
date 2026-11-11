# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.requires_runtime, pytest.mark.ppu]

torch = pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

pytestmark.append(
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="fused_moe Triton kernels require CUDA",
    )
)

NUM_EXPERTS = 8
HIDDEN = 64
INTERMEDIATE = 64
TOP_K = 2


def _make_inputs(M: int):
    torch.manual_seed(42)
    hidden_states = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device="cuda")
    w1 = (
        torch.randn(
            NUM_EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.1
    )
    w2 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device="cuda"
    )
    topk_weights = torch.softmax(torch.randn(M, TOP_K, device="cuda"), dim=-1)
    return hidden_states, w1, w2, topk_weights, topk_ids


@pytest.mark.parametrize("M", [1, 16, 128])
def test_sail_modular_moe_nvtx_matches_fused_experts(M: int, monkeypatch):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    from vllm_sail.profiling import modular_moe, moe, nvtx

    assert nvtx.has_nvtx
    hidden_states, w1, w2, topk_weights, topk_ids = _make_inputs(M)
    ref_out = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
    kernel = object()
    events = []

    def delegate(self, hidden_states, w1, w2, topk_ids, topk_weights):
        assert self is kernel
        events.append(("dispatch", None))
        return fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)

    monkeypatch.setattr(modular_moe, "_original", delegate)
    monkeypatch.setattr(moe, "range_push", lambda label: events.append(("push", label)))
    monkeypatch.setattr(moe, "range_pop", lambda: events.append(("pop", None)))
    mode = "D" if torch.cuda.is_current_stream_capturing() else "P"
    nvtx_out = modular_moe.apply(kernel, hidden_states, w1, w2, topk_ids, topk_weights)

    label = (
        f"{mode}_MoE,M_{M}_E_{NUM_EXPERTS}_H_{HIDDEN}"
        f"_In_{2 * INTERMEDIATE}_topk_{TOP_K}"
    )
    assert events == [("push", label), ("dispatch", None), ("pop", None)]
    assert nvtx_out.shape == hidden_states.shape
    torch.testing.assert_close(nvtx_out, ref_out)


_SUBPROC_SCRIPT = """
import os
import sys

# Drop cwd entries so the installed packages are not shadowed by the source tree.
sys.path[:] = [p for p in sys.path if p not in ("", ".", os.getcwd())]

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
import vllm_sail.profiling as profiling

profiling.install()
assert profiling.is_enabled()
assert profiling._installed

from vllm_sail.profiling import modular_moe, moe

assert modular_moe._modular.FusedMoEKernelModularImpl.apply == modular_moe.apply

torch.manual_seed(42)
hs = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
w1 = torch.randn(8, 128, 64, dtype=torch.bfloat16, device="cuda") * 0.1
w2 = torch.randn(8, 64, 64, dtype=torch.bfloat16, device="cuda") * 0.1
ids = torch.randint(0, 8, (16, 2), dtype=torch.int32, device="cuda")
w = torch.softmax(torch.randn(16, 2, device="cuda"), dim=-1)
ref_out = fused_experts(hs, w1, w2, w, ids)
kernel = object()
events = []


def delegate(self, hidden_states, w1, w2, topk_ids, topk_weights):
    assert self is kernel
    events.append(("dispatch", None))
    return fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)


modular_moe._original = delegate
moe.range_push = lambda label: events.append(("push", label))
moe.range_pop = lambda: events.append(("pop", None))
mode = "D" if torch.cuda.is_current_stream_capturing() else "P"
out = modular_moe._modular.FusedMoEKernelModularImpl.apply(
    kernel, hs, w1, w2, ids, w
)
assert events == [
    ("push", f"{mode}_MoE,M_16_E_8_H_64_In_128_topk_2"),
    ("dispatch", None),
    ("pop", None),
]
assert out.shape == (16, 64)
torch.testing.assert_close(out, ref_out)
print("NVTX_OK")
"""


@pytest.mark.parametrize("env_var", ["VLLM_SAIL_NVTX_PROFILE", "VLLM_PPU_NVTX_PROFILE"])
def test_nvtx_enabled_via_env_end_to_end(env_var):
    env = dict(os.environ)
    for name in (
        "VLLM_SAIL_NVTX_PROFILE",
        "VLLM_PPU_NVTX_PROFILE",
        "SAIL_NVTX_PROFILE",
    ):
        env.pop(name, None)
    env[env_var] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROC_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "NVTX_OK" in proc.stdout, f"stdout={proc.stdout}\nstderr={proc.stderr}"
