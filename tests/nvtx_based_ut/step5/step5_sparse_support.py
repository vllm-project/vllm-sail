# SPDX-License-Identifier: Apache-2.0
"""Device runtime and correctness helpers from the Step5 replay suites."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

PROFILE_SWITCHES = (
    "VLLM_SAIL_NVTX_PROFILE",
    "VLLM_PPU_NVTX_PROFILE",
    "SAIL_NVTX_PROFILE",
)


class DeviceBusyError(RuntimeError):
    """Device occupancy is not a compute failure; report blocking only, never start acceptance."""

    def __init__(self, message, snapshot=None):
        super().__init__(message)
        self.snapshot = snapshot


def idle_gate(gpu_uuid=None):
    """All-device gate by default; an explicit UUID checks only that device, and a failed query still refuses measurement."""
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader",
    ]
    if gpu_uuid is not None:
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu_uuid):
            raise ValueError("a full GPU UUID is required")
        command.append("--id=" + gpu_uuid)
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=15, check=True
    )
    others = [
        line
        for line in result.stdout.splitlines()
        if line.strip()
        and str(os.getpid()) not in [part.strip() for part in line.split(",")]
    ]
    snapshot = {
        "queried_at": time.time(),
        "compute_processes": result.stdout,
        "command": result.args,
        "gpu_uuid": gpu_uuid,
    }
    if others:
        raise DeviceBusyError(
            "device has other compute processes; acceptance paused without killing them: "
            + "; ".join(others),
            snapshot,
        )
    return snapshot


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class ScoreNumericalError(AssertionError):
    """Keeps the error under the fixed tolerance; failed results must not feed the performance gate."""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def direct_cpu_oracle(data, ids):
    """Independent CPU FP32 dot and the original four FMA-chain epilogue."""
    torch = data.torch
    q = data.query.cpu().float()
    raw = data.mean.cpu()
    key = (
        raw.float()
        if raw.dtype == torch.float16
        else raw.view(torch.float8_e4m3fn).float()
    )
    weights = data.weights.cpu().float()
    expected = []
    for row in ids:
        _, _, _, _, _, owner, token, group, written = data.host[row]
        regions = torch.arange(written)
        col = regions // data.spec["rpb"]
        physical = torch.tensor(data.host_table[owner])[col]
        valid = (physical >= 0) & (physical < key.shape[0])
        keys = key[
            physical.clamp(0, key.shape[0] - 1), regions % data.spec["rpb"]
        ].clone()
        keys[~valid] = 0
        dot = (keys @ q[token, group].T).clamp_min(0)
        sums = torch.zeros((written, 4), dtype=torch.float32)
        for head_group in range(data.spec["heads"] // 4):
            first = head_group * 4
            sums = (
                dot[:, first : first + 4].double()
                * weights[token, group, first : first + 4].double()
                + sums.double()
            ).float()
        expected.append(((sums[:, 0] + sums[:, 1]) + sums[:, 2]) + sums[:, 3])
    return expected


def error_metrics(torch, actual, expected, atol=1e-4, rtol=1e-4):
    if not torch.equal(torch.isneginf(actual), torch.isneginf(expected)):
        raise AssertionError("score -inf masks are not equivalent")
    mask = torch.isfinite(expected)
    if not bool(torch.isfinite(actual[mask]).all()):
        raise AssertionError("NaN/+inf in valid scores")
    delta = (actual[mask] - expected[mask]).abs()
    relative = delta / expected[mask].abs().clamp_min(1e-12)
    result = {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "max_rel": float(relative.max()) if delta.numel() else 0.0,
        "mean_rel": float(relative.mean()) if delta.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
    }
    result["outside_tolerance"] = int(
        (delta > atol + rtol * expected[mask].abs()).sum()
    )
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise ScoreNumericalError(str(exc), result) from exc
    return result


def reject_instrumentation():
    from vllm_sail import envs

    if envs.VLLM_SAIL_NVTX_PROFILE or any(
        os.environ.get(name, "").lower() in ("1", "true", "yes", "on")
        for name in PROFILE_SWITCHES
    ):
        raise ValueError(
            "formal performance UTs must disable NVTX and its aliases; profiling runs in a separate pass"
        )


def require_runtime(implementation="default", extension_root=None):
    reject_instrumentation()
    import torch

    import vllm_sail

    if extension_root is not None:
        directory = str(Path(extension_root).resolve())
        if not Path(directory).is_dir():
            raise ValueError("extension-root directory does not exist")
        if directory not in vllm_sail.__path__:
            # Only extend the binary search path; Python sources from older wheels must not shadow the current package.
            vllm_sail.__path__.append(directory)
    if not torch.cuda.is_available():
        raise RuntimeError("no PPU device available")
    name = torch.cuda.get_device_name(0)
    if not any(word in name.upper() for word in ("PPU", "ZW-", "ZW8", "M890")):
        raise RuntimeError(f"this UT requires PPU, actual device: {name}")
    if implementation not in ("default", "flashinfer", "native"):
        raise ValueError("unknown implementation")
    os.environ["VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND"] = (
        "flashinfer" if implementation == "default" else implementation
    )
    os.environ["VLLM_SAIL_STEP5_DSA_PREFILL_PPU"] = str(int(implementation == "native"))
    vllm_sail.register()
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia import sparse_attention
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        raise RuntimeError(
            "PPU platform not registered; set VLLM_PLUGINS=ppu before process start"
        )
    package_root = os.environ.get("VLLM_SAIL_TEST_PACKAGE_ROOT")
    if package_root:
        import sys

        root = Path(package_root).resolve()
        for module_name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if module_name.startswith("vllm_sail") and path and path.endswith(".py"):
                if not Path(path).resolve().is_relative_to(root):
                    raise RuntimeError(
                        f"isolated wheel acceptance loaded a worktree module: {module_name}: {path}"
                    )
    return torch, sparse_attention


def check_selection_oracle(data, records, sample_rows=26, observed=None):
    """Standalone FP32 oracle reads the original per-row selection without relying on Q8 unions; fixed selections use this entry too."""
    torch, geo, syn = data.torch, data.case["geometry"], data.case["synthetic"]
    count = min(positive(sample_rows, "sample_rows"), geo["rows"])
    row_ids = sorted(
        {round(i * (geo["rows"] - 1) / max(count - 1, 1)) for i in range(count)}
    )
    for row in row_ids:
        outer = data.case.get("entry") == "forward"
        token = (
            row % geo["tokens"]
            if outer
            else row // geo["groups"]
            if data.case["pd_type"] == "decode"
            else row
        )
        group = (
            row // geo["tokens"]
            if outer
            else row % geo["groups"]
            if data.case["pd_type"] == "decode"
            else 0
        )
        owner = observed["owners"][row] if observed else data.host_owners[token]
        pos = observed["positions"][row] if observed else data.host_positions[token]
        active = observed["active"][row] if observed else True
        tokens = sorted(
            {
                region * 8 + j
                for region in records[row]
                if active and region >= 0
                for j in range(8)
                if region * 8 + j <= pos and region * 8 + j < syn["seq_lens"][owner]
            }
        )
        tokens = [
            t
            for t in tokens
            if t // geo["block_size"] < len(data.host_tables[owner])
            and 0 <= data.host_tables[owner][t // geo["block_size"]] < geo["pages"]
        ]
        head_slice = slice(group * geo["heads"], (group + 1) * geo["heads"])
        q = (
            (
                data.tensors["query"][token, head_slice]
                if outer
                else data.tensors["query"][row]
            )
            .float()
            .cpu()
        )
        if tokens:
            physical = [data.host_tables[owner][t // geo["block_size"]] for t in tokens]
            offsets = [t % geo["block_size"] for t in tokens]
            cache = data.tensors["kv_cache"]
            if cache.dtype in (torch.uint8, torch.float8_e4m3fn):
                packed = (
                    cache.view(torch.uint8)[physical, group, offsets]
                    .cpu()
                    .view(torch.float8_e4m3fn)
                    .float()
                )
                packed[:, :192] *= syn["k_scale"]
                packed[:, 192:] *= syn["v_scale"]
                packed = packed.to(torch.bfloat16).float()
            else:
                packed = cache[physical, group, offsets].float().cpu()
            expected = (q @ packed[:, :192].T * data.impl.scale).softmax(-1) @ packed[
                :, 192:
            ]
        else:
            expected = torch.zeros_like(q)
        gate = data.tensors.get("gate")
        if gate is not None:
            expected *= (
                gate[token, group * geo["heads"] : (group + 1) * geo["heads"]]
                .float()
                .cpu()
                .sigmoid()
                .unsqueeze(-1)
            )
        actual = (
            data.tensors["output"][token, head_slice]
            if outer
            else data.tensors["output"][row]
        )
        torch.testing.assert_close(
            actual.float().cpu(), expected, atol=0.025, rtol=0.025
        )
    return row_ids
