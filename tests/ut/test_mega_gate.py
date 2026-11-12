# SPDX-License-Identifier: Apache-2.0
"""SAIL adaptation of the native V4.1 MegaMoE gate branch."""

import ast
import math
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import ROOT, definition, function

DEEP_GEMM = "vllm_sail/utils/deep_gemm.py"
HELPER = "vllm_sail/models/deepseek_v41/mega_gate.py"
BODY = "vllm_sail/patch/_upstream/deepseek_v41_mega_gate.py"
PATCH = "vllm_sail/patch/enhancement/models/deepseek_v41_mega_gate.py"


class Tensor:
    def __init__(self, shape, dtype="bf16", device="ppu"):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 16

    def view(self, shape):
        assert shape == self.shape
        return self


def test_native_api_preserves_rank_masks_output_order_and_owned_buffers():
    calls = []

    def api(*args, **kwargs):
        calls.append((args, kwargs))

    namespace = {
        "torch": NS(
            float32="fp32",
            int64="int64",
            empty=lambda shape, dtype, device: Tensor(shape, dtype, device),
        ),
        "_lazy_init": lambda: None,
        "_bf16_mega_gate_impl": api,
    }
    x, weight = Tensor((17, 256)), Tensor((256, 256))
    image_mask, fix_mask, logical_ids = object(), object(), object()
    weights, ids = function(DEEP_GEMM, "bf16_mega_gate", namespace)(
        x,
        weight,
        6,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        ep_rank=3,
        image_token_mask=image_mask,
        fix_routing_mask=fix_mask,
        unmapped_topk_idx=logical_ids,
    )
    assert weights.dtype == "fp32" and ids.dtype == "int64"
    args, kwargs = calls[0]
    assert args == (x, weight, 6)
    assert kwargs["out"] == (ids, weights)
    assert kwargs["ep_rank"] == 3
    assert kwargs["scoring_func"] == "sqrtsoftplus"
    assert kwargs["routed_scaling_factor"] == 2.5
    assert kwargs["image_token_mask"] is image_mask
    assert kwargs["fix_routing_mask"] is fix_mask
    assert kwargs["unmapped_topk_idx"] is logical_ids
    assert not kwargs["use_shared_as_routed"]
    assert kwargs["num_shared_experts"] == 0
    assert "to_physical_map" not in kwargs


def _eligibility():
    metadata = NS(num_prefill_tokens=0, num_decode_tokens=17)
    moe = NS(
        use_sequence_parallel=False,
        scoring_func="sqrtsoftplus",
        renormalize=True,
        routed_scaling_factor=2.5,
        n_activated_experts=6,
        gate=NS(weight=Tensor((256, 256)), e_score_correction_bias=None, bias_vl=None),
    )
    namespace = {
        "envs": NS(VLLM_SAIL_DEEPGEMM_MEGA_GATE=True),
        "vllm_envs": NS(VLLM_BATCH_INVARIANT=False),
        "current_platform": NS(
            is_ppu=lambda: True, has_device_capability=lambda sm: sm == 89
        ),
        "is_deep_gemm_supported": lambda: True,
        "has_mega_gate": lambda: True,
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: NS(attn_metadata={"indexer": metadata}),
        "torch": NS(bfloat16="bf16", float32="fp32"),
        "math": math,
        "_MAX_ROWS": 2048,
    }
    return moe, metadata, namespace


@pytest.mark.parametrize(
    "case,reason",
    [
        ("decode", None),
        ("disabled", "disabled"),
        ("missing_wheel", "unavailable"),
        ("prefill", "only decode"),
        ("sp", "sequence parallelism"),
        ("batch_invariant", "batch-invariant"),
        ("fp32", "BF16"),
        ("too_many_rows", "row limit"),
    ],
)
def test_native_gate_eligibility(case, reason):
    moe, metadata, namespace = _eligibility()
    x = Tensor((17, 256))
    if case == "disabled":
        namespace["envs"].VLLM_SAIL_DEEPGEMM_MEGA_GATE = False
    elif case == "missing_wheel":
        namespace["has_mega_gate"] = lambda: False
    elif case == "prefill":
        metadata.num_prefill_tokens = 1
    elif case == "sp":
        moe.use_sequence_parallel = True
    elif case == "batch_invariant":
        namespace["vllm_envs"].VLLM_BATCH_INVARIANT = True
    elif case == "fp32":
        x.dtype = "fp32"
    elif case == "too_many_rows":
        x.shape = (2049, 256)
    result = function(HELPER, "_fallback_reason", namespace)(moe, x)
    if reason is None:
        assert result is None
    else:
        assert reason in result


@pytest.mark.parametrize("rows", [0, 17])
def test_idle_gate_does_not_log_a_fallback(rows):
    logs = []
    namespace = {
        "_fallback_reason": lambda *args: "not eligible",
        "envs": NS(VLLM_SAIL_DEEPGEMM_MEGA_GATE=True),
        "logger": NS(info_once=lambda *args: logs.append(args)),
    }
    should_use = function(HELPER, "should_use_mega_gate", namespace)
    assert not should_use(NS(), Tensor((rows, 256)))
    assert bool(logs) == bool(rows)


@pytest.mark.parametrize(
    "experts,rows,eligible,fused",
    [
        (256, 16, True, False),
        (256, 17, True, True),
        (128, 1, True, False),
        (128, 2, True, True),
        (256, 17, False, False),
    ],
)
def test_native_forward_retains_threshold_and_expert_call(
    modules, experts, rows, eligible, fused
):
    calls = []

    class Gate:
        weight = Tensor((experts, 256))
        tid2eid = None
        e_score_correction_bias = None
        bias_vl = None

        def __call__(self, x):
            calls.append("gate")
            return "logits", None

    def route(*args, **kwargs):
        calls.append(("mega" if args else "regular", kwargs))
        return "weights", "ids"

    x = Tensor((rows, 256))

    def run_experts(*args):
        calls.append(("experts", args))
        return x

    modules(
        "vllm_sail.models.deepseek_v41.mega_gate",
        should_use_mega_gate=lambda *args: eligible,
    )
    moe = NS(
        gate=Gate(),
        use_native_mega_moe=True,
        n_activated_experts=6,
        scoring_func="sqrtsoftplus",
        renormalize=True,
        routed_scaling_factor=2.5,
        hash_indices_dtype="int64",
        ep_rank=3,
        image_sentinel_lo=0,
        experts=run_experts,
        shared_experts="shared",
    )
    metadata = NS(image_token_mask="prepared-once")
    forward = function(
        BODY, "forward", {"fused_topk_bias": route, "bf16_mega_gate": route}
    )
    assert forward(moe, x, None, metadata) is x
    routing = calls[0] if fused else calls[1]
    assert routing[0] == ("mega" if fused else "regular")
    if fused:
        assert routing[1]["ep_rank"] == 3
        assert routing[1]["image_token_mask"] == "prepared-once"
    assert calls[-1] == ("experts", (x, "weights", "ids", "shared"))


def test_patch_rebinds_native_alias_and_delegates_other_paths(modules, patch_utils_module):
    calls = []
    platform = NS(is_ppu=lambda: True)

    def upstream_api(*args, **kwargs):
        calls.append("cuda-api")
        return kwargs

    def ppu_api(*args, **kwargs):
        calls.append("ppu-api")
        return kwargs

    def upstream_forward(*args):
        return "upstream-forward"

    class V41MoE:
        forward = upstream_forward

    modules("torch")
    provider = modules("vllm.utils.deep_gemm", bf16_mega_gate=upstream_api)
    model = modules("vllm.models.deepseek_v4.nvidia.model", bf16_mega_gate=upstream_api)
    modules("vllm.models.deepseek_v41.nvidia.model", DeepseekV4MoE=V41MoE)
    modules("vllm.platforms", current_platform=platform)
    modules("vllm_sail.utils.deep_gemm", bf16_mega_gate=ppu_api)
    installed = load_patch(PATCH)
    assert provider.bf16_mega_gate is model.bf16_mega_gate is installed.bf16_mega_gate
    assert V41MoE.forward is installed.forward
    assert (
        patch_utils_module.original_of(installed.forward, installed._FORWARD_TARGET)
        is upstream_forward
    )
    for target in (
        installed._API_TARGET,
        "vllm.models.deepseek_v4.nvidia.model.bf16_mega_gate",
    ):
        assert (
            patch_utils_module.original_of(installed.bf16_mega_gate, target)
            is upstream_api
        )
    kwargs = dict(scoring_func="sqrtsoftplus", routed_scaling_factor=2.5, ep_rank=3)
    assert model.bf16_mega_gate("x", "weight", 6, **kwargs)["ep_rank"] == 3
    platform.is_ppu = lambda: False
    provider.bf16_mega_gate("x", "weight", 6, **kwargs)
    assert calls == ["ppu-api", "cuda-api"]
    assert (
        V41MoE.forward(NS(use_native_mega_moe=True), "x") == "upstream-forward"
    )
    platform.is_ppu = lambda: True
    assert (
        V41MoE.forward(NS(use_native_mega_moe=False), "x") == "upstream-forward"
    )
    with pytest.raises(RuntimeError, match="already"):
        load_patch(PATCH)


@pytest.mark.upstream_source
def test_native_body_matches_upstream_except_marked_eligibility(upstream_source_root):
    original = definition(
        upstream_source_root / "vllm/models/deepseek_v4/nvidia/model.py",
        "DeepseekV4MoE.forward",
    )
    adapted = definition(ROOT / BODY, "forward")
    adapted.body = [n for n in adapted.body if not isinstance(n, ast.ImportFrom)]
    adapted_gate = next(
        n for n in adapted.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.BoolOp)
        and isinstance(n.test.op, ast.Or)
    )
    original_gate = next(
        n for n in original.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
    )
    adapted_gate.test = original_gate.test
    assert ast.dump(adapted, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )
