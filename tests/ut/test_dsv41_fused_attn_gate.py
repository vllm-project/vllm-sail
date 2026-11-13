# SPDX-License-Identifier: Apache-2.0
"""Offline UT for the PR#229 fused norm-rope gate wiring.

CPU-only tier: no torch, no vLLM, no device. Covers the three pieces of
PR#229 that are decidable without a kernel:

* ``vllm_sail.envs`` parsing of the two new switches
  (``VLLM_SAIL_FUSED_NORM_ROPE_ATTN`` / ``VLLM_SAIL_PPU_KV_MXFP8``)
  including their auto-derived ``VLLM_PPU_*`` aliases and SAIL precedence;
* ``fused_gate`` in ``vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py``:
  every gate branch (env, kv-record, platform, capability, wheel probe,
  topology) and the fully-open report;
* the ``dsv41_kv_record`` patch module: non-PPU delegation, once-per-process
  env caching, the forensic log line, and the patch metadata (target,
  affected_versions against the repo's SUPPORTED_RANGE);
* the ``_select_dsv4_attn_cls`` selector patch: which attention class is
  returned for each (platform, selected, gate) combination.

Torch-dependent bodies (``_o_proj``, ``_dequant_fused_output``,
``_forward_{decode,prefill}_mega``) are intentionally out of scope here;
they are covered by the FlashMLA PR#54 kernel UTs plus on-device smoke.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.upstream import SUPPORTED_RANGE
from vllm_sail import envs

_REAL_FIND_SPEC = importlib.util.find_spec

_SWITCH_NAMES = (
    "VLLM_SAIL_FUSED_NORM_ROPE_ATTN",
    "VLLM_PPU_FUSED_NORM_ROPE_ATTN",
    "VLLM_SAIL_PPU_KV_MXFP8",
    "VLLM_PPU_PPU_KV_MXFP8",
    "VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE",
    "VLLM_PPU_PPU_FUSED_OUTPUT_CHANNELWISE",
)


class _UpstreamMegaAttnStub:
    """Stand-in for upstream's ``DeepseekV4MegaAttnAttention``.

    Carries the attributes the PPU overrides read (topology, block size) and a
    ``_alloc_attn_out`` that returns a sentinel, so the channel-wise override
    can be checked for delegating to ``super()`` when it is off.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._o_proj_block_size = 128
        self.n_wv_group = 1
        self.n_local_groups = 1
        self.o_lora_rank = 4

    def _alloc_attn_out(self, num_tokens, hidden_states):
        return "super-alloc"


def _load_attention_module(modules, monkeypatch, *, real_torch: bool = False):
    """Load ``flash_mla_mega_attn.py`` against the stubbed vLLM module tree.

    ``real_torch=False`` keeps the module on the CPU-only tier's torch stub.
    ``real_torch=True`` binds the real (CPU) torch instead, which the tensor
    level helpers need; vLLM and any device stay stubbed either way.
    """
    logs = []
    logger = SimpleNamespace(info_once=lambda msg, *a: logs.append((msg, a)))
    if real_torch:
        monkeypatch.setitem(sys.modules, "torch", pytest.importorskip("torch"))
    else:
        modules("torch")
    modules("vllm.logger", init_logger=lambda name: logger)
    modules(
        "vllm.model_executor.layers.fusion.quant_activation",
        QuantizedActivation=make_dataclass(
            "QuantizedActivation",
            ["data", "scale", "orig_dtype", "orig_shape", "quant_key"],
        ),
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda weight, scale: weight,
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        kMxfp8Dynamic=object(),
    )
    modules(
        "vllm.models.deepseek_v41.common.ops",
        combine_topk_swa_indices=lambda *a, **k: None,
        dequantize_and_gather_k_cache=lambda *a, **k: None,
    )
    modules("vllm.models.deepseek_v41.common.ops.fused_layout", WV_GROUP_SIZE=8)
    modules(
        "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
        _HEAD_DIM_V=512,
        _QUANT_GROUP=32,
        _ROPE_DIM=64,
        _token_slice=lambda out, start, end: out,
        DeepseekV4MegaAttnAttention=_UpstreamMegaAttnStub,
    )
    modules(
        "vllm.models.deepseek_v41.sparse_mla",
        DeepseekV4FlashMLABackend=type("DeepseekV4FlashMLABackend", (), {}),
    )
    modules("vllm.utils.math_utils", round_up=lambda x, y: -(-x // y) * y)
    modules("vllm.v1.worker.workspace", current_workspace_manager=lambda: None)
    platform = SimpleNamespace(
        is_ppu=lambda: True,
        get_device_capability=lambda device_id=0: SimpleNamespace(major=8, minor=9),
    )
    modules("vllm.platforms", current_platform=platform)
    module = load_patch("vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py")
    return SimpleNamespace(module=module, platform=platform, logs=logs)


@pytest.fixture(autouse=True)
def _clean_switch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SWITCH_NAMES:
        monkeypatch.delenv(name, raising=False)


def _config(heads: int = 64, groups: int = 8, tp: int = 8, *, nested: bool = False):
    hf = SimpleNamespace(num_attention_heads=heads, o_groups=groups)
    if nested:
        hf = SimpleNamespace(text_config=hf, num_attention_heads=None, o_groups=None)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


# ---------------------------------------------------------------------------
# env switch parsing
# ---------------------------------------------------------------------------


def test_new_switches_default_off() -> None:
    """Opt-in contract: unconfigured processes keep the baseline path."""
    assert envs.VLLM_SAIL_FUSED_NORM_ROPE_ATTN is False
    assert envs.VLLM_SAIL_PPU_KV_MXFP8 is False
    snapshot = envs.snapshot()
    assert snapshot["VLLM_SAIL_FUSED_NORM_ROPE_ATTN"] is False
    assert snapshot["VLLM_SAIL_PPU_KV_MXFP8"] is False


@pytest.mark.parametrize(
    ("sail", "ppu"),
    [
        ("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "VLLM_PPU_FUSED_NORM_ROPE_ATTN"),
        ("VLLM_SAIL_PPU_KV_MXFP8", "VLLM_PPU_PPU_KV_MXFP8"),
    ],
)
def test_switches_turn_on_via_either_name(
    monkeypatch: pytest.MonkeyPatch, sail: str, ppu: str
) -> None:
    for name in (sail, ppu):
        monkeypatch.setenv(name, "1")
        assert getattr(envs, sail) is True
        assert getattr(envs, ppu) is True
        monkeypatch.delenv(name)
        assert getattr(envs, sail) is False


@pytest.mark.parametrize(
    ("sail", "ppu"),
    [
        ("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "VLLM_PPU_FUSED_NORM_ROPE_ATTN"),
        ("VLLM_SAIL_PPU_KV_MXFP8", "VLLM_PPU_PPU_KV_MXFP8"),
    ],
)
def test_sail_name_takes_precedence_over_ppu_alias(
    monkeypatch: pytest.MonkeyPatch, sail: str, ppu: str
) -> None:
    monkeypatch.setenv(ppu, "0")
    monkeypatch.setenv(sail, "1")
    assert getattr(envs, sail) is True
    monkeypatch.setenv(sail, "0")
    monkeypatch.setenv(ppu, "1")
    assert getattr(envs, ppu) is False


# ---------------------------------------------------------------------------
# fused_gate
# ---------------------------------------------------------------------------


@pytest.fixture
def gate(modules, monkeypatch):
    loaded = _load_attention_module(modules, monkeypatch)
    module, platform, logs = loaded.module, loaded.platform, loaded.logs
    # Opt-in contract: gate tests explicitly enable both switches; the
    # default-off behaviour is asserted separately below.
    monkeypatch.setenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "1")
    monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "1")

    def set_capability(value):
        monkeypatch.setattr(
            platform, "get_device_capability", lambda device_id=0: value
        )

    def set_wheel(state: str) -> None:
        """state: missing | api | no-api | no-decode | import-error"""
        if state == "missing":
            monkeypatch.setattr(
                importlib.util,
                "find_spec",
                lambda name, *a, **k: (
                    None if name == "flash_mla" else _REAL_FIND_SPEC(name, *a, **k)
                ),
            )
            return
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: (
                SimpleNamespace(name=name)
                if name == "flash_mla"
                else _REAL_FIND_SPEC(name, *a, **k)
            ),
        )
        if state == "api":
            modules(
                "flash_mla",
                fused_norm_rope_attn_rope_cast=SimpleNamespace(
                    prefill=object(), decode=object()
                ),
            )
        elif state == "no-api":
            modules("flash_mla")
        elif state == "no-decode":
            modules(
                "flash_mla",
                fused_norm_rope_attn_rope_cast=SimpleNamespace(prefill=object()),
            )
        elif state == "import-error":
            monkeypatch.setitem(
                __import__("sys").modules, "flash_mla", None
            )  # import of flash_mla halts -> ImportError
        else:  # pragma: no cover - guard against typos in tests
            raise AssertionError(f"unknown wheel state {state!r}")

    return SimpleNamespace(
        module=module,
        platform=platform,
        logs=logs,
        set_capability=set_capability,
        set_wheel=set_wheel,
        monkeypatch=monkeypatch,
    )


def test_gate_closed_by_default_without_env_opt_in(gate) -> None:
    """Requirement: the fused path never runs on an unconfigured process."""
    gate.monkeypatch.delenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN")
    gate.monkeypatch.delenv("VLLM_SAIL_PPU_KV_MXFP8")
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-env:")
    assert (
        gate.module.PPUDeepseekV4MegaAttnAttention.is_available_for(_config()) is False
    )


def test_gate_env_switch_closes_first(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "0")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-env:")


def test_gate_kv_record_switch_closes(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "0")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-kvrecord:")


def test_gate_non_ppu_platform_closes(gate) -> None:
    gate.platform.is_ppu = lambda: False
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-platform:")


@pytest.mark.parametrize("cap", [None, (9, 0), (8, 6), (10, 0)])
def test_gate_capability_outside_ppu_sm80_sm89_closes(gate, cap) -> None:
    value = None if cap is None else SimpleNamespace(major=cap[0], minor=cap[1])
    gate.set_capability(value)
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-cap:")


@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
def test_gate_accepts_both_ppu_capabilities(gate, cap) -> None:
    gate.set_capability(SimpleNamespace(major=cap[0], minor=cap[1]))
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert ok
    assert report.startswith(f"all gates open: cap={cap[0]}.{cap[1]} ")


def test_gate_missing_wheel_closes(gate) -> None:
    gate.set_wheel("missing")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report == "gate-wheel: flash_mla not installed"


@pytest.mark.parametrize("state", ["no-api", "no-decode"])
def test_gate_baseline_wheel_without_fused_api_closes(gate, state) -> None:
    gate.set_wheel(state)
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith(
        "gate-wheel: flash_mla lacks fused_norm_rope_attn_rope_cast"
    )


def test_gate_wheel_import_error_closes_with_repr(gate) -> None:
    gate.set_wheel("import-error")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-wheel: flash_mla import failed:")


@pytest.mark.parametrize(
    ("heads", "groups", "tp"),
    [
        (60, 8, 8),  # heads not divisible by tp
        (64, 4, 8),  # o_groups not divisible by tp
        (64, 8, 16),  # local_heads=4 < WV_GROUP_SIZE
        (64, 16, 8),  # 4 local heads per local group, need exactly 8
        (64, 0, 8),  # no o_groups at all
    ],
)
def test_gate_topology_mismatch_closes(gate, heads, groups, tp) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(heads, groups, tp))
    assert not ok
    assert report.startswith("gate-topology:")


def test_gate_fully_open_tp8_reports_padded_heads(gate) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert ok
    assert "tp=8 local_heads=8 padded_heads=64" in report


def test_gate_reads_nested_text_config(gate) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(nested=True))
    assert ok
    assert "local_heads=8" in report


def test_ppu_class_contract(gate) -> None:
    cls = gate.module.PPUDeepseekV4MegaAttnAttention
    # backend stays FLASHMLA_SPARSE_DSV41, KV record stays fp8_ds_mla.
    assert cls.backend_cls is gate.module.DeepseekV4FlashMLABackend
    instance = cls()
    assert instance.packed_kv_cache_dtype == "fp8_ds_mla"
    # v1.1 pins: per-32 packed UE8M0 recipe, no TMA-aligned scales.
    assert instance._einsum_recipe == (1, 1, 32)
    assert instance._tma_aligned_scales is False
    assert gate.module._PPU_EINSUM_RECIPE == (1, 1, 32)
    # is_available_for mirrors fused_gate.
    gate.set_wheel("api")
    assert cls.is_available_for(_config()) is True
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "0")
    assert cls.is_available_for(_config()) is False


# ---------------------------------------------------------------------------
# dsv41_kv_record patch module
# ---------------------------------------------------------------------------


@pytest.fixture
def kv_record(modules, patch_utils_module, monkeypatch):
    logs = []
    logger = SimpleNamespace(info=lambda msg, *a: logs.append((msg, a)))
    modules("vllm.logger", init_logger=lambda name: logger)
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return "upstream-record"

    attention_stub = modules(
        "vllm.models.deepseek_v41.attention",
        _use_v41_mxfp8_kv_record=upstream,
    )
    platform = SimpleNamespace(is_ppu=lambda: True)
    modules("vllm.platforms", current_platform=platform)
    module = load_patch("vllm_sail/patch/enhancement/attention/dsv41_kv_record.py")
    return SimpleNamespace(
        module=module,
        patched=module._use_v41_mxfp8_kv_record,
        attention_stub=attention_stub,
        platform=platform,
        logs=logs,
        upstream_calls=upstream_calls,
        registry=patch_utils_module.PATCH_REGISTRY,
        marker=patch_utils_module.PATCH_MARKER,
        monkeypatch=monkeypatch,
    )


def test_kv_record_non_ppu_delegates_upstream(kv_record) -> None:
    kv_record.platform.is_ppu = lambda: False
    assert kv_record.patched() == "upstream-record"
    assert kv_record.upstream_calls == [1]
    assert kv_record.logs == []


def test_kv_record_ppu_default_false_with_forensic_log(kv_record) -> None:
    assert kv_record.patched() is False
    assert kv_record.upstream_calls == []
    assert kv_record.logs == [
        ("[VLLM_SAIL] kv_mxfp8=%s kv_bytes_per_token=%d", (False, 584))
    ]
    # Cached: a second call neither re-reads nor re-logs.
    assert kv_record.patched() is False
    assert len(kv_record.logs) == 1


def test_kv_record_opt_in_selects_528(kv_record) -> None:
    kv_record.monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "1")
    assert kv_record.patched() is True
    assert kv_record.logs == [
        ("[VLLM_SAIL] kv_mxfp8=%s kv_bytes_per_token=%d", (True, 528))
    ]


def test_kv_record_env_read_once_per_process(kv_record) -> None:
    assert kv_record.patched() is False
    kv_record.monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "1")
    assert kv_record.patched() is False


def test_kv_record_patch_metadata(kv_record) -> None:
    target = "vllm.models.deepseek_v41.attention._use_v41_mxfp8_kv_record"
    # The module attribute was replaced on the (stubbed) upstream module.
    assert kv_record.attention_stub._use_v41_mxfp8_kv_record is kv_record.patched
    # The replacement remembers the original for drift tooling.
    originals = getattr(kv_record.patched, kv_record.marker, None)
    assert originals is not None and target in originals
    assert callable(originals[target])
    records = [r for r in kv_record.registry if r.target == target]
    assert records, "dsv41_kv_record did not register a PatchRecord"
    record = records[-1]
    assert record.kind == "module"
    assert record.affected_versions == SUPPORTED_RANGE, (
        "dsv41_kv_record must declare the repo's supported vLLM range "
        f"({SUPPORTED_RANGE}), got {record.affected_versions!r}"
    )
    assert record.remove_when.strip()
    assert record.reason.strip()


# ---------------------------------------------------------------------------
# _select_dsv4_attn_cls selector patch
# ---------------------------------------------------------------------------


@pytest.fixture
def selector(modules, patch_utils_module):
    logs = []
    logger = SimpleNamespace(info_once=lambda msg, *a: logs.append((msg, a)))
    modules("vllm.logger", init_logger=lambda name: logger)
    upstream_flashmla = type("UpstreamFlashMLAAttention", (), {})
    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla",
        DeepseekV4FlashMLAAttention=upstream_flashmla,
    )
    state = {"selected": upstream_flashmla, "calls": []}

    def original(vllm_config):
        state["calls"].append(vllm_config)
        return state["selected"]

    def original_scale_name(vllm_config, expert_dtype):
        return "upstream-scale-name"

    model_stub = modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _select_dsv4_attn_cls=original,
        _linear_scale_param_name=original_scale_name,
    )
    platform = SimpleNamespace(is_ppu=lambda: True)
    modules("vllm.platforms", current_platform=platform)
    baseline = type("DeepseekV41FlashMLAAttention", (), {})
    modules(
        "vllm_sail.models.deepseek_v41.flashmla",
        DeepseekV41FlashMLAAttention=baseline,
    )
    upstream_mega = type("DeepseekV4MegaAttnAttention", (), {})
    modules(
        "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
        DeepseekV4MegaAttnAttention=upstream_mega,
    )
    ppu_fused = type("PPUDeepseekV4MegaAttnAttention", (), {})
    gate_state = {"result": (True, "all gates open: test"), "calls": []}

    def fake_gate(vllm_config):
        gate_state["calls"].append(vllm_config)
        return gate_state["result"]

    modules(
        "vllm_sail.models.deepseek_v41.flash_mla_mega_attn",
        PPUDeepseekV4MegaAttnAttention=ppu_fused,
        fused_gate=fake_gate,
    )
    module = load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")
    return SimpleNamespace(
        select=module._select_dsv4_attn_cls,
        model_stub=model_stub,
        platform=platform,
        logs=logs,
        state=state,
        gate_state=gate_state,
        registry=patch_utils_module.PATCH_REGISTRY,
        marker=patch_utils_module.PATCH_MARKER,
        classes=SimpleNamespace(
            upstream_flashmla=upstream_flashmla,
            upstream_mega=upstream_mega,
            baseline=baseline,
            ppu_fused=ppu_fused,
        ),
    )


def test_selector_non_ppu_passthrough(selector) -> None:
    selector.platform.is_ppu = lambda: False
    other = type("Other", (), {})
    selector.state["selected"] = other
    config = object()
    assert selector.select(config) is other
    assert selector.state["calls"] == [config]
    assert selector.gate_state["calls"] == []


def test_selector_ppu_gate_open_returns_fused_class(selector) -> None:
    config = object()
    assert selector.select(config) is selector.classes.ppu_fused
    assert selector.gate_state["calls"] == [config]
    assert any("PPUDeepseekV4MegaAttnAttention" in msg for msg, _ in selector.logs)


def test_selector_ppu_gate_closed_falls_back_to_baseline(selector) -> None:
    selector.gate_state["result"] = (False, "gate-cap: capability (9,0)")
    assert selector.select(object()) is selector.classes.baseline
    assert any("gate closed" in msg for msg, _ in selector.logs)


def test_selector_upstream_mega_selection_also_gated(selector) -> None:
    selector.state["selected"] = selector.classes.upstream_mega
    assert selector.select(object()) is selector.classes.ppu_fused
    selector.gate_state["result"] = (False, "gate-wheel: flash_mla not installed")
    assert selector.select(object()) is selector.classes.baseline


def test_selector_baseline_selection_kept_when_gate_closed(selector) -> None:
    selector.state["selected"] = selector.classes.baseline
    selector.gate_state["result"] = (False, "gate-env: off")
    assert selector.select(object()) is selector.classes.baseline
    assert len(selector.gate_state["calls"]) == 1


def test_selector_foreign_class_never_gated(selector) -> None:
    other = type("FlashInferAttention", (), {})
    selector.state["selected"] = other
    assert selector.select(object()) is other
    assert selector.gate_state["calls"] == []


def test_selector_patch_metadata(selector) -> None:
    target = "vllm.models.deepseek_v41.nvidia.model._select_dsv4_attn_cls"
    assert selector.model_stub._select_dsv4_attn_cls is selector.select
    originals = getattr(selector.select, selector.marker, None)
    assert originals is not None and target in originals
    records = [r for r in selector.registry if r.target == target]
    assert records, "selector patch did not register a PatchRecord"
    assert records[-1].affected_versions == SUPPORTED_RANGE


# ---------------------------------------------------------------------------
# int32 positions cast regression (runtime bug found on lingjun-08, 2026-11-12)
# ---------------------------------------------------------------------------


def test_forward_mega_methods_cast_positions_to_int32() -> None:
    """vllm main (8b10bda) forwards the runner's int64 positions unchanged.

    The PPU fused kernel enforces int32 (``KU_CHECK_DTYPE(token_positions,
    torch::kInt32)``), so both PPU forward methods must defensively cast
    before the kernel call; without it the server dies in warmup with
    ``RuntimeError: token_positions must have dtype torch::kInt32``.
    """
    import ast

    from tests.support.source import ROOT

    source = (ROOT / "vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "PPUDeepseekV4MegaAttnAttention"
    )
    for name in ("_forward_decode_mega", "_forward_prefill_mega"):
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        body_source = ast.get_source_segment(source, method) or ""
        assert "positions_int32.to(torch.int32)" in body_source, (
            f"{name} lost the defensive int32 cast for token_positions; the "
            "PPU fused kernel rejects int64 positions (seen on lingjun-08)."
        )


# ---------------------------------------------------------------------------
# channel-wise (per-token FP32) output scales — FlashMLA PR#59 macro=1
# ---------------------------------------------------------------------------
#
# The switch pairs a runtime env var with a wheel compiled with
# FLASHMLA_FUSED_OUTPUT_CHANNELWISE=1. FlashMLA cannot be probed at runtime,
# so the tests below pin: the env wiring, the SM89-only gate, the buffer and
# dequantization contract, the call-argument matrix, and the actionable
# fail-fast when the wheel macro and the switch disagree.


@pytest.fixture
def tensor_gate(modules, monkeypatch):
    """The attention module loaded against the real (CPU) torch."""
    loaded = _load_attention_module(modules, monkeypatch, real_torch=True)
    monkeypatch.setenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "1")
    monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "1")
    return SimpleNamespace(
        module=loaded.module,
        logs=loaded.logs,
        torch=sys.modules["torch"],
        monkeypatch=monkeypatch,
    )


def test_channelwise_switch_defaults_off() -> None:
    assert envs.VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE is False
    assert envs.VLLM_PPU_PPU_FUSED_OUTPUT_CHANNELWISE is False
    assert envs.snapshot()["VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE"] is False


@pytest.mark.parametrize(
    "sail, ppu",
    [
        (
            "VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE",
            "VLLM_PPU_PPU_FUSED_OUTPUT_CHANNELWISE",
        )
    ],
)
def test_channelwise_switch_turns_on_via_either_name(
    monkeypatch: pytest.MonkeyPatch, sail: str, ppu: str
) -> None:
    for name in (sail, ppu):
        monkeypatch.setenv(name, "1")
        assert getattr(envs, sail) is True
        assert getattr(envs, ppu) is True
        monkeypatch.delenv(name)
        assert getattr(envs, sail) is False


def test_channelwise_sail_name_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sail = "VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE"
    ppu = "VLLM_PPU_PPU_FUSED_OUTPUT_CHANNELWISE"
    monkeypatch.setenv(ppu, "0")
    monkeypatch.setenv(sail, "1")
    assert envs.VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE is True
    monkeypatch.setenv(sail, "0")
    monkeypatch.setenv(ppu, "1")
    assert envs.VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE is False


def test_gate_channelwise_opens_on_sm89_with_macro_report(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE", "1")
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert ok
    assert "out_scale=per_token_fp32" in report
    assert "required_flashmla_macro=1" in report


def test_gate_channelwise_closes_on_sm80(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE", "1")
    gate.set_capability(SimpleNamespace(major=8, minor=0))
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-outscale-cap:")
    assert "capability 8.9" in report


def test_gate_default_report_still_per32(gate) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert ok
    assert "out_scale=per32_ue8m0" in report
    assert "required_flashmla_macro=0" in report


@pytest.mark.parametrize(
    ("channelwise", "expected"),
    [
        (True, (4096, False, False, False)),
        (False, (32, True, True, True)),
    ],
)
def test_out_scale_args_matrix(gate, channelwise, expected) -> None:
    assert gate.module._out_scale_args(channelwise) == expected
    # The kernel hardcodes 4096 channels per WV group; the module derives it
    # from the geometry (8 heads x 512) and asserts the two agree.
    assert gate.module._PPU_OUT_SCALE_CHANNELS == 4096


def test_alloc_attn_out_channelwise_allocates_fp32_scale(tensor_gate) -> None:
    tensor_gate.monkeypatch.setenv("VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE", "1")
    cls = tensor_gate.module.PPUDeepseekV4MegaAttnAttention
    out = cls()._alloc_attn_out(3, tensor_gate.torch.empty(0))
    assert out.data.shape == (3, 1, 4096)
    assert out.data.dtype == tensor_gate.torch.float8_e4m3fn
    assert out.scale.shape == (3, 1, 1)
    assert out.scale.dtype == tensor_gate.torch.float32
    assert out.scale.is_contiguous()
    assert out.orig_dtype == tensor_gate.torch.bfloat16
    assert out.orig_shape == out.data.shape
    # The quant key stays upstream's per-32 key: on PPU the only consumer is
    # the BF16 bmm in _o_proj, which never reads it.
    assert out.quant_key is tensor_gate.module.kMxfp8Dynamic


def test_alloc_attn_out_default_delegates_to_super(gate) -> None:
    cls = gate.module.PPUDeepseekV4MegaAttnAttention
    instance = cls()
    assert instance._output_channelwise is False
    assert instance._alloc_attn_out(3, SimpleNamespace(device="cpu")) == "super-alloc"


def test_alloc_attn_out_reads_switch_at_construction(tensor_gate) -> None:
    tensor_gate.monkeypatch.setenv("VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE", "1")
    on = tensor_gate.module.PPUDeepseekV4MegaAttnAttention()
    assert on._output_channelwise is True
    tensor_gate.monkeypatch.delenv("VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE")
    off = tensor_gate.module.PPUDeepseekV4MegaAttnAttention()
    assert off._output_channelwise is False


def test_dequant_per_token_applies_one_scale_per_group(tensor_gate) -> None:
    torch = tensor_gate.torch
    module = tensor_gate.module
    data = torch.tensor(
        [[[1.5, -2.0, 0.5, 3.0]], [[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    # One FP32 scale per (token, group): 2^-1 for token 0, 3.0 for token 1.
    scale = torch.tensor([[[0.5]], [[3.0]]], dtype=torch.float32)
    out = module._dequant_fused_output(data, scale)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 1, 4)
    expected = data.to(torch.float32) * scale
    assert torch.allclose(out.float(), expected, rtol=2**-8, atol=0)
    # Spot values on the power-of-two scale are exact: no bf16 rounding.
    assert out[0, 0, 0].item() == 0.75
    assert out[0, 0, 1].item() == -1.0


def test_dequant_per_token_rejects_non_singleton_scale_dim(tensor_gate) -> None:
    torch = tensor_gate.torch
    data = torch.ones((1, 1, 8), dtype=torch.float8_e4m3fn)
    scale = torch.ones((1, 1, 2), dtype=torch.float32)
    with pytest.raises(AssertionError, match=r"channel-wise.*scale shape"):
        tensor_gate.module._dequant_fused_output(data, scale)


def test_dequant_per32_ue8m0_path_unchanged(tensor_gate) -> None:
    torch = tensor_gate.torch
    module = tensor_gate.module
    data = torch.ones((1, 1, 128), dtype=torch.float8_e4m3fn)
    # Four UE8M0 exponent bytes per int32 word, one per 32-dim block:
    # 127 -> 2^0, 128 -> 2^1, 126 -> 2^-1, 127 -> 2^0.
    packed = torch.tensor([[[127, 128, 126, 127]]], dtype=torch.uint8).view(torch.int32)
    out = module._dequant_fused_output(data, packed)
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 1, 128)
    expected = torch.cat(
        [
            torch.full((32,), 1.0),
            torch.full((32,), 2.0),
            torch.full((32,), 0.5),
            torch.full((32,), 1.0),
        ]
    )
    assert torch.equal(out[0, 0].float(), expected)


def test_dequant_rejects_unknown_scale_dtype(tensor_gate) -> None:
    torch = tensor_gate.torch
    data = torch.ones((1, 1, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones((1, 1, 1), dtype=torch.int64)
    with pytest.raises(AssertionError, match="must be int32 .* or float32"):
        tensor_gate.module._dequant_fused_output(data, scale)


def _o_proj_instance(module, torch, *, channelwise: bool, scale_dtype, scale_shape):
    instance = module.PPUDeepseekV4MegaAttnAttention()
    instance._output_channelwise = channelwise
    instance.wo_a = SimpleNamespace(
        weight=torch.zeros((4, 128), dtype=torch.bfloat16), weight_scale=None
    )
    instance._wo_b_proj = lambda z: z
    data = torch.ones((2, 1, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones(scale_shape, dtype=scale_dtype)
    attn_out = module.QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=module.kMxfp8Dynamic,
    )
    return instance, attn_out


def test_o_proj_channelwise_accepts_fp32_singleton_scale(tensor_gate) -> None:
    instance, attn_out = _o_proj_instance(
        tensor_gate.module,
        tensor_gate.torch,
        channelwise=True,
        scale_dtype=tensor_gate.torch.float32,
        scale_shape=(2, 1, 1),
    )
    out = instance._o_proj(attn_out, None)
    assert out.shape == (2, 4)
    assert out.dtype == tensor_gate.torch.bfloat16
    assert tensor_gate.logs, "the first-call ledger line must still be emitted"
    assert "output_channelwise=%s" in tensor_gate.logs[-1][0]
    assert tensor_gate.logs[-1][1][-1] is True


def test_o_proj_channelwise_rejects_per32_int32_scale(tensor_gate) -> None:
    instance, attn_out = _o_proj_instance(
        tensor_gate.module,
        tensor_gate.torch,
        channelwise=True,
        scale_dtype=tensor_gate.torch.int32,
        scale_shape=(2, 1, 1),
    )
    with pytest.raises(AssertionError, match="per-token FP32"):
        instance._o_proj(attn_out, None)


def test_o_proj_default_rejects_fp32_scale(tensor_gate) -> None:
    instance, attn_out = _o_proj_instance(
        tensor_gate.module,
        tensor_gate.torch,
        channelwise=False,
        scale_dtype=tensor_gate.torch.float32,
        scale_shape=(2, 1, 32),
    )
    with pytest.raises(AssertionError, match="per-32 packed UE8M0"):
        instance._o_proj(attn_out, None)


def test_o_proj_channelwise_rejects_non_singleton_scale_shape(tensor_gate) -> None:
    instance, attn_out = _o_proj_instance(
        tensor_gate.module,
        tensor_gate.torch,
        channelwise=True,
        scale_dtype=tensor_gate.torch.float32,
        scale_shape=(2, 1, 4),
    )
    with pytest.raises(AssertionError, match=r"scale shape.*per-token FP32"):
        instance._o_proj(attn_out, None)


@pytest.mark.parametrize("channelwise", [True, False])
def test_macro_mismatch_hint_names_required_macro(gate, channelwise) -> None:
    required = 1 if channelwise else 0
    exc = RuntimeError(
        "num_per_channels must match this FLASHMLA_FUSED_OUTPUT_CHANNELWISE="
        f"{1 - required} build"
    )
    hint = gate.module._macro_mismatch_hint(exc, channelwise)
    assert hint is not None
    assert f"VLLM_SAIL_PPU_FUSED_OUTPUT_CHANNELWISE={required}" in hint
    assert f"FLASHMLA_FUSED_OUTPUT_CHANNELWISE={required}" in hint
    assert "compile-time constant" in hint
    # The hint must stay portable: no local build fingerprints.
    assert "sha256" not in hint


def test_macro_mismatch_hint_passes_other_errors_through(gate) -> None:
    assert gate.module._macro_mismatch_hint(RuntimeError("boom"), True) is None


def test_source_channelwise_helpers_and_default_quad_are_pinned() -> None:
    """Source-level guards for what the offline tier cannot execute."""
    from tests.support.source import ROOT

    source = (ROOT / "vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py").read_text()
    # Both fused call sites share the argument matrix (definition + 2 callers).
    assert source.count("_out_scale_args(") == 3
    assert source.count("_macro_mismatch_hint(exc, self._output_channelwise)") == 2
    # The default (per-32) quartet and the ledger recipe stay as they were.
    assert "return (_QUANT_GROUP, True, True, True)" in source
    assert "_PPU_EINSUM_RECIPE = (1, 1, 32)" in source
    # The output copy stays layout-agnostic (element-wise copy_): decode copies
    # into ``out`` and each prefill chunk into ``chunk_out``.
    assert source.count(".data.copy_(out_fp8)") == 2
    assert source.count(".scale.copy_(out_sf)") == 2
    assert source.count("chunk_out.scale.copy_(out_sf)") == 1
