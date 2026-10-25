# SPDX-License-Identifier: Apache-2.0
"""Tests for ``vllm_sail.native.stubs``, the runtime honesty layer.

Only the pure surface is exercised: ``unsupported_ops`` and ``load_schemas`` are
stdlib-only, while ``install`` registers into ``torch.library`` and therefore
belongs to the hardware suite. The diagnostic text is tested directly because it
is the whole point of the module — a maintainer reading a NotImplementedError
must learn which op is missing, why, and where it is tracked.
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

import pytest

from vllm_sail.native import manifest, stubs


def test_unsupported_ops_are_derived_from_the_manifest() -> None:
    assert stubs.unsupported_ops() == tuple(
        name for name in manifest.load().excluded_ops() if name != "merge_attn_states"
    )


def test_unsupported_ops_include_the_accepted_cutlass_regression() -> None:
    assert "cutlass_scaled_mm" in stubs.unsupported_ops()


_SCHEMAS = """\
[[op]]
namespace = "_C"
name = "cutlass_scaled_mm"
schema = "cutlass_scaled_mm(Tensor! out, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()"
kind = "compute"
tier = "x"
reason = "CUTLASS / CuTe in the include closure"

[[op]]
namespace = "_C"
name = "cutlass_scaled_mm_supports_fp8"
schema = "cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool"
kind = "probe"
tier = "x"
reason = "CUTLASS / CuTe in the include closure"

[[op]]
namespace = "_moe_C"
name = "shuffle_rows"
schema = "shuffle_rows(Tensor input_tensor, Tensor dst2src_map, Tensor! output_tensor) -> ()"
"""


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    src = tmp_path / "excluded_ops.toml"
    src.write_text(textwrap.dedent(_SCHEMAS), encoding="utf-8")
    return src


def test_load_schemas_round_trips_every_field(schema_file: Path) -> None:
    first, second, _ = stubs.load_schemas(schema_file)

    assert first.namespace == "_C"
    assert first.name == "cutlass_scaled_mm"
    assert first.schema.startswith("cutlass_scaled_mm(Tensor! out,")
    assert first.kind == "compute"
    assert first.tier == "x"
    assert first.reason == "CUTLASS / CuTe in the include closure"

    assert second.namespace == "_C"
    assert second.name == "cutlass_scaled_mm_supports_fp8"
    assert second.kind == "probe"


def test_load_schemas_defaults_the_optional_fields(schema_file: Path) -> None:
    """A minimal entry must still register, so only the signature is required."""
    minimal = stubs.load_schemas(schema_file)[-1]
    assert minimal.namespace == "_moe_C"
    assert minimal.name == "shuffle_rows"
    assert minimal.kind == "compute"
    assert minimal.tier == "x"
    assert minimal.reason == ""


def test_load_schemas_tolerates_an_ungenerated_file(tmp_path: Path) -> None:
    """The file legitimately does not exist until the port tool has run."""
    assert stubs.load_schemas(tmp_path / "excluded_ops.toml") == ()


def test_generated_stub_schemas_parse_with_installed_torch() -> None:
    torch = pytest.importorskip("torch")
    schemas = stubs.load_schemas()
    assert schemas, "the generated excluded-op schema inventory is empty"
    for op in schemas:
        torch._C.parse_schema(op.schema)


@pytest.mark.parametrize(
    ("kind", "is_probe"),
    [("probe", True), ("compute", False), ("", False), ("capability", False)],
)
def test_is_probe_is_exactly_the_probe_kind(kind: str, is_probe: bool) -> None:
    op = stubs.ExcludedOp(
        namespace="_C", name="op", schema="op() -> ()", kind=kind, tier="x", reason=""
    )
    assert op.is_probe is is_probe


def test_compute_stub_names_the_op_the_reason_and_the_tier() -> None:
    """Reaches into ``_unsupported`` because the public path needs torch.

    ``install`` builds this callable through ``torch.library``, which the merge
    gate does not have, but the message it produces is the user-facing contract
    of the module and must not regress.
    """
    op = stubs.ExcludedOp(
        namespace="_C",
        name="cutlass_scaled_mm",
        schema="cutlass_scaled_mm() -> ()",
        kind="compute",
        tier="x",
        reason="CUTLASS / CuTe in the include closure",
    )
    with pytest.raises(NotImplementedError) as excinfo:
        stubs._unsupported(op)(1, 2, out=3)

    message = str(excinfo.value)
    assert "_C::cutlass_scaled_mm" in message
    assert "CUTLASS / CuTe in the include closure" in message
    assert "'x'" in message
    assert "docs/developer_guide/kernels.md" in message


def test_compute_stub_message_survives_a_missing_reason() -> None:
    op = stubs.ExcludedOp(
        namespace="_moe_C",
        name="shuffle_rows",
        schema="shuffle_rows() -> ()",
        kind="compute",
        tier="x",
        reason="",
    )
    with pytest.raises(NotImplementedError, match="unsupported"):
        stubs._unsupported(op)()


def test_capability_probe_stub_answers_false_instead_of_raising() -> None:
    """Failure mode 2: a probe that throws breaks vLLM's backend selection."""
    assert stubs._probe(1) is False


class _FakeLibrary:
    def __init__(self, torch, namespace: str, kind: str) -> None:
        assert kind == "FRAGMENT"
        self._torch = torch
        self._namespace = namespace

    def define(self, schema: str) -> None:
        name = schema.split("(", 1)[0]
        key = (self._namespace, name)
        if key in self._torch.defined:
            raise RuntimeError(f"duplicate schema: {self._namespace}::{name}")
        self._torch.defined.add(key)

        namespace = getattr(self._torch.ops, self._namespace)
        packet_name, separator, overload = name.partition(".")
        packet = getattr(namespace, packet_name, None)
        if packet is None:
            packet = types.SimpleNamespace()
            setattr(namespace, packet_name, packet)
        if separator:
            setattr(packet, overload, object())
        else:
            packet.default = object()

    def impl(self, name: str, function, dispatch: str) -> None:
        assert (self._namespace, name) in self._torch.defined
        assert dispatch == "CompositeExplicitAutograd"


def _fake_torch(*registered: str):
    torch = types.ModuleType("torch")
    torch.ops = types.SimpleNamespace(_C=types.SimpleNamespace())
    torch.defined = set()
    for name in registered:
        packet_name, separator, overload = name.partition(".")
        packet = getattr(torch.ops._C, packet_name, None)
        if packet is None:
            packet = types.SimpleNamespace()
            setattr(torch.ops._C, packet_name, packet)
        if separator:
            setattr(packet, overload, object())
        else:
            packet.default = object()
        torch.defined.add(("_C", name))
    torch.library = types.SimpleNamespace(
        Library=lambda namespace, kind: _FakeLibrary(torch, namespace, kind)
    )
    return torch


def _overload_schema_file(tmp_path: Path) -> Path:
    path = tmp_path / "excluded_ops.toml"
    path.write_text(
        textwrap.dedent(
            """\
            [[op]]
            namespace = "_C"
            name = "quant"
            schema = "quant(Tensor input) -> Tensor"

            [[op]]
            namespace = "_C"
            name = "quant.out"
            schema = "quant.out(Tensor input, Tensor(a!) out) -> Tensor(a!)"
            """
        ),
        encoding="utf-8",
    )
    return path


def test_install_is_idempotent_for_default_and_named_overloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(stubs, "_KEEPALIVE", [])
    schemas = _overload_schema_file(tmp_path)

    assert stubs.install(schemas) == ("_C::quant", "_C::quant.out")
    assert stubs.install(schemas) == ()


def test_install_preserves_an_exact_externally_registered_overload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _fake_torch("quant", "quant.out")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(stubs, "_KEEPALIVE", [])

    assert stubs.install(_overload_schema_file(tmp_path)) == ()


def test_base_packet_does_not_hide_a_missing_named_overload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _fake_torch("quant")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(stubs, "_KEEPALIVE", [])

    assert stubs.install(_overload_schema_file(tmp_path)) == ("_C::quant.out",)


def test_named_overload_packet_does_not_hide_a_missing_default_overload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _fake_torch("quant.out")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(stubs, "_KEEPALIVE", [])

    assert stubs.install(_overload_schema_file(tmp_path)) == ("_C::quant",)
