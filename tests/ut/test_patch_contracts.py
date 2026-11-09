# SPDX-License-Identifier: Apache-2.0
"""Independent contracts replacing the derived lock/Markdown checks."""

from __future__ import annotations

import ast
import inspect

import pytest

from tests.support.patch_inventory import (
    dispatcher_hooks,
    registry_alias_hooks,
    scan_patch_file,
    scan_patches,
)
from tests.support.upstream import (
    FUNC_NODES,
    PATCH_ROOT,
    ROOT,
    callable_signature,
    compatibility_problems,
    module_file,
    resolve_symbol,
)


def _feature_calls(root):
    names = {
        "_get_config_dtype_str",
        "moe_kernel_quantize_input",
        "Qwen3MoeSparseMoeBlock",
        "Qwen3MoeDecoderLayer",
        "decoder_layer_type",
    }
    calls = []
    for path in sorted((root / "vllm").rglob("*.py")):
        for call in ast.walk(ast.parse(path.read_text())):
            if isinstance(call, ast.Call):
                name = (
                    call.func.id
                    if isinstance(call.func, ast.Name)
                    else getattr(call.func, "attr", None)
                )
                if name in names:
                    calls.append((path, name, call))
    return calls


def actual_call_problems(root, hook, original, calls):
    """Narrow contracts for existing adapters whose extra PPU args alter order.

    Quantization helpers and Qwen layer factories are called with keywords by
    pinned upstream. Verify those real calls, rather than promising every call
    admitted by the old signature. Starred/unresolved forms fail closed.
    """
    leaf = hook.target.rsplit(".", 1)[-1]
    constructor = leaf == "__init__"
    if constructor:
        leaf = hook.target.split(".")[-2]
    upstream = callable_signature(original)
    replacement = callable_signature(hook.node)
    problems, checked = [], 0
    for path, name, call in calls:
        if constructor and path != root / "vllm/model_executor/models/qwen3_moe.py":
            continue
        if name != leaf and not (
            constructor
            and leaf == "Qwen3MoeDecoderLayer"
            and name == "decoder_layer_type"
        ):
            continue
        checked += 1
        location = f"{path.relative_to(root)}:{call.lineno}"
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
            kw.arg is None for kw in call.keywords
        ):
            problems.append(
                f"{location}: unresolved expanded call needs a feature contract"
            )
            continue
        args = ([object()] if constructor else []) + [object() for _ in call.args]
        kwargs = {kw.arg: object() for kw in call.keywords}
        try:
            before = upstream.bind(*args, **kwargs)
            after = replacement.bind(*args, **kwargs)
        except TypeError as exc:
            problems.append(f"{location}: {exc}")
            continue
        for name, value in before.arguments.items():
            if (
                name in after.arguments
                and upstream.parameters[name].kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                and after.arguments[name] is not value
            ):
                problems.append(f"{location}: argument {name} changes meaning")
    return problems if checked else [f"no source calls found for {hook.target}"]


def target_problems(root, hooks):
    failures = []
    calls = None
    registry = ROOT / "vllm_sail/registry/moe_backends/__init__.py"
    alias_bodies = {
        hook.target: hook.node
        for hook in hooks
        if hook.path == registry and isinstance(hook.node, FUNC_NODES)
    }
    for hook in hooks:
        if hook.path == registry and hook.node is None:
            hook = hook._replace(node=alias_bodies.get(hook.target))
        resolved = resolve_symbol(root, hook.target)
        if not resolved.module_found:
            failures.append(f"{hook.target}: missing upstream module")
        elif hook.additive:
            if len(resolved.symbol_path) > 1:
                owner = hook.target.rsplit(".", 1)[0]
                if not resolve_symbol(root, owner).present:
                    failures.append(f"{hook.target}: missing additive owner {owner}")
        elif not resolved.present and not hook.conditional_alias:
            failures.append(f"{hook.target}: missing required target")
        elif isinstance(resolved.node, FUNC_NODES):
            if hook.node is None:
                # The Autotuner rebuild retains the original fn. The real
                # installer is exercised in test_fla_autotune_cache.py.
                if hook.path != PATCH_ROOT / "performance/fla/autotune_cache.py":
                    failures.append(
                        f"{hook.target}: unresolved callable replacement at {hook.path}:{hook.line}"
                    )
                continue
            problems = compatibility_problems(hook.node, resolved.node)
            # Existing feature-specific adapters deliberately add PPU inputs.
            # This small set expresses external call dependencies, not an
            # inventory of all patches or a signature snapshot.
            if hook.target.endswith(
                (
                    "._get_config_dtype_str",
                    ".moe_kernel_quantize_input",
                    ".Qwen3MoeSparseMoeBlock.__init__",
                    ".Qwen3MoeDecoderLayer.__init__",
                )
            ):
                if calls is None:
                    calls = _feature_calls(root)
                problems = actual_call_problems(root, hook, resolved.node, calls)
            failures.extend(f"{hook.target}: {problem}" for problem in problems)
    return failures


def _function(text):
    return ast.parse(text).body[0]


@pytest.mark.parametrize(
    "upstream,replacement,compatible",
    [
        ("def f(a: Unknown = dangerous()): ...", "def f(a=1): ...", True),
        ("def f(a): ... # comment", "def f(a, extra=None): ...", True),
        ("def f(a, *, new=1): ...", "def f(a, **kwargs): ...", True),
        ("def f(a, *, used=1): ...", "def f(a): ...", False),
        ("def f(a, b): ...", "def f(b, a, **kwargs): ...", False),
        ("def f(a): ...", "def f(a, required): ...", False),
        ("def f(a): ...", "def f(a, /, **kwargs): ...", False),
        ("def f(a, /): ...", "def f(*, a, **kwargs): ...", False),
        ("def f(a, *args): ...", "def f(a, **kwargs): ...", False),
        ("def f(a, **kwargs): ...", "def f(a, *args): ...", False),
    ],
)
def test_binding_contract(upstream, replacement, compatible):
    assert (
        not compatibility_problems(_function(replacement), _function(upstream))
    ) is compatible


@pytest.mark.parametrize("fault", [None, "missing-target", "missing-keyword"])
def test_discovery_to_source_gate_rejects_real_contract_faults(tmp_path, fault):
    upstream = tmp_path / "vllm/example.py"
    upstream.parent.mkdir()
    upstream.write_text("def run(a, *, used=None): ...\nclass Owner: pass\n")
    plugin = tmp_path / "plugin.py"
    name = "missing" if fault == "missing-target" else "run"
    signature = "a" if fault == "missing-keyword" else "a, *, used=None"
    plugin.write_text(
        f'@patch("vllm.example", "{name}", reason="test")\ndef replacement({signature}): ...\n'
    )
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved
    problems = target_problems(tmp_path, hooks)
    assert bool(problems) is (fault is not None), problems
    if fault:
        assert name in " ".join(problems)


def test_discovery_covers_supported_forms_and_reports_unknowns(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
from vllm import example as provider
from vllm_sail.patch.utils import patch as install
@patch("vllm.example")
def inferred(a): ...
@patch(provider.__name__, "Owner.method")
def method(self, a): ...
@install("vllm.example", "aliased")
def alias(a): ...
if condition:
    @patch("vllm.example", "Added", allow_missing=True)
    class Added: pass
for name in ("one", "two"):
    patch_value("vllm.example", name, object(), allow_missing=True)
patch(dynamic_module(), "unsupported")(method)
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert {hook.target for hook in hooks} == {
        "vllm.example.inferred",
        "vllm.example.Owner.method",
        "vllm.example.aliased",
        "vllm.example.Added",
        "vllm.example.one",
        "vllm.example.two",
    }
    assert len(unresolved) == 1 and "dynamic_module()" in unresolved[0].expression


def test_discovery_reports_dictionary_unpack_instead_of_partial_success(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
TARGETS = {"vllm.example": ("known",), **dynamic_targets()}
for module, names in TARGETS.items():
    for name in names:
        patch(module, name)(replacement)
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert not hooks
    assert len(unresolved) == 1


@pytest.mark.parametrize("present", [True, False])
def test_discovery_to_gate_resolves_reexport_providers(tmp_path, present):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "provider.py").write_text("def run(a): ..." if present else "# removed")
    (package / "consumer.py").write_text("from vllm.provider import run")
    plugin = tmp_path / "plugin.py"
    plugin.write_text('@patch("vllm.consumer", "run")\ndef replacement(a): ...')
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved
    assert bool(target_problems(tmp_path, hooks)) is (not present)


def test_conditional_import_with_explicit_fallback_remains_a_real_target(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "consumer.py").write_text("""
if available:
    from vllm.optional import run
else:
    run = unavailable
""")
    assert resolve_symbol(tmp_path, "vllm.consumer.run").present


def test_unconditional_dangling_import_cannot_be_repaired_by_later_assignment(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "consumer.py").write_text(
        "from vllm.missing import run\nrun = wrapper(run)\n"
    )
    assert not resolve_symbol(tmp_path, "vllm.consumer.run").present


def test_gate_checks_each_conditional_replacement_for_the_same_target(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "example.py").write_text("def run(a, *, used=None): ...")
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
if is_ppu:
    @patch("vllm.example", "run")
    def bad(a): ...
else:
    @patch("vllm.example", "run")
    def good(a, *, used=None): ...
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved and len(hooks) == 2
    assert target_problems(tmp_path, hooks)


@pytest.mark.parametrize("variation", ["replacement", "additive"])
def test_gate_preserves_distinct_variants_at_one_literal_loop_site(tmp_path, variation):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "example.py").write_text("def run(a, *, used=None): ...")
    plugin = tmp_path / "plugin.py"
    declarations = "def good(a, *, used=None): ...\ndef bad(a): ...\n"
    if variation == "replacement":
        declarations += """
for flag, body in ((True, good), (False, bad)):
    if is_ppu == flag:
        patch("vllm.example", "run")(body)
"""
    else:
        declarations += """
for additive in (True, False):
    patch("vllm.example", "missing", allow_missing=additive)(good)
"""
    plugin.write_text(declarations)
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved and len(hooks) == 2
    assert target_problems(tmp_path, hooks)


def test_every_unresolved_installation_has_a_feature_contract():
    _, unresolved = scan_patches()
    # Only these two dispatcher mechanisms and dataclass-generated methods need
    # feature-specific source handling. A new opaque installation fails here.
    expected = {
        ("patch/enhancement/runtime.py", "_install", "module.__name__", "target"),
        ("patch/enhancement/runtime.py", "_install", "name", "attribute"),
        (
            "patch/enhancement/triton_moe.py",
            "_install",
            "module.__name__",
            "target_name",
        ),
        ("patch/enhancement/triton_moe.py", "_install", "consumer_name", "name"),
        (
            "registry/moe_backends/__init__.py",
            "_rebind_loaded_oracle_aliases",
            "consumer_name",
            "alias_name",
        ),
        *(
            (
                "patch/enhancement/attention/kv_cache_interface.py",
                "_extend_existing_subclass",
                "_MODULE",
                f"f'{{cls.__name__}}.{method}'",
            )
            for method in ("__init__", "__eq__", "__hash__")
        ),
    }
    for site in unresolved:
        call = ast.parse(site.expression, mode="eval").body
        key = (
            site.path.relative_to(ROOT / "vllm_sail").as_posix(),
            site.scope,
            *(ast.unparse(arg) for arg in call.args[:2]),
        )
        assert key in expected, (
            f"uncovered installation at {site.path}:{site.line}: {site.expression}"
        )


@pytest.mark.upstream_source
def test_declared_targets_and_calls_fit_pinned_source(upstream_source_root):
    hooks, _ = scan_patches()
    for name in ("runtime.py", "triton_moe.py"):
        hooks.extend(dispatcher_hooks(PATCH_ROOT / "enhancement" / name))
    hooks.extend(registry_alias_hooks(hooks))
    problems = target_problems(upstream_source_root, hooks)
    assert not problems, "\n".join(problems)


@pytest.mark.upstream_source
def test_patch_upstream_import_modules_resolve(upstream_source_root):
    imported = set()
    for path in (ROOT / "vllm_sail").rglob("*.py"):
        if "template" in path.parts or path.name.startswith("."):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and not node.level
                and node.module
                and node.module.startswith("vllm.")
            ):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(
                    alias.name for alias in node.names if alias.name.startswith("vllm.")
                )
    assert imported
    missing = [
        name
        for name in sorted(imported)
        if module_file(upstream_source_root, name)[0] is None
    ]
    assert not missing, f"missing upstream import modules: {missing}"
