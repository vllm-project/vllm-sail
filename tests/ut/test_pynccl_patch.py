# SPDX-License-Identifier: Apache-2.0
"""Deterministic optional-symbol coverage without torch, vLLM, or a device."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from tests.support.source import definition, function

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL = ("ncclCommSuspend", "ncclCommResume")


@pytest.fixture
def runtime(monkeypatch, patch_utils_module):
    modules = {}

    def add(name, **attrs):
        if name not in modules:
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
            monkeypatch.setitem(sys.modules, name, module)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(add(parent), leaf, module)
        modules[name].__dict__.update(attrs)
        return modules[name]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    add("vllm_sail.patch.utils", patch=patch_utils_module.patch)
    helpers = load("_pynccl_bodies", ROOT / "vllm_sail/patch/bodies.py")
    add("vllm_sail.patch.bodies", bind_body=helpers.bind_body)
    platform = types.SimpleNamespace(backend="ppu")
    platform.is_ppu = lambda: platform.backend == "ppu"
    platform.is_rocm = lambda: platform.backend == "rocm"
    add("vllm.platforms", current_platform=platform)
    libraries, opened, delegated, warnings = {}, [], [], []

    def cdll(path):
        opened.append(path)
        library = libraries[path]
        if isinstance(library, Exception):
            raise library
        return library

    logger = types.SimpleNamespace(
        error=lambda *args: None,
        warning_once=lambda *args: warnings.append(args),
    )

    class Library:
        exported_functions = [
            types.SimpleNamespace(name=name, restype=None, argtypes=[])
            for name in (
                "ncclAllReduce",
                *OPTIONAL,
                "ncclGetLastError",
                "ncclCommQueryProperties",
            )
        ]
        path_to_library_cache = {}
        path_to_dict_mapping = {}

        def __init__(self, so_file=None):
            raise AssertionError("unpatched loader")

        def has_symbol(self, name):
            return name in self._funcs

    class Communicator:
        def suspend(self):
            delegated.append(self)
            self._suspended = True

    wrapper = add(
        "vllm.distributed.device_communicators.pynccl_wrapper",
        NCCLLibrary=Library,
        ctypes=types.SimpleNamespace(CDLL=cdll),
        current_platform=platform,
        envs=types.SimpleNamespace(VLLM_USE_NCCL_SYMM_MEM=False),
        logger=logger,
        platform=types.SimpleNamespace(platform=lambda: "test"),
        find_nccl_library=lambda: "default.so",
    )
    add(
        "vllm.distributed.device_communicators.pynccl",
        PyNcclCommunicator=Communicator,
        NCCLLibrary=Library,
        logger=logger,
    )
    original_init, original_suspend = Library.__init__, Communicator.suspend
    load("_pynccl_patch", ROOT / "vllm_sail/patch/enhancement/pynccl.py")
    assert Library.__init__ is not original_init
    assert Communicator.suspend is not original_suspend
    assert len(patch_utils_module.PATCH_REGISTRY) == 2

    def library(path="default.so", symbols=OPTIONAL):
        libraries[path] = types.SimpleNamespace(
            **{name: types.SimpleNamespace() for name in ("ncclAllReduce", *symbols)}
        )
        return Library(path)

    return types.SimpleNamespace(
        Library=Library,
        Communicator=Communicator,
        platform=platform,
        libraries=libraries,
        opened=opened,
        delegated=delegated,
        warnings=warnings,
        library=library,
        wrapper=wrapper,
    )


@pytest.mark.parametrize("backend", ["ppu", "cuda", "rocm"])
@pytest.mark.parametrize("symbols", [(), OPTIONAL[:1], OPTIONAL[1:], OPTIONAL])
def test_optional_symbol_policy(runtime, backend, symbols):
    runtime.platform.backend = backend
    if backend == "cuda" and len(symbols) != 2:
        with pytest.raises(AttributeError, match="ncclComm"):
            runtime.library(symbols=symbols)
    else:
        library = runtime.library(symbols=symbols)
        assert library.has_symbol("ncclAllReduce")
        assert {name for name in OPTIONAL if library.has_symbol(name)} == set(symbols)


@pytest.mark.parametrize("symbols", [(), OPTIONAL[:1], OPTIONAL[1:], OPTIONAL])
def test_ppu_suspends_only_with_both_apis(runtime, symbols):
    communicator = runtime.Communicator()
    communicator.nccl = runtime.library(symbols=symbols)
    communicator.disabled = communicator._suspended = False
    communicator.suspend()
    assert bool(runtime.delegated) == (len(symbols) == 2)
    assert communicator._suspended == (len(symbols) == 2)
    assert bool(runtime.warnings) == (len(symbols) != 2)


def test_missing_required_symbol_and_library_load_errors_propagate(runtime):
    runtime.libraries["required.so"] = types.SimpleNamespace()
    with pytest.raises(AttributeError, match="ncclAllReduce"):
        runtime.Library("required.so")
    runtime.libraries["broken.so"] = OSError("invalid ELF header")
    with pytest.raises(OSError, match="invalid ELF header"):
        runtime.Library("broken.so")


def test_repeated_and_distinct_libraries_keep_separate_symbol_caches(runtime):
    alias = runtime.Library
    absent = runtime.library("absent.so", symbols=())
    cached = alias("absent.so")
    present = runtime.library("present.so")
    assert absent._funcs is cached._funcs
    assert present._funcs is not absent._funcs
    assert not absent.has_symbol("ncclCommSuspend")
    assert present.has_symbol("ncclCommSuspend")
    assert runtime.opened == ["absent.so", "present.so"]
    assert [f.name for f in alias.exported_functions] == [
        "ncclAllReduce",
        *OPTIONAL,
        "ncclGetLastError",
        "ncclCommQueryProperties",
    ]


@pytest.mark.parametrize("backend", ["ppu", "cuda", "rocm"])
@pytest.mark.parametrize("present", [False, True])
def test_diagnostic_symbol_is_optional_on_every_backend(runtime, backend, present):
    runtime.platform.backend = backend
    symbols = (*OPTIONAL, "ncclGetLastError") if present else OPTIONAL
    library = runtime.library(symbols=symbols)
    assert library.has_symbol("ncclAllReduce")
    assert all(library.has_symbol(name) for name in OPTIONAL)
    assert library.has_symbol("ncclGetLastError") is present


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("backend", ["cuda", "rocm"])
@pytest.mark.parametrize("present", [False, True])
def test_actual_provider_allows_missing_diagnostic_symbol(
    runtime, upstream_source_root, copied, backend, present
):
    path = (
        upstream_source_root / "vllm/distributed/device_communicators/pynccl_wrapper.py"
    )
    cls = definition(path, "NCCLLibrary")
    (exported,) = [
        node
        for node in cls.body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "exported_functions"
            for target in node.targets
        )
    ]
    names = {
        node.args[0].value
        for node in ast.walk(exported)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Function"
    }
    assert {"ncclAllReduce", "ncclGetLastError"} <= names
    if not copied:
        runtime.Library.__init__ = function(
            path, "NCCLLibrary.__init__", vars(runtime.wrapper)
        )
    runtime.platform.backend = backend
    symbols = (*OPTIONAL, "ncclGetLastError") if present else OPTIONAL
    library = runtime.library(symbols=symbols)
    assert library.has_symbol("ncclAllReduce")
    assert library.has_symbol("ncclGetLastError") is present
    assert all(library.has_symbol(name) for name in OPTIONAL)


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "result,message,expected",
    [
        (7, None, None),
        (7, b"device detail", "device detail"),
        (7, b"\xff", "\ufffd"),
        (0, b"unused", None),
    ],
)
def test_actual_error_reporting_keeps_optional_details(
    upstream_source_root, result, message, expected
):
    path = (
        upstream_source_root / "vllm/distributed/device_communicators/pynccl_wrapper.py"
    )
    calls = []

    def base(code):
        calls.append(("base", code))
        return b"base failure"

    def last(comm):
        calls.append(("last", comm))
        return message

    funcs = {"ncclGetErrorString": base}
    if message is not None:
        funcs["ncclGetLastError"] = last
    library = types.SimpleNamespace(_funcs=funcs)
    base_getter = function(path, "NCCLLibrary.ncclGetErrorString", {})
    last_getter = function(path, "NCCLLibrary.ncclGetLastError", {})
    library.ncclGetErrorString = lambda code: base_getter(library, code)
    library.ncclGetLastError = lambda: last_getter(library)
    check = function(path, "NCCLLibrary.NCCL_CHECK", {})
    if result == 0:
        check(library, result)
        assert calls == []
    else:
        with pytest.raises(RuntimeError) as error:
            check(library, result)
        assert str(error.value) == "NCCL error: base failure" + (
            "\nLast error:\n" + expected if expected is not None else ""
        )
        assert calls == [("base", 7)] + (
            [("last", None)] if message is not None else []
        )
