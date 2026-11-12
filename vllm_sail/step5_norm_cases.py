# SPDX-License-Identifier: Apache-2.0
"""Step5 Norm 元数据协议；仅依赖标准库，不恢复真实激活值。"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DTYPES = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}
_LABEL_PREFIX = "[FW_NATIVE] op:_Step5Norm"
_FIELDS = (
    "v", "op", "M", "H", "shape", "stride", "X", "W", "wshape", "wstride",
    "R", "rshape", "rstride", "Y", "eps", "Z", "out", "oshape", "ostride", "capture",
)


def _vector(values) -> str:
    return "x".join(str(int(v)) for v in values)


def _tensor(tensor) -> tuple[str, str, str]:
    if tensor is None:
        return "-", "-", "-"
    return DTYPES.get(str(tensor.dtype), str(tensor.dtype)), _vector(tensor.shape), _vector(tensor.stride())


def make_label(x, weight, eps, zero_centered=False, residual=None, out=None,
               out_dtype=None, capture=False) -> str:
    """只查询 shape/stride/dtype；不读内容、不同步设备。Y 是请求的输出 dtype。"""
    xd, shape, stride = _tensor(x)
    wd, ws, wt = _tensor(weight)
    rd, rs, rt = _tensor(residual)
    _, os, ot = _tensor(out)
    yd = DTYPES.get(str(out_dtype or x.dtype), str(out_dtype or x.dtype))
    values = ("1", "add_rms" if residual is not None else "rms",
              str(math.prod(x.shape[:-1])), str(x.shape[-1]), shape, stride,
              xd, wd, ws, wt, rd, rs, rt, yd, repr(float(eps)),
              str(int(zero_centered)), str(int(out is not None)), os, ot,
              str(int(capture)))
    kind = "D" if capture else "P"
    return f"{_LABEL_PREFIX},type:{kind}," + ",".join(
        f"{key}={value}" for key, value in zip(_FIELDS, values)
    )


def _parse_vector(raw: str) -> list[int]:
    values = [int(v) for v in raw.split("x")]
    if not values or any(v < 0 for v in values):
        raise ValueError("shape/stride 必须是非负整数向量")
    return values


def is_norm_label(label) -> bool:
    """只识别 Norm 候选前缀；字段完整性仍由 parse_label 严格检查。"""
    return isinstance(label, str) and label.startswith(
        (_LABEL_PREFIX + ",", "D_Step5Norm,", "P_Step5Norm,")
    )


def parse_label(label: str) -> dict[str, Any]:
    """严格版本化解析；损坏或未知版本不能悄悄成为 benchmark 输入。"""
    prefix, *parts = label.split(",")
    if prefix == _LABEL_PREFIX:
        if not parts or parts[0] not in ("type:D", "type:P"):
            raise ValueError("Norm type 必须为 D/P，且与元数据用逗号分隔")
        expected_capture = int(parts.pop(0) == "type:D")
    elif prefix in ("D_Step5Norm", "P_Step5Norm"):
        expected_capture = int(prefix.startswith("D"))
    else:
        raise ValueError("不是 Step5 Norm 标签")
    pairs = [part.split("=", 1) for part in parts]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("标签字段格式错误")
    raw = dict(pairs)
    if len(raw) != len(pairs) or set(raw) != set(_FIELDS) or raw["v"] != "1":
        raise ValueError("重复/缺失字段或不支持的 schema 版本")
    if raw["op"] not in ("rms", "add_rms"):
        raise ValueError("不支持的 Norm op")
    result: dict[str, Any] = dict(raw)
    for key in ("v", "M", "H", "Z", "out", "capture"):
        result[key] = int(raw[key])
    if any(result[key] not in (0, 1) for key in ("Z", "out", "capture")):
        raise ValueError("布尔字段必须是 0/1")
    if result["capture"] != expected_capture:
        raise ValueError("capture 与标签前缀不一致")
    result["eps"] = float(raw["eps"])
    if not math.isfinite(result["eps"]) or result["eps"] <= 0:
        raise ValueError("epsilon 必须有限且大于零")
    for key in ("shape", "stride", "wshape", "wstride", "rshape", "rstride", "oshape", "ostride"):
        result[key] = None if raw[key] == "-" else _parse_vector(raw[key])
    for dtype, shape, stride, required in (
        ("X", "shape", "stride", True), ("W", "wshape", "wstride", True),
        ("R", "rshape", "rstride", result["op"] == "add_rms"),
        (None, "oshape", "ostride", bool(result["out"])),
    ):
        if required:
            if result[shape] is None or result[stride] is None or len(result[shape]) != len(result[stride]):
                raise ValueError("张量 shape/stride 不一致")
            if dtype and raw[dtype] not in DTYPES.values():
                raise ValueError("不支持的 dtype")
        elif result[shape] is not None or result[stride] is not None or (dtype and raw[dtype] != "-"):
            raise ValueError("缺省张量携带了元数据")
    if raw["Y"] not in DTYPES.values():
        raise ValueError("不支持的 output dtype")
    if result["H"] <= 0 or result["H"] != result["shape"][-1] or result["M"] != math.prod(result["shape"][:-1]):
        raise ValueError("M/H 与完整 shape 不一致")
    if result["wshape"] != [result["H"]]:
        raise ValueError("weight 必须是 hidden-size 向量")
    for key in ("rshape", "oshape"):
        if result[key] is not None and result[key] != result["shape"]:
            raise ValueError("residual/out shape 必须等于 input")
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return result


def labels_from_file(path: str | Path):
    """接受 torch/NVTX 导出的 JSON 或逐行标签，支持 gzip。"""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        yield from (line.strip() for line in text.splitlines() if is_norm_label(line.strip()))
        return
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("JSON 应包含 traceEvents 列表")
    for event in events:
        if not isinstance(event, dict):
            continue
        # 同一事件优先使用 name，避免 name 与 args 同时携带标签而双计数。
        args = event.get("args", {})
        candidates = [event.get("name", "")]
        if isinstance(args, dict):
            candidates += [args.get(key, "") for key in ("Text", "text", "message")]
        for name in candidates:
            if is_norm_label(name):
                yield name
                break


def extract_cases(paths) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        source = str(Path(path).resolve())
        for label in labels_from_file(path):
            case = parse_label(label)
            item = unique.setdefault(case["id"], {**case, "sources": [], "observations": 0})
            if source not in item["sources"]:
                item["sources"].append(source)
            item["observations"] += 1
    return {"schema": 1, "input_values": "shape-matched synthetic (not real activations)",
            "frequency_note": "observations 是导出范围数；capture/replay 不能据此推算真实调用频次",
            "cases": sorted(unique.values(), key=lambda case: case["id"])}


def load_cases(path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("cases"), list):
        raise ValueError("不支持的 cases.json schema")
    result = []
    for case in data["cases"]:
        values = []
        for key in _FIELDS:
            value = case[key]
            if key.endswith("shape") or key.endswith("stride"):
                value = "-" if value is None else _vector(value)
            values.append(f"{key}={value}")
        kind = "D" if case["capture"] else "P"
        checked = parse_label(f"{_LABEL_PREFIX},type:{kind}," + ",".join(values))
        if checked["id"] != case["id"]:
            raise ValueError("case id 与元数据不一致")
        result.append({**case, **checked})
    return result
