import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import torch
from vllm.platforms import current_platform

SOURCE_SHA = "21d2c37e55e280a1d32d1cda78f16637f07a6da7"
EXPECTED_SOURCE_HASH = "f56a3cfa8ac3a1c0eb27454892c9b23f2d06e4559360eabafa819f0e336103b2"


def main():
    print("PROBE_STAGE load_candidate", flush=True)
    spec = importlib.util.spec_from_file_location("pr472_deep_select", "/probe/candidate/deep_select.py")
    deep_select = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deep_select)
    source_hash = hashlib.sha256(Path(deep_select.__file__).read_bytes()).hexdigest()
    assert source_hash == EXPECTED_SOURCE_HASH, ("unexpected plugin source", source_hash)
    print("PROBE_STAGE check_device", flush=True)
    assert current_platform.is_ppu(), "requires the vLLM PPU platform"
    assert current_platform.has_device_capability(89), "requires capability 8.9"
    device_name = torch.cuda.get_device_name(0)
    assert "m890" in device_name.casefold(), ("requires M890", device_name)

    print("PROBE_STAGE load_wheel", flush=True)
    wheel_spec = importlib.util.find_spec("deep_select")
    if wheel_spec and wheel_spec.origin:
        assert not Path(wheel_spec.origin).is_relative_to("/probe"), "probe must import the installed wheel"
    interface = deep_select._wheel()
    if interface is None:
        print("PROBE_STAGE wheel_interface_missing", flush=True)
        raise RuntimeError("DeepSelect wheel interface missing")
    print("PROBE_STAGE prepare_cpu_scores", flush=True)
    topk = interface.topk
    width = 4096
    k = 512
    values = 1.0 + torch.arange(width, dtype=torch.float32) * (2**-22)
    cpu_scores = torch.stack((values.roll(997), values.flip(0).roll(123)))
    assert cpu_scores.bfloat16().unique().numel() == 1
    print("PROBE_STAGE copy_fp32_to_device", flush=True)
    scores = cpu_scores.cuda()
    lengths = (4096, 3072)
    ends = torch.tensor(lengths, dtype=torch.int32, device=scores.device)

    print("PROBE_STAGE compute_cpu_reference", flush=True)
    expected = torch.stack(
        [cpu_scores[row, :end].topk(k).indices.sort().values for row, end in enumerate(lengths)]
    )
    calls = []

    def record_native_call(logits, *args, **kwargs):
        calls.append((str(logits.dtype), str(logits.device), list(logits.shape)))
        assert logits.dtype == torch.float32 and logits.device == scores.device
        return topk(logits, *args, **kwargs)

    print("PROBE_STAGE check_fp32_eligibility", flush=True)
    assert deep_select.can_use_deep_select(scores, k), "plugin rejected the FP32 inputs"
    print("PROBE_STAGE call_wheel", flush=True)
    with patch.object(interface, "topk", record_native_call):
        actual = deep_select.deep_select_topk(scores, k, end=ends)
        torch.cuda.synchronize()
    assert calls == [("torch.float32", str(scores.device), [2, width])], calls
    print("PROBE_STAGE compare_indices", flush=True)
    assert torch.equal(actual.cpu().to(torch.int64).sort(dim=1).values, expected)
    print("PPU_FP32_RECEIPT " + json.dumps({
        "image_digest": os.environ["PROBE_IMAGE_DIGEST"],
        "image_id": os.environ["PROBE_IMAGE_ID"],
        "source_sha": SOURCE_SHA,
        "wrapper_sha256": source_hash,
        "wheel_version": importlib.metadata.version("deep-select"),
        "torch_version": torch.__version__,
        "device": device_name,
        "native_calls": calls,
        "topk": k,
        "row_lengths": lengths,
        "index_sets_match": True,
    }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PROBE_EXCEPTION {type(error).__name__}", flush=True)
        if isinstance(error, ImportError) and re.fullmatch(r"[a-zA-Z0-9_.]+", error.name or ""):
            print(f"PROBE_MODULE {error.name}", flush=True)
        for marker in ("undefined symbol", "cannot open shared object file", "cannot import name", "supports bfloat16 input only"):
            if marker in str(error):
                print(f"PROBE_DIAGNOSTIC {marker}", flush=True)
        raise
