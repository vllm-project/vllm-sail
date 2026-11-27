# SPDX-License-Identifier: Apache-2.0
"""Check mixed-history radix selection and M890 FP8 verification scores."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("threads", [256, 1024])
@pytest.mark.parametrize("width", [16384, 32768, 131072])
def test_radix_mixed_history(threads, width):
    import torch

    from vllm_sail.models.step5.ops.selector_v2 import allocate, raw_select

    torch.manual_seed(29)
    k = 512
    current = [0, 31, 511, 512, 513, 8192, 16384, 16385, 25000]
    current = [min(width - 1, n) for n in current]
    scores = torch.rand(len(current), width, device="cuda")
    # Make the scores unique to check exact membership across a chunk boundary.
    scores = scores.argsort(dim=-1).float()
    positions = torch.tensor(current, dtype=torch.int64, device="cuda")
    workspace, output, starts, table = allocate(scores, k)
    workspace.fill_(222)
    raw_select(
        scores.unsqueeze(1),
        positions,
        table,
        workspace,
        output,
        starts,
        topk=k,
        threads=threads,
    )
    for row, n in enumerate(current):
        count = min(k, n)
        ids = scores[row, :n].topk(count).indices.tolist()
        actual = output[row, 0, :k].tolist()
        assert sorted(actual) == sorted(ids + [-1] * (k - count))
        assert output[row, 0, k].item() == n


@pytest.mark.parametrize("rows", [32, 64, 96, 128])
@pytest.mark.parametrize("capacity", [8192, 16384, 131072])
def test_direct_fp8_decode_schedule(monkeypatch, rows, capacity):
    import torch
    from vllm.models.step5.nvidia import flashinfer_sparse
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu() or not current_platform.is_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia.ops import sparse_indexer_scores

    launch = flashinfer_sparse.decode_summary_scores
    assert launch is sparse_indexer_scores.decode_summary_scores
    assert hasattr(launch, "__wrapped__")
    torch.manual_seed(29)
    page_size = 8
    pages = capacity // page_size
    q = torch.randn(rows, 1, 8, 256, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.rand(rows, 1, 8, device="cuda")
    mean = torch.randn(pages, page_size, 256, device="cuda").to(torch.float8_e4m3fn)
    table = torch.arange(pages, dtype=torch.int32, device="cuda")
    table = table.expand(rows // 4, -1).contiguous()
    slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    slots[0] = -1
    lengths = torch.linspace(
        437, capacity * 8, rows // 4, dtype=torch.int32, device="cuda"
    )
    args = (q, weights, mean.view(torch.uint8), table, slots, lengths, capacity)
    monkeypatch.setenv("VLLM_SAIL_STEP5_DSA_SCORES", "v0")
    expected, expected_meta = launch(*args, num_sms=39, query_len=4)
    kernel = sparse_indexer_scores._direct_scores
    launches = []

    class RecordLaunch:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launches.append((grid, kwargs["num_warps"]))
                return kernel[grid](*args, **kwargs)

            return run

    monkeypatch.setattr(sparse_indexer_scores, "_direct_scores", RecordLaunch())
    monkeypatch.setenv("VLLM_SAIL_STEP5_DSA_SCORES", "v1")
    actual, actual_meta = launch(*args, num_sms=39, query_len=4)
    assert [warps for _, warps in launches] == [
        4 if capacity > 16384 and rows == 96 else 1
    ]
    if capacity == 131072 and rows == 64:
        assert launches == [((19, 64), 1)]
    for name in ("active", "positions", "current", "lengths", "topk_lengths"):
        torch.testing.assert_close(
            getattr(actual_meta, name), getattr(expected_meta, name)
        )
    live = (
        torch.arange(capacity, device="cuda")[None, :] < expected_meta.lengths[:, None]
    )
    torch.testing.assert_close(actual[live], expected[live], atol=0, rtol=0)


@pytest.mark.parametrize("rows", [16, 32, 64])
@pytest.mark.parametrize("gate_enabled", [False, True])
def test_native_paired_decode(gate_enabled, rows):
    import torch
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu() or not current_platform.is_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    vllm_sail.register_out_of_tree()
    from vllm_sail.models.step5.ops.dsa_native import compiled_op

    decode = compiled_op("decode")
    assert decode is not None
    torch.manual_seed(421)
    device, dtype = "cuda", torch.bfloat16
    q = torch.randn((rows, 16, 192), device=device, dtype=dtype)
    cache = torch.randn((128, 1, 64, 384), device=device, dtype=dtype)
    cache.mul_(0.25)
    table = torch.stack(
        [
            torch.randperm(128, device=device, dtype=torch.int32)
            for _ in range(rows // 4)
        ]
    )
    positions = torch.arange(rows, device=device) % 4 + 8188
    active = torch.ones(rows, device=device, dtype=torch.bool)
    active[2::8] = False
    regions = torch.stack(
        [torch.randperm(1023, device=device)[:513] for _ in range(rows)]
    )
    regions[:, -1] = 1023
    regions[1, :4] = -1
    regions[3, :4] = torch.iinfo(torch.int64).max
    gate = torch.randn((rows, 16), device=device, dtype=dtype) if gate_enabled else None
    expected, actual = (torch.full_like(q, 2) for _ in range(2))
    args = (q, cache, regions, table, positions, active)
    decode(*args, expected, 192**-0.5, 4, False, gate, False)
    decode(*args, actual, 192**-0.5, 4, False, gate, True)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=0.03)
    assert torch.all(actual[~active] == (0 if gate_enabled else 2))


@pytest.mark.parametrize(("rows", "width"), [(3072, 4096), (4096, 8192)])
def test_prefill_selector_preserves_cutoff_ties(monkeypatch, rows, width):
    import torch
    from vllm.models.step5.nvidia.ops.sparse_indexer import _stable_topk_regions

    from vllm_sail.models.step5.ops.selector_v2 import (
        allocate,
        raw_select,
        select_regions,
    )

    monkeypatch.setenv("VLLM_SAIL_STEP5_DSA_PREFILL_FP8", "0")
    torch.manual_seed(164)
    scores = torch.randn(rows, width, device="cuda")
    scores[::2].mul_(64).round_()
    scores[1::8].fill_(0.5)
    scores[3::8, :4] = -torch.inf
    current = (torch.arange(rows, device="cuda") * 3).remainder(width)
    workspace, candidates, starts, table = allocate(scores, 512)
    raw_select(scores.unsqueeze(1), current, table, workspace, candidates, starts)
    scratch = torch.empty((rows, 513), device="cuda", dtype=torch.int32)
    expected = torch.empty_like(scratch, dtype=torch.int64)
    _stable_topk_regions[(rows,)](
        scores,
        current,
        candidates,
        scratch,
        expected,
        width,
        513,
        512,
        512,
        1024,
        1024,
        SORT=True,
    )
    torch.testing.assert_close(select_regions(scores, current, 512), expected)


@pytest.mark.parametrize("rows", [16, 64, 128])
@pytest.mark.parametrize("width", [4096, 16384, 32768])
def test_selector_topk_lengths_mirrors_current(rows, width):
    """Reusing int64 current for the int32 topk_lengths mirror must be exact.

    The decode scores and prefill-indexer metadata kernels store
    topk_lengths = current (position >> 3), so selector_v2 feeds current
    straight to the radix kernel instead of casting topk_lengths back to int64.
    """
    import torch

    from vllm_sail.models.step5.ops.selector_v2 import select_regions

    torch.manual_seed(165)
    scores = torch.randn(rows, width, device="cuda")
    scores[::2].mul_(64).round_()
    current = (torch.arange(rows, device="cuda") * 3).remainder(width)
    topk_lengths = current.to(torch.int32)
    baseline = select_regions(scores, current, 512)
    mirrored = select_regions(scores, current, 512, topk_lengths=topk_lengths)
    torch.testing.assert_close(mirrored, baseline, atol=0, rtol=0)
    decode = select_regions(scores, current, 512, topk_lengths=topk_lengths, sort=False)
    torch.testing.assert_close(decode, baseline, atol=0, rtol=0)


@pytest.mark.parametrize("threads", [256, 1024])
@pytest.mark.parametrize("kind", ["finite", "ties", "nan", "inf", "zeros", "short"])
def test_native_stable_sort_matches_triton(threads, kind):
    """The fused selector epilogue must equal _stable_topk_regions SORT=True.

    Covers the ignore-NaN cutoff, smallest-ID tie fill, ascending order, -1 left
    padding and current-last, including non-finite and full-context rows.
    """
    import torch
    from vllm.models.step5.nvidia.ops.sparse_indexer import _stable_topk_regions
    from vllm.triton_utils import triton

    from vllm_sail.models.step5.ops.selector_v2 import allocate, raw_select

    torch.manual_seed(77)
    rows, regions, k = 48, 8192, 512
    scores = torch.randn(rows, regions, device="cuda")
    if kind == "ties":
        scores.mul_(8).round_()
    elif kind == "nan":
        scores[:, ::37] = float("nan")
        scores[::3] = float("nan")
    elif kind == "inf":
        scores[:, ::53] = float("inf")
        scores[:, ::71] = float("-inf")
    elif kind == "zeros":
        scores.mul_(4).round_().mul_(0.25)
        scores[:, ::2] = torch.where(
            scores[:, ::2] >= 0,
            torch.tensor(0.0, device="cuda"),
            torch.tensor(-0.0, device="cuda"),
        )
    if kind == "short":
        current = torch.randint(0, k + 1, (rows,), device="cuda", dtype=torch.int64)
    else:
        current = torch.randint(
            0, regions + 1, (rows,), device="cuda", dtype=torch.int64
        )
        current[::5] = regions  # full-context boundary (native clamps to regions-1)

    workspace, candidates, starts, table = allocate(scores, k)
    raw_select(
        scores.view(rows, 1, regions),
        current,
        table,
        workspace,
        candidates,
        starts,
        topk=k,
        threads=threads,
    )
    cand = candidates.view(rows, k + 1).clone()
    native = torch.empty((rows, k + 1), device="cuda", dtype=torch.int64)
    raw_select(
        scores.view(rows, 1, regions),
        current,
        table,
        workspace,
        candidates,
        starts,
        topk=k,
        threads=threads,
        sorted=native,
    )

    scratch = torch.empty((rows, k + 1), device="cuda", dtype=torch.int32)
    ref = torch.empty((rows, k + 1), device="cuda", dtype=torch.int64)
    _stable_topk_regions[(rows,)](
        scores,
        current,
        cand,
        scratch,
        ref,
        scores.stride(0),
        k + 1,
        k,
        triton.next_power_of_2(k),
        1024,
        triton.next_power_of_2(k + 1),
        SORT=True,
    )
    torch.testing.assert_close(native, ref, atol=0, rtol=0)


@pytest.mark.parametrize("threads", [256, 1024])
@pytest.mark.parametrize("width", [4096, 32768])
def test_radix_keeps_entire_cutoff_bin(threads, width):
    import torch

    from vllm_sail.models.step5.ops.selector_v2 import allocate, raw_select

    topk = 512
    current = width - 1
    scores = torch.full((1, width), -2.0, device="cuda")
    # Exactly enough winners to take the entire cutoff bin. Give each later
    # streaming chunk its own lower-scoring cutoff bin to exercise merging too.
    scores[0, 4 : 4 + 2 * topk : 2] = 2.0
    if width > 16384:
        scores[0, 16384 : 16384 + topk] = 1.0
    scores[0, current] = 4.0
    positions = torch.tensor([current], device="cuda", dtype=torch.int64)
    workspace, output, starts, table = allocate(scores, topk)
    raw_select(
        scores.unsqueeze(1),
        positions,
        table,
        workspace,
        output,
        starts,
        topk=topk,
        threads=threads,
    )
    assert sorted(output[0, 0, :topk].tolist()) == list(range(4, 4 + 2 * topk, 2))
    assert output[0, 0, topk].item() == current


@pytest.mark.parametrize("threads", [256, 1024])
@pytest.mark.parametrize("kind", ["finite", "ties", "nan", "inf", "short"])
@pytest.mark.parametrize("regions", [4096, 8192])
@pytest.mark.parametrize("rows", [3072, 4096])
def test_native_sort_matches_prefill_stabilization(rows, regions, kind, threads):
    """The native epilogue must equal _stable_prefill_regions at prefill shapes.

    _stable_prefill_regions is the single-cutoff specialization of
    _stable_topk_regions SORT=True. This guards bitwise parity, but selector_v2
    keeps the Triton kernel for prefill_special: its specialization is faster
    than the native epilogue at those shapes.
    """
    import torch
    from vllm.triton_utils import triton

    from vllm_sail.models.step5.ops.selector_prefill import _stable_prefill_regions
    from vllm_sail.models.step5.ops.selector_v2 import allocate, raw_select

    torch.manual_seed(166)
    k = 512
    scores = torch.randn(rows, regions, device="cuda")
    if kind == "ties":
        scores.mul_(8).round_()
    elif kind == "nan":
        scores[:, ::37] = float("nan")
        scores[::3] = float("nan")
    elif kind == "inf":
        scores[:, ::53] = float("inf")
        scores[:, ::71] = float("-inf")
    if kind == "short":
        current = torch.randint(0, k + 1, (rows,), device="cuda", dtype=torch.int64)
    else:
        current = torch.randint(
            0, regions + 1, (rows,), device="cuda", dtype=torch.int64
        )
        current[::5] = regions

    workspace, candidates, starts, table = allocate(scores, k)
    raw_select(
        scores.view(rows, 1, regions),
        current,
        table,
        workspace,
        candidates,
        starts,
        topk=k,
        threads=threads,
    )
    cand = candidates.view(rows, k + 1).clone()
    native = torch.empty((rows, k + 1), device="cuda", dtype=torch.int64)
    raw_select(
        scores.view(rows, 1, regions),
        current,
        table,
        workspace,
        candidates,
        starts,
        topk=k,
        threads=threads,
        sorted=native,
    )

    scratch = torch.empty((rows, k + 1), device="cuda", dtype=torch.int32)
    ref = torch.empty((rows, k + 1), device="cuda", dtype=torch.int64)
    _stable_prefill_regions[(rows,)](
        scores,
        current,
        cand,
        scratch,
        ref,
        scores.stride(0),
        regions,
        k,
        512,
        1024,
        triton.next_power_of_2(regions),
    )
    torch.testing.assert_close(native, ref, atol=0, rtol=0)
