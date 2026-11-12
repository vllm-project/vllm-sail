# SPDX-License-Identifier: Apache-2.0
"""Bounded kpool decode must preserve paged INT8 score semantics."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def runtime():
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        pytest.skip("requires PPU")
    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    from vllm_sail.attention.ops import mla_sparse_kpool

    return torch, mla_sparse_kpool


@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("heads,next_n", [(16, 1), (32, 3)])
def test_paged_logits_match_reference(
    runtime, monkeypatch, block_size, unified, heads, next_n
):
    torch, module = runtime
    monkeypatch.setenv("VLLM_SAIL_KPOOL_MAX_DENSE_POOLS", "1")
    generator = torch.Generator().manual_seed(51)
    batch, width, dim = 3, 80, 128
    blocks_per_seq = (width + block_size - 1) // block_size
    blocks = batch * blocks_per_seq
    keys = torch.randint(
        -8, 9, (blocks, block_size, dim), generator=generator, dtype=torch.int8
    )
    scales = torch.rand(blocks, block_size, generator=generator) * 0.03 + 0.01
    backing = torch.full(
        (blocks, 3 if unified else 1, block_size, dim + 4), 0xA5, dtype=torch.uint8
    )
    for page in range(blocks):
        flat = backing[page, 1 if unified else 0].view(-1)
        flat[: block_size * dim] = keys[page].reshape(-1).view(torch.uint8)
        flat[block_size * dim :] = scales[page].view(torch.uint8)
    cache = backing.cuda()[:, 1 if unified else 0]
    table = (
        torch.randperm(blocks, generator=generator)
        .to(torch.int32)
        .reshape(batch, blocks_per_seq)
    )
    q = torch.randint(
        -8, 9, (batch, next_n, heads, dim), generator=generator, dtype=torch.int8
    )
    weights = torch.randn(batch * next_n, heads, generator=generator) * 0.1
    seq = torch.tensor(
        [
            [0] * next_n,
            [35 + i for i in range(next_n)],
            [65 + i for i in range(next_n)],
        ],
        dtype=torch.int32,
    )
    decode = torch.tensor([0, 1, next_n], dtype=torch.int32)
    expected = torch.full((batch * next_n, width), -float("inf"))
    for b in range(batch):
        for n in range(int(decode[b])):
            length = int(seq[b, n])
            slots = torch.arange(length)
            pages = table[b, slots // block_size].long()
            k = keys[pages, slots % block_size].float()
            s = scales[pages, slots % block_size]
            scores = (q[b, n].float() @ k.T).clamp_min(0)
            expected[b * next_n + n, :length] = (
                scores * weights[b * next_n + n, :, None]
            ).sum(0) * s
    actual, ks, ke = module._decode_logits_non_paged(
        cache,
        q.cuda(),
        weights.cuda(),
        seq.cuda(),
        decode.cuda(),
        table.cuda(),
        batch,
        next_n,
        batch * next_n,
        width * 4,
        4,
        dim,
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
    assert ks.cpu().tolist() == [0] * (batch * next_n)
    assert ke.cpu().tolist() == [0] * next_n + [35] + [0] * (next_n - 1) + [
        65 + i for i in range(next_n)
    ]
    topk = torch.empty((batch * next_n, 16), dtype=torch.int32, device="cuda")
    torch.ops._C.top_k_per_row_prefill(
        actual, ks, ke, topk, batch * next_n, actual.stride(0), actual.stride(1), 16
    )
    real_rows = (
        torch.arange(next_n, device="cuda")[None, :] < decode.cuda()[:, None]
    ).reshape(-1)
    topk[~real_rows] = -1
    for row, (indices, length) in enumerate(
        zip(topk.cpu(), ke.cpu().tolist(), strict=False)
    ):
        if length == 0:
            assert (indices == -1).all()
        else:
            assert ((indices >= 0) & (indices < length)).all()
            selected = expected[row, indices.long()].sort(descending=True).values
            torch.testing.assert_close(
                selected, expected[row, :length].topk(16).values, atol=1e-4, rtol=1e-4
            )


def test_graph_profile_geometry_does_not_gather_all_request_contexts(runtime):
    torch, module = runtime
    batch, dim, block = 512, 128, 128
    values = torch.ones(block * dim, dtype=torch.int8, device="cuda")
    scales = torch.full((block,), 0.01, device="cuda")
    cache = torch.empty((1, block, dim + 4), dtype=torch.uint8, device="cuda")
    cache.view(-1)[: block * dim] = values.view(torch.uint8)
    cache.view(-1)[block * dim :] = scales.view(torch.uint8)
    logits, ks, ke = module._decode_logits_non_paged(
        cache,
        torch.ones(batch, 1, 32, dim, dtype=torch.int8, device="cuda"),
        torch.ones(batch, 32, device="cuda"),
        torch.ones(batch, dtype=torch.int32, device="cuda"),
        torch.ones(batch, dtype=torch.int32, device="cuda"),
        torch.zeros(batch, 2048, dtype=torch.int32, device="cuda"),
        batch,
        1,
        batch,
        1048576,
        4,
        dim,
    )
    assert logits.shape == (512, 262144)
    torch.testing.assert_close(logits[:, 0], torch.full_like(logits[:, 0], 40.96))
    assert torch.isneginf(logits[:, 1:]).all()
    assert (ks == 0).all() and (ke == 1).all()
    topk = torch.empty((batch, 512), dtype=torch.int32, device="cuda")
    torch.ops._C.top_k_per_row_prefill(
        logits, ks, ke, topk, batch, logits.stride(0), logits.stride(1), 512
    )
    assert (topk[:, 0] == 0).all()
    assert (topk[:, 1:] == -1).all()
