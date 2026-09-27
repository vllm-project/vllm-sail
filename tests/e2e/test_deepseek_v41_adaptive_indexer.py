# SPDX-License-Identifier: Apache-2.0
"""Device-length flattening and paged-logits oracle for adaptive verification."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


def test_adaptive_indexer_backend_gate():
    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.v1.worker.gpu.attn_utils import (
        get_query_lens_mismatch_unsupported_backend,
    )

    group = SimpleNamespace(backend=DeepseekV41IndexerBackend, layer_names=["indexer"])
    assert get_query_lens_mismatch_unsupported_backend([[group]]) is None


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_device_lengths_flattened_logits_graph_replay(compress_ratio):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

    from vllm_sail.utils.deep_gemm import (
        fp8_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
    )

    # Same total budget; CPU plans [3,3], device reallocates to [5,1] or [1,5].
    capacity, tokens, block_size, heads, dim = 8, 6, 128, 64, 128
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True)
    )
    builder.supports_varlen = False
    builder.decode_seq_lens_buffer = torch.zeros(
        capacity, dtype=torch.int32, device="cuda"
    )
    builder.decode_lens_buffer = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    builder.expanded_block_table_buffer = torch.zeros(
        capacity, 2, dtype=torch.int32, device="cuda"
    )
    builder.arange_buffer = torch.arange(capacity, dtype=torch.int32, device="cuda")
    cpu_lens = torch.tensor([3, 3], dtype=torch.int32)
    lens = torch.tensor([5, 1], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    context = torch.tensor([127, 185], dtype=torch.int32, device="cuda")
    seq_lens = context + lens
    table_cpu = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    table = table_cpu.cuda()
    generator = torch.Generator().manual_seed(83)
    q_cpu = (torch.randn(capacity, 1, heads, dim, generator=generator) / 8).to(
        torch.float8_e4m3fn
    )
    k_cpu = (torch.randn(4, block_size, dim, generator=generator) / 8).to(
        torch.float8_e4m3fn
    )
    scales_cpu = torch.rand(4, block_size, generator=generator) / 2 + 0.25
    weight_cpu = torch.rand(capacity, heads, generator=generator) / heads
    packed = torch.cat(
        (k_cpu.view(torch.uint8).flatten(1), scales_cpu.view(torch.uint8).flatten(1)),
        dim=1,
    )
    cache = packed.reshape(4, block_size, 1, dim + 4).cuda()
    q, weights = q_cpu.cuda(), weight_cpu.cuda()
    sms = get_num_sms()

    def step():
        ends, blocks, lengths, batch, padding = builder._prepare_decode_tensors(
            seq_lens,
            table,
            lens,
            cpu_lens,
            starts,
            num_decodes=2,
            num_decode_tokens=capacity,
            use_native=False,
            next_n=6,
            max_decode_len=3,
        )
        assert batch == capacity and not padding
        ends = (ends // compress_ratio).unsqueeze(-1)
        schedule = get_paged_mqa_logits_metadata(ends, block_size, sms, None)
        logits = fp8_paged_mqa_logits(
            q, cache, weights, ends, blocks, schedule, 256, True
        )
        return ends, blocks, lengths, logits

    # Compile first, then capture and replay with changed device-only boundaries.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    for device_lens in ([5, 1], [1, 5]):
        lens.copy_(torch.tensor(device_lens, dtype=torch.int32))
        starts.copy_(torch.tensor([0, device_lens[0]], dtype=torch.int32))
        seq_lens.copy_(context + lens)
        graph.replay()
        ends, blocks, lengths, logits = (tensor.cpu() for tensor in actual)
        expected_ends, request_ids = [], []
        for req, count in enumerate(device_lens):
            base = (127, 185)[req]
            expected_ends.extend((base + i + 1) // compress_ratio for i in range(count))
            request_ids.extend([req] * count)
        assert ends.flatten().tolist() == expected_ends + [0] * (capacity - tokens)
        assert lengths.tolist() == [1] * capacity
        torch.testing.assert_close(
            blocks[:tokens], table_cpu[request_ids], rtol=0, atol=0
        )
        for row, (req, end) in enumerate(zip(request_ids, expected_ends, strict=False)):
            positions = torch.arange(end)
            pages = table_cpu[req, positions // block_size].long()
            slots = positions % block_size
            keys = k_cpu.float()[pages, slots] * scales_cpu[pages, slots, None]
            reference = (
                (q_cpu[row, 0].float() @ keys.T).clamp_min(0) * weight_cpu[row, :, None]
            ).sum(0)
            torch.testing.assert_close(
                logits[row, :end], reference, rtol=0.01, atol=0.002
            )
            assert torch.isneginf(logits[row, end:]).all()
        assert torch.isneginf(logits[tokens:]).all()
