# SPDX-License-Identifier: Apache-2.0
"""Execute the indexer metadata producer and preserve its shared block table."""

import ast
from types import SimpleNamespace as NS

import pytest

from tests.support.source import definition, function

PATCH = "vllm_sail/patch/enhancement/attention/mla_indexer.py"


class CpuValues(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return CpuValues(value) if isinstance(index, slice) else value

    def max(self):
        return NS(item=lambda: max(self))


@pytest.fixture
def producer():
    def prepare(fp4=False, empty=False):
        splits = []

        def split(*args, **kwargs):
            splits.append((args, kwargs))
            return []

        class Metadata(NS):
            def __init__(self, *, block_table=None, **kwargs):
                super().__init__(block_table=block_table, **kwargs)

        common = NS(
            num_reqs=1,
            num_actual_tokens=0 if empty else 1,
            query_start_loc=object(),
            query_start_loc_cpu=CpuValues([0, 1]),
            seq_lens=object(),
            slot_mapping=object(),
            block_table_tensor=object(),
            dcp_local_seq_lens=None,
            max_seq_len=17,
            seq_lens_cpu_upper_bound=CpuValues([17]),
        )
        builder = NS(
            compress_ratio=1,
            decode_threshold=1,
            use_flattening=False,
            supports_varlen=False,
            use_pcp=False,
            max_prefill_buffer_size=128,
            indexer_uses_fp4=fp4,
            _prefill_split_seq_lens=lambda values: values,
            _split_indexer_prefill_chunks=split,
        )
        namespace = {
            "torch": NS(
                diff=lambda values: [values[1] - values[0]],
                bfloat16="bfloat16",
                float32="float32",
            ),
            "envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1),
            "split_decodes_and_prefills": lambda *args, **kwargs: (0, 1, 0, 1),
            "DeepseekV32IndexerMetadata": Metadata,
            "DeepseekV32IndexerPrefillMetadata": lambda chunks, **kwargs: NS(
                chunks=chunks, **kwargs
            ),
        }
        return builder, common, namespace, splits

    return prepare


@pytest.mark.parametrize("fp4", [False, True])
def test_builder_preserves_shared_block_table_and_logits_dtype(producer, fp4):
    builder, common, namespace, splits = producer(fp4=fp4)
    result = function(PATCH, "_build_body", namespace)(builder, 0, common)
    assert result.block_table is common.block_table_tensor
    assert result.seq_lens is common.seq_lens
    assert result.slot_mapping is common.slot_mapping
    assert splits[0][1]["logits_dtype"] == ("bfloat16" if fp4 else "float32")


def test_empty_placeholder_keeps_optional_block_table_default(producer):
    builder, common, namespace, splits = producer(empty=True)
    result = function(PATCH, "_build_body", namespace)(builder, 0, common)
    assert result.block_table is None
    assert result.num_decode_tokens == result.num_prefill_tokens == 0
    assert not splits


@pytest.mark.upstream_source
def test_selected_upstream_producer_preserves_the_same_table(
    upstream_source_root, producer
):
    path = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    node = definition(path, "DeepseekV32IndexerMetadataBuilder.build")
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    builder, common, namespace, _ = producer()
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    assert (
        namespace["build"](builder, 0, common).block_table is common.block_table_tensor
    )
