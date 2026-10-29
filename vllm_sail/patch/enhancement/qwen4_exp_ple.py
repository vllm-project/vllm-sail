# SPDX-License-Identifier: Apache-2.0
"""Compressed-tensors coverage for the Qwen4Exp PLE n-gram embedding.

``Qwen4ExpPLEEmbeddingMethod.from_quant_config`` picks the storage format of the
PLE n-gram table. Upstream dispatches over an explicit whitelist — ``None``,
``ModelOptMixedPrecisionConfig``, ``ModelOptQuantConfigBase`` and ``Fp8Config``
— and rejects everything else:

.. code-block:: python

    if not isinstance(quant_config, Fp8Config):
        raise NotImplementedError(
            "Qwen4Exp PLE embedding does not support quantization config "
            f"{type(quant_config).__name__}"
        )

The PPU quantization pipeline emits ``CompressedTensorsConfig`` checkpoints
(FP8 per-channel weights with per-token dynamic activations), so loading a
Qwen4Exp model from one kills every worker inside ``load_model`` before the API
server can bind a port:

.. code-block:: text

    NotImplementedError: Qwen4Exp PLE embedding does not support quantization
    config CompressedTensorsConfig

Those checkpoints do not quantize the PLE table at all. Every PLE weight is
listed in ``quantization_config.ignore`` — the ``ple.ple_embedding.ngram_embedding.shard_*``
tensors (one per ``split_ngram_parts`` entry) plus ``ple.conv1d``,
``ple.key_proj`` and the PLE norms — so the table is stored in ``params_dtype``
and must be loaded that way. The ``ModelOptQuantConfigBase`` branch
(``is_layer_excluded``) and the ``Fp8Config`` branch (``is_layer_skipped`` plus
a ``shard_`` prefix scan) already honour exactly this exclusion; compressed
tensors is the one family that never reaches a check.

This patch adds that missing branch and delegates everything else, so a
compressed-tensors checkpoint that genuinely *does* quantize the PLE table
still fails loudly upstream instead of silently loading unquantized weights
over quantized ones.

Leaf-style module (like ``compressed_tensors_ppu.py``): importable on a bare
CPU runner, patches install when ``install()`` runs.
"""

from __future__ import annotations

from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.qwen4_exp.common.ngram_embedding"
AFFECTED_VERSIONS = ">=0.31.0,<0.32.0"

REASON_PLE_IGNORE = (
    "Upstream's Qwen4ExpPLEEmbeddingMethod.from_quant_config whitelists None, "
    "ModelOpt and Fp8Config and raises NotImplementedError for anything else, "
    "so a CompressedTensorsConfig checkpoint that lists every PLE weight in "
    "quantization_config.ignore still fails to load. The ModelOpt and Fp8 "
    "branches already honour that exclusion; compressed tensors never reaches "
    "a check."
)
REMOVE_WHEN_PLE_IGNORE = (
    "upstream's Qwen4ExpPLEEmbeddingMethod.from_quant_config grows a "
    "CompressedTensorsConfig branch that returns the unquantized method when "
    "the PLE table is excluded by quantization_config.ignore."
)

METADATA = (
    (
        f"{_MODULE}.Qwen4ExpPLEEmbeddingMethod.from_quant_config",
        REASON_PLE_IGNORE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_PLE_IGNORE,
    ),
)

_installed = False


def install() -> None:
    """Apply the patch. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )
    from vllm.models.qwen4_exp.common import ngram_embedding as _ng

    # Captured before patching so the replacement can delegate to upstream.
    # from_quant_config is a staticmethod, so class access yields the plain
    # function and it is called without a leading self/cls.
    _upstream_from_quant_config = _ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def _ple_table_is_ignored(quant_config: CompressedTensorsConfig, prefix: str):
        """Whether the checkpoint stores this PLE table unquantized."""
        ignore = quant_config.ignore
        fused_mapping = quant_config.packed_modules_mapping
        if should_ignore_layer(prefix, ignore=ignore, fused_mapping=fused_mapping):
            return True
        # The table is one runtime parameter assembled from shard_* checkpoint
        # tensors, so an ignore list naming the shards excludes the whole table
        # without ever naming its parent. Mirrors the upstream Fp8Config branch.
        shard_prefix = f"{prefix}.shard_"
        return any(name.startswith(shard_prefix) for name in ignore)

    @patch(
        _MODULE,
        "Qwen4ExpPLEEmbeddingMethod.from_quant_config",
        reason=REASON_PLE_IGNORE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_PLE_IGNORE,
    )
    def _from_quant_config(quant_config, prefix, embedding_dtype=None):
        # Upstream resolves an explicitly declared FP8 PLE dtype before it ever
        # looks at the quantization config, so delegating first keeps that
        # precedence (and every config family upstream does handle) intact.
        # Only the blanket rejection of CompressedTensorsConfig is rescued, and
        # only when the checkpoint's ignore list says the table is unquantized;
        # a compressed-tensors checkpoint that really does quantize the PLE
        # table re-raises upstream's error instead of loading it wrong.
        try:
            return _upstream_from_quant_config(quant_config, prefix, embedding_dtype)
        except NotImplementedError:
            if isinstance(
                quant_config, CompressedTensorsConfig
            ) and _ple_table_is_ignored(quant_config, prefix):
                return _ng.Qwen4ExpPLEUnquantizedEmbeddingMethod()
            raise

    _installed = True
