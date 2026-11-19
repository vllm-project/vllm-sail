# SPDX-License-Identifier: Apache-2.0
"""Upstream fixes with explicit installation and dependency-free imports."""


def install() -> None:
    from vllm_sail.patch.bugfix import (
        async_llm_shutdown,
        glm5_next_int8_mla_dequant,
        qwen4_exp_ple_quant_scheme,
    )

    async_llm_shutdown.install()
    glm5_next_int8_mla_dequant.install()
    qwen4_exp_ple_quant_scheme.install()
