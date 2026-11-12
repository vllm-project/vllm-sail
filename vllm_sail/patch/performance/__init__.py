# SPDX-License-Identifier: Apache-2.0
"""PPU performance patches.

Optimisations that PPU needs on top of upstream's code paths. Unlike the
`enhancement` category, removing these does not break correctness — it costs
performance.

Contains `fla` (flash-linear-attention): the fork's cache_results autotune
additions and the Qwen3-Next fused GDN decode path; and `step5_dsa`: the
opt-in resident-block Step5 DSA indexer scoring kernel.
"""

from vllm_sail.patch.performance import fla  # noqa: F401
from vllm_sail.patch.performance import step5_dsa  # noqa: F401
from vllm_sail.patch.performance import step5_dsa_selector  # noqa: F401
from vllm_sail.patch.performance import step5_dsa_prefill  # noqa: F401
from vllm_sail.patch.performance import step5_norm  # noqa: F401
