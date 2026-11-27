# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-tree chat renderers for PPU.

Importing this package pulls in nothing: the leaf modules import ``vllm`` at
module top level and are reached only through the registration in
``vllm_sail/registry/renderers.py``, which stores their module path and lets
vLLM resolve the class lazily when the renderer mode is first loaded.
"""
