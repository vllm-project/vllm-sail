# SPDX-License-Identifier: Apache-2.0


def register():
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    from vllm_sail.models.step5.ops.gate_linear import PPUStep5GateLinear

    registered = op_registry_oot.get("GateLinear")
    if registered is None:
        GateLinear.register_oot(PPUStep5GateLinear)
    elif registered is not PPUStep5GateLinear:
        raise RuntimeError("GateLinear already has an out-of-tree implementation")
