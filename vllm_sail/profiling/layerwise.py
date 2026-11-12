# SPDX-License-Identifier: Apache-2.0
"""在插件中补齐 V1/V2 的 Layer NVTX 和 draft/MTP 模块遍历。

沿用 --enable-layerwise-nvtx-tracing 和上游 PytHooks 标签，不依赖
VLLM_SAIL_NVTX_PROFILE 或 PyPI nvtx。只在首次 target/draft tracing 完成后
注册，不改变模型、采样、编译或 CUDA Graph 配置。
"""

from __future__ import annotations

import functools
import importlib

from vllm.logger import init_logger

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.33.0,<0.34.0"
_REMOVE_WHEN = "上游 V1/V2 均原生支持 draft/MTP Layer NVTX 并对共享模块去重。"
_INSTALLED: set[str] = set()


def _make_hooks():
    """复用 v1 输入/输出标签，仅调整注册去重和异常时的范围闭合。"""
    import torch
    from vllm.utils.nvtx_pytorch_hooks import PytHooks

    class SharedModuleHooks(PytHooks):
        def register_hooks(self, network_model, module_prefix="top"):
            skip_types = (
                torch.nn.Identity,
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
            )
            for name, module in network_model.named_modules(prefix=module_prefix):
                if isinstance(module, skip_types) or module in self.module_to_name_map:
                    continue
                # 先注册 target，再遍历 draft；共享层始终保留首次名称。
                self.module_to_name_map[module] = name
                module.register_forward_pre_hook(
                    self.module_fwd_pre_hook, with_kwargs=True
                )
                module.register_forward_hook(self.module_fwd_hook, always_call=True)

    return SharedModuleHooks()


def register_layerwise_nvtx_hooks(self) -> None:
    """在 runner 已完成首次 tracing 的调用点注册，关闭时不遍历模型。"""
    config = self.vllm_config
    if not config.observability_config.enable_layerwise_nvtx_tracing:
        return
    if getattr(self, "layerwise_nvtx_hooks_registered", False):
        return

    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    compilation = self.compilation_config
    if compilation.mode == CompilationMode.STOCK_TORCH_COMPILE:
        logger.warning_once(
            "STOCK_TORCH_COMPILE 不支持逐层 NVTX hooks，跳过注册。"
            "请使用 --enforce-eager 查看逐层标记。"
        )
        return
    if (
        compilation.mode != CompilationMode.NONE
        or compilation.cudagraph_mode != CUDAGraphMode.NONE
    ):
        logger.warning_once(
            "编译路径可能绕过 module hooks，CUDA Graph replay 不会重新执行"
            "逐层 NVTX hooks。请使用 --enforce-eager 查看逐层标记。"
        )

    # 先解析 draft 类型再注册，避免可选导入失败留下半套 target hooks。
    # 仅最后一个 PP rank 持有 drafter/speculator，其他 rank 只注册 target。
    if hasattr(self, "speculator"):
        from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

        proposer = self.speculator
        draft = proposer.model if isinstance(proposer, DraftModelSpeculator) else None
    elif config.speculative_config:
        from vllm.v1.spec_decode.dflash import DFlashProposer
        from vllm.v1.spec_decode.draft_model import DraftModelProposer
        from vllm.v1.spec_decode.eagle import EagleProposer
        from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

        proposer = getattr(self, "drafter", None)
        draft = (
            proposer.model
            if isinstance(
                proposer,
                (EagleProposer, DraftModelProposer, DFlashProposer, Gemma4Proposer),
            )
            else None
        )
    else:
        draft = None
    hooks = _make_hooks()
    hooks.register_hooks(self.model, self.model.__class__.__name__)
    if draft is not None:
        hooks.register_hooks(draft, f"draft.{draft.__class__.__name__}")
    self._sail_layerwise_nvtx_hooks = hooks
    self.layerwise_nvtx_hooks_registered = True


def install() -> None:
    """安装配置控制的兼容入口，与外层 profiling 环境变量独立且幂等。"""
    for module_path, is_v2 in (
        ("vllm.v1.worker.gpu_model_runner", False),
        ("vllm.v1.worker.gpu.model_runner", True),
    ):
        if module_path in _INSTALLED:
            continue
        try:
            module = importlib.import_module(module_path)
            runner = module.GPUModelRunner
            # 先校验全部入口，再安装；V1 原调用点已位于 draft dummy_run 之后。
            original_dummy_run = runner._dummy_run if is_v2 else None
            if not is_v2:
                getattr(runner, "_register_layerwise_nvtx_hooks")

            @functools.wraps(register_layerwise_nvtx_hooks)
            def register(self):
                return register_layerwise_nvtx_hooks(self)

            patch(
                module_path,
                "GPUModelRunner._register_layerwise_nvtx_hooks",
                reason="[profiling, 配置控制] 支持 V1/V2 target 与 draft/MTP 逐层标记及共享模块去重。",
                affected_versions=_AFFECTED,
                remove_when=_REMOVE_WHEN,
                allow_missing=is_v2,
            )(register)
            if is_v2:
                _install_v2_dummy_run(module_path, original_dummy_run)
            _INSTALLED.add(module_path)
        except Exception as exc:
            # 可选依赖缺失或上游接口漂移不应阻止引擎启动。
            logger.warning("Layer NVTX 兼容入口未安装到 %s：%s", module_path, exc)


def _install_v2_dummy_run(module_path, original):
    @patch(
        module_path,
        "GPUModelRunner._dummy_run",
        reason="[profiling, 配置控制] 在 V2 target/draft 首次 dummy forward 均完成后注册 Layer NVTX。",
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )
    @functools.wraps(original)
    def dummy_run(self, *args, **kwargs):
        # 不在 execute_model(dummy_run=True) 返回时注册：此时 draft 尚未 tracing。
        result = original(self, *args, **kwargs)
        self._register_layerwise_nvtx_hooks()
        return result
