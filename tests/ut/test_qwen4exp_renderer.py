# SPDX-License-Identifier: Apache-2.0
"""Runtime tests for the out-of-tree Qwen4Exp renderer and its registration.

Needs installed torch/vLLM (``requires_runtime``): the renderer subclasses
upstream ``BaseRenderer`` and registration mutates the real upstream registries.
The native encoder is stood in for by a strict stub that mirrors the remote-code
tokenizer's signature, so no checkpoint vocabulary is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.requires_runtime


class StubQwen4ExpTokenizer:
    """Mimics the remote-code Qwen4Exp tokenizer's native Python encoder.

    Like the real encoder it names the kwargs it understands, consumes
    ``enable_thinking`` from ``**kwargs``, and raises ``TypeError`` on anything
    else. It has no Jinja ``chat_template`` so the renderer takes the native path.
    """

    chat_template = None

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.calls: list[dict[str, Any]] = []
        self.conversations: list[list[dict[str, Any]]] = []

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
        thinking=None,
        reasoning_effort=None,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_dict=False,
        preserve_thinking=True,
        *,
        return_extra_metadata=False,
        **kwargs,
    ) -> list[int]:
        if "enable_thinking" in kwargs:
            thinking = bool(kwargs.pop("enable_thinking"))
        if kwargs:
            raise TypeError(
                "unsupported apply_chat_template arguments: "
                + ", ".join(sorted(kwargs))
            )
        self.conversations.append(conversation)
        self.calls.append(
            {
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "thinking": thinking,
                "reasoning_effort": reasoning_effort,
                "return_dict": return_dict,
            }
        )
        return list(self.token_ids)


@dataclass
class MockHFConfig:
    model_type: str = "qwen4_exp"


@dataclass
class MockModelConfig:
    runner_type: str = "generate"
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    multimodal_config: Any = None
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    allowed_local_media_path: str = ""
    allowed_media_domains: Any = None
    enable_prompt_embeds: bool = False
    renderer_num_workers: int = 1
    revision: str | None = None
    code_revision: str | None = None
    trust_remote_code: bool = True
    tokenizer_revision: str | None = None


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    from vllm.exceptions import VLLMValidationError
    from vllm.renderers.params import ChatParams
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    from vllm_sail.registry import renderers as renderers_registry
    from vllm_sail.renderers.qwen4exp import Qwen4ExpRenderer

    return SimpleNamespace(
        Qwen4ExpRenderer=Qwen4ExpRenderer,
        ChatParams=ChatParams,
        VLLMValidationError=VLLMValidationError,
        RENDERER_REGISTRY=RENDERER_REGISTRY,
        TokenizerRegistry=TokenizerRegistry,
        renderers_registry=renderers_registry,
    )


def _make_renderer(runtime, tokenizer: StubQwen4ExpTokenizer):
    config = MockVllmConfig(MockModelConfig(), MockParallelConfig())
    return runtime.Qwen4ExpRenderer(config, tokenizer)


def test_qwen4exp_mode_registered(runtime):
    runtime.renderers_registry.register()
    assert (
        runtime.RENDERER_REGISTRY.load_renderer_cls("qwen4exp").__name__
        == "Qwen4ExpRenderer"
    )
    assert (
        runtime.TokenizerRegistry.load_tokenizer_cls("qwen4exp").__name__
        == "CachedHfTokenizer"
    )


def test_native_encoder_forces_tokenize_and_pins_return_dict(runtime):
    tokenizer = StubQwen4ExpTokenizer([7, 8, 9])
    renderer = _make_renderer(runtime, tokenizer)
    params = runtime.ChatParams(chat_template_kwargs={"tokenize": False})

    token_ids = renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}], params
    )

    assert token_ids == [7, 8, 9]
    kwargs = tokenizer.calls[-1]
    # Native rendering keeps special-vs-ordinary token ids, so tokenize is
    # forced on and return_dict pinned False for a flat token list.
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is False


def test_native_encoder_drops_jinja_and_api_only_kwargs(runtime):
    tokenizer = StubQwen4ExpTokenizer([1, 2])
    renderer = _make_renderer(runtime, tokenizer)
    # chat_template / continue_final_message / documents are not understood by
    # the native encoder and must be filtered out, not forwarded.
    params = runtime.ChatParams(
        chat_template="{{ some jinja }}",
        chat_template_kwargs={
            "continue_final_message": False,
            "documents": [{"title": "d"}],
            "add_generation_prompt": True,
        },
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    # No TypeError raised means the unsupported kwargs were dropped.
    assert tokenizer.calls[-1]["add_generation_prompt"] is True


def test_native_encoder_forwards_thinking_and_tools(runtime):
    tokenizer = StubQwen4ExpTokenizer([1, 2, 3])
    renderer = _make_renderer(runtime, tokenizer)
    tools = [{"type": "function", "function": {"name": "search"}}]
    params = runtime.ChatParams(
        chat_template_kwargs={
            "tools": tools,
            "enable_thinking": False,
            "reasoning_effort": "high",
        }
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["tools"] == tools
    # enable_thinking is consumed by the encoder to drive thinking.
    assert kwargs["thinking"] is False
    assert kwargs["reasoning_effort"] == "high"


def test_native_encoder_rejects_continue_final_message(runtime):
    tokenizer = StubQwen4ExpTokenizer([1])
    renderer = _make_renderer(runtime, tokenizer)
    params = runtime.ChatParams(chat_template_kwargs={"continue_final_message": True})

    with pytest.raises(runtime.VLLMValidationError, match="continue_final_message"):
        renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls == []


def test_render_messages_returns_token_prompt(runtime):
    renderer = _make_renderer(runtime, StubQwen4ExpTokenizer([1, 2, 3]))

    conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], runtime.ChatParams()
    )

    assert prompt == {"prompt_token_ids": [1, 2, 3]}
    assert "multi_modal_data" not in prompt
    assert conversation[0]["role"] == "user"


def test_render_messages_async_returns_token_prompt(runtime):
    import asyncio

    renderer = _make_renderer(runtime, StubQwen4ExpTokenizer([4, 5]))

    async def scenario():
        return await renderer.render_messages_async(
            [{"role": "user", "content": "hi"}], runtime.ChatParams()
        )

    conversation, prompt = asyncio.run(scenario())

    assert prompt == {"prompt_token_ids": [4, 5]}
    assert conversation[0]["role"] == "user"
