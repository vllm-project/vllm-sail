# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen4Exp chat renderer for the Python frontend.

Qwen4Exp checkpoints render chat in one of two ways: some ship a Jinja chat
template, while others ship no template and instead provide a remote-code
tokenizer that renders conversations through a hand-written Python encoder
exposed via ``apply_chat_template``. The Python encoder emits structural markers
as special tokens interleaved with ordinary text, so we tokenize eagerly instead
of re-tokenizing a rendered string.

Upstream vLLM's HF renderer requires a resolved Jinja template, so a
template-less Qwen4Exp checkpoint fails every chat request. This renderer picks
the path from the loaded tokenizer: when a Jinja template is present it defers to
the standard HF rendering, and when absent it drives the native encoder. The
native encoder rejects any keyword it does not explicitly understand, so we
forward only the kwargs its signature accepts (plus ``enable_thinking``, which it
consumes from ``**kwargs``) and never pass a Jinja ``chat_template``.
"""

from typing import Any

from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.renderers.base import BaseRenderer
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.hf import (
    resolve_chat_template_content_format,
    safe_apply_chat_template,
)
from vllm.renderers.inputs import DictPrompt
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.renderers.params import ChatParams
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async
from vllm.utils.func_utils import supports_kw


def _qwen4exp_template_kwargs(
    tokenizer: HfTokenizer, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Keep only the kwargs the native Qwen4Exp encoder accepts.

    The remote-code ``apply_chat_template`` raises ``TypeError`` on unknown
    keywords, so drop everything its signature does not name. ``enable_thinking``
    is the one exception: the encoder pops it from ``**kwargs`` to drive
    ``thinking``, so it must survive the filter. ``chat_template`` is never
    forwarded because the native encoder has no Jinja template.
    """
    apply_chat_template = tokenizer.apply_chat_template
    filtered = {
        k: v
        for k, v in kwargs.items()
        if k != "chat_template"
        and (
            k == "enable_thinking"
            or supports_kw(apply_chat_template, k, allow_var_kwargs=False)
        )
    }
    # Native rendering must keep special-vs-ordinary token ids, so tokenize
    # eagerly and pin a flat token list regardless of what the request asked.
    filtered["tokenize"] = True
    filtered["return_dict"] = False
    return filtered


class Qwen4ExpRenderer(BaseRenderer[HfTokenizer]):
    """Render chat prompts for the Qwen4Exp architectures.

    Registered out-of-tree for the ``qwen4exp`` renderer/tokenizer mode, which the
    PPU platform selects for ``Qwen4ExpForCausalLM`` and
    ``Qwen4ExpForConditionalGeneration``. Uses the standard HF tokenizer (loaded
    via ``trust_remote_code`` when the checkpoint ships remote code). Checkpoints
    with a Jinja chat template render through the standard HF path; checkpoints
    without one render through the tokenizer's native Python encoder.
    """

    def __init__(self, config: VllmConfig, tokenizer: HfTokenizer | None) -> None:
        super().__init__(config, tokenizer)
        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _uses_native_encoder(self) -> bool:
        return getattr(self.get_tokenizer(), "chat_template", None) is None

    def _apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        params: ChatParams,
    ) -> list[int]:
        tokenizer = self.get_tokenizer()
        kwargs = params.get_apply_chat_template_kwargs()

        if not self._uses_native_encoder():
            kwargs["tokenize"] = True
            return safe_apply_chat_template(
                self.model_config, tokenizer, conversation, **kwargs
            )

        if kwargs.get("continue_final_message"):
            raise VLLMValidationError(
                "Qwen4Exp renderer does not support continue_final_message"
            )

        template_kwargs = _qwen4exp_template_kwargs(tokenizer, kwargs)
        try:
            return tokenizer.apply_chat_template(conversation, **template_kwargs)
        except (TypeError, ValueError) as e:
            # Malformed request content or an unsupported encoder argument;
            # surface as a 400 rather than a generic upstream wrapper.
            raise VLLMValidationError(str(e)) from e

    def _content_format(self, params: ChatParams):
        # The native encoder renders structural markers as special tokens in a
        # flat string prompt, so its content format is always "string". Only the
        # Jinja path needs to detect the template's content format.
        if self._uses_native_encoder():
            return "string"
        return resolve_chat_template_content_format(
            chat_template=params.chat_template,
            tools=params.chat_template_kwargs.get("tools"),
            given_format=params.chat_template_content_format,
            tokenizer=self.get_tokenizer(),
            model_config=self.model_config,
        )

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format=self._content_format(params),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt = parse_dec_only_prompt(self._apply_chat_template(conversation, params))
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format=self._content_format(params),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        token_ids = await self._apply_chat_template_async(conversation, params)
        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
