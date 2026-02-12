"""Anthropic Claude AI provider."""

from __future__ import annotations

import logging
from typing import Sequence

import anthropic

from .base import AIProvider, AIResponse, Message

logger = logging.getLogger(__name__)


class AnthropicProvider(AIProvider):
    """Anthropic API provider using the official SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: int | None = None,
    ) -> AIResponse:
        api_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)

        content_parts = [
            block.text for block in response.content if hasattr(block, "text")
        ]

        return AIResponse(
            content="\n".join(content_parts),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason or "",
            raw=response.model_dump(),
        )

    async def is_available(self) -> bool:
        try:
            await self._client.messages.create(
                model=self._model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
