"""Ollama (local LLM) AI provider via httpx REST calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from .base import AIProvider, AIResponse, Message, Role

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OllamaModel:
    name: str
    size_gb: float


class OllamaProvider(AIProvider):
    """Ollama REST API provider."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=300)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: int | None = None,
    ) -> AIResponse:
        payload_messages: list[dict[str, str]] = []
        if system_prompt:
            payload_messages.append({"role": Role.SYSTEM.value, "content": system_prompt})
        for msg in messages:
            payload_messages.append({"role": msg.role.value, "content": msg.content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": payload_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        effective_timeout = timeout if timeout is not None else 300
        resp = await self._client.post("/api/chat", json=payload, timeout=effective_timeout)
        if resp.status_code != 200:
            # Ollama returns 404 with error body for missing models — surface it
            try:
                err_body = resp.json().get("error", resp.text)
            except Exception:
                err_body = resp.text
            raise httpx.HTTPStatusError(
                f"Ollama error: {err_body}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()

        return AIResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", self._model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            finish_reason=data.get("done_reason", ""),
            raw=data,
        )

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get("/", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    async def get_loaded_models(self) -> list[OllamaModel]:
        """Fetch currently loaded models via GET /api/tags."""
        try:
            resp = await self._client.get("/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models: list[OllamaModel] = []
            for m in data.get("models", []):
                size_bytes = m.get("size", 0)
                models.append(
                    OllamaModel(
                        name=m.get("name", "unknown"),
                        size_gb=round(size_bytes / (1024**3), 1),
                    )
                )
            return models
        except Exception:
            logger.debug("Failed to fetch Ollama models", exc_info=True)
            return []

    async def close(self) -> None:
        await self._client.aclose()
