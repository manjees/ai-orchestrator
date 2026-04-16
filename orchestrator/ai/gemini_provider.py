"""Gemini CLI subprocess wrapper — headless mode via `-p` flag."""

from __future__ import annotations

import asyncio
import logging

from .base import AIProvider, AIResponse, Message

logger = logging.getLogger(__name__)

_GEMINI_BIN = "/opt/homebrew/bin/gemini"


class GeminiCLIProvider(AIProvider):
    """Wraps the Gemini CLI for text-only analysis tasks (design critique, cross-review)."""

    def __init__(self, model: str = "gemini-2.5-pro") -> None:
        self._model = model

    async def chat(
        self,
        messages: list[Message] | tuple[Message, ...],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: int | None = None,
    ) -> AIResponse:
        """Send a prompt to Gemini CLI and return the response."""
        # Build the prompt from messages (combine system + user content)
        parts: list[str] = []
        if system_prompt:
            parts.append(system_prompt)
        for msg in messages:
            parts.append(msg.content)
        prompt = "\n\n".join(parts)

        proc = await asyncio.create_subprocess_exec(
            _GEMINI_BIN, "-p", prompt, "-m", self._model, "-o", "text",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout or 600,
        )
        output = stdout.decode(errors="replace") if stdout else ""

        if proc.returncode != 0:
            err = stderr.decode(errors="replace") if stderr else ""
            raise RuntimeError(
                f"Gemini CLI failed (exit={proc.returncode}): {err[:300]}"
            )

        if not output.strip():
            raise RuntimeError("Gemini CLI returned empty response")

        return AIResponse(content=output, model=self._model)

    async def is_available(self) -> bool:
        """Health check: run a real prompt to verify Gemini CLI works end-to-end."""
        try:
            proc = await asyncio.create_subprocess_exec(
                _GEMINI_BIN, "-p", "ping", "-o", "text",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode(errors="replace") if stdout else ""
            return proc.returncode == 0 and len(output.strip()) > 0
        except Exception:
            return False

    async def close(self) -> None:
        """No-op — stateless subprocess provider."""
