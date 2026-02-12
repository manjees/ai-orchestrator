"""Abstract AI provider interface and shared data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self, Sequence


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class AIResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class AIProvider(ABC):
    """Base class for all AI providers."""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: int | None = None,
    ) -> AIResponse: ...

    @abstractmethod
    async def is_available(self) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
