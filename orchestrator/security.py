"""Telegram user ID whitelist filter and secret masking utilities."""

import re

from telegram.ext.filters import BaseFilter, UpdateFilter


_SECRET_PATTERNS = re.compile(
    r"(?:"
    r"sk-ant-[A-Za-z0-9_-]{10,}"  # Anthropic API keys
    r"|sk-[A-Za-z0-9_-]{20,}"     # OpenAI-style API keys
    r"|ghp_[A-Za-z0-9]{36,}"      # GitHub personal access tokens
    r"|gho_[A-Za-z0-9]{36,}"      # GitHub OAuth tokens
    r"|xoxb-[A-Za-z0-9-]+"        # Slack bot tokens
    r"|xoxp-[A-Za-z0-9-]+"        # Slack user tokens
    r")"
)


def mask_secrets(text: str) -> str:
    """Replace known secret patterns with [MASKED]."""
    return _SECRET_PATTERNS.sub("[MASKED]", text)


class AuthFilter(UpdateFilter):
    """Allow only messages from whitelisted Telegram user IDs."""

    def __init__(self, allowed_user_id: int) -> None:
        super().__init__()
        self._allowed = {allowed_user_id}

    def filter(self, update: object) -> bool:  # noqa: A003
        from telegram import Update

        if not isinstance(update, Update) or update.effective_user is None:
            return False
        return update.effective_user.id in self._allowed
