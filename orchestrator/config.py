"""Pydantic-settings based configuration loaded from .env."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_PROJECTS_FILE = Path(__file__).parent / "projects.json"


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Required
    telegram_bot_token: str
    telegram_allowed_user_id: int

    # Optional
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:latest"
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    tmux_session_name: str = "ai_factory"
    log_level: str = "INFO"
    cmd_timeout: int = 30
    cmd_long_timeout: int = 600
    solve_timeout: int = 1800  # per-issue timeout (30 min)

    # Dual-Check System
    dual_check_enabled: bool = True
    deepseek_design_timeout: int = 600   # Step 1: 10 min
    deepseek_audit_timeout: int = 600    # Step 4: 10 min
    deepseek_design_max_tokens: int = 8192
    deepseek_audit_max_tokens: int = 4096


def load_projects() -> dict[str, dict]:
    """Read projects.json and return {name: {path: ...}} with expanded paths."""
    if not _PROJECTS_FILE.exists():
        logger.warning("projects.json not found at %s", _PROJECTS_FILE)
        return {}
    with open(_PROJECTS_FILE) as f:
        data: dict[str, dict] = json.load(f)
    # Expand ~ in paths
    for info in data.values():
        if "path" in info:
            info["path"] = os.path.expanduser(info["path"])
    return data
