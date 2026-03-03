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
    solve_timeout: int = 3600  # per-issue timeout (60 min)

    # Dual-Check System
    dual_check_enabled: bool = True
    deepseek_design_timeout: int = 600   # Step 1: 10 min
    claude_review_timeout: int = 900     # Step 3: 15 min
    deepseek_audit_timeout: int = 600    # Step 4: 10 min
    deepseek_design_max_tokens: int = 8192
    deepseek_audit_max_tokens: int = 4096
    max_review_retries: int = 2

    # Triple-Model Hybrid (Qwen Coder)
    qwen_coder_model: str = "qwen2.5-coder:32b"
    enable_data_mining: bool = True
    qwen_impl_timeout: int = 600          # Step 2: Qwen pre-impl (10 min)
    data_mining_timeout: int = 300         # Step 5: training data gen (5 min)
    data_mining_max_tokens: int = 4096
    training_data_dir: str = ""            # Empty = default to ai-orchestrator/data/training/

    # Five-brid Pipeline
    pipeline_mode: str = "legacy"          # "legacy" | "fivebrid"

    # Gemini
    gemini_model: str = "gemini-2.5-pro"

    # Phase 0: Research (Haiku CLI)
    haiku_model: str = "claude-haiku-4-5-20251001"
    research_timeout: int = 120
    research_max_tokens: int = 4096

    # Phase A: Design (Opus CLI + Gemini critique)
    opus_model: str = "claude-opus-4-6"
    opus_design_timeout: int = 900         # 15 min
    opus_design_max_tokens: int = 16384
    gemini_critique_timeout: int = 600     # 10 min
    max_design_retries: int = 2

    # Phase B: Self-Review (Sonnet CLI)
    sonnet_model: str = "claude-sonnet-4-6"
    sonnet_self_review_timeout: int = 600  # 10 min

    # Phase C: Cross-Review (Gemini CLI)
    gemini_cross_review_timeout: int = 600  # 10 min

    # Service
    launchd_label: str = "com.ai-orchestrator"

    # Init Pipeline
    projects_base_dir: str = "~/Desktop/dev"
    github_user: str = ""
    default_repo_visibility: str = "private"   # "private" | "public"
    init_timeout: int = 1800                   # 전체 타임아웃 30분
    init_exec_timeout: int = 600               # Sonnet 실행 단계 10분
    init_ci_watch_timeout: int = 300           # CI 대기 최대 5분
    init_ci_fix_retries: int = 2               # CI 실패 시 자동 수정 최대 횟수
    init_issue_planning_timeout: int = 900     # Opus 이슈 기획 15분

    # Plan & Discuss Commands
    plan_timeout: int = 900               # Opus 이슈 기획 (15분)
    discuss_timeout: int = 600            # Opus 기술 상담 (10분)
    discuss_issue_timeout: int = 300      # 논의 → 이슈 변환 (5분)


def save_projects(projects: dict[str, dict]) -> None:
    """Write projects dict to projects.json, contracting home dir to ~ for portability."""
    home = str(Path.home())
    out: dict[str, dict] = {}
    for name, info in projects.items():
        entry = dict(info)
        if "path" in entry:
            p = str(entry["path"])
            if p.startswith(home):
                p = "~" + p[len(home):]
            entry["path"] = p
        out[name] = entry
    with open(_PROJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Saved %d project(s) to %s", len(out), _PROJECTS_FILE)


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
