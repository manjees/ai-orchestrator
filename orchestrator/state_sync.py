"""Project state synchronization — persistent architectural decisions."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _summary_path(project_path: str) -> Path:
    return Path(project_path) / ".claude" / "project_summary.json"


def load_project_summary(project_path: str) -> dict:
    """Load project_summary.json. Returns empty dict if missing."""
    path = _summary_path(project_path)
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        logger.warning("Failed to load project_summary.json")
        return {}


def append_project_summary(
    project_path: str,
    issue_num: int,
    issue_title: str,
    decisions: list[str],
    files_changed: list[str],
) -> None:
    """Append an issue's decisions to project_summary.json."""
    path = _summary_path(project_path)
    try:
        data = load_project_summary(project_path)
        if "issues" not in data:
            data["issues"] = []
        data["issues"].append({
            "num": issue_num,
            "title": issue_title,
            "decisions": decisions,
            "files_changed": files_changed,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep last 20 issues to avoid unbounded growth
        data["issues"] = data["issues"][-20:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        logger.warning("Failed to write project_summary.json")


def format_state_context(project_path: str) -> str:
    """Format recent decisions as prompt context (last 5 issues)."""
    data = load_project_summary(project_path)
    issues = data.get("issues", [])[-5:]
    if not issues:
        return ""
    lines = ["## Recent Project Decisions"]
    for entry in issues:
        lines.append(f"\n### #{entry['num']}: {entry['title']}")
        for dec in entry.get("decisions", []):
            lines.append(f"- {dec}")
        files = entry.get("files_changed", [])
        if files:
            lines.append(f"  Files: {', '.join(files)}")
    return "\n".join(lines)
