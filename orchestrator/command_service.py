"""Shared command validation utilities — used by both Telegram handlers and API routes."""

from __future__ import annotations

import re
import uuid

# ── Dangerous shell command patterns ─────────────────────────────────────────

_DANGEROUS_PATTERNS = re.compile(
    r"rm\s+-[^\s]*r[^\s]*f"                      # rm -rf / rm -fr variants
    r"|rm\s+-[^\s]*f[^\s]*r"
    r"|mkfs"
    r"|dd\s+if="                                  # dd if=... (disk wipe)
    r"|:\(\)\{.*\}.*:"                            # fork bomb
    r"|kill\s+-9"
    r"|killall"
    r"|pkill"
    r"|>\s*/dev/sd"
    r"|chmod\s+-[^\s]*R[^\s]*\s+777"             # chmod -R 777
    r"|chmod\s+-[^\s]*777[^\s]*\s+-R"
    r"|shutdown"
    r"|reboot"
    r"|halt"
    r"|curl\s+.*\|\s*(?:bash|sh)"
    r"|wget\s+.*\|\s*(?:bash|sh)",
    re.IGNORECASE,
)


def generate_command_id() -> str:
    """Return a new random UUID string."""
    return str(uuid.uuid4())


def resolve_project(name: str, projects: dict) -> tuple[str | None, str]:
    """Resolve a project name with prefix matching (no HTML escaping).

    Returns (resolved_name, error_message).
    If resolved_name is None, error_message explains the problem.
    """
    # Exact match wins
    if name in projects:
        return name, ""

    # Prefix match
    matches = [p for p in projects if p.startswith(name)]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, f"Ambiguous project: {', '.join(sorted(matches))}"

    return None, f"Project not found: {name}"


def is_dangerous_command(command: str) -> str | None:
    """Return a warning string if the command matches dangerous patterns, else None."""
    match = _DANGEROUS_PATTERNS.search(command)
    if match:
        return f"Potentially dangerous command detected: '{match.group()}'"
    return None
