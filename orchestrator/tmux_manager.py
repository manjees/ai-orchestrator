"""Tmux subprocess wrapper with automatic session creation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TmuxSession:
    name: str
    windows: int
    created: str


async def _run_tmux(*args: str, timeout: float = 10) -> tuple[int, str, str]:
    """Execute a tmux command and return (returncode, stdout, stderr).

    Uses ``create_subprocess_exec`` to avoid shell injection.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "tmux",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )
    except FileNotFoundError:
        logger.error("tmux binary not found")
        return (1, "", "tmux not found")
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[possibly-undefined]
        return (1, "", "tmux command timed out")


async def ensure_session(session_name: str) -> None:
    """Create the session if it does not already exist."""
    rc, _, _ = await _run_tmux("has-session", "-t", session_name)
    if rc != 0:
        logger.info("Creating tmux session %s", session_name)
        await _run_tmux("new-session", "-d", "-s", session_name)


async def list_sessions() -> list[TmuxSession]:
    """Return a list of all active tmux sessions."""
    rc, stdout, _ = await _run_tmux(
        "list-sessions", "-F", "#{session_name}|#{session_windows}|#{session_created}"
    )
    if rc != 0 or not stdout.strip():
        return []
    sessions: list[TmuxSession] = []
    for line in stdout.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            sessions.append(
                TmuxSession(name=parts[0], windows=int(parts[1]), created=parts[2])
            )
    return sessions


async def capture_pane(session_name: str, lines: int = 20) -> str:
    """Capture the last *lines* lines from pane 0 of the given session."""
    await ensure_session(session_name)
    rc, stdout, stderr = await _run_tmux(
        "capture-pane", "-t", f"{session_name}:0", "-p", "-S", f"-{lines}"
    )
    if rc != 0:
        return f"(capture failed: {stderr.strip()})"
    return stdout


async def send_keys(session_name: str, keys: str) -> bool:
    """Send keystrokes to the session (for future expansion)."""
    await ensure_session(session_name)
    rc, _, stderr = await _run_tmux("send-keys", "-t", f"{session_name}:0", keys, "Enter")
    if rc != 0:
        logger.error("send_keys failed: %s", stderr)
        return False
    return True
