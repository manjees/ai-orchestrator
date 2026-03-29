"""In-process registry for active pipeline contexts."""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any

from orchestrator.pipeline import PipelineContext

_lock = threading.Lock()
_active: dict[str, dict[str, Any]] = {}  # pipeline_id → ctx dict


def pipeline_id(project_name: str, issue_num: int) -> str:
    return f"{project_name}_{issue_num}"


def register(ctx: PipelineContext) -> None:
    pid = pipeline_id(ctx.project_name, ctx.issue_num)
    with _lock:
        _active[pid] = asdict(ctx)


def update(ctx: PipelineContext) -> None:
    """Update the registry entry (e.g. after step completion)."""
    register(ctx)  # same operation — overwrite


def unregister(project_name: str, issue_num: int) -> None:
    pid = pipeline_id(project_name, issue_num)
    with _lock:
        _active.pop(pid, None)


def get(pid: str) -> dict[str, Any] | None:
    with _lock:
        return _active.get(pid)


def list_all() -> dict[str, dict[str, Any]]:
    with _lock:
        return dict(_active)


def clear() -> None:
    """For testing only."""
    with _lock:
        _active.clear()
