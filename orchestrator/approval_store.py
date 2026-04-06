"""Shared in-process registry for pending approvals (Strategy + Supreme Court)."""

import asyncio
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ApprovalType(str, Enum):
    STRATEGY = "strategy"
    SUPREME_COURT = "supreme_court"


@dataclass
class PendingApproval:
    approval_id: str              # "{chat_id}:{issue_num}" or UUID
    approval_type: ApprovalType
    context: dict[str, Any]       # metadata (issue_num, project, ruling, etc.)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    decision: str | None = None
    comment: str = ""
    resolved: bool = False


_lock = threading.Lock()
_pending: dict[str, PendingApproval] = {}


def register(approval: PendingApproval) -> None:
    with _lock:
        _pending[approval.approval_id] = approval


def get(approval_id: str) -> PendingApproval | None:
    with _lock:
        return _pending.get(approval_id)


def resolve(approval_id: str, decision: str, comment: str = "") -> PendingApproval | None:
    """Set decision and signal event. Returns None if not found or already resolved."""
    with _lock:
        approval = _pending.get(approval_id)
        if not approval or approval.resolved:
            return None
        approval.decision = decision
        approval.comment = comment
        approval.resolved = True
    # Signal outside lock to avoid deadlocks
    approval.event.set()
    return approval


def remove(approval_id: str) -> None:
    with _lock:
        _pending.pop(approval_id, None)


def list_pending() -> list[PendingApproval]:
    with _lock:
        return [a for a in _pending.values() if not a.resolved]


def clear() -> None:
    """For testing only."""
    with _lock:
        _pending.clear()
