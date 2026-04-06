"""Shared approval state — single source of truth for both Telegram and API."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Approval:
    approval_id: str
    type: str                            # "strategy" | "supreme_court"
    status: str = "pending"              # "pending" | "decided" | "expired"
    decision: str | None = None
    decision_options: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    event: asyncio.Event = field(default_factory=asyncio.Event)


# ── Module-level store ────────────────────────────────────────────────────────
_approvals: dict[str, Approval] = {}


def create_approval(
    approval_id: str,
    approval_type: str,
    decision_options: list[str],
    context: dict | None = None,
) -> Approval:
    """Register a new pending approval. Returns the Approval object."""
    approval = Approval(
        approval_id=approval_id,
        type=approval_type,
        decision_options=decision_options,
        context=context or {},
    )
    _approvals[approval_id] = approval
    return approval


def get_approval(approval_id: str) -> Approval | None:
    return _approvals.get(approval_id)


def respond(approval_id: str, decision: str) -> bool:
    """
    Record a decision. Returns True if applied, False if already decided or missing.
    Raises ValueError if decision not in decision_options.
    """
    approval = _approvals.get(approval_id)
    if approval is None:
        return False
    if approval.status != "pending":
        return False
    if decision not in approval.decision_options:
        raise ValueError(
            f"Invalid decision '{decision}'. Must be one of: {approval.decision_options}"
        )
    approval.decision = decision
    approval.status = "decided"
    approval.event.set()
    return True


async def wait_for_decision(approval_id: str, timeout: float) -> str | None:
    """Block until decided or timeout. Returns decision or None."""
    approval = _approvals.get(approval_id)
    if approval is None:
        return None
    try:
        await asyncio.wait_for(approval.event.wait(), timeout=timeout)
        return approval.decision
    except asyncio.TimeoutError:
        approval.status = "expired"
        return None


def remove_approval(approval_id: str) -> None:
    """Clean up an approval from the store."""
    _approvals.pop(approval_id, None)


def list_pending() -> list[Approval]:
    return [a for a in _approvals.values() if a.status == "pending"]


def clear_all() -> None:
    """For testing only."""
    _approvals.clear()
