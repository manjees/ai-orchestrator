"""REST endpoints for approval management."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from orchestrator import approval_store
from orchestrator.approval_store import ApprovalType

from .approval_models import (
    ApprovalInfo,
    ApprovalListResponse,
    ApprovalRespondRequest,
    ApprovalRespondResponse,
)
from .events import EventType, get_event_bus

logger = logging.getLogger(__name__)
approval_router = APIRouter(prefix="/api/approvals", tags=["approvals"])


@approval_router.get("", response_model=ApprovalListResponse)
async def list_approvals():
    pending = approval_store.list_pending()
    return ApprovalListResponse(
        approvals=[
            ApprovalInfo(
                approval_id=a.approval_id,
                type=a.type,
                status=a.status,
                decision=a.decision,
                decision_options=a.decision_options,
                context=a.context,
            )
            for a in pending
        ]
    )


@approval_router.post("/{approval_id}/respond", response_model=ApprovalRespondResponse)
async def respond_to_approval(approval_id: str, body: ApprovalRespondRequest):
    approval = approval_store.get_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")

    try:
        applied = approval_store.respond(approval_id, body.decision)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not applied:
        raise HTTPException(status_code=409, detail="Approval already decided")

    # For Supreme Court approvals, "accept" defers to the AI ruling — resolve the
    # actual decision so WS clients stay in sync with what the pipeline will act on.
    emit_decision = body.decision
    if approval.type == ApprovalType.SUPREME_COURT and body.decision == "accept":
        emit_decision = approval.context.get("ruling", body.decision).lower()

    try:
        await get_event_bus().emit(EventType.APPROVAL_RESPONDED, {
            "approval_id": approval_id,
            "decision": emit_decision,
            "source": "api",
        })
    except Exception:
        logger.warning("Failed to emit approval.responded event", exc_info=True)

    return ApprovalRespondResponse(
        approval_id=approval_id,
        status="decided",
        decision=body.decision,
    )
