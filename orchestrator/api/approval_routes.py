"""POST /api/approvals/* endpoints — respond to pipeline approval requests."""

from fastapi import APIRouter, HTTPException

from orchestrator import approval_store
from orchestrator.api.command_models import (
    ApprovalRespondRequest,
    ApprovalRespondResponse,
    PendingApprovalInfo,
)
from orchestrator.api.events import EventType, get_event_bus

approval_router = APIRouter(prefix="/api/approvals", tags=["approvals"])

_VALID_DECISIONS = {
    "strategy": {"approve", "nosplit", "cancel"},
    "supreme_court": {"uphold", "overturn", "accept"},
}


@approval_router.post("/{approval_id}/respond", response_model=ApprovalRespondResponse)
async def respond_to_approval(approval_id: str, req: ApprovalRespondRequest):
    # 1. Look up
    pending = approval_store.get(approval_id)
    if pending is None:
        raise HTTPException(404, detail=f"Approval '{approval_id}' not found")

    # 2. Validate decision for approval type
    valid = _VALID_DECISIONS.get(pending.approval_type.value, set())
    if req.decision not in valid:
        raise HTTPException(
            422,
            detail=f"Invalid decision '{req.decision}' for {pending.approval_type.value}. "
                   f"Valid: {sorted(valid)}",
        )

    # 3. Resolve (first-responder-wins)
    resolved = approval_store.resolve(approval_id, req.decision, req.comment)
    if resolved is None:
        raise HTTPException(409, detail="Approval already resolved")

    # 4. Emit WebSocket event
    bus = get_event_bus()
    await bus.emit(EventType.APPROVAL_RESPONDED, {
        "approval_id": approval_id,
        "approval_type": pending.approval_type.value,
        "decision": req.decision,
        "source": "dashboard",
    })

    return ApprovalRespondResponse(
        approval_id=approval_id,
        decision=req.decision,
        message=f"{pending.approval_type.value} approval resolved: {req.decision}",
    )


@approval_router.get("/pending", response_model=list[PendingApprovalInfo])
async def list_pending_approvals():
    pending = approval_store.list_pending()
    return [
        PendingApprovalInfo(
            approval_id=a.approval_id,
            approval_type=a.approval_type.value,
            context=a.context,
        )
        for a in pending
    ]
