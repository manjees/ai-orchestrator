"""Pydantic models for approval endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field

from orchestrator.approval_store import ApprovalStatus, ApprovalType


class ApprovalInfo(BaseModel):
    """Read-only representation of an approval."""
    approval_id: str
    type: ApprovalType
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision: str | None = None
    decision_options: list[str]
    context: dict = {}                   # triage_result, issue_num, etc.


class ApprovalRespondRequest(BaseModel):
    """Body for POST /api/approvals/{id}/respond"""
    decision: str = Field(..., min_length=1)


class ApprovalRespondResponse(BaseModel):
    approval_id: str
    status: str       # "decided"
    decision: str


class ApprovalListResponse(BaseModel):
    approvals: list[ApprovalInfo]
