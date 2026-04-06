"""Pydantic request/response models for POST /api/commands/* endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Response models ───────────────────────────────────────────────────────────


class CommandResponse(BaseModel):
    """Standard response for all command endpoints (async, immediate accept)."""

    command_id: str
    status: str = "accepted"
    message: str = ""


class CommandWarningResponse(CommandResponse):
    """Response when a command is accepted but carries a warning (e.g. dangerous shell)."""

    warning: str = ""


# ── Request models ────────────────────────────────────────────────────────────


class SolveRequest(BaseModel):
    issues: list[int] = Field(..., min_length=1)
    project: str
    mode: str | None = None       # "express" | "standard" | "full" | None (auto)
    parallel: bool = False


class RetryRequest(BaseModel):
    project: str
    issue_num: int


class CancelRequest(BaseModel):
    pipeline_id: str              # "{project_name}_{issue_num}"


class InitRequest(BaseModel):
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str
    visibility: str = "private"   # "public" | "private"


class PlanRequest(BaseModel):
    project: str


class DiscussRequest(BaseModel):
    project: str
    question: str


class DesignRequest(BaseModel):
    project: str
    figma_url: str
    create_issue: bool = False


class RebaseRequest(BaseModel):
    project: str
    pr_number: int


class ShellRequest(BaseModel):
    command: str = Field(..., min_length=1)
    timeout: int | None = None    # override default timeout
    stream: bool = False
