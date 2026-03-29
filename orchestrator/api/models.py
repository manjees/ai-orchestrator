"""Pydantic response models for GET endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class TmuxSessionResponse(BaseModel):
    name: str
    windows: int
    created: str


class StatusResponse(BaseModel):
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    cpu_percent: float
    thermal_pressure: str
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    tmux_sessions: list[TmuxSessionResponse]


class ProjectSummary(BaseModel):
    name: str
    path: str


class ProjectDetail(BaseModel):
    name: str
    path: str
    summary: dict


class GithubIssue(BaseModel):
    number: int
    title: str
    labels: list[str]
    url: str


class PipelineStepResponse(BaseModel):
    name: str
    status: str
    detail: str
    elapsed_sec: float


class PipelineSummary(BaseModel):
    id: str
    project_name: str
    issue_num: int
    mode: str
    steps: list[PipelineStepResponse]


class PipelineDetail(BaseModel):
    id: str
    project_name: str
    issue_num: int
    branch_name: str
    mode: str
    issue_title: str
    issue_body: str
    design_doc: str
    git_diff: str
    ci_check_log: str
    review_report: str
    ai_audit_result: str
    ai_audit_passed: bool
    steps: list[PipelineStepResponse]


class CheckpointSummary(BaseModel):
    file: str
    project_name: str
    issue_num: int
    pipeline_mode: str
    failed_step_name: str
