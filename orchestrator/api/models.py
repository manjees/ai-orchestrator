"""Pydantic response models for GET endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class OllamaModelInfo(BaseModel):
    name: str
    size_gb: float


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
    ollama_models: list[OllamaModelInfo]
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
    status: str  # pending | running | passed | failed | skipped
    detail: str = ""
    elapsed_sec: float = 0.0


class PipelineSummary(BaseModel):
    pipeline_id: str  # "{project_name}_{issue_num}"
    project_name: str
    issue_num: int
    status: str  # pending | running | completed | failed
    mode: str = "standard"  # express | standard | full
    issue_title: str = ""
    branch_name: str = ""
    steps: list[PipelineStepResponse] = []


class PipelineDetail(PipelineSummary):
    """Full detail — extends summary with large text fields (masked)."""

    design_doc: str = ""
    git_diff: str = ""
    review_report: str = ""
    audit_result: str = ""
    ai_audit_result: str = ""
    self_review_report: str = ""
    gemini_cross_review: str = ""
    gemini_design_critique: str = ""
    research_log: str = ""
    ci_check_log: str = ""
    triage_reason: str = ""
    retry_count: int = 0


class PipelineListResponse(BaseModel):
    pipelines: list[PipelineSummary]


class CheckpointSummary(BaseModel):
    file: str
    project_name: str
    issue_num: int
    pipeline_mode: str
    failed_step_name: str
