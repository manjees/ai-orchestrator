"""API route definitions."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from orchestrator.ai.ollama_provider import OllamaModel, OllamaProvider
from orchestrator.checkpoint import list_checkpoints
from orchestrator.security import mask_secrets
from orchestrator.state_sync import load_project_summary
from orchestrator.system_monitor import get_system_status
from orchestrator.tmux_manager import list_sessions

from .models import (
    CheckpointSummary,
    GithubIssue,
    OllamaModelInfo,
    PipelineDetail,
    PipelineStepResponse,
    PipelineSummary,
    ProjectDetail,
    ProjectSummary,
    StatusResponse,
    TmuxSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _masked_response(data) -> JSONResponse:
    """Serialize model, mask secrets, return JSON response."""
    if isinstance(data, list):
        raw = json.dumps([item.model_dump() for item in data])
    else:
        raw = data.model_dump_json()
    masked = mask_secrets(raw)
    return JSONResponse(content=json.loads(masked))


async def _fetch_github_issues(project_path: str) -> list[dict]:
    """Fetch open GitHub issues via `gh` CLI. Returns [] on any failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "gh", "issue", "list", "--state", "open",
            "--json", "number,title,labels,url",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode == 0:
            return json.loads(stdout.decode())
    except Exception:
        pass
    return []


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/health")
async def health():
    return {"status": "ok"}


async def _get_ollama_models(base_url: str) -> list[OllamaModel]:
    provider = OllamaProvider(base_url=base_url, model="")
    try:
        return await provider.get_loaded_models()
    finally:
        await provider.close()


async def _safe_list_sessions() -> list:
    """Fetch tmux sessions, returning [] on failure."""
    try:
        return await list_sessions()
    except Exception:
        logger.warning("Failed to list tmux sessions", exc_info=True)
        return []


async def _safe_get_ollama_models(base_url: str) -> list[OllamaModel]:
    """Fetch ollama models, returning [] on failure."""
    try:
        return await _get_ollama_models(base_url)
    except Exception:
        logger.warning("Failed to get ollama models", exc_info=True)
        return []


@router.get("/status")
async def get_status(request: Request):
    settings = request.app.state.settings

    sys_status = await get_system_status()

    sessions, models = await asyncio.gather(
        _safe_list_sessions(),
        _safe_get_ollama_models(settings.ollama_base_url),
    )

    tmux_resp = [
        TmuxSessionResponse(name=s.name, windows=s.windows, created=str(s.created))
        for s in sessions
    ]
    resp = StatusResponse(
        ram_total_gb=sys_status.ram_total_gb,
        ram_used_gb=sys_status.ram_used_gb,
        ram_percent=sys_status.ram_percent,
        cpu_percent=sys_status.cpu_percent,
        thermal_pressure=sys_status.thermal_pressure,
        disk_total_gb=sys_status.disk_total_gb,
        disk_used_gb=sys_status.disk_used_gb,
        disk_percent=sys_status.disk_percent,
        ollama_models=[OllamaModelInfo(name=m.name, size_gb=m.size_gb) for m in models],
        tmux_sessions=tmux_resp,
    )
    return _masked_response(resp)


@router.get("/projects")
async def get_projects(request: Request):
    projects = getattr(request.app.state, "projects", {})
    result = [
        ProjectSummary(name=name, path=p["path"])
        for name, p in projects.items()
    ]
    return _masked_response(result)


@router.get("/projects/{name}")
async def get_project(name: str, request: Request):
    projects = getattr(request.app.state, "projects", {})
    if name not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    project_path = projects[name]["path"]
    summary = load_project_summary(project_path)
    return _masked_response(ProjectDetail(name=name, path=project_path, summary=summary))


@router.get("/projects/{name}/issues")
async def get_project_issues(name: str, request: Request):
    projects = getattr(request.app.state, "projects", {})
    if name not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    project_path = projects[name]["path"]
    issues_raw = await _fetch_github_issues(project_path)
    result = [
        GithubIssue(
            number=i["number"],
            title=i["title"],
            url=i["url"],
            labels=[lbl["name"] if isinstance(lbl, dict) else lbl for lbl in i.get("labels", [])],
        )
        for i in issues_raw
    ]
    return _masked_response(result)


@router.get("/pipelines")
async def get_pipelines(request: Request):
    pipelines = getattr(request.app.state, "pipelines", {})
    result = []
    for p_id, ctx in pipelines.items():
        steps = [
            PipelineStepResponse(
                name=s.name, status=s.status, detail=s.detail, elapsed_sec=s.elapsed_sec,
            )
            for s in ctx.steps
        ]
        result.append(PipelineSummary(
            id=p_id, project_name=ctx.project_name, issue_num=ctx.issue_num,
            mode=ctx.mode, steps=steps,
        ))
    return _masked_response(result)


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str, request: Request):
    pipelines = getattr(request.app.state, "pipelines", {})
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    ctx = pipelines[pipeline_id]
    steps = [
        PipelineStepResponse(
            name=s.name, status=s.status, detail=s.detail, elapsed_sec=s.elapsed_sec,
        )
        for s in ctx.steps
    ]
    return _masked_response(PipelineDetail(
        id=pipeline_id, project_name=ctx.project_name, issue_num=ctx.issue_num,
        branch_name=ctx.branch_name, mode=ctx.mode, issue_title=ctx.issue_title,
        issue_body=ctx.issue_body, design_doc=ctx.design_doc, git_diff=ctx.git_diff,
        ci_check_log=ctx.ci_check_log, review_report=ctx.review_report,
        ai_audit_result=ctx.ai_audit_result, ai_audit_passed=ctx.ai_audit_passed,
        steps=steps,
    ))


@router.get("/checkpoints")
async def get_checkpoints():
    raw = list_checkpoints()
    result = [CheckpointSummary(**c) for c in raw]
    return _masked_response(result)
