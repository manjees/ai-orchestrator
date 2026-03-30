"""API route definitions."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from orchestrator.ai.ollama_provider import OllamaModel, OllamaProvider
from orchestrator.api import registry
from orchestrator.checkpoint import list_checkpoints, load_checkpoint
from orchestrator.security import mask_secrets
from orchestrator.state_sync import load_project_summary
from orchestrator.system_monitor import get_system_status
from orchestrator.tmux_manager import list_sessions

from .models import (
    CheckpointSummary,
    GithubIssue,
    OllamaModelInfo,
    PipelineDetail,
    PipelineListResponse,
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


def _derive_status(steps: list[dict]) -> str:
    """Derive pipeline status from step statuses."""
    statuses = [s.get("status", "pending") for s in steps]
    if "running" in statuses:
        return "running"
    if "failed" in statuses:
        return "failed"
    if all(s in ("passed", "skipped") for s in statuses) and statuses:
        return "completed"
    return "pending"


def _mask_dict_strings(d: dict) -> dict:
    """Recursively apply mask_secrets() to all string values."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = mask_secrets(v)
        elif isinstance(v, dict):
            result[k] = _mask_dict_strings(v)
        elif isinstance(v, list):
            result[k] = [
                _mask_dict_strings(item) if isinstance(item, dict)
                else mask_secrets(item) if isinstance(item, str)
                else item
                for item in v
            ]
        else:
            result[k] = v
    return result


def _build_summary(ctx_data: dict, pid: str) -> PipelineSummary:
    """Build a PipelineSummary from a ctx dict. Status derived from steps."""
    steps_raw = ctx_data.get("steps", [])
    masked = _mask_dict_strings(ctx_data)
    return PipelineSummary(
        pipeline_id=pid,
        project_name=masked.get("project_name", ""),
        issue_num=ctx_data.get("issue_num", 0),
        status=_derive_status(steps_raw),
        mode=masked.get("mode", "standard"),
        issue_title=masked.get("issue_title", ""),
        branch_name=masked.get("branch_name", ""),
        steps=[PipelineStepResponse(**s) for s in masked.get("steps", [])],
    )


def _build_detail(ctx_data: dict, pid: str) -> PipelineDetail:
    """Build full PipelineDetail from a ctx dict."""
    steps_raw = ctx_data.get("steps", [])
    masked = _mask_dict_strings(ctx_data)
    return PipelineDetail(
        pipeline_id=pid,
        project_name=masked.get("project_name", ""),
        issue_num=ctx_data.get("issue_num", 0),
        status=_derive_status(steps_raw),
        mode=masked.get("mode", "standard"),
        issue_title=masked.get("issue_title", ""),
        branch_name=masked.get("branch_name", ""),
        steps=[PipelineStepResponse(**s) for s in masked.get("steps", [])],
        design_doc=masked.get("design_doc", ""),
        git_diff=masked.get("git_diff", ""),
        review_report=masked.get("review_report", ""),
        audit_result=masked.get("audit_result", ""),
        ai_audit_result=masked.get("ai_audit_result", ""),
        self_review_report=masked.get("self_review_report", ""),
        gemini_cross_review=masked.get("gemini_cross_review", ""),
        gemini_design_critique=masked.get("gemini_design_critique", ""),
        research_log=masked.get("research_log", ""),
        ci_check_log=masked.get("ci_check_log", ""),
        triage_reason=masked.get("triage_reason", ""),
        retry_count=ctx_data.get("retry_count", 0),
    )


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


@router.get("/pipelines", response_model=PipelineListResponse)
async def list_pipelines():
    """Return active + recently completed pipelines."""
    pipelines: list[PipelineSummary] = []

    # Live pipelines take priority so the UI always shows latest state
    for pid, ctx_data in registry.list_all().items():
        pipelines.append(_build_summary(ctx_data, pid))

    # Backfill from checkpoints so completed/failed runs remain visible
    for cp_meta in list_checkpoints():
        pid = f"{cp_meta['project_name']}_{cp_meta['issue_num']}"
        if any(p.pipeline_id == pid for p in pipelines):
            continue
        cp_data = load_checkpoint(cp_meta["project_name"], cp_meta["issue_num"])
        if cp_data and "ctx" in cp_data:
            pipelines.append(_build_summary(cp_data["ctx"], pid))

    return PipelineListResponse(pipelines=pipelines)


@router.get("/pipelines/history")
async def get_pipeline_history(limit: int = 20):
    """Return recent pipeline runs from JSONL event log."""
    from orchestrator.event_logger import get_event_logger
    history = get_event_logger().read_pipeline_history(limit=min(limit, 100))
    return {"history": history}


@router.get("/pipelines/{pipeline_id}", response_model=PipelineDetail)
async def get_pipeline(pipeline_id: str):
    """Return pipeline detail by ID ({project_name}_{issue_num})."""
    # Prefer live registry data — it reflects real-time step progress
    ctx_data = registry.get(pipeline_id)
    if ctx_data:
        return _build_detail(ctx_data, pipeline_id)

    # Fall back to checkpoint for completed/failed pipelines no longer in memory
    parts = pipeline_id.rsplit("_", 1)
    if len(parts) == 2:
        project_name, issue_str = parts
        try:
            issue_num = int(issue_str)
        except ValueError:
            pass
        else:
            cp_data = load_checkpoint(project_name, issue_num)
            if cp_data and "ctx" in cp_data:
                return _build_detail(cp_data["ctx"], pipeline_id)

    return JSONResponse(status_code=404, content={"detail": "Pipeline not found"})


@router.get("/checkpoints", response_model=list[CheckpointSummary])
async def get_checkpoints():
    raw = list_checkpoints()
    result = [CheckpointSummary(**c) for c in raw]
    return _masked_response(result)


# ── WebSocket ───────────────────────────────────────────────────────────────


@router.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """Real-time event stream for dashboard clients."""
    from .events import get_event_bus

    settings = ws.app.state.settings
    token = ws.query_params.get("token")

    if settings.dashboard_api_key:
        if not token or not hmac.compare_digest(
            token.encode(), settings.dashboard_api_key.encode()
        ):
            await ws.close(code=1008, reason="Invalid or missing token")
            return

    await ws.accept()
    bus = get_event_bus()
    await bus.register(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await bus.unregister(ws)
