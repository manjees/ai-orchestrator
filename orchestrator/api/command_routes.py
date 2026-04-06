"""POST /api/commands/* endpoints — launch background pipeline tasks from the API."""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from orchestrator.checkpoint import load_checkpoint
from orchestrator.command_service import (
    generate_command_id,
    is_dangerous_command,
    resolve_project,
)
from orchestrator.config import PIPELINE_MODES

from .command_models import (
    CancelRequest,
    CommandResponse,
    CommandWarningResponse,
    DesignRequest,
    DiscussRequest,
    InitRequest,
    PlanRequest,
    RebaseRequest,
    RetryRequest,
    ShellRequest,
    SolveRequest,
)

logger = logging.getLogger(__name__)

command_router = APIRouter(prefix="/api/commands", tags=["commands"])

# Cancel events for API-initiated pipeline tasks: pipeline_id → asyncio.Event
_api_cancel_events: dict[str, asyncio.Event] = {}

# Lock to make project-name check-then-reserve atomic in /init
_project_creation_lock = asyncio.Lock()


# ── Background task stubs ─────────────────────────────────────────────────────
# These are named functions so tests can patch them cleanly.


async def _bg_solve(
    project_name: str,
    project_path: str,
    project_info: dict,
    issue_nums: list[int],
    solve_mode: str | None,
    parallel: bool,
    settings,
    cancel_events: dict[int, asyncio.Event],
) -> None:
    """Background task: solve one or more issues via the fivebrid pipeline."""
    from orchestrator.core_commands import core_solve

    try:
        await core_solve(
            project_name, project_path, project_info,
            issue_nums, solve_mode, parallel, settings, cancel_events,
        )
    except Exception:
        pass  # core_solve already logs
    finally:
        for num in issue_nums:
            _api_cancel_events.pop(f"{project_name}_{num}", None)


async def _bg_retry(
    project_name: str,
    project_path: str,
    project_info: dict,
    issue_num: int,
    settings,
    cancel_event: asyncio.Event,
) -> None:
    """Background task: retry a pipeline from a saved checkpoint."""
    from orchestrator.core_commands import core_retry

    try:
        await core_retry(
            project_name, project_path, project_info,
            issue_num, settings, cancel_event,
        )
    except Exception:
        pass  # core_retry already logs
    finally:
        _api_cancel_events.pop(f"{project_name}_{issue_num}", None)


async def _bg_init(
    project_name: str,
    description: str,
    project_path: str,
    visibility: str,
    settings,
    cancel_event: asyncio.Event,
) -> None:
    """Background task: bootstrap a new project via the init pipeline."""
    from orchestrator.pipeline import InitContext, run_init_pipeline

    async def _noop_progress(msg: str) -> None:
        logger.debug("[api/init] %s", msg)

    ctx = InitContext(
        project_name=project_name,
        description=description,
        project_path=project_path,
        github_user=settings.github_user,
        repo_visibility=visibility,
    )
    try:
        await asyncio.wait_for(
            run_init_pipeline(ctx, settings, cancel_event, _noop_progress),
            timeout=settings.init_timeout,
        )
    except Exception:
        logger.exception("API init error for %s", project_name)
    finally:
        _api_cancel_events.pop(f"init_{project_name}", None)


async def _bg_plan(
    project_name: str,
    project_path: str,
    settings,
    cancel_event: asyncio.Event,
) -> None:
    """Background task: run step_plan_issues for an existing project."""
    import json

    from orchestrator.ai.ollama_provider import OllamaProvider
    from orchestrator.pipeline import step_plan_issues

    async def _noop_progress(msg: str) -> None:
        logger.debug("[api/plan] %s", msg)

    # Fetch existing issues
    try:
        proc = await asyncio.create_subprocess_exec(
            "gh", "issue", "list",
            "-R", f"{settings.github_user}/{project_name}",
            "--state", "all", "--limit", "50",
            "--json", "number,title,labels,state",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        existing_issues_text = stdout.decode(errors="replace") if stdout else "[]"
    except Exception:
        existing_issues_text = "[]"

    # Read CLAUDE.md
    claude_md = ""
    for candidate in [
        os.path.join(project_path, "CLAUDE.md"),
        os.path.join(project_path, ".claude", "CLAUDE.md"),
    ]:
        if os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8") as f:
                    claude_md = f.read()
                break
            except Exception:
                pass

    if cancel_event.is_set():
        return

    ollama = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.qwen_model,
    )
    try:
        await asyncio.wait_for(
            step_plan_issues(
                project_name, project_path,
                settings.github_user, existing_issues_text,
                claude_md, settings, _noop_progress, ollama,
            ),
            timeout=settings.plan_timeout,
        )
    except Exception:
        logger.exception("API plan error for %s", project_name)
    finally:
        _api_cancel_events.pop(f"plan_{project_name}", None)
        await ollama.close()


async def _bg_discuss(
    project_name: str,
    project_path: str,
    question: str,
    settings,
    cancel_event: asyncio.Event,
) -> None:
    """Background task: run step_discuss_consult for technical consultation."""
    import os

    from orchestrator.ai.ollama_provider import OllamaProvider
    from orchestrator.pipeline import step_discuss_consult

    async def _noop_progress(msg: str) -> None:
        logger.debug("[api/discuss] %s", msg)

    # Read CLAUDE.md
    claude_md = ""
    for candidate in [
        os.path.join(project_path, "CLAUDE.md"),
        os.path.join(project_path, ".claude", "CLAUDE.md"),
    ]:
        if os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8") as f:
                    claude_md = f.read()
                break
            except Exception:
                pass

    if cancel_event.is_set():
        return

    ollama = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.qwen_model,
    )
    try:
        await asyncio.wait_for(
            step_discuss_consult(
                project_name, claude_md,
                file_tree="", build_context="",
                question=question, settings=settings,
                progress_cb=_noop_progress, ollama=ollama,
            ),
            timeout=settings.discuss_timeout,
        )
    except Exception:
        logger.exception("API discuss error for %s", project_name)
    finally:
        _api_cancel_events.pop(f"discuss_{project_name}", None)
        await ollama.close()


async def _bg_design(
    project_name: str,
    project_path: str,
    figma_url: str,
    create_issue: bool,
    settings,
    cancel_event: asyncio.Event,
) -> None:
    """Background task: extract Figma frame and save design spec."""
    from orchestrator.figma import extract_figma_frame, parse_figma_url

    try:
        frame_key = parse_figma_url(figma_url)
        if not frame_key:
            logger.error("API design: invalid Figma URL %s", figma_url)
            return
        await asyncio.wait_for(
            extract_figma_frame(
                frame_key, settings.figma_access_token,
                project_path, create_issue=create_issue,
            ),
            timeout=settings.design_timeout,
        )
    except Exception:
        logger.exception("API design error for %s", project_name)
    finally:
        _api_cancel_events.pop(f"design_{project_name}", None)


async def _bg_rebase(
    project_name: str,
    project_path: str,
    pr_number: int,
    settings,
    cancel_event: asyncio.Event,
) -> None:
    """Background task: rebase a PR branch onto main."""
    import json as _json

    try:
        # Fetch PR metadata: branch name and whether it's from a fork.
        proc = await asyncio.create_subprocess_exec(
            "gh", "pr", "view", str(pr_number),
            "--json", "headRefName,isCrossRepository,headRepository",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode != 0:
            logger.error(
                "API rebase: gh pr view failed for #%d (exit=%d): %s",
                pr_number, proc.returncode,
                stdout.decode(errors="replace")[:500] if stdout else "",
            )
            return

        try:
            pr_info = _json.loads(stdout.decode(errors="replace"))
        except _json.JSONDecodeError as exc:
            logger.error("API rebase: failed to parse gh pr view output for #%d: %s", pr_number, exc)
            return

        branch_name = pr_info.get("headRefName")
        if not branch_name:
            logger.error("API rebase: could not determine head branch for #%d", pr_number)
            return

        is_fork = pr_info.get("isCrossRepository", False)
        if is_fork:
            head_repo = (pr_info.get("headRepository") or {}).get("nameWithOwner", "<unknown>")
            logger.error(
                "API rebase: PR #%d is from a fork (%s). "
                "Rebasing fork branches is not supported — push access is required.",
                pr_number, head_repo,
            )
            return

        if cancel_event.is_set():
            return

        # fetch + checkout + rebase onto origin/main + force-push
        pre_rebase_steps = [
            ["git", "fetch", "origin", "main", branch_name],
            ["git", "checkout", branch_name],
        ]
        post_rebase_steps = [
            ["git", "push", "--force-with-lease", "origin", branch_name],
        ]

        async def _run_git(git_cmd: list[str], timeout: int = 60) -> bool:
            p = await asyncio.create_subprocess_exec(
                *git_cmd,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
            if p.returncode != 0:
                logger.error(
                    "API rebase git command failed (exit=%d): %s\n%s",
                    p.returncode, " ".join(git_cmd),
                    out.decode(errors="replace")[:500] if out else "",
                )
                return False
            return True

        for git_cmd in pre_rebase_steps:
            if cancel_event.is_set():
                logger.info("API rebase: cancelled before %s", git_cmd[1])
                return
            if not await _run_git(git_cmd):
                return

        # Run rebase with timeout; abort if it hangs (conflict) or fails.
        if cancel_event.is_set():
            logger.info("API rebase: cancelled before rebase")
            return
        rebase_ok = False
        try:
            rebase_ok = await _run_git(["git", "rebase", "origin/main"], timeout=120)
        except asyncio.TimeoutError:
            logger.error("API rebase: git rebase timed out for %s#%d — aborting", project_name, pr_number)
        if not rebase_ok:
            # Clean up any in-progress rebase state.
            try:
                abort_proc = await asyncio.create_subprocess_exec(
                    "git", "rebase", "--abort",
                    cwd=project_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await asyncio.wait_for(abort_proc.communicate(), timeout=15)
            except Exception:
                logger.warning("API rebase: git rebase --abort failed for %s#%d", project_name, pr_number)
            return

        for git_cmd in post_rebase_steps:
            if cancel_event.is_set():
                logger.info("API rebase: cancelled before %s", git_cmd[1])
                return
            if not await _run_git(git_cmd):
                return

        logger.info("API rebase: %s#%d (%s) completed", project_name, pr_number, branch_name)
    except Exception:
        logger.exception("API rebase error for %s#%d", project_name, pr_number)
    finally:
        _api_cancel_events.pop(f"rebase_{project_name}_{pr_number}", None)


async def _bg_shell(
    command: str,
    timeout: int,
    settings,
) -> None:
    """Background task: execute a shell command."""
    from orchestrator.core_commands import core_shell

    try:
        result = await core_shell(command, timeout)
        if result.timed_out:
            logger.warning("API shell command timed out after %ds: %s", timeout, command[:80])
        else:
            logger.info(
                "API shell command completed (exit=%d): %s",
                result.exit_code,
                command[:80],
            )
            logger.debug("API shell output: %s", result.output[:500])
    except Exception:
        logger.exception("API shell error for command: %s", command[:80])


# ── Endpoints ─────────────────────────────────────────────────────────────────


@command_router.post("/solve", response_model=CommandResponse, status_code=202)
async def solve(req: SolveRequest, request: Request):
    """Launch the solve pipeline for one or more issues."""
    projects = getattr(request.app.state, "projects", {})
    settings = request.app.state.settings

    project_name, err = resolve_project(req.project, projects)
    if project_name is None:
        raise HTTPException(status_code=404, detail=err)

    if req.mode and req.mode not in PIPELINE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{req.mode}'. Valid: {', '.join(PIPELINE_MODES)}",
        )

    command_id = generate_command_id()
    project_path = projects[project_name]["path"]
    project_info = projects[project_name]

    cancel_events: dict[int, asyncio.Event] = {}
    for num in req.issues:
        pid = f"{project_name}_{num}"
        event = asyncio.Event()
        cancel_events[num] = event
        _api_cancel_events[pid] = event

    asyncio.create_task(
        _bg_solve(
            project_name, project_path, project_info,
            req.issues, req.mode, req.parallel,
            settings, cancel_events,
        )
    )

    nums_str = ", ".join(f"#{n}" for n in req.issues)
    mode_hint = f" [{req.mode}]" if req.mode else ""
    return CommandResponse(
        command_id=command_id,
        status="accepted",
        message=f"Solving {nums_str} for {project_name}{mode_hint}",
    )


@command_router.post("/retry", response_model=CommandResponse, status_code=202)
async def retry(req: RetryRequest, request: Request):
    """Retry a failed pipeline from its last checkpoint."""
    projects = getattr(request.app.state, "projects", {})
    settings = request.app.state.settings

    project_name, err = resolve_project(req.project, projects)
    if project_name is None:
        raise HTTPException(status_code=404, detail=err)

    cp_data = load_checkpoint(project_name, req.issue_num)
    if not cp_data or "ctx" not in cp_data:
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint found for {project_name}#{req.issue_num}",
        )

    command_id = generate_command_id()
    project_path = projects[project_name]["path"]
    project_info = projects[project_name]
    resume_from_step = cp_data.get("failed_step_index", -1)

    pid = f"{project_name}_{req.issue_num}"
    cancel_event = asyncio.Event()
    _api_cancel_events[pid] = cancel_event

    asyncio.create_task(
        _bg_retry(
            project_name, project_path, project_info,
            req.issue_num, settings, cancel_event,
        )
    )

    return CommandResponse(
        command_id=command_id,
        status="accepted",
        message=f"Retrying {project_name}#{req.issue_num} from step {resume_from_step}",
    )


@command_router.post("/cancel", response_model=CommandResponse, status_code=202)
async def cancel(req: CancelRequest, request: Request):
    """Cancel a running pipeline by pipeline_id ({project_name}_{issue_num})."""
    from . import registry

    pid = req.pipeline_id
    ctx_data = registry.get(pid)
    if ctx_data is None:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pid}")

    cancel_event = _api_cancel_events.get(pid)
    if cancel_event:
        cancel_event.set()
        logger.info("API cancel: signalled %s", pid)

    return CommandResponse(
        command_id=generate_command_id(),
        status="accepted",
        message=f"Cancel requested for {pid}",
    )


@command_router.post("/init", response_model=CommandResponse, status_code=202)
async def init(req: InitRequest, request: Request):
    """Bootstrap a new project via the init pipeline."""
    projects = getattr(request.app.state, "projects", {})
    settings = request.app.state.settings

    if not settings.github_user:
        raise HTTPException(status_code=400, detail="GITHUB_USER is not configured")

    async with _project_creation_lock:
        if req.name in projects:
            raise HTTPException(
                status_code=409,
                detail=f"Project '{req.name}' already exists",
            )
        # Reserve the name immediately so concurrent requests see a 409.
        projects[req.name] = {"name": req.name, "status": "initializing"}

    command_id = generate_command_id()
    base_dir = os.path.expanduser(settings.projects_base_dir)
    project_path = os.path.join(base_dir, req.name)
    visibility = req.visibility if req.visibility in ("public", "private") else settings.default_repo_visibility

    cancel_event = asyncio.Event()
    _api_cancel_events[f"init_{req.name}"] = cancel_event

    asyncio.create_task(
        _bg_init(
            req.name, req.description, project_path,
            visibility, settings, cancel_event,
        )
    )

    return CommandResponse(
        command_id=command_id,
        status="accepted",
        message=f"Bootstrapping project '{req.name}' ({visibility})",
    )


@command_router.post("/plan", response_model=CommandResponse, status_code=202)
async def plan(req: PlanRequest, request: Request):
    """Plan next-stage GitHub issues for an existing project."""
    projects = getattr(request.app.state, "projects", {})
    settings = request.app.state.settings

    project_name, err = resolve_project(req.project, projects)
    if project_name is None:
        raise HTTPException(status_code=404, detail=err)

    command_id = generate_command_id()
    project_path = projects[project_name]["path"]

    cancel_event = asyncio.Event()
    _api_cancel_events[f"plan_{project_name}"] = cancel_event

    asyncio.create_task(
        _bg_plan(project_name, project_path, settings, cancel_event)
    )

    return CommandResponse(
        command_id=command_id,
        status="accepted",
        message=f"Planning issues for {project_name}",
    )


@command_router.post("/discuss", response_model=CommandResponse, status_code=202)
async def discuss(req: DiscussRequest, request: Request):
    """Technical consultation with the AI for a project."""
    projects = getattr(request.app.state, "projects", {})
    settings = request.app.state.settings

    project_name, err = resolve_project(req.project, projects)
    if project_name is None:
        raise HTTPException(status_code=404, detail=err)

    command_id = generate_command_id()
    project_path = projects[project_name]["path"]

    cancel_event = asyncio.Event()
    _api_cancel_events[f"discuss_{project_name}"] = cancel_event

    asyncio.create_task(
        _bg_discuss(project_name, project_path, req.question, settings, cancel_event)
    )

    return CommandResponse(
        command_id=command_id,
        status="accepted",
        message=f"Consulting about {project_name}: {req.question[:80]}",
    )


@command_router.post("/design", response_model=CommandResponse, status_code=202)
async def design(req: DesignRequest, request: Request):
    """Extract a Figma frame and save as a UI design spec."""
    projects = getattr(request.app.state, "projects", {})
    settings = request.app.state.settings

    project_name, err = resolve_project(req.project, projects)
    if project_name is None:
        raise HTTPException(status_code=404, detail=err)

    if not settings.figma_access_token:
        raise HTTPException(status_code=400, detail="FIGMA_ACCESS_TOKEN is not configured")

    command_id = generate_command_id()
    project_path = projects[project_name]["path"]

    cancel_event = asyncio.Event()
    _api_cancel_events[f"design_{project_name}"] = cancel_event

    asyncio.create_task(
        _bg_design(
            project_name, project_path,
            req.figma_url, req.create_issue,
            settings, cancel_event,
        )
    )

    return CommandResponse(
        command_id=command_id,
        status="accepted",
        message=f"Extracting Figma frame for {project_name}",
    )


@command_router.post("/rebase", response_model=CommandResponse, status_code=202)
async def rebase(req: RebaseRequest, request: Request):
    """Rebase a PR branch onto main."""
    projects = getattr(request.app.state, "projects", {})
    settings = request.app.state.settings

    project_name, err = resolve_project(req.project, projects)
    if project_name is None:
        raise HTTPException(status_code=404, detail=err)

    command_id = generate_command_id()
    project_path = projects[project_name]["path"]

    cancel_event = asyncio.Event()
    _api_cancel_events[f"rebase_{project_name}_{req.pr_number}"] = cancel_event

    asyncio.create_task(
        _bg_rebase(project_name, project_path, req.pr_number, settings, cancel_event)
    )

    return CommandResponse(
        command_id=command_id,
        status="accepted",
        message=f"Rebasing {project_name} PR #{req.pr_number} onto main",
    )


@command_router.post("/shell", response_model=CommandWarningResponse, status_code=202)
async def shell(req: ShellRequest, request: Request):
    """Execute a shell command in the background."""
    settings = request.app.state.settings
    command_id = generate_command_id()

    warning = is_dangerous_command(req.command)
    if warning:
        return CommandWarningResponse(
            command_id=command_id,
            status="warning",
            message="Command flagged as potentially dangerous",
            warning=warning,
        )

    timeout = req.timeout if req.timeout is not None else settings.cmd_long_timeout

    asyncio.create_task(_bg_shell(req.command, timeout, settings))

    return CommandResponse(
        command_id=command_id,
        status="accepted",
        message=f"Shell command started: {req.command[:80]}",
    )
