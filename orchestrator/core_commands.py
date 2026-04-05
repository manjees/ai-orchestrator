"""Core command logic — pure async functions with no Telegram dependency.

Both Telegram handlers (handlers.py) and API routes (command_routes.py) call
these functions. Telegram-specific concerns (keyboard markup, chat messages,
argument parsing) remain in handlers.py.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from orchestrator.ai.gemini_provider import GeminiCLIProvider
from orchestrator.ai.ollama_provider import OllamaProvider
from orchestrator.checkpoint import load_checkpoint, restore_context
from orchestrator.pipeline import PipelineContext, run_fivebrid_pipeline
from orchestrator.security import mask_secrets
from orchestrator.api import registry

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]


async def _noop_progress(msg: str) -> None:
    logger.debug("[core] %s", msg)


@dataclass
class ShellResult:
    exit_code: int
    output: str
    timed_out: bool = False


async def core_solve(
    project_name: str,
    project_path: str,
    project_info: dict,
    issue_nums: list[int],
    solve_mode: str | None,
    parallel: bool,
    settings,
    cancel_events: dict[int, asyncio.Event],
    progress_cb: ProgressCallback | None = None,
) -> None:
    """Run the fivebrid solve pipeline for one or more issues.

    Callers:
      - handlers.py (_start_solve → _solve_issues)
      - command_routes.py (_bg_solve)
    """
    cb = progress_cb or _noop_progress
    resolved_mode = solve_mode or "standard"

    async def _solve_one(issue_num: int) -> None:
        cancel_event = cancel_events[issue_num]
        branch_name = f"solve/issue-{issue_num}"
        ctx = PipelineContext(
            project_path=project_path,
            project_name=project_name,
            issue_num=issue_num,
            branch_name=branch_name,
            mode=resolved_mode,
        )
        registry.register(ctx)
        ollama = OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.reasoning_model,
        )
        gemini = GeminiCLIProvider()
        try:
            await asyncio.wait_for(
                run_fivebrid_pipeline(
                    ctx, ollama, gemini, settings, cancel_event, cb,
                    project_info=project_info,
                ),
                timeout=settings.solve_timeout,
            )
        except Exception:
            logger.exception("Solve error for %s#%d", project_name, issue_num)
            raise
        finally:
            registry.unregister(project_name, issue_num)
            await ollama.close()

    if len(issue_nums) == 1:
        await _solve_one(issue_nums[0])
    elif parallel:
        await asyncio.gather(*[_solve_one(n) for n in issue_nums], return_exceptions=True)
    else:
        for num in issue_nums:
            try:
                await _solve_one(num)
            except Exception:
                # _solve_one already logged the exception details.
                pass


async def core_retry(
    project_name: str,
    project_path: str,
    project_info: dict,
    issue_num: int,
    settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback | None = None,
) -> None:
    """Retry a pipeline from its last checkpoint."""
    cb = progress_cb or _noop_progress

    cp_data = load_checkpoint(project_name, issue_num)
    if not cp_data or "ctx" not in cp_data:
        logger.warning("Retry: checkpoint missing for %s#%d", project_name, issue_num)
        return

    ctx = restore_context(cp_data["ctx"])
    resume_from_step = cp_data.get("failed_step_index", -1)

    registry.register(ctx)
    ollama = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.reasoning_model,
    )
    gemini = GeminiCLIProvider()
    try:
        await asyncio.wait_for(
            run_fivebrid_pipeline(
                ctx, ollama, gemini, settings, cancel_event, cb,
                project_info=project_info,
                resume_from_step=resume_from_step,
            ),
            timeout=settings.solve_timeout,
        )
    except Exception:
        logger.exception("Retry error for %s#%d", project_name, issue_num)
        raise
    finally:
        registry.unregister(project_name, issue_num)
        await ollama.close()


def core_cancel(
    pipeline_id: str,
    cancel_events: dict[str, asyncio.Event],
) -> bool:
    """Signal cancellation for a running pipeline. Returns True if found."""
    event = cancel_events.get(pipeline_id)
    if event is None:
        return False
    event.set()
    return True


async def core_shell(
    command: str,
    timeout: int,
) -> ShellResult:
    """Execute a shell command and return masked output."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode(errors="replace") if stdout else "(no output)"
        return ShellResult(
            exit_code=proc.returncode or 0,
            output=mask_secrets(output),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return ShellResult(exit_code=-1, output="", timed_out=True)
