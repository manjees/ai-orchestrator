"""Dual-Check Pipeline: DeepSeek Design → Claude Implement → Claude Review → DeepSeek Audit."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .ai.base import AIResponse, Message, Role
from .ai.anthropic_provider import AnthropicProvider
from .ai.ollama_provider import OllamaProvider
from .config import Settings
from .security import mask_secrets

logger = logging.getLogger(__name__)

# ── Data Structures ──────────────────────────────────────────────────────────

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass
class PipelineStep:
    name: str
    status: str = "pending"  # pending | running | passed | failed | skipped
    detail: str = ""
    elapsed_sec: float = 0.0


@dataclass
class PipelineContext:
    project_path: str
    project_name: str
    issue_num: int
    branch_name: str
    issue_body: str = ""
    design_doc: str = ""
    git_diff: str = ""
    review_report: str = ""
    audit_result: str = ""
    audit_passed: bool = False
    steps: list[PipelineStep] = field(default_factory=lambda: [
        PipelineStep(name="DeepSeek Design"),
        PipelineStep(name="Claude Implement"),
        PipelineStep(name="Claude Review"),
        PipelineStep(name="DeepSeek Audit"),
    ])


# ── Verdict Parsing ──────────────────────────────────────────────────────────

def parse_verdict(text: str) -> bool | None:
    """Extract [VERDICT: PASS] or [VERDICT: FAIL] from response tail."""
    tail = text[-500:]
    match = re.search(r"\[VERDICT:\s*(PASS|FAIL)\]", tail, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "PASS"
    return None


def parse_final(text: str) -> bool | None:
    """Extract [FINAL: APPROVED] or [FINAL: REJECTED] from response tail."""
    tail = text[-500:]
    match = re.search(r"\[FINAL:\s*(APPROVED|REJECTED)\]", tail, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "APPROVED"
    return None


# ── Git Diff Filtering ───────────────────────────────────────────────────────

DIFF_EXCLUDE_PATTERNS = [
    "*.lock", "*.min.js", "*.min.css", "*.map",
    "*.pyc", "__pycache__/*", "*.egg-info/*",
    "dist/*", "build/*", "node_modules/*",
    "*.png", "*.jpg", "*.gif", "*.ico", "*.woff*",
]

_MAX_DIFF_CHARS = 50_000


async def _capture_filtered_diff(project_path: str) -> str:
    """Capture git diff main..HEAD excluding binary/generated files."""
    exclude_args = [f":(exclude){pat}" for pat in DIFF_EXCLUDE_PATTERNS]

    proc = await asyncio.create_subprocess_exec(
        "git", "diff", "main..HEAD", "--", ".", *exclude_args,
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    diff = stdout.decode(errors="replace") if stdout else ""

    if len(diff) > _MAX_DIFF_CHARS:
        # Capture stat summary as fallback context
        stat_proc = await asyncio.create_subprocess_exec(
            "git", "diff", "main..HEAD", "--stat",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stat_out, _ = await asyncio.wait_for(stat_proc.communicate(), timeout=10)
        stat_text = stat_out.decode(errors="replace") if stat_out else ""

        diff = (
            f"[DIFF TRUNCATED — {len(diff)} chars, showing first {_MAX_DIFF_CHARS}]\n"
            f"--- stat summary ---\n{stat_text}\n--- diff (truncated) ---\n"
            + diff[:_MAX_DIFF_CHARS]
        )

    return diff


# ── DeepSeek Progress Helper ─────────────────────────────────────────────────

async def _call_deepseek_with_progress(
    ollama: OllamaProvider,
    messages: list[Message],
    *,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    timeout: int,
    progress_cb: ProgressCallback,
    step_name: str,
) -> AIResponse:
    """Call DeepSeek with periodic 'Thinking...' progress updates."""
    task = asyncio.create_task(
        ollama.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            timeout=timeout,
        )
    )
    elapsed = 0
    while not task.done():
        await asyncio.sleep(10)
        elapsed += 10
        mins, secs = divmod(elapsed, 60)
        await progress_cb(f"{step_name} (Thinking... {mins}m {secs}s)")
    return task.result()


# ── Step Functions ────────────────────────────────────────────────────────────

async def step_fetch_issue(ctx: PipelineContext) -> None:
    """Fetch issue details via gh CLI."""
    proc = await asyncio.create_subprocess_exec(
        "gh", "issue", "view", str(ctx.issue_num),
        "--json", "title,body,labels,comments",
        cwd=ctx.project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    if proc.returncode != 0:
        err = stderr.decode(errors="replace") if stderr else "unknown error"
        raise RuntimeError(f"gh issue view failed: {err}")
    ctx.issue_body = stdout.decode(errors="replace") if stdout else ""


async def step_deepseek_design(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
) -> None:
    """Step 1: DeepSeek designs the implementation approach."""
    step = ctx.steps[0]
    step.status = "running"
    start = time.monotonic()

    system_prompt = (
        "You are a senior software engineer with 7+ years of experience. "
        "Provide the optimal implementation design for the given GitHub issue. "
        "Include: files to modify, detailed changes, edge cases, and testing strategy. "
        "Be specific and actionable."
    )
    user_content = (
        f"GitHub Issue #{ctx.issue_num}:\n"
        f"{ctx.issue_body}\n\n"
        "How should this be implemented? Include:\n"
        "1. Files to modify/create\n"
        "2. Detailed implementation steps\n"
        "3. Edge cases and pitfalls\n"
        "4. Testing approach"
    )

    messages = [Message(role=Role.USER, content=user_content)]

    try:
        response = await _call_deepseek_with_progress(
            ollama, messages,
            max_tokens=settings.deepseek_design_max_tokens,
            temperature=0.7,
            system_prompt=system_prompt,
            timeout=settings.deepseek_design_timeout,
            progress_cb=progress_cb,
            step_name="DeepSeek Design",
        )
        ctx.design_doc = response.content
        step.status = "passed"
        step.detail = f"{response.output_tokens} tokens"
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_claude_implement(
    ctx: PipelineContext,
    settings: Settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback,
) -> None:
    """Step 2: Claude CLI implements based on DeepSeek's design."""
    step = ctx.steps[1]
    step.status = "running"
    start = time.monotonic()

    prompt = (
        f"Read .claude/CLAUDE.md first and follow the defined pipeline. "
        f"Then read GitHub issue #{ctx.issue_num} with `gh issue view {ctx.issue_num}`. "
        f"\n\n--- Design Guide (from DeepSeek) ---\n{ctx.design_doc}\n---\n\n"
        f"Implement the solution following this design guide. "
        f"Complete all steps including testing."
    )

    claude_cmd = (
        f'claude -p --dangerously-skip-permissions '
        f'{_shell_quote(prompt)}'
    )

    proc = await asyncio.create_subprocess_shell(
        claude_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=ctx.project_path,
    )
    assert proc.stdout is not None

    collected = ""

    async def _read_output() -> None:
        nonlocal collected
        async for line in proc.stdout:  # type: ignore[union-attr]
            collected += line.decode(errors="replace")

    read_task = asyncio.create_task(_read_output())

    try:
        while not read_task.done():
            await asyncio.sleep(10)

            if cancel_event.is_set():
                proc.kill()
                await read_task
                step.status = "failed"
                step.detail = "Cancelled"
                step.elapsed_sec = time.monotonic() - start
                raise asyncio.CancelledError("Cancelled by user")

            elapsed = int(time.monotonic() - start)
            if elapsed > settings.solve_timeout:
                proc.kill()
                await read_task
                step.status = "failed"
                step.detail = f"Timed out after {settings.solve_timeout}s"
                step.elapsed_sec = time.monotonic() - start
                raise TimeoutError(step.detail)

            mins, secs = divmod(elapsed, 60)
            await progress_cb(f"Claude Implement [{mins}m {secs}s]")

        try:
            await asyncio.wait_for(read_task, timeout=10)
        except asyncio.TimeoutError:
            pass
        await proc.wait()

        rc = proc.returncode
        if rc != 0:
            step.status = "failed"
            preview = mask_secrets(collected[-500:]) if collected else "(no output)"
            step.detail = f"exit={rc}: {preview[:200]}"
            step.elapsed_sec = time.monotonic() - start
            raise RuntimeError(f"Claude CLI failed with exit={rc}")

        # Snapshot commit so diff is clean
        await _snapshot_commit(ctx.project_path, ctx.issue_num)

        # Capture filtered diff
        ctx.git_diff = await _capture_filtered_diff(ctx.project_path)

        if not ctx.git_diff.strip():
            step.status = "failed"
            step.detail = "No changes produced"
            step.elapsed_sec = time.monotonic() - start
            raise RuntimeError("Claude produced no code changes")

        step.status = "passed"
        step.detail = f"exit=0, diff={len(ctx.git_diff)} chars"
    except (asyncio.CancelledError, TimeoutError, RuntimeError):
        raise
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_claude_review(
    ctx: PipelineContext,
    anthropic: AnthropicProvider | None,
    settings: Settings,
) -> None:
    """Step 3: Claude reviews the implementation via Anthropic API (or CLI fallback)."""
    step = ctx.steps[2]
    step.status = "running"
    start = time.monotonic()

    review_prompt = (
        f"You are a meticulous code reviewer. Review the following implementation "
        f"for GitHub issue #{ctx.issue_num}.\n\n"
        f"--- Issue ---\n{ctx.issue_body}\n\n"
        f"--- Design Guide ---\n{ctx.design_doc}\n\n"
        f"--- Git Diff ---\n{ctx.git_diff}\n\n"
        f"Review criteria:\n"
        f"1. Does the implementation correctly address the issue?\n"
        f"2. Are there bugs, security issues, or edge cases missed?\n"
        f"3. Is the code quality acceptable?\n"
        f"4. Are there any regressions?\n\n"
        f"After your review, you MUST end with exactly one of:\n"
        f"[VERDICT: PASS] — if the implementation is acceptable\n"
        f"[VERDICT: FAIL] — if there are critical issues"
    )

    try:
        if anthropic:
            # Use Anthropic API directly (faster, no filesystem needed)
            messages = [Message(role=Role.USER, content=review_prompt)]
            response = await anthropic.chat(
                messages,
                max_tokens=4096,
                temperature=0.2,
            )
            review_text = response.content
        else:
            # Fallback: Claude CLI
            review_text = await _claude_cli_review(ctx.project_path, review_prompt)

        ctx.review_report = review_text
        verdict = parse_verdict(review_text)

        if verdict is True:
            step.status = "passed"
            step.detail = "VERDICT: PASS"
        elif verdict is False:
            step.status = "failed"
            step.detail = "VERDICT: FAIL"
            raise RuntimeError("Claude review returned FAIL")
        else:
            step.status = "failed"
            step.detail = "No verdict tag found — treated as FAIL"
            raise RuntimeError("Claude review did not include [VERDICT: PASS/FAIL]")

    except RuntimeError:
        raise
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_deepseek_audit(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
) -> None:
    """Step 4: DeepSeek final audit of the implementation."""
    step = ctx.steps[3]
    step.status = "running"
    start = time.monotonic()

    system_prompt = (
        "You are a senior code auditor performing a final quality check. "
        "Examine the diff and review report carefully."
    )
    user_content = (
        f"GitHub Issue #{ctx.issue_num}:\n{ctx.issue_body}\n\n"
        f"--- Git Diff ---\n{ctx.git_diff}\n\n"
        f"--- Peer Review Report ---\n{ctx.review_report}\n\n"
        "Perform a final audit. Check for:\n"
        "1. Security vulnerabilities\n"
        "2. Logic errors\n"
        "3. Missing edge cases\n"
        "4. Consistency with the issue requirements\n\n"
        "After your audit, you MUST end with exactly one of:\n"
        "[FINAL: APPROVED] — if the implementation is ready to merge\n"
        "[FINAL: REJECTED] — if there are critical issues"
    )

    messages = [Message(role=Role.USER, content=user_content)]

    try:
        response = await _call_deepseek_with_progress(
            ollama, messages,
            max_tokens=settings.deepseek_audit_max_tokens,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=settings.deepseek_audit_timeout,
            progress_cb=progress_cb,
            step_name="DeepSeek Audit",
        )
        ctx.audit_result = response.content
        verdict = parse_final(response.content)

        if verdict is True:
            ctx.audit_passed = True
            step.status = "passed"
            step.detail = "FINAL: APPROVED"
        elif verdict is False:
            step.status = "failed"
            step.detail = "FINAL: REJECTED"
            raise RuntimeError("DeepSeek audit returned REJECTED")
        else:
            step.status = "failed"
            step.detail = "No final tag found — treated as REJECTED"
            raise RuntimeError("DeepSeek audit did not include [FINAL: APPROVED/REJECTED]")

    except RuntimeError:
        raise
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


# ── Orchestrator ─────────────────────────────────────────────────────────────

async def run_dual_check_pipeline(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    anthropic: AnthropicProvider | None,
    settings: Settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback,
) -> tuple[str, str]:
    """Run the full 4-step pipeline. Returns (status, detail)."""
    try:
        # Fetch issue
        await progress_cb("Fetching issue...")
        await step_fetch_issue(ctx)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 1: DeepSeek Design
        await progress_cb("[1/4] DeepSeek Design...")
        await step_deepseek_design(ctx, ollama, settings, progress_cb)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 2: Claude Implement
        await progress_cb("[2/4] Claude Implement...")
        await step_claude_implement(ctx, settings, cancel_event, progress_cb)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 3: Claude Review
        await progress_cb("[3/4] Claude Review...")
        await step_claude_review(ctx, anthropic, settings)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 4: DeepSeek Audit
        await progress_cb("[4/4] DeepSeek Audit...")
        await step_deepseek_audit(ctx, ollama, settings, progress_cb)

        return "success", "All 4 checks passed"

    except asyncio.CancelledError:
        return "skipped", "Cancelled by user"
    except Exception as exc:
        # Find which step failed
        for s in ctx.steps:
            if s.status == "failed":
                return "failed", f"{s.name}: {s.detail}"
        return "failed", str(exc)[:200]


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _shell_quote(s: str) -> str:
    """Quote a string for safe shell use (single-quote wrapping)."""
    return "'" + s.replace("'", "'\\''") + "'"


async def _snapshot_commit(project_path: str, issue_num: int) -> None:
    """Stage all and create a WIP commit so diff is based on committed state."""
    add_proc = await asyncio.create_subprocess_exec(
        "git", "add", "-A",
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(add_proc.communicate(), timeout=15)

    commit_proc = await asyncio.create_subprocess_exec(
        "git", "commit", "-m", f"wip: auto-solve #{issue_num}",
        "--allow-empty",
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(commit_proc.communicate(), timeout=15)


async def _claude_cli_review(project_path: str, prompt: str) -> str:
    """Fallback: run Claude CLI for review when Anthropic API is unavailable."""
    proc = await asyncio.create_subprocess_shell(
        f"claude -p {_shell_quote(prompt)}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=project_path,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
    return stdout.decode(errors="replace") if stdout else ""


def format_pipeline_summary(ctx: PipelineContext) -> str:
    """Format pipeline steps into a Telegram-friendly summary."""
    lines: list[str] = []
    for s in ctx.steps:
        icon = {
            "passed": "\u2705",
            "failed": "\u274c",
            "skipped": "\u23ed",
            "running": "\u23f3",
            "pending": "\u2b1c",
        }.get(s.status, "\u2753")

        elapsed_str = ""
        if s.elapsed_sec > 0:
            mins, secs = divmod(int(s.elapsed_sec), 60)
            elapsed_str = f" ({mins}m {secs}s)" if mins else f" ({secs}s)"

        lines.append(f"  {icon} {s.name}{elapsed_str}")
    return "\n".join(lines)
