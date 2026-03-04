"""Triple-Model Pipeline (legacy) and Five-brid 9-Step Pipeline.

Legacy: DeepSeek Design → Qwen Pre-Implement → Claude Implement → Claude Review → DeepSeek Audit → Data Mining.
Fivebrid: Haiku Research → Opus Design → Gemini Critique → Qwen Hints → Sonnet Implement → Sonnet Self-Review → Gemini Cross-Review → DeepSeek Audit → Data Mining.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from .ai.base import AIResponse, Message, Role
from .ai.anthropic_provider import AnthropicProvider
from .ai.gemini_provider import GeminiCLIProvider
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
    issue_title: str = ""
    base_commit: str = ""  # HEAD hash at worktree creation (before Claude runs)
    design_doc: str = ""
    qwen_hints: str = ""
    git_diff: str = ""
    review_report: str = ""
    audit_result: str = ""
    audit_passed: bool = False
    data_mining_result: str = ""
    retry_count: int = 0
    review_feedback: str = ""
    review_passed: bool = False
    # Fivebrid pipeline fields
    research_log: str = ""
    gemini_design_critique: str = ""
    design_iteration: int = 0
    self_review_report: str = ""
    gemini_cross_review: str = ""
    impl_snapshot_ref: str = ""
    ci_check_log: str = ""
    ai_audit_result: str = ""
    ai_audit_passed: bool = False
    ci_fix_history: list[str] = field(default_factory=list)
    audit_fix_history: list[str] = field(default_factory=list)
    steps: list[PipelineStep] = field(default_factory=list)


# ── Verdict Parsing ──────────────────────────────────────────────────────────

def parse_verdict(text: str) -> bool | None:
    """Extract [VERDICT: PASS] or [VERDICT: FAIL] from response (last match wins)."""
    matches = list(re.finditer(r"\[VERDICT:\s*(PASS|FAIL)\]", text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper() == "PASS"
    return None


def parse_final(text: str) -> bool | None:
    """Extract [FINAL: APPROVED] or [FINAL: REJECTED] from response (last match wins)."""
    matches = list(re.finditer(r"\[FINAL:\s*(APPROVED|REJECTED)\]", text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper() == "APPROVED"
    return None


def parse_design_verdict(text: str) -> bool | None:
    """Extract [DESIGN: APPROVED] or [DESIGN: NEEDS_REVISION] (last match wins, case-insensitive)."""
    matches = list(re.finditer(
        r"\[DESIGN:\s*(APPROVED|NEEDS_REVISION)\]", text, re.IGNORECASE,
    ))
    if matches:
        return matches[-1].group(1).upper() == "APPROVED"
    return None


def parse_audit(text: str) -> bool | None:
    """Extract [AUDIT: PASS] or [AUDIT: FAIL] from response (last match wins)."""
    matches = list(re.finditer(r"\[AUDIT:\s*(PASS|FAIL)\]", text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper() == "PASS"
    return None


# ── Git Diff Filtering ───────────────────────────────────────────────────────

DIFF_EXCLUDE_PATTERNS = [
    "*.lock", "*.min.js", "*.min.css", "*.map",
    "*.pyc", "__pycache__/*", "*.egg-info/*",
    "dist/*", "build/*", "node_modules/*",
    "*.png", "*.jpg", "*.gif", "*.ico", "*.woff*",
]

_MAX_DIFF_CHARS = 50_000
_MAX_CI_LOG_CHARS = 3_000


def detect_ci_commands(project_path: str) -> list[str]:
    """Auto-detect build/lint/test commands based on project files."""
    p = Path(project_path)

    # Gradle (KMP, Android, JVM)
    if (p / "gradlew").exists():
        return ["./gradlew check"]

    # Node.js
    pkg_json = p / "package.json"
    if pkg_json.exists():
        try:
            scripts = json.loads(pkg_json.read_text()).get("scripts", {})
            cmds: list[str] = []
            for key in ("lint", "build", "test"):
                if key in scripts:
                    cmds.append(f"npm run {key}")
            return cmds if cmds else []
        except Exception:
            return []

    # Python
    if (p / "pyproject.toml").exists():
        cmds = []
        toml_text = (p / "pyproject.toml").read_text()
        if "ruff" in toml_text:
            cmds.append("ruff check .")
        if "mypy" in toml_text:
            cmds.append("mypy .")
        if "pytest" in toml_text or (p / "tests").is_dir():
            cmds.append("pytest")
        return cmds if cmds else []

    # Rust
    if (p / "Cargo.toml").exists():
        return ["cargo build", "cargo test"]

    # Go
    if (p / "go.mod").exists():
        return ["go build ./...", "go test ./..."]

    return []


def _resolve_ci_commands(project_path: str, project_info: dict | None) -> list[str]:
    """Return CI commands: explicit from project_info if set, else auto-detect."""
    if project_info and "ci_commands" in project_info:
        return list(project_info["ci_commands"])
    return detect_ci_commands(project_path)


async def _run_local_ci(
    project_path: str,
    commands: list[str],
    timeout: int,
) -> tuple[bool, str]:
    """Run CI commands sequentially. Return (success, log). Stop on first failure."""
    full_log = ""
    for cmd in commands:
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=project_path,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace") if stdout else ""
            full_log += f"$ {cmd}\n{output}\n"
            if proc.returncode != 0:
                # Truncate to tail
                if len(full_log) > _MAX_CI_LOG_CHARS:
                    full_log = f"...(truncated)\n{full_log[-_MAX_CI_LOG_CHARS:]}"
                return False, full_log
        except asyncio.TimeoutError:
            full_log += f"$ {cmd}\n[TIMEOUT after {timeout}s]\n"
            if len(full_log) > _MAX_CI_LOG_CHARS:
                full_log = f"...(truncated)\n{full_log[-_MAX_CI_LOG_CHARS:]}"
            return False, full_log

    if len(full_log) > _MAX_CI_LOG_CHARS:
        full_log = f"...(truncated)\n{full_log[-_MAX_CI_LOG_CHARS:]}"
    return True, full_log


def _extract_test_failures(log: str, max_chars: int = 500) -> str:
    """Extract failed test names/errors from CI log for concise reporting."""
    patterns = [
        re.compile(r"^.*FAILED.*$", re.MULTILINE),           # pytest / Gradle
        re.compile(r"^.*FAIL:.*$", re.MULTILINE),            # Go
        re.compile(r"^.*failing.*$", re.MULTILINE | re.IGNORECASE),  # npm/jest
        re.compile(r"^.*Error:.*$", re.MULTILINE),           # Generic errors
        re.compile(r"^E\s+.*$", re.MULTILINE),               # pytest assertion details
    ]
    failures: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in pat.finditer(log):
            line = m.group(0).strip()
            if line not in seen:
                seen.add(line)
                failures.append(line)
    result = "\n".join(failures)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n...(truncated)"
    return result or "(no specific failures extracted)"


async def step_local_ci_check(
    ctx: PipelineContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    ci_commands: list[str],
    step_index: int,
) -> None:
    """Run local CI commands and auto-fix with Sonnet on failure."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()
    max_retries = settings.local_ci_fix_retries

    try:
        for attempt in range(max_retries + 1):
            attempt_label = f" (attempt {attempt + 1}/{max_retries + 1})" if attempt > 0 else ""
            await progress_cb(f"Local CI Check{attempt_label}...")

            success, log = await _run_local_ci(
                ctx.project_path, ci_commands, settings.local_ci_timeout,
            )
            ctx.ci_check_log = log

            if success:
                step.status = "passed"
                step.detail = "CI passed"
                return

            # Failed — try auto-fix if retries remain
            if attempt < max_retries:
                await progress_cb(f"Local CI Check — fixing (attempt {attempt + 1})...")
                fix_prompt = (
                    f"You are a Senior Engineer. Fix the following build/test error "
                    f"while maintaining the architectural integrity defined in CLAUDE.md.\n\n"
                    f"Read .claude/CLAUDE.md first to understand the project conventions.\n\n"
                    f"Failed commands:\n{', '.join(ci_commands)}\n\n"
                    f"--- Error Log ---\n{log}\n---\n\n"
                    f"Fix the errors in the source files. Do NOT create commits or push.\n"
                    f"Do NOT create branches. Only modify the files to fix the errors."
                )
                try:
                    await _call_claude_cli_with_progress(
                        fix_prompt,
                        model=settings.sonnet_model,
                        timeout=settings.local_ci_fix_timeout,
                        progress_cb=progress_cb,
                        step_name="CI Fix",
                        cwd=ctx.project_path,
                        dangerously_skip_permissions=True,
                    )
                    # Snapshot commit so changes are visible in diff
                    await _snapshot_commit(ctx.project_path, ctx.issue_num)
                    # Update diff
                    ctx.git_diff = await _capture_filtered_diff(
                        ctx.project_path, base_ref=ctx.base_commit or "main",
                    )
                    fail_count = len(_extract_test_failures(log).split("\n"))
                    ctx.ci_fix_history.append(f"attempt {attempt + 1}: {fail_count} failures → auto-fixed")
                except Exception as fix_exc:
                    logger.warning("CI auto-fix attempt %d failed: %s", attempt + 1, fix_exc)
            else:
                # No retries left
                if settings.local_ci_fatal:
                    fail_summary = _extract_test_failures(log)
                    step.status = "failed"
                    step.detail = f"CI failed after {max_retries + 1} attempts\n{fail_summary}"
                    raise RuntimeError(step.detail)
                else:
                    step.status = "skipped"
                    step.detail = f"CI failed (non-fatal, {max_retries + 1} attempts)"
                    logger.warning("Local CI check failed but non-fatal for issue #%d", ctx.issue_num)
                    return

    except RuntimeError:
        raise
    except Exception as exc:
        step.status = "skipped"
        step.detail = f"Non-fatal: {str(exc)[:150]}"
        logger.warning("Local CI check error (non-fatal): %s", exc)
    finally:
        step.elapsed_sec = time.monotonic() - start


async def _capture_filtered_diff(project_path: str, base_ref: str = "main") -> str:
    """Capture git diff base_ref..HEAD excluding binary/generated files."""
    diff_range = f"{base_ref}..HEAD"
    exclude_args = [f":(exclude){pat}" for pat in DIFF_EXCLUDE_PATTERNS]

    proc = await asyncio.create_subprocess_exec(
        "git", "diff", diff_range, "--", ".", *exclude_args,
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    diff = stdout.decode(errors="replace") if stdout else ""

    if len(diff) > _MAX_DIFF_CHARS:
        # Capture stat summary as fallback context
        stat_proc = await asyncio.create_subprocess_exec(
            "git", "diff", diff_range, "--stat",
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


# ── Ollama Progress Helper ───────────────────────────────────────────────────

async def _call_ollama_with_progress(
    ollama: OllamaProvider,
    messages: list[Message],
    *,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    timeout: int,
    progress_cb: ProgressCallback,
    step_name: str,
    model: str | None = None,
    num_ctx: int | None = None,
) -> AIResponse:
    """Call an Ollama model with periodic 'Thinking...' progress updates."""
    task = asyncio.create_task(
        ollama.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            timeout=timeout,
            model=model,
            num_ctx=num_ctx,
        )
    )
    elapsed = 0
    while not task.done():
        await asyncio.sleep(10)
        elapsed += 10
        mins, secs = divmod(elapsed, 60)
        await progress_cb(f"{step_name} (Thinking... {mins}m {secs}s)")
    return task.result()


# ── Claude CLI Progress Helper ───────────────────────────────────────────────

async def _call_claude_cli_with_progress(
    prompt: str,
    *,
    model: str,
    timeout: int,
    progress_cb: ProgressCallback,
    step_name: str,
    cwd: str | None = None,
    dangerously_skip_permissions: bool = False,
) -> str:
    """Call Claude CLI with periodic progress updates. Returns the text output."""
    cmd = ["claude", "-p", "--model", model, "--output-format", "text"]
    if dangerously_skip_permissions:
        cmd.append("--dangerously-skip-permissions")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )

    async def _communicate() -> tuple[bytes, bytes]:
        return await proc.communicate(input=prompt.encode())

    task = asyncio.create_task(_communicate())
    elapsed = 0
    while not task.done():
        await asyncio.sleep(10)
        elapsed += 10
        mins, secs = divmod(elapsed, 60)
        await progress_cb(f"{step_name} (Thinking... {mins}m {secs}s)")

    stdout, stderr = await asyncio.wait_for(task, timeout=timeout)
    output = stdout.decode(errors="replace") if stdout else ""

    if proc.returncode != 0:
        err = stderr.decode(errors="replace") if stderr else ""
        raise RuntimeError(f"Claude CLI failed (exit={proc.returncode}): {err[:300]}")

    if not output.strip():
        raise RuntimeError("Claude CLI returned empty response")

    return output


# ── Gemini CLI Progress Helper ───────────────────────────────────────────────

async def _call_gemini_with_progress(
    gemini: GeminiCLIProvider,
    messages: list[Message],
    *,
    system_prompt: str,
    timeout: int,
    progress_cb: ProgressCallback,
    step_name: str,
) -> AIResponse:
    """Call Gemini CLI with periodic progress updates."""
    task = asyncio.create_task(
        gemini.chat(
            messages,
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
    raw = stdout.decode(errors="replace") if stdout else ""
    ctx.issue_body = raw
    try:
        import json
        ctx.issue_title = json.loads(raw).get("title", "")
    except Exception:
        ctx.issue_title = ""


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
        response = await _call_ollama_with_progress(
            ollama, messages,
            max_tokens=settings.deepseek_design_max_tokens,
            temperature=0.7,
            system_prompt=system_prompt,
            timeout=settings.deepseek_design_timeout,
            progress_cb=progress_cb,
            step_name="DeepSeek Design",
            model=settings.reasoning_model,
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


async def step_qwen_pre_implement(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 1,
) -> None:
    """Step 1.5: Qwen2.5-Coder generates implementation hints (code-only, non-fatal)."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    system_prompt = (
        "You are a code generation assistant. "
        "No explanations. Output ONLY: file paths, code snippets, function signatures, "
        "and import statements. Be maximally concise."
    )
    user_content = (
        f"GitHub Issue #{ctx.issue_num}:\n{ctx.issue_body}\n\n"
        f"--- Design Guide ---\n{ctx.design_doc}\n---\n\n"
        "Generate the implementation code for the above design. "
        "Output ONLY code: file paths, code snippets, function signatures, and imports."
    )

    messages = [Message(role=Role.USER, content=user_content)]

    try:
        response = await _call_ollama_with_progress(
            ollama, messages,
            max_tokens=settings.data_mining_max_tokens,
            temperature=0.4,
            system_prompt=system_prompt,
            timeout=settings.qwen_impl_timeout,
            progress_cb=progress_cb,
            step_name="Qwen Pre-Implement",
            model=settings.qwen_model,
        )
        ctx.qwen_hints = response.content
        step.status = "passed"
        step.detail = f"{response.output_tokens} tokens"
    except Exception as exc:
        step.status = "skipped"
        step.detail = f"Non-fatal: {str(exc)[:150]}"
        logger.warning("Qwen pre-implement failed (non-fatal): %s", exc)
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_claude_implement(
    ctx: PipelineContext,
    settings: Settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback,
    step_index: int = 2,
) -> None:
    """Step 2: Claude CLI implements based on DeepSeek's design and Qwen's hints."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    qwen_section = ""
    if ctx.qwen_hints:
        qwen_section = (
            f"\n\n--- Code Suggestions (from Qwen2.5-Coder) ---\n{ctx.qwen_hints}\n---\n"
            f"Use the above code suggestions as a starting reference.\n"
        )

    prompt = (
        f"Read .claude/CLAUDE.md first and follow the defined pipeline. "
        f"Then read GitHub issue #{ctx.issue_num} with `gh issue view {ctx.issue_num}`. "
        f"\n\n--- Design Guide (from DeepSeek) ---\n{ctx.design_doc}\n---\n"
        f"{qwen_section}\n"
        f"Implement the solution following this design guide using TDD:\n"
        f"1. Write failing tests FIRST that define the expected behavior\n"
        f"2. Run the tests to confirm they fail\n"
        f"3. Implement the minimum code to make the tests pass\n"
        f"4. Run the tests again to confirm they pass\n"
        f"5. Refactor if needed while keeping tests green\n"
        f"IMPORTANT: After implementation, run the project's compile/build command "
        f"AND the full test suite to verify everything passes. "
        f"Fix any failures before proceeding.\n\n"
        f"GIT RESTRICTIONS (OVERRIDE CLAUDE.md Section 6):\n"
        f"- Do NOT create any branches. You are already on the correct branch.\n"
        f"- Do NOT run git push.\n"
        f"- Do NOT run gh pr create.\n"
        f"- Only commit your changes locally. Branch management and PR creation are handled externally."
    )

    if ctx.review_feedback:
        prompt += (
            f"\n\n--- Previous Review Feedback (MUST address these issues) ---\n"
            f"{ctx.review_feedback}\n---\n\n"
            f"The previous implementation was rejected. Fix ALL issues mentioned above."
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
        ctx.git_diff = await _capture_filtered_diff(
            ctx.project_path, base_ref=ctx.base_commit or "main",
        )

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
    progress_cb: ProgressCallback | None = None,
    step_index: int = 3,
) -> None:
    """Step 3: Claude reviews the implementation via Anthropic API (or CLI fallback)."""
    step = ctx.steps[step_index]
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
        f"[VERDICT: FAIL] — if there are critical issues\n\n"
        f"IMPORTANT: [VERDICT: ...] must be the VERY LAST line of your response. "
        f"Do not write anything after the verdict tag."
    )

    # Periodic progress updates while review is running
    async def _tick() -> None:
        while True:
            await asyncio.sleep(15)
            if progress_cb:
                elapsed = int(time.monotonic() - start)
                mins, secs = divmod(elapsed, 60)
                await progress_cb(f"[3/5] Claude Review [{mins}m {secs}s]...")

    tick_task = asyncio.create_task(_tick())

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
            review_text = await _claude_cli_review(
                ctx.project_path, review_prompt, timeout=settings.claude_review_timeout,
            )

        ctx.review_report = review_text
        verdict = parse_verdict(review_text)

        if verdict is True:
            ctx.review_passed = True
            step.status = "passed"
            step.detail = "VERDICT: PASS"
            logger.info("Review PASSED for issue #%d", ctx.issue_num)
        elif verdict is False:
            ctx.review_passed = False
            step.status = "failed"
            step.detail = "VERDICT: FAIL"
            logger.warning(
                "Review FAILED for issue #%d. Full review report:\n%s",
                ctx.issue_num, review_text,
            )
        else:
            ctx.review_passed = False
            step.status = "failed"
            step.detail = "No verdict tag found — treated as FAIL"
            logger.warning(
                "Review returned no verdict for issue #%d. Full review report:\n%s",
                ctx.issue_num, review_text,
            )

    except Exception as exc:
        ctx.review_passed = False
        step.status = "failed"
        step.detail = str(exc)[:200]
        logger.warning("Claude review error for issue #%d: %s", ctx.issue_num, exc)
    finally:
        tick_task.cancel()
        step.elapsed_sec = time.monotonic() - start


async def step_deepseek_audit(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = -1,
) -> None:
    """Step 4: DeepSeek final audit of the implementation."""
    step = ctx.steps[step_index]
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
        response = await _call_ollama_with_progress(
            ollama, messages,
            max_tokens=settings.deepseek_audit_max_tokens,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=settings.deepseek_audit_timeout,
            progress_cb=progress_cb,
            step_name="DeepSeek Audit",
            model=settings.reasoning_model,
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


async def step_ai_audit(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = -1,
) -> None:
    """Intent-based adversarial AI audit using DeepSeek reasoning model."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    # Read project CLAUDE.md if available
    claude_md_path = Path(ctx.project_path) / ".claude" / "CLAUDE.md"
    claude_md_content = ""
    if claude_md_path.exists():
        try:
            claude_md_content = claude_md_path.read_text(errors="replace")[:3000]
        except Exception:
            claude_md_content = "(could not read CLAUDE.md)"

    system_prompt = (
        "You are an adversarial code auditor. Your job is to FIND FLAWS, "
        "not to praise. Assume the implementer made mistakes. "
        "Think like an attacker exploiting this code in production."
    )

    user_content = (
        f"## Original Intent\n"
        f"Issue #{ctx.issue_num}: {ctx.issue_title}\n{ctx.issue_body}\n\n"
        f"## Design Document\n{ctx.design_doc}\n\n"
        f"## Implementation Diff\n{ctx.git_diff}\n\n"
    )
    if claude_md_content:
        user_content += f"## CLAUDE.md Guidelines\n{claude_md_content}\n\n"
    if ctx.review_report:
        user_content += f"## Previous Reviews\n{ctx.review_report}\n\n"

    user_content += (
        "Audit this implementation against the ORIGINAL INTENT:\n"
        "1. INTENT ALIGNMENT: Does the code fully satisfy the issue requirements? Any missing functionality?\n"
        "2. LOGIC FLAWS: Edge cases, race conditions, off-by-one errors the implementer likely missed.\n"
        "3. SECURITY: Injection, XSS, auth bypass, data exposure vulnerabilities.\n"
        "4. GUIDELINE VIOLATIONS: Does the code violate CLAUDE.md conventions?\n"
        "   - COMMENT POLICY: If the code contains 'what' comments instead of 'why' comments, mark as [MAJOR].\n"
        "     Only 'why' comments are allowed. Self-documenting naming is mandatory.\n\n"
        "For each finding, classify as [CRITICAL], [MAJOR], or [MINOR].\n\n"
        "End with exactly one of:\n"
        "[AUDIT: PASS] — no critical or major issues\n"
        "[AUDIT: FAIL] — critical issues found that must be fixed"
    )

    messages = [Message(role=Role.USER, content=user_content)]

    try:
        response = await _call_ollama_with_progress(
            ollama, messages,
            max_tokens=settings.ai_audit_max_tokens,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=settings.ai_audit_timeout,
            progress_cb=progress_cb,
            step_name="AI Audit",
            model=settings.reasoning_model,
            num_ctx=settings.ai_audit_num_ctx,
        )
        ctx.ai_audit_result = response.content
        verdict = parse_audit(response.content)

        if verdict is None:
            # Fallback: try legacy [FINAL: APPROVED/REJECTED] format
            legacy = parse_final(response.content)
            if legacy is not None:
                verdict = legacy

        if verdict is True:
            ctx.ai_audit_passed = True
            # Also set legacy fields for backward compat
            ctx.audit_result = response.content
            ctx.audit_passed = True
            step.status = "passed"
            step.detail = "AUDIT: PASS"
        elif verdict is False:
            ctx.audit_result = response.content
            ctx.audit_passed = False
            step.status = "failed"
            step.detail = "AUDIT: FAIL"
            raise RuntimeError("AI Audit returned FAIL — critical issues found")
        else:
            ctx.audit_result = response.content
            ctx.audit_passed = False
            step.status = "failed"
            step.detail = "No audit tag found — treated as FAIL"
            raise RuntimeError("AI Audit did not include [AUDIT: PASS/FAIL]")

    except RuntimeError:
        raise
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


# ── Data Mining ──────────────────────────────────────────────────────────────

async def step_data_mining(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = -1,
) -> None:
    """Step 5: Qwen generates training data from the successful solve (non-fatal)."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    system_prompt = (
        "You are a training data generator. Analyze the code diff and issue context. "
        "Output JSONL (one JSON object per line). Each object must have exactly two keys: "
        '"instruction" (a natural-language coding task) and "output" (the code that solves it). '
        "Output ONLY valid JSONL lines. No markdown fences. No explanations."
    )
    user_content = (
        f"GitHub Issue #{ctx.issue_num}:\n{ctx.issue_body}\n\n"
        f"--- Git Diff ---\n{ctx.git_diff}\n\n"
        "Generate instruction-output pairs as JSONL for fine-tuning a code model."
    )

    messages = [Message(role=Role.USER, content=user_content)]

    try:
        response = await _call_ollama_with_progress(
            ollama, messages,
            max_tokens=settings.data_mining_max_tokens,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=settings.data_mining_timeout,
            progress_cb=progress_cb,
            step_name="Data Mining",
            model=settings.qwen_model,
        )
        ctx.data_mining_result = response.content
        valid, dropped = _write_training_data(ctx, settings)
        step.status = "passed"
        step.detail = f"{valid} pairs saved, {dropped} dropped"
    except Exception as exc:
        step.status = "skipped"
        step.detail = f"Non-fatal: {str(exc)[:150]}"
        logger.warning("Data mining failed (non-fatal): %s", exc)
    finally:
        step.elapsed_sec = time.monotonic() - start


def _write_training_data(ctx: PipelineContext, settings: Settings) -> tuple[int, int]:
    """Write JSONL training data to disk. Returns (valid_count, dropped_count)."""
    raw = ctx.data_mining_result
    if not raw.strip():
        return 0, 0

    # Resolve output directory
    if settings.training_data_dir:
        out_dir = Path(settings.training_data_dir)
    else:
        out_dir = Path(__file__).parent.parent / "data" / "training" / ctx.project_name

    os.makedirs(out_dir, exist_ok=True)

    # Strip markdown code fences that LLMs often wrap JSONL in
    cleaned = re.sub(r"```(?:json|jsonl)?\s*\n?", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)

    valid_lines: list[str] = []
    dropped = 0
    for line in cleaned.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                valid_lines.append(json.dumps(obj, ensure_ascii=False))
            else:
                dropped += 1
        except json.JSONDecodeError:
            dropped += 1

    if not valid_lines:
        return 0, dropped

    timestamp = int(time.time())
    filename = f"issue-{ctx.issue_num}-{timestamp}.jsonl"
    filepath = out_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_lines) + "\n")

    logger.info("Training data written: %s (%d valid, %d dropped)", filepath, len(valid_lines), dropped)
    return len(valid_lines), dropped


# ── Fivebrid Step Functions ───────────────────────────────────────────────────

async def step_haiku_research(
    ctx: PipelineContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 0,
) -> None:
    """Step 0: Haiku CLI investigates the issue and explores existing code patterns."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    prompt = (
        f"You are a research assistant preparing context for a senior engineer.\n\n"
        f"GitHub Issue #{ctx.issue_num}:\n{ctx.issue_body}\n\n"
        f"Tasks:\n"
        f"1. Summarize the issue requirements clearly.\n"
        f"2. Identify which files, modules, and patterns in this project are most relevant.\n"
        f"3. Note any existing similar patterns in the codebase that could be reused.\n"
        f"4. Flag potential pitfalls or edge cases.\n\n"
        f"Be concise and actionable. Focus on facts, not opinions."
    )

    try:
        output = await _call_claude_cli_with_progress(
            prompt,
            model=settings.haiku_model,
            timeout=settings.research_timeout,
            progress_cb=progress_cb,
            step_name="Haiku Research",
            cwd=ctx.project_path,
        )
        ctx.research_log = output
        step.status = "passed"
        step.detail = f"{len(output)} chars"
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_opus_design(
    ctx: PipelineContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 1,
) -> None:
    """Step 1: Opus CLI creates a detailed implementation design document."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    critique_section = ""
    if ctx.gemini_design_critique:
        critique_section = (
            f"\n\n--- Previous Design Critique (from Gemini) ---\n"
            f"{ctx.gemini_design_critique}\n---\n"
            f"Address ALL issues raised in the critique above.\n"
        )

    prompt = (
        f"You are a senior architect creating a detailed implementation plan.\n\n"
        f"GitHub Issue #{ctx.issue_num}:\n{ctx.issue_body}\n\n"
        f"--- Research Context ---\n{ctx.research_log}\n---\n"
        f"{critique_section}\n"
        f"Create a comprehensive design document including:\n"
        f"1. Files to create/modify with exact paths\n"
        f"2. TDD test plan: list the tests to write FIRST (before implementation)\n"
        f"3. Detailed implementation steps with code structure\n"
        f"4. Data flow and component interactions\n"
        f"5. Edge cases and error handling strategy\n\n"
        f"IMPORTANT: Follow TDD methodology. The test plan (step 2) must come before "
        f"implementation details (step 3). Tests define the expected behavior.\n\n"
        f"Be specific and actionable — this document will guide the implementation directly."
    )

    try:
        output = await _call_claude_cli_with_progress(
            prompt,
            model=settings.opus_model,
            timeout=settings.opus_design_timeout,
            progress_cb=progress_cb,
            step_name="Opus Design",
            cwd=ctx.project_path,
        )
        ctx.design_doc = output
        step.status = "passed"
        step.detail = f"{len(output)} chars"
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_gemini_design_critique(
    ctx: PipelineContext,
    gemini: GeminiCLIProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 2,
) -> None:
    """Step 2: Gemini critiques the design. Non-fatal — APPROVED if verdict missing or error."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    system_prompt = (
        "You are a design reviewer. Evaluate the implementation plan critically. "
        "Check for: architectural soundness, missing edge cases, security concerns, "
        "and adherence to the issue requirements."
    )
    user_content = (
        f"GitHub Issue #{ctx.issue_num}:\n{ctx.issue_body}\n\n"
        f"--- Research Context ---\n{ctx.research_log}\n\n"
        f"--- Design Document ---\n{ctx.design_doc}\n\n"
        f"Review this design. End your review with exactly one of:\n"
        f"[DESIGN: APPROVED] — if the design is solid\n"
        f"[DESIGN: NEEDS_REVISION] — if there are critical issues\n\n"
        f"If NEEDS_REVISION, clearly list what must be changed."
    )

    messages = [Message(role=Role.USER, content=user_content)]

    try:
        response = await _call_gemini_with_progress(
            gemini, messages,
            system_prompt=system_prompt,
            timeout=settings.gemini_critique_timeout,
            progress_cb=progress_cb,
            step_name="Gemini Design Critique",
        )
        ctx.gemini_design_critique = response.content
        verdict = parse_design_verdict(response.content)

        if verdict is True:
            step.status = "passed"
            step.detail = "DESIGN: APPROVED"
        elif verdict is False:
            step.status = "failed"
            step.detail = "DESIGN: NEEDS_REVISION"
        else:
            # No verdict tag found — treat as APPROVED to prevent infinite loop
            step.status = "passed"
            step.detail = "No verdict tag — treated as APPROVED"
            logger.warning("Gemini design critique returned no verdict tag, treating as APPROVED")
    except Exception as exc:
        # Non-fatal: proceed with current design
        step.status = "skipped"
        step.detail = f"Non-fatal: {str(exc)[:150]}"
        logger.warning("Gemini design critique failed (non-fatal): %s", exc)
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_sonnet_self_review(
    ctx: PipelineContext,
    settings: Settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback,
    step_index: int = 5,
) -> None:
    """Step 5: Sonnet self-reviews and fixes its own implementation. Safe-fail with snapshot recovery."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    prompt = (
        f"You are reviewing code you just wrote for GitHub issue #{ctx.issue_num}.\n\n"
        f"--- Issue ---\n{ctx.issue_body}\n\n"
        f"--- Design Document ---\n{ctx.design_doc}\n\n"
        f"--- Current Git Diff ---\n{ctx.git_diff}\n\n"
        f"Tasks:\n"
        f"1. Review your implementation for bugs, missing edge cases, and code quality issues.\n"
        f"2. Verify test coverage: are there tests for all new/changed behavior? Add missing tests.\n"
        f"3. Fix any issues you find directly in the code files.\n"
        f"4. Run the full test suite and the project's compile/build command.\n"
        f"5. Fix any test failures or compile errors.\n\n"
        f"GIT RESTRICTIONS:\n"
        f"- Do NOT create branches, push, or create PRs.\n"
        f"- Only modify files and commit locally."
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", "--model", settings.sonnet_model,
            "--output-format", "text", "--dangerously-skip-permissions",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=ctx.project_path,
        )

        async def _communicate() -> tuple[bytes, bytes]:
            return await proc.communicate(input=prompt.encode())

        comm_task = asyncio.create_task(_communicate())
        elapsed = 0
        while not comm_task.done():
            await asyncio.sleep(10)
            elapsed += 10
            if cancel_event.is_set():
                proc.kill()
                await comm_task
                raise asyncio.CancelledError("Cancelled by user")
            mins, secs = divmod(elapsed, 60)
            await progress_cb(f"Sonnet Self-Review [{mins}m {secs}s]")

        stdout, stderr = await asyncio.wait_for(
            comm_task, timeout=settings.sonnet_self_review_timeout,
        )
        output = stdout.decode(errors="replace") if stdout else ""

        if proc.returncode != 0:
            raise RuntimeError(f"Sonnet self-review CLI failed (exit={proc.returncode})")

        # Snapshot commit after self-review modifications
        await _snapshot_commit(ctx.project_path, ctx.issue_num)

        # Re-capture diff after self-review changes
        ctx.git_diff = await _capture_filtered_diff(
            ctx.project_path, base_ref=ctx.base_commit or "main",
        )

        ctx.self_review_report = output
        step.status = "passed"
        step.detail = f"Self-review complete, diff={len(ctx.git_diff)} chars"

    except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
        # Safe-fail: recover snapshot
        if ctx.impl_snapshot_ref:
            logger.warning("Self-review %s — recovering snapshot %s",
                           type(exc).__name__, ctx.impl_snapshot_ref[:8])
            await _reset_to_snapshot(ctx.project_path, ctx.impl_snapshot_ref)
            ctx.git_diff = await _capture_filtered_diff(
                ctx.project_path, base_ref=ctx.base_commit or "main",
            )
        step.status = "skipped"
        step.detail = f"Safe-fail: {type(exc).__name__}, snapshot recovered"
        if isinstance(exc, asyncio.CancelledError):
            raise
    except Exception as exc:
        # Safe-fail: recover snapshot on any error
        if ctx.impl_snapshot_ref:
            logger.warning("Self-review error — recovering snapshot %s: %s",
                           ctx.impl_snapshot_ref[:8], exc)
            await _reset_to_snapshot(ctx.project_path, ctx.impl_snapshot_ref)
            ctx.git_diff = await _capture_filtered_diff(
                ctx.project_path, base_ref=ctx.base_commit or "main",
            )
        step.status = "skipped"
        step.detail = f"Safe-fail: {str(exc)[:150]}, snapshot recovered"
        logger.warning("Sonnet self-review failed (safe-fail): %s", exc)
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_gemini_cross_review(
    ctx: PipelineContext,
    gemini: GeminiCLIProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 6,
) -> None:
    """Step 6: Gemini cross-reviews the implementation. Non-fatal — defaults to PASS."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    system_prompt = (
        "You are a cross-reviewer examining an implementation from a different perspective. "
        "Focus on business logic correctness, edge cases, and potential regressions."
    )

    review_context = ""
    if ctx.self_review_report:
        review_context = f"\n--- Self-Review Report ---\n{ctx.self_review_report}\n"

    user_content = (
        f"GitHub Issue #{ctx.issue_num}:\n{ctx.issue_body}\n\n"
        f"--- Design Document ---\n{ctx.design_doc}\n\n"
        f"--- Git Diff ---\n{ctx.git_diff}\n"
        f"{review_context}\n"
        f"Review the implementation for:\n"
        f"1. Business logic correctness\n"
        f"2. Edge cases and error handling\n"
        f"3. Potential regressions\n"
        f"4. Code quality and maintainability\n\n"
        f"End with exactly one of:\n"
        f"[VERDICT: PASS] — if the implementation is acceptable\n"
        f"[VERDICT: FAIL] — if there are critical issues"
    )

    messages = [Message(role=Role.USER, content=user_content)]

    try:
        response = await _call_gemini_with_progress(
            gemini, messages,
            system_prompt=system_prompt,
            timeout=settings.gemini_cross_review_timeout,
            progress_cb=progress_cb,
            step_name="Gemini Cross-Review",
        )
        ctx.gemini_cross_review = response.content
        verdict = parse_verdict(response.content)

        if verdict is True:
            step.status = "passed"
            step.detail = "VERDICT: PASS"
        elif verdict is False:
            step.status = "failed"
            step.detail = "VERDICT: FAIL"
            logger.warning("Gemini cross-review FAILED for issue #%d", ctx.issue_num)
        else:
            # No verdict — default to PASS (non-fatal)
            step.status = "passed"
            step.detail = "No verdict tag — treated as PASS"
    except Exception as exc:
        # Non-fatal: proceed to DeepSeek audit
        step.status = "skipped"
        step.detail = f"Non-fatal: {str(exc)[:150]}"
        logger.warning("Gemini cross-review failed (non-fatal): %s", exc)
    finally:
        step.elapsed_sec = time.monotonic() - start


# ── Fivebrid Data Mining (Enhanced) ──────────────────────────────────────────

async def step_data_mining_fivebrid(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = -1,
) -> None:
    """Step 8: Enhanced data mining — bundles research + design + code as training data."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    system_prompt = (
        "You are a training data generator. Create high-quality fine-tuning data. "
        "Output JSONL (one JSON object per line). Each object must have exactly these keys: "
        '"instruction" (background context + design intent summary + task), '
        '"output" (the final code), '
        '"metadata" (object with "issue" number and "project" name). '
        "IMPORTANT: Compress unnecessary logs and intermediate steps. "
        "The 'instruction' should clearly show the correlation between the design intent "
        "and the final diff. Summarize the research background and design core intent concisely. "
        "Output ONLY valid JSONL lines. No markdown fences. No explanations."
    )
    user_content = (
        f"GitHub Issue #{ctx.issue_num} (project: {ctx.project_name}):\n{ctx.issue_body}\n\n"
        f"--- Research Background ---\n{ctx.research_log}\n\n"
        f"--- Design Document ---\n{ctx.design_doc}\n\n"
        f"--- Final Git Diff ---\n{ctx.git_diff}\n\n"
        "Generate instruction-output pairs as JSONL. Bundle the research background, "
        "design intent, and final code into cohesive training examples."
    )

    messages = [Message(role=Role.USER, content=user_content)]

    try:
        response = await _call_ollama_with_progress(
            ollama, messages,
            max_tokens=settings.data_mining_max_tokens,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=settings.data_mining_timeout,
            progress_cb=progress_cb,
            step_name="Data Mining",
            model=settings.qwen_model,
        )
        ctx.data_mining_result = response.content
        valid, dropped = _write_training_data(ctx, settings)
        step.status = "passed"
        step.detail = f"{valid} pairs saved, {dropped} dropped"
    except Exception as exc:
        step.status = "skipped"
        step.detail = f"Non-fatal: {str(exc)[:150]}"
        logger.warning("Data mining (fivebrid) failed (non-fatal): %s", exc)
    finally:
        step.elapsed_sec = time.monotonic() - start


# ── Fivebrid Orchestrator ────────────────────────────────────────────────────

async def run_fivebrid_pipeline(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    gemini: GeminiCLIProvider,
    settings: Settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback,
    project_info: dict | None = None,
) -> tuple[str, str]:
    """Run the 9-step Five-brid pipeline (10 with Local CI Check). Returns (status, detail)."""

    # Resolve CI commands once at start
    ci_commands = _resolve_ci_commands(ctx.project_path, project_info) if settings.local_ci_enabled else []

    # Initialize fivebrid steps
    if not ctx.steps:
        steps = [
            PipelineStep(name="Haiku Research"),        # 0
            PipelineStep(name="Opus Design"),           # 1
            PipelineStep(name="Gemini Design Critique"),# 2
            PipelineStep(name="Qwen Hints"),            # 3
            PipelineStep(name="Sonnet Implement"),      # 4
        ]
        if ci_commands:
            steps.append(PipelineStep(name="Local CI Check"))  # 5
        steps += [
            PipelineStep(name="Sonnet Self-Review"),    # 5 or 6
            PipelineStep(name="Gemini Cross-Review"),   # 6 or 7
            PipelineStep(name="AI Audit"),               # 7 or 8
        ]
        ctx.steps = steps

    try:
        # Fetch issue
        await progress_cb("Fetching issue...")
        await step_fetch_issue(ctx)

        total_steps = len(ctx.steps)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # ── Phase 0: Research ──
        await progress_cb(f"[0/{total_steps}] Haiku Research...")
        await step_haiku_research(ctx, settings, progress_cb, step_index=0)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # ── Phase A: Design Loop ──
        for iteration in range(settings.max_design_retries + 1):
            ctx.design_iteration = iteration

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

            # Step 1: Opus Design
            step_label = f" (iteration {iteration + 1})" if iteration > 0 else ""
            await progress_cb(f"[1/{total_steps}] Opus Design{step_label}...")

            if iteration > 0:
                # Insert new design + critique step pairs after previous critique
                # Previous critique is at index 2 + (iteration-1)*2
                insert_at = 3 + (iteration - 1) * 2
                ctx.steps.insert(insert_at, PipelineStep(name=f"Opus Design (retry {iteration})"))
                ctx.steps.insert(insert_at + 1, PipelineStep(name=f"Gemini Critique (retry {iteration})"))
                total_steps = len(ctx.steps)

            design_idx = 1 + (iteration * 2)
            await step_opus_design(ctx, settings, progress_cb, step_index=design_idx)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

            # Step 2: Gemini Design Critique (non-fatal)
            critique_idx = design_idx + 1
            await progress_cb(f"[2/{total_steps}] Gemini Design Critique{step_label}...")
            await step_gemini_design_critique(
                ctx, gemini, settings, progress_cb, step_index=critique_idx,
            )

            critique_step = ctx.steps[critique_idx]
            # Check if critique says NEEDS_REVISION and we have retries left
            if critique_step.status == "failed" and "NEEDS_REVISION" in critique_step.detail:
                if iteration < settings.max_design_retries:
                    await progress_cb(
                        f"Design needs revision (iteration {iteration + 1}/{settings.max_design_retries + 1}), retrying..."
                    )
                    continue
            # APPROVED, skipped, or no more retries — proceed
            break

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # ── Phase B: Implementation ──

        # Find step indices dynamically (design loop may have added extra steps)
        total_steps = len(ctx.steps)
        qwen_idx = next(i for i, s in enumerate(ctx.steps) if s.name == "Qwen Hints")
        impl_idx = next(i for i, s in enumerate(ctx.steps) if s.name == "Sonnet Implement")

        # Step 3: Qwen Hints (non-fatal)
        try:
            await progress_cb(f"[{qwen_idx}/{total_steps}] Qwen Pre-Implement...")
            await step_qwen_pre_implement(ctx, ollama, settings, progress_cb, step_index=qwen_idx)
        except Exception as exc:
            logger.warning("Qwen pre-implement failed (non-fatal): %s", exc)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 4: Sonnet Implement (uses existing step_claude_implement)
        await progress_cb(f"[4/{total_steps}] Sonnet Implement...")
        await step_claude_implement(ctx, settings, cancel_event, progress_cb, step_index=impl_idx)

        # Capture snapshot ref for safe-fail recovery
        snap_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=ctx.project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        snap_out, _ = await asyncio.wait_for(snap_proc.communicate(), timeout=5)
        ctx.impl_snapshot_ref = snap_out.decode().strip() if snap_out else ""

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 5 (optional): Local CI Check
        if ci_commands:
            ci_idx = next(i for i, s in enumerate(ctx.steps) if s.name == "Local CI Check")
            await progress_cb(f"[5/{total_steps}] Local CI Check...")
            await step_local_ci_check(ctx, settings, progress_cb, ci_commands, step_index=ci_idx)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

        # Re-resolve dynamic indices after possible CI step
        self_review_idx = next(i for i, s in enumerate(ctx.steps) if s.name == "Sonnet Self-Review")

        # Step 5/6: Sonnet Self-Review (safe-fail)
        await progress_cb(f"[{self_review_idx}/{total_steps}] Sonnet Self-Review...")
        await step_sonnet_self_review(ctx, settings, cancel_event, progress_cb, step_index=self_review_idx)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # ── Phase C: Audit ──
        cross_review_idx = next(i for i, s in enumerate(ctx.steps) if s.name == "Gemini Cross-Review")
        audit_idx = next(i for i, s in enumerate(ctx.steps) if s.name == "AI Audit")

        # Gemini Cross-Review (non-fatal)
        await progress_cb(f"[{cross_review_idx}/{total_steps}] Gemini Cross-Review...")
        await step_gemini_cross_review(ctx, gemini, settings, progress_cb, step_index=cross_review_idx)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # AI Audit with retry loop
        ctx.review_report = ctx.gemini_cross_review or ctx.self_review_report or "(no review available)"

        for audit_attempt in range(settings.ai_audit_max_retries + 1):
            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

            audit_label = f" (retry {audit_attempt})" if audit_attempt > 0 else ""
            await progress_cb(f"[{audit_idx}/{total_steps}] AI Audit{audit_label}...")

            try:
                await step_ai_audit(ctx, ollama, settings, progress_cb, step_index=audit_idx)
                break  # PASS
            except RuntimeError:
                if audit_attempt < settings.ai_audit_max_retries:
                    # Audit failed — feed back to Sonnet for re-implementation
                    ctx.audit_fix_history.append(
                        f"audit retry {audit_attempt + 1}: {ctx.ai_audit_result[:200]}"
                    )
                    ctx.review_feedback = ctx.ai_audit_result

                    await progress_cb(f"AI Audit FAIL — Sonnet re-implementing (retry {audit_attempt + 1})...")
                    # Reset to snapshot
                    if ctx.impl_snapshot_ref:
                        await _reset_to_snapshot(ctx.project_path, ctx.impl_snapshot_ref)

                    # Re-implement
                    await step_claude_implement(ctx, settings, cancel_event, progress_cb, step_index=impl_idx)
                    await _snapshot_commit(ctx.project_path, ctx.issue_num)
                    ctx.git_diff = await _capture_filtered_diff(
                        ctx.project_path, base_ref=ctx.base_commit or "main",
                    )

                    # Re-run CI if applicable
                    if ci_commands:
                        await step_local_ci_check(ctx, settings, progress_cb, ci_commands, step_index=ci_idx)

                    # Reset audit step for retry
                    ctx.steps[audit_idx] = PipelineStep(name=f"AI Audit (retry {audit_attempt + 1})")
                else:
                    raise  # Final failure

        # ── Phase D: Post-Process ──
        if settings.enable_data_mining and ctx.ai_audit_passed:
            ctx.steps.append(PipelineStep(name="Data Mining"))
            try:
                await progress_cb(f"[{total_steps}/{total_steps}] Data Mining...")
                await step_data_mining_fivebrid(ctx, ollama, settings, progress_cb, step_index=-1)
            except Exception as exc:
                logger.warning("Data mining step failed (non-fatal): %s", exc)

        return "success", "All checks passed"

    except asyncio.CancelledError:
        return "skipped", "Cancelled by user"
    except Exception as exc:
        for s in ctx.steps:
            if s.status == "failed":
                return "failed", f"{s.name}: {s.detail}"
        return "failed", str(exc)[:200]


# ── Legacy Orchestrator ──────────────────────────────────────────────────────

async def run_dual_check_pipeline(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    anthropic: AnthropicProvider | None,
    settings: Settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback,
    project_info: dict | None = None,
) -> tuple[str, str]:
    """Run the full 6-step pipeline with automatic retry on review failure. Returns (status, detail)."""
    max_retries = settings.max_review_retries

    # Resolve CI commands once at start
    ci_commands = _resolve_ci_commands(ctx.project_path, project_info) if settings.local_ci_enabled else []

    # Initialize steps if empty (default_factory is now list)
    if not ctx.steps:
        steps = [
            PipelineStep(name="DeepSeek Design"),      # 0
            PipelineStep(name="Qwen Pre-Implement"),   # 1
            PipelineStep(name="Claude Implement"),      # 2
        ]
        if ci_commands:
            steps.append(PipelineStep(name="Local CI Check"))  # 3
        steps += [
            PipelineStep(name="Claude Review"),         # 3 or 4
            PipelineStep(name="AI Audit"),              # 4 or 5
        ]
        ctx.steps = steps

    try:
        # Fetch issue
        await progress_cb("Fetching issue...")
        await step_fetch_issue(ctx)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 1: DeepSeek Design (once)
        await progress_cb("[1/6] DeepSeek Design...")
        await step_deepseek_design(ctx, ollama, settings, progress_cb)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 1.5: Qwen Pre-Implement (non-fatal)
        try:
            await progress_cb("[1.5/6] Qwen Pre-Implement...")
            await step_qwen_pre_implement(ctx, ollama, settings, progress_cb, step_index=1)
        except Exception as exc:
            logger.warning("Qwen pre-implement step failed (non-fatal): %s", exc)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Steps 2+3+4: Implement + CI Check + Review (retry loop)
        ci_step_offset = 1 if ci_commands else 0  # extra step per attempt when CI enabled
        steps_per_attempt = 2 + ci_step_offset  # impl + (ci) + review

        for attempt in range(max_retries + 1):
            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

            # On retry: reset worktree and add new step entries
            if attempt > 0:
                ctx.retry_count = attempt
                await _reset_worktree(ctx.project_path)
                # Insert new steps before the last step (DeepSeek Audit)
                ctx.steps.insert(-1, PipelineStep(name=f"Claude Implement (retry {attempt})"))
                if ci_commands:
                    ctx.steps.insert(-1, PipelineStep(name=f"Local CI Check (retry {attempt})"))
                ctx.steps.insert(-1, PipelineStep(name=f"Claude Review (retry {attempt})"))

            step_idx_impl = 2 + (attempt * steps_per_attempt)
            step_idx_ci = step_idx_impl + 1 if ci_commands else None
            step_idx_review = step_idx_impl + ci_step_offset + 1

            # Step 2: Claude Implement
            attempt_label = f" (retry {attempt})" if attempt > 0 else ""
            await progress_cb(f"[2/6] Claude Implement{attempt_label}...")
            await step_claude_implement(ctx, settings, cancel_event, progress_cb, step_index=step_idx_impl)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

            # Step 3: Local CI Check (if CI commands detected)
            if ci_commands and step_idx_ci is not None:
                await progress_cb(f"[3/6] Local CI Check{attempt_label}...")
                await step_local_ci_check(
                    ctx, settings, progress_cb, ci_commands, step_index=step_idx_ci,
                )

                if cancel_event.is_set():
                    return "skipped", "Cancelled by user"

            # Step 4: Claude Review
            await progress_cb(f"[4/6] Claude Review{attempt_label}...")
            await step_claude_review(ctx, anthropic, settings, progress_cb=progress_cb, step_index=step_idx_review)

            if ctx.review_passed:
                break

            # Review failed — retry if possible
            if attempt < max_retries:
                ctx.review_feedback = ctx.review_report
                await progress_cb(
                    f"Review failed (attempt {attempt + 1}/{max_retries + 1}), retrying..."
                )
            else:
                return "failed", f"Claude Review: VERDICT: FAIL (after {attempt + 1} attempts)"

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 5: AI Audit with retry loop
        audit_idx = next(i for i, s in enumerate(ctx.steps) if s.name == "AI Audit")

        for audit_attempt in range(settings.ai_audit_max_retries + 1):
            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

            audit_label = f" (retry {audit_attempt})" if audit_attempt > 0 else ""
            await progress_cb(f"[5/6] AI Audit{audit_label}...")

            try:
                await step_ai_audit(ctx, ollama, settings, progress_cb, step_index=audit_idx)
                break  # PASS
            except RuntimeError:
                if audit_attempt < settings.ai_audit_max_retries:
                    ctx.audit_fix_history.append(
                        f"audit retry {audit_attempt + 1}: {ctx.ai_audit_result[:200]}"
                    )
                    ctx.review_feedback = ctx.ai_audit_result
                    await progress_cb(f"AI Audit FAIL — Sonnet re-implementing (retry {audit_attempt + 1})...")
                    await _reset_worktree(ctx.project_path)

                    # Re-implement
                    impl_retry_idx = len(ctx.steps)
                    ctx.steps.insert(audit_idx, PipelineStep(name=f"Claude Implement (audit-fix {audit_attempt + 1})"))
                    audit_idx += 1  # Shift audit index
                    await step_claude_implement(ctx, settings, cancel_event, progress_cb, step_index=impl_retry_idx)

                    # Re-run CI if applicable
                    if ci_commands:
                        ci_fix_idx = next((i for i, s in enumerate(ctx.steps) if "Local CI Check" in s.name), None)
                        if ci_fix_idx is not None:
                            await step_local_ci_check(ctx, settings, progress_cb, ci_commands, step_index=ci_fix_idx)

                    # Reset audit step for retry
                    ctx.steps[audit_idx] = PipelineStep(name=f"AI Audit (retry {audit_attempt + 1})")
                else:
                    raise  # Final failure

        # Step 6: Data Mining (conditional, non-fatal)
        if settings.enable_data_mining and ctx.ai_audit_passed:
            ctx.steps.append(PipelineStep(name="Data Mining"))
            try:
                await progress_cb("[6/6] Data Mining...")
                await step_data_mining(ctx, ollama, settings, progress_cb, step_index=-1)
            except Exception as exc:
                logger.warning("Data mining step failed (non-fatal): %s", exc)

        return "success", "All checks passed"

    except asyncio.CancelledError:
        return "skipped", "Cancelled by user"
    except Exception as exc:
        # Find which step failed
        for s in ctx.steps:
            if s.status == "failed":
                return "failed", f"{s.name}: {s.detail}"
        return "failed", str(exc)[:200]


# ── Internal Helpers ─────────────────────────────────────────────────────────

async def _reset_to_snapshot(project_path: str, snapshot_ref: str) -> None:
    """Reset worktree to a specific commit snapshot (safe-fail recovery)."""
    proc = await asyncio.create_subprocess_exec(
        "git", "reset", "--hard", snapshot_ref,
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(proc.communicate(), timeout=15)


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


async def _reset_worktree(project_path: str) -> None:
    """Reset worktree to clean state for retry."""
    proc = await asyncio.create_subprocess_exec(
        "git", "reset", "--hard", "origin/main",
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(proc.communicate(), timeout=15)
    proc = await asyncio.create_subprocess_exec(
        "git", "clean", "-fd",
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(proc.communicate(), timeout=15)


async def _claude_cli_review(project_path: str, prompt: str, timeout: int = 900) -> str:
    """Run Claude CLI for review, passing prompt via stdin to avoid shell length limits."""
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", "--output-format", "text",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=project_path,
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=prompt.encode()), timeout=timeout,
    )
    output = stdout.decode(errors="replace") if stdout else ""
    if proc.returncode != 0:
        err = stderr.decode(errors="replace") if stderr else ""
        logger.error("Claude CLI review failed (exit=%d): %s", proc.returncode, err[:500])
        raise RuntimeError(f"Claude CLI exit={proc.returncode}: {err[:200]}")
    if not output.strip():
        logger.error("Claude CLI review returned empty output")
        raise RuntimeError("Claude CLI returned empty response")
    return output


def _extract_review_summary(review_report: str, max_chars: int = 1500) -> str:
    """Extract key failure points from review report for Telegram display."""
    if not review_report:
        return "(no review details)"
    idx = review_report.upper().rfind("VERDICT")
    if idx > 200:
        summary = review_report[max(0, idx - 1500):idx + 200]
    else:
        summary = review_report[-max_chars:]
    return summary.strip()


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
        if s.status == "failed" and s.detail:
            detail_lines = s.detail[:500].split("\n")
            for dl in detail_lines[:5]:
                lines.append(f"      \u2514 {dl}")

    # Append review feedback when review failed
    review_step = next((s for s in ctx.steps if "Review" in s.name and s.status == "failed"), None)
    if review_step and ctx.review_report:
        summary = _extract_review_summary(ctx.review_report)
        lines.append(f"\n\U0001f4cb Review Feedback:\n{summary[:3000]}")

    return "\n".join(lines)


def format_intent_summary(ctx: PipelineContext) -> str:
    """Format an intent-centric summary for Telegram reporting."""
    lines: list[str] = []

    # Intent result
    any_failed = any(s.status == "failed" for s in ctx.steps)
    intent_verdict = "\u274c FAIL" if any_failed else "\u2705 PASS"
    lines.append(f"\U0001f3af Intent: {ctx.issue_title}")
    lines.append(f"   \u2192 {intent_verdict}")

    # Test-Gate
    ci_step = next((s for s in ctx.steps if "CI Check" in s.name), None)
    if ci_step:
        ci_verdict = "\u2705 PASS" if ci_step.status == "passed" else "\u274c FAIL"
        lines.append(f"\n\U0001f9ea Tests: {ci_verdict}")
        if ctx.ci_fix_history:
            history = " \u2192 ".join(ctx.ci_fix_history)
            lines.append(f"   \u2192 {history} \u2192 final {ci_step.status}")
        elif ci_step.status == "passed":
            lines.append("   \u2192 All tests passed on first run")
        elif ci_step.detail:
            lines.append(f"   \u2192 {ci_step.detail[:200]}")

    # AI Audit
    audit_step = next((s for s in ctx.steps if "AI Audit" in s.name), None)
    if audit_step:
        audit_verdict = "\u2705 PASS" if audit_step.status == "passed" else "\u274c FAIL"
        lines.append(f"\n\U0001f50d AI Audit: {audit_verdict}")
        if ctx.ai_audit_result:
            # Extract first few meaningful lines (skip thinking tokens)
            audit_lines = [
                l.strip() for l in ctx.ai_audit_result.split("\n")
                if l.strip() and not l.strip().startswith("<")
            ]
            summary = "\n".join(audit_lines[:3])
            if len(summary) > 300:
                summary = summary[:300] + "..."
            lines.append(f"   \u2192 {summary}")
        if ctx.audit_fix_history:
            for h in ctx.audit_fix_history:
                lines.append(f"   \u2192 {h[:200]}")

    return "\n".join(lines)


# ── Init Pipeline ────────────────────────────────────────────────────────────

@dataclass
class InitContext:
    project_name: str
    description: str
    project_path: str
    github_user: str
    repo_visibility: str
    stack_research: str = ""
    claude_md: str = ""
    agents_md: str = ""
    repo_url: str = ""
    issues_created: list[dict] = field(default_factory=list)
    steps: list[PipelineStep] = field(default_factory=list)


def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <TAG>...</TAG> markers."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


async def step_init_stack_scout(
    ctx: InitContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 0,
) -> None:
    """Step 0: Haiku CLI investigates tech stack and latest versions."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    prompt = (
        f"You are a tech stack research assistant.\n\n"
        f"Project: {ctx.project_name}\n"
        f"Description: {ctx.description}\n\n"
        f"Tasks:\n"
        f"1. Identify the optimal tech stack based on the project description.\n"
        f"2. For each technology, find the latest stable version.\n"
        f"3. List the recommended build tools, package managers, and test frameworks.\n"
        f"4. Identify the standard project structure and key configuration files.\n"
        f"5. List the build/test commands for CI.\n"
        f"6. Note any important setup steps or dependencies.\n\n"
        f"Be concise and factual. Output should be directly usable by an architect."
    )

    try:
        output = await _call_claude_cli_with_progress(
            prompt,
            model=settings.haiku_model,
            timeout=settings.research_timeout,
            progress_cb=progress_cb,
            step_name="Stack Scout",
        )
        ctx.stack_research = output
        step.status = "passed"
        step.detail = f"{len(output)} chars"
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_init_architecting(
    ctx: InitContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    reference_claude_md: str = "",
    step_index: int = 1,
) -> None:
    """Step 1: Opus CLI generates CLAUDE.md and agents.md."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    ref_section = ""
    if reference_claude_md:
        ref_section = (
            f"\n\n--- Reference CLAUDE.md (use as style/quality template) ---\n"
            f"{reference_claude_md}\n---\n"
            f"Adapt the structure and coding standards from this reference "
            f"to the new project's tech stack. Keep the senior-level quality bar.\n"
        )

    prompt = (
        f"You are a senior architect bootstrapping a new project.\n\n"
        f"Project: {ctx.project_name}\n"
        f"Description: {ctx.description}\n\n"
        f"--- Stack Research ---\n{ctx.stack_research}\n---\n"
        f"{ref_section}\n"
        f"Generate TWO files:\n\n"
        f"1. CLAUDE.md — Project-specific coding standards, architecture decisions, "
        f"file structure overview, naming conventions, testing requirements, "
        f"and build/run/test commands. This file guides all future AI development.\n"
        f"   MANDATORY: Include a 'Development Methodology' section enforcing TDD:\n"
        f"   - Always write failing tests FIRST, then implement to make them pass\n"
        f"   - Every feature/bugfix must have corresponding tests before implementation\n"
        f"   - Specify the test framework, test directory structure, and how to run tests\n"
        f"   - Define minimum test coverage expectations\n\n"
        f"   MANDATORY: Include a 'Comment Policy' section:\n"
        f"   - NO 'what' comments: Do not write comments that describe what code does.\n"
        f"     Function names, variable names, and code structure must be self-documenting.\n"
        f"   - Self-Documenting Naming: Prefer long, descriptive names over short names with comments.\n"
        f"     e.g., `isRetryLimitExceeded` instead of `isLimit // Check if retry limit is reached`.\n"
        f"   - ONLY 'why' comments: Comments are reserved for explaining WHY something exists.\n"
        f"   - Allowed: magic numbers, non-obvious business logic, architectural trade-offs,\n"
        f"     workarounds with context, and anything where the reasoning isn't obvious from code alone.\n"
        f"   - KDoc/docstring: Only on public API boundaries where callers need usage context.\n"
        f"     Skip parameter docs when names are self-explanatory.\n\n"
        f"2. agents.md — Agent configuration defining roles (architect, implementer, "
        f"reviewer) with their responsibilities and handoff protocols.\n\n"
        f"Output format (STRICT — use these exact tags):\n"
        f"<CLAUDE_MD>\n(full CLAUDE.md content)\n</CLAUDE_MD>\n"
        f"===SPLIT===\n"
        f"<AGENTS_MD>\n(full agents.md content)\n</AGENTS_MD>\n\n"
        f"Be thorough and specific to this project's tech stack."
    )

    try:
        output = await _call_claude_cli_with_progress(
            prompt,
            model=settings.opus_model,
            timeout=settings.opus_design_timeout,
            progress_cb=progress_cb,
            step_name="Architecting",
        )
        ctx.claude_md = _extract_tag(output, "CLAUDE_MD")
        ctx.agents_md = _extract_tag(output, "AGENTS_MD")

        if not ctx.claude_md:
            raise RuntimeError("Failed to extract CLAUDE_MD from Opus output")

        step.status = "passed"
        step.detail = f"CLAUDE.md: {len(ctx.claude_md)} chars, agents.md: {len(ctx.agents_md)} chars"
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_init_execution(
    ctx: InitContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 2,
) -> None:
    """Step 2: Sonnet CLI creates directories, files, git init, gh repo create."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    prompt = (
        f"You are a project scaffolding engineer. Create a new project from scratch.\n\n"
        f"Project name: {ctx.project_name}\n"
        f"Target directory: {ctx.project_path}\n"
        f"Description: {ctx.description}\n\n"
        f"--- Stack Research ---\n{ctx.stack_research}\n---\n\n"
        f"--- CLAUDE.md (write to .claude/CLAUDE.md) ---\n{ctx.claude_md}\n---\n\n"
        f"--- agents.md (write to .claude/agents.md) ---\n{ctx.agents_md}\n---\n\n"
        f"TASKS (execute ALL in order):\n"
        f"1. Create the project directory: mkdir -p {ctx.project_path}\n"
        f"2. Create .claude/ directory and write CLAUDE.md and agents.md inside it\n"
        f"3. Generate a comprehensive .gitignore based on the tech stack: include standard "
        f"patterns for languages, IDEs, OS from Stack Research. Cover: build outputs, "
        f"IDE configs (.idea/, .vscode/), OS artifacts (.DS_Store), dependency caches, secrets (.env), "
        f"and .claude/ directory (AI agent configs — must NOT be committed)\n"
        f"4. Create a README.md with project name, description, setup instructions, and tech stack\n"
        f"5. Generate a MINIMAL but BUILDABLE project skeleton based on the Stack Research. "
        f"Include all necessary config files (build configs, dependency files, etc.) "
        f"and a minimal source structure so the project can build immediately. "
        f"IMPORTANT: If the stack uses Gradle, run 'gradle wrapper' to generate gradlew. "
        f"If it uses Maven, include mvnw. CI depends on these wrappers existing.\n"
        f"   TESTING: Set up the test framework and directory structure. Include at least "
        f"one sample test that passes, so the test pipeline is verified from day one.\n"
        f"6. Create .github/workflows/ci.yml with push/PR triggers for build+test "
        f"based on the build commands from Stack Research\n"
        f"7. Run: git init && git branch -M main\n"
        f"8. Run: git add -A && git commit -m \"Initial project scaffold\"\n"
        f"9. Run: gh repo create {ctx.github_user}/{ctx.project_name} "
        f"--{ctx.repo_visibility} --source=. --remote=origin --push\n\n"
        f"SAFETY: If the directory {ctx.project_path} already exists or "
        f"gh repo create fails because the repo already exists, "
        f"STOP immediately and report the error. Do NOT overwrite.\n\n"
        f"After completion, output the GitHub repo URL."
    )

    try:
        output = await _call_claude_cli_with_progress(
            prompt,
            model=settings.sonnet_model,
            timeout=settings.init_exec_timeout,
            progress_cb=progress_cb,
            step_name="Execution",
            cwd=os.path.expanduser("~"),
            dangerously_skip_permissions=True,
        )

        # Verify directory was created
        if not os.path.isdir(ctx.project_path):
            raise RuntimeError(f"Project directory not created: {ctx.project_path}")

        # Extract GitHub URL
        url_match = re.search(r"https://github\.com/[^\s\"'<>]+", output)
        if url_match:
            ctx.repo_url = url_match.group(0).rstrip(".,;)")

        step.status = "passed"
        step.detail = ctx.repo_url or "dir created (no URL extracted)"
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def _poll_ci_status(repo: str, timeout: int) -> tuple[str, str]:
    """Poll GitHub Actions until the latest run completes or timeout. Returns (status, log)."""
    deadline = time.monotonic() + timeout

    # Wait a bit for the run to appear
    await asyncio.sleep(10)

    while time.monotonic() < deadline:
        proc = await asyncio.create_subprocess_exec(
            "gh", "run", "list", "-R", repo, "--limit", "1",
            "--json", "status,conclusion,databaseId",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if not stdout:
            await asyncio.sleep(15)
            continue

        runs = json.loads(stdout.decode())
        if not runs:
            await asyncio.sleep(15)
            continue

        run = runs[0]
        status = run.get("status", "")
        conclusion = run.get("conclusion", "")
        run_id = run.get("databaseId", "")

        if status == "completed":
            if conclusion == "success":
                return "success", ""
            # Fetch failed log
            log_proc = await asyncio.create_subprocess_exec(
                "gh", "run", "view", str(run_id), "-R", repo, "--log-failed",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            log_out, _ = await asyncio.wait_for(log_proc.communicate(), timeout=30)
            log_text = log_out.decode(errors="replace") if log_out else "(no log)"
            # Truncate to last 3000 chars
            if len(log_text) > 3000:
                log_text = log_text[-3000:]
            return "failed", log_text

        await asyncio.sleep(15)

    return "timeout", "CI did not complete within timeout"


async def step_init_ci_watch(
    ctx: InitContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 3,
) -> None:
    """Step 2.5: Wait for CI, auto-fix with Sonnet if it fails."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    repo = f"{ctx.github_user}/{ctx.project_name}"
    max_retries = settings.init_ci_fix_retries

    try:
        for attempt in range(max_retries + 1):
            attempt_label = f" (attempt {attempt + 1})" if attempt > 0 else ""
            await progress_cb(f"CI Watch{attempt_label} — waiting for GitHub Actions...")

            ci_status, ci_log = await _poll_ci_status(repo, settings.init_ci_watch_timeout)

            if ci_status == "success":
                step.status = "passed"
                step.detail = f"CI passed{attempt_label}"
                return

            if ci_status == "timeout":
                step.status = "skipped"
                step.detail = "CI did not complete in time — skipping"
                return

            # CI failed
            if attempt >= max_retries:
                step.status = "failed"
                step.detail = f"CI failed after {attempt + 1} fix attempts"
                raise RuntimeError(step.detail)

            # Auto-fix with Sonnet
            await progress_cb(f"CI Watch — fixing CI failure (attempt {attempt + 1}/{max_retries})...")

            fix_prompt = (
                f"The GitHub Actions CI for this project has FAILED.\n\n"
                f"--- CI Failure Log ---\n{ci_log}\n---\n\n"
                f"Fix the issue so CI passes. Common causes:\n"
                f"- Missing gradle wrapper (run: gradle wrapper)\n"
                f"- Missing dependencies or config files\n"
                f"- Incorrect build commands in .github/workflows/ci.yml\n"
                f"- Missing source files referenced in build config\n\n"
                f"After fixing, commit and push:\n"
                f"git add -A && git commit -m \"fix: CI failure\" && git push"
            )

            await _call_claude_cli_with_progress(
                fix_prompt,
                model=settings.sonnet_model,
                timeout=settings.init_exec_timeout,
                progress_cb=progress_cb,
                step_name=f"CI Fix (attempt {attempt + 1})",
                cwd=ctx.project_path,
                dangerously_skip_permissions=True,
            )

    except RuntimeError:
        raise
    except Exception as exc:
        step.status = "failed"
        step.detail = str(exc)[:200]
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


async def step_init_issue_planning(
    ctx: InitContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    step_index: int = 4,
) -> None:
    """Step 4: Opus CLI generates GitHub issues as JSON, then creates them via gh."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()

    prompt = (
        f"You are a project manager planning the initial development roadmap.\n\n"
        f"Project: {ctx.project_name}\n"
        f"Description: {ctx.description}\n\n"
        f"--- Stack Research ---\n{ctx.stack_research}\n---\n\n"
        f"--- CLAUDE.md ---\n{ctx.claude_md}\n---\n\n"
        f"Create 5-10 GitHub issues that break down the initial implementation into "
        f"manageable, well-scoped tasks. Each issue should be solvable by an AI agent "
        f"in a single session.\n\n"
        f"Output a JSON array (no markdown fences, no explanation, ONLY the JSON):\n"
        f'[{{"title": "...", "body": "...", "labels": ["..."]}}]\n\n'
        f"Rules:\n"
        f"- Each issue body should include clear acceptance criteria\n"
        f"- Each issue MUST include a 'Test Requirements' section specifying what tests "
        f"to write BEFORE implementation (TDD). Example: 'Write tests for X, Y, Z first, "
        f"then implement to make them pass.'\n"
        f"- Order issues by dependency (foundational first)\n"
        f"- Labels should be relevant (e.g., 'setup', 'feature', 'testing', 'ci')\n"
        f"- Issues should cover: core features, testing, CI/CD, documentation\n"
        f"- Be specific about file paths and implementation details"
    )

    try:
        output = await _call_claude_cli_with_progress(
            prompt,
            model=settings.opus_model,
            timeout=settings.init_issue_planning_timeout,
            progress_cb=progress_cb,
            step_name="Issue Planning",
        )

        # Extract JSON array from output (robust: find first [ ... last ])
        cleaned = re.sub(r"```(?:json)?\s*\n?", "", output)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

        # Try direct parse first, then bracket extraction
        issues = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                issues = parsed
        except json.JSONDecodeError:
            pass

        if issues is None:
            # Find JSON array by bracket matching
            start_idx = cleaned.find("[")
            end_idx = cleaned.rfind("]")
            if start_idx != -1 and end_idx > start_idx:
                try:
                    issues = json.loads(cleaned[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    pass

        if not issues or not isinstance(issues, list):
            raise RuntimeError(f"Could not extract JSON array from Opus output ({len(output)} chars)")

        # Create ai-managed label (ignore if exists)
        label_proc = await asyncio.create_subprocess_exec(
            "gh", "label", "create", "ai-managed",
            "--color", "7057ff",
            "--description", "Auto-generated by AI orchestrator",
            "-R", f"{ctx.github_user}/{ctx.project_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(label_proc.communicate(), timeout=15)

        created = 0
        total = len(issues)
        for issue in issues:
            title = issue.get("title", "")
            body = issue.get("body", "")
            labels = issue.get("labels", [])
            if not title:
                continue

            # Always add ai-managed label
            if "ai-managed" not in labels:
                labels.append("ai-managed")

            cmd = [
                "gh", "issue", "create",
                "-R", f"{ctx.github_user}/{ctx.project_name}",
                "--title", title,
                "--body", body,
            ]
            for label in labels:
                cmd.extend(["--label", label])

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode == 0:
                    url = stdout.decode().strip() if stdout else ""
                    ctx.issues_created.append({"title": title, "url": url, "labels": labels})
                    created += 1
                else:
                    logger.warning("Failed to create issue '%s': %s", title, stderr.decode()[:200])
            except Exception as exc:
                logger.warning("Failed to create issue '%s': %s", title, exc)

        step.status = "passed"
        step.detail = f"{created}/{total} issues created"
    except Exception as exc:
        step.status = "skipped"
        step.detail = f"Non-fatal: {str(exc)[:180]}"
        logger.warning("Issue planning failed (non-fatal): %s", exc)
    finally:
        step.elapsed_sec = time.monotonic() - start


# ── Plan & Discuss Pipeline Functions ────────────────────────────────────────


def _extract_json_array(text: str) -> list[dict] | None:
    """Extract a JSON array from LLM output with multiple fallback strategies."""
    # 1. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    # 2. Try direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3. Bracket matching (first [ to last ])
    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx != -1 and end_idx > start_idx:
        try:
            parsed = json.loads(cleaned[start_idx:end_idx + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # 4. Balanced bracket extraction — find the [ that properly closes
    if start_idx != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start_idx, len(cleaned)):
            ch = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[start_idx:i + 1])
                        if isinstance(parsed, list):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    logger.warning("JSON extraction failed. First 500 chars: %s", cleaned[:500])
    return None


async def _ensure_labels_and_create_issues(
    issues: list[dict],
    github_user: str,
    project_name: str,
) -> tuple[int, int, list[dict]]:
    """Ensure all labels exist and create GitHub issues. Returns (created, total, issues_list)."""
    # Collect all unique labels and ensure they exist on the repo
    all_labels: set[str] = {"ai-managed"}
    for issue in issues:
        for label in issue.get("labels", []):
            all_labels.add(label)

    repo = f"{github_user}/{project_name}"
    for label in all_labels:
        try:
            lp = await asyncio.create_subprocess_exec(
                "gh", "label", "create", label,
                "--color", "7057ff",
                "--description", "Auto-generated by AI orchestrator",
                "-R", repo,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(lp.communicate(), timeout=15)
        except Exception:
            pass  # label may already exist

    created = 0
    total = len(issues)
    issues_list: list[dict] = []
    for issue in issues:
        title = issue.get("title", "")
        body = issue.get("body", "")
        labels = issue.get("labels", [])
        if not title:
            continue

        if "ai-managed" not in labels:
            labels.append("ai-managed")

        cmd = [
            "gh", "issue", "create",
            "-R", repo,
            "--title", title,
            "--body", body,
        ]
        for label in labels:
            cmd.extend(["--label", label])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                url = stdout.decode().strip() if stdout else ""
                issues_list.append({"title": title, "url": url, "labels": labels})
                created += 1
            else:
                logger.warning("Failed to create issue '%s': %s", title, stderr.decode()[:200])
        except Exception as exc:
            logger.warning("Failed to create issue '%s': %s", title, exc)

    return created, total, issues_list


async def step_plan_issues(
    project_name: str,
    project_path: str,
    github_user: str,
    existing_issues_text: str,
    claude_md: str,
    settings: Settings,
    progress_cb: ProgressCallback,
    ollama: OllamaProvider,
) -> tuple[int, int, list[dict]]:
    """Plan next-stage issues for an existing project. Returns (created, total, issues_list)."""

    system_prompt = (
        "You are a project manager planning the NEXT development stage. "
        "Analyze the existing backlog keywords and imagine the Next Stage architecture "
        "after those tasks are completed. Plan a roadmap-level set of issues that reflect "
        "the project's evolution direction — NOT simple feature listings.\n\n"
        "Output a JSON array (no markdown fences, no explanation, ONLY the JSON):\n"
        '[{"title": "...", "body": "...", "labels": ["..."]}]\n\n'
        "Rules:\n"
        "- Each issue body should include clear acceptance criteria\n"
        "- Each issue MUST include a 'Test Requirements' section specifying what tests "
        "to write BEFORE implementation (TDD). Example: 'Write tests for X, Y, Z first, "
        "then implement to make them pass.'\n"
        "- Order issues by dependency (foundational first)\n"
        "- Labels should be relevant (e.g., 'enhancement', 'feature', 'testing', 'refactor')\n"
        "- Do NOT duplicate existing issues\n"
        "- Be specific about file paths and implementation details"
    )

    user_content = (
        f"Project: {project_name}\n\n"
        f"--- CLAUDE.md ---\n{claude_md}\n---\n\n"
        f"--- Existing Issues (open + closed) ---\n{existing_issues_text}\n---\n\n"
        f"Create 5-10 GitHub issues for the next development phase."
    )

    messages = [Message(role=Role.USER, content=user_content)]

    response = await _call_ollama_with_progress(
        ollama, messages,
        max_tokens=16384,
        temperature=0.4,
        system_prompt=system_prompt,
        timeout=settings.plan_timeout,
        progress_cb=progress_cb,
        step_name="Issue Planning",
        model=settings.qwen_model,
        num_ctx=32768,
    )

    issues = _extract_json_array(response.content)
    if not issues:
        raise RuntimeError(f"Could not extract JSON array from Qwen output ({len(response.content)} chars)")

    return await _ensure_labels_and_create_issues(issues, github_user, project_name)


async def step_discuss_consult(
    project_name: str,
    claude_md: str,
    file_tree: str,
    build_context: str,
    question: str,
    settings: Settings,
    progress_cb: ProgressCallback,
    ollama: OllamaProvider,
) -> str:
    """Qwen tech-lead consultation. Returns response text."""

    system_prompt = (
        "You are a senior tech lead providing technical consulting for this project. "
        "Provide a direct, opinionated answer with:\n"
        "- Concrete recommendations\n"
        "- Code examples where helpful\n"
        "- Trade-offs analysis\n"
        "- Clear next steps\n\n"
        "Be decisive and specific. Do not hedge unnecessarily."
    )

    user_content = (
        f"Project: {project_name}\n\n"
        f"--- CLAUDE.md ---\n{claude_md}\n---\n\n"
        f"--- Project File Tree ---\n{file_tree}\n---\n\n"
        f"--- Build Context (libraries, versions) ---\n{build_context}\n---\n\n"
        f"Question: {question}"
    )

    messages = [Message(role=Role.USER, content=user_content)]

    response = await _call_ollama_with_progress(
        ollama, messages,
        max_tokens=16384,
        temperature=0.7,
        system_prompt=system_prompt,
        timeout=settings.discuss_timeout,
        progress_cb=progress_cb,
        step_name="Tech Consultation",
        model=settings.qwen_model,
        num_ctx=32768,
    )

    return response.content


async def step_discuss_to_issues(
    project_name: str,
    github_user: str,
    project_path: str,
    question: str,
    discussion_text: str,
    settings: Settings,
    progress_cb: ProgressCallback,
    ollama: OllamaProvider,
) -> tuple[int, int, list[dict]]:
    """Convert discussion into 1-3 GitHub issues. Returns (created, total, issues_list)."""

    system_prompt = (
        "Based on the following technical discussion, extract the key action items "
        "and create 1-3 focused GitHub issues.\n\n"
        "Output a JSON array (no markdown fences, no explanation, ONLY the JSON):\n"
        '[{"title": "...", "body": "...", "labels": ["..."]}]\n\n'
        "Rules:\n"
        "- Each issue should be a concrete, actionable task\n"
        "- Each issue body should include acceptance criteria\n"
        "- Each issue MUST include a 'Test Requirements' section (TDD)\n"
        "- Labels should be relevant\n"
        "- 1-3 issues maximum — focus on the most impactful action items"
    )

    user_content = (
        f"--- Original Question ---\n{question}\n---\n\n"
        f"--- Discussion ---\n{discussion_text}\n---"
    )

    messages = [Message(role=Role.USER, content=user_content)]

    response = await _call_ollama_with_progress(
        ollama, messages,
        max_tokens=8192,
        temperature=0.4,
        system_prompt=system_prompt,
        timeout=settings.discuss_issue_timeout,
        progress_cb=progress_cb,
        step_name="Discussion → Issues",
        model=settings.qwen_model,
        num_ctx=16384,
    )

    issues = _extract_json_array(response.content)
    if not issues:
        raise RuntimeError(f"Could not extract JSON array from Qwen output ({len(response.content)} chars)")

    return await _ensure_labels_and_create_issues(issues, github_user, project_name)


async def run_init_pipeline(
    ctx: InitContext,
    settings: Settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback,
    reference_claude_md: str = "",
) -> tuple[str, str]:
    """Run the 5-step init pipeline. Returns (status, detail)."""
    ctx.steps = [
        PipelineStep(name="Stack Scout (Haiku)"),
        PipelineStep(name="Architecting (Opus)"),
        PipelineStep(name="Execution (Sonnet)"),
        PipelineStep(name="CI Watch (Sonnet)"),
        PipelineStep(name="Issue Planning (Opus)"),
    ]

    step_funcs = [
        lambda: step_init_stack_scout(ctx, settings, progress_cb, step_index=0),
        lambda: step_init_architecting(ctx, settings, progress_cb, reference_claude_md, step_index=1),
        lambda: step_init_execution(ctx, settings, progress_cb, step_index=2),
        lambda: step_init_ci_watch(ctx, settings, progress_cb, step_index=3),
        lambda: step_init_issue_planning(ctx, settings, progress_cb, step_index=4),
    ]

    for i, func in enumerate(step_funcs):
        if cancel_event.is_set():
            for j in range(i, len(ctx.steps)):
                ctx.steps[j].status = "skipped"
                ctx.steps[j].detail = "Cancelled"
            return "skipped", "Cancelled by user"

        try:
            await func()
        except Exception as exc:
            # All steps are fatal
            for j in range(i + 1, len(ctx.steps)):
                ctx.steps[j].status = "skipped"
            return "failed", f"Step {i} ({ctx.steps[i].name}) failed: {str(exc)[:200]}"

    return "success", f"Project created: {ctx.repo_url}"
