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

import httpx

from .ai.base import AIResponse, Message, Role
from .ai.gemini_provider import GeminiCLIProvider
from .ai.ollama_provider import OllamaProvider
from .config import PIPELINE_MODES, Settings
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
    # Adaptive pipeline fields
    mode: str = "standard"  # "express" | "standard" | "full"
    triage_reason: str = ""  # Haiku triage reasoning
    split_plan: str = ""     # Opus split proposal (empty if not needed)
    steps: list[PipelineStep] = field(default_factory=list)
    # Supreme Court fields
    supreme_court_ruling: str = ""
    draft_context_diff: str = ""
    predecessor_issue_num: int = 0


# ── Exception Detail Helper ──────────────────────────────────────────────────

def _exc_detail(exc: Exception, max_chars: int = 200) -> str:
    """Return a human-readable exception detail, never empty."""
    msg = str(exc).strip()
    if msg:
        return msg[:max_chars]
    # Fallback: type name + repr for exceptions with empty str() (e.g. TimeoutError, CancelledError)
    type_name = type(exc).__name__
    return f"{type_name}: {repr(exc)}"[:max_chars]


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


def parse_ruling(text: str) -> str:
    """Extract [RULING: UPHOLD/OVERTURN/REDESIGN] from Supreme Court output."""
    match = re.search(r"\[RULING:\s*(UPHOLD|OVERTURN|REDESIGN)\]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    text_upper = text.upper()
    if "OVERTURN" in text_upper:
        return "OVERTURN"
    if "REDESIGN" in text_upper:
        return "REDESIGN"
    if "UPHOLD" in text_upper:
        return "UPHOLD"
    return "UPHOLD"


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
        return ["./gradlew ktlintFormat", "./gradlew check"]

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
        re.compile(r"^.*[Ee]rror[:\s].*$", re.MULTILINE),    # Generic errors
        re.compile(r"^E\s+.*$", re.MULTILINE),               # pytest assertion details
        re.compile(r"^.*couldn't be completed.*$", re.MULTILINE | re.IGNORECASE),  # macOS system
        re.compile(r"^.*not found.*$", re.MULTILINE | re.IGNORECASE),  # command not found
        re.compile(r"^.*Unable to .*$", re.MULTILINE),       # Unable to locate...
        re.compile(r"^.*exception.*$", re.MULTILINE | re.IGNORECASE),  # exceptions
    ]
    failures: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in pat.finditer(log):
            line = m.group(0).strip()
            if line and line not in seen:
                seen.add(line)
                failures.append(line)
    if not failures:
        # Fallback: show last non-empty lines of log
        tail_lines = [l.strip() for l in log.strip().split("\n") if l.strip()]
        failures = tail_lines[-5:] if tail_lines else ["(empty CI log)"]
    result = "\n".join(failures)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n...(truncated)"
    return result


# ── Adaptive Pipeline Step Builder ──────────────────────────────────────────

def build_fivebrid_steps(mode: str, ci_commands: list[str]) -> list[PipelineStep]:
    """Build pipeline step list based on adaptive mode."""
    mode_config = PIPELINE_MODES.get(mode, PIPELINE_MODES["standard"])
    allowed = mode_config["steps"]  # None = all

    all_steps = [
        "Haiku Research",
        "Opus Design",
        "Gemini Design Critique",
        "Qwen Hints",
        "Sonnet Implement",
    ]
    if ci_commands:
        all_steps.append("Local CI Check")
    all_steps += [
        "Sonnet Self-Review",
        "Gemini Cross-Review",
        "AI Audit",
    ]

    if allowed is None:
        return [PipelineStep(name=n) for n in all_steps]

    return [
        PipelineStep(name=n) for n in all_steps
        if n in allowed and (n != "Local CI Check" or ci_commands)
    ]


# ── Triage & Split Detection ───────────────────────────────────────────────

async def step_triage_and_split(
    ctx: PipelineContext,
    settings: Settings,
    progress_cb: ProgressCallback,
) -> dict:
    """Triage issue complexity and detect if splitting is needed.

    Returns dict with:
        mode: "express" | "standard" | "full"
        reason: str (Haiku's reasoning)
        estimated_files: str
        split_needed: bool
        split_plan: str (Opus's sub-issue proposal, empty if not needed)
        sub_issues: list[dict] (parsed sub-issues: [{title, body}])
    """
    # ── Phase 1: Fetch issue ──
    if not ctx.issue_body:
        await progress_cb("Fetching issue...")
        await step_fetch_issue(ctx)

    # ── Phase 2: Haiku Triage ──
    await progress_cb("Analyzing complexity (Haiku)...")
    triage_prompt = (
        "Analyze this GitHub issue and classify its implementation complexity.\n\n"
        f"Title: {ctx.issue_title}\n"
        f"Body:\n{ctx.issue_body[:3000]}\n\n"
        "Rules:\n"
        "- EXPRESS: Typo fix, config change, dependency bump, simple rename, "
        "one-line fix, README update, version bump\n"
        "- STANDARD: New feature, bug fix requiring investigation, refactoring, "
        "API change, test additions\n"
        "- FULL: Architecture change, security-critical fix, multi-system integration, "
        "breaking change, performance optimization requiring multiple files\n\n"
        "IMPORTANT: If in doubt between EXPRESS and STANDARD, choose STANDARD.\n"
        "Domain-specific logic (matching algorithms, scoring, business rules) is always STANDARD+.\n\n"
        "Reply in this EXACT format:\n"
        "MODE: EXPRESS|STANDARD|FULL\n"
        "REASON: <one sentence why>\n"
        "FILES: <comma-separated list of likely files to modify>"
    )
    try:
        triage_output = await _call_claude_cli_with_progress(
            triage_prompt,
            model=settings.haiku_model,
            timeout=settings.triage_timeout,
            progress_cb=progress_cb,
            step_name="Triage",
        )
        mode = "standard"
        reason = ""
        estimated_files = ""
        for line in triage_output.strip().splitlines():
            if line.startswith("MODE:"):
                val = line.split(":", 1)[1].strip().upper()
                if "EXPRESS" in val:
                    mode = "express"
                elif "FULL" in val:
                    mode = "full"
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
            elif line.startswith("FILES:"):
                estimated_files = line.split(":", 1)[1].strip()
    except Exception:
        mode, reason, estimated_files = "standard", "Triage failed, defaulting to standard", ""

    ctx.triage_reason = reason

    # ── Phase 3: Opus Split Detection (standard/full only) ──
    split_needed = False
    split_plan = ""
    sub_issues: list[dict] = []

    if mode != "express":
        await progress_cb("Checking if issue should be split (Opus)...")
        split_prompt = (
            "You are an expert software architect. Analyze this GitHub issue "
            "and determine if it should be split into smaller, independently "
            "solvable sub-issues.\n\n"
            f"Title: {ctx.issue_title}\n"
            f"Body:\n{ctx.issue_body[:4000]}\n\n"
            "Rules:\n"
            "- Only split if the issue clearly contains 2+ independent tasks\n"
            "- Each sub-issue must be independently implementable and testable\n"
            "- Do NOT split if the tasks are tightly coupled\n"
            "- Max 5 sub-issues\n\n"
            "If splitting is NOT needed, reply: SPLIT: NO\n\n"
            "If splitting IS needed, reply in this format:\n"
            "SPLIT: YES\n"
            "---\n"
            "TITLE: <sub-issue 1 title>\n"
            "BODY: <sub-issue 1 description>\n"
            "---\n"
            "TITLE: <sub-issue 2 title>\n"
            "BODY: <sub-issue 2 description>\n"
            "---"
        )
        try:
            split_output = await _call_claude_cli_with_progress(
                split_prompt,
                model=settings.opus_model,
                timeout=settings.split_timeout,
                progress_cb=progress_cb,
                step_name="Split Analysis",
            )
            if "SPLIT: YES" in split_output.upper():
                split_needed = True
                split_plan = split_output
                # Parse sub-issues
                blocks = split_output.split("---")
                for block in blocks:
                    title_match = re.search(r"TITLE:\s*(.+)", block)
                    body_match = re.search(r"BODY:\s*(.+)", block, re.DOTALL)
                    if title_match and body_match:
                        sub_issues.append({
                            "title": title_match.group(1).strip(),
                            "body": body_match.group(1).strip(),
                        })
        except Exception:
            logger.warning("Split analysis failed, proceeding without split")

    ctx.split_plan = split_plan

    return {
        "mode": mode,
        "reason": reason,
        "estimated_files": estimated_files,
        "split_needed": split_needed,
        "split_plan": split_plan,
        "sub_issues": sub_issues,
    }


async def step_local_ci_check(
    ctx: PipelineContext,
    settings: Settings,
    progress_cb: ProgressCallback,
    ci_commands: list[str],
    step_index: int,
    max_retries_override: int | None = None,
) -> None:
    """Run local CI commands and auto-fix with Sonnet on failure."""
    step = ctx.steps[step_index]
    step.status = "running"
    start = time.monotonic()
    max_retries = max_retries_override if max_retries_override is not None else settings.local_ci_fix_retries

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

    # Fallback: if filtered diff is empty, check unfiltered diff
    if not diff.strip():
        raw_proc = await asyncio.create_subprocess_exec(
            "git", "diff", diff_range,
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        raw_out, _ = await asyncio.wait_for(raw_proc.communicate(), timeout=30)
        raw_diff = raw_out.decode(errors="replace") if raw_out else ""
        if raw_diff.strip():
            logger.warning(
                "Filtered diff was empty but raw diff has %d chars — "
                "exclude patterns may be too broad", len(raw_diff),
            )
            diff = raw_diff

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
    try:
        return task.result()
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
        raise type(exc)(f"Ollama {step_name} timed out after {elapsed}s") from exc
    except httpx.HTTPStatusError:
        raise
    except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
        raise type(exc)(f"Ollama {step_name} cancelled/timed out after {elapsed}s") from exc


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
    try:
        return task.result()
    except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
        raise type(exc)(f"Gemini {step_name} cancelled/timed out after {elapsed}s") from exc


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
        f"\n\n--- Design Guide (from Opus) ---\n{ctx.design_doc}\n---\n"
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

    if ctx.draft_context_diff:
        prompt += f"\n\n{ctx.draft_context_diff}\n"

    if ctx.review_feedback:
        prompt += (
            f"\n\n--- Previous Review Feedback (MUST address these issues) ---\n"
            f"{ctx.review_feedback}\n---\n\n"
            f"The previous implementation was rejected. Fix ALL issues mentioned above."
        )

    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", "--dangerously-skip-permissions",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=ctx.project_path,
    )
    proc.stdin.write(prompt.encode())  # type: ignore[union-attr]
    proc.stdin.close()  # type: ignore[union-attr]
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
            tail = collected[-500:].strip() if collected else "(no output)"
            logger.warning("Sonnet produced no changes. Output tail:\n%s", tail)
            step.detail = f"No changes produced. Output: {tail[:300]}"
            step.elapsed_sec = time.monotonic() - start
            raise RuntimeError("Claude produced no code changes")

        step.status = "passed"
        step.detail = f"exit=0, diff={len(ctx.git_diff)} chars"
    except (asyncio.CancelledError, TimeoutError, RuntimeError):
        raise
    except Exception as exc:
        step.status = "failed"
        step.detail = _exc_detail(exc)
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
        "You are an adversarial code auditor with the personality of a brutally honest "
        "senior developer. Your job is to FIND FLAWS, not to praise. "
        "Assume the implementer made mistakes. Think like an attacker exploiting this code.\n\n"
        "When you find 'what' comments (comments that describe what the code does instead of why), "
        "mock them like a senior developer would: 'Oh great, // increment counter — "
        "I never would have guessed that i++ increments a counter. Delete this noise.'"
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
        "5. COMMENT AUDIT (MANDATORY — zero tolerance for 'what' comments):\n"
        "   Examine EVERY comment in the diff. For each comment:\n"
        "   - If it describes WHAT the code does → [MAJOR] violation. "
        "     Provide a sarcastic correction explaining why this comment is noise. "
        "     The code should be self-documenting with clear naming.\n"
        "   - If it explains WHY (design decision, non-obvious constraint, "
        "     business rule, workaround reason) → ACCEPTABLE.\n"
        "   - If no comments exist and the code is self-explanatory → ACCEPTABLE.\n"
        "   - Examples of INEXCUSABLE comments: '// increment counter', '// set user name', "
        "'// loop through items', '// check if null', '// create instance', '// return result'\n"
        "   - Examples of GOOD comments: '// OAuth spec requires nonce per-request', "
        "'// Workaround for SQLite 3.x locking bug', '// Business rule: 30-day retention'\n\n"
        "For each finding, classify as:\n"
        "- [CRITICAL]: Will cause runtime crash, data loss, or security breach in production\n"
        "- [MAJOR]: Clearly violates the issue requirements OR introduces a real bug\n"
        "- [MINOR]: Best practice suggestion, style preference, or 'nice to have'\n\n"
        "IMPORTANT classification rules:\n"
        "- Missing validation is [MINOR] unless the issue EXPLICITLY requires it\n"
        "- Alternative API design choices (query param vs body) are [MINOR]\n"
        "- If the code works correctly and tests pass, do NOT mark working design choices as [MAJOR]\n"
        "- 'Could be improved' is ALWAYS [MINOR], never [MAJOR]\n\n"
        "End with exactly one of:\n"
        "[AUDIT: PASS] — no critical or major issues found\n"
        "[AUDIT: FAIL] — critical or major issues found that MUST be fixed"
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
        step.detail = _exc_detail(exc)
        raise
    finally:
        step.elapsed_sec = time.monotonic() - start


# ── Supreme Court ────────────────────────────────────────────────────────

async def step_supreme_court(
    ctx: PipelineContext,
    gemini: GeminiCLIProvider,
    settings: Settings,
    progress_cb: ProgressCallback,
) -> str:
    """Gemini mediates when Sonnet (PASS) conflicts with DeepSeek (FAIL).

    Returns ruling: "UPHOLD" | "OVERTURN" | "REDESIGN"
    """
    prompt = (
        "You are a Supreme Court judge mediating a conflict between two AI reviewers.\n\n"
        f"## Issue\n#{ctx.issue_num}: {ctx.issue_title}\n{ctx.issue_body[:2000]}\n\n"
        f"## Implementation Diff\n{ctx.git_diff[:4000]}\n\n"
        f"## Self-Review (Sonnet) — VERDICT: PASS\n{ctx.self_review_report[:2000]}\n\n"
        f"## AI Audit (DeepSeek R1) — VERDICT: FAIL\n{ctx.ai_audit_result[:2000]}\n\n"
        "Analyze both reviews carefully:\n"
        "1. Is the Audit's FAIL justified? Are the critical issues real?\n"
        "2. Is the Self-Review's PASS justified? Did it miss real issues?\n"
        "3. Could both be partially right?\n\n"
        "CRITICAL INSTRUCTION: You MUST conclude with exactly ONE ruling. "
        "Do NOT hedge or say 'both have valid points without choosing'. "
        "Even if both sides have merit, you must pick the BEST option.\n\n"
        "Reply with exactly one of:\n"
        "[RULING: UPHOLD] — Audit is correct, implementation needs fixes\n"
        "[RULING: OVERTURN] — Self-review is correct, audit findings are false positives\n"
        "[RULING: REDESIGN] — Both reviews expose fundamental design issues\n\n"
        "Then explain your reasoning in 2-3 sentences."
    )

    try:
        output = await gemini.generate(prompt, timeout=settings.supreme_court_timeout)
        ctx.supreme_court_ruling = output
        return parse_ruling(output)
    except Exception as exc:
        logger.warning("Supreme Court failed: %s", exc)
        ctx.supreme_court_ruling = f"Error: {exc}"
        return "UPHOLD"


# ── State Sync Helpers ───────────────────────────────────────────────────

async def _extract_decisions_llm(
    ctx: PipelineContext,
    settings: Settings,
    progress_cb: ProgressCallback,
) -> list[str]:
    """Extract key architectural decisions from design doc using Haiku."""
    if not ctx.design_doc:
        return []
    try:
        prompt = (
            "Extract the key architectural/design DECISIONS from this design document. "
            "Focus on WHY choices were made (library selection, pattern choice, tradeoffs).\n"
            "Return 3-7 bullet points, each starting with '-'.\n"
            "Only include actual decisions, not descriptions of what the code does.\n\n"
            f"{ctx.design_doc[:3000]}"
        )
        output = await _call_claude_cli_with_progress(
            prompt,
            model=settings.haiku_model,
            timeout=30,
            progress_cb=progress_cb,
            step_name="State Sync",
        )
        decisions = []
        for line in output.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                decisions.append(stripped[2:])
        return decisions[:7]
    except Exception:
        decisions = []
        for line in ctx.design_doc.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") and any(
                kw in stripped.lower()
                for kw in ("chose", "decided", "approach", "pattern", "strategy", "because", "using", "instead of")
            ):
                decisions.append(stripped[2:])
        return decisions[:10]


def _extract_files_from_diff(diff: str) -> list[str]:
    """Extract file paths from git diff output."""
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:])
    return files


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
        step.detail = _exc_detail(exc)
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

    if ctx.draft_context_diff:
        prompt += f"\n\n{ctx.draft_context_diff}\n"

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
        step.detail = _exc_detail(exc)
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
            step.status = "revised"
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
    resume_from_step: int = -1,
    scheduler: "StaggeredScheduler | None" = None,
    supreme_court_cb: Callable | None = None,
) -> tuple[str, str]:
    """Run the adaptive Five-brid pipeline. Returns (status, detail)."""
    from .scheduler import StaggeredScheduler  # noqa: F811

    # Resolve CI commands once at start
    ci_commands = _resolve_ci_commands(ctx.project_path, project_info) if settings.local_ci_enabled else []

    # Initialize steps based on adaptive mode
    if not ctx.steps:
        ctx.steps = build_fivebrid_steps(ctx.mode, ci_commands)

    # Mode-specific config overrides
    mode_config = PIPELINE_MODES.get(ctx.mode, PIPELINE_MODES["standard"])
    effective_max_design_retries = mode_config["max_design_retries"]
    effective_ai_audit_max_retries = mode_config["ai_audit_max_retries"]
    effective_ai_audit_enabled = mode_config["ai_audit_enabled"]
    effective_ci_fix_retries = mode_config["local_ci_fix_retries"]

    resuming = resume_from_step >= 0

    def _step_done(idx: int) -> bool:
        return _is_step_done(ctx, idx, resuming)

    def _find_step(name: str) -> int | None:
        return next((i for i, s in enumerate(ctx.steps) if s.name == name), None)

    try:
        # Fetch issue (skip if already populated by triage or checkpoint)
        if ctx.issue_body:
            if resuming:
                await progress_cb("Issue data restored from checkpoint")
        else:
            await progress_cb("Fetching issue...")
            await step_fetch_issue(ctx)

        total_steps = len(ctx.steps)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # ── Phase 0: Research ──
        research_idx = _find_step("Haiku Research")
        if research_idx is not None:
            if _step_done(research_idx):
                await progress_cb(f"[{research_idx}/{total_steps}] Haiku Research (cached)")
            else:
                await progress_cb(f"[{research_idx}/{total_steps}] Haiku Research...")
                await step_haiku_research(ctx, settings, progress_cb, step_index=research_idx)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

        # ── Inject State Context from prior issues ──
        from .state_sync import format_state_context
        state_ctx = format_state_context(ctx.project_path)
        if state_ctx:
            ctx.draft_context_diff = state_ctx

        # ── Phase A: Design Loop ──
        design_idx = _find_step("Opus Design")
        if design_idx is not None:
            if not resuming or not _step_done(design_idx):
                for iteration in range(effective_max_design_retries + 1):
                    ctx.design_iteration = iteration

                    if cancel_event.is_set():
                        return "skipped", "Cancelled by user"

                    # Step: Opus Design
                    step_label = f" (iteration {iteration + 1})" if iteration > 0 else ""
                    await progress_cb(f"[{design_idx}/{total_steps}] Opus Design{step_label}...")

                    if iteration > 0:
                        critique_base = _find_step("Gemini Design Critique")
                        if critique_base is not None:
                            insert_at = critique_base + 1 + (iteration - 1) * 2
                        else:
                            insert_at = design_idx + 1 + (iteration - 1) * 2
                        ctx.steps.insert(insert_at, PipelineStep(name=f"Opus Design (retry {iteration})"))
                        ctx.steps.insert(insert_at + 1, PipelineStep(name=f"Gemini Critique (retry {iteration})"))
                        total_steps = len(ctx.steps)

                    current_design_idx = design_idx + (iteration * 2)
                    await step_opus_design(ctx, settings, progress_cb, step_index=current_design_idx)

                    if cancel_event.is_set():
                        return "skipped", "Cancelled by user"

                    # Gemini Design Critique
                    critique_idx = _find_step("Gemini Design Critique")
                    if critique_idx is not None:
                        current_critique_idx = current_design_idx + 1
                        await progress_cb(f"[{current_critique_idx}/{total_steps}] Gemini Design Critique{step_label}...")
                        await step_gemini_design_critique(
                            ctx, gemini, settings, progress_cb, step_index=current_critique_idx,
                        )

                        critique_step = ctx.steps[current_critique_idx]
                        if critique_step.status == "revised" and "NEEDS_REVISION" in critique_step.detail:
                            if iteration < effective_max_design_retries:
                                await progress_cb(
                                    f"Design needs revision (iteration {iteration + 1}/{effective_max_design_retries + 1}), retrying..."
                                )
                                continue
                    break
            else:
                await progress_cb(f"[{design_idx}/{total_steps}] Design phase (cached)")

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

        # ── Phase B: Implementation ──

        # Find step indices dynamically (design loop may have added extra steps)
        total_steps = len(ctx.steps)

        # Qwen Hints (non-fatal, optional)
        qwen_idx = _find_step("Qwen Hints")
        if qwen_idx is not None:
            if _step_done(qwen_idx):
                await progress_cb(f"[{qwen_idx}/{total_steps}] Qwen Pre-Implement (cached)")
            else:
                try:
                    await progress_cb(f"[{qwen_idx}/{total_steps}] Qwen Pre-Implement...")
                    await step_qwen_pre_implement(ctx, ollama, settings, progress_cb, step_index=qwen_idx)
                except Exception as exc:
                    logger.warning("Qwen pre-implement failed (non-fatal): %s", exc)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

        # Sonnet Implement (always present)
        impl_idx = _find_step("Sonnet Implement")
        if impl_idx is None:
            return "failed", "Sonnet Implement step not found"

        if _step_done(impl_idx):
            await progress_cb(f"[{impl_idx}/{total_steps}] Sonnet Implement (cached)")
        else:
            await progress_cb(f"[{impl_idx}/{total_steps}] Sonnet Implement...")
            await step_claude_implement(ctx, settings, cancel_event, progress_cb, step_index=impl_idx)

        # Capture snapshot ref for safe-fail recovery
        if not (resuming and ctx.impl_snapshot_ref):
            snap_proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "HEAD",
                cwd=ctx.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            snap_out, _ = await asyncio.wait_for(snap_proc.communicate(), timeout=5)
            ctx.impl_snapshot_ref = snap_out.decode().strip() if snap_out else ""

        # Notify scheduler: implement done
        if scheduler:
            scheduler.notify_implement_done(ctx.issue_num)

        if cancel_event.is_set():
            if scheduler:
                scheduler.notify_audit_failed(ctx.issue_num)
            return "skipped", "Cancelled by user"

        # Local CI Check (optional)
        ci_idx = _find_step("Local CI Check")
        if ci_idx is not None and ci_commands:
            if _step_done(ci_idx):
                await progress_cb(f"[{ci_idx}/{total_steps}] Local CI Check (cached)")
            else:
                await progress_cb(f"[{ci_idx}/{total_steps}] Local CI Check...")
                await step_local_ci_check(ctx, settings, progress_cb, ci_commands,
                                          step_index=ci_idx, max_retries_override=effective_ci_fix_retries)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

        # Sonnet Self-Review (optional)
        self_review_idx = _find_step("Sonnet Self-Review")
        if self_review_idx is not None:
            if _step_done(self_review_idx):
                await progress_cb(f"[{self_review_idx}/{total_steps}] Sonnet Self-Review (cached)")
            else:
                await progress_cb(f"[{self_review_idx}/{total_steps}] Sonnet Self-Review...")
                await step_sonnet_self_review(ctx, settings, cancel_event, progress_cb, step_index=self_review_idx)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

        # ── Phase C: Audit ──

        # Gemini Cross-Review (optional)
        cross_review_idx = _find_step("Gemini Cross-Review")
        if cross_review_idx is not None:
            if _step_done(cross_review_idx):
                await progress_cb(f"[{cross_review_idx}/{total_steps}] Gemini Cross-Review (cached)")
            else:
                await progress_cb(f"[{cross_review_idx}/{total_steps}] Gemini Cross-Review...")
                await step_gemini_cross_review(ctx, gemini, settings, progress_cb, step_index=cross_review_idx)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

        # Wait-Gate: wait for dependency audit approval before starting audit
        if scheduler:
            await progress_cb("Waiting for dependency audit approval...")
            deps_ok = await scheduler.wait_dependencies(
                ctx.issue_num, timeout=settings.stagger_gate_timeout,
            )
            if not deps_ok:
                scheduler.notify_audit_failed(ctx.issue_num)
                return "failed", "Dependency wait timed out or dependency failed"

        # AI Audit (optional, with retry loop)
        audit_idx = _find_step("AI Audit")
        if audit_idx is not None and effective_ai_audit_enabled:
            ctx.review_report = ctx.gemini_cross_review or ctx.self_review_report or "(no review available)"

            for audit_attempt in range(effective_ai_audit_max_retries + 1):
                if cancel_event.is_set():
                    if scheduler:
                        scheduler.notify_audit_failed(ctx.issue_num)
                    return "skipped", "Cancelled by user"

                audit_label = f" (retry {audit_attempt})" if audit_attempt > 0 else ""
                await progress_cb(f"[{audit_idx}/{total_steps}] AI Audit{audit_label}...")

                try:
                    await step_ai_audit(ctx, ollama, settings, progress_cb, step_index=audit_idx)
                    break  # PASS
                except RuntimeError:
                    # Supreme Court: self-review PASS vs audit FAIL
                    if ctx.self_review_report and parse_verdict(ctx.self_review_report) is True:
                        await progress_cb("Conflict detected — Supreme Court (Gemini)...")
                        ruling = await step_supreme_court(ctx, gemini, settings, progress_cb)

                        if supreme_court_cb:
                            user_decision = await supreme_court_cb(ctx, ruling)
                        else:
                            user_decision = ruling.lower()

                        if user_decision == "overturn":
                            ctx.ai_audit_passed = True
                            ctx.steps[audit_idx].status = "passed"
                            ctx.steps[audit_idx].detail = "OVERTURNED by Supreme Court"
                            break
                        elif user_decision == "redesign":
                            if scheduler:
                                scheduler.notify_audit_failed(ctx.issue_num)
                            return "failed", "Supreme Court: REDESIGN required"
                        # else "uphold" → continue normal retry

                    if audit_attempt < effective_ai_audit_max_retries:
                        ctx.audit_fix_history.append(
                            f"audit retry {audit_attempt + 1}: {ctx.ai_audit_result[:200]}"
                        )
                        ctx.review_feedback = ctx.ai_audit_result

                        await progress_cb(f"AI Audit FAIL — Sonnet re-implementing (retry {audit_attempt + 1})...")
                        if ctx.impl_snapshot_ref:
                            await _reset_to_snapshot(ctx.project_path, ctx.impl_snapshot_ref)

                        await step_claude_implement(ctx, settings, cancel_event, progress_cb, step_index=impl_idx)
                        await _snapshot_commit(ctx.project_path, ctx.issue_num)
                        ctx.git_diff = await _capture_filtered_diff(
                            ctx.project_path, base_ref=ctx.base_commit or "main",
                        )

                        if ci_idx is not None and ci_commands:
                            await step_local_ci_check(ctx, settings, progress_cb, ci_commands,
                                                      step_index=ci_idx, max_retries_override=effective_ci_fix_retries)

                        ctx.steps[audit_idx] = PipelineStep(name=f"AI Audit (retry {audit_attempt + 1})")
                    else:
                        raise  # Final failure
        else:
            # AI Audit skipped for this mode
            ctx.ai_audit_passed = True

        # Notify scheduler: audit approved
        if scheduler:
            scheduler.notify_audit_approved(ctx.issue_num)

        # ── State Sync: record decisions after audit approval ──
        if ctx.ai_audit_passed or not effective_ai_audit_enabled:
            try:
                from .state_sync import append_project_summary
                decisions = await _extract_decisions_llm(ctx, settings, progress_cb)
                files = _extract_files_from_diff(ctx.git_diff)
                append_project_summary(ctx.project_path, ctx.issue_num, ctx.issue_title, decisions, files)
            except Exception:
                logger.warning("State sync failed (non-fatal)")

        # ── Phase D: Post-Process ──
        if settings.enable_data_mining and (ctx.ai_audit_passed or not effective_ai_audit_enabled):
            data_mining_idx = _find_step("Data Mining")
            if data_mining_idx is None:
                ctx.steps.append(PipelineStep(name="Data Mining"))
                data_mining_idx = len(ctx.steps) - 1
            try:
                await progress_cb(f"[{data_mining_idx}/{len(ctx.steps)}] Data Mining...")
                await step_data_mining_fivebrid(ctx, ollama, settings, progress_cb, step_index=data_mining_idx)
            except Exception as exc:
                logger.warning("Data mining step failed (non-fatal): %s", exc)

        return "success", "All checks passed"

    except asyncio.CancelledError:
        return "skipped", "Cancelled by user"
    except Exception as exc:
        for s in reversed(ctx.steps):
            if s.status == "failed":
                return "failed", f"{s.name}: {s.detail}"
        return "failed", str(exc)[:200]
    finally:
        # Ensure scheduler is notified on any exit path (set() is idempotent)
        if scheduler and not ctx.ai_audit_passed:
            scheduler.notify_audit_failed(ctx.issue_num)


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _is_step_done(ctx: PipelineContext, idx: int, resuming: bool) -> bool:
    """Check if a step was already completed (for resume)."""
    return resuming and idx < len(ctx.steps) and ctx.steps[idx].status == "passed"


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
            "revised": "\U0001f504",
            "skipped": "\u23ed",
            "running": "\u23f3",
            "pending": "\u2b1c",
        }.get(s.status, "\u2753")

        elapsed_str = ""
        if s.elapsed_sec > 0:
            mins, secs = divmod(int(s.elapsed_sec), 60)
            elapsed_str = f" ({mins}m {secs}s)" if mins else f" ({secs}s)"

        lines.append(f"  {icon} {s.name}{elapsed_str}")
        if s.status in ("failed", "revised") and s.detail:
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
        step.detail = _exc_detail(exc)
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
        step.detail = _exc_detail(exc)
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
        step.detail = _exc_detail(exc)
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
        step.detail = _exc_detail(exc)
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
