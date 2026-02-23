"""Triple-Model Pipeline: DeepSeek Design → Qwen Pre-Implement → Claude Implement → Claude Review → DeepSeek Audit → Data Mining."""

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
    steps: list[PipelineStep] = field(default_factory=lambda: [
        PipelineStep(name="DeepSeek Design"),
        PipelineStep(name="Qwen Pre-Implement"),
        PipelineStep(name="Claude Implement"),
        PipelineStep(name="Claude Review"),
        PipelineStep(name="DeepSeek Audit"),
    ])


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


# ── Git Diff Filtering ───────────────────────────────────────────────────────

DIFF_EXCLUDE_PATTERNS = [
    "*.lock", "*.min.js", "*.min.css", "*.map",
    "*.pyc", "__pycache__/*", "*.egg-info/*",
    "dist/*", "build/*", "node_modules/*",
    "*.png", "*.jpg", "*.gif", "*.ico", "*.woff*",
]

_MAX_DIFF_CHARS = 50_000


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
            model=settings.qwen_coder_model,
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
        f"Implement the solution following this design guide. "
        f"IMPORTANT: After implementation, you MUST run the project's compile/build command "
        f"to verify there are no compilation errors before finishing. "
        f"Fix any compile errors before proceeding. "
        f"Complete all steps including testing.\n\n"
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
            model=settings.qwen_coder_model,
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


# ── Orchestrator ─────────────────────────────────────────────────────────────

async def run_dual_check_pipeline(
    ctx: PipelineContext,
    ollama: OllamaProvider,
    anthropic: AnthropicProvider | None,
    settings: Settings,
    cancel_event: asyncio.Event,
    progress_cb: ProgressCallback,
) -> tuple[str, str]:
    """Run the full 5-step pipeline with automatic retry on review failure. Returns (status, detail)."""
    max_retries = settings.max_review_retries

    try:
        # Fetch issue
        await progress_cb("Fetching issue...")
        await step_fetch_issue(ctx)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 1: DeepSeek Design (once)
        await progress_cb("[1/5] DeepSeek Design...")
        await step_deepseek_design(ctx, ollama, settings, progress_cb)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Step 1.5: Qwen Pre-Implement (non-fatal)
        try:
            await progress_cb("[1.5/5] Qwen Pre-Implement...")
            await step_qwen_pre_implement(ctx, ollama, settings, progress_cb, step_index=1)
        except Exception as exc:
            logger.warning("Qwen pre-implement step failed (non-fatal): %s", exc)

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Steps 2+3: Implement + Review (retry loop)
        for attempt in range(max_retries + 1):
            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

            # On retry: reset worktree and add new step entries
            if attempt > 0:
                ctx.retry_count = attempt
                await _reset_worktree(ctx.project_path)
                # Insert new step pair before the last step (DeepSeek Audit)
                ctx.steps.insert(-1, PipelineStep(name=f"Claude Implement (retry {attempt})"))
                ctx.steps.insert(-1, PipelineStep(name=f"Claude Review (retry {attempt})"))

            step_idx_impl = 2 + (attempt * 2)
            step_idx_review = 3 + (attempt * 2)

            # Step 2: Claude Implement
            attempt_label = f" (retry {attempt})" if attempt > 0 else ""
            await progress_cb(f"[2/5] Claude Implement{attempt_label}...")
            await step_claude_implement(ctx, settings, cancel_event, progress_cb, step_index=step_idx_impl)

            if cancel_event.is_set():
                return "skipped", "Cancelled by user"

            # Step 3: Claude Review
            await progress_cb(f"[3/5] Claude Review{attempt_label}...")
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

        # Step 4: DeepSeek Audit (always last non-mining step)
        await progress_cb("[4/5] DeepSeek Audit...")
        await step_deepseek_audit(ctx, ollama, settings, progress_cb, step_index=-1)

        # Step 5: Data Mining (conditional, non-fatal)
        if settings.enable_data_mining and ctx.audit_passed:
            ctx.steps.append(PipelineStep(name="Data Mining"))
            try:
                await progress_cb("[5/5] Data Mining...")
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

    # Append review feedback when review failed
    review_step = next((s for s in ctx.steps if "Review" in s.name and s.status == "failed"), None)
    if review_step and ctx.review_report:
        summary = _extract_review_summary(ctx.review_report)
        lines.append(f"\n\U0001f4cb Review Feedback:\n{summary[:3000]}")

    return "\n".join(lines)
