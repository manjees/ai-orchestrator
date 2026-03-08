"""Checkpoint persistence for pipeline resume (/retry).

Saves PipelineContext as JSON and manages git tags to preserve implementation
commits after worktree deletion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from .pipeline import PipelineContext, PipelineStep

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path(__file__).parent / ".checkpoints"
_TAG_PREFIX = "checkpoint"


def _checkpoint_path(project_name: str, issue_num: int) -> Path:
    return _CHECKPOINT_DIR / f"{project_name}_{issue_num}.json"


def save_checkpoint(
    ctx: PipelineContext,
    pipeline_mode: str,
    failed_step_name: str,
    failed_step_index: int,
) -> Path | None:
    """Serialize PipelineContext + metadata to a JSON checkpoint file.

    Returns the path on success, None on failure.  Never raises — checkpoint
    save must not break the pipeline.
    """
    try:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # Derive the original project path from the worktree path
        # worktree_dir is typically: <project_path>/.worktrees/solve-issue-<N>
        original_project_path = _extract_original_project_path(ctx.project_path)

        data = {
            "pipeline_mode": pipeline_mode,
            "failed_step_name": failed_step_name,
            "failed_step_index": failed_step_index,
            "original_project_path": original_project_path,
            "ctx": asdict(ctx),
        }
        path = _checkpoint_path(ctx.project_name, ctx.issue_num)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(
            "Checkpoint saved: %s (failed at %s)", path.name, failed_step_name,
        )
        return path
    except Exception:
        logger.exception("Failed to save checkpoint")
        return None


def load_checkpoint(project_name: str, issue_num: int) -> dict | None:
    """Load a checkpoint file.  Returns the parsed dict or None."""
    path = _checkpoint_path(project_name, issue_num)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("Failed to load checkpoint %s", path)
        return None


def delete_checkpoint(project_name: str, issue_num: int) -> None:
    """Remove the checkpoint file if it exists."""
    path = _checkpoint_path(project_name, issue_num)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to delete checkpoint %s", path)


def list_checkpoints() -> list[dict]:
    """Return metadata for all stored checkpoints."""
    results: list[dict] = []
    if not _CHECKPOINT_DIR.exists():
        return results
    for f in sorted(_CHECKPOINT_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            results.append({
                "file": f.name,
                "project_name": data["ctx"]["project_name"],
                "issue_num": data["ctx"]["issue_num"],
                "pipeline_mode": data.get("pipeline_mode", "unknown"),
                "failed_step_name": data.get("failed_step_name", "unknown"),
            })
        except Exception:
            logger.warning("Skipping corrupt checkpoint: %s", f)
    return results


def restore_context(checkpoint: dict) -> PipelineContext:
    """Rebuild a PipelineContext from checkpoint data.

    The returned ctx has ``project_path=""`` — caller must set it after
    creating a new worktree.
    """
    ctx_data = checkpoint["ctx"]

    # Rebuild PipelineStep list
    steps = [
        PipelineStep(
            name=s.get("name", ""),
            status=s.get("status", "pending"),
            detail=s.get("detail", ""),
            elapsed_sec=s.get("elapsed_sec", 0.0),
        )
        for s in ctx_data.get("steps", [])
    ]

    ctx = PipelineContext(
        project_path="",  # caller sets after worktree creation
        project_name=ctx_data.get("project_name", ""),
        issue_num=ctx_data.get("issue_num", 0),
        branch_name=ctx_data.get("branch_name", ""),
        issue_body=ctx_data.get("issue_body", ""),
        issue_title=ctx_data.get("issue_title", ""),
        base_commit=ctx_data.get("base_commit", ""),
        design_doc=ctx_data.get("design_doc", ""),
        qwen_hints=ctx_data.get("qwen_hints", ""),
        git_diff=ctx_data.get("git_diff", ""),
        review_report=ctx_data.get("review_report", ""),
        audit_result=ctx_data.get("audit_result", ""),
        audit_passed=ctx_data.get("audit_passed", False),
        data_mining_result=ctx_data.get("data_mining_result", ""),
        retry_count=ctx_data.get("retry_count", 0),
        review_feedback=ctx_data.get("review_feedback", ""),
        review_passed=ctx_data.get("review_passed", False),
        research_log=ctx_data.get("research_log", ""),
        gemini_design_critique=ctx_data.get("gemini_design_critique", ""),
        design_iteration=ctx_data.get("design_iteration", 0),
        self_review_report=ctx_data.get("self_review_report", ""),
        gemini_cross_review=ctx_data.get("gemini_cross_review", ""),
        impl_snapshot_ref=ctx_data.get("impl_snapshot_ref", ""),
        ci_check_log=ctx_data.get("ci_check_log", ""),
        ai_audit_result=ctx_data.get("ai_audit_result", ""),
        ai_audit_passed=ctx_data.get("ai_audit_passed", False),
        ci_fix_history=ctx_data.get("ci_fix_history", []),
        audit_fix_history=ctx_data.get("audit_fix_history", []),
        mode=ctx_data.get("mode", "standard"),
        triage_reason=ctx_data.get("triage_reason", ""),
        split_plan=ctx_data.get("split_plan", ""),
        steps=steps,
    )
    return ctx


# ── Git Tag Helpers ──────────────────────────────────────────────────────────

def _tag_name(project_name: str, issue_num: int) -> str:
    return f"{_TAG_PREFIX}/{project_name}/{issue_num}"


async def create_checkpoint_tag(
    project_path: str, project_name: str, issue_num: int, commit_ref: str,
) -> bool:
    """Create (or move) a lightweight git tag to preserve the commit from GC."""
    tag = _tag_name(project_name, issue_num)
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "tag", "-f", tag, commit_ref,
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode == 0:
            logger.info("Checkpoint tag created: %s -> %s", tag, commit_ref[:8])
            return True
        logger.warning("Failed to create checkpoint tag %s (rc=%d)", tag, proc.returncode)
        return False
    except Exception:
        logger.exception("Failed to create checkpoint tag %s", tag)
        return False


async def delete_checkpoint_tag(
    project_path: str, project_name: str, issue_num: int,
) -> None:
    """Delete the checkpoint tag (cleanup after success)."""
    tag = _tag_name(project_name, issue_num)
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "tag", "-d", tag,
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
    except Exception:
        logger.warning("Failed to delete checkpoint tag %s", tag)


async def check_staleness(project_path: str, commit_ref: str) -> bool:
    """Return True if commit_ref is an ancestor of origin/main (i.e. not stale)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "merge-base", "--is-ancestor", commit_ref, "origin/main",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode == 0
    except Exception:
        return False  # Can't verify → treat as potentially stale


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_original_project_path(worktree_path: str) -> str:
    """Extract original project path from worktree path.

    Worktree: /path/to/project/.worktrees/solve-issue-42
    Returns:  /path/to/project
    """
    parts = worktree_path.split(os.sep)
    try:
        idx = parts.index(".worktrees")
        return os.sep.join(parts[:idx])
    except ValueError:
        # Not a worktree path — return as-is
        return worktree_path
