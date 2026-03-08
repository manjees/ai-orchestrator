"""Staggered parallel scheduler for multi-issue solve."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ScheduleMode:
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    STAGGERED = "staggered"


@dataclass
class IssueSlot:
    """Single issue's scheduling state."""
    issue_num: int
    depends_on: list[int] = field(default_factory=list)
    mode: str = "standard"
    triage_reason: str = ""

    estimated_files: list[str] = field(default_factory=list)

    # Phase gate events — set by pipeline, awaited by scheduler
    implement_done: asyncio.Event = field(default_factory=asyncio.Event)
    audit_approved: asyncio.Event = field(default_factory=asyncio.Event)
    audit_failed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class StaggeredScheduler:
    """Coordinates staggered execution of multiple issues."""
    slots: dict[int, IssueSlot] = field(default_factory=dict)

    def add_slot(self, slot: IssueSlot) -> None:
        self.slots[slot.issue_num] = slot

    def get_slot(self, issue_num: int) -> IssueSlot | None:
        return self.slots.get(issue_num)

    async def wait_dependencies(self, issue_num: int, timeout: int = 3600) -> bool:
        """Wait until all dependencies' audits are approved.
        Returns False on timeout or if any dependency failed."""
        slot = self.slots.get(issue_num)
        if not slot or not slot.depends_on:
            return True

        for dep_num in slot.depends_on:
            dep_slot = self.slots.get(dep_num)
            if dep_slot and dep_slot.audit_failed.is_set():
                logger.warning("#%d dependency #%d already failed", issue_num, dep_num)
                return False

        async def _wait_dep(dep_num: int) -> bool:
            dep_slot = self.slots.get(dep_num)
            if not dep_slot:
                return True
            approved_task = asyncio.create_task(dep_slot.audit_approved.wait())
            failed_task = asyncio.create_task(dep_slot.audit_failed.wait())
            done, pending = await asyncio.wait(
                [approved_task, failed_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            return dep_slot.audit_approved.is_set()

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[_wait_dep(d) for d in slot.depends_on]),
                timeout=timeout,
            )
            return all(results)
        except asyncio.TimeoutError:
            logger.warning("Dependency wait timed out for #%d", issue_num)
            return False

    def notify_implement_done(self, issue_num: int) -> None:
        slot = self.slots.get(issue_num)
        if slot:
            slot.implement_done.set()

    def notify_audit_approved(self, issue_num: int) -> None:
        slot = self.slots.get(issue_num)
        if slot:
            slot.audit_approved.set()

    def notify_audit_failed(self, issue_num: int) -> None:
        """Signal that this issue's audit failed — wake up dependents immediately."""
        slot = self.slots.get(issue_num)
        if slot:
            slot.audit_failed.set()

    def detect_file_conflicts(self) -> list[tuple[int, int]]:
        """Detect pairs of issues that may conflict on shared files."""
        conflicts = []
        nums = list(self.slots.keys())
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                slot_a, slot_b = self.slots[a], self.slots[b]
                if slot_a.estimated_files and slot_b.estimated_files:
                    overlap = set(slot_a.estimated_files) & set(slot_b.estimated_files)
                    if overlap:
                        conflicts.append((a, b))
                        logger.info("File conflict detected: #%d and #%d share %s", a, b, overlap)
        return conflicts

    def enforce_file_conflict_deps(self) -> None:
        """Auto-add depends_on for issues with file conflicts (force sequential)."""
        for a, b in self.detect_file_conflicts():
            slot_b = self.slots[b]
            if a not in slot_b.depends_on:
                slot_b.depends_on.append(a)
                logger.info("Auto-dependency: #%d → #%d (file conflict)", b, a)


async def triage_issue_dependencies(
    issues: list[dict],
    settings,
    progress_cb,
) -> dict[int, IssueSlot]:
    """Analyze all issues together for inter-dependencies.

    Returns dict of issue_num -> IssueSlot with depends_on populated.
    """
    from .pipeline import _call_claude_cli_with_progress

    issue_list = "\n".join(
        f"#{d['num']}: {d['title']}\n{d['body'][:500]}" for d in issues
    )
    prompt = (
        "Analyze these GitHub issues and determine execution dependencies.\n\n"
        f"{issue_list}\n\n"
        "Rules:\n"
        "- An issue DEPENDS ON another only if it modifies the SAME files/APIs "
        "and the second must see the first's changes\n"
        "- Independent issues can run in parallel\n"
        "- Most issues are independent unless they clearly overlap\n\n"
        "Reply in this EXACT format for each issue:\n"
        "#<num>: MODE=EXPRESS|STANDARD|FULL, DEPENDS=<comma-separated #nums or NONE>, "
        "FILES=<comma-separated likely files to modify>, REASON=<one sentence>\n"
    )

    try:
        output = await _call_claude_cli_with_progress(
            prompt,
            model=settings.haiku_model,
            timeout=settings.dep_triage_timeout,
            progress_cb=progress_cb,
            step_name="Dependency Triage",
        )
        slots: dict[int, IssueSlot] = {}
        for line in output.strip().splitlines():
            match = re.match(
                r"#(\d+):\s*MODE=(\w+),\s*DEPENDS=(.+?),\s*FILES=(.+?),\s*REASON=(.+)", line,
            )
            if match:
                num = int(match.group(1))
                mode_str = match.group(2).strip().lower()
                if mode_str not in ("express", "standard", "full"):
                    mode_str = "standard"
                deps_str = match.group(3).strip()
                depends: list[int] = []
                if deps_str.upper() != "NONE":
                    for d in re.findall(r"#?(\d+)", deps_str):
                        dep_num = int(d)
                        if dep_num != num:
                            depends.append(dep_num)
                files_str = match.group(4).strip()
                est_files = [f.strip() for f in files_str.split(",") if f.strip() and f.strip().upper() != "NONE"]
                slots[num] = IssueSlot(
                    issue_num=num,
                    depends_on=depends,
                    mode=mode_str,
                    triage_reason=match.group(5).strip(),
                    estimated_files=est_files,
                )
        # Ensure all issues have slots (fallback for unparsed)
        for d in issues:
            if d["num"] not in slots:
                slots[d["num"]] = IssueSlot(issue_num=d["num"])
        return slots
    except Exception:
        logger.warning("Dependency triage failed, treating all as independent")
        return {d["num"]: IssueSlot(issue_num=d["num"]) for d in issues}
