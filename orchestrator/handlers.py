"""Telegram command handlers: /status, /cmd, /view, /projects, /issues, /solve, /rebase, and process control."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import signal
import time

import psutil
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from .ai.anthropic_provider import AnthropicProvider
from .ai.gemini_provider import GeminiCLIProvider
from .ai.ollama_provider import OllamaProvider
from .pipeline import (
    InitContext,
    PipelineContext,
    format_pipeline_summary,
    run_dual_check_pipeline,
    run_fivebrid_pipeline,
    run_init_pipeline,
    step_plan_issues,
    step_discuss_consult,
    step_discuss_to_issues,
)
from .security import mask_secrets
from .system_monitor import get_system_status
from .tmux_manager import capture_pane, list_sessions

logger = logging.getLogger(__name__)

# Process names to monitor for inline keyboard control
_MONITORED_PROCESSES = {"ollama", "python", "node"}

# Active solve sessions: chat_id → {issue_num → asyncio.Event}
_solve_cancels: dict[int, dict[int, asyncio.Event]] = {}
_solve_active: dict[int, int] = {}  # chat_id → number of active solve sessions

# Active init sessions: chat_id → asyncio.Event
_init_cancels: dict[int, asyncio.Event] = {}

# Active plan sessions: chat_id → asyncio.Event
_plan_cancels: dict[int, asyncio.Event] = {}

# Active discuss sessions: chat_id → asyncio.Event
_discuss_cancels: dict[int, asyncio.Event] = {}

# Discuss results for [Create Issues] button: message_id → context dict
_discuss_results: dict[int, dict] = {}

# ANSI escape codes: CSI sequences (incl. ?/= private modes), OSC sequences, single ESC codes
_ANSI_RE = re.compile(r"\x1b\[[^A-Za-z]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[^[\]()]")


def _resolve_project(name: str, projects: dict[str, dict]) -> tuple[str | None, str]:
    """Resolve project name with prefix matching.

    Returns (resolved_name, error_message).
    If resolved_name is None, error_message explains why.
    """
    # Exact match first
    if name in projects:
        return name, ""

    # Prefix match
    matches = [p for p in projects if p.startswith(name)]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        names = ", ".join(f"<code>{html.escape(m)}</code>" for m in sorted(matches))
        return None, f"Ambiguous project: {names}"

    return None, f"Unknown project: <code>{html.escape(name)}</code>"


def _sanitize_output(text: str) -> str:
    """Strip ANSI escape codes, control characters, then HTML-escape."""
    text = _ANSI_RE.sub("", text)
    # Remove remaining control chars (keep \n, \r, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return html.escape(text)


# ── /status ──────────────────────────────────────────────────────────────────

async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Report system stats, Ollama status, tmux sessions, and controllable processes."""
    try:
        # Gather system status and tmux sessions concurrently
        sys_status, sessions = await asyncio.gather(
            get_system_status(),
            list_sessions(),
        )

        lines = [
            "<b>System Status</b>",
            f"RAM: {sys_status.ram_used_gb}/{sys_status.ram_total_gb} GB ({sys_status.ram_percent}%)",
            f"CPU: {sys_status.cpu_percent}%",
            f"Thermal: {sys_status.thermal_pressure}",
            f"Disk: {sys_status.disk_used_gb}/{sys_status.disk_total_gb} GB ({sys_status.disk_percent}%)",
        ]

        # Ollama status
        ollama: OllamaProvider | None = context.bot_data.get("ollama")
        if ollama and await ollama.is_available():
            models = await ollama.get_loaded_models()
            if models:
                model_strs = [f"{m.name}, {m.size_gb}GB" for m in models]
                lines.append(f"Ollama: online ({'; '.join(model_strs)})")
            else:
                lines.append("Ollama: online (no models loaded)")
        else:
            lines.append("Ollama: offline")

        # Tmux sessions
        if sessions:
            sess_list = ", ".join(f"{s.name}({s.windows}w)" for s in sessions)
            lines.append(f"Tmux: {sess_list}")
        else:
            lines.append("Tmux: no sessions")

        # Process inline keyboard
        keyboard = _build_process_keyboard()

        await update.message.reply_text(  # type: ignore[union-attr]
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard if keyboard.inline_keyboard else None,
        )
    except Exception:
        logger.exception("/status handler error")
        await _safe_reply(update, "Error collecting system status.")


# ── /cmd ─────────────────────────────────────────────────────────────────────

async def cmd_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute a shell command with timeout. Supports --long and --stream flags."""
    try:
        if not context.args:
            await _safe_reply(update, "Usage: `/cmd [--long|--stream] <command>`")
            return

        args = list(context.args)
        settings = context.bot_data["settings"]
        timeout = settings.cmd_timeout
        stream = False

        # Parse flags
        if args and args[0] == "--long":
            args.pop(0)
            timeout = settings.cmd_long_timeout
        elif args and args[0] == "--stream":
            args.pop(0)
            stream = True
            timeout = settings.cmd_long_timeout

        command = " ".join(args)
        if not command:
            await _safe_reply(update, "No command provided.")
            return

        logger.info("Executing command: %s (timeout=%ds, stream=%s)", command, timeout, stream)

        if stream:
            await _cmd_stream(update, command, timeout)
        else:
            await _cmd_simple(update, command, timeout)
    except Exception:
        logger.exception("/cmd handler error")
        await _safe_reply(update, "Error executing command.")


async def _cmd_simple(update: Update, command: str, timeout: int) -> None:
    """Run command, wait for completion, send output."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode(errors="replace") if stdout else "(no output)"
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[possibly-undefined]
        output = f"(command timed out after {timeout}s)"
    except FileNotFoundError:
        output = "(command not found)"

    output = mask_secrets(output)
    # Truncate to 4000 chars for Telegram message limit
    if len(output) > 4000:
        output = output[-4000:]

    await _safe_reply(update, f"<pre>{_sanitize_output(output)}</pre>")


async def _cmd_stream(update: Update, command: str, timeout: int) -> None:
    """Run command with streaming: update message every 10 seconds."""
    start = time.monotonic()
    msg = await update.message.reply_text("<pre>(starting...)</pre>", parse_mode=ParseMode.HTML)  # type: ignore[union-attr]
    collected = ""
    last_sent = ""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None

        async def _read_output() -> None:
            nonlocal collected
            async for line in proc.stdout:  # type: ignore[union-attr]
                collected += line.decode(errors="replace")

        read_task = asyncio.create_task(_read_output())

        # Periodically update the Telegram message
        while not read_task.done():
            await asyncio.sleep(10)
            elapsed = int(time.monotonic() - start)
            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"

            if collected:
                snippet = mask_secrets(collected[-3800:])
                sanitized = _sanitize_output(snippet)
                text = f"<b>[running {time_str}]</b>\n<pre>{sanitized}</pre>"
            else:
                text = f"<b>[running {time_str} — waiting for output...]</b>"

            # Always update because elapsed time changes
            try:
                await msg.edit_text(text, parse_mode=ParseMode.HTML)
                last_sent = text
            except Exception as exc:
                logger.warning("Stream edit failed: %s", exc)
                try:
                    plain = f"[running {time_str}]\n{collected[-3900:]}" if collected else f"[running {time_str} — waiting...]"
                    if plain != last_sent:
                        await msg.edit_text(plain)
                        last_sent = plain
                except Exception:
                    pass

        await asyncio.wait_for(read_task, timeout=timeout)
        await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[possibly-undefined]
        collected += f"\n(timed out after {timeout}s)"
    except Exception:
        logger.exception("Stream command error")
        collected += "\n(stream error)"

    # Final result
    elapsed = int(time.monotonic() - start)
    mins, secs = divmod(elapsed, 60)
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    final = mask_secrets(collected[-3800:]) if collected else "(no output)"
    sanitized = _sanitize_output(final)
    rc = proc.returncode if proc else None  # type: ignore[possibly-undefined]
    header = f"<b>[done in {time_str}, exit={rc}]</b>"
    try:
        await msg.edit_text(f"{header}\n<pre>{sanitized}</pre>", parse_mode=ParseMode.HTML)
    except Exception:
        try:
            await msg.edit_text(f"[done in {time_str}, exit={rc}]\n{final[-3900:]}")
        except Exception:
            pass


# ── /view ────────────────────────────────────────────────────────────────────

async def view_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Capture and send the last 20 lines from the ai_factory tmux session."""
    try:
        settings = context.bot_data["settings"]
        output = await capture_pane(settings.tmux_session_name)
        output = mask_secrets(output)
        if not output.strip():
            output = "(empty pane)"
        # Truncate for Telegram
        if len(output) > 4000:
            output = output[-4000:]
        await _safe_reply(update, f"<pre>{_sanitize_output(output)}</pre>")
    except Exception:
        logger.exception("/view handler error")
        await _safe_reply(update, "Error capturing tmux pane.")


# ── Process control (InlineKeyboard callbacks) ───────────────────────────────

def _build_process_keyboard() -> InlineKeyboardMarkup:
    """Build inline keyboard with pause/resume buttons for monitored processes."""
    buttons: list[list[InlineKeyboardButton]] = []
    try:
        for proc in psutil.process_iter(["pid", "name", "status"]):
            info = proc.info  # type: ignore[attr-defined]
            name = info.get("name", "")
            if not name:
                continue
            name_lower = name.lower()
            if not any(m in name_lower for m in _MONITORED_PROCESSES):
                continue
            # Only show processes owned by the current user
            try:
                if proc.username() != os.getlogin():
                    continue
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue

            pid = info["pid"]
            status = info.get("status", "")
            if status == psutil.STATUS_STOPPED:
                buttons.append([
                    InlineKeyboardButton(
                        f"▶ Resume {name} ({pid})",
                        callback_data=f"proc:cont:{pid}",
                    )
                ])
            else:
                buttons.append([
                    InlineKeyboardButton(
                        f"⏸ Pause {name} ({pid})",
                        callback_data=f"proc:stop:{pid}",
                    )
                ])
    except Exception:
        logger.debug("Error building process keyboard", exc_info=True)
    return InlineKeyboardMarkup(buttons)


async def process_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle process pause/resume button presses."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()

    try:
        data = query.data or ""
        if not data.startswith("proc:"):
            return

        _, action, pid_str = data.split(":", 2)
        pid = int(pid_str)

        if action == "stop":
            os.kill(pid, signal.SIGSTOP)
            await query.edit_message_text(f"Paused process {pid} (SIGSTOP)")
        elif action == "cont":
            os.kill(pid, signal.SIGCONT)
            await query.edit_message_text(f"Resumed process {pid} (SIGCONT)")
        else:
            await query.edit_message_text(f"Unknown action: {action}")
    except ProcessLookupError:
        await query.edit_message_text("Process no longer exists.")
    except PermissionError:
        await query.edit_message_text("Permission denied.")
    except Exception:
        logger.exception("Process control callback error")
        await query.edit_message_text("Error controlling process.")


# ── /service ─────────────────────────────────────────────────────────────────

def _get_launchd_label(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Return launchd label from settings, falling back to default."""
    try:
        return context.bot_data["settings"].launchd_label
    except (KeyError, AttributeError):
        return "com.ai-orchestrator"


def _get_plist_path(label: str) -> str:
    return os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")


async def service_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show service menu: /service [status|restart|stop|start|logs]"""
    try:
        sub = (context.args[0].lower() if context.args else "").strip()

        if sub == "status":
            await _svc_status(update, context)
        elif sub == "restart":
            await _svc_restart(update, context)
        elif sub == "stop":
            await _svc_stop(update, context)
        elif sub == "start":
            await _svc_start(update, context)
        elif sub == "logs":
            await _svc_logs(update, context, _get_launchd_label(context))
        else:
            await _safe_reply(
                update,
                "<b>Service Control</b>\n"
                "/service status  - launchd 상태 확인\n"
                "/service restart - 봇 재시작\n"
                "/service stop    - 봇 중지\n"
                "/service start   - 봇 시작\n"
                "/service logs    - 최근 로그 (stderr)\n"
                "/service logs stdout - 최근 로그 (stdout)",
            )
    except Exception:
        logger.exception("/service handler error")
        await _safe_reply(update, "Error in service handler.")


async def _run_launchctl(*args: str) -> tuple[int, str]:
    """Run a launchctl command and return (returncode, combined output)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "launchctl", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode or 0, stdout.decode(errors="replace")
    except asyncio.TimeoutError:
        return 1, "(launchctl timed out)"
    except FileNotFoundError:
        return 1, "(launchctl not found)"


async def _svc_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    label = _get_launchd_label(context)
    rc, output = await _run_launchctl("list", label)
    if rc != 0:
        await _safe_reply(update, f"Service not loaded\n<pre>{html.escape(output)}</pre>")
        return
    # Parse PID and last exit status from output
    lines = ["<b>Service Status</b>"]
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('"PID"'):
            pid = line.split("=")[-1].strip().rstrip(";")
            lines.append(f"PID: {pid}")
        elif line.startswith('"LastExitStatus"'):
            code = line.split("=")[-1].strip().rstrip(";")
            lines.append(f"Last Exit: {code}")
        elif line.startswith('"Label"'):
            label = line.split("=")[-1].strip().strip('";')
            lines.append(f"Label: {label}")
    if len(lines) == 1:
        # Fallback: show raw output
        lines.append(f"<pre>{html.escape(output.strip())}</pre>")
    await _safe_reply(update, "\n".join(lines))


async def _svc_restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    label = _get_launchd_label(context)
    uid = os.getuid()
    await _safe_reply(update, "Restarting bot...")
    await _run_launchctl("kickstart", "-k", f"gui/{uid}/{label}")


async def _svc_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    plist = _get_plist_path(_get_launchd_label(context))
    await _safe_reply(update, "Stopping bot... (KeepAlive will NOT restart)")
    # unload prevents KeepAlive from restarting the process
    await _run_launchctl("unload", plist)


async def _svc_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    plist = _get_plist_path(_get_launchd_label(context))
    rc, output = await _run_launchctl("load", plist)
    if rc == 0:
        await _safe_reply(update, "Service loaded and starting.")
    else:
        await _safe_reply(update, f"Failed to start:\n<pre>{html.escape(output)}</pre>")


async def _svc_logs(update: Update, context: ContextTypes.DEFAULT_TYPE, _label: str = "") -> None:
    log_type = "stderr"
    if context.args and len(context.args) > 1:
        log_type = context.args[1].lower()

    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs"
    )
    if log_type == "stdout":
        log_file = os.path.join(log_dir, "launchd-stdout.log")
    else:
        log_file = os.path.join(log_dir, "launchd-stderr.log")

    try:
        proc = await asyncio.create_subprocess_exec(
            "tail", "-n", "40", log_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        output = stdout.decode(errors="replace") if stdout else "(empty)"
    except FileNotFoundError:
        output = f"(log file not found: {log_file})"
    except asyncio.TimeoutError:
        output = "(tail timed out)"

    output = mask_secrets(output)
    if len(output) > 4000:
        output = output[-4000:]
    await _safe_reply(update, f"<b>{html.escape(log_type)} logs</b>\n<pre>{_sanitize_output(output)}</pre>")


# ── /projects ────────────────────────────────────────────────────────────────

async def projects_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List registered projects from projects.json."""
    try:
        projects: dict = context.bot_data.get("projects", {})
        if not projects:
            await _safe_reply(update, "No projects registered. Edit <code>orchestrator/projects.json</code>.")
            return

        lines = ["<b>Registered Projects</b>"]
        for name, info in projects.items():
            path = info.get("path", "?")
            lines.append(f"  <code>{name}</code> → <code>{html.escape(path)}</code>")
        await _safe_reply(update, "\n".join(lines))
    except Exception:
        logger.exception("/projects handler error")
        await _safe_reply(update, "Error listing projects.")


# ── /issues ──────────────────────────────────────────────────────────────────

async def issues_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List open GitHub issues for a project: /issues <project>"""
    try:
        if not context.args:
            await _safe_reply(update, "Usage: <code>/issues &lt;project&gt;</code>")
            return

        project_name, err = _resolve_project(context.args[0], context.bot_data.get("projects", {}))
        if project_name is None:
            await _safe_reply(update, err)
            return
        projects: dict = context.bot_data.get("projects", {})

        project_path = projects[project_name]["path"]
        proc = await asyncio.create_subprocess_exec(
            "gh", "issue", "list", "--state", "open", "--limit", "10",
            "--json", "number,title,labels",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode(errors="replace") if stdout else ""

        if proc.returncode != 0:
            await _safe_reply(update, f"<code>gh</code> failed:\n<pre>{_sanitize_output(output)}</pre>")
            return

        import json
        issues = json.loads(output)
        if not issues:
            await _safe_reply(update, f"<b>{html.escape(project_name)}</b>: No open issues.")
            return

        # Build inline keyboard — each issue is a button that triggers solve
        buttons: list[list[InlineKeyboardButton]] = []
        lines = [f"<b>{html.escape(project_name)}</b> — Open Issues"]
        for issue in issues:
            num = issue["number"]
            title = issue["title"]
            label_tags = ""
            if issue.get("labels"):
                label_tags = " " + " ".join(
                    f"[{l['name']}]" for l in issue["labels"]
                )
            lines.append(f"  #{num} {html.escape(title)}{html.escape(label_tags)}")
            buttons.append([
                InlineKeyboardButton(
                    f"#{num} Solve",
                    callback_data=f"solve:{project_name}:{num}",
                )
            ])

        keyboard = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(  # type: ignore[union-attr]
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )
    except asyncio.TimeoutError:
        await _safe_reply(update, "Timed out fetching issues.")
    except Exception:
        logger.exception("/issues handler error")
        await _safe_reply(update, "Error fetching issues.")


# ── /solve ───────────────────────────────────────────────────────────────────

async def solve_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start solving issues: /solve <project> <issue#> [issue#] ..."""
    try:
        if not context.args or len(context.args) < 2:
            await _safe_reply(update, "Usage: <code>/solve &lt;project&gt; &lt;issue#&gt; [issue#] ...</code>")
            return

        project_name, err = _resolve_project(context.args[0], context.bot_data.get("projects", {}))
        if project_name is None:
            await _safe_reply(update, err)
            return
        projects: dict = context.bot_data.get("projects", {})

        # Parse issue numbers
        issue_nums: list[int] = []
        for arg in context.args[1:]:
            try:
                issue_nums.append(int(arg))
            except ValueError:
                await _safe_reply(update, f"Invalid issue number: <code>{html.escape(arg)}</code>")
                return

        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        await _start_solve(update, context, chat_id, project_name, issue_nums)
    except Exception:
        logger.exception("/solve handler error")
        await _safe_reply(update, "Error starting solve.")


async def solve_inline_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button click from /issues list: solve:<project>:<issue#>"""
    query = update.callback_query
    if query is None:
        return
    await query.answer()

    try:
        data = query.data or ""
        _, project_name, issue_str = data.split(":", 2)
        issue_num = int(issue_str)

        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        # Send a new message to start solve (don't edit the issues list)
        fake_update = update
        await _start_solve(fake_update, context, chat_id, project_name, [issue_num], from_callback=True)
    except Exception:
        logger.exception("solve inline callback error")
        if query:
            try:
                await query.edit_message_text("Error starting solve.")
            except Exception:
                pass


async def solve_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Cancel button press during solve."""
    query = update.callback_query
    if query is None:
        return
    await query.answer("Cancelling...")

    try:
        data = query.data or ""
        parts = data.split(":")
        # cancel_solve:<chat_id>:<issue_num> (new) or cancel_solve:<chat_id> (legacy)
        chat_id = int(parts[1])
        issue_num = int(parts[2]) if len(parts) > 2 else None

        events = _solve_cancels.get(chat_id)
        if events:
            if issue_num and issue_num in events:
                events[issue_num].set()
                await query.edit_message_text(f"Cancel requested for #{issue_num}. Stopping after current step...")
            else:
                # Cancel all issues for this chat
                for ev in events.values():
                    ev.set()
                await query.edit_message_text("Cancel requested for all issues. Stopping after current step...")
        else:
            await query.edit_message_text("No active solve session to cancel.")
    except Exception:
        logger.exception("solve cancel callback error")


async def _start_solve(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    issue_nums: list[int],
    *,
    from_callback: bool = False,
) -> None:
    """Guard duplicate issues and launch the solve loop."""
    # Block only if the same issue is already running
    existing = _solve_cancels.get(chat_id, {})
    duplicates = [num for num in issue_nums if num in existing]
    if duplicates:
        dup_str = ", ".join(f"#{n}" for n in duplicates)
        msg = f"Already running: {dup_str}. Use Cancel to stop first."
        if from_callback:
            await context.bot.send_message(chat_id, msg)
        else:
            await _safe_reply(update, msg)
        return

    _solve_active[chat_id] = _solve_active.get(chat_id, 0) + 1
    # Merge per-issue cancel events (don't overwrite existing)
    cancel_events = {num: asyncio.Event() for num in issue_nums}
    _solve_cancels.setdefault(chat_id, {}).update(cancel_events)

    projects: dict = context.bot_data.get("projects", {})
    project_path = projects[project_name]["path"]
    settings = context.bot_data["settings"]
    timeout = settings.solve_timeout

    # Run in background so the handler returns immediately
    asyncio.create_task(
        _solve_issues(context, chat_id, project_name, project_path, issue_nums, timeout, cancel_events)
    )

    nums_str = ", ".join(f"#{n}" for n in issue_nums)
    await context.bot.send_message(
        chat_id,
        f"Starting solve for <b>{html.escape(project_name)}</b> issues: {nums_str}",
        parse_mode=ParseMode.HTML,
    )


async def _solve_issues(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    project_path: str,
    issue_nums: list[int],
    timeout: int,
    cancel_events: dict[int, asyncio.Event],
) -> None:
    """Solve issues in parallel using git worktrees."""
    results: list[tuple[int, str, str]] = []  # (issue#, status, detail)

    try:
        # Ensure main repo is on 'main' so solve branches aren't "in use" by worktrees
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", "main",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)

        if len(issue_nums) == 1:
            # Single issue — no need for gather overhead
            num = issue_nums[0]
            status, detail = await _solve_single_issue(
                context, chat_id, project_name, project_path,
                num, timeout, cancel_events[num],
            )
            results.append((num, status, detail))
        else:
            # Multiple issues — run in parallel with independent cancel events
            tasks = [
                _solve_single_issue(
                    context, chat_id, project_name, project_path,
                    issue_num, timeout, cancel_events[issue_num],
                )
                for issue_num in issue_nums
            ]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            for issue_num, outcome in zip(issue_nums, outcomes):
                if isinstance(outcome, Exception):
                    logger.exception("Parallel solve error for #%d", issue_num, exc_info=outcome)
                    results.append((issue_num, "failed", str(outcome)[:100]))
                else:
                    results.append((issue_num, outcome[0], outcome[1]))
    except Exception:
        logger.exception("Solve loop error")
    finally:
        # Decrement active count; remove only this batch's cancel events
        count = _solve_active.get(chat_id, 1) - 1
        if count <= 0:
            _solve_active.pop(chat_id, None)
        else:
            _solve_active[chat_id] = count
        events = _solve_cancels.get(chat_id, {})
        for num in issue_nums:
            events.pop(num, None)
        if not events:
            _solve_cancels.pop(chat_id, None)

    # Summary message
    lines = [f"<b>Solve Summary — {html.escape(project_name)}</b>"]
    for num, status, detail in results:
        icon = {"success": "✅", "failed": "❌", "skipped": "⏭"}.get(status, "❓")
        lines.append(f"  {icon} #{num}: {html.escape(detail)}")

    # Mark remaining as skipped
    for num in issue_nums:
        if not any(r[0] == num for r in results):
            lines.append(f"  ⏭ #{num}: skipped")

    await context.bot.send_message(chat_id, "\n".join(lines), parse_mode=ParseMode.HTML)


async def _solve_single_issue(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    project_path: str,
    issue_num: int,
    timeout: int,
    cancel_event: asyncio.Event,
) -> tuple[str, str]:
    """Solve one issue. Routes to fivebrid, dual-check, or direct-claude based on settings."""
    settings = context.bot_data["settings"]

    if settings.pipeline_mode == "fivebrid":
        ollama: OllamaProvider | None = context.bot_data.get("ollama")
        gemini: GeminiCLIProvider | None = context.bot_data.get("gemini")
        if not ollama:
            return "failed", "Fivebrid pipeline requires Ollama but it is not configured"
        if not gemini:
            return "failed", "Fivebrid pipeline requires Gemini CLI but it is not available"
        return await _solve_with_fivebrid(
            context, chat_id, project_name, project_path,
            issue_num, timeout, cancel_event,
            ollama, gemini, settings,
        )
    elif settings.dual_check_enabled:
        ollama = context.bot_data.get("ollama")
        anthropic: AnthropicProvider | None = context.bot_data.get("anthropic")
        if not ollama:
            return "failed", "Dual-check requires Ollama but it is not configured"
        return await _solve_with_dual_check(
            context, chat_id, project_name, project_path,
            issue_num, timeout, cancel_event,
            ollama, anthropic, settings,
        )
    else:
        return await _solve_direct_claude(
            context, chat_id, project_name, project_path,
            issue_num, timeout, cancel_event,
        )


async def _solve_with_dual_check(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    project_path: str,
    issue_num: int,
    timeout: int,
    cancel_event: asyncio.Event,
    ollama: OllamaProvider,
    anthropic: AnthropicProvider | None,
    settings,
) -> tuple[str, str]:
    """Solve via 4-step dual-check pipeline."""
    branch_name = f"solve/issue-{issue_num}"

    cancel_btn = InlineKeyboardMarkup([
        [InlineKeyboardButton("Cancel", callback_data=f"cancel_solve:{chat_id}:{issue_num}")]
    ])
    msg = await context.bot.send_message(
        chat_id,
        f"<b>#{issue_num}</b> [0/4] Preparing git worktree...",
        parse_mode=ParseMode.HTML,
        reply_markup=cancel_btn,
    )

    pipeline_start = time.monotonic()

    worktree_dir = ""
    try:
        # ── Git Worktree Setup ──
        git_ok, git_result = await _git_fresh_start(project_path, branch_name)
        if not git_ok:
            await _edit_msg(msg, f"<b>#{issue_num}</b> — Git setup failed:\n<pre>{_sanitize_output(git_result)}</pre>")
            return "failed", f"Git setup failed: {git_result[:100]}"

        worktree_dir = git_result  # on success, this is the worktree path

        # Capture base commit for accurate diffing (only this session's changes)
        base_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=worktree_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        base_out, _ = await asyncio.wait_for(base_proc.communicate(), timeout=5)
        base_commit = base_out.decode().strip() if base_out else ""

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Build pipeline context — use worktree_dir as working directory
        ctx = PipelineContext(
            project_path=worktree_dir,
            project_name=project_name,
            issue_num=issue_num,
            branch_name=branch_name,
            base_commit=base_commit,
        )

        # Progress callback: update Telegram message
        async def progress_cb(status_text: str) -> None:
            elapsed = int(time.monotonic() - pipeline_start)
            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            try:
                sys_status = await get_system_status()
                sys_line = f"CPU: {sys_status.cpu_percent}% | RAM: {sys_status.ram_percent}%"
            except Exception:
                sys_line = ""

            text = (
                f"<b>#{issue_num}</b> {html.escape(status_text)}\n"
                f"[{time_str}] {sys_line}"
            )
            await _edit_msg(msg, text, reply_markup=cancel_btn)

        # Run pipeline
        status, detail = await run_dual_check_pipeline(
            ctx, ollama, anthropic, settings, cancel_event, progress_cb,
        )

        elapsed = int(time.monotonic() - pipeline_start)
        mins, secs = divmod(elapsed, 60)
        total_time = f"{mins}m {secs}s" if mins else f"{secs}s"

        if status == "success" and cancel_event.is_set():
            summary = format_pipeline_summary(ctx)
            await _edit_msg(
                msg,
                f"<b>#{issue_num}</b> ⏭ Cancelled before PR creation\n[{total_time}]\n\n"
                f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
            )
            return "skipped", "Cancelled before PR creation"

        if status == "success":
            # Create PR — push from worktree
            await _edit_msg(msg, f"<b>#{issue_num}</b> — Creating PR...")
            pr_url, pr_err = await _create_pr(worktree_dir, issue_num, branch_name, ctx.issue_title, ctx)

            summary = format_pipeline_summary(ctx)
            if pr_url:
                await _edit_msg(
                    msg,
                    f"<b>#{issue_num}</b> \u2705 Solved in {total_time}\nPR: {pr_url}\n\n"
                    f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
                )
                return "success", f"PR created: {pr_url}"
            else:
                await _edit_msg(
                    msg,
                    f"<b>#{issue_num}</b> — Pipeline passed but PR failed:\n"
                    f"<pre>{_sanitize_output(pr_err)}</pre>\n\n"
                    f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
                )
                return "failed", f"PR creation failed: {pr_err[:100]}"
        else:
            summary = format_pipeline_summary(ctx)
            await _edit_msg(
                msg,
                f"<b>#{issue_num}</b> \u274c {html.escape(status)}: {html.escape(detail[:200])}\n"
                f"[{total_time}]\n\n"
                f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
            )
            return status, detail

    except Exception as exc:
        logger.exception("Error in dual-check pipeline for issue #%d", issue_num)
        await _edit_msg(msg, f"<b>#{issue_num}</b> — Error: {html.escape(str(exc)[:200])}")
        return "failed", str(exc)[:100]
    finally:
        if worktree_dir:
            await _cleanup_worktree(project_path, worktree_dir, branch_name)


async def _solve_with_fivebrid(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    project_path: str,
    issue_num: int,
    timeout: int,
    cancel_event: asyncio.Event,
    ollama: OllamaProvider,
    gemini: GeminiCLIProvider,
    settings,
) -> tuple[str, str]:
    """Solve via 9-step Five-brid pipeline."""
    branch_name = f"solve/issue-{issue_num}"

    cancel_btn = InlineKeyboardMarkup([
        [InlineKeyboardButton("Cancel", callback_data=f"cancel_solve:{chat_id}:{issue_num}")]
    ])
    msg = await context.bot.send_message(
        chat_id,
        f"<b>#{issue_num}</b> [0/8] Preparing git worktree (Five-brid)...",
        parse_mode=ParseMode.HTML,
        reply_markup=cancel_btn,
    )

    pipeline_start = time.monotonic()

    worktree_dir = ""
    try:
        # ── Git Worktree Setup ──
        git_ok, git_result = await _git_fresh_start(project_path, branch_name)
        if not git_ok:
            await _edit_msg(msg, f"<b>#{issue_num}</b> — Git setup failed:\n<pre>{_sanitize_output(git_result)}</pre>")
            return "failed", f"Git setup failed: {git_result[:100]}"

        worktree_dir = git_result

        # Capture base commit for accurate diffing
        base_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=worktree_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        base_out, _ = await asyncio.wait_for(base_proc.communicate(), timeout=5)
        base_commit = base_out.decode().strip() if base_out else ""

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Build pipeline context
        ctx = PipelineContext(
            project_path=worktree_dir,
            project_name=project_name,
            issue_num=issue_num,
            branch_name=branch_name,
            base_commit=base_commit,
        )

        # Progress callback
        async def progress_cb(status_text: str) -> None:
            elapsed = int(time.monotonic() - pipeline_start)
            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            try:
                sys_status = await get_system_status()
                sys_line = f"CPU: {sys_status.cpu_percent}% | RAM: {sys_status.ram_percent}%"
            except Exception:
                sys_line = ""

            text = (
                f"<b>#{issue_num}</b> {html.escape(status_text)}\n"
                f"[{time_str}] {sys_line}"
            )
            await _edit_msg(msg, text, reply_markup=cancel_btn)

        # Run fivebrid pipeline
        status, detail = await run_fivebrid_pipeline(
            ctx, ollama, gemini, settings, cancel_event, progress_cb,
        )

        elapsed = int(time.monotonic() - pipeline_start)
        mins, secs = divmod(elapsed, 60)
        total_time = f"{mins}m {secs}s" if mins else f"{secs}s"

        if status == "success" and cancel_event.is_set():
            summary = format_pipeline_summary(ctx)
            await _edit_msg(
                msg,
                f"<b>#{issue_num}</b> \u23ed Cancelled before PR creation\n[{total_time}]\n\n"
                f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
            )
            return "skipped", "Cancelled before PR creation"

        if status == "success":
            await _edit_msg(msg, f"<b>#{issue_num}</b> — Creating PR...")
            pr_url, pr_err = await _create_pr(worktree_dir, issue_num, branch_name, ctx.issue_title, ctx)

            summary = format_pipeline_summary(ctx)
            if pr_url:
                await _edit_msg(
                    msg,
                    f"<b>#{issue_num}</b> \u2705 Solved in {total_time}\nPR: {pr_url}\n\n"
                    f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
                )
                return "success", f"PR created: {pr_url}"
            else:
                await _edit_msg(
                    msg,
                    f"<b>#{issue_num}</b> — Pipeline passed but PR failed:\n"
                    f"<pre>{_sanitize_output(pr_err)}</pre>\n\n"
                    f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
                )
                return "failed", f"PR creation failed: {pr_err[:100]}"
        else:
            summary = format_pipeline_summary(ctx)
            await _edit_msg(
                msg,
                f"<b>#{issue_num}</b> \u274c {html.escape(status)}: {html.escape(detail[:200])}\n"
                f"[{total_time}]\n\n"
                f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
            )
            return status, detail

    except Exception as exc:
        logger.exception("Error in fivebrid pipeline for issue #%d", issue_num)
        await _edit_msg(msg, f"<b>#{issue_num}</b> — Error: {html.escape(str(exc)[:200])}")
        return "failed", str(exc)[:100]
    finally:
        if worktree_dir:
            await _cleanup_worktree(project_path, worktree_dir, branch_name)


async def _solve_direct_claude(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    project_path: str,
    issue_num: int,
    timeout: int,
    cancel_event: asyncio.Event,
) -> tuple[str, str]:
    """Original Claude-only solve logic (dual_check_enabled=false)."""
    branch_name = f"solve/issue-{issue_num}"

    cancel_btn = InlineKeyboardMarkup([
        [InlineKeyboardButton("Cancel", callback_data=f"cancel_solve:{chat_id}:{issue_num}")]
    ])
    msg = await context.bot.send_message(
        chat_id,
        f"<b>#{issue_num}</b> — Preparing git worktree...",
        parse_mode=ParseMode.HTML,
        reply_markup=cancel_btn,
    )

    worktree_dir = ""
    try:
        # ── Git Worktree Setup ──
        git_ok, git_result = await _git_fresh_start(project_path, branch_name)
        if not git_ok:
            await _edit_msg(msg, f"<b>#{issue_num}</b> — Git setup failed:\n<pre>{_sanitize_output(git_result)}</pre>")
            return "failed", f"Git setup failed: {git_result[:100]}"

        worktree_dir = git_result

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # ── Claude execution ──
        await _edit_msg(
            msg,
            f"<b>#{issue_num}</b> — Running Claude...",
            reply_markup=cancel_btn,
        )

        claude_cmd = (
            f'claude -p --dangerously-skip-permissions '
            f'"Read .claude/CLAUDE.md first and follow the defined pipeline. '
            f'Then read GitHub issue #{issue_num} with gh issue view {issue_num}. '
            f'Implement the solution. Complete all steps including testing."'
        )

        start = time.monotonic()
        proc = await asyncio.create_subprocess_shell(
            claude_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=worktree_dir,
        )
        assert proc.stdout is not None

        collected = ""

        async def _read_output() -> None:
            nonlocal collected
            async for line in proc.stdout:  # type: ignore[union-attr]
                collected += line.decode(errors="replace")

        read_task = asyncio.create_task(_read_output())

        # Streaming updates every 10s
        while not read_task.done():
            await asyncio.sleep(10)

            if cancel_event.is_set():
                proc.kill()
                await read_task
                return "skipped", "Cancelled by user"

            elapsed = int(time.monotonic() - start)
            if elapsed > timeout:
                proc.kill()
                await read_task
                return "failed", f"Timed out after {timeout}s"

            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"

            # Get system stats for the progress message
            try:
                sys_status = await get_system_status()
                sys_line = f"CPU: {sys_status.cpu_percent}% | RAM: {sys_status.ram_percent}%"
            except Exception:
                sys_line = ""

            preview = _sanitize_output(mask_secrets(collected[-500:])) if collected else "(waiting...)"
            text = (
                f"<b>#{issue_num}</b> — Running Claude [{time_str}]\n"
                f"{sys_line}\n"
                f"<pre>{preview}</pre>"
            )
            await _edit_msg(msg, text, reply_markup=cancel_btn)

        try:
            await asyncio.wait_for(read_task, timeout=10)
        except asyncio.TimeoutError:
            pass
        await proc.wait()

        elapsed = int(time.monotonic() - start)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        rc = proc.returncode

        if rc != 0:
            preview = _sanitize_output(mask_secrets(collected[-1000:])) if collected else "(no output)"
            await _edit_msg(
                msg,
                f"<b>#{issue_num}</b> — Claude failed (exit={rc}, {time_str})\n<pre>{preview}</pre>",
            )
            return "failed", f"Claude exit={rc} after {time_str}"

        # ── Cancel check before PR ──
        if cancel_event.is_set():
            return "skipped", f"Cancelled before PR creation (Claude done in {time_str})"

        # ── Auto PR ──
        await _edit_msg(msg, f"<b>#{issue_num}</b> — Claude done ({time_str}). Creating PR...")

        pr_url, pr_err = await _create_pr(worktree_dir, issue_num, branch_name)
        if pr_url:
            await _edit_msg(msg, f"<b>#{issue_num}</b> \u2705 Solved in {time_str}\nPR: {pr_url}")
            return "success", f"PR created: {pr_url}"
        else:
            await _edit_msg(
                msg,
                f"<b>#{issue_num}</b> — Claude done but PR failed:\n<pre>{_sanitize_output(pr_err)}</pre>",
            )
            return "failed", f"PR creation failed: {pr_err[:100]}"

    except Exception as exc:
        logger.exception("Error solving issue #%d", issue_num)
        await _edit_msg(msg, f"<b>#{issue_num}</b> — Error: {html.escape(str(exc)[:200])}")
        return "failed", str(exc)[:100]
    finally:
        if worktree_dir:
            await _cleanup_worktree(project_path, worktree_dir, branch_name)


async def _git_fresh_start(project_path: str, branch_name: str) -> tuple[bool, str]:
    """Create a git worktree for the branch. Returns (ok, error_or_worktree_path).

    On success, the second element is the worktree path.
    On failure, the second element is the error message.
    """
    import os
    import shutil

    worktree_dir = os.path.join(project_path, ".worktrees", branch_name.replace("/", "-"))

    # 1. Force remove worktree directory if it exists
    if os.path.exists(worktree_dir):
        shutil.rmtree(worktree_dir, ignore_errors=True)

    # 3. Prune stale worktree references
    proc = await asyncio.create_subprocess_exec(
        "git", "worktree", "prune",
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    await asyncio.wait_for(proc.communicate(), timeout=10)

    # 4. Force delete branch if it exists (from a previous run)
    proc = await asyncio.create_subprocess_exec(
        "git", "branch", "-D", branch_name,
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    await asyncio.wait_for(proc.communicate(), timeout=10)
    # Ignore errors — branch may not exist

    # 5. Fetch latest main
    proc = await asyncio.create_subprocess_exec(
        "git", "fetch", "origin", "main",
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    if proc.returncode != 0:
        output = stdout.decode(errors="replace") if stdout else ""
        return False, f"git fetch failed: {output}"

    # 5. Create worktree with new branch based on origin/main
    proc = await asyncio.create_subprocess_exec(
        "git", "worktree", "add", "-B", branch_name, worktree_dir, "origin/main",
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
    if proc.returncode != 0:
        output = stdout.decode(errors="replace") if stdout else ""
        return False, f"git worktree add failed: {output}"

    return True, worktree_dir


async def _cleanup_worktree(project_path: str, worktree_dir: str, branch_name: str) -> None:
    """Remove worktree and optionally the branch after solve completes."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "remove", "--force", worktree_dir,
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await asyncio.wait_for(proc.communicate(), timeout=15)
    except Exception:
        logger.warning("Failed to remove worktree %s", worktree_dir)


def _build_pr_body(issue_num: int, ctx: PipelineContext | None = None) -> str:
    """Build a rich PR body from pipeline context."""
    sections: list[str] = []

    sections.append(f"Closes #{issue_num}")

    if ctx and ctx.design_doc:
        # Trim design doc to first ~1500 chars
        design_summary = ctx.design_doc[:1500]
        if len(ctx.design_doc) > 1500:
            design_summary += "\n\n_(truncated)_"
        sections.append(f"## Design\n{design_summary}")

    if ctx and ctx.git_diff:
        # Count changed files from diff
        diff_files = [
            line.split(" b/")[-1]
            for line in ctx.git_diff.split("\n")
            if line.startswith("diff --git")
        ]
        if diff_files:
            file_list = "\n".join(f"- `{f}`" for f in diff_files[:20])
            sections.append(f"## Changed Files ({len(diff_files)})\n{file_list}")

    if ctx and ctx.review_report:
        review_summary = ctx.review_report[:800]
        if len(ctx.review_report) > 800:
            review_summary += "\n\n_(truncated)_"
        sections.append(f"## Review\n{review_summary}")

    if ctx and ctx.audit_result:
        verdict = "PASS" if ctx.audit_passed else "FAIL"
        sections.append(f"## Audit: {verdict}")

    return "\n\n".join(sections)


async def _create_pr(
    project_path: str,
    issue_num: int,
    branch_name: str,
    issue_title: str = "",
    ctx: PipelineContext | None = None,
) -> tuple[str, str]:
    """Squash commits, push branch, and create PR. Returns (pr_url, error_msg)."""
    # Build PR title from issue title
    if issue_title:
        pr_title = f"feat: {issue_title}"
    else:
        pr_title = f"feat: resolve #{issue_num}"
    squash_msg = f"{pr_title}\n\nCloses #{issue_num}"

    # Squash all commits on this branch into one
    proc = await asyncio.create_subprocess_exec(
        "git", "reset", "--soft", "origin/main",
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    await asyncio.wait_for(proc.communicate(), timeout=15)

    proc = await asyncio.create_subprocess_exec(
        "git", "commit", "-m", squash_msg,
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
    if proc.returncode != 0:
        output = stdout.decode(errors="replace") if stdout else "squash commit failed"
        logger.warning("Squash commit failed for #%d: %s", issue_num, output[:200])

    # Push
    proc = await asyncio.create_subprocess_exec(
        "git", "push", "-u", "origin", branch_name,
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
    if proc.returncode != 0:
        return "", stdout.decode(errors="replace") if stdout else "push failed"

    # Create PR
    pr_body = _build_pr_body(issue_num, ctx)
    proc = await asyncio.create_subprocess_exec(
        "gh", "pr", "create",
        "--title", pr_title,
        "--body", pr_body,
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
    output = stdout.decode(errors="replace").strip() if stdout else ""
    if proc.returncode != 0:
        return "", output or "gh pr create failed"

    return output, ""


async def _edit_msg(msg, text: str, reply_markup=None) -> None:
    """Safely edit a message, ignoring errors."""
    try:
        await msg.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
    except Exception:
        try:
            # Fallback: strip HTML
            import re as _re
            plain = _re.sub(r"<[^>]+>", "", text)
            await msg.edit_text(plain, reply_markup=reply_markup)
        except Exception:
            pass


# ── /rebase ──────────────────────────────────────────────────────────────────

async def rebase_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Rebase a PR branch onto main: /rebase <project> <pr#>"""
    try:
        if not context.args or len(context.args) < 2:
            await _safe_reply(update, "Usage: <code>/rebase &lt;project&gt; &lt;pr#&gt;</code>")
            return

        project_name, err = _resolve_project(context.args[0], context.bot_data.get("projects", {}))
        if project_name is None:
            await _safe_reply(update, err)
            return
        projects: dict = context.bot_data.get("projects", {})

        try:
            pr_number = int(context.args[1])
        except ValueError:
            await _safe_reply(update, f"Invalid PR number: <code>{html.escape(context.args[1])}</code>")
            return

        project_path = projects[project_name]["path"]
        settings = context.bot_data["settings"]
        chat_id = update.effective_chat.id  # type: ignore[union-attr]

        # Get branch name from PR
        proc = await asyncio.create_subprocess_exec(
            "gh", "pr", "view", str(pr_number), "--json", "headRefName",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode != 0:
            output = stdout.decode(errors="replace") if stdout else "gh pr view failed"
            await _safe_reply(update, f"Failed to get PR info:\n<pre>{_sanitize_output(output)}</pre>")
            return

        import json as _json
        pr_info = _json.loads(stdout.decode(errors="replace"))
        branch_name = pr_info["headRefName"]

        msg = await context.bot.send_message(
            chat_id,
            f"<b>#{pr_number}</b> Rebasing <code>{html.escape(branch_name)}</code> onto main...",
            parse_mode=ParseMode.HTML,
        )

        # Run rebase in background
        asyncio.create_task(
            _rebase_pr(context, chat_id, msg, project_path, pr_number, branch_name, settings)
        )
    except Exception:
        logger.exception("/rebase handler error")
        await _safe_reply(update, "Error starting rebase.")


async def _rebase_pr(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    msg,
    project_path: str,
    pr_number: int,
    branch_name: str,
    settings,
) -> None:
    """Rebase a PR branch onto origin/main, using Claude to resolve conflicts if needed."""
    worktree_dir = os.path.join(project_path, ".worktrees", f"rebase-{branch_name.replace('/', '-')}")

    try:
        # ── Step A: Git preparation ──
        import shutil
        if os.path.exists(worktree_dir):
            shutil.rmtree(worktree_dir, ignore_errors=True)

        # Prune stale worktrees
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "prune",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)

        # Fetch latest main and the PR branch
        proc = await asyncio.create_subprocess_exec(
            "git", "fetch", "origin", "main", branch_name,
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            output = stdout.decode(errors="replace") if stdout else ""
            await _edit_msg(msg, f"<b>#{pr_number}</b> ❌ Fetch failed:\n<pre>{_sanitize_output(output)}</pre>")
            return

        # Create worktree on the existing PR branch
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "add", worktree_dir, f"origin/{branch_name}",
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode != 0:
            output = stdout.decode(errors="replace") if stdout else ""
            await _edit_msg(msg, f"<b>#{pr_number}</b> ❌ Worktree setup failed:\n<pre>{_sanitize_output(output)}</pre>")
            return

        # Ensure the worktree tracks the remote branch for push
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", "-B", branch_name, f"origin/{branch_name}",
            cwd=worktree_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)

        # ── Step B: Rebase attempt ──
        proc = await asyncio.create_subprocess_exec(
            "git", "rebase", "origin/main",
            cwd=worktree_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        rebase_rc = proc.returncode

        if rebase_rc != 0:
            # ── Step C: Claude resolves conflicts ──
            await _edit_msg(
                msg,
                f"<b>#{pr_number}</b> Rebase conflict detected, resolving with Claude...",
            )

            # Abort the failed rebase first so Claude starts clean
            proc = await asyncio.create_subprocess_exec(
                "git", "rebase", "--abort",
                cwd=worktree_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10)

            claude_cmd = (
                'claude -p --dangerously-skip-permissions '
                '"There are git merge conflicts from rebasing onto main. '
                'First run: git rebase origin/main. '
                'Then resolve ALL conflicts in the current working directory. '
                'For each conflicted file, read it, resolve the conflict markers, and write the fixed version. '
                'Then run: git add <file> for each resolved file. '
                'After ALL conflicts are resolved, run: git rebase --continue. '
                'Repeat until the rebase is fully complete. '
                'Do NOT create branches, push, or create PRs. '
                'GIT RESTRICTIONS: Do NOT run git push or gh pr create."'
            )

            claude_proc = await asyncio.create_subprocess_shell(
                claude_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=worktree_dir,
            )
            claude_stdout, _ = await asyncio.wait_for(
                claude_proc.communicate(), timeout=settings.solve_timeout
            )

            if claude_proc.returncode != 0:
                # Abort rebase if Claude failed
                proc = await asyncio.create_subprocess_exec(
                    "git", "rebase", "--abort",
                    cwd=worktree_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await asyncio.wait_for(proc.communicate(), timeout=10)

                output = claude_stdout.decode(errors="replace") if claude_stdout else "(no output)"
                preview = _sanitize_output(mask_secrets(output[-500:]))
                await _edit_msg(
                    msg,
                    f"<b>#{pr_number}</b> ❌ Rebase failed: Claude could not resolve conflicts\n<pre>{preview}</pre>",
                )
                return

        # ── Step D: Force push ──
        proc = await asyncio.create_subprocess_exec(
            "git", "push", "--force-with-lease", "origin", branch_name,
            cwd=worktree_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            output = stdout.decode(errors="replace") if stdout else ""
            await _edit_msg(msg, f"<b>#{pr_number}</b> ❌ Push failed:\n<pre>{_sanitize_output(output)}</pre>")
            return

        await _edit_msg(msg, f"<b>#{pr_number}</b> ✅ Rebased and pushed successfully")

    except asyncio.TimeoutError:
        await _edit_msg(msg, f"<b>#{pr_number}</b> ❌ Rebase timed out")
    except Exception as exc:
        logger.exception("Error rebasing PR #%d", pr_number)
        await _edit_msg(msg, f"<b>#{pr_number}</b> ❌ Rebase failed: {html.escape(str(exc)[:200])}")
    finally:
        # ── Step E: Cleanup worktree ──
        await _cleanup_worktree(project_path, worktree_dir, branch_name="")


# ── /extract ──────────────────────────────────────────────────────────────────

async def extract_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate training data from a file: /extract <project> <file_path>"""
    try:
        if not context.args or len(context.args) < 2:
            await _safe_reply(update, "Usage: <code>/extract &lt;project&gt; &lt;file_path&gt;</code>")
            return

        project_name, err = _resolve_project(context.args[0], context.bot_data.get("projects", {}))
        if project_name is None:
            await _safe_reply(update, err)
            return
        projects: dict = context.bot_data.get("projects", {})
        file_rel_path = " ".join(context.args[1:])

        project_path = projects[project_name]["path"]
        file_path = os.path.join(project_path, file_rel_path)

        if not os.path.isfile(file_path):
            await _safe_reply(update, f"File not found: <code>{html.escape(file_rel_path)}</code>")
            return

        settings = context.bot_data["settings"]
        ollama: OllamaProvider | None = context.bot_data.get("ollama")
        if not ollama:
            await _safe_reply(update, "Ollama is not configured.")
            return

        # Read file content (cap at 50K chars)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(50_000)

        await _safe_reply(update, f"Generating training data from <code>{html.escape(file_rel_path)}</code>...")

        from .ai.base import Message, Role

        system_prompt = (
            "You are a training data generator. Analyze the source code. "
            "Output JSONL (one JSON object per line). Each object must have exactly two keys: "
            '"instruction" (a natural-language coding task) and "output" (the code that solves it). '
            "Output ONLY valid JSONL lines. No markdown fences. No explanations."
        )
        user_content = (
            f"File: {file_rel_path}\n\n"
            f"```\n{content}\n```\n\n"
            "Generate instruction-output pairs as JSONL for fine-tuning a code model."
        )

        messages = [Message(role=Role.USER, content=user_content)]
        response = await ollama.chat(
            messages,
            max_tokens=settings.data_mining_max_tokens,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=settings.data_mining_timeout,
            model=settings.qwen_coder_model,
        )

        result = response.content.strip()

        if len(result) <= 4000:
            await _safe_reply(update, f"<pre>{_sanitize_output(result)}</pre>")
        else:
            # Send as document to avoid Telegram message truncation
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", prefix="extract-", delete=False, encoding="utf-8",
            ) as tmp:
                tmp.write(result)
                tmp_path = tmp.name

            try:
                await update.message.reply_document(  # type: ignore[union-attr]
                    document=open(tmp_path, "rb"),
                    filename=f"extract-{project_name}-{os.path.basename(file_rel_path)}.jsonl",
                    caption=f"Training data from {file_rel_path} ({len(result)} chars)",
                )
            finally:
                os.unlink(tmp_path)

    except Exception:
        logger.exception("/extract handler error")
        await _safe_reply(update, "Error generating training data.")


# ── /init ────────────────────────────────────────────────────────────────────

_PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


async def init_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bootstrap a new project: /init [--public|--private] <project_name> <description...>"""
    try:
        if not context.args or len(context.args) < 2:
            await _safe_reply(
                update,
                "Usage: <code>/init [--public|--private] &lt;name&gt; &lt;description&gt;</code>\n"
                "Example: <code>/init --public my-app A KMP mobile app</code>",
            )
            return

        args = list(context.args)

        # Parse --public / --private flag
        visibility_override: str | None = None
        if args[0] in ("--public", "--private"):
            visibility_override = args.pop(0).lstrip("-")  # "public" or "private"

        if len(args) < 2:
            await _safe_reply(update, "Missing project name or description.")
            return

        project_name = args[0]
        description = " ".join(args[1:])

        # Validate project name
        if not _PROJECT_NAME_RE.match(project_name):
            await _safe_reply(
                update,
                f"Invalid project name: <code>{html.escape(project_name)}</code>\n"
                f"Must match: <code>[a-z0-9][a-z0-9-]*</code>",
            )
            return

        settings: Settings = context.bot_data["settings"]

        if not settings.github_user:
            await _safe_reply(update, "GITHUB_USER is not set in .env")
            return

        projects: dict = context.bot_data.get("projects", {})

        # Duplicate check
        if project_name in projects:
            await _safe_reply(update, f"Project <b>{html.escape(project_name)}</b> already exists in projects.json")
            return

        # Directory check
        base_dir = os.path.expanduser(settings.projects_base_dir)
        project_path = os.path.join(base_dir, project_name)
        if os.path.exists(project_path):
            await _safe_reply(update, f"Directory already exists: <code>{html.escape(project_path)}</code>")
            return

        chat_id = update.effective_chat.id  # type: ignore[union-attr]

        # Prevent double-init
        if chat_id in _init_cancels:
            await _safe_reply(update, "An init is already running. Cancel it first.")
            return

        cancel_event = asyncio.Event()
        _init_cancels[chat_id] = cancel_event

        visibility = visibility_override or settings.default_repo_visibility

        # Background task
        asyncio.create_task(
            _run_init(context, chat_id, project_name, description, project_path, settings, cancel_event, visibility)
        )

        vis_label = f" ({visibility})" if visibility_override else ""
        await _safe_reply(
            update,
            f"Bootstrapping <b>{html.escape(project_name)}</b>{vis_label}...\n"
            f"<i>{html.escape(description)}</i>",
        )
    except Exception:
        logger.exception("/init handler error")
        await _safe_reply(update, "Error starting init.")


def _find_reference_claude_md(projects: dict[str, dict]) -> str:
    """Find a reference CLAUDE.md using 3-tier priority."""
    # Priority 1: Global template
    global_template = os.path.expanduser("~/.ai_orchestrator/templates/CLAUDE.md.template")
    if os.path.isfile(global_template):
        try:
            with open(global_template, encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                logger.info("Using global CLAUDE.md template: %s", global_template)
                return content
        except Exception:
            pass

    # Priority 2: Most recently modified CLAUDE.md from existing projects
    candidates: list[tuple[float, str]] = []
    for info in projects.values():
        p = info.get("path", "")
        if not p:
            continue
        claude_path = os.path.join(p, ".claude", "CLAUDE.md")
        if os.path.isfile(claude_path):
            try:
                mtime = os.path.getmtime(claude_path)
                candidates.append((mtime, claude_path))
            except Exception:
                pass

    if candidates:
        candidates.sort(reverse=True)  # newest first
        best_path = candidates[0][1]
        try:
            with open(best_path, encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                logger.info("Using reference CLAUDE.md: %s", best_path)
                return content
        except Exception:
            pass

    # Priority 3: Empty (Opus generates from scratch)
    logger.info("No reference CLAUDE.md found — Opus will generate from scratch")
    return ""


async def _run_init(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    description: str,
    project_path: str,
    settings,
    cancel_event: asyncio.Event,
    visibility: str = "private",
) -> None:
    """Background task that runs the 4-step init pipeline."""
    cancel_btn = InlineKeyboardMarkup([
        [InlineKeyboardButton("Cancel", callback_data=f"cancel_init:{chat_id}")]
    ])
    msg = await context.bot.send_message(
        chat_id,
        f"<b>{html.escape(project_name)}</b> [0/4] Starting init pipeline...",
        parse_mode=ParseMode.HTML,
        reply_markup=cancel_btn,
    )

    pipeline_start = time.monotonic()

    try:
        # Find reference CLAUDE.md
        projects: dict = context.bot_data.get("projects", {})
        reference_claude_md = _find_reference_claude_md(projects)

        ctx = InitContext(
            project_name=project_name,
            description=description,
            project_path=project_path,
            github_user=settings.github_user,
            repo_visibility=visibility,
        )

        async def progress_cb(status_text: str) -> None:
            elapsed = int(time.monotonic() - pipeline_start)
            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            text = (
                f"<b>{html.escape(project_name)}</b> {html.escape(status_text)}\n"
                f"[{time_str}]"
            )
            await _edit_msg(msg, text, reply_markup=cancel_btn)

        status, detail = await run_init_pipeline(
            ctx, settings, cancel_event, progress_cb, reference_claude_md,
        )

        elapsed = int(time.monotonic() - pipeline_start)
        mins, secs = divmod(elapsed, 60)
        total_time = f"{mins}m {secs}s" if mins else f"{secs}s"

        # Format pipeline summary (works with InitContext since it has .steps)
        summary_lines: list[str] = []
        for s in ctx.steps:
            icon = {
                "passed": "\u2705", "failed": "\u274c", "skipped": "\u23ed",
                "running": "\u23f3", "pending": "\u2b1c",
            }.get(s.status, "\u2753")
            elapsed_str = ""
            if s.elapsed_sec > 0:
                m, sc = divmod(int(s.elapsed_sec), 60)
                elapsed_str = f" ({m}m {sc}s)" if m else f" ({sc}s)"
            summary_lines.append(f"  {icon} {s.name}{elapsed_str}")
        summary = "\n".join(summary_lines)

        if status == "success":
            # Register in projects.json
            from .config import save_projects

            projects[project_name] = {"path": project_path}
            save_projects(projects)
            context.bot_data["projects"] = projects

            # Build issues list
            issues_text = ""
            if ctx.issues_created:
                issue_lines = []
                for iss in ctx.issues_created:
                    url = iss.get("url", "")
                    title = iss.get("title", "")
                    if url:
                        issue_lines.append(f"  - <a href=\"{url}\">{html.escape(title)}</a>")
                    else:
                        issue_lines.append(f"  - {html.escape(title)}")
                issues_text = "\n<b>Issues Created:</b>\n" + "\n".join(issue_lines)

            repo_text = f"\nRepo: {ctx.repo_url}" if ctx.repo_url else ""

            await _edit_msg(
                msg,
                f"<b>{html.escape(project_name)}</b> \u2705 Bootstrapped in {total_time}\n"
                f"{repo_text}\n\n"
                f"<b>Pipeline Steps:</b>\n{html.escape(summary)}"
                f"{issues_text}\n\n"
                f"Ready for <code>/solve {html.escape(project_name)} 1</code>",
            )
        else:
            await _edit_msg(
                msg,
                f"<b>{html.escape(project_name)}</b> \u274c {html.escape(status)}: "
                f"{html.escape(detail[:200])}\n[{total_time}]\n\n"
                f"<b>Pipeline Steps:</b>\n{html.escape(summary)}",
            )
    except Exception as exc:
        logger.exception("Init pipeline error for %s", project_name)
        await _edit_msg(msg, f"<b>{html.escape(project_name)}</b> \u274c Error: {html.escape(str(exc)[:200])}")
    finally:
        _init_cancels.pop(chat_id, None)


async def init_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Cancel button press during init."""
    query = update.callback_query
    if query is None:
        return
    await query.answer("Cancelling...")

    try:
        data = query.data or ""
        parts = data.split(":")
        chat_id = int(parts[1])

        event = _init_cancels.get(chat_id)
        if event:
            event.set()
            await query.edit_message_text("Cancel requested. Stopping after current step...")
        else:
            await query.edit_message_text("No active init session to cancel.")
    except Exception:
        logger.exception("init cancel callback error")


# ── /plan ─────────────────────────────────────────────────────────────────────


async def plan_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/plan <project> — Plan next-stage issues for an existing project."""
    try:
        if not context.args:
            await _safe_reply(update, "Usage: <code>/plan &lt;project&gt;</code>")
            return

        project_name, err = _resolve_project(context.args[0], context.bot_data.get("projects", {}))
        if project_name is None:
            await _safe_reply(update, err)
            return
        projects: dict = context.bot_data.get("projects", {})
        settings: Settings = context.bot_data["settings"]

        if not settings.github_user:
            await _safe_reply(update, "GITHUB_USER is not set in .env")
            return

        chat_id = update.effective_chat.id  # type: ignore[union-attr]

        if chat_id in _plan_cancels:
            await _safe_reply(update, "A plan is already running. Cancel it first.")
            return

        cancel_event = asyncio.Event()
        _plan_cancels[chat_id] = cancel_event

        project_path = projects[project_name].get("path", "")

        asyncio.create_task(
            _run_plan(context, chat_id, project_name, project_path, settings, cancel_event)
        )

        await _safe_reply(
            update,
            f"Planning next-stage issues for <b>{html.escape(project_name)}</b>...",
        )
    except Exception:
        logger.exception("/plan handler error")
        await _safe_reply(update, "Error starting plan.")


async def _run_plan(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    project_path: str,
    settings: Settings,
    cancel_event: asyncio.Event,
) -> None:
    """Background task that plans next-stage issues."""
    cancel_btn = InlineKeyboardMarkup([
        [InlineKeyboardButton("Cancel", callback_data=f"cancel_plan:{chat_id}")]
    ])
    msg = await context.bot.send_message(
        chat_id,
        f"<b>{html.escape(project_name)}</b> Fetching existing issues...",
        parse_mode=ParseMode.HTML,
        reply_markup=cancel_btn,
    )

    pipeline_start = time.monotonic()

    try:
        # 1. Fetch existing issues (open + closed)
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

        # 2. Read CLAUDE.md
        claude_md_path = os.path.join(project_path, ".claude", "CLAUDE.md")
        claude_md = ""
        if os.path.isfile(claude_md_path):
            try:
                with open(claude_md_path, encoding="utf-8") as f:
                    claude_md = f.read()
            except Exception:
                pass

        # 3. Cancel check
        if cancel_event.is_set():
            await _edit_msg(msg, f"<b>{html.escape(project_name)}</b> Plan cancelled.")
            return

        # 4. Call step_plan_issues
        async def progress_cb(status_text: str) -> None:
            elapsed = int(time.monotonic() - pipeline_start)
            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            text = (
                f"<b>{html.escape(project_name)}</b> {html.escape(status_text)}\n"
                f"[{time_str}]"
            )
            await _edit_msg(msg, text, reply_markup=cancel_btn)

        created, total, issues_list = await step_plan_issues(
            project_name=project_name,
            project_path=project_path,
            github_user=settings.github_user,
            existing_issues_text=existing_issues_text,
            claude_md=claude_md,
            settings=settings,
            progress_cb=progress_cb,
        )

        # 5. Report results
        elapsed = int(time.monotonic() - pipeline_start)
        mins, secs = divmod(elapsed, 60)
        total_time = f"{mins}m {secs}s" if mins else f"{secs}s"

        issue_lines = []
        for iss in issues_list:
            url = iss.get("url", "")
            title = iss.get("title", "")
            if url:
                issue_lines.append(f"  - <a href=\"{url}\">{html.escape(title)}</a>")
            else:
                issue_lines.append(f"  - {html.escape(title)}")
        issues_text = "\n".join(issue_lines) if issue_lines else "  (none)"

        await _edit_msg(
            msg,
            f"<b>{html.escape(project_name)}</b> \u2705 Plan complete in {total_time}\n"
            f"Issues created: {created}/{total}\n\n"
            f"{issues_text}",
        )
    except Exception as exc:
        logger.exception("Plan pipeline error for %s", project_name)
        await _edit_msg(msg, f"<b>{html.escape(project_name)}</b> \u274c Error: {html.escape(str(exc)[:200])}")
    finally:
        _plan_cancels.pop(chat_id, None)


async def plan_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Cancel button press during plan."""
    query = update.callback_query
    if query is None:
        return
    await query.answer("Cancelling...")

    try:
        data = query.data or ""
        parts = data.split(":")
        chat_id = int(parts[1])

        event = _plan_cancels.get(chat_id)
        if event:
            event.set()
            await query.edit_message_text("Cancel requested. Stopping after current step...")
        else:
            await query.edit_message_text("No active plan session to cancel.")
    except Exception:
        logger.exception("plan cancel callback error")


# ── /discuss ──────────────────────────────────────────────────────────────────


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split text into chunks respecting Telegram's 4096 char limit, at paragraph boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        # Try paragraph boundary
        cut = remaining.rfind("\n\n", 0, max_len)
        if cut == -1:
            # Try line boundary
            cut = remaining.rfind("\n", 0, max_len)
        if cut == -1:
            # Hard cut
            cut = max_len

        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")

    return chunks


async def discuss_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/discuss <project> <question> — Technical consultation with Opus."""
    try:
        if not context.args or len(context.args) < 2:
            await _safe_reply(
                update,
                "Usage: <code>/discuss &lt;project&gt; &lt;question&gt;</code>\n"
                "Example: <code>/discuss my-app What features should we add for monitoring?</code>",
            )
            return

        project_name, err = _resolve_project(context.args[0], context.bot_data.get("projects", {}))
        if project_name is None:
            await _safe_reply(update, err)
            return
        projects: dict = context.bot_data.get("projects", {})
        question = " ".join(context.args[1:])

        chat_id = update.effective_chat.id  # type: ignore[union-attr]

        if chat_id in _discuss_cancels:
            await _safe_reply(update, "A discussion is already running. Cancel it first.")
            return

        settings: Settings = context.bot_data["settings"]
        cancel_event = asyncio.Event()
        _discuss_cancels[chat_id] = cancel_event

        project_path = projects[project_name].get("path", "")

        asyncio.create_task(
            _run_discuss(context, chat_id, project_name, project_path, question, settings, cancel_event)
        )

        await _safe_reply(
            update,
            f"Consulting Opus about <b>{html.escape(project_name)}</b>...\n"
            f"<i>{html.escape(question)}</i>",
        )
    except Exception:
        logger.exception("/discuss handler error")
        await _safe_reply(update, "Error starting discussion.")


async def _run_discuss(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    project_name: str,
    project_path: str,
    question: str,
    settings: Settings,
    cancel_event: asyncio.Event,
) -> None:
    """Background task for Opus tech consultation."""
    cancel_btn = InlineKeyboardMarkup([
        [InlineKeyboardButton("Cancel", callback_data=f"cancel_discuss:{chat_id}")]
    ])
    msg = await context.bot.send_message(
        chat_id,
        f"<b>{html.escape(project_name)}</b> Gathering project context...",
        parse_mode=ParseMode.HTML,
        reply_markup=cancel_btn,
    )

    pipeline_start = time.monotonic()

    try:
        # 1. Read CLAUDE.md
        claude_md_path = os.path.join(project_path, ".claude", "CLAUDE.md")
        claude_md = ""
        if os.path.isfile(claude_md_path):
            try:
                with open(claude_md_path, encoding="utf-8") as f:
                    claude_md = f.read()
            except Exception:
                pass

        # 2. File tree (excluding common dirs)
        try:
            tree_proc = await asyncio.create_subprocess_exec(
                "find", ".", "-type", "f",
                "-not", "-path", "./.git/*",
                "-not", "-path", "*/node_modules/*",
                "-not", "-path", "*/build/*",
                "-not", "-path", "*/__pycache__/*",
                "-not", "-path", "*/.venv/*",
                "-not", "-path", "*/.gradle/*",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_path,
            )
            tree_stdout, _ = await asyncio.wait_for(tree_proc.communicate(), timeout=15)
            file_tree_lines = tree_stdout.decode(errors="replace").strip().split("\n") if tree_stdout else []
            file_tree = "\n".join(file_tree_lines[:200])
        except Exception:
            file_tree = "(could not collect file tree)"

        # 3. Build context — auto-detect project type and collect key build files
        build_files: dict[str, str] = {}
        candidates = [
            "build.gradle.kts",
            "settings.gradle.kts",
            "gradle/libs.versions.toml",
            "package.json",
            "pyproject.toml",
            "requirements.txt",
        ]
        for candidate in candidates:
            full_path = os.path.join(project_path, candidate)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, encoding="utf-8") as f:
                        lines = f.readlines()
                    build_files[candidate] = "".join(lines[:200])
                except Exception:
                    pass

        build_context = ""
        if build_files:
            parts = []
            for fname, content in build_files.items():
                parts.append(f"### {fname}\n{content}")
            build_context = "\n\n".join(parts)

        # 4. Cancel check
        if cancel_event.is_set():
            await _edit_msg(msg, f"<b>{html.escape(project_name)}</b> Discussion cancelled.")
            return

        # 5. Call step_discuss_consult
        async def progress_cb(status_text: str) -> None:
            elapsed = int(time.monotonic() - pipeline_start)
            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            text = (
                f"<b>{html.escape(project_name)}</b> {html.escape(status_text)}\n"
                f"[{time_str}]"
            )
            await _edit_msg(msg, text, reply_markup=cancel_btn)

        response = await step_discuss_consult(
            project_name=project_name,
            claude_md=claude_md,
            file_tree=file_tree,
            build_context=build_context,
            question=question,
            settings=settings,
            progress_cb=progress_cb,
        )

        # 6. Send response in chunks
        chunks = _split_message(response)
        last_sent_msg = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            if is_last:
                # Add [Create Issues] button on last chunk
                issues_btn = InlineKeyboardMarkup([
                    [InlineKeyboardButton("Create Issues", callback_data=f"discuss_issues:{msg.message_id}")]
                ])
                if i == 0:
                    # Edit the progress message with the response
                    await _edit_msg(msg, chunk, reply_markup=issues_btn)
                    last_sent_msg = msg
                else:
                    last_sent_msg = await context.bot.send_message(
                        chat_id, chunk, reply_markup=issues_btn,
                    )
            else:
                if i == 0:
                    await _edit_msg(msg, chunk)
                else:
                    await context.bot.send_message(chat_id, chunk)

        # 7. Store discussion result for [Create Issues] button
        # Memory management: keep max 10 entries
        if len(_discuss_results) >= 10:
            oldest_key = next(iter(_discuss_results))
            _discuss_results.pop(oldest_key, None)

        _discuss_results[msg.message_id] = {
            "project_name": project_name,
            "project_path": project_path,
            "github_user": settings.github_user,
            "question": question,
            "response": response,
        }

        # 8. Save discussion to file
        try:
            discussions_dir = os.path.join(project_path, "discussions")
            os.makedirs(discussions_dir, exist_ok=True)

            # Generate slug from question
            slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:60].strip("-")
            from datetime import date
            filename = f"{date.today().isoformat()}_{slug}.md"
            filepath = os.path.join(discussions_dir, filename)

            content = (
                f"## Question\n{question}\n\n"
                f"## Response\n{response}\n\n"
                f"## Metadata\n"
                f"- Date: {date.today().isoformat()}\n"
                f"- Model: opus\n"
            )
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Saved discussion to %s", filepath)
        except Exception as exc:
            logger.warning("Failed to save discussion file: %s", exc)

    except Exception as exc:
        logger.exception("Discuss pipeline error for %s", project_name)
        await _edit_msg(msg, f"<b>{html.escape(project_name)}</b> \u274c Error: {html.escape(str(exc)[:200])}")
    finally:
        _discuss_cancels.pop(chat_id, None)


async def discuss_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Cancel button press during discuss."""
    query = update.callback_query
    if query is None:
        return
    await query.answer("Cancelling...")

    try:
        data = query.data or ""
        parts = data.split(":")
        chat_id = int(parts[1])

        event = _discuss_cancels.get(chat_id)
        if event:
            event.set()
            await query.edit_message_text("Cancel requested. Stopping after current step...")
        else:
            await query.edit_message_text("No active discussion to cancel.")
    except Exception:
        logger.exception("discuss cancel callback error")


async def discuss_create_issues_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle [Create Issues] button press after discuss."""
    query = update.callback_query
    if query is None:
        return
    await query.answer("Creating issues...")

    try:
        data = query.data or ""
        parts = data.split(":")
        message_id = int(parts[1])

        result = _discuss_results.get(message_id)
        if not result:
            await query.edit_message_text("Discussion expired. Please run /discuss again.")
            return

        # Remove button, show progress
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        settings: Settings = context.bot_data["settings"]
        chat_id = update.effective_chat.id  # type: ignore[union-attr]

        progress_msg = await context.bot.send_message(
            chat_id, "Creating issues from discussion..."
        )

        async def progress_cb(status_text: str) -> None:
            await _edit_msg(progress_msg, status_text)

        created, total, issues_list = await step_discuss_to_issues(
            project_name=result["project_name"],
            github_user=result["github_user"],
            project_path=result["project_path"],
            question=result["question"],
            discussion_text=result["response"],
            settings=settings,
            progress_cb=progress_cb,
        )

        # Show results
        issue_lines = []
        for iss in issues_list:
            url = iss.get("url", "")
            title = iss.get("title", "")
            if url:
                issue_lines.append(f"  - <a href=\"{url}\">{html.escape(title)}</a>")
            else:
                issue_lines.append(f"  - {html.escape(title)}")
        issues_text = "\n".join(issue_lines) if issue_lines else "  (none)"

        await _edit_msg(
            progress_msg,
            f"\u2705 Issues created: {created}/{total}\n\n{issues_text}",
        )

        # Cleanup
        _discuss_results.pop(message_id, None)

    except Exception as exc:
        logger.exception("discuss create issues callback error")
        try:
            chat_id = update.effective_chat.id  # type: ignore[union-attr]
            await context.bot.send_message(
                chat_id, f"\u274c Error creating issues: {html.escape(str(exc)[:200])}"
            )
        except Exception:
            pass


# ── /help ────────────────────────────────────────────────────────────────────

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show available commands."""
    text = (
        "<b>Available Commands</b>\n"
        "\n"
        "/status — System stats (CPU, RAM, Disk, Ollama, Tmux)\n"
        "/cmd &lt;command&gt; — Run shell command (--long, --stream)\n"
        "/view — Capture tmux pane output\n"
        "/service — Service control (status/restart/stop/start/logs)\n"
        "\n"
        "/projects — List registered projects\n"
        "/init [--public] &lt;name&gt; &lt;desc&gt; — Bootstrap a new project\n"
        "/plan &lt;project&gt; — Plan new development issues\n"
        "/issues &lt;project&gt; — Open GitHub issues (with Solve buttons)\n"
        "/solve &lt;project&gt; &lt;#&gt; [#...] — Auto-solve issues via Claude\n"
        "/rebase &lt;project&gt; &lt;pr#&gt; — Rebase PR onto latest main\n"
        "/discuss &lt;project&gt; &lt;question&gt; — Technical consultation with Opus\n"
        "/extract &lt;project&gt; &lt;file&gt; — Generate training data from file\n"
        "\n"
        "/help — This message"
    )
    await _safe_reply(update, text)


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _safe_reply(update: Update, text: str) -> None:
    """Reply with HTML parse mode, falling back to plain text on parse errors."""
    try:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)  # type: ignore[union-attr]
    except Exception:
        await update.message.reply_text(text)  # type: ignore[union-attr]
