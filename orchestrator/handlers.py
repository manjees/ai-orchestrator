"""Telegram command handlers: /status, /cmd, /view, /projects, /issues, /solve, and process control."""

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
from .ai.ollama_provider import OllamaProvider
from .pipeline import PipelineContext, format_pipeline_summary, run_dual_check_pipeline
from .security import mask_secrets
from .system_monitor import get_system_status
from .tmux_manager import capture_pane, list_sessions

logger = logging.getLogger(__name__)

# Process names to monitor for inline keyboard control
_MONITORED_PROCESSES = {"ollama", "python", "node"}

# Active solve sessions keyed by chat_id → asyncio.Event (set = cancel requested)
_solve_cancels: dict[int, asyncio.Event] = {}
_solve_active: set[int] = set()

# ANSI escape codes: CSI sequences (incl. ?/= private modes), OSC sequences, single ESC codes
_ANSI_RE = re.compile(r"\x1b\[[^A-Za-z]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[^[\]()]")


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

_LAUNCHD_LABEL = "com.manjee.ai-orchestrator"
_PLIST_PATH = os.path.expanduser(
    f"~/Library/LaunchAgents/{_LAUNCHD_LABEL}.plist"
)


async def service_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show service menu: /service [status|restart|stop|start|logs]"""
    try:
        sub = (context.args[0].lower() if context.args else "").strip()

        if sub == "status":
            await _svc_status(update)
        elif sub == "restart":
            await _svc_restart(update)
        elif sub == "stop":
            await _svc_stop(update)
        elif sub == "start":
            await _svc_start(update)
        elif sub == "logs":
            await _svc_logs(update, context)
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


async def _svc_status(update: Update) -> None:
    rc, output = await _run_launchctl("list", _LAUNCHD_LABEL)
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


async def _svc_restart(update: Update) -> None:
    uid = os.getuid()
    await _safe_reply(update, "Restarting bot...")
    await _run_launchctl("kickstart", "-k", f"gui/{uid}/{_LAUNCHD_LABEL}")


async def _svc_stop(update: Update) -> None:
    await _safe_reply(update, "Stopping bot... (KeepAlive will NOT restart)")
    # unload prevents KeepAlive from restarting the process
    await _run_launchctl("unload", _PLIST_PATH)


async def _svc_start(update: Update) -> None:
    rc, output = await _run_launchctl("load", _PLIST_PATH)
    if rc == 0:
        await _safe_reply(update, "Service loaded and starting.")
    else:
        await _safe_reply(update, f"Failed to start:\n<pre>{html.escape(output)}</pre>")


async def _svc_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

        project_name = context.args[0]
        projects: dict = context.bot_data.get("projects", {})
        if project_name not in projects:
            await _safe_reply(update, f"Unknown project: <code>{html.escape(project_name)}</code>")
            return

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

        project_name = context.args[0]
        projects: dict = context.bot_data.get("projects", {})
        if project_name not in projects:
            await _safe_reply(update, f"Unknown project: <code>{html.escape(project_name)}</code>")
            return

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
        # cancel_solve:<chat_id>
        _, chat_id_str = data.split(":", 1)
        chat_id = int(chat_id_str)

        cancel_event = _solve_cancels.get(chat_id)
        if cancel_event:
            cancel_event.set()
            await query.edit_message_text("Cancel requested. Stopping after current step...")
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
    """Guard concurrent runs and launch the solve loop."""
    if chat_id in _solve_active:
        msg = "A solve session is already running. Use Cancel to stop it first."
        if from_callback:
            await context.bot.send_message(chat_id, msg)
        else:
            await _safe_reply(update, msg)
        return

    _solve_active.add(chat_id)
    cancel_event = asyncio.Event()
    _solve_cancels[chat_id] = cancel_event

    projects: dict = context.bot_data.get("projects", {})
    project_path = projects[project_name]["path"]
    settings = context.bot_data["settings"]
    timeout = settings.solve_timeout

    # Run in background so the handler returns immediately
    asyncio.create_task(
        _solve_issues(context, chat_id, project_name, project_path, issue_nums, timeout, cancel_event)
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
    cancel_event: asyncio.Event,
) -> None:
    """Sequentially solve each issue: git fresh start → claude → PR."""
    results: list[tuple[int, str, str]] = []  # (issue#, status, detail)

    try:
        for issue_num in issue_nums:
            if cancel_event.is_set():
                results.append((issue_num, "skipped", "Cancelled by user"))
                continue

            status, detail = await _solve_single_issue(
                context, chat_id, project_name, project_path, issue_num, timeout, cancel_event,
            )
            results.append((issue_num, status, detail))
    except Exception:
        logger.exception("Solve loop error")
    finally:
        _solve_active.discard(chat_id)
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
    """Solve one issue. Routes to dual-check or direct-claude based on settings."""
    settings = context.bot_data["settings"]

    if settings.dual_check_enabled:
        ollama: OllamaProvider | None = context.bot_data.get("ollama")
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
        [InlineKeyboardButton("Cancel", callback_data=f"cancel_solve:{chat_id}")]
    ])
    msg = await context.bot.send_message(
        chat_id,
        f"<b>#{issue_num}</b> [0/4] Preparing git worktree...",
        parse_mode=ParseMode.HTML,
        reply_markup=cancel_btn,
    )

    pipeline_start = time.monotonic()

    try:
        # ── Git Fresh Start ──
        git_ok, git_err = await _git_fresh_start(project_path, branch_name)
        if not git_ok:
            await _edit_msg(msg, f"<b>#{issue_num}</b> — Git setup failed:\n<pre>{_sanitize_output(git_err)}</pre>")
            return "failed", f"Git setup failed: {git_err[:100]}"

        if cancel_event.is_set():
            return "skipped", "Cancelled by user"

        # Build pipeline context
        ctx = PipelineContext(
            project_path=project_path,
            project_name=project_name,
            issue_num=issue_num,
            branch_name=branch_name,
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

        if status == "success":
            # Create PR
            await _edit_msg(msg, f"<b>#{issue_num}</b> — Creating PR...")
            pr_url, pr_err = await _create_pr(project_path, issue_num, branch_name)

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
        [InlineKeyboardButton("Cancel", callback_data=f"cancel_solve:{chat_id}")]
    ])
    msg = await context.bot.send_message(
        chat_id,
        f"<b>#{issue_num}</b> — Preparing git worktree...",
        parse_mode=ParseMode.HTML,
        reply_markup=cancel_btn,
    )

    try:
        # ── Git Fresh Start ──
        git_ok, git_err = await _git_fresh_start(project_path, branch_name)
        if not git_ok:
            await _edit_msg(msg, f"<b>#{issue_num}</b> — Git setup failed:\n<pre>{_sanitize_output(git_err)}</pre>")
            return "failed", f"Git setup failed: {git_err[:100]}"

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
            cwd=project_path,
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

        # ── Auto PR ──
        await _edit_msg(msg, f"<b>#{issue_num}</b> — Claude done ({time_str}). Creating PR...")

        pr_url, pr_err = await _create_pr(project_path, issue_num, branch_name)
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


async def _git_fresh_start(project_path: str, branch_name: str) -> tuple[bool, str]:
    """Reset to main, pull, create new branch. Returns (ok, error_msg)."""
    commands = [
        # Stash any dirty changes
        ["git", "stash", "--include-untracked"],
        # Checkout main
        ["git", "checkout", "main"],
        # Pull latest
        ["git", "pull"],
        # Create and checkout new branch (delete if exists)
        ["git", "branch", "-D", branch_name],  # may fail — that's ok
    ]

    for cmd in commands:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace") if stdout else ""
        # Allow branch -D to fail (branch may not exist)
        if proc.returncode != 0 and cmd[1] != "branch":
            return False, f"{' '.join(cmd)} failed: {output}"

    # Create new branch
    proc = await asyncio.create_subprocess_exec(
        "git", "checkout", "-b", branch_name,
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    if proc.returncode != 0:
        output = stdout.decode(errors="replace") if stdout else ""
        return False, f"git checkout -b failed: {output}"

    return True, ""


async def _create_pr(project_path: str, issue_num: int, branch_name: str) -> tuple[str, str]:
    """Push branch and create PR. Returns (pr_url, error_msg)."""
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
    proc = await asyncio.create_subprocess_exec(
        "gh", "pr", "create",
        "--title", f"fix: resolve #{issue_num}",
        "--body", f"Automatically solved by ai-orchestrator\n\nCloses #{issue_num}",
        cwd=project_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
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
        "/issues &lt;project&gt; — Open GitHub issues (with Solve buttons)\n"
        "/solve &lt;project&gt; &lt;#&gt; [#...] — Auto-solve issues via Claude\n"
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
