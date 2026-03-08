"""Telegram Application factory with handler registration and global error handling."""

from __future__ import annotations

import logging

from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from .ai.anthropic_provider import AnthropicProvider
from .ai.gemini_provider import GeminiCLIProvider
from .ai.ollama_provider import OllamaProvider
from .config import Settings, load_projects
from .handlers import (
    cmd_handler,
    discuss_cancel_callback,
    discuss_create_issues_callback,
    discuss_handler,
    extract_handler,
    help_handler,
    init_cancel_callback,
    init_handler,
    issues_handler,
    plan_cancel_callback,
    plan_handler,
    process_callback_handler,
    projects_handler,
    rebase_handler,
    retry_handler,
    service_handler,
    solve_cancel_callback,
    solve_handler,
    solve_inline_callback,
    status_handler,
    view_handler,
)
from .security import AuthFilter

logger = logging.getLogger(__name__)


async def _post_init(app: Application) -> None:
    """Initialize AI providers and store references in bot_data."""
    settings: Settings = app.bot_data["settings"]

    # Load project registry
    app.bot_data["projects"] = load_projects()
    logger.info("Loaded %d project(s)", len(app.bot_data["projects"]))

    # Ollama provider (always created — availability checked at runtime)
    ollama = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.reasoning_model,
    )
    app.bot_data["ollama"] = ollama

    # Verify Ollama models exist
    missing_models: list[str] = []
    for model_name in (settings.reasoning_model, settings.qwen_model):
        try:
            resp = await ollama._client.post("/api/show", json={"name": model_name})
            if resp.status_code == 200:
                logger.info("Ollama model verified: %s", model_name)
            else:
                missing_models.append(model_name)
                logger.warning("Ollama model NOT found: %s", model_name)
        except Exception:
            logger.warning("Could not verify Ollama model: %s", model_name)

    if missing_models:
        pull_cmds = "\n".join(f"  ollama pull {m}" for m in missing_models)
        try:
            await app.bot.send_message(
                settings.telegram_allowed_user_id,
                f"⚠️ Missing Ollama models:\n{pull_cmds}",
            )
        except Exception:
            pass  # Bot init phase — send may fail

    # Anthropic provider (only if API key is configured)
    if settings.anthropic_api_key:
        anthropic = AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
        app.bot_data["anthropic"] = anthropic

    # Gemini CLI provider (for fivebrid pipeline)
    if settings.pipeline_mode == "fivebrid":
        gemini = GeminiCLIProvider(model=settings.gemini_model)
        if await gemini.is_available():
            app.bot_data["gemini"] = gemini
            logger.info("Gemini CLI provider initialized (model=%s)", settings.gemini_model)
        else:
            logger.warning("Gemini CLI not available — fivebrid pipeline will fail on Gemini steps")

    # Register command menu for Telegram autocomplete
    await app.bot.set_my_commands([
        BotCommand("help", "Show available commands"),
        BotCommand("status", "System stats (CPU, RAM, Disk)"),
        BotCommand("cmd", "Run shell command"),
        BotCommand("view", "Capture tmux pane output"),
        BotCommand("service", "Service control"),
        BotCommand("projects", "List registered projects"),
        BotCommand("issues", "Open GitHub issues for a project"),
        BotCommand("init", "Bootstrap a new project"),
        BotCommand("plan", "Plan new issues for existing project"),
        BotCommand("discuss", "Technical consultation with Opus"),
        BotCommand("solve", "Auto-solve issues via Claude"),
        BotCommand("retry", "Resume failed solve from checkpoint"),
        BotCommand("rebase", "Rebase PR onto main"),
        BotCommand("extract", "Generate training data from file"),
    ])

    logger.info("Bot initialized — polling started")


async def _post_shutdown(app: Application) -> None:
    """Gracefully close AI provider clients."""
    for key in ("ollama", "anthropic", "gemini"):
        provider = app.bot_data.get(key)
        if provider is not None:
            await provider.close()
    logger.info("Bot shutdown complete")


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — logs the exception and notifies the user."""
    logger.exception("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                f"Internal error: {type(context.error).__name__}"
            )
        except Exception:
            pass  # Can't even reply — just log


def create_application(settings: Settings) -> Application:
    """Build and configure the Telegram Application."""
    auth = AuthFilter(settings.telegram_allowed_user_id)

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .concurrent_updates(True)
        .build()
    )

    # Store settings for handlers to access
    app.bot_data["settings"] = settings

    # Register command handlers — all filtered through AuthFilter
    app.add_handler(CommandHandler("status", status_handler, filters=auth))
    app.add_handler(CommandHandler("cmd", cmd_handler, filters=auth))
    app.add_handler(CommandHandler("view", view_handler, filters=auth))
    app.add_handler(CommandHandler("service", service_handler, filters=auth))
    app.add_handler(CommandHandler("projects", projects_handler, filters=auth))
    app.add_handler(CommandHandler("issues", issues_handler, filters=auth))
    app.add_handler(CommandHandler("solve", solve_handler, filters=auth))
    app.add_handler(CommandHandler("retry", retry_handler, filters=auth))
    app.add_handler(CommandHandler("init", init_handler, filters=auth))
    app.add_handler(CommandHandler("plan", plan_handler, filters=auth))
    app.add_handler(CommandHandler("discuss", discuss_handler, filters=auth))
    app.add_handler(CommandHandler("rebase", rebase_handler, filters=auth))
    app.add_handler(CommandHandler("extract", extract_handler, filters=auth))
    app.add_handler(CommandHandler("help", help_handler, filters=auth))

    # Inline keyboard callbacks
    app.add_handler(CallbackQueryHandler(process_callback_handler, pattern=r"^proc:"))
    app.add_handler(CallbackQueryHandler(solve_inline_callback, pattern=r"^solve:"))
    app.add_handler(CallbackQueryHandler(solve_cancel_callback, pattern=r"^cancel_solve:"))
    app.add_handler(CallbackQueryHandler(init_cancel_callback, pattern=r"^cancel_init:"))
    app.add_handler(CallbackQueryHandler(plan_cancel_callback, pattern=r"^cancel_plan:"))
    app.add_handler(CallbackQueryHandler(discuss_cancel_callback, pattern=r"^cancel_discuss:"))
    app.add_handler(CallbackQueryHandler(discuss_create_issues_callback, pattern=r"^discuss_issues:"))

    # Global error handler — never let exceptions kill the bot
    app.add_error_handler(_error_handler)

    return app
