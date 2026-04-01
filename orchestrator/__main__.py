"""Entrypoint: python -m orchestrator"""

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn

from .api.app import create_api_app
from .bot import create_application
from .config import Settings

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / "orchestrator.log",
        maxBytes=10_485_760,
        backupCount=5,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler())


async def run_all(settings: Settings) -> None:
    telegram_app = create_application(settings)

    api_app = create_api_app(settings)
    config = uvicorn.Config(
        api_app, host="0.0.0.0", port=settings.api_port, log_level="info"
    )
    server = uvicorn.Server(config)

    async with telegram_app:
        if telegram_app.post_init:
            await telegram_app.post_init(telegram_app)
        await telegram_app.start()
        await telegram_app.updater.start_polling()

        try:
            await server.serve()
        finally:
            try:
                await telegram_app.updater.stop()
            except Exception:
                logger.exception("Error stopping Telegram updater")
            try:
                await telegram_app.stop()
            except Exception:
                logger.exception("Error stopping Telegram app")


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_level)
    asyncio.run(run_all(settings))


if __name__ == "__main__":
    main()
