"""Entrypoint: python -m orchestrator"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .bot import create_application
from .config import Settings


def setup_logging(level: str) -> None:
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / "orchestrator.log",
        maxBytes=10_485_760,  # 10 MB
        backupCount=5,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler())


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_level)

    app = create_application(settings)
    app.run_polling()


if __name__ == "__main__":
    main()
