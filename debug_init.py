"""Debug script: test _post_init flow without network calls."""
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from orchestrator.config import Settings, load_projects
from orchestrator.bot import create_application


async def test():
    s = Settings()
    app = create_application(s)
    print("settings in bot_data:", "settings" in app.bot_data)
    p = load_projects()
    print("projects loaded:", len(p), p)
    app.bot_data["projects"] = p
    logger = logging.getLogger("orchestrator.bot")
    logger.info("Loaded %d project(s)", len(p))
    print("done")


asyncio.run(test())
