"""Debug: check if bot_data survives Application.initialize()"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from orchestrator.config import Settings
from orchestrator.bot import create_application


async def test():
    s = Settings()
    app = create_application(s)
    print("BEFORE initialize - settings in bot_data:", "settings" in app.bot_data)
    await app.initialize()
    print("AFTER initialize - settings in bot_data:", "settings" in app.bot_data)
    print("AFTER initialize - projects in bot_data:", "projects" in app.bot_data)
    print("projects:", app.bot_data.get("projects"))
    await app.shutdown()


asyncio.run(test())
