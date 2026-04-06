import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from orchestrator.config import Settings

from .approval_routes import approval_router
from .auth import APIKeyAuthMiddleware
from .command_routes import command_router
from .events import get_event_bus
from .routes import router

logger = logging.getLogger(__name__)


def _parse_cors_origins(raw: str) -> list[str]:
    origins = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parsed = urlparse(entry)
        if parsed.scheme and parsed.netloc:
            origins.append(entry)
        else:
            logger.warning("Ignoring invalid CORS origin: %s", entry)
    return origins


@asynccontextmanager
async def _lifespan(app: FastAPI):
    bus = get_event_bus()
    await bus.start_status_loop(interval=10.0)
    yield
    await bus.stop_status_loop()


def create_api_app(
    settings: Settings,
    projects: dict[str, dict] | None = None,
    pipelines: dict | None = None,
) -> FastAPI:
    app = FastAPI(title="AI Orchestrator API", lifespan=_lifespan)
    app.state.projects = projects if projects is not None else {}
    app.state.pipelines = pipelines if pipelines is not None else {}
    app.state.settings = settings

    if settings.cors_origins:
        origins = _parse_cors_origins(settings.cors_origins)
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    app.add_middleware(APIKeyAuthMiddleware, api_key=settings.dashboard_api_key)
    app.include_router(router)
    app.include_router(command_router)
    app.include_router(approval_router)

    return app
