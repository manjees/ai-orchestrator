import logging
from urllib.parse import urlparse

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from orchestrator.config import Settings

from .auth import APIKeyAuthMiddleware
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


def create_api_app(
    settings: Settings,
    projects: dict[str, dict] | None = None,
    pipelines: dict | None = None,
) -> FastAPI:
    app = FastAPI(title="AI Orchestrator API")
    app.state.projects = projects if projects is not None else {}
    app.state.pipelines = pipelines if pipelines is not None else {}

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

    return app
