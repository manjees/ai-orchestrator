import hmac
import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_LOCALHOST_ADDRESSES = ("127.0.0.1", "::1")


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    PUBLIC_PATHS = {"/api/health"}

    def __init__(self, app, api_key: str = "") -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization")

        if self.api_key:
            if not auth_header or not auth_header.startswith("Bearer "):
                logger.warning("Auth failed: missing or malformed Bearer token from %s", self._client_ip(request))
                return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
            token = auth_header[7:]
            if not hmac.compare_digest(token.encode(), self.api_key.encode()):
                logger.warning("Auth failed: invalid API key from %s", self._client_ip(request))
                return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
        else:
            client_host = self._client_ip(request)
            if not self._is_localhost(client_host):
                logger.warning("Access denied: non-localhost request from %s", client_host)
                return JSONResponse(status_code=403, content={"detail": "Forbidden: localhost only"})

        return await call_next(request)

    @staticmethod
    def _client_ip(request: Request) -> str | None:
        return request.client.host if request.client else None

    @staticmethod
    def _is_localhost(host: str | None) -> bool:
        if host is None:
            return False
        if host in _LOCALHOST_ADDRESSES:
            return True
        # Handle IPv6 scoped addresses like ::1%lo0
        if host.startswith("::1%"):
            return True
        return False
