"""Simple sliding-window rate limiter middleware."""

from __future__ import annotations

import threading
import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed on client IP.

    Tracks request timestamps per IP in a 60-second rolling window.
    When a client exceeds *requests_per_minute* requests in that window,
    subsequent requests receive a 429 response until the window clears.

    Note: state is in-process memory, so limits are per-worker and reset
    on restart. This is appropriate for single-process local deployments.
    """

    def __init__(self, app, *, requests_per_minute: int) -> None:
        super().__init__(app)
        self.limit = requests_per_minute
        self._window = 60.0
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        if not self._is_allowed(ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        return await call_next(request)

    def _is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.setdefault(ip, deque())
            cutoff = now - self._window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True
