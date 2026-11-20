"""Request guard: who may talk to the API.

The API is a local tool with no accounts. Two browser attacks still reach it:

* **DNS rebinding.** A page on ``evil.example`` re-points its name at 127.0.0.1 and then
  calls the API "same-origin". The request carries ``Host: evil.example``, so only
  requests whose ``Host`` is a loopback name (or one listed in ``BAKUMINE_ALLOWED_HOSTS``,
  comma separated; ``*`` allows any) are served; others get 400.
* **Cross-site requests.** A page on another origin can send simple requests (no
  preflight). State-changing methods with an ``Origin`` header that is not the server's
  own (``Origin`` host:port equal to ``Host``) get 403. Clients that send no ``Origin``
  (the CLI, curl, scripts) are not browsers and pass.

No CORS headers are sent, so other origins can never read a response either.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

ALLOWED_HOSTS_ENV = "BAKUMINE_ALLOWED_HOSTS"
#: loopback names, and the host name of Starlette's test client
DEFAULT_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def host_name(netloc: str) -> str:
    """``host[:port]`` or ``[v6]:port`` -> lower-case host without port or brackets."""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[1 : netloc.find("]")] if "]" in netloc else netloc
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def allowed_hosts() -> frozenset[str]:
    extra = {h.strip().lower() for h in os.environ.get(ALLOWED_HOSTS_ENV, "").split(",")}
    return DEFAULT_HOSTS | {h for h in extra if h}


class RequestGuard:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        allowed = allowed_hosts()
        if "*" not in allowed and host_name(host) not in allowed:
            await PlainTextResponse("host not allowed", status_code=400)(scope, receive, send)
            return
        origin = headers.get("origin")
        if scope["method"] not in SAFE_METHODS and origin is not None:
            netloc = urlsplit(origin).netloc.lower() if origin != "null" else ""
            if not netloc or netloc != host.strip().lower():
                response = PlainTextResponse("cross-origin request refused", status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
