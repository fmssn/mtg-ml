"""The hosted play server: a Starlette app over `manager.HostedManager`.

Same protocol as `replay.make_server` (the play page at /play/, the replay
viewer at /, /api/live/... and /api/replays) with these differences:

- `/healthz` is the only unauthenticated path; it answers 503 until the
  configuration and every pinned checkpoint are validated and the recorded
  games are restored;
- everything else needs a Cloudflare Access JWT of an allowlisted email
  (`auth`), or the insecure dev user when configured;
- the game token only in the `X-Game-Token` header (`?token=` is refused);
- JSON mutations: POST, `Origin` equal to the public origin, `Content-Type:
  application/json`, at most 64 KiB (counted while reading);
- games and replays belong to the account that made them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from ..live import LiveError, load_play_config, validate_offer
from ..replay import PLAY_APP, VIEWER, _PLAY_TYPES
from .auth import AccessVerifier, AuthError, request_token
from .config import HostedConfig
from .manager import HostedError, HostedManager

MAX_BODY = 64 * 1024
log = logging.getLogger("mtg_ml.hosted")
SECURITY_HEADERS = [(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"same-origin"),
                    (b"x-frame-options", b"DENY")]


class Refused(Exception):
    def __init__(self, message: str, status: int, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra


def _error(message: str, status: int, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status)


class AccessMiddleware:
    """Verifies the account of every request but /healthz; sets scope["account"]."""

    def __init__(self, app, cfg: HostedConfig, verifier: AccessVerifier | None):
        self.app, self.cfg, self.verifier = app, cfg, verifier

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_secure(msg):
            if msg["type"] == "http.response.start":
                msg = {**msg, "headers": [h for h in msg.get("headers", []) if h[0].lower() not in dict(SECURITY_HEADERS)] + SECURITY_HEADERS}
            await send(msg)

        if scope["path"] == "/healthz":
            return await self.app(scope, receive, send_secure)
        req = Request(scope)
        if self.cfg.dev_user:
            account = self.cfg.dev_user
        else:
            try:
                account = await run_in_threadpool(self.verifier.verify, request_token(req.headers, req.cookies))
            except AuthError as e:
                return await _error(str(e), e.status, auth=True)(scope, receive, send_secure)
        scope["account"] = account
        await self.app(scope, receive, send_secure)


def create_app(cfg: HostedConfig, fetch_certs=None, background: bool = True, manager_kw: dict | None = None) -> Starlette:
    """`background`: validate and restore on a thread while /healthz answers 503
    (serve); False: before the app takes requests (tests). `fetch_certs`: the
    Access key fetcher (tests). `manager_kw`: HostedManager limits (tests)."""
    state = {"manager": None, "ready": False, "status": "starting", "models": {}}

    def startup_work() -> None:
        try:
            offer = load_play_config(cfg.play_config)
            pinned, problems = validate_offer(cfg.models, offer, strict=True)
            if problems:
                state["status"] = "invalid"
                for p in problems:
                    log.error("play offer: %s", p)
                return
            mgr = HostedManager(cfg.models, cfg.replays, cfg.state, config=offer, pinned=pinned, runtime=cfg.runtime, engine=cfg.engine,
                                filer=_filer(cfg), **(manager_kw or {}))
            state["manager"] = mgr
            restored = mgr.submit(mgr.h_restore_all).result()
            log.info("restored %d games; unrecoverable: %s", restored["restored"], restored["unrecoverable"] or "none")
            state["models"] = {k: v["sha256"] for k, v in pinned.items()}
            state["ready"], state["status"] = True, "ok"
        except Exception:
            state["status"] = "invalid"
            log.exception("startup failed")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if background:
            threading.Thread(target=startup_work, name="hosted-startup", daemon=True).start()
        else:
            startup_work()
        app.state.hosted = state
        try:
            yield
        finally:
            if state["manager"] is not None:
                await run_in_threadpool(state["manager"].close)

    def manager() -> HostedManager:
        if not state["ready"]:
            raise Refused("the server is starting" if state["status"] == "starting" else "the server is misconfigured", 503)
        return state["manager"]

    async def run(fn, *args):
        return await asyncio.wrap_future(manager().submit(fn, *args))

    def same_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        want = cfg.public_origin or f"{request.url.scheme}://{request.headers.get('host', '')}".lower()
        if not origin or origin.lower() != want:
            raise Refused("cross-origin request refused", 403)

    async def body_json(request: Request) -> dict:
        same_origin(request)
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise Refused("send JSON (Content-Type: application/json)", 415)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY:
                raise Refused("request too large", 413)
        try:
            req = json.loads(bytes(body) or b"{}")
        except ValueError:
            raise Refused("malformed JSON", 400) from None
        if not isinstance(req, dict):
            raise Refused("expected a JSON object", 400)
        return req

    def handle(fn):
        async def endpoint(request: Request):
            try:
                return await fn(request)
            except Refused as e:
                return _error(str(e), e.status, **e.extra)
            except HostedError as e:
                return _error(str(e), e.status, **e.extra)
            except LiveError as e:
                return _error(str(e), 403 if getattr(e, "forbidden", False) else 400)
            except ValueError as e:
                return _error(str(e), 400)
            except Exception as e:  # noqa: BLE001
                log.exception("request failed")
                return _error(f"server error: {type(e).__name__}", 500)

        return endpoint

    def game_token(request: Request) -> str | None:
        if "token" in request.query_params:
            raise Refused("send the game token in the X-Game-Token header, never in the URL", 400)
        return request.headers.get("x-game-token")

    # -- routes
    async def healthz(request: Request):
        body = {"ok": state["ready"], "status": state["status"], "models": state["models"], "runtime": cfg.runtime}
        return JSONResponse(body, 200 if state["ready"] else 503)

    async def viewer(request: Request):
        return FileResponse(VIEWER, media_type="text/html; charset=utf-8")

    async def play_redirect(request: Request):
        return RedirectResponse("/play/", 301)

    async def play_asset(request: Request):
        name = request.path_params.get("name") or "index.html"
        f = (PLAY_APP / name).resolve()
        if f.parent == PLAY_APP.resolve() and f.suffix in _PLAY_TYPES and f.is_file():
            return FileResponse(f, media_type=_PLAY_TYPES[f.suffix])
        return Response(b"not found", 404, media_type="text/plain")

    async def replays(request: Request):
        return JSONResponse(await run_in_threadpool(manager().replay_list, request.scope["account"]))

    async def replay_file(request: Request):
        p = await run_in_threadpool(manager().replay_path, request.scope["account"], request.path_params["name"])
        if p is None:
            return Response(b"not found", 404, media_type="text/plain")
        return FileResponse(p, media_type="application/json")

    async def options(request: Request):
        game_token(request)
        return JSONResponse(await run(manager().h_options, request.scope["account"]))

    async def sideboard(request: Request):
        q = request.query_params
        return JSONResponse(manager().sideboard(q.get("matchup", ""), int(q.get("seat", "0"))))

    async def decks(request: Request):
        q = request.query_params
        return JSONResponse(manager().decklists(q.get("matchup", ""), int(q.get("seat", "0")), int(q.get("game", "1"))))

    async def new(request: Request):
        game_token(request)
        req = await body_json(request)
        return JSONResponse(await run(manager().h_new, request.scope["account"], req))

    async def view(request: Request):
        token = game_token(request)
        return JSONResponse(await run(manager().h_view, request.scope["account"], request.path_params["gid"], token, int(request.query_params.get("since", "0"))))

    async def review(request: Request):
        token = game_token(request)
        return JSONResponse(await run(manager().h_review, request.scope["account"], request.path_params["gid"], token))

    async def action(request: Request):
        token = game_token(request)
        req = await body_json(request)
        acct, gid, what, mgr = request.scope["account"], request.path_params["gid"], request.path_params["action"], manager()
        if what == "choose":
            return JSONResponse(await run(mgr.h_choose, acct, gid, token, req))
        if what == "concede":
            return JSONResponse(await run(mgr.h_concede, acct, gid, token))
        if what == "next":
            return JSONResponse(await run(mgr.h_next, acct, gid, token, req))
        if what == "survey":
            return JSONResponse(await run(mgr.h_survey, acct, gid, token, req))
        if what == "flag":
            out = await run(mgr.h_flag, acct, gid, token, req)
            return JSONResponse(await run_in_threadpool(mgr.file_flag, gid, out))
        return _error("not found", 404)

    routes = [
        Route("/healthz", handle(healthz)),
        Route("/", handle(viewer)),
        Route("/index.html", handle(viewer)),
        Route("/play", handle(play_redirect)),
        Route("/play/", handle(play_asset)),
        Route("/play/{name}", handle(play_asset)),
        Route("/api/replays", handle(replays)),
        Route("/replays/{name}", handle(replay_file)),
        Route("/api/live/options", handle(options)),
        Route("/api/live/sideboard", handle(sideboard)),
        Route("/api/live/decks", handle(decks)),
        Route("/api/live/new", handle(new), methods=["POST"]),
        Route("/api/live/{gid}", handle(view)),
        Route("/api/live/{gid}/review", handle(review)),
        Route("/api/live/{gid}/{action}", handle(action), methods=["POST"]),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.hosted = state
    verifier = None if cfg.dev_user else AccessVerifier(cfg.team_domain, cfg.aud, cfg.allowed_emails, fetch_certs=fetch_certs)
    app.add_middleware(AccessMiddleware, cfg=cfg, verifier=verifier)
    return app


def _filer(cfg: HostedConfig):
    if not cfg.github_filing:
        return None
    from ..live_issues import GitHubFiler

    return GitHubFiler()
