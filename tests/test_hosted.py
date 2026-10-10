"""The hosted play server (mtg_ml.hosted): Cloudflare Access verification,
accounts owning games and replays, header-only game tokens, private seeds,
request limits and capacity. Tokens are signed with a local RSA key whose
JWKS is injected in place of the team's certs endpoint."""

import concurrent.futures
import json
import threading
import time

import pytest

pytest.importorskip("starlette")
jwt = pytest.importorskip("jwt")
pytest.importorskip("torch")

from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from mtg_ml.hosted.app import create_app  # noqa: E402
from mtg_ml.hosted.auth import AccessVerifier, AuthError  # noqa: E402
from mtg_ml.hosted.config import ConfigError, HostedConfig  # noqa: E402
from tests.test_play_offer import TINY, pilot_config, tiny7  # noqa: E402,F401 (fixture)

TEAM = "https://team.cloudflareaccess.com"
AUD = "aud-tag-123"
ORIGIN = "https://play.example.com"
ALICE, BOB, EVE = "alice@example.com", "bob@example.com", "eve@example.com"


class Keys:
    """A local Access issuer: RSA keys, a JWKS and signed tokens."""

    def __init__(self):
        self.keys = {"k1": rsa.generate_private_key(public_exponent=65537, key_size=2048)}
        self.fetches = 0

    def jwks(self) -> dict:
        self.fetches += 1
        out = []
        for kid, k in self.keys.items():
            jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(k.public_key()))
            out.append({**jwk, "kid": kid, "alg": "RS256", "use": "sig"})
        return {"keys": out}

    def token(self, email=ALICE, kid="k1", key=None, **over) -> str:
        now = int(time.time())
        claims = {"iss": TEAM, "aud": [AUD], "email": email, "iat": now, "nbf": now, "exp": now + 600, "sub": "x", **over}
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, key or self.keys[kid], algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def keys():
    return Keys()


def write_offer(tmp_path, models):
    """The packaged offer, pinned to the tiny checkpoint, as a TOML file."""
    cfg = pilot_config(models)
    lines = [f"player_decks = {json.dumps(cfg['player_decks'])}", ""]
    for name, pin in cfg["models"].items():
        lines.append(f'[models."{name}"]')
        lines += [f"{k} = {json.dumps(v)}" for k, v in pin.items()]
        lines.append("")
    for o in cfg["opponents"]:
        lines.append("[[opponents]]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in o.items()]
        lines.append("")
    path = tmp_path / "play.toml"
    path.write_text("\n".join(lines))
    return path


def make_cfg(tmp_path, models, engine="python", dev_user="", **over) -> HostedConfig:
    base = dict(models=models, state=tmp_path / "state", replays=tmp_path / "replays", play_config=over.pop("play_config", None) or write_offer(tmp_path, models), engine=engine,
                team_domain=TEAM, aud=AUD, allowed_emails=frozenset({ALICE, BOB}), public_origin=ORIGIN, dev_user=dev_user, runtime="test-rev")
    base.update(over)
    return HostedConfig(**base)


@pytest.fixture
def server(tmp_path, tiny7, keys):  # noqa: F811
    """A started app (python engine) and a client per account."""
    cfg = make_cfg(tmp_path, tiny7)
    app = create_app(cfg, fetch_certs=keys.jwks, background=False, manager_kw={"max_pending": 4})
    with TestClient(app, base_url="http://testserver") as c:
        yield Server(c, keys, app)


class Server:
    def __init__(self, client, keys, app):
        self.c, self.keys, self.app = client, keys, app

    @property
    def mgr(self):
        return self.app.state.hosted["manager"]

    def h(self, email=ALICE, token=None, **extra):
        h = {"Cf-Access-Jwt-Assertion": self.keys.token(email), "Origin": ORIGIN, **extra}
        if token:
            h["X-Game-Token"] = token
        return h

    def get(self, path, email=ALICE, token=None, **kw):
        return self.c.get(path, headers=self.h(email, token), **kw)

    def post(self, path, body=None, email=ALICE, token=None, headers=None, **kw):
        return self.c.post(path, json={} if body is None else body, headers={**self.h(email, token), **(headers or {})}, **kw)

    def new(self, email=ALICE, deck="elves", opponent="r9-elves-pilot"):
        r = self.post("/api/live/new", {"deck": deck, "opponent": opponent}, email=email)
        assert r.status_code == 200, r.text
        v = r.json()
        return v, v["live"]["id"], v["live"]["token"]


# ---------------------------------------------------------------- Access JWTs
def test_verifier_accepts_good_and_refuses_bad_tokens(keys):
    v = AccessVerifier(TEAM, AUD, [ALICE.upper()], fetch_certs=keys.jwks)
    assert v.verify(keys.token(ALICE.upper())) == ALICE  # lower-cased on both sides
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = int(time.time())
    bad = {
        "expired": keys.token(exp=now - 120, iat=now - 700, nbf=now - 700),
        "not yet valid": keys.token(nbf=now + 600),
        "issued in the future": keys.token(iat=now + 600),
        "wrong aud": keys.token(aud=["other"]),
        "wrong iss": keys.token(iss="https://evil.cloudflareaccess.com"),
        "no exp": keys.token(exp=None),
        "bad signature": keys.token(key=other),
        "unknown kid": jwt.encode({"iss": TEAM, "aud": AUD, "email": ALICE, "iat": now, "exp": now + 600}, other, algorithm="RS256", headers={"kid": "k9"}),
        "unsigned": jwt.encode({"iss": TEAM, "aud": AUD, "email": ALICE, "iat": now, "exp": now + 600}, None, algorithm="none"),
        "hs256": jwt.encode({"iss": TEAM, "aud": AUD, "email": ALICE, "iat": now, "exp": now + 600}, "x" * 32, algorithm="HS256", headers={"kid": "k1"}),
        "garbage": "not.a.jwt",
        "empty": "",
        "no email": keys.token(email=None),
    }
    for what, tok in bad.items():
        with pytest.raises(AuthError) as e:
            v.verify(tok)
        assert e.value.status == 401, what
    with pytest.raises(AuthError) as e:
        v.verify(keys.token(EVE))
    assert e.value.status == 403


def test_unknown_kid_refetches_bounded_and_picks_up_rotation(keys):
    clock = [1000.0]
    v = AccessVerifier(TEAM, AUD, [ALICE], fetch_certs=keys.jwks, clock=lambda: clock[0])
    v.clock = lambda: clock[0]
    assert v.verify(keys.token()) == ALICE and keys.fetches == 1
    keys.keys["k2"] = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    clock[0] += 5
    for _ in range(5):  # within a minute: no refetch for an unknown kid
        with pytest.raises(AuthError):
            v.verify(keys.token(kid="k2"))
    assert keys.fetches == 1
    clock[0] += 60
    # pyjwt checks iat against the real clock; the verifier's own clock only paces fetches
    assert v.verify(keys.token(kid="k2")) == ALICE and keys.fetches == 2
    assert v.verify(keys.token()) == ALICE and keys.fetches == 2


def test_config_from_env(tmp_path):
    env = {"MTG_HOSTED_MODELS": "/m", "MTG_HOSTED_STATE": "/s", "MTG_HOSTED_REPLAYS": "/r"}
    with pytest.raises(ConfigError) as e:
        HostedConfig.from_env(env)
    assert "MTG_ACCESS_AUD" in str(e.value)
    full = {**env, "MTG_ACCESS_TEAM_DOMAIN": TEAM + "/", "MTG_ACCESS_AUD": AUD, "MTG_ALLOWED_EMAILS": "A@x.com, b@y.com", "MTG_HOSTED_PUBLIC_ORIGIN": ORIGIN}
    cfg = HostedConfig.from_env(full)
    assert cfg.team_domain == TEAM and cfg.allowed_emails == {"a@x.com", "b@y.com"} and not cfg.dev and not cfg.github_filing
    with pytest.raises(ConfigError, match="refused when MTG_ACCESS_AUD"):
        HostedConfig.from_env({**full, "MTG_HOSTED_INSECURE_DEV_USER": "me@x.com"})
    with pytest.raises(ConfigError, match="https"):
        HostedConfig.from_env({**full, "MTG_HOSTED_PUBLIC_ORIGIN": "http://play.example.com"})
    dev = HostedConfig.from_env({**env, "MTG_HOSTED_INSECURE_DEV_USER": "Me@X.com", "MTG_GIT_REV": "abc123"})
    assert dev.dev_user == "me@x.com" and dev.runtime == "abc123"


def test_requests_need_a_valid_access_token(server):
    assert server.c.get("/healthz").status_code == 200
    assert server.c.get("/healthz").json()["models"] == {TINY: server.mgr.pinned[TINY]["sha256"]}
    for path in ("/", "/play/", "/api/replays", "/api/live/options"):
        r = server.c.get(path)
        assert r.status_code == 401 and r.json()["auth"], path
    r = server.c.get("/api/live/options", headers={"Cf-Access-Jwt-Assertion": server.keys.token(EVE)})
    assert r.status_code == 403
    assert server.c.get("/api/live/options", headers={"Cf-Access-Jwt-Assertion": server.keys.token(aud=["x"])}).status_code == 401
    r = server.c.get("/api/live/options", cookies={"CF_Authorization": server.keys.token(BOB)})  # the cookie works too
    assert r.status_code == 200 and r.json()["account"] == BOB
    opt = server.get("/api/live/options").json()
    assert opt["mode"] == "play" and opt["hosted"] and not opt["filing"] and "models" not in opt and "scenarios" not in opt
    assert len(opt["player_decks"]) == 6
    assert server.get("/play/").status_code == 200 and server.get("/").status_code == 200
    assert server.get("/play/").headers["x-content-type-options"] == "nosniff"


# ---------------------------------------------------------------- ownership and tokens
def test_games_belong_to_their_account(server):
    v, gid, tok = server.new()
    assert v["meta"]["seed"] is None  # private until the game is over
    assert server.get(f"/api/live/{gid}", token=tok).status_code == 200
    # Bob, even with Alice's token (stolen), sees no game: the same answer as an unknown id
    r = server.get(f"/api/live/{gid}", email=BOB, token=tok)
    unknown = server.get("/api/live/nope", email=BOB, token=tok)
    assert r.status_code == unknown.status_code == 404 and r.json()["error"] == unknown.json()["error"]
    frame = len(v["frames"]) - 1
    for path, body in ((f"/api/live/{gid}/choose", {"frame": frame, "index": 0}), (f"/api/live/{gid}/concede", {}), (f"/api/live/{gid}/next", {}),
                       (f"/api/live/{gid}/flag", {"category": "bug", "what": "x"}), (f"/api/live/{gid}/survey", {"strength": 3})):
        assert server.post(path, body, email=BOB, token=tok).status_code == 404, path
    assert server.get(f"/api/live/{gid}/review", email=BOB, token=tok).status_code == 404
    # Alice without the token, or with another game's token: refused
    assert server.get(f"/api/live/{gid}").status_code == 403
    _, gid2, tok2 = server.new(deck="tron", opponent="r9-tron-pilot")
    assert server.get(f"/api/live/{gid}", token=tok2).status_code == 403
    # the token in the URL is refused outright
    r = server.c.get(f"/api/live/{gid}?token={tok}", headers=server.h())
    assert r.status_code == 400 and "header" in r.json()["error"]
    assert server.c.get(f"/api/live/{gid}/review?token={tok}", headers=server.h()).status_code == 400


def test_player_seeds_and_dev_requests_are_refused(server):
    r = server.post("/api/live/new", {"deck": "elves", "opponent": "r9-elves-pilot", "seed": 3})
    assert r.status_code == 400 and "seed" in r.json()["error"]
    for body in ({"model": TINY, "matchup": "jund_blue", "seat": 0}, {"scenario": "wildfire", "model": "scripted-bot"}, {"deck": "elves", "opponent": "nope"}):
        assert server.post("/api/live/new", body).status_code == 400, body


def test_seed_and_reconstruction_data_stay_private_until_the_end(server):
    v, gid, tok = server.new()
    frame = len(v["frames"]) - 1
    out = server.post(f"/api/live/{gid}/flag", {"category": "bug", "what": "odd", "frame": frame}, token=tok).json()
    assert "seed" not in json.dumps(out).lower().replace("the seed", "")
    assert server.get(f"/api/live/{gid}/review", token=tok).status_code == 400
    r = server.post(f"/api/live/{gid}/flag", {"category": "bug", "what": "x", "review_frame": 0}, token=tok)
    assert r.status_code == 400  # the review (and its rebuild command) opens when the game is over
    row = server.mgr.store.game(gid)
    end = server.post(f"/api/live/{gid}/concede", {}, token=tok).json()
    assert end["meta"]["seed"] == int(row["seed"]) and 0 <= int(row["seed"]) < 2**63
    rep = server.get(f"/api/live/{gid}/review", token=tok).json()
    assert rep["meta"]["seed"] == int(row["seed"]) and "choices" in rep["feedback"]


def test_replays_are_private(server, tmp_path):
    (tmp_path / "replays").mkdir(exist_ok=True)
    (tmp_path / "replays" / "legacy-bot-bot-s1.json").write_text(json.dumps({"format": 1, "meta": {"seed": 1}, "frames": []}))
    _, gid, tok = server.new()
    end = server.post(f"/api/live/{gid}/concede", {}, token=tok).json()
    name = end["live"]["replay"]
    assert [r["file"] for r in server.get("/api/replays").json()] == [name]
    assert server.get("/api/replays", email=BOB).json() == []
    assert server.get(f"/replays/{name}").status_code == 200
    assert server.get(f"/replays/{name}", email=BOB).status_code == 404
    assert server.get("/replays/legacy-bot-bot-s1.json").status_code == 404  # unowned: never served online
    assert server.get("/replays/..%2Fstate%2Fplay.sqlite3").status_code == 404


# ---------------------------------------------------------------- limits
def test_mutations_must_be_same_origin_json_and_small(server):
    v, gid, tok = server.new()
    frame = len(v["frames"]) - 1
    ok = {"frame": frame, "index": 0}
    assert server.post(f"/api/live/{gid}/choose", ok, token=tok, headers={"Origin": "https://evil.example"}).status_code == 403
    h = server.h(token=tok)
    del h["Origin"]
    assert server.c.post(f"/api/live/{gid}/choose", json=ok, headers=h).status_code == 403
    r = server.c.post(f"/api/live/{gid}/choose", content=json.dumps(ok), headers={**server.h(token=tok), "Content-Type": "text/plain"})
    assert r.status_code == 415
    big = json.dumps({"frame": frame, "index": 0, "pad": "x" * 70000})
    assert server.c.post(f"/api/live/{gid}/choose", content=big, headers={**server.h(token=tok), "Content-Type": "application/json"}).status_code == 413

    def chunks():  # no Content-Length to trust: the limit holds while streaming
        for _ in range(80):
            yield b" " * 1024

    r = server.c.post(f"/api/live/{gid}/choose", content=chunks(), headers={**server.h(token=tok), "Content-Type": "application/json"})
    assert r.status_code == 413
    assert server.post(f"/api/live/{gid}/choose", ok, token=tok).status_code == 200


def test_capacity_per_account_and_global(server):
    mgr = server.mgr
    server.new()
    server.new()
    r = server.post("/api/live/new", {"deck": "elves", "opponent": "r9-elves-pilot"})
    assert r.status_code == 429
    mgr.max_games = 3
    server.new(email=BOB)
    r = server.post("/api/live/new", {"deck": "elves", "opponent": "r9-elves-pilot"}, email=BOB)
    assert r.status_code == 503 and "full" in r.json()["error"]


def test_idle_games_expire(server):
    _, gid, tok = server.new()
    server.mgr.idle_seconds = 0.0
    time.sleep(0.01)
    r = server.get(f"/api/live/{gid}", token=tok)
    assert r.status_code == 410 and r.json()["expired"]
    server.mgr.idle_seconds = 3600
    server.new()
    server.new()  # the expired game no longer counts


def test_full_queue_is_refused(server):
    mgr, gate = server.mgr, threading.Event()
    held = [mgr.submit(gate.wait) for _ in range(mgr.max_pending)]
    try:
        r = server.get("/api/live/options")
        assert r.status_code == 503 and "busy" in r.json()["error"]
    finally:
        gate.set()
        concurrent.futures.wait(held)
    assert server.get("/api/live/options").status_code == 200


def test_flags_and_surveys_are_saved_not_filed(server):
    v, gid, tok = server.new()
    frame = len(v["frames"]) - 1
    body = {"category": "bug", "what": "the stack looked wrong", "frame": frame, "pseudonym": "tester"}
    out = server.post(f"/api/live/{gid}/flag", body, token=tok).json()
    assert not out["filed"] and len(out["flags"]) == 1
    again = server.post(f"/api/live/{gid}/flag", body, token=tok).json()  # a retry: no second flag
    assert len(again["flags"]) == 1
    server.post(f"/api/live/{gid}/concede", {}, token=tok)
    assert server.post(f"/api/live/{gid}/survey", {"strength": 4, "hardest": "turn 3"}, token=tok).status_code == 200
    rep = server.get(f"/api/live/{gid}/review", token=tok).json()
    assert rep["feedback"]["survey"]["strength"] == 4 and rep["feedback"]["player"] == "tester" and len(rep["feedback"]["flags"]) == 1
    assert server.mgr.store.one("SELECT survey FROM games WHERE id = ?", (gid,))[0]


def test_dev_user_needs_no_jwt_but_keeps_origin_checks(tmp_path, tiny7):  # noqa: F811
    cfg = make_cfg(tmp_path, tiny7, dev_user="me@local", public_origin="", team_domain="", aud="", allowed_emails=frozenset())
    with TestClient(create_app(cfg, background=False), base_url="http://127.0.0.1:8080") as c:
        assert c.get("/api/live/options").json()["account"] == "me@local"
        assert c.post("/api/live/new", json={"deck": "elves", "opponent": "r9-elves-pilot"}).status_code == 403  # no Origin
        r = c.post("/api/live/new", json={"deck": "elves", "opponent": "r9-elves-pilot"}, headers={"Origin": "http://127.0.0.1:8080"})
        assert r.status_code == 200


def test_zero_opponents_starts_healthy(tmp_path):
    (tmp_path / "models").mkdir()
    offer = tmp_path / "empty.toml"
    offer.write_text('player_decks = ["elves"]\n')
    cfg = make_cfg(tmp_path, tmp_path / "models", dev_user="ci@local", play_config=offer)
    with TestClient(create_app(cfg, background=False)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/api/live/options").json()["opponents"] == []


def test_bad_pin_keeps_healthz_unready(tmp_path, tiny7):  # noqa: F811
    cfg = make_cfg(tmp_path, tiny7, dev_user="ci@local")
    text = cfg.play_config.read_text()
    cfg.play_config.write_text(text.replace(pilot_config(tiny7)["models"][TINY]["sha256"], "0" * 64))
    with TestClient(create_app(cfg, background=False)) as c:
        r = c.get("/healthz")
        assert r.status_code == 503 and r.json()["status"] == "invalid"
        assert c.get("/api/live/options").status_code == 503
