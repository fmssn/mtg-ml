"""Record bot games as replay files and serve the web viewer.

    python -m mtg_ml.replay record --agents bot,bot --seed 3       # -> replays/bot-bot-s3.json
    python -m mtg_ml.replay record --agents bot,bot --games 20     # seeds 0..19
    python -m mtg_ml.replay record --agents model:runs/x/latest.pt,bot   # policy + value per decision
    python -m mtg_ml.replay record --matchup jund_madness --agents model:jund.pt,model:red.pt --greedy --games 20
    python -m mtg_ml.replay serve                                  # http://127.0.0.1:8765

A replay is omniscient (both hands, library sizes, every permanent's state):
one frame per decision, holding the full state at that point, the decision
with all its options and the chosen index, plus the log lines that led
there. The viewer (`viz/viewer.html`) only needs the JSON file.
"""

from __future__ import annotations

import argparse
import http.server
import json
import pathlib
import time
import urllib.parse

from .agents import take
from .backend import engine_name, game_class
from .match import DEFAULT_MATCHUP, game_args, matchup_decks

FORMAT = 1
ROOT = pathlib.Path(__file__).resolve().parent.parent
VIEWER = ROOT / "viz" / "viewer.html"
DECK_TITLES = {"jund_wildfire": "Jund Wildfire", "mono_blue_terror": "Mono Blue Terror", "red_madness": "Red Madness"}


def _card_info(face, token: bool) -> dict:
    return {
        "types": sorted(face.types),
        "subtypes": sorted(face.subtypes),
        "cost": str(face.cost) if not ("Land" in face.types or token) else "",
        "text": face.text,
        "power": face.power,
        "toughness": face.toughness,
        "token": token,
    }


def _card(c, info: dict) -> dict:
    info.setdefault(c.name, _card_info(c.face, c.is_token))
    return {"uid": c.uid, "name": c.name}


def _perm(g, c, info: dict) -> dict:
    d = _card(c, info)
    creature = g.is_creature(c)
    d.update(oid=c.oid, controller=c.controller, types=sorted(g.types(c)))
    if creature:
        d.update(power=g.power(c), toughness=g.toughness(c), keywords=sorted(g.keywords(c)))
    # Default-valued flags are left out to keep replay files small.
    flags = {
        "tapped": c.tapped,
        "sick": c.sick and creature,
        "damage": c.damage,
        "counters": c.counters,
        "attached_to": c.attached_to,
        "attacking": c.oid in g.attackers,
        "blocking": g.blocks.get(c.oid),
        "token": c.is_token,
    }
    d.update((k, v) for k, v in flags.items() if v)
    return d


def _ref_label(g, ref) -> str | None:
    if ref[0] == "player":
        return f"player {ref[1]}"
    if ref[0] == "stack":
        it = g.stack_item(ref[1])
        return None if it is None else f"spell {it.name}"
    c = g.perm(ref[1])
    return None if c is None else c.name


def _known_top(library, n: int = 3) -> list[str]:
    """Names of the top cards someone knows, stopping at the first unknown one,
    so every entry's position is exact (index 0 = top)."""
    out = []
    for c in library[:n]:
        if not c.known_to:
            break
        out.append(c.name)
    return out


def snapshot(g, info: dict) -> dict:
    """Full (omniscient) game state. `info` collects static card data by name."""
    players = []
    for p in g.players:
        players.append(
            {
                "life": p.life,
                "hand": [_card(c, info) for c in p.hand],
                "library": len(p.library),
                "library_top_known": _known_top(p.library),
                "graveyard": [_card(c, info) for c in p.graveyard],
                "exile": [_card(c, info) for c in p.exile],
                "pool": {k: v for k, v in p.pool.items() if v},
                "mulligans": g.mulligans_taken[p.idx],
            }
        )
    stack = []
    for it in g.stack:
        src = it.card or it.source
        if src is not None:
            info.setdefault(src.name, _card_info(src.face, src.is_token))
        stack.append(
            {
                "sid": it.sid,
                "name": it.name,
                "kind": it.kind,
                "card": src.name if src is not None else None,
                "controller": it.controller,
                "targets": [t for t in (_ref_label(g, r) for r in it.targets) if t],
                "target_oids": [r[1] for r in it.targets if r[0] == "perm"],
                "x": it.x,
                "method": it.method,
            }
        )
    return {
        "turn": g.turn,
        "active": g.active,
        "step": g.step_name,
        "players": players,
        "battlefield": [_perm(g, c, info) for c in g.battlefield],
        "stack": stack,
    }


def record(agents, seed: int = 0, decks=None, engine: str | None = None, names=("?", "?"), matchup: str = DEFAULT_MATCHUP, match_game: int = 1, **game_kw) -> dict:
    """Play one game with `agents` and return the replay dict. The decks are
    game `match_game` of `matchup` unless `decks` are given."""
    args = game_args(match_game, matchup)
    if decks is not None:
        args["decks"] = decks
    g = game_class(engine)(**args, seed=seed, log=True, **game_kw)
    info: dict = {}
    frames = []
    seen = 0
    while not g.over:
        d = g.decision
        agent = agents[d.player]
        choice = agent.act(g)
        decision = {
            "player": d.player,
            "kind": d.kind,
            "prompt": d.prompt,
            "options": [o.label for o in d.options],
            "chosen": choice,
        }
        # Model agents report their policy and value estimate for the viewer.
        decision.update(getattr(agent, "last_info", None) or {})
        frames.append({"state": snapshot(g, info), "events": g.log[seen:], "decision": decision})
        seen = len(g.log)
        take(g, agents, choice)
    frames.append({"state": snapshot(g, info), "events": g.log[seen:], "decision": None})
    return {
        "format": FORMAT,
        "meta": {
            "seed": seed,
            "engine": engine_name(engine),
            "agents": list(names),
            "decks": [DECK_TITLES[d] for d in matchup_decks(matchup)],
            "match_game": match_game,
            "starting_player": g.starting_player,
            "winner": g.winner,
            "end_reason": g.end_reason,
            "turns": g.turn,
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "cards": info,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Viewer server
# ---------------------------------------------------------------------------


def _replay_summary(path: pathlib.Path) -> dict:
    """Listing entry for one replay file; malformed files list without metadata."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = None
    meta = data.get("meta") if isinstance(data, dict) else None
    return {**(meta if isinstance(meta, dict) else {}), "file": path.name, "mtime": path.stat().st_mtime}


def serve(directory: pathlib.Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    directory = directory.resolve()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(VIEWER.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/replays":
                files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                return self._send(json.dumps([_replay_summary(p) for p in files]).encode(), "application/json")
            if path.startswith("/replays/"):
                f = (directory / urllib.parse.unquote(path[len("/replays/"):])).resolve()
                if f.parent == directory and f.suffix == ".json" and f.is_file():
                    return self._send(f.read_bytes(), "application/json")
            self._send(b"not found", "text/plain", 404)

        def log_message(self, fmt, *args) -> None:
            pass

    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    print(f"serving {directory} at http://{host}:{port}  (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def _agent_name(kind: str) -> str:
    """model:runs/foo/latest.pt -> model:foo (the run directory names the model)."""
    if not kind.startswith("model:"):
        return kind
    p = pathlib.Path(kind[len("model:"):])
    return "model:" + (p.parent.name if p.stem in ("latest", "model") else p.stem)


def main(argv=None) -> None:
    from .play import make_agent

    ap = argparse.ArgumentParser(prog="mtg_ml.replay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="play games and write replay files")
    r.add_argument("--agents", default="bot,bot", help="seat0,seat1 from: random, bot, search[:playouts], model:<checkpoint>")
    r.add_argument("--greedy", action="store_true", help="models take their most likely option instead of sampling")
    r.add_argument("--seed", type=int, default=0, help="first seed")
    r.add_argument("--games", type=int, default=1)
    r.add_argument("--out", default="replays", help="output directory")
    r.add_argument("--engine", default=None, help="python or native; default: $MTG_ENGINE, else python")
    r.add_argument("--matchup", default=DEFAULT_MATCHUP, help="match.MATCHUPS: jund_blue or jund_madness")
    r.add_argument("--match-game", type=int, default=1, help="1: maindecks; 2/3: sideboarded")
    r.add_argument("--starting-player", type=int, default=None, help="0 or 1 (default: by the seed)")
    s = sub.add_parser("serve", help="serve the web viewer")
    s.add_argument("--dir", default="replays")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    if args.cmd == "record":
        kinds = args.agents.split(",")
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for seed in range(args.seed, args.seed + args.games):
            decks = matchup_decks(args.matchup)
            agents = [make_agent(k, i, seed, greedy=args.greedy, deck=decks[i]) for i, k in enumerate(kinds)]
            names = [_agent_name(k) for k in kinds]
            rep = record(agents, seed=seed, engine=args.engine, names=names, matchup=args.matchup, match_game=args.match_game, starting_player=args.starting_player)
            tag = "" if args.matchup == DEFAULT_MATCHUP else f"{args.matchup}-g{args.match_game}-"
            path = out / f"{tag}{'-'.join(n.replace(':', '_') for n in names)}-s{seed}.json"
            path.write_text(json.dumps(rep, separators=(",", ":")))
            m = rep["meta"]
            print(f"{path}: {len(rep['frames'])} frames, {m['turns']} turns, winner {m['winner']} ({m['end_reason']})")
    else:
        serve(pathlib.Path(args.dir), args.host, args.port)


if __name__ == "__main__":
    main()
