"""Play against a checkpoint in the web viewer.

    python -m mtg_ml.replay serve --models runs/     # then "Play" in the viewer

The server holds each game; the browser only ever gets what the human
player may see (`replay.snapshot(..., viewer=seat)`): the model's hand is
face down, its private choices are cut from the log, and its policy and
value stay on the server. The model moves until the human has a decision,
so one request answers each human choice. A finished game is written to the
replay directory as an ordinary (omniscient) replay, model policy included.

The player also gets a frame per model decision, so the viewer can play the
model's turn back step by step. Those show the chosen option only for
decision kinds whose choice is public (`rl.features.PUBLIC_KINDS`, the rule
the model itself sees its opponent by); a scry or a put-back shows just
its kind.

The browser picks a checkpoint by name from the `--models` directory, never
by path: loading a checkpoint unpickles it, which can run code.
"""

from __future__ import annotations

import gc
import pathlib
import secrets
import time
from concurrent.futures import ThreadPoolExecutor

from .agents import take
from .backend import engine_name, game_class
from .match import MATCHUPS, game_args, matchup_decks
from .replay import DECK_TITLES, FORMAT, snapshot, visible_events
from .rl.features import PUBLIC_KINDS

MAX_GAMES = 8  # games held at once; the oldest idle one makes room
IDLE_SECONDS = 3600  # a game nobody touched for this long is dropped


class LiveError(ValueError):
    """A request the server refuses; the message is shown to the player."""


class Human:
    """Placeholder agent for the browser's seat (`agents.take` only needs `observe` on models)."""


def find_models(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    """Checkpoints under `directory` (up to two levels deep) by display name:
    the path relative to it without `.pt`, e.g. `v3-entity/model`."""
    directory = directory.resolve()
    found = sorted(directory.glob("*.pt")) + sorted(directory.glob("*/*.pt"))
    return {p.relative_to(directory).with_suffix("").as_posix(): p for p in found if p.is_file()}


class LiveGame:
    def __init__(self, gid: str, model_name: str, model_path: pathlib.Path, seat: int, matchup: str, greedy: bool, seed: int, engine: str | None):
        from .rl.agent import ModelAgent

        self.id, self.seat, self.matchup, self.engine = gid, seat, matchup, engine
        self.touched = time.monotonic()
        self.model = ModelAgent(str(model_path), 1 - seat, sample=not greedy, seed=seed)
        self.agents = [None, None]
        self.agents[seat], self.agents[1 - seat] = Human(), self.model
        self.names = ["", ""]
        self.names[seat], self.names[1 - seat] = "you", f"model:{model_name}" + (" (greedy)" if greedy else "")
        self.g = game_class(engine)(**game_args(1, matchup), seed=seed, log=True)
        self.seed = seed
        self.full_info: dict = {}  # card data for the saved replay
        self.view_info: dict = {}  # card data the player has seen
        self.full_frames: list[dict] = []  # omniscient, one per decision (the saved replay)
        self.frames: list[dict] = []  # the player's view, one per player decision
        self.full_seen = self.view_seen = 0  # log lines already in a frame
        self.replay_file: str | None = None
        self._advance()

    # -- game loop
    def _record(self, decision: dict | None) -> None:
        g = self.g
        self.full_frames.append({"state": snapshot(g, self.full_info), "events": g.log[self.full_seen :], "decision": decision})
        self.full_seen = len(g.log)

    def _advance(self) -> None:
        """Let the model play until the player decides or the game ends."""
        g = self.g
        while not g.over and g.decision.player != self.seat:
            d = g.decision
            choice = self.model.act(g)
            options = [o.label for o in d.options]
            self._record({"player": d.player, "kind": d.kind, "prompt": d.prompt, "options": options, "chosen": choice, **(self.model.last_info or {})})
            if d.kind in PUBLIC_KINDS:
                shown = {"prompt": d.prompt, "options": [options[choice]]}
            else:
                shown = {"prompt": "", "options": [f"(hidden {d.kind.replace('_', ' ')})"]}
            self._view_frame({"player": d.player, "kind": d.kind, **shown, "chosen": 0})
            take(g, self.agents, choice)
        decision = None
        if not g.over:
            d = g.decision
            decision = {"player": d.player, "kind": d.kind, "prompt": d.prompt, "options": [o.label for o in d.options], "chosen": None}
        self._view_frame(decision)

    def _view_frame(self, decision: dict | None) -> None:
        g = self.g
        self.frames.append({"state": snapshot(g, self.view_info, viewer=self.seat), "events": visible_events(g.log[self.view_seen :], self.seat), "decision": decision})
        self.view_seen = len(g.log)

    def choose(self, frame: int, index: int) -> None:
        """Take option `index` at the player's decision `frame` (the frame
        number guards against a repeated or stale click)."""
        g = self.g
        if g.over:
            raise LiveError("the game is over")
        if frame != len(self.frames) - 1:
            raise LiveError("that decision was already made")
        d = self.frames[-1]["decision"]
        if not (isinstance(index, int) and 0 <= index < len(d["options"])):
            raise LiveError("no such option")
        d["chosen"] = index
        self._record({k: d[k] for k in ("player", "kind", "prompt", "options")} | {"chosen": index})
        take(g, self.agents, index)
        self._advance()

    # -- output
    def _meta(self) -> dict:
        g = self.g
        return {
            "seed": self.seed,
            "engine": engine_name(self.engine),
            "agents": self.names,
            "decks": [DECK_TITLES[d] for d in matchup_decks(self.matchup)],
            "match_game": 1,
            "starting_player": g.starting_player,
            "winner": g.winner if g.over else None,
            "end_reason": g.end_reason if g.over else "",
            "turns": g.turn,
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def replay(self) -> dict:
        """The finished game as an ordinary omniscient replay."""
        frames = self.full_frames + [{"state": snapshot(self.g, self.full_info), "events": self.g.log[self.full_seen :], "decision": None}]
        return {"format": FORMAT, "meta": self._meta(), "cards": self.full_info, "frames": frames}

    def view(self, since: int = 0) -> dict:
        """The player's view: frames from `since` on (the client keeps the
        rest), in the replay format, plus the live bookkeeping."""
        since = max(0, min(since, len(self.frames) - 1))
        return {
            "format": FORMAT,
            "meta": self._meta(),
            "cards": self.view_info,
            "frames": self.frames[since:],
            "live": {"id": self.id, "seat": self.seat, "since": since, "over": self.g.over, "replay": self.replay_file},
        }


class LiveManager:
    """The games a server holds, by unguessable id. Every game is created and
    played on one worker thread: native games may only be touched by the
    thread that made them, and the HTTP server answers each request on a
    thread of its own. A move takes milliseconds, so one thread is plenty."""

    def __init__(self, models_dir: pathlib.Path, replay_dir: pathlib.Path, engine: str | None = None, max_games: int = MAX_GAMES):
        import torch

        torch.set_num_threads(1)  # one forward pass per decision; threads only contend
        self.models_dir, self.replay_dir, self.engine, self.max_games = models_dir, replay_dir, engine, max_games
        self.games: dict[str, LiveGame] = {}
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="live")

    def options(self) -> dict:
        return {
            "models": sorted(find_models(self.models_dir)),
            "matchups": {k: [DECK_TITLES[d] for d in v] for k, v in MATCHUPS.items()},
        }

    def close(self) -> None:
        """Drop every game (on the worker, where native games must be freed)."""
        self._run(self._drop, list(self.games))
        self.worker.shutdown()

    def _drop(self, gids: list[str]) -> None:
        for gid in gids:
            del self.games[gid]
        gc.collect()  # native objects in reference cycles must die on this thread too

    def _run(self, fn, *args):
        return self.worker.submit(fn, *args).result()

    def new(self, req: dict) -> dict:
        return self._run(self._new, req)

    def view(self, gid: str, since: int = 0) -> dict:
        return self._run(lambda: self._get(gid).view(since))

    def choose(self, gid: str, req: dict) -> dict:
        return self._run(self._choose, gid, req)

    def _new(self, req: dict) -> dict:
        models = find_models(self.models_dir)
        name, seat, matchup = req.get("model"), req.get("seat", 0), req.get("matchup", "jund_blue")
        if name not in models:
            raise LiveError(f"unknown model {name!r}")
        if seat not in (0, 1):
            raise LiveError("seat must be 0 or 1")
        if matchup not in MATCHUPS:
            raise LiveError(f"unknown matchup {matchup!r}")
        seed = req.get("seed")
        seed = secrets.randbelow(2**31) if seed is None else int(seed)
        gid = secrets.token_urlsafe(9)
        try:
            game = LiveGame(gid, name, models[name], seat, matchup, bool(req.get("greedy")), seed, self.engine)
        except LiveError:
            raise
        except Exception as e:  # a checkpoint this code cannot run (old features, other config)
            raise LiveError(f"cannot play {name}: {e}") from e
        self._evict()
        self.games[gid] = game
        self._finish(game)
        return game.view()

    def _evict(self) -> None:
        now = time.monotonic()
        old = sorted(self.games, key=lambda k: self.games[k].touched)
        n = max(len(old) - self.max_games + 1, sum(now - self.games[k].touched > IDLE_SECONDS for k in old))
        if n > 0:
            self._drop(old[:n])

    def _get(self, gid: str) -> LiveGame:
        game = self.games.get(gid)
        if game is None:
            raise LiveError("no such game (it may have expired)")
        game.touched = time.monotonic()
        return game

    def _choose(self, gid: str, req: dict) -> dict:
        game = self._get(gid)
        game.choose(req.get("frame"), req.get("index"))
        self._finish(game)
        return game.view(int(req.get("since", 0)))

    def _finish(self, game: LiveGame) -> None:
        """Write the replay of a game that just ended."""
        import json

        if not game.g.over or game.replay_file:
            return
        model = game.names[1 - game.seat].split(":", 1)[1].split(" ")[0].replace("/", "_")
        name = f"human-vs-{model}-{game.id}.json" if game.seat == 0 else f"{model}-vs-human-{game.id}.json"
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        (self.replay_dir / name).write_text(json.dumps(game.replay(), separators=(",", ":")))
        game.replay_file = name
