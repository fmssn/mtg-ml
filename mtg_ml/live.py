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
its kind. The prompt never goes out: a public choice is safe to name, but a
prompt can mention a card only the decider knows (Delver of Secrets asks
"reveal <top of my library>?").

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
from .engine import DECKS, SIDEBOARDS, expand, plan_for, postboard
from .match import MATCHUPS, game_args, matchup_decks
from .live_proto import add_tap_previews, auto_pay_index, combat_and_life_lines, option_ref, option_refs, parse_events, status, visible_ids
from .replay import DECK_TITLES, FORMAT, snapshot, visible_events
from .rl.features import PUBLIC_KINDS

SCRIPTED = "scripted-bot"  # the deck's hand-written bot, offered with --scripted-bot (dev, no checkpoint needed)
MAX_GAMES = 8  # games held at once; only an idle or finished game makes room
IDLE_SECONDS = 3600  # a game nobody touched for this long is dropped
ACTIVE_SECONDS = 600  # a game touched within this long is being played: never evicted


def live_dev_options() -> dict[str, str]:
    from .live_dev import options

    return options()


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
    def __init__(self, gid: str, model_name: str, model_path: pathlib.Path | None, seat: int, matchup: str, greedy: bool, seed: int, engine: str | None,
                 scenario: str | None = None, match: dict | None = None, game_no: int = 1, start: int | None = None, plan: str = "standard"):
        """`model_path` None: the deck's scripted bot plays instead (`SCRIPTED`).
        `scenario`: a `live_dev` scenario (dev servers only); it sets the matchup and seat.
        `match`: the best-of-three this game belongs to (shared with its other
        games); `game_no` 2 and 3 are sideboarded: the model plays the plan
        table's plan, the player `plan` ("standard": the same table's plan
        for their deck, "maindeck": no changes). `start`: the starting player."""
        game = None
        self.match, self.game_no, self.plan = match, game_no, plan
        self.conceded = False
        self.model_name, self.greedy, self.scenario = model_name, greedy, scenario
        if scenario is not None:
            from . import live_dev

            game, matchup, seat = live_dev.new_game(scenario, seed, engine)
        self.id, self.seat, self.matchup, self.engine = gid, seat, matchup, engine
        self.touched = time.monotonic()
        if model_path is None:
            from .bots import make_bot

            self.model = make_bot(1 - seat, matchup_decks(matchup)[1 - seat])
        else:
            from .rl.agent import ModelAgent

            self.model = ModelAgent(str(model_path), 1 - seat, sample=not greedy, seed=seed)
        self.agents = [None, None]
        self.agents[seat], self.agents[1 - seat] = Human(), self.model
        self.names = ["", ""]
        self.names[seat], self.names[1 - seat] = "you", f"model:{model_name}" + (" (greedy)" if greedy else "")
        if game is None:
            args = game_args(game_no, matchup)
            if game_no > 1:  # sideboarded: the model takes the table's plan, the player the plan they chose
                decks = list(matchup_decks(matchup))
                mine = postboard(decks[seat], decks[1 - seat]) if plan == "standard" else DECKS[decks[seat]]
                lists = [None, None]
                lists[seat], lists[1 - seat] = expand(mine), expand(postboard(decks[1 - seat], decks[seat]))
                args["decks"] = tuple(lists)
            game = game_class(engine)(**args, seed=seed, log=True, **({} if start is None else {"starting_player": start}))
        self.g = game
        self.seed = seed
        self.full_info: dict = {}  # card data for the saved replay
        self.view_info: dict = {}  # card data the player has seen
        self.full_frames: list[dict] = []  # omniscient, one per decision (the saved replay)
        self.frames: list[dict] = []  # the player's view, one per player decision
        self.full_seen = self.view_seen = 0  # log lines already in a frame
        # The engine's log plus lines the live layer adds (combat damage, life
        # changes: live_proto.combat_and_life_lines); frames read this one.
        self.log: list[str] = []
        self.engine_seen = 0  # engine log lines copied into self.log
        self.replay_file: str | None = None
        self._advance()

    def _sync_log(self) -> list[str]:
        new = list(self.g.log[self.engine_seen :])
        self.engine_seen += len(new)
        self.log.extend(new)
        return new

    def _take(self, index: int) -> None:
        """Take option `index` and log the combat damage and life changes it led to."""
        g = self.g
        self._sync_log()
        before = status(g)
        take(g, self.agents, index)
        new = self._sync_log()
        extra = combat_and_life_lines(before, status(g), new)
        if not extra:
            return
        # Put the lines where they happened: right after the combat damage step
        # began, else before the next turn starts or the game ends.
        start = len(self.log) - len(new)
        at = next((k + 1 for k, line in enumerate(new) if line == "-- combat_damage"), None)
        if at is None:
            at = next((k for k, line in enumerate(new) if line.startswith("=== Turn") or line.startswith("GAME OVER")), len(new))
        self.log[start + at : start + at] = extra

    # -- game loop
    def _record(self, decision: dict | None) -> None:
        g = self.g
        self._sync_log()
        self.full_frames.append({"state": snapshot(g, self.full_info), "events": self.log[self.full_seen :], "decision": decision})
        self.full_seen = len(self.log)

    def _advance(self) -> None:
        """Let the model play until the player decides or the game ends."""
        g = self.g
        while not g.over and g.decision.player != self.seat:
            d = g.decision
            choice = self.model.act(g)
            options = [o.label for o in d.options]
            self._record({"player": d.player, "kind": d.kind, "prompt": d.prompt, "options": options, "chosen": choice, **(getattr(self.model, "last_info", None) or {})})
            # The chosen option of a public kind is safe to show; the prompt is
            # not: it can name a card only the decider knows (Delver of
            # Secrets: "reveal <top of my library>?"), so it stays on the server.
            state = self._view_state()
            if d.kind in PUBLIC_KINDS:
                o = d.options[choice]
                shown, refs = [options[choice]], [option_ref(d.kind, o.label, o.key, o.value, visible_ids(state))]
            else:
                shown, refs = [f"(hidden {d.kind.replace('_', ' ')})"], [{"type": "hidden"}]
            self._view_frame(state, {"player": d.player, "kind": d.kind, "prompt": "", "options": shown, "refs": refs, "chosen": 0})
            self._take(choice)
        decision = None
        state = self._view_state()
        if not g.over:
            d = g.decision
            decision = {"player": d.player, "kind": d.kind, "prompt": d.prompt, "options": [o.label for o in d.options], "refs": option_refs(d, state), "chosen": None}
            if d.kind == "pay_mana":
                decision["auto"] = auto_pay_index(g, d)
            elif d.kind == "priority":
                add_tap_previews(g, decision["refs"], self.seat, visible_ids(state))
        self._view_frame(state, decision)

    def _view_state(self) -> dict:
        return snapshot(self.g, self.view_info, viewer=self.seat)

    def _view_frame(self, state: dict, decision: dict | None) -> None:
        """A frame of the player's view: the state, the visible log lines since
        the last frame, the same lines parsed for animation (`actions`), and
        the decision (with structured option `refs`)."""
        self._sync_log()
        events = visible_events(self.log[self.view_seen :], self.seat)
        self.frames.append({"state": state, "events": events, "actions": parse_events(events), "decision": decision})
        self.view_seen = len(self.log)

    # -- result (a concession ends the game without the engine)
    @property
    def over(self) -> bool:
        return self.conceded or self.g.over

    @property
    def winner(self) -> int | None:
        return 1 - self.seat if self.conceded else self.g.winner if self.g.over else None

    @property
    def end_reason(self) -> str:
        return "concede" if self.conceded else self.g.end_reason if self.g.over else ""

    def concede(self) -> None:
        if self.over:
            raise LiveError("the game is over")
        self.conceded = True
        self.log.append(f"p{self.seat} concedes")
        self._view_frame(self._view_state(), None)

    def choose(self, frame: int, index: int) -> None:
        """Take option `index` at the player's decision `frame` (the frame
        number guards against a repeated or stale click)."""
        if self.over:
            raise LiveError("the game is over")
        if frame != len(self.frames) - 1:
            raise LiveError("that decision was already made")
        d = self.frames[-1]["decision"]
        if not (isinstance(index, int) and 0 <= index < len(d["options"])):
            raise LiveError("no such option")
        d["chosen"] = index
        self._record({k: d[k] for k in ("player", "kind", "prompt", "options")} | {"chosen": index})
        self._take(index)
        self._advance()

    # -- output
    def _meta(self) -> dict:
        g = self.g
        return {
            "seed": self.seed,
            "engine": engine_name(self.engine),
            "agents": self.names,
            "decks": [DECK_TITLES[d] for d in matchup_decks(self.matchup)],
            "match_game": self.game_no,
            "starting_player": g.starting_player,
            "winner": self.winner,
            "end_reason": self.end_reason,
            "match": self._match_meta(),
            "turns": g.turn,
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _match_meta(self) -> dict | None:
        m = self.match
        if m is None:
            return None
        results = m["results"]
        wins = [sum(1 for _, w in results if w == p) for p in (0, 1)]
        over = max(wins) >= 2 or len(results) >= 3
        last_loser = None if not results or results[-1][1] is None else 1 - results[-1][1]
        return {"game_no": self.game_no, "results": results, "wins": wins, "over": over, "you_choose_play": last_loser == self.seat, "plan": self.plan}

    def replay(self) -> dict:
        """The finished game as an ordinary omniscient replay."""
        self._sync_log()
        frames = self.full_frames + [{"state": snapshot(self.g, self.full_info), "events": self.log[self.full_seen :], "decision": None}]
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
            "live": {"id": self.id, "seat": self.seat, "since": since, "over": self.over, "replay": self.replay_file, "scenario": self.scenario, "matchup": self.matchup},
        }


class LiveManager:
    """The games a server holds, by unguessable id. Every game is created and
    played on one worker thread: native games may only be touched by the
    thread that made them, and the HTTP server answers each request on a
    thread of its own. A move takes milliseconds, so one thread is plenty."""

    def __init__(self, models_dir: pathlib.Path | None, replay_dir: pathlib.Path, engine: str | None = None, max_games: int = MAX_GAMES, scripted: bool = False):
        """`models_dir` None: no checkpoints (only the scripted bot, with `scripted`)."""
        if models_dir is not None:
            import torch

            torch.set_num_threads(1)  # one forward pass per decision; threads only contend
        self.models_dir, self.replay_dir, self.engine, self.max_games, self.scripted = models_dir, replay_dir, engine, max_games, scripted
        self.games: dict[str, LiveGame] = {}
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="live")

    def options(self) -> dict:
        return {
            "models": sorted(self._models()) + ([SCRIPTED] if self.scripted else []),
            "scenarios": live_dev_options() if self.scripted else {},
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

    def concede(self, gid: str) -> dict:
        return self._run(self._concede, gid)

    def next_game(self, gid: str, req: dict) -> dict:
        return self._run(self._next, gid, req)

    def sideboard(self, matchup: str, seat: int) -> dict:
        """The standard plan for the player's deck in `matchup` (what "standard" swaps)."""
        if matchup not in MATCHUPS or seat not in (0, 1):
            raise LiveError("unknown matchup or seat")
        decks = matchup_decks(matchup)
        plan = plan_for(decks[seat], decks[1 - seat])
        return {"in": dict(plan.cards_in), "out": dict(plan.cards_out), "deck": DECK_TITLES[decks[seat]]}

    def decklists(self, matchup: str, seat: int, game_no: int = 1) -> dict:
        """Both decks of `matchup` (maindeck and sideboard): public in Pauper, and
        what a player wants to know about the opponent."""
        if matchup not in MATCHUPS or seat not in (0, 1):
            raise LiveError("unknown matchup or seat")
        decks = matchup_decks(matchup)
        from .engine.cards import CARDS

        def deck(d: str) -> dict:
            names = {**DECKS[d], **SIDEBOARDS[d]}
            return {"title": DECK_TITLES[d], "main": dict(DECKS[d]), "side": dict(SIDEBOARDS[d]),
                    "types": {n: sorted(CARDS[n].types) for n in names if n in CARDS}}

        return {"mine": deck(decks[seat]), "theirs": deck(decks[1 - seat])}

    def _concede(self, gid: str) -> dict:
        game = self._get(gid)
        game.concede()
        self._finish(game)
        return game.view(len(game.frames) - 2)

    def _next(self, gid: str, req: dict) -> dict:
        """The next game of the match (or a rematch: a new match, same settings)."""
        prev = self._get(gid)
        if not prev.over:
            raise LiveError("finish or concede this game first")
        if prev.scenario is not None:
            return self._new({"model": prev.model_name, "scenario": prev.scenario, "greedy": prev.greedy})
        mm = prev._match_meta()
        if mm is None or mm["over"]:
            return self._new({"model": prev.model_name, "seat": prev.seat, "matchup": prev.matchup, "greedy": prev.greedy})
        plan = req.get("plan", "standard")
        if plan not in ("standard", "maindeck"):
            raise LiveError("plan must be 'standard' or 'maindeck'")
        last_start, last_winner = prev.match["results"][-1]
        if last_winner is None:
            start = last_start
        elif last_winner == prev.seat:  # the model lost: it chooses to play first
            start = 1 - prev.seat
        else:  # you lost: you choose
            start = prev.seat if req.get("play", True) else 1 - prev.seat
        return self._new({"model": prev.model_name, "seat": prev.seat, "matchup": prev.matchup, "greedy": prev.greedy},
                         match=prev.match, game_no=prev.game_no + 1, start=start, plan=plan)

    def _models(self) -> dict[str, pathlib.Path]:
        return {} if self.models_dir is None else find_models(self.models_dir)

    def _new(self, req: dict, match: dict | None = None, game_no: int = 1, start: int | None = None, plan: str = "standard") -> dict:
        models: dict = self._models()
        if self.scripted:
            models[SCRIPTED] = None
        name, seat, matchup = req.get("model"), req.get("seat", 0), req.get("matchup", "jund_blue")
        scenario = req.get("scenario") or None
        if scenario is not None and (not self.scripted or scenario not in live_dev_options()):
            raise LiveError(f"unknown scenario {scenario!r}")
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
            if match is None and scenario is None:
                match = {"id": secrets.token_urlsafe(6), "results": []}
            self._evict()  # make room first: a full server refuses before any work
            game = LiveGame(gid, name, models[name], seat, matchup, bool(req.get("greedy")), seed, self.engine, scenario, match, game_no, start, plan)
        except LiveError:
            raise
        except Exception as e:  # a checkpoint this code cannot run (old features, other config)
            raise LiveError(f"cannot play {name}: {e}") from e
        self.games[gid] = game
        self._finish(game)
        return game.view()

    def _evict(self) -> None:
        """Drop expired games, then make room for one more by dropping idle or
        finished games (oldest first). A game being played is never dropped:
        when every slot holds one, the new game is refused."""
        now = time.monotonic()
        self._drop([k for k, g in self.games.items() if now - g.touched > IDLE_SECONDS])
        if len(self.games) < self.max_games:
            return
        spare = sorted((k for k, g in self.games.items() if g.over or now - g.touched > ACTIVE_SECONDS), key=lambda k: self.games[k].touched)
        need = len(self.games) - self.max_games + 1
        if len(spare) < need:
            raise LiveError(f"The server is full: {len(self.games)} games are being played. Please try again in a few minutes.")
        self._drop(spare[:need])

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

        if not game.over or game.replay_file:
            return
        if game.match is not None:
            game.match["results"].append((game.g.starting_player, game.winner))
        model = game.names[1 - game.seat].split(":", 1)[1].split(" ")[0].replace("/", "_")
        name = f"human-vs-{model}-{game.id}.json" if game.seat == 0 else f"{model}-vs-human-{game.id}.json"
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        (self.replay_dir / name).write_text(json.dumps(game.replay(), separators=(",", ":")))
        game.replay_file = name
