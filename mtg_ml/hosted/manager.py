"""Games of verified accounts, recorded in SQLite and rebuilt after a restart.

`HostedManager` is a `live.LiveManager` (the same games, the same single
worker thread: native games may only be touched by the thread that made
them) with what a public server adds:

- every game, match and replay belongs to one account; a game request needs
  the owner's account *and* the game's token (`X-Game-Token`). Another
  account gets "no such game", never a hint that the id exists;
- the seed is drawn here (63 bits) and stays private until the game is over;
- capacity: two unfinished games per account, eight in all, an hour of idle
  time, a bounded queue of pending requests;
- each action is committed (`store.Store`) before it is answered: the new
  choices (the player's and the model's), the result, the replay, the
  feedback. A game that is not in memory (a restart) is rebuilt on the
  worker from its seed and recorded choices; the model plays its own
  decisions again, which rebuilds its recurrent memory, and each must equal
  the recorded one. A changed checkpoint, code revision or engine, or any
  divergence, marks the game unrecoverable, and the API says so;
- retries are harmless: a `choose` for a decision already taken with the
  same option, a repeated `next` (the same next game), a repeated flag.

Methods named `h_*` run on the worker (`submit`); nothing else touches a game.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import threading
import time

from ..backend import engine_name
from ..live import IDLE_SECONDS, MAX_GAMES, LiveError, LiveGame, LiveManager
from ..live_issues import MAX_ISSUES_PER_GAME, FilingError
from .store import Store

PER_ACCOUNT = 2  # unfinished games one account may hold
MAX_PENDING = 16  # requests waiting for the game worker before new ones are refused
KEEP_FINISHED = 8  # finished games kept in memory (others are rebuilt when asked for)


class HostedError(LiveError):
    """A refusal with an HTTP status (and extra fields for the page)."""

    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra
        self.forbidden = status == 403


class Divergence(Exception):
    pass


NO_GAME = "no such game (it may have expired)"


class HostedManager(LiveManager):
    def __init__(self, models_dir: pathlib.Path, replay_dir: pathlib.Path, state_dir: pathlib.Path, *, config: dict, pinned: dict[str, dict],
                 runtime: str, engine: str | None = None, filer=None, max_games: int = MAX_GAMES, per_account: int = PER_ACCOUNT,
                 idle_seconds: float = IDLE_SECONDS, max_pending: int = MAX_PENDING):
        super().__init__(models_dir, replay_dir, engine=engine, max_games=max_games, config=config, filer=filer, pinned=pinned)
        self.runtime, self.per_account, self.idle_seconds, self.max_pending = runtime, per_account, idle_seconds, max_pending
        state_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(state_dir / "play.sqlite3")
        self.key = _token_key(state_dir / "token.key")
        self._pending = 0
        self._plock = threading.Lock()

    # -- the worker, bounded
    def submit(self, fn, *args):
        """Queue `fn` on the game worker; refused (503) when too many requests wait."""
        with self._plock:
            if self._pending >= self.max_pending:
                raise HostedError("The server is busy. Please try again in a moment.", 503)
            self._pending += 1
        try:
            fut = self.worker.submit(fn, *args)
        except BaseException:
            self._done(None)
            raise
        fut.add_done_callback(self._done)
        return fut

    def _done(self, _fut) -> None:
        with self._plock:
            self._pending -= 1

    def _run(self, fn, *args):
        return self.submit(fn, *args).result()

    def close(self) -> None:
        super().close()
        self.store.close()

    @staticmethod
    def new_seed() -> int:
        """A private 63-bit seed (engines and ModelAgent's seed*2+seat generator take it)."""
        return secrets.randbits(63)

    # -- tokens: derived from the game id with a server key (state dir), so a
    # retried `next` and a restored game hand out the same one; the database
    # keeps only its hash.
    def token_for(self, gid: str) -> str:
        return base64.urlsafe_b64encode(hmac.new(self.key, b"game-token:" + gid.encode(), hashlib.sha256).digest())[:22].decode()

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    # -- options
    def h_options(self, account: str) -> dict:
        out = self.options()
        out.update(hosted=True, account=account, limits={"per_account": self.per_account, "idle_minutes": round(self.idle_seconds / 60)})
        return out

    # -- authorization and loading (worker)
    def _sweep(self) -> None:
        for gid in self.store.expire(time.time() - self.idle_seconds):
            if gid in self.games:
                self._drop([gid])

    def _authorize(self, account: str, gid: str, token: str | None):
        self._sweep()
        row = self.store.game(gid) if isinstance(gid, str) else None
        if row is None or row["owner"] != account:
            raise HostedError(NO_GAME, 404)
        if not token or not hmac.compare_digest(self.token_hash(str(token)), row["token_hash"]):
            raise HostedError("this game's key is missing or wrong (it was started in another browser?)", 403)
        if row["status"] == "expired":
            raise HostedError("This game expired: nobody moved for an hour.", 410, expired=True)
        if row["status"] == "unrecoverable":
            raise HostedError(f"This game cannot be resumed: {row['problem']}", 409, unrecoverable=True)
        return row

    def _load(self, row) -> LiveGame:
        game = self.games.get(row["id"])
        if game is None:
            game = self._restore(row)
            self.games[row["id"]] = game
            self._trim()
        game.touched = time.monotonic()
        return game

    def _unrecoverable(self, row, problem: str):
        if row["status"] == "active":
            self.store.mark_unrecoverable(row["id"], problem)
        return HostedError(f"This game cannot be resumed: {problem}", 409, unrecoverable=True)

    def _restore(self, row) -> LiveGame:
        """Rebuild a recorded game (on the worker) and check it against the record."""
        pin = (self.pinned or {}).get(row["model"])
        model_path = self._models().get(row["model"])
        if pin is None or model_path is None:
            raise self._unrecoverable(row, f"its model {row['model']} is no longer offered on this server")
        if pin.get("sha256") != row["checkpoint_sha256"]:
            raise self._unrecoverable(row, "the model's checkpoint changed since the game started")
        if row["runtime"] != self.runtime:
            raise self._unrecoverable(row, f"the server's code changed since the game started ({row['runtime']} -> {self.runtime})")
        if row["engine"] != engine_name(self.engine):
            raise self._unrecoverable(row, f"the server's engine changed ({row['engine']} -> {engine_name(self.engine)})")
        match = {"id": row["match_id"], "results": self.store.match_results(row["match_id"], row["game_no"] - 1)}
        record = self.store.choices(row["id"])
        try:
            game = LiveGame(row["id"], row["model"], model_path, row["seat"], row["matchup"], bool(row["greedy"]), int(row["seed"]), self.engine,
                            None, match, row["game_no"], row["start_arg"], row["plan"])
            done = _verify(game, record, 0)
            while done < len(record):
                player, index = record[done]
                if game.over or player != game.seat:
                    raise Divergence(f"the record has a choice by p{player} where the game waits for p{game.g.decision.player if not game.over else '-'}")
                game.choose(len(game.frames) - 1, index)
                done = _verify(game, record, done)
            if row["conceded"] and not game.over:
                game.concede()
            if game.over != (row["status"] == "finished"):
                raise Divergence(f"the rebuilt game is {'over' if game.over else 'running'}, the record says {row['status']}")
        except (Divergence, LiveError) as e:
            raise self._unrecoverable(row, f"replaying its record diverged ({e})") from None
        except Exception as e:  # noqa: BLE001 (a checkpoint or engine that cannot rebuild it)
            raise self._unrecoverable(row, f"rebuilding it failed ({type(e).__name__}: {e})") from None
        game.hide_seed = True
        game.token = self.token_for(row["id"])
        game.recorded = len(game.full_frames)
        game.flags = self.store.flags(row["id"])
        game.survey = json.loads(row["survey"]) if row["survey"] else None
        game.pseudonym = row["pseudonym"] or ""
        game.owner, game.opponent = row["owner"], row["opponent"]
        if game.over:
            game.replay_file = row["replay_file"]
            match["results"] = self.store.match_results(row["match_id"], row["game_no"])
        return game

    def _trim(self) -> None:
        """Keep every unfinished game in memory; drop the oldest finished ones beyond KEEP_FINISHED."""
        done = sorted((k for k, g in self.games.items() if g.over), key=lambda k: self.games[k].touched)
        if len(done) > KEEP_FINISHED:
            self._drop(done[: len(done) - KEEP_FINISHED])

    def _forget(self, gid: str) -> None:
        """After a failure mid-action: memory may be ahead of the record, so rebuild from the record next time."""
        if gid in self.games:
            self._drop([gid])

    # -- committing (worker)
    def _replay_name(self, game: LiveGame) -> str:
        model = game.model_name.replace("/", "_")
        return f"human-vs-{model}-{game.id}.json" if game.seat == 0 else f"{model}-vs-human-{game.id}.json"

    def _write_replay(self, game: LiveGame) -> None:
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        path = self.replay_dir / game.replay_file
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps(game.replay(), separators=(",", ":")))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _commit(self, game: LiveGame, db_extra=None) -> None:
        """Record what the game did since the last commit, in one transaction."""
        new = [(i, f["decision"]["player"], f["decision"]["chosen"]) for i, f in enumerate(game.full_frames[game.recorded :], start=game.recorded)]
        over = game.over
        if over and not game.replay_file:
            game.replay_file = self._replay_name(game)
            self._write_replay(game)
        now = time.time()
        reveal = []
        with self.store.tx() as db:
            db.executemany("INSERT INTO choices VALUES (?, ?, ?, ?)", [(game.id, i, p, c) for i, p, c in new])
            db.execute("UPDATE games SET last_active = ?, conceded = ?, status = ?, winner = ?, end_reason = ?, replay_file = ?, finished = COALESCE(finished, ?) "
                       "WHERE id = ?", (now, int(game.conceded), "finished" if over else "active", game.winner, game.end_reason or None, game.replay_file,
                                       now if over else None, game.id))
            if over:
                db.execute("INSERT OR IGNORE INTO replays VALUES (?, ?, ?, ?)", (game.replay_file, game.owner, game.id, now))
                reveal = [r[0] for r in db.execute("SELECT url FROM issues WHERE game_id = ? AND url IS NOT NULL AND revealed = 0", (game.id,))]
                db.execute("UPDATE issues SET revealed = 1 WHERE game_id = ? AND url IS NOT NULL", (game.id,))
            if db_extra:
                db_extra(db)
        game.recorded = len(game.full_frames)
        if over and game.match is not None:
            game.match["results"] = self.store.match_results(game.match["id"], game.game_no)
        if reveal:
            self._reveal(game.reveal_comment(), reveal)

    def _mutate(self, game: LiveGame, action, db_extra=None):
        try:
            out = action()
        except LiveError:
            raise  # refused before anything changed
        except BaseException:
            self._forget(game.id)
            raise
        try:
            self._commit(game, db_extra)
        except BaseException:
            self._forget(game.id)
            raise
        return out

    # -- new games (worker)
    def _capacity(self, account: str) -> None:
        if self.store.count_active(account) >= self.per_account:
            raise HostedError(f"You already have {self.per_account} unfinished games: finish or concede one first.", 429)
        if self.store.count_active() >= self.max_games:
            raise HostedError(f"The server is full: {self.max_games} games are being played. Please try again in a few minutes.", 503)

    def _create(self, account: str, opponent: str, model: str, matchup: str, seat: int, greedy: bool, match_id: str | None, game_no: int,
                start: int | None, plan: str, prev_id: str | None) -> dict:
        self._capacity(account)
        new_match = match_id is None
        match_id = match_id or secrets.token_urlsafe(9)
        gid, seed = secrets.token_urlsafe(9), self.new_seed()
        path = self._models().get(model)
        pin = (self.pinned or {}).get(model)
        if path is None or pin is None:
            raise HostedError("that opponent is not offered on this server")
        match = {"id": match_id, "results": [] if new_match else self.store.match_results(match_id, game_no - 1)}
        try:
            game = LiveGame(gid, model, path, seat, matchup, greedy, seed, self.engine, None, match, game_no, start, plan)
        except LiveError:
            raise
        except Exception as e:  # noqa: BLE001
            raise HostedError(f"cannot play {model}: {e}") from e
        game.hide_seed, game.token, game.recorded, game.owner, game.opponent = True, self.token_for(gid), 0, account, opponent
        now = time.time()

        def rows(db):
            db.execute("INSERT INTO accounts VALUES (?, ?, ?) ON CONFLICT(email) DO UPDATE SET last_seen = excluded.last_seen", (account, now, now))
            if new_match:
                db.execute("INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (match_id, account, opponent, model, matchup, seat, int(greedy), now))
            db.execute("INSERT INTO games (id, owner, token_hash, match_id, game_no, opponent, model, checkpoint_sha256, runtime, engine, matchup, seat, greedy, "
                       "seed, start_arg, starting_player, plan, status, created, last_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                       (gid, account, self.token_hash(game.token), match_id, game_no, opponent, model, pin["sha256"], self.runtime, engine_name(self.engine),
                        matchup, seat, int(greedy), str(seed), start, game.g.starting_player, plan, now, now))
            if prev_id is not None and db.execute("UPDATE games SET next_game_id = ? WHERE id = ? AND next_game_id IS NULL", (gid, prev_id)).rowcount != 1:
                raise HostedError("the next game was already started")

        # Insert the row first (the choices reference it), then the choices and the result.
        with self.store.tx() as db:
            rows(db)
        self.games[gid] = game
        try:
            self._commit(game)
        except BaseException:
            self._forget(gid)
            raise
        self._trim()
        view = game.view()
        view["live"]["token"] = game.token
        return view

    def h_new(self, account: str, req: dict) -> dict:
        if "seed" in req:
            raise HostedError("the server picks the seed: a hosted game takes no seed")
        self._sweep()
        r = self._resolve(req)
        return self._create(account, str(req.get("opponent")), r["model"], r["matchup"], r["seat"], r["greedy"], None, 1, None, "standard", None)

    def h_next(self, account: str, gid: str, token: str | None, req: dict) -> dict:
        """The next game of the match (or a rematch). Asked twice, it answers with the same game."""
        row = self._authorize(account, gid, token)
        if row["next_game_id"]:
            nrow = self.store.game(row["next_game_id"])
            if nrow["status"] in ("expired", "unrecoverable"):
                return self._authorize(account, nrow["id"], self.token_for(nrow["id"]))  # raises with the reason
            view = self._load(nrow).view()
            view["live"]["token"] = self.token_for(nrow["id"])
            return view
        if row["status"] != "finished":
            raise HostedError("finish or concede this game first")
        results = self.store.match_results(row["match_id"], row["game_no"])
        wins = [sum(1 for _, w in results if w == p) for p in (0, 1)]
        args = (account, row["opponent"], row["model"], row["matchup"], row["seat"], bool(row["greedy"]))
        if max(wins) >= 2 or len(results) >= 3:
            return self._create(*args, None, 1, None, "standard", row["id"])
        plan = req.get("plan", "standard")
        if plan not in ("standard", "maindeck"):
            raise HostedError("plan must be 'standard' or 'maindeck'")
        last_start, last_winner = results[-1]
        seat = row["seat"]
        if last_winner is None:
            start = last_start
        elif last_winner == seat:  # the model lost: it plays first
            start = 1 - seat
        else:  # the player lost: they choose
            start = seat if req.get("play", True) else 1 - seat
        return self._create(*args, row["match_id"], row["game_no"] + 1, start, plan, row["id"])

    # -- one game (worker)
    def h_view(self, account: str, gid: str, token: str | None, since: int) -> dict:
        row = self._authorize(account, gid, token)
        game = self._load(row)
        if row["status"] == "active":
            self.store.touch_game(gid)
        return game.view(since)

    def h_choose(self, account: str, gid: str, token: str | None, req: dict) -> dict:
        row = self._authorize(account, gid, token)
        game = self._load(row)
        frame, index, since = req.get("frame"), req.get("index"), int(req.get("since", 0) or 0)
        if isinstance(frame, int) and 0 <= frame < len(game.frames) - 1:
            d = game.frames[frame]["decision"]
            if d is not None and d["player"] == game.seat and d.get("chosen") == index:
                return game.view(since)  # a retry of a choice already made
        self._mutate(game, lambda: game.choose(frame, index))
        return game.view(since)

    def h_concede(self, account: str, gid: str, token: str | None) -> dict:
        row = self._authorize(account, gid, token)
        game = self._load(row)
        if not game.conceded:
            self._mutate(game, game.concede)
        return game.view(len(game.frames) - 2)

    def h_review(self, account: str, gid: str, token: str | None) -> dict:
        row = self._authorize(account, gid, token)
        if row["status"] != "finished":
            raise HostedError("the review opens when the game is over")
        try:
            game = self._load(row)
        except HostedError:  # finished before an upgrade: its saved replay is the review
            path = self.replay_dir / (row["replay_file"] or "")
            if not row["replay_file"] or not path.is_file():
                raise
            rep = json.loads(path.read_text())
            rep["review"] = {"seat": row["seat"], "bot": 1 - row["seat"], "model": row["model"], "greedy": bool(row["greedy"]), "replay": row["replay_file"]}
            return rep
        rep = game.replay()
        rep["review"] = {"seat": game.seat, "bot": 1 - game.seat, "model": game.model_name, "greedy": game.greedy, "replay": game.replay_file}
        return rep

    def h_survey(self, account: str, gid: str, token: str | None, req: dict) -> dict:
        row = self._authorize(account, gid, token)
        game = self._load(row)
        out = game.set_survey(req)
        self._save_feedback(game)
        return out

    def h_flag(self, account: str, gid: str, token: str | None, req: dict) -> dict:
        row = self._authorize(account, gid, token)
        game = self._load(row)
        if not isinstance(req.get("category"), str):
            raise HostedError("category must be 'bot' or 'bug'")
        same = _same_flag(game.flags, req)
        if same is not None:  # a retried report: nothing new
            return {"entry": same, "issue": None, "over": game.over, "flags": game.flags, "duplicate": True}
        out = game.flag(req)
        self._save_feedback(game, out["entry"])
        if self.filer is not None and out["issue"]:
            out["reveal"] = game.reveal_comment() if game.over and not out.get("revealed") else None
        return out

    def _save_feedback(self, game: LiveGame, entry: dict | None = None) -> None:
        def extra(db):
            db.execute("UPDATE games SET pseudonym = ?, survey = ? WHERE id = ?", (game.pseudonym, json.dumps(game.survey) if game.survey else None, game.id))
            if entry is not None:
                db.execute("INSERT OR IGNORE INTO flags VALUES (?, ?, ?)", (game.id, entry["id"], json.dumps(entry)))

        if game.replay_file:
            self._write_replay(game)  # the saved replay carries the feedback
        with self.store.tx() as db:
            extra(db)

    # -- GitHub filing (request thread, only when enabled)
    def file_flag(self, gid: str, out: dict) -> dict:
        flags, entry, issue = out["flags"], out["entry"], out.get("issue")
        if out.get("duplicate"):
            return {"flags": flags, "filed": bool(entry.get("issue")), "issue": entry.get("issue"), "message": "Already saved."}
        if self.filer is None or issue is None:
            return {"flags": flags, "filed": False, "issue": None, "message": "Saved with the game for the developers."}
        if not self.filer.available():
            return {"flags": flags, "filed": False, "issue": None, "message": "Saved with the game, not filed (no GitHub access on this server)."}
        with self.store.tx() as db:
            if db.execute("SELECT COUNT(*) FROM issues WHERE game_id = ?", (gid,)).fetchone()[0] >= MAX_ISSUES_PER_GAME:
                return {"flags": flags, "filed": False, "issue": None, "message": f"Saved with the game, not filed (at most {MAX_ISSUES_PER_GAME} issues per game)."}
            claimed = db.execute("INSERT OR IGNORE INTO issues (game_id, flag_id, created) VALUES (?, ?, ?)", (gid, entry["id"], time.time())).rowcount
        if not claimed:
            return {"flags": flags, "filed": False, "issue": None, "message": "Already being filed."}
        try:
            url = self.filer.create(issue["category"], issue["title"], issue["body"])
        except FilingError as e:
            with self.store.tx() as db:
                db.execute("DELETE FROM issues WHERE game_id = ? AND flag_id = ? AND url IS NULL", (gid, entry["id"]))
            return {"flags": flags, "filed": False, "issue": None, "message": f"Saved with the game, not filed ({e})."}
        revealed = bool(out.get("revealed") or out.get("reveal"))
        with self.store.tx() as db:
            db.execute("UPDATE issues SET url = ?, revealed = ? WHERE game_id = ? AND flag_id = ?", (url, int(revealed), gid, entry["id"]))
        self.submit(self._filed_hosted, gid, entry["id"], url).result()
        if out.get("reveal"):
            self._reveal(out["reveal"], [url])
        return {"flags": flags, "filed": True, "issue": url, "message": "Filed as a GitHub issue."}

    def _filed_hosted(self, gid: str, fid: int, url: str) -> None:
        game = self.games.get(gid)
        if game is None:
            return
        for f in game.flags:
            if f["id"] == fid:
                f["issue"] = url
                with self.store.tx() as db:
                    db.execute("UPDATE flags SET entry = ? WHERE game_id = ? AND flag_id = ?", (json.dumps(f), gid, fid))
        if game.replay_file:
            self._write_replay(game)

    # -- startup (worker)
    def h_restore_all(self) -> dict:
        """Rebuild every unfinished game; returns how many were restored and which could not be."""
        self._sweep()
        ok, bad = 0, {}
        for row in self.store.all("SELECT * FROM games WHERE status = 'active'"):
            try:
                self._load(row)
                ok += 1
            except HostedError as e:
                bad[row["id"]] = str(e)
        return {"restored": ok, "unrecoverable": bad}

    # -- replays (any thread: no game is touched)
    def replay_list(self, account: str) -> list[dict]:
        from ..replay import _replay_summary

        out = []
        for r in self.store.replays(account):
            p = self.replay_dir / r["file"]
            if p.is_file():
                out.append(_replay_summary(p))
        return out

    def replay_path(self, account: str, name: str) -> pathlib.Path | None:
        """The replay file `name` if `account` owns it (unowned files are never served)."""
        if self.store.replay_owner(name) != account:
            return None
        p = (self.replay_dir / name).resolve()
        if p.parent != self.replay_dir.resolve() or p.suffix != ".json" or not p.is_file():
            return None
        return p


def _verify(game: LiveGame, record: list[tuple[int, int]], done: int) -> int:
    """Check the game's decisions from `done` on against the record; returns how many match."""
    frames = game.full_frames
    for j in range(done, len(frames)):
        d = frames[j]["decision"]
        if j >= len(record):
            raise Divergence(f"decision {j} ({d['kind']} by p{d['player']}) is not in the record")
        if (d["player"], d["chosen"]) != record[j]:
            raise Divergence(f"decision {j} ({d['kind']}): p{d['player']} chose {d['chosen']}, the record says p{record[j][0]} chose {record[j][1]}")
    return len(frames)


def _same_flag(flags: list[dict], req: dict) -> dict | None:
    what, note = str(req.get("what", "")).strip()[:200], str(req.get("note", ""))[:2000]
    for f in flags:
        if f["category"] == req.get("category") and f["what"] == what and f["note"] == note and (
                (req.get("review_frame") is not None and f.get("from") == "review" and f["replay_frame"] == req.get("review_frame"))
                or (req.get("review_frame") is None and f.get("from") != "review" and f["frame"] == req.get("frame", f["frame"]))):
            return f
    return None


def _token_key(path: pathlib.Path) -> bytes:
    """The server's key for game tokens (created once, owner-readable only)."""
    try:
        key = path.read_bytes()
        if len(key) >= 32:
            return key
    except FileNotFoundError:
        pass
    key = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
        f.flush()
        os.fsync(f.fileno())
    return key
