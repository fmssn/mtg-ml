"""The hosted server's durable record: `<MTG_HOSTED_STATE>/play.sqlite3`.

One row per account, match and game; every choice of every game in order
(the player's and the model's), so an unfinished game can be rebuilt from its
seed after a restart and checked against what was played; the feedback
(flags, survey, pseudonym), the exported replays and the filed issues. Each
action is committed before the server answers it. WAL journal, synchronous
FULL; `schema_version` says which layout the file has.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import sqlite3
import threading
import time

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS accounts (
    email TEXT PRIMARY KEY,
    created REAL NOT NULL,
    last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL REFERENCES accounts(email),
    opponent TEXT NOT NULL,
    model TEXT NOT NULL,
    matchup TEXT NOT NULL,
    seat INTEGER NOT NULL,
    greedy INTEGER NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL REFERENCES accounts(email),
    token_hash TEXT NOT NULL,
    match_id TEXT NOT NULL REFERENCES matches(id),
    game_no INTEGER NOT NULL,
    opponent TEXT NOT NULL,
    model TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    runtime TEXT NOT NULL,
    engine TEXT NOT NULL,
    matchup TEXT NOT NULL,
    seat INTEGER NOT NULL,
    greedy INTEGER NOT NULL,
    seed TEXT NOT NULL,                 -- 63-bit; text keeps sqlite from rounding anything
    start_arg INTEGER,                  -- the starting player asked for (games 2, 3), NULL: by the seed
    starting_player INTEGER,
    plan TEXT NOT NULL,
    status TEXT NOT NULL,               -- active | finished | expired | unrecoverable
    problem TEXT,                       -- why it is unrecoverable
    conceded INTEGER NOT NULL DEFAULT 0,
    winner INTEGER,
    end_reason TEXT,
    pseudonym TEXT NOT NULL DEFAULT '',
    survey TEXT,
    next_game_id TEXT UNIQUE,
    replay_file TEXT,
    created REAL NOT NULL,
    last_active REAL NOT NULL,
    finished REAL,
    UNIQUE (match_id, game_no)
);
CREATE INDEX IF NOT EXISTS games_owner_status ON games(owner, status);
CREATE INDEX IF NOT EXISTS games_status ON games(status);
CREATE TABLE IF NOT EXISTS choices (
    game_id TEXT NOT NULL REFERENCES games(id),
    seq INTEGER NOT NULL,
    player INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    PRIMARY KEY (game_id, seq)
);
CREATE TABLE IF NOT EXISTS flags (
    game_id TEXT NOT NULL REFERENCES games(id),
    flag_id INTEGER NOT NULL,
    entry TEXT NOT NULL,
    PRIMARY KEY (game_id, flag_id)
);
CREATE TABLE IF NOT EXISTS replays (
    file TEXT PRIMARY KEY,
    owner TEXT NOT NULL REFERENCES accounts(email),
    game_id TEXT NOT NULL UNIQUE REFERENCES games(id),
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS issues (
    game_id TEXT NOT NULL REFERENCES games(id),
    flag_id INTEGER NOT NULL,
    url TEXT,
    revealed INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    PRIMARY KEY (game_id, flag_id)
);
"""


class Store:
    """A single connection, serialized by a lock (the game worker writes; request
    threads read replay listings)."""

    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript(SCHEMA)
            rows = self.db.execute("SELECT version FROM schema_version").fetchall()
            if not rows:
                self.db.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
            elif rows[0][0] != SCHEMA_VERSION:
                raise RuntimeError(f"{path}: schema version {rows[0][0]}, this code reads {SCHEMA_VERSION}")

    def close(self) -> None:
        with self.lock:
            self.db.close()

    @contextlib.contextmanager
    def tx(self):
        """One transaction: committed when the block ends, rolled back on an error."""
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")

    def one(self, sql: str, args=()) -> sqlite3.Row | None:
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def all(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # -- accounts
    def touch_account(self, email: str) -> None:
        now = time.time()
        with self.tx() as db:
            db.execute("INSERT INTO accounts VALUES (?, ?, ?) ON CONFLICT(email) DO UPDATE SET last_seen = excluded.last_seen", (email, now, now))

    # -- games
    def game(self, gid: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM games WHERE id = ?", (gid,))

    def choices(self, gid: str) -> list[tuple[int, int]]:
        return [(r["player"], r["idx"]) for r in self.all("SELECT player, idx FROM choices WHERE game_id = ? ORDER BY seq", (gid,))]

    def flags(self, gid: str) -> list[dict]:
        return [json.loads(r["entry"]) for r in self.all("SELECT entry FROM flags WHERE game_id = ? ORDER BY flag_id", (gid,))]

    def match_results(self, match_id: str, up_to: int) -> list[tuple[int | None, int | None]]:
        """(starting player, winner) of the match's finished games numbered up to `up_to`."""
        return [(r["starting_player"], r["winner"]) for r in self.all(
            "SELECT starting_player, winner FROM games WHERE match_id = ? AND game_no <= ? AND status = 'finished' ORDER BY game_no", (match_id, up_to))]

    def count_active(self, owner: str | None = None) -> int:
        if owner is None:
            return self.one("SELECT COUNT(*) FROM games WHERE status = 'active'")[0]
        return self.one("SELECT COUNT(*) FROM games WHERE status = 'active' AND owner = ?", (owner,))[0]

    def expire(self, before: float) -> list[str]:
        """Mark games idle since `before` expired; returns their ids."""
        with self.tx() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM games WHERE status = 'active' AND last_active < ?", (before,))]
            db.execute("UPDATE games SET status = 'expired' WHERE status = 'active' AND last_active < ?", (before,))
        return ids

    def mark_unrecoverable(self, gid: str, problem: str) -> None:
        with self.tx() as db:
            db.execute("UPDATE games SET status = 'unrecoverable', problem = ? WHERE id = ? AND status = 'active'", (problem[:500], gid))

    def touch_game(self, gid: str) -> None:
        with self.tx() as db:
            db.execute("UPDATE games SET last_active = ? WHERE id = ?", (time.time(), gid))

    # -- replays
    def replays(self, owner: str) -> list[sqlite3.Row]:
        return self.all("SELECT file, game_id, created FROM replays WHERE owner = ? ORDER BY created DESC", (owner,))

    def replay_owner(self, file: str) -> str | None:
        r = self.one("SELECT owner FROM replays WHERE file = ?", (file,))
        return None if r is None else r["owner"]
