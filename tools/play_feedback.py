"""Pull hosted play feedback through SSH, without changing the running server.

The same file runs as a streamed worker in the deployed container. Keep imports
stdlib-only here; reconstruction imports that container's matching mtg_ml runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile

FORMAT = 1
GAME_FIELDS = (
    "id", "match_id", "game_no", "opponent", "model", "checkpoint_sha256", "runtime", "engine",
    "matchup", "seat", "greedy", "status", "conceded", "winner", "end_reason", "pseudonym",
    "created", "last_active", "finished",
)
FLAG_FIELDS = ("category", "frame", "replay_frame", "from", "action", "kind", "turn", "step", "what", "note", "at", "issue")
SURVEY_FIELDS = ("strength", "hardest", "note", "at")
META_FIELDS = ("seed", "engine", "agents", "decks", "match_game", "starting_player", "winner", "end_reason", "turns", "recorded")
CARD_FIELDS = ("uid", "name", "hidden", "oid", "controller", "types", "power", "toughness", "keywords", "tapped", "sick",
               "damage", "counters", "attached_to", "attacking", "blocking", "token")
INFO_FIELDS = ("types", "subtypes", "cost", "text", "power", "toughness", "token", "mana", "mana_sac")
SAFE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


class FeedbackError(ValueError):
    """An operator-facing error whose message contains no private source data."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def only(value, keys):
    if not isinstance(value, dict):
        raise FeedbackError("expected a JSON object")
    return {k: value[k] for k in keys if k in value}


def scalar_fields(value, keys):
    return {k: v for k, v in only(value, keys).items() if v is None or type(v) in (str, int, float, bool)}


def read_records(db_path, game_id=None):
    """Private working data, never serialized directly. One WAL read snapshot.

Do not use hosted.Store: its constructor creates tables and sets write pragmas.
mode=ro (not immutable=1) includes committed WAL data while the app is running.
"""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        if [r[0] for r in db.execute("SELECT version FROM schema_version")] != [1]:
            raise FeedbackError("unsupported hosted database schema (expected version 1)")
        where = "(survey IS NOT NULL OR EXISTS (SELECT 1 FROM flags WHERE game_id=games.id))"
        args = ()
        if game_id is not None:
            where += " AND id=?"
            args = (game_id,)
        columns = ",".join((*GAME_FIELDS, "survey", "seed", "start_arg", "plan", "replay_file"))
        records = []
        for row in db.execute(f"SELECT {columns} FROM games WHERE {where} ORDER BY created,id", args):
            record = dict(row)
            record["flags"] = [dict(r) for r in db.execute("SELECT flag_id,entry FROM flags WHERE game_id=? ORDER BY flag_id", (row["id"],))]
            record["choices"] = [dict(r) for r in db.execute("SELECT seq,player,idx FROM choices WHERE game_id=? ORDER BY seq", (row["id"],))]
            records.append(record)
        db.rollback()
    return records


def parse_feedback(record):
    flags, warnings = [], []
    for row in record["flags"]:
        flag = {"id": row["flag_id"]}
        try:
            raw = json.loads(row["entry"])
            flag.update(scalar_fields(raw, FLAG_FIELDS))
            if raw.get("id") != row["flag_id"]:
                raise FeedbackError("flag ID differs from database key")
        except (ValueError, TypeError):
            flag["error"] = "Malformed flag record; context unavailable."
            warnings.append(f"Flag {row['flag_id']}: malformed feedback.")
        flags.append(flag)
    survey = None
    if record["survey"] is not None:
        try:
            survey = scalar_fields(json.loads(record["survey"]), SURVEY_FIELDS)
        except (ValueError, TypeError):
            warnings.append("Malformed survey; survey exists but could not be decoded.")
    return flags, survey, warnings


def safe_state(state):
    """Known replay schema only; no future extensions containing account data."""
    out = only(state, ("turn", "active", "step"))
    out["players"] = []
    for player in state["players"]:
        p = only(player, ("life", "library", "library_top_known", "pool", "mulligans"))
        for zone in ("hand", "graveyard", "exile"):
            p[zone] = [only(c, CARD_FIELDS) for c in player[zone]]
        out["players"].append(p)
    out["battlefield"] = [only(c, CARD_FIELDS) for c in state["battlefield"]]
    out["stack"] = [only(s, ("sid", "name", "kind", "card", "controller", "targets", "target_oids", "x", "method")) for s in state["stack"]]
    return out


def completed_replay(record, replay_dir, feedback):
    """Use the file for completed games only; database feedback wins any race.

The immutable game frames are independent of later feedback rewrites, which
the server publishes via rename. Never follow a replay filename outside its dir.
"""
    if record["status"] != "finished":
        return {"status": "unavailable", "reason": "Game is not finished; no completed replay exported."}
    name = record["replay_file"]
    if not name:
        return {"status": "unavailable", "reason": "No replay is recorded."}
    root = Path(replay_dir).resolve()
    path = (root / name).resolve()
    if path.parent != root or path.suffix != ".json":
        return {"status": "unavailable", "reason": "Invalid replay path."}
    try:
        raw = json.loads(path.read_text())
        if raw.get("format") != 1 or str(raw["meta"]["seed"]) != record["seed"]:
            raise FeedbackError("replay version or seed mismatch")
        frames = []
        for frame in raw["frames"]:
            decision = frame["decision"]
            frames.append({"state": safe_state(frame["state"]), "events": frame["events"],
                           "decision": None if decision is None else only(decision, ("player", "kind", "prompt", "options", "chosen", "policy", "value"))})
        data = {"format": 1, "meta": only(raw["meta"], META_FIELDS),
                "cards": {k: only(v, INFO_FIELDS) for k, v in raw["cards"].items()}, "frames": frames,
                "feedback": {"format": 1, "player": record["pseudonym"] or "anonymous", "seat": record["seat"],
                             "seed": int(record["seed"]), "matchup": record["matchup"], "match_game": record["game_no"],
                             "choices": [r["idx"] for r in record["choices"]], "flags": feedback[0], "survey": feedback[1]}}
        return {"status": "available", "data": data}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"status": "unavailable", "reason": "Completed replay is missing, malformed, or does not match this game."}


def reconstruct(record, flags, runtime):
    """Replay saved decisions, without a model or a live manager. No writes.

Only snapshots from the human seat and public log lines leave this function.
Opponent alternatives, prompts, seeds and reconstruction choices stay private.
"""
    if not runtime or runtime in ("unknown", "unavailable") or runtime.startswith("mtg-ml ") or record["runtime"] != runtime:
        return {"status": "unavailable", "reason": "Runtime mismatch or unavailable revision; use the original release for reconstruction."}
    try:
        from mtg_ml.live import PUBLIC_KINDS
        from mtg_ml.live_issues import rebuild
        from mtg_ml.replay import snapshot, visible_events

        game = rebuild(record["matchup"], int(record["seed"]), record["seat"], record["game_no"],
                       record["plan"], record["start_arg"], [], record["engine"])
        targets = {}
        contexts = {}
        for flag in flags:
            frame = flag.get("replay_frame")
            if "error" in flag or type(frame) is not int or not 0 <= frame <= len(record["choices"]):
                contexts[str(flag["id"])] = {"status": "unavailable", "reason": "Missing or invalid replay_frame."}
            else:
                targets.setdefault(frame, []).append(flag)
        for n in range(len(record["choices"]) + 1):
            decision = None if game.over or (n == len(record["choices"]) and record["conceded"]) else game.decision
            choice = record["choices"][n] if n < len(record["choices"]) else None
            if choice is not None:
                if choice["seq"] != n or decision is None or choice["player"] != decision.player:
                    raise FeedbackError(f"Recorded decision player or sequence diverges at frame {n}.")
                if type(choice["idx"]) is not int or not 0 <= choice["idx"] < len(decision.options):
                    raise FeedbackError(f"Recorded option index is invalid at frame {n}.")
            for flag in targets.get(n, []):
                if flag.get("turn") != game.turn or flag.get("step") != game.step_name or flag.get("kind") != (decision.kind if decision else None):
                    raise FeedbackError(f"Flag {flag['id']} state differs at frame {n}.")
                action = decision.options[choice["idx"]].label if choice is not None else None
                public = decision is not None and (decision.player == record["seat"] or decision.kind in PUBLIC_KINDS)
                # Live private bot decisions are displayed as '(hidden ...)'.
                expected = action if public or action is None else f"(hidden {decision.kind.replace('_', ' ')})"
                if flag.get("from") == "review":
                    expected = action
                if flag.get("action") is not None and flag["action"] != expected:
                    raise FeedbackError(f"Flag {flag['id']} action differs at frame {n}.")
                if flag.get("category") == "bot" and (decision is None or decision.player == record["seat"]):
                    raise FeedbackError(f"Flag {flag['id']} is not a bot decision at frame {n}.")
                contexts[str(flag["id"])] = {"status": "available", "replay_frame": n,
                    "state": safe_state(snapshot(game, {}, viewer=record["seat"])),
                    "events": visible_events(list(game.log), record["seat"])[-20:],
                    "decision": None if decision is None else {"player": decision.player, "kind": decision.kind,
                                                               "action": action if public else "(hidden decision)"}}
            if choice is not None:
                game.step(choice["idx"])
        return {"status": "available", "verified_choices": len(record["choices"]), "flags": contexts}
    except FeedbackError as e:
        return {"status": "unavailable", "reason": str(e)}
    except Exception as e:  # source exceptions may contain seeds or hidden cards
        return {"status": "unavailable", "reason": f"Reconstruction failed ({type(e).__name__}); no context exported."}


def collect(db_path, replay_dir, runtime, game_id=None):
    records = read_records(db_path, game_id)
    out = {"format": FORMAT, "exporter": "mtg-ml-play-feedback", "retrieved_at": utc_now(), "runtime": runtime,
           "filter": {"game": game_id}, "games": []}
    for record in records:
        feedback = parse_feedback(record)
        flags, survey, warnings = feedback
        game = {k: record[k] for k in GAME_FIELDS}
        game.update(flags=flags, survey=survey, survey_present=record["survey"] is not None,
                    choice_count=len(record["choices"]), warnings=warnings)
        game["replay"] = completed_replay(record, replay_dir, feedback)
        game["context"] = reconstruct(record, flags, runtime) if flags else {"status": "not_requested", "reason": "No flags."}
        out["games"].append(game)
    return out


def ssh_command(host, compose_dir, game_id):
    # SSH destinations starting '-' are options, even without a local shell.
    if not host or host.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", host):
        raise FeedbackError("host must be an SSH alias or [user@]hostname, without shell syntax")
    if not compose_dir.startswith("/"):
        raise FeedbackError("compose-dir must be absolute")
    worker = ["docker", "compose", "-f", "deploy/compose.yaml", "exec", "-T", "app", "python", "-", "_worker"]
    if game_id is not None:
        worker += ["--game", game_id]
    remote = f"cd {shlex.quote(compose_dir)} && exec {shlex.join(worker)}"
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote]


def pull(host, compose_dir, game_id, timeout):
    proc = subprocess.run(ssh_command(host, compose_dir, game_id), input=Path(__file__).read_text(),
                          text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        # Don't echo arbitrary remote stderr (it can contain environment data).
        raise FeedbackError(f"SSH worker failed (exit {proc.returncode}). Check SSH access, running app, database access and schema version 1.")
    try:
        result = json.loads(proc.stdout)
        if result["format"] != FORMAT or result["exporter"] != "mtg-ml-play-feedback" or not isinstance(result["games"], list):
            raise ValueError
        return result
    except (ValueError, KeyError, TypeError):
        raise FeedbackError("worker returned an invalid feedback export") from None


def quoted(value):
    """Keep player text as quoted data, including multiline Markdown/HTML."""
    text = html.escape(str(value))
    text = re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)
    return "\n".join("> " + line for line in text.splitlines()) or "> (empty)"


def board_text(state):
    lines = [f"Turn {state['turn']} · {state['step']} · active player {state['active']}"]
    for i, player in enumerate(state["players"]):
        hand = ["(hidden)" if c.get("hidden") else c["name"] for c in player["hand"]]
        lines += [f"Player {i}: life {player['life']}, library {player['library']}, hand: {', '.join(hand) or '-'}"]
        permanents = []
        for card in state["battlefield"]:
            if card["controller"] != i:
                continue
            label = f"{card['name']}#{card['oid']}"
            if "power" in card:
                label += f" {card['power']}/{card['toughness']}"
            label += "".join(f" ({k})" for k in ("tapped", "sick", "attacking") if card.get(k))
            permanents.append(label)
        lines += ["Battlefield: " + (", ".join(permanents) or "-")]
        for zone in ("graveyard", "exile"):
            lines += [zone.title() + ": " + (", ".join(c["name"] for c in player[zone]) or "-")]
    if state["stack"]:
        lines += ["Stack: " + "; ".join(s["name"] + " → " + ", ".join(s["targets"]) for s in state["stack"])]
    return "\n".join(lines)


def markdown(report):
    games = report["games"]
    lines = ["# Hosted play feedback", "", f"Retrieved: {report['retrieved_at']}", "",
             f"{len(games)} games · {sum(len(g['flags']) for g in games)} flags · {sum(g['survey_present'] for g in games)} surveys", "",
             "Feedback is player testimony, not a verified finding. Times in `at` retain the server's original format; numeric timestamps are Unix seconds.", ""]
    if not games:
        lines += ["No feedback matches this selection.", ""]
    for game in games:
        identity = (f"{game['pseudonym'] or 'anonymous'} · {game['matchup']} · human seat {game['seat']} · "
                    f"match {game['match_id']}, game {game['game_no']} · {game['status']}")
        lines += [f"## Game {game['id']}", "", quoted(identity), "",
                  quoted(f"Model: {game['model']} · opponent: {game['opponent']} · engine: {game['engine']} · greedy: {bool(game['greedy'])}"), "",
                  quoted(f"Runtime: {game['runtime']}\nCheckpoint SHA-256: {game['checkpoint_sha256']}"), "",
                  quoted(f"Created: {game['created']} · last active: {game['last_active']} · finished: {game['finished']}"), "",
                  f"Recorded choices: {game['choice_count']}. Survey: {'present' if game['survey_present'] else 'not submitted'}.", ""]
        replay = game["replay"]
        lines += [f"Replay: `{replay['file']}`" if replay["status"] == "available" else f"Replay unavailable: {replay['reason']}", ""]
        context = game["context"]
        lines += [f"Reconstruction: {context['status']}. " + (f"Verified {context['verified_choices']} choices." if context["status"] == "available" else context.get("reason", "")), ""]
        for warning in game["warnings"]:
            lines += [quoted(warning), ""]
        for flag in game["flags"]:
            lines += [f"### Flag {flag['id']} (game {game['id']})", "",
                      quoted(f"{flag.get('category', '?')} · turn {flag.get('turn', '?')} / {flag.get('step', '?')} · "
                             f"replay frame {flag.get('replay_frame', '?')} · submitted {flag.get('at', '?')}"), "",
                      "Reported:", "", quoted(flag.get("what", "(malformed feedback)")), "",
                      "Player's note:", "", quoted(flag.get("note", "")), "",
                      "Flagged action:", "", quoted(flag.get("action") or "(no action)"), ""]
            if flag.get("issue"):
                lines += ["Issue:", "", quoted(flag["issue"]), ""]
            evidence = context.get("flags", {}).get(str(flag["id"]))
            if evidence:
                if evidence["status"] == "available":
                    lines += ["Player-visible board:", "", quoted(board_text(evidence["state"])), "",
                              "Recent public log:", "", quoted("\n".join(evidence["events"])), ""]
                else:
                    lines += [quoted(evidence["reason"]), ""]
        if game["survey"] is not None:
            lines += ["### Survey", ""]
            for key, value in game["survey"].items():
                lines += [key.title() + ":", "", quoted(value), ""]
    return "\n".join(lines)


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save_report(report, directory):
    """Content-addressed replay files; feedback.json is the snapshot commit point.

Report regeneration is deterministic from the manifest. Only obsolete replay
files listed by our previous manifest are removed; unrelated files are retained.
"""
    directory = Path(directory)
    manifest = directory / "feedback.json"
    previous = []
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get("exporter") != "mtg-ml-play-feedback" or old.get("format") != FORMAT:
            raise FeedbackError("output already contains an unrecognized feedback.json")
        previous = [g["replay"].get("file") for g in old["games"]]
    current = set()
    # Work on a copy so retries can reuse the same received payload.
    report = json.loads(json.dumps(report))
    for game in report["games"]:
        if not SAFE_ID.fullmatch(game["id"]):
            raise FeedbackError("invalid game ID in worker response")
        replay = game["replay"]
        if replay["status"] == "available":
            payload = json.dumps(replay.pop("data"), ensure_ascii=False, separators=(",", ":")).encode()
            digest = hashlib.sha256(payload).hexdigest()
            name = f"replays/{game['id']}-{digest}.json"
            atomic_write(directory / name, payload)
            replay.update(file=name, sha256=digest)
            current.add(name)
    atomic_write(directory / "report.md", markdown(report).encode())
    atomic_write(manifest, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    for name in previous:
        if name and name not in current and re.fullmatch(r"replays/[A-Za-z0-9_-]+-[0-9a-f]{64}\.json", name):
            (directory / name).unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    cmd = commands.add_parser("pull", help="retrieve a read-only snapshot from the deployed app")
    cmd.add_argument("--host", required=True, help="SSH alias or user@host")
    cmd.add_argument("--compose-dir", default="/opt/mtg-ml")
    cmd.add_argument("--game", help="only this game; default: every game with feedback")
    cmd.add_argument("--out", type=Path, default=Path(".context/feedback"))
    cmd.add_argument("--timeout", type=int, default=300, help="total SSH timeout in seconds")
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--game")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "_worker":
            # Suppress library diagnostics, which might include hidden game data.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from mtg_ml.hosted.config import runtime_revision
                result = collect(Path(os.environ.get("MTG_HOSTED_STATE", "/data/state")) / "play.sqlite3",
                                 Path(os.environ.get("MTG_HOSTED_REPLAYS", "/data/replays")), runtime_revision(), args.game)
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        else:
            report = save_report(pull(args.host, args.compose_dir, args.game, args.timeout), args.out)
            games = report["games"]
            print(f"Retrieved {len(games)} games, {sum(len(g['flags']) for g in games)} flags, {sum(g['survey_present'] for g in games)} surveys.")
            gaps = sum(g["context"]["status"] == "unavailable" or bool(g["warnings"]) or any(
                c["status"] == "unavailable" for c in g["context"].get("flags", {}).values()) for g in games)
            if gaps:
                print(f"Evidence gaps in {gaps} games; see report. Feedback was retained.")
            print(f"Report: {args.out / 'report.md'}\nJSON: {args.out / 'feedback.json'}")
        return 0
    except FeedbackError as e:
        print(f"Feedback retrieval: {e}", file=sys.stderr)
    except (OSError, sqlite3.Error, ValueError, subprocess.TimeoutExpired):
        print("Feedback retrieval failed: check SSH, container/database access, JSON and local output permissions. Existing snapshot retained if collection failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
