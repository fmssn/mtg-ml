"""Feedback export contracts: disposable WAL databases, no SSH or live models."""

import importlib.util
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from mtg_ml.hosted.store import SCHEMA

SPEC = importlib.util.spec_from_file_location("play_feedback", Path(__file__).resolve().parents[1] / "tools/play_feedback.py")
feedback = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(feedback)
PRIVATE = "PRIVATE-ACCOUNT-CREDENTIAL"
REV = "a" * 40


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "play.sqlite3"
    db = sqlite3.connect(path, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    db.execute("INSERT INTO schema_version VALUES (1)")
    yield path, db
    db.close()


def add_game(database, gid="game-1", status="active", survey=None, flags=True, engine="python"):
    path, db = database
    values = dict(id=gid, owner=PRIVATE, token_hash=PRIVATE, match_id="match-1", game_no=1, opponent="r9-blue-pilot",
                  model="r9-blue-pilot/policy", checkpoint_sha256="b" * 64, runtime=REV, engine=engine,
                  matchup="jund_blue", seat=0, greedy=1, seed="7654321987654321", start_arg=None,
                  starting_player=None, plan="standard", status=status, problem=PRIVATE, conceded=0,
                  winner=None, end_reason=None, pseudonym="Tester", survey=survey, next_game_id=None,
                  replay_file=None, created=10, last_active=20, finished=None)
    # Avoid the schema's match_id/game_no unique constraint across fixture games.
    values["match_id"] = "match-" + gid
    columns = ",".join(values)
    db.execute(f"INSERT INTO games ({columns}) VALUES ({','.join('?' for _ in values)})", tuple(values.values()))
    if flags:
        flag = dict(id=1, category="bug", frame=0, replay_frame=0, action=None, kind="mulligan", turn=0,
                    step="mulligan", what="Something happened", note="Please inspect", at="2026-10-09 10:17:29",
                    issue=None, owner=PRIVATE, token=PRIVATE)
        db.execute("INSERT INTO flags VALUES (?,?,?)", (gid, 1, json.dumps(flag)))
    return feedback.read_records(path, gid)[0] if flags or survey else values


@pytest.mark.parametrize("status", ["active", "expired", "unrecoverable"])
def test_unfinished_feedback_without_replay(database, tmp_path, status):
    add_game(database, status=status)
    report = feedback.collect(database[0], tmp_path, "different-runtime")
    game = report["games"][0]
    assert game["status"] == status and len(game["flags"]) == 1
    assert not game["survey_present"] and game["survey"] is None
    assert game["replay"]["status"] == "unavailable"
    assert "Runtime mismatch" in game["context"]["reason"]
    assert PRIVATE not in json.dumps(report)
    assert "7654321987654321" not in json.dumps(report)
    assert "choices" not in game and "seed" not in game


def test_survey_only_filter_and_no_feedback(database, tmp_path):
    add_game(database, "survey", flags=False, survey=json.dumps({"strength": 2, "note": "Too easy", "email": PRIVATE}))
    add_game(database, "flags")
    add_game(database, "empty", flags=False)
    all_games = feedback.collect(database[0], tmp_path, REV)["games"]
    assert {g["id"] for g in all_games} == {"survey", "flags"}
    survey = feedback.collect(database[0], tmp_path, REV, "survey")["games"][0]
    assert survey["survey"] == {"strength": 2, "note": "Too easy"}
    assert survey["context"]["status"] == "not_requested"
    assert feedback.collect(database[0], tmp_path, REV, "missing")["games"] == []
    assert feedback.collect(database[0], tmp_path, REV, "' OR 1=1 --")["games"] == []


def test_database_snapshot_includes_wal_and_does_not_write(database, monkeypatch):
    path, writer = database
    add_game(database, survey=json.dumps({"note": "before"}))
    original_connect = sqlite3.connect
    changed = False
    statements = []

    def connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro") and kwargs["uri"] is True
        reader = original_connect(database_uri, **kwargs)
        def trace(sql):
            nonlocal changed
            statements.append(sql)
            if sql.startswith("SELECT flag_id") and not changed:
                changed = True
                writer.execute("UPDATE games SET survey=?", (json.dumps({"note": "after"}),))
                writer.execute("UPDATE flags SET entry=?", (json.dumps({"id": 1, "what": "new flag text"}),))
        reader.set_trace_callback(trace)
        return reader

    monkeypatch.setattr(feedback.sqlite3, "connect", connect)
    record = feedback.read_records(path)[0]
    assert changed
    assert not any(sql.startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "BEGIN IMMEDIATE")) for sql in statements)
    assert json.loads(record["survey"])["note"] == "before"
    assert json.loads(record["flags"][0]["entry"])["what"] == "Something happened"
    assert json.loads(writer.execute("SELECT survey FROM games").fetchone()[0])["note"] == "after"
    assert json.loads(feedback.read_records(path)[0]["survey"])["note"] == "after"


def test_unsupported_schema_and_missing_db_fail_without_creation(database, tmp_path):
    database[1].execute("UPDATE schema_version SET version=2")
    with pytest.raises(feedback.FeedbackError, match="schema"):
        feedback.read_records(database[0])
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        feedback.read_records(missing)
    assert not missing.exists()


def test_malformed_feedback_retains_game_and_valid_text(database, tmp_path):
    add_game(database, survey="{bad json with " + PRIVATE)
    database[1].execute("UPDATE flags SET entry=?", ('{"id":2,"what":"preserve me"}',))
    report = feedback.collect(database[0], tmp_path, "mismatch")
    game = report["games"][0]
    assert game["flags"][0]["id"] == 1
    assert game["flags"][0]["what"] == "preserve me"
    assert len(game["warnings"]) == 2 and game["survey_present"]
    assert PRIVATE not in json.dumps(report)


@pytest.fixture(params=["python", "native"])
def recorded(database, request):
    from mtg_ml.backend import native_available
    from mtg_ml.live import PUBLIC_KINDS
    from mtg_ml.live_issues import rebuild

    if request.param == "native" and not native_available():
        pytest.skip("native engine not built")
    record = add_game(database, engine=request.param)
    g = rebuild(record["matchup"], int(record["seed"]), 0, 1, "standard", None, [], request.param)
    # Real engine decisions; no checkpoint required. Export behavior is under test.
    record["choices"] = []
    flags = []
    for n in range(12):
        d = g.decision
        label = d.options[0].label
        flags.append(dict(id=n+1, category="bug", frame=n, replay_frame=n, kind=d.kind, turn=g.turn, step=g.step_name,
                          action=label if d.player == 1 and d.kind in PUBLIC_KINDS else None, what="inspect", note=""))
        record["choices"].append({"seq": n, "player": d.player, "idx": 0})
        g.step(0)
    return record, flags, g


def test_reconstruction_has_only_player_visible_information(recorded):
    record, flags, _ = recorded
    result = feedback.reconstruct(record, flags, REV)
    assert result["status"] == "available", result
    assert result["verified_choices"] == 12 and len(result["flags"]) == 12
    for context in result["flags"].values():
        assert "options" not in context["decision"] and "prompt" not in context["decision"]
        assert all(c["hidden"] and c["name"] == "" for c in context["state"]["players"][1]["hand"])
        assert not any(line.startswith("  p1 ") for line in context["events"])
    assert record["seed"] not in json.dumps(result)


@pytest.mark.parametrize("breakage", ["seq", "player", "idx", "action", "turn", "kind"])
def test_divergent_records_export_no_partial_context(recorded, breakage):
    record, flags, _ = recorded
    if breakage in ("seq", "player", "idx"):
        record["choices"][-1][breakage] = 999
    else:
        flags[-1][breakage] = "incorrect" if breakage != "turn" else 999
    result = feedback.reconstruct(record, flags, REV)
    assert result["status"] == "unavailable"
    assert "flags" not in result


def test_bad_frame_preserves_other_context_and_conceded_final_frame(recorded):
    record, flags, game = recorded
    flags[0]["replay_frame"] = 999999
    record["conceded"] = 1
    flags.append(dict(id=99, category="bug", replay_frame=12, kind=None, turn=game.turn, step=game.step_name, action=None))
    result = feedback.reconstruct(record, flags, REV)
    assert result["status"] == "available", result
    assert result["flags"]["1"]["status"] == "unavailable"
    assert result["flags"]["2"]["status"] == "available"
    assert result["flags"]["99"]["decision"] is None


def test_runtime_mismatch_never_reconstructs(recorded, monkeypatch):
    import mtg_ml.live_issues
    def forbidden(*args, **kwargs):
        pytest.fail("must not replay a different runtime")
    monkeypatch.setattr(mtg_ml.live_issues, "rebuild", forbidden)
    for runtime in ("different", "", "unknown", "mtg-ml 0.1.0"):
        assert feedback.reconstruct(recorded[0], recorded[1], runtime)["status"] == "unavailable"


def test_reconstruction_exception_does_not_leak(recorded, monkeypatch):
    import mtg_ml.live_issues
    def broken(*args, **kwargs):
        raise ValueError(PRIVATE)
    monkeypatch.setattr(mtg_ml.live_issues, "rebuild", broken)
    result = feedback.reconstruct(recorded[0], recorded[1], REV)
    assert result["status"] == "unavailable" and PRIVATE not in json.dumps(result)


def replay_payload(record):
    from mtg_ml.live_issues import rebuild
    from mtg_ml.replay import snapshot
    game = rebuild(record["matchup"], int(record["seed"]), 0, 1, "standard", None, [], "python")
    return {"format": 1, "meta": {"seed": int(record["seed"]), "engine": "python", "owner": PRIVATE, "token": PRIVATE},
            "cards": {}, "frames": [{"state": snapshot(game, {}), "events": [], "decision": None, "token": PRIVATE}],
            "feedback": {"flags": [], "survey": None, "email": PRIVATE}, "owner": PRIVATE}


def test_completed_replay_redaction_updates_repeated_pulls_and_cleanup(database, tmp_path):
    record = add_game(database, status="finished", survey=json.dumps({"note": "first"}))
    replay_dir = tmp_path / "source-replays"
    replay_dir.mkdir()
    (replay_dir / "saved.json").write_text(json.dumps(replay_payload(record)))
    database[1].execute("UPDATE games SET replay_file='saved.json'")
    output = tmp_path / "output"
    first = feedback.save_report(feedback.collect(database[0], replay_dir, "mismatch"), output)
    saved = output / first["games"][0]["replay"]["file"]
    assert saved.exists() and PRIVATE not in saved.read_text()
    assert json.loads(saved.read_text())["feedback"]["survey"]["note"] == "first"
    (output / "unrelated.txt").write_text("keep")
    database[1].execute("UPDATE games SET survey=?", (json.dumps({"note": "updated after finishing"}),))
    second = feedback.save_report(feedback.collect(database[0], replay_dir, "mismatch"), output)
    assert not saved.exists()
    assert len(list((output / "replays").glob("*.json"))) == 1
    assert (output / "unrelated.txt").read_text() == "keep"
    final_replay = json.loads((output / second["games"][0]["replay"]["file"]).read_text())
    assert final_replay["feedback"]["survey"]["note"] == "updated after finishing"
    assert len(final_replay["feedback"]["flags"]) == 1
    third = feedback.save_report(feedback.collect(database[0], replay_dir, "mismatch"), output)
    assert len(third["games"]) == 1 and len(third["games"][0]["flags"]) == 1
    assert "not submitted" not in (output / "report.md").read_text()
    # A narrower subsequent pull removes only previously owned replay exports.
    feedback.save_report(feedback.collect(database[0], replay_dir, REV, "absent"), output)
    assert list((output / "replays").glob("*.json")) == []
    assert (output / "unrelated.txt").exists()


@pytest.mark.parametrize("filename", ["../private.json", "missing.json", "bad.json", "symlink.json"])
def test_missing_malformed_or_unsafe_completed_replay(database, tmp_path, filename):
    record = add_game(database, status="finished")
    record["replay_file"] = filename
    root = tmp_path / "replays"
    root.mkdir()
    (root / "bad.json").write_text("not JSON " + PRIVATE)
    (tmp_path / "private.json").write_text(PRIVATE)
    (root / "symlink.json").symlink_to(tmp_path / "private.json")
    result = feedback.completed_replay(record, root, feedback.parse_feedback(record))
    assert result["status"] == "unavailable" and PRIVATE not in json.dumps(result)


def test_shell_arguments_are_quoted_and_worker_is_streamed(monkeypatch):
    directory = "/opt/path with 'quote';$(touch /tmp/not-run)"
    gid = "'; echo private; #"
    argv = feedback.ssh_command("root@server", directory, gid)
    remote_parts = shlex.split(argv[-1])
    assert remote_parts[:4] == ["cd", directory, "&&", "exec"]
    assert remote_parts[-2:] == ["--game", gid]
    assert "BatchMode=yes" in argv
    def run(command, **kwargs):
        assert command == argv
        assert "def read_records" in kwargs["input"]
        assert "shell" not in kwargs
        return SimpleNamespace(returncode=0, stdout=json.dumps({"format": 1, "exporter": "mtg-ml-play-feedback", "games": []}))
    monkeypatch.setattr(feedback.subprocess, "run", run)
    assert feedback.pull("root@server", directory, gid, 60)["games"] == []


@pytest.mark.parametrize("host", ["-oProxyCommand=evil", "user@host;echo x", "host\nother", "", "user@$(evil)"])
def test_ssh_host_options_and_injection_rejected(host):
    with pytest.raises(feedback.FeedbackError):
        feedback.ssh_command(host, "/opt/mtg-ml", None)


def test_failed_collection_preserves_existing_snapshot(tmp_path, monkeypatch, capsys):
    path = tmp_path / "feedback.json"
    path.write_text("previous contents")
    def fail(*args):
        raise subprocess.TimeoutExpired("ssh", 1)
    monkeypatch.setattr(feedback, "pull", fail)
    assert feedback.main(["pull", "--host", "host", "--out", str(tmp_path)]) == 1
    assert path.read_text() == "previous contents"
    assert "retained" in capsys.readouterr().err


def test_report_escapes_user_markdown(database, tmp_path):
    add_game(database, survey=json.dumps({"note": "<script>run()</script>\n![load](https://example.test)"}))
    report = feedback.save_report(feedback.collect(database[0], tmp_path, "mismatch"), tmp_path / "out")
    assert report["games"][0]["survey"]["note"].startswith("<script>")
    rendered = (tmp_path / "out/report.md").read_text()
    assert "<script>" not in rendered and "![load]" not in rendered
