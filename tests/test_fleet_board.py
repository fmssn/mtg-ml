"""Fleet board renderer: pure functions on a fixture snapshot, no ssh."""
import re
import tomllib
from pathlib import Path

from tools.fleet_board.render import render, summary_line

CONFIG = tomllib.loads((Path(__file__).resolve().parents[1] / "tools/fleet_board/fleet.toml").read_text())


def _gpu(bus, util=50.0, mem=40000.0):
    return {"bus": bus, "util": util, "util_min": util, "util_max": util, "mem_mb": mem,
            "mem_total_mb": 81559.0, "power_w": 250.0, "name": "H100"}


def snapshot():
    elo = [{"games": g, "bench": 0.4, "greedy": 0.55, "elo": e, "se": 13.0}
           for g, e in ((250000, -200.0), (500000, -100.0), (750000, -80.0))]
    return {
        "collected_at": "2026-10-10T15:00:00+00:00",
        "machines": {
            "h100-private": {"status": "unreachable", "error": "timeout after 45s"},
            "h100-private2": {"status": "ok", "nproc": 64, "cpu_busy": 40.0,
                              "gpus": [_gpu("0A", 99.0), _gpu("18", 60.0, 11000), _gpu("2F", 0.0, 0)],
                              "runs": [{"dir": "/x/r9-tron-pilot", "name": "r9-tron-pilot", "live": True,
                                        "status": "running", "learner_bus": "0A", "server_buses": ["18"],
                                        "cpus": [[0, 2], [4, 31]], "target_games": 3000000,
                                        "metrics": {"games_total": 900000, "win_vs_frozen": 0.55,
                                                    "win_vs_frozen_first": 0.5, "games_per_h": 880000.0,
                                                    "evals": []}}]},
            "h100-private3": {"status": "ok", "nproc": 64, "cpu_busy": 20.0,
                              "gpus": [_gpu("0A"), _gpu("18")],
                              "runs": [{"dir": "/x/b", "name": "20261010-r9-fs7h256", "handle": "r9-base-h256",
                                        "live": True, "status": "running", "learner_bus": "0A",
                                        "server_buses": ["18"], "cpus": [[0, 31]], "hidden": "256",
                                        "lr": "7.5e-5", "target_games": 40000000,
                                        "metrics": {"games_total": 750000, "evals": elo}}]},
            "h100-private4": {"status": "ok", "nproc": 128, "cpu_busy": 5.0, "gpus": [], "runs": []},
        },
    }


def test_render_follows_artifact_rules():
    html = render(snapshot(), CONFIG)
    assert html.startswith("<title>mtg-ml Fleet Board</title>")
    assert not re.search(r"<(html|head|body|script)[\s>]", html)
    urls = set(re.findall(r"https?://[^\s\"')]+", html))
    assert all(u.startswith("https://fonts.g") for u in urls), urls


def test_unreachable_machine_does_not_fail_page():
    html = render(snapshot(), CONFIG)
    assert "Unreachable" in html and "timeout after 45s" in html
    assert "h100-private unreachable" in html  # runs table row
    assert "1 of 4" not in html and "3 of 4 machines reachable" in html
    assert "unreachable: h100-private" in summary_line(snapshot())


def test_gpu_tiles_roles_and_annotations():
    html = render(snapshot(), CONFIG)
    assert "r9-tron-pilot · learner" in html and "r9-tron-pilot · server" in html
    assert "ComfyUI" in html and "off limits" in html  # annotated bus 2F on private2
    assert "r9-base-h256 · learner" in html


def test_chart_only_for_runs_with_elo_and_reference_line():
    html = render(snapshot(), CONFIG)
    assert html.count("<svg") == 1
    assert "r7 lr075 final" in html and "L1 -80" in html
    # the pilot run has no elo: in the table, no chart, no crash
    assert "r9-tron-pilot" in html and "55% vs frozen (from 50%)" in html


def test_forced_chart_without_data_shows_placeholder():
    cfg = dict(CONFIG, charts={"include": ["r9-tron-pilot"]})
    html = render(snapshot(), cfg)
    assert "No L1 Elo data yet" in html


def test_html_escaping():
    snap = snapshot()
    snap["machines"]["h100-private3"]["runs"][0]["handle"] = "<b>x</b>"
    assert "<b>x</b>" not in render(snap, CONFIG)


# --- watchdog -------------------------------------------------------------------------------------

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from tools.fleet_board import remote, watch  # noqa: E402

WS = watch.watch_settings(CONFIG)


def _run(status="running", live=True, **kw):
    r = {"dir": "/x/r9-jund-pilot", "name": "r9-jund-pilot", "status": status, "live": live,
         "log_size": 100, "log_age_s": 5.0, "metrics": {"age_s": 5.0, "games_total": 1000}}
    r.update(kw)
    return r


def _snap(*runs, status="ok", machine="h100-private"):
    m = {"status": status, "error": "timeout"} if status != "ok" else {"status": "ok", "runs": list(runs)}
    return {"machines": {machine: m}}


def _step(state, snap, **kw):
    return watch.detect(state, snap, WS, **kw)


def test_watch_baseline_reports_nothing_then_crash_once():
    ev, st = _step({}, _snap(_run(status="failed", live=False)))
    assert ev == []  # already failed before we looked
    ev, st = _step({}, _snap(_run()))
    assert ev == []
    ev, st = _step(st, _snap(_run(status="failed", live=False, errors=["RuntimeError: CUDA error: out of memory"])))
    assert ev == ["crash h100-private r9-jund-pilot: status failed; last error: RuntimeError: CUDA error: out of memory"]
    ev, st = _step(st, _snap(_run(status="failed", live=False)))
    assert ev == []  # dedup


def test_watch_vanished_process_is_a_crash():
    _, st = _step({}, _snap(_run()))
    ev, _ = _step(st, _snap(_run(live=False)))
    assert len(ev) == 1 and ev[0].startswith("crash") and "trainer process gone" in ev[0]


def test_watch_error_lines_and_state_roundtrip():
    _, st = _step({}, _snap(_run()))
    st = json.loads(json.dumps(st))  # survives a restart
    assert watch.scan_request(st, WS)["offsets"] == {"h100-private": {"/x/r9-jund-pilot/train.log": 100}}
    ev, st = _step(st, _snap(_run(log_size=300, errors=["Traceback: ValueError: boom"])))
    assert ev == ["error h100-private r9-jund-pilot: Traceback: ValueError: boom"]
    assert watch.scan_request(st, WS)["offsets"]["h100-private"]["/x/r9-jund-pilot/train.log"] == 300
    ev, _ = _step(st, _snap(_run(log_size=300)))
    assert ev == []


def test_watch_stall_once_per_episode():
    _, st = _step({}, _snap(_run()))
    stale = _run(log_age_s=2000.0, metrics={"age_s": 1500.0})
    ev, st = _step(st, _snap(stale))
    assert ev == ["stall h100-private r9-jund-pilot: no log or metrics write for 25 min"]
    assert _step(st, _snap(stale))[0] == []
    # fresh metrics but old log is not a stall; a custom threshold works
    assert _step(st, _snap(_run(log_age_s=2000.0)))[0] == []
    _, st2 = _step({}, _snap(_run()))
    assert len(_step(st2, _snap(_run(log_age_s=400.0, metrics={"age_s": 400.0})), stall_minutes=5)[0]) == 1


def test_watch_finished_started_and_unreachable():
    _, st = _step({}, _snap(_run()))
    ev, st = _step(st, _snap(_run(status="complete", live=False)))
    assert ev == ["finished h100-private r9-jund-pilot: 1,000 games"]
    other = _run(name="r9-new", dir="/x/r9-new")
    assert _step(st, _snap(_run(status="complete", live=False), other))[0] == []
    ev, _ = _step(st, _snap(_run(status="complete", live=False), other), report_starts=True)
    assert ev == ["started h100-private r9-new: status running"]
    down = _snap(status="unreachable")
    ev, st = _step(st, down)
    assert ev == []
    assert "/x/r9-jund-pilot" in next(iter(st["runs"]))  # run state kept while down
    ev, st = _step(st, down)
    assert len(ev) == 1 and ev[0].startswith("unreachable h100-private: 2 checks")
    assert _step(st, down)[0] == []
    ev, st = _step(st, _snap(_run(status="complete", live=False)))
    assert ev == [] and st["down"] == {}


def test_remote_scan_log_reads_only_new_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("fine\n")
    size = log.stat().st_size
    pats = WS["error_patterns"]
    assert remote.scan_log(str(log), {}, pats, 3).get("errors") is None  # first look: size only
    with log.open("a") as f:
        f.write("Traceback (most recent call last):\n  File x\nValueError: boom\nProcess Killed\n"
                "CUDA error: out of memory\nNativeRulesError: a\nNativeRulesError: b\n")
    out = remote.scan_log(str(log), {str(log): size}, pats, 3)
    assert out["error_count"] == 5 and len(out["errors"]) == 3
    assert out["errors"][0] == "CUDA error: out of memory" and out["log_size"] > size
    assert remote.scan_log(str(log), {str(log): out["log_size"]}, pats, 3)["errors"] == []


def _watch_main(monkeypatch, tmp_path, snaps):
    it = iter(snaps)
    monkeypatch.setattr(watch, "collect", lambda *a, **k: next(it))
    monkeypatch.setattr(watch, "render", lambda s, c: "<title>x</title>")
    monkeypatch.setattr(watch.time, "sleep", lambda s: None)
    out = tmp_path / "fleet.html"
    return watch.main(["--out", str(out), "--interval", "0", "--max-minutes", "0"]), out


def test_watch_exit_codes_and_output(monkeypatch, tmp_path, capsys):
    code, out = _watch_main(monkeypatch, tmp_path, [_snap(_run())])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0 and lines[-1] == str(out) and len(lines) == 2 and out.exists()
    assert (tmp_path / "watch-state.json").exists() and (tmp_path / "fleet.json").exists()
    code, _ = _watch_main(monkeypatch, tmp_path, [_snap(_run(status="failed", live=False))])
    lines = capsys.readouterr().out.splitlines()
    assert code == 1 and lines[0].startswith("crash h100-private r9-jund-pilot: status failed")
    assert len(lines) == 2  # one event line plus the summary
    code, _ = _watch_main(monkeypatch, tmp_path, [_snap(_run(status="failed", live=False))])
    assert code == 0  # same event is not reported again after a restart


def test_watch_caps_event_lines(monkeypatch, tmp_path, capsys):
    runs = [_run(name=f"r{i}", dir=f"/x/r{i}") for i in range(14)]
    assert _watch_main(monkeypatch, tmp_path, [_snap(*runs)])[0] == 0
    capsys.readouterr()
    bad = [dict(r, status="failed", live=False) for r in runs]
    code, _ = _watch_main(monkeypatch, tmp_path, [_snap(*bad)])
    lines = capsys.readouterr().out.splitlines()
    assert code == 1 and len(lines) == 12 and "4 more events" in lines[-2]


def test_remote_imports_without_running_probe():
    out = subprocess.run([sys.executable, "-c", "import tools.fleet_board.remote"], capture_output=True)
    assert out.returncode == 0
