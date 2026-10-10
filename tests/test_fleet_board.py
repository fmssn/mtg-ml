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
