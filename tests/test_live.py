"""Live play against a checkpoint (mtg_ml.live): the player's view leaks
nothing hidden, choices are validated, finished games become replays."""

import json
import random
import threading
import urllib.error
import urllib.request

import pytest

from mtg_ml.agents import RandomAgent, take
from mtg_ml.backend import game_class
from mtg_ml.match import game_args
from mtg_ml.replay import snapshot, visible_events

torch = pytest.importorskip("torch")


@pytest.fixture
def models(tmp_path):
    from mtg_ml.rl.model import PolicyNet

    config = {"hidden": 16, "memory": "gru", "trunk": "entity"}
    (tmp_path / "models" / "tiny").mkdir(parents=True)
    torch.save({"config": config, "model": PolicyNet(**config).state_dict()}, tmp_path / "models" / "tiny" / "model.pt")
    (tmp_path / "models" / "notes.txt").write_text("not a checkpoint")
    return tmp_path / "models"


@pytest.fixture
def manager(models, tmp_path):
    from mtg_ml.live import LiveManager

    m = LiveManager(models, tmp_path / "replays")
    yield m
    m.close()


def play_out(manager, view, rng):
    """Answer every decision with a random option; returns the final view and all frames seen."""
    gid, frames = view["live"]["id"], list(view["frames"])
    while not view["live"]["over"]:
        frame = len(frames) - 1
        view = manager.choose(gid, {"frame": frame, "index": rng.randrange(len(frames[-1]["decision"]["options"])), "since": frame})
        frames[view["live"]["since"] :] = view["frames"]
    return view, frames


def test_player_view_hides_opponent_hand_and_private_choices():
    g = game_class()(**game_args(1), seed=4, log=True)
    agents = [RandomAgent(1), RandomAgent(2)]
    info: dict = {}
    seen = 0
    while not g.over:
        s = snapshot(g, info, viewer=0)
        opp, me = s["players"][1], s["players"][0]
        assert len(opp["hand"]) == len(g.players[1].hand)
        for c, real in zip(opp["hand"], g.players[1].hand):
            if 0 in real.known_to:
                assert c["name"] == real.name
            else:
                assert c == {"uid": c["uid"], "name": "", "hidden": True} and c["uid"] < 0
        assert [c["name"] for c in me["hand"]] == [c.name for c in g.players[0].hand]
        assert all(not line.startswith("  p1 ") for line in visible_events(g.log[seen:], 0))
        seen = len(g.log)
        take(g, agents, agents[g.decision.player].act(g))
    assert "" not in info  # face-down cards never reach the card data


def test_find_models_lists_checkpoints_by_name(models):
    from mtg_ml.live import find_models

    assert list(find_models(models)) == ["tiny/model"]


@pytest.mark.parametrize("seat", [0, 1])
def test_live_game_to_the_end_writes_a_replay(manager, seat, tmp_path):
    view = manager.new({"model": "tiny/model", "seat": seat, "seed": 3})
    assert view["meta"]["agents"][seat] == "you"
    view, frames = play_out(manager, view, random.Random(seat))
    for f in frames[:-1]:
        d = f["decision"]
        assert d["player"] == seat and 0 <= d["chosen"] < len(d["options"])
        assert "policy" not in d and "value" not in d  # the model's thoughts stay on the server
        assert all(not line.startswith(f"  p{1 - seat} ") for line in f["events"])
    assert frames[-1]["decision"] is None
    rep = json.loads((tmp_path / "replays" / view["live"]["replay"]).read_text())
    assert rep["meta"]["winner"] == view["meta"]["winner"]
    model_moves = [f["decision"] for f in rep["frames"][:-1] if f["decision"]["player"] != seat]
    assert model_moves and all("policy" in d for d in model_moves)
    human_moves = [f["decision"]["chosen"] for f in rep["frames"][:-1] if f["decision"]["player"] == seat]
    assert human_moves == [f["decision"]["chosen"] for f in frames[:-1]]


def test_bad_requests_are_refused(manager):
    from mtg_ml.live import LiveError

    with pytest.raises(LiveError):
        manager.new({"model": "../../etc/passwd"})
    with pytest.raises(LiveError):
        manager.new({"model": "tiny/model", "seat": 2})
    view = manager.new({"model": "tiny/model", "seed": 1})
    gid = view["live"]["id"]
    with pytest.raises(LiveError):
        manager.choose(gid, {"frame": 0, "index": 99})
    manager.choose(gid, {"frame": 0, "index": 0})
    with pytest.raises(LiveError, match="already"):
        manager.choose(gid, {"frame": 0, "index": 0})  # a double click
    with pytest.raises(LiveError):
        manager.choose("nope", {"frame": 0, "index": 0})


def test_oldest_game_makes_room(manager):
    manager.max_games = 2
    ids = [manager.new({"model": "tiny/model", "seed": s})["live"]["id"] for s in range(3)]
    assert sorted(manager.games) == sorted(ids[1:])


@pytest.mark.parametrize("engine", ["python", "native"])
def test_http_endpoints(manager, tmp_path, engine):
    """Each request runs on its own thread; native games must still work
    (they may only be touched by the thread that created them)."""
    from mtg_ml.replay import make_server

    if engine == "native":
        pytest.importorskip("mtg_ml_native")
    manager.engine = engine

    srv = make_server(tmp_path / "replays", port=0, live=manager)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert call("/api/live/options")[1]["models"] == ["tiny/model"]
        code, view = call("/api/live/new", {"model": "tiny/model", "seed": 2})
        assert code == 200
        gid = view["live"]["id"]
        code, again = call(f"/api/live/{gid}?since=0")
        assert code == 200 and again["frames"] == view["frames"]
        code, after = call(f"/api/live/{gid}/choose", {"frame": 0, "index": 0})
        assert code == 200 and after["frames"][0]["decision"]["chosen"] == 0
        while not after["live"]["over"]:  # play on, a request (thread) per move
            frame = after["live"]["since"] + len(after["frames"]) - 1
            code, after = call(f"/api/live/{gid}/choose", {"frame": frame, "index": 0, "since": frame})
            assert code == 200, after
        assert after["live"]["replay"]
        assert call(f"/api/live/{gid}/choose", {"frame": 0, "index": 0})[0] == 400
        assert call("/api/live/new", {"model": "missing"})[0] == 400
    finally:
        srv.shutdown()
