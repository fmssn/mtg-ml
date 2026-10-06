import json

import pytest

from mtg_ml.agents import RandomAgent
from mtg_ml.bots import make_bot
from mtg_ml.replay import record


def test_record_bot_game_is_complete_and_serializable():
    rep = record([make_bot(0), make_bot(1)], seed=3, names=("bot", "bot"))
    json.dumps(rep)
    frames = rep["frames"]
    assert frames[-1]["decision"] is None
    assert all(f["decision"] is not None for f in frames[:-1])
    for f in frames[:-1]:
        d = f["decision"]
        assert 0 <= d["chosen"] < len(d["options"])
    assert rep["meta"]["winner"] in (0, 1, None)
    # Every card on screen has static info for the viewer.
    for f in frames:
        s = f["state"]
        names = [c["name"] for c in s["battlefield"]]
        for p in s["players"]:
            names += [c["name"] for z in ("hand", "graveyard", "exile") for c in p[z]]
        assert set(names) <= set(rep["cards"])


def test_uids_are_stable_across_zones():
    rep = record([RandomAgent(1), RandomAgent(2)], seed=5)
    seen = {}
    for f in rep["frames"]:
        s = f["state"]
        cards = list(s["battlefield"])
        for p in s["players"]:
            cards += p["hand"] + p["graveyard"] + p["exile"]
        uids = [c["uid"] for c in cards]
        assert len(uids) == len(set(uids))  # no card in two zones at once
        for c in cards:
            if not rep["cards"][c["name"]]["token"]:
                seen.setdefault(c["uid"], set()).add(c["name"])
    # A physical card keeps its uid; only transforming changes its name.
    assert all(len(n) <= 2 for n in seen.values())


def test_model_agent_records_policy_and_value(tmp_path):
    torch = pytest.importorskip("torch")

    from mtg_ml.rl.agent import ModelAgent
    from mtg_ml.rl.model import PolicyNet

    config = {"hidden": 16, "memory": "gru", "trunk": "entity"}
    path = tmp_path / "tiny.pt"
    torch.save({"config": config, "model": PolicyNet(**config).state_dict()}, path)
    rep = record([ModelAgent(str(path), 0, seed=1), make_bot(1)], seed=2, names=("model", "bot"))
    model_decisions = [f["decision"] for f in rep["frames"][:-1] if f["decision"]["player"] == 0]
    assert model_decisions
    for d in model_decisions:
        assert len(d["policy"]) == len(d["options"])
        assert abs(sum(d["policy"]) - 1) < 1e-2
        assert -1.5 < d["value"] < 1.5


def test_replay_listing_survives_malformed_files(tmp_path):
    from mtg_ml.replay import _replay_summary

    for name, body in [("a.json", '{"meta": null}'), ("b.json", "[]"), ("c.json", "not json"), ("d.json", '{"meta": {"seed": 4}}')]:
        (tmp_path / name).write_text(body)
    rows = {p.name: _replay_summary(p) for p in tmp_path.glob("*.json")}
    assert all(r["file"] == n for n, r in rows.items())
    assert rows["d.json"]["seed"] == 4


def test_known_top_stops_at_first_unknown_card():
    from types import SimpleNamespace as C

    from mtg_ml.replay import _known_top

    lib = [C(name="Island", known_to=set()), C(name="Mental Note", known_to={1})]
    assert _known_top(lib) == []  # a known second card must not be shown as the top
    lib[0].known_to = {0}
    assert _known_top(lib) == ["Island", "Mental Note"]


def test_model_agent_resets_between_games(tmp_path):
    torch = pytest.importorskip("torch")
    from mtg_ml.match import play_match
    from mtg_ml.rl.agent import ModelAgent
    from mtg_ml.rl.model import PolicyNet

    config = {"hidden": 16, "memory": "gru", "trunk": "entity"}
    path = tmp_path / "tiny.pt"
    torch.save({"config": config, "model": PolicyNet(**config).state_dict()}, path)
    agent = ModelAgent(str(path), 0, seed=1)
    seen = []
    act = agent.act

    def spy(game):
        # At a game's first decision the old memory is gone: either act is
        # about to switch games, or observe already did and cleared it.
        seen.append((game, game is not agent.game or agent.hidden is None))
        return act(game)

    agent.act = spy
    res = play_match([agent, make_bot(1)], seed=3)
    firsts = [fresh for i, (g, fresh) in enumerate(seen) if i == 0 or seen[i - 1][0] is not g]
    assert len(res.games) >= 2 and len(firsts) == len(res.games)
    assert all(firsts)  # every game starts from a fresh memory
