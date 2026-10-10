"""The play offer (mtg_ml/play_config.toml): r7 with all six decks, the legacy
r4 Delver, pinned checkpoints (live.validate_offer), and every one of the 36
ordered deck pairs as a playable best-of-three."""

import copy
import random
import re

import pytest

from mtg_ml.engine import plan_for
from mtg_ml.live import TESTED_LEVELS, LiveError, LiveManager, load_play_config, matchup_for, offer_text_problems, sha256_file, validate_offer
from mtg_ml.match import matchup_decks

from tests.test_live_proto import native_ok, play_out, prefer_plays

DECKS6 = ["jund_wildfire", "mono_blue_terror", "red_madness", "grixis_affinity", "elves", "tron"]
R7 = "r7-lr075/policy"


def test_packaged_offer_pins_r7_on_every_deck_and_keeps_legacy_delver():
    cfg = load_play_config()
    assert cfg["player_decks"] == DECKS6
    r7 = [o for o in cfg["opponents"] if o["model"] == R7]
    assert sorted(o["deck"] for o in r7) == sorted(DECKS6) and all(o["greedy"] for o in r7)
    delver = next(o for o in cfg["opponents"] if o["id"] == "delver")
    assert delver["greedy"] is False and "legacy" in delver["label"].lower() and delver["deck"] == "mono_blue_terror"
    for o in cfg["opponents"]:
        pin = cfg["models"][o["model"]]
        assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) and isinstance(pin["features"], int) and pin["source"]
        assert not o["model"].endswith("latest")
    assert cfg["models"][R7]["features"] == 7 and cfg["models"][R7]["games"] == 8353792
    assert len({o["id"] for o in cfg["opponents"]}) == len(cfg["opponents"])


def test_every_ordered_pair_is_a_matchup_with_sideboard_plans():
    """36 ordered pairs, mirrors included; each side has a plan for games 2 and 3."""
    for mine in DECKS6:
        for theirs in DECKS6:
            m = matchup_for(mine, theirs)
            assert m is not None, (mine, theirs)
            decks = matchup_decks(m[0])
            assert (decks[m[1]], decks[1 - m[1]]) == (mine, theirs)
            assert plan_for(mine, theirs).swaps and plan_for(theirs, mine).swaps


def test_default_jund_opponent_is_the_pilot_and_testing_fields_are_valid():
    cfg = load_play_config()
    ids = [o["id"] for o in cfg["opponents"]]
    assert ids[0] == "r8-jund-pilot" and {"r8-jund", "r7-jund"} <= set(ids)
    pilot = cfg["models"]["r8-jund-pilot/policy"]
    assert pilot["sha256"] == "71c758dcf4513deb3bc7021652410d7c573c366c97d8c01a4da1b5f036078379" and pilot["features"] == 7 and pilot["games"] == 3000320
    assert pilot["trained_at"] == "889798b" and "ledger 20261010-r8-pilot-matrix" in pilot["evaluation"]
    assert next(o for o in cfg["opponents"] if o["id"] == "r8-jund-pilot")["model"] == "r8-jund-pilot/policy"
    for o in cfg["opponents"]:
        assert o["tested"] in TESTED_LEVELS and o["description"].strip(), o["id"]
    assert {o["id"] for o in cfg["opponents"] if o["tested"] == "experimental"} == {"r7-madness", "r7-affinity", "r7-elves", "r7-tron"}
    assert sorted(cfg["recommended_matchup"]["decks"]) == ["jund_wildfire", "mono_blue_terror"]
    assert offer_text_problems(cfg) == []


def test_offer_text_problems():
    base = {"opponents": [{"id": "x", "tested": "well", "description": "d"}], "recommended_matchup": {"decks": ["jund_wildfire", "elves"], "note": "n"}}
    assert offer_text_problems(base) == [] and offer_text_problems({"opponents": [{"id": "x"}]}) == []  # the fields are optional
    assert "tested must be one of" in offer_text_problems({"opponents": [{"id": "x", "tested": "great"}]})[0]
    assert "description must be a string" in offer_text_problems({"opponents": [{"id": "x", "description": 3}]})[0]
    for bad in (["jund_wildfire"], ["elves", "elves"], ["elves", "nope"], "elves"):
        assert "recommended_matchup.decks" in offer_text_problems({"recommended_matchup": {"decks": bad}})[0]


def test_options_carry_testing_and_recommendation():
    cfg = copy.deepcopy(load_play_config())
    cfg["opponents"] = [o for o in cfg["opponents"] if o["model"] == R7]
    cfg["models"] = {R7: cfg["models"][R7]}
    m = LiveManager(None, None, config=cfg, pinned={R7: cfg["models"][R7]})
    opt = m.options()
    assert opt["recommended_matchup"]["decks"] == [{"deck": "jund_wildfire", "title": "Jund Wildfire"}, {"deck": "mono_blue_terror", "title": "Mono Blue Terror"}]
    assert "most tested" in opt["recommended_matchup"]["note"]
    rec = {d["deck"]: d["recommended"] for d in opt["player_decks"]}
    assert rec == {"jund_wildfire": ["r7-blue"], "mono_blue_terror": ["r7-jund"], "red_madness": [], "grixis_affinity": [], "elves": [], "tron": []}
    by = {o["id"]: o for o in opt["opponents"]}
    assert by["r7-blue"]["tested"] == "well" and by["r7-tron"]["tested"] == "experimental" and "Experimental" in by["r7-tron"]["description"]
    # an older config without any of the fields still works
    old = {"player_decks": cfg["player_decks"], "opponents": [{"id": "a", "label": "A", "model": R7, "deck": "elves"}]}
    o2 = LiveManager(None, None, config=old, pinned={R7: {}}).options()
    assert "recommended_matchup" not in o2 and o2["opponents"][0]["tested"] == "" and o2["opponents"][0]["description"] == ""
    assert all(d["recommended"] == [] for d in o2["player_decks"])
    # a recommended matchup nobody can play here is not shown
    only = {**old, "recommended_matchup": {"decks": ["jund_wildfire", "mono_blue_terror"]}}
    assert "recommended_matchup" not in LiveManager(None, None, config=only, pinned={R7: {}}).options()


@pytest.fixture
def tiny7(tmp_path):
    """A tiny feature-set-7 checkpoint stored under the pinned r7 name."""
    torch = pytest.importorskip("torch")
    from mtg_ml.rl.model import PolicyNet

    config = {"hidden": 16, "memory": "gru", "trunk": "entity", "features": 7}
    path = tmp_path / "models" / f"{R7}.pt"
    path.parent.mkdir(parents=True)
    torch.save({"config": config, "model": PolicyNet(**config).state_dict()}, path)
    return tmp_path / "models"


def r7_config(models, opponents=None) -> dict:
    cfg = copy.deepcopy(load_play_config())
    cfg["opponents"] = [o for o in cfg["opponents"] if o["model"] == R7] if opponents is None else opponents
    cfg["models"] = {R7: {**cfg["models"][R7], "sha256": sha256_file(models / f"{R7}.pt")}}
    return cfg


def test_validate_offer(tiny7):
    cfg = r7_config(tiny7)
    ok, problems = validate_offer(tiny7, cfg, strict=True)
    assert problems == [] and ok[R7]["features"] == 7
    bad = copy.deepcopy(cfg)
    bad["models"][R7]["sha256"] = "0" * 64
    assert "does not match" in validate_offer(tiny7, bad, strict=True)[1][0]
    assert validate_offer(tiny7, bad)[0] == {}  # a local server leaves it out
    bad = copy.deepcopy(cfg)
    bad["models"][R7]["features"] = 6
    assert "feature set 7" in validate_offer(tiny7, bad, strict=True)[1][0]
    bad = copy.deepcopy(cfg)
    del bad["models"][R7]
    assert "not pinned" in validate_offer(tiny7, bad, strict=True)[1][0]
    assert R7 in validate_offer(tiny7, bad)[0]  # unpinned is fine locally
    (tiny7 / "r7-lr075" / "latest.pt").write_bytes((tiny7 / f"{R7}.pt").read_bytes())
    moving = {**cfg, "opponents": [{"id": "x", "label": "x", "model": "r7-lr075/latest", "deck": "elves"}]}
    assert "latest" in validate_offer(tiny7, moving, strict=True)[1][0]
    gone = {**cfg, "opponents": [{"id": "x", "label": "x", "model": "r9/model", "deck": "elves"}]}
    assert "no r9/model.pt" in validate_offer(tiny7, gone, strict=True)[1][0]
    assert validate_offer(tiny7, gone) == ({}, [])


@pytest.fixture(params=["python", "native"])
def offer(request, tiny7, tmp_path):
    if request.param == "native" and not native_ok():
        pytest.skip("native engine not built")
    m = LiveManager(tiny7, tmp_path / "replays", engine=request.param, config=r7_config(tiny7), max_games=64)
    yield m
    m.close()


def test_all_36_pairs_start_and_sideboard(offer):
    """Every player deck against every r7 opponent: game 1 with the maindecks,
    then game 2 sideboarded (the player's standard plan or the maindeck)."""
    opt = offer.options()
    assert {d["deck"]: sorted(d["opponents"]) for d in opt["player_decks"]} == {d: sorted(o["id"] for o in opt["opponents"]) for d in DECKS6}
    for k, (mine, opp) in enumerate((d, o) for d in DECKS6 for o in opt["opponents"]):
        view = offer.new({"deck": mine, "opponent": opp["id"]})
        live = view["live"]
        decks = matchup_decks(live["matchup"])
        assert (decks[live["seat"]], decks[1 - live["seat"]]) == (mine, opp["deck"])
        offer.concede(live["id"])
        nxt = offer.next_game(live["id"], {"plan": "standard" if k % 2 else "maindeck"})
        mm = nxt["meta"]["match"]
        assert mm["game_no"] == 2 and mm["opp_swaps"] == plan_for(opp["deck"], mine).swaps
        offer.concede(nxt["live"]["id"])
        offer._run(offer._drop, list(offer.games))


@pytest.mark.slow
def test_representative_complete_games(offer):
    """A mirror, a cross matchup and a sideboarded game 2, played to the end."""
    rng = random.Random(7)
    for mine, opp, plan in (("elves", "r7-elves", None), ("tron", "r7-madness", None), ("grixis_affinity", "r7-blue", "standard")):
        view = offer.new({"deck": mine, "opponent": opp})
        if plan:
            offer.concede(view["live"]["id"])
            view = offer.next_game(view["live"]["id"], {"plan": plan})
        frames = play_out(offer, view, rng, prefer_plays)
        assert frames[-1]["decision"] is None
        game = offer.games[view["live"]["id"]]
        assert offer._run(lambda: game.over) and game.replay_file  # native games are read on their thread


def test_seeds_and_unknown_opponents_are_refused_where_offered(offer):
    with pytest.raises(LiveError):
        offer.new({"deck": "elves", "opponent": "delver"})  # not in this offer
