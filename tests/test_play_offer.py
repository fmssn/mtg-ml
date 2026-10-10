"""The play offer (mtg_ml/play_config.toml): the r9 pilots of all six decks, the legacy
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
PILOTS = {d: f"r9-{d}-pilot" for d in ("jund", "blue", "madness", "affinity", "elves", "tron")}
PILOT_DECKS = dict(zip(PILOTS.values(), DECKS6))
TINY = "r9-jund-pilot/policy"  # the tiny test checkpoint is stored under this pinned name
SHA256 = {
    "jund": "4ffe266218ee165f12950b620a79506fd6cc0f7114031ee688b4995c171d5bc9",
    "blue": "4ff018a63ea65175f6f8fe5313cbd26d70055882c5edc720f3c0050eeb24d92d",
    "madness": "4dbb86c94df919c9b05d9f1e7d25a4b4dcda32a134575e201cda13596f96d117",
    "affinity": "3b8478ef83c802bb8070d2de5b4b6f0f1864e6799121febeb04644dddbac771b",
    "elves": "7eb8c30aa0e4cfa1b11745cdae4ba84e8a51e481e69bacfb5460d971a889bf99",
    "tron": "37e8622a0619964bd3483c4f550541fb0de98e586a8038f19e2edbfc484bd11c",
}


def test_packaged_offer_pins_the_r9_pilot_of_every_deck_and_keeps_two_references():
    cfg = load_play_config()
    assert cfg["player_decks"] == DECKS6
    ids = {o["id"] for o in cfg["opponents"]}
    assert ids == {*PILOTS.values(), "r8-jund", "delver"}  # no r7 or r8-pilot opponents are left
    assert not [k for k in cfg["models"] if k.startswith(("r7-", "r8-jund-pilot", "r8-blue-pilot", "r8-tron-pilot"))]
    for (name, deck), (k, sha) in zip(PILOT_DECKS.items(), SHA256.items()):
        o = next(o for o in cfg["opponents"] if o["id"] == name)
        assert o["model"] == f"{name}/policy" and o["deck"] == deck and o["greedy"] is True
        pin = cfg["models"][o["model"]]
        assert pin["sha256"] == sha and pin["features"] == 7 and pin["games"] == 3000320 and pin["trained_at"] == "b33b6cd"
        assert "ledger 20261010-r9-pilots" in pin["evaluation"] and k in pin["source"]
    delver = next(o for o in cfg["opponents"] if o["id"] == "delver")
    assert delver["greedy"] is False and "legacy" in delver["label"].lower() and delver["deck"] == "mono_blue_terror"
    spec = next(o for o in cfg["opponents"] if o["id"] == "r8-jund")
    assert spec["model"] == "r8-jund-blue/policy" and spec["deck"] == "jund_wildfire"
    for o in cfg["opponents"]:
        pin = cfg["models"][o["model"]]
        assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) and isinstance(pin["features"], int) and pin["source"]
        assert not o["model"].endswith("latest")
    assert len({o["id"] for o in cfg["opponents"]}) == len(cfg["opponents"])
    assert set(cfg["models"]) == {o["model"] for o in cfg["opponents"]}  # nothing pinned that is not offered


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
    assert ids[0] == "r9-jund-pilot" and "r8-jund" in ids
    for o in cfg["opponents"]:
        assert o["tested"] in TESTED_LEVELS and o["description"].strip(), o["id"]
    assert {o["id"] for o in cfg["opponents"] if o["tested"] == "well"} == {"r9-jund-pilot", "r9-blue-pilot", "r8-jund"}
    assert not [o for o in cfg["opponents"] if o["tested"] == "experimental"]
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
    keep = [f"{PILOTS[d]}/policy" for d in ("jund", "blue", "tron")]
    cfg["opponents"] = [o for o in cfg["opponents"] if o["model"] in keep]
    cfg["models"] = {k: cfg["models"][k] for k in keep}
    m = LiveManager(None, None, config=cfg, pinned=cfg["models"])
    opt = m.options()
    assert opt["recommended_matchup"]["decks"] == [{"deck": "jund_wildfire", "title": "Jund Wildfire"}, {"deck": "mono_blue_terror", "title": "Mono Blue Terror"}]
    assert "most tested" in opt["recommended_matchup"]["note"]
    rec = {d["deck"]: d["recommended"] for d in opt["player_decks"]}
    assert rec == {"jund_wildfire": ["r9-blue-pilot"], "mono_blue_terror": ["r9-jund-pilot"], "red_madness": [], "grixis_affinity": [], "elves": [], "tron": []}
    by = {o["id"]: o for o in opt["opponents"]}
    assert by["r9-blue-pilot"]["tested"] == "well" and by["r9-tron-pilot"]["tested"] == "some" and "overrates" in by["r9-tron-pilot"]["description"]
    # an older config without any of the fields still works
    old = {"player_decks": cfg["player_decks"], "opponents": [{"id": "a", "label": "A", "model": TINY, "deck": "elves"}]}
    o2 = LiveManager(None, None, config=old, pinned={TINY: {}}).options()
    assert "recommended_matchup" not in o2 and o2["opponents"][0]["tested"] == "" and o2["opponents"][0]["description"] == ""
    assert all(d["recommended"] == [] for d in o2["player_decks"])
    # a recommended matchup nobody can play here is not shown
    only = {**old, "recommended_matchup": {"decks": ["jund_wildfire", "mono_blue_terror"]}}
    assert "recommended_matchup" not in LiveManager(None, None, config=only, pinned={TINY: {}}).options()


@pytest.fixture
def tiny7(tmp_path):
    """A tiny feature-set-7 checkpoint stored under a pinned r9 name."""
    torch = pytest.importorskip("torch")
    from mtg_ml.rl.model import PolicyNet

    config = {"hidden": 16, "memory": "gru", "trunk": "entity", "features": 7}
    path = tmp_path / "models" / f"{TINY}.pt"
    path.parent.mkdir(parents=True)
    torch.save({"config": config, "model": PolicyNet(**config).state_dict()}, path)
    return tmp_path / "models"


def pilot_config(models, opponents=None) -> dict:
    cfg = copy.deepcopy(load_play_config())
    if opponents is None:
        # the six r9 pilots are served from the same tiny test checkpoint
        opponents = [{**o, "model": TINY} for o in cfg["opponents"] if o["id"] in PILOT_DECKS]
    cfg["opponents"] = opponents
    cfg["models"] = {TINY: {**cfg["models"][TINY], "sha256": sha256_file(models / f"{TINY}.pt")}}
    return cfg


def test_validate_offer(tiny7):
    cfg = pilot_config(tiny7)
    ok, problems = validate_offer(tiny7, cfg, strict=True)
    assert problems == [] and ok[TINY]["features"] == 7
    bad = copy.deepcopy(cfg)
    bad["models"][TINY]["sha256"] = "0" * 64
    assert "does not match" in validate_offer(tiny7, bad, strict=True)[1][0]
    assert validate_offer(tiny7, bad)[0] == {}  # a local server leaves it out
    bad = copy.deepcopy(cfg)
    bad["models"][TINY]["features"] = 6
    assert "feature set 7" in validate_offer(tiny7, bad, strict=True)[1][0]
    bad = copy.deepcopy(cfg)
    del bad["models"][TINY]
    assert "not pinned" in validate_offer(tiny7, bad, strict=True)[1][0]
    assert TINY in validate_offer(tiny7, bad)[0]  # unpinned is fine locally
    (tiny7 / "r9-jund-pilot" / "latest.pt").write_bytes((tiny7 / f"{TINY}.pt").read_bytes())
    moving = {**cfg, "opponents": [{"id": "x", "label": "x", "model": "r9-jund-pilot/latest", "deck": "elves"}]}
    assert "latest" in validate_offer(tiny7, moving, strict=True)[1][0]
    gone = {**cfg, "opponents": [{"id": "x", "label": "x", "model": "r9/model", "deck": "elves"}]}
    assert "no r9/model.pt" in validate_offer(tiny7, gone, strict=True)[1][0]
    assert validate_offer(tiny7, gone) == ({}, [])


@pytest.fixture(params=["python", "native"])
def offer(request, tiny7, tmp_path):
    if request.param == "native" and not native_ok():
        pytest.skip("native engine not built")
    m = LiveManager(tiny7, tmp_path / "replays", engine=request.param, config=pilot_config(tiny7), max_games=64)
    yield m
    m.close()


def test_all_36_pairs_start_and_sideboard(offer):
    """Every player deck against every pilot opponent: game 1 with the maindecks,
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
    for mine, opp, plan in (("elves", "r9-elves-pilot", None), ("tron", "r9-madness-pilot", None), ("grixis_affinity", "r9-blue-pilot", "standard")):
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
