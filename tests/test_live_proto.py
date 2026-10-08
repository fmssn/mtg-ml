"""The play client's protocol (mtg_ml.live_proto): structured option refs
and parsed events, and that neither leaks what the player may not see."""

import json
import random
import re
import threading
import urllib.error
import urllib.request

import pytest

from mtg_ml.live import SCRIPTED, LiveError, LiveManager
from mtg_ml.live_proto import parse_event, parse_events, visible_ids
from mtg_ml.rl.features import PUBLIC_KINDS


def native_ok():
    try:
        import mtg_ml_native  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(params=["python", "native"])
def manager(request, tmp_path):
    if request.param == "native" and not native_ok():
        pytest.skip("native engine not built")
    m = LiveManager(None, tmp_path / "replays", engine=request.param, scripted=True)
    yield m
    m.close()


def play_out(manager, view, rng, pick=None):
    """Answer every decision (randomly unless `pick` says otherwise); returns all frames seen."""
    gid, frames = view["live"]["id"], list(view["frames"])
    while not view["live"]["over"]:
        frame = len(frames) - 1
        d = frames[-1]["decision"]
        i = pick(d) if pick else None
        view = manager.choose(gid, {"frame": frame, "index": rng.randrange(len(d["options"])) if i is None else i, "since": frame})
        frames[view["live"]["since"] :] = view["frames"]
    return frames


def prefer_plays(d):
    """Play lands and spells, attack and block when possible: exercises every ref type."""
    refs = d["refs"]
    for want in ("play_land", "cast", "attack", "block", "target"):
        for i, r in enumerate(refs):
            if r["type"] == want and not r.get("done") and not r.get("none"):
                return i
    return None


def ids_in(ref: dict) -> list[tuple[str, int]]:
    out = [(k, ref[k]) for k in ("uid", "oid", "sid", "attacker", "blocker") if k in ref]
    return out + [("oid", o) for o in ref.get("to", [])]


@pytest.mark.parametrize("seat,matchup", [(0, "jund_blue"), (1, "jund_madness")])
def test_refs_point_only_at_what_the_player_sees(manager, seat, matchup):
    view = manager.new({"model": SCRIPTED, "seat": seat, "seed": 5, "matchup": matchup})
    frames = play_out(manager, view, random.Random(seat), prefer_plays)
    seen_types = set()
    for f in frames:
        d = f["decision"]
        if d is None:
            continue
        assert len(d["refs"]) == len(d["options"])
        vis = visible_ids(f["state"])
        for ref in d["refs"]:
            seen_types.add(ref["type"])
            for kind, n in ids_in(ref):
                assert n in vis["uid" if kind == "uid" else "sid" if kind == "sid" else "oid"], (d["kind"], ref)
        if d["player"] != seat:  # the model's frame: one shown option, private kinds by kind only
            assert len(d["refs"]) == 1
            if d["kind"] not in PUBLIC_KINDS:
                assert d["refs"] == [{"type": "hidden"}] and d["options"][0].startswith("(hidden")
    assert {"pass", "play_land", "cast", "attack"} <= seen_types


def test_hand_refs_name_the_card_they_play(manager):
    view = manager.new({"model": SCRIPTED, "seat": 0, "seed": 2})
    rng = random.Random(0)
    checked = 0
    gid, frames = view["live"]["id"], list(view["frames"])
    while not view["live"]["over"] and checked < 40:
        f = frames[-1]
        d, s = f["decision"], f["state"]
        if d["kind"] == "priority":
            hand = {c["uid"]: c["name"] for c in s["players"][0]["hand"]}
            mine = {c["oid"]: c for c in s["battlefield"] if c["controller"] == 0}
            for label, r in zip(d["options"], d["refs"]):
                if r.get("zone") == "hand":
                    assert hand[r["uid"]] == r["name"] and r["name"] in label
                    checked += 1
                elif r["type"] in ("activate", "mana") and "oid" in r:
                    assert mine[r["oid"]]["name"] == r["name"]
        if d["kind"] == "declare_blocker":
            for r in d["refs"]:
                assert r["blocker"] in {c["oid"] for c in s["battlefield"] if c["controller"] == 0}
                if "attacker" in r:
                    assert any(c["oid"] == r["attacker"] and c.get("attacking") for c in s["battlefield"])
        frame = len(frames) - 1
        view = manager.choose(gid, {"frame": frame, "index": prefer_plays(d) or rng.randrange(len(d["options"])), "since": frame})
        frames[view["live"]["since"] :] = view["frames"]
    assert checked


def test_model_hand_uids_never_reach_refs(manager):
    """The model's face-down hand has placeholder uids; a cast from it names
    the card (public) but never its real uid."""
    view = manager.new({"model": SCRIPTED, "seat": 0, "seed": 9})
    frames = play_out(manager, view, random.Random(1))
    for f in frames:
        hidden = {c["uid"] for c in f["state"]["players"][1]["hand"] if c.get("hidden")}
        assert all(u < 0 for u in hidden)
        d = f["decision"]
        if d and d["player"] == 1:
            assert all(r.get("uid", 0) >= 0 for r in d["refs"])


def test_parse_events():
    assert parse_event("=== Turn 3: player 1 ===") == {"t": "turn", "turn": 3, "p": 1}
    assert parse_event("-- main1") == {"t": "step", "step": "main1"}
    assert parse_event("p1 casts Lightning Bolt (normal) from hand") == {"t": "cast", "p": 1, "name": "Lightning Bolt", "mode": "normal", "zone": "hand"}
    assert parse_event("p0 activates Twisted Landscape: search for a basic land") == {"t": "activate", "p": 0, "name": "Twisted Landscape: search for a basic land"}
    assert parse_event("p0 activates Ash Barrens#12 for G") == {"t": "mana", "p": 0, "name": "Ash Barrens", "oid": 12, "color": "G"}
    assert parse_event("enters: Island#135 (p1)") == {"t": "enter", "name": "Island", "oid": 135, "p": 1}
    assert parse_event("leaves: Nihil Spellbomb#142 (p0) -> graveyard") == {"t": "leave", "name": "Nihil Spellbomb", "oid": 142, "p": 0, "to": "graveyard"}
    assert parse_event("p1 attacks with [Delver of Secrets#7, Tolarian Terror#20]") == {"t": "attack", "p": 1, "oids": [7, 20]}
    assert parse_event("p0 blocks: {31: 7, 33: 20}") == {"t": "block", "p": 0, "pairs": [[31, 7], [33, 20]]}
    assert parse_event("  p0 priority: Pass priority") is None
    assert parse_event("Delver of Secrets#169 transforms into Insectile Aberration")["t"] == "note"
    assert [e["t"] for e in parse_events(["-- draw", "  p1 priority: Play Island", "p1 plays Island"])] == ["step", "play"]


def test_scripted_bot_needs_no_checkpoints(tmp_path):
    m = LiveManager(None, tmp_path, scripted=True)
    try:
        assert m.options()["models"] == [SCRIPTED]
        with pytest.raises(LiveError):
            m.new({"model": "tiny/model"})
        view = m.new({"model": SCRIPTED, "seed": 1, "matchup": "blue_madness", "seat": 1})
        assert view["meta"]["agents"] == [f"model:{SCRIPTED}", "you"]
    finally:
        m.close()


def test_play_app_is_served_and_contained(tmp_path):
    from mtg_ml.replay import make_server

    srv = make_server(tmp_path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path):
        try:
            with urllib.request.urlopen(base + path) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, None, b""

    try:
        code, ctype, body = get("/play/")
        assert code == 200 and ctype.startswith("text/html") and b"play.js" in body
        assert get("/play/play.js")[0] == 200 and get("/play/play.css")[1].startswith("text/css")
        assert get("/play/../live.py")[0] == 404
        assert get("/play/%2e%2e/%2e%2e/mtg_ml/live.py")[0] == 404
        assert get("/play/missing.js")[0] == 404
    finally:
        srv.shutdown()


@pytest.mark.parametrize("scenario,seed", [("crowded", 1), ("floating", 2), ("instant", 3), (None, 4), (None, 7)])
def test_tap_preview_is_what_auto_pay_taps(manager, scenario, seed):
    """Each cast's `taps` preview equals the sources the client's automatic
    payment (decision["auto"]) really taps, other choices taking option 0
    as the preview does."""
    req = {"model": SCRIPTED, "seed": seed, "seat": 1, "matchup": "madness_elves"}
    if scenario:
        req["scenario"] = scenario
    view = manager.new(req)
    seat, gid, frames = view["live"]["seat"], view["live"]["id"], list(view["frames"])
    pending, checked = [], 0
    while not view["live"]["over"] and checked < 12 and len(frames) < 1500:
        f = len(frames) - 1
        d = frames[-1]["decision"]
        if d["kind"] == "priority":
            land = next((k for k, r in enumerate(d["refs"]) if r["type"] == "play_land"), 0)
            i = land or next((k for k, r in enumerate(d["refs"]) if "taps" in r), 0)
            if i and not land:
                untapped = {c["oid"] for c in frames[-1]["state"]["battlefield"] if c["controller"] == seat and not c.get("tapped")}
                mine = {c["oid"] for c in frames[-1]["state"]["battlefield"] if c["controller"] == seat}
                pending.append((f, set(d["refs"][i]["taps"]), set(d["refs"][i].get("sacs", [])), untapped, mine))
        elif d["kind"] == "pay_mana":
            assert 0 <= d["auto"] < len(d["options"])
            i = d["auto"]
        else:
            i = 0
        view = manager.choose(gid, {"frame": f, "index": i, "since": f})
        frames[view["live"]["since"] :] = view["frames"]
        while pending:  # the first frame after the cast's own decisions shows what was tapped
            start, taps, sacs, untapped, mine = pending[0]
            after = next((g for g in range(start + 1, len(frames)) if not frames[g]["decision"] or frames[g]["decision"]["player"] != seat or frames[g]["decision"]["kind"] == "priority"), None)
            if after is None:
                break
            tapped = {c["oid"] for c in frames[after]["state"]["battlefield"] if c["oid"] in untapped and c.get("tapped")}
            assert tapped == taps, (frames[start]["decision"]["options"], taps, tapped)
            gone = mine - {c["oid"] for c in frames[after]["state"]["battlefield"]}
            assert gone == sacs, (frames[start]["decision"]["options"], sacs, gone)
            pending.pop(0)
            checked += 1
    assert checked


def test_life_changes_and_combat_damage_are_logged(manager):
    """Every life change between two frames has `life` events in the later
    frame that chain from the old total to the new one; unblocked attackers
    get `hit` events; the saved replay has the same lines."""
    view = manager.new({"model": SCRIPTED, "seed": 5, "seat": 1, "matchup": "madness_elves"})
    frames = play_out(manager, view, random.Random(2), prefer_plays)
    hits = 0
    for prev, f in zip(frames, frames[1:]):
        lives = [e for e in f["actions"] if e["t"] == "life"]
        hits += sum(e["t"] == "hit" for e in f["actions"])
        for p in (0, 1):
            old, new = prev["state"]["players"][p]["life"], f["state"]["players"][p]["life"]
            mine = [e for e in lives if e["p"] == p]
            if old == new:
                assert all(e["old"] != e["new"] for e in mine)
                continue
            assert mine and mine[0]["old"] == old and mine[-1]["new"] == new
            assert all(a["new"] == b["old"] for a, b in zip(mine, mine[1:]))
    assert hits
    final = manager.view(view["live"]["id"])
    rep = json.loads((manager.replay_dir / final["live"]["replay"]).read_text())
    assert any(line.startswith("life: ") for f in rep["frames"] for line in f["events"])


def test_dev_scenarios_only_on_dev_servers(tmp_path):
    from mtg_ml.live_dev import SCENARIOS

    m = LiveManager(None, tmp_path, scripted=False)
    try:
        assert m.options()["scenarios"] == {}
        with pytest.raises(LiveError):
            m.new({"model": SCRIPTED, "scenario": "crowded"})
    finally:
        m.close()
    m = LiveManager(None, tmp_path, scripted=True)
    try:
        for name in SCENARIOS:
            view = m.new({"model": SCRIPTED, "scenario": name, "seed": 0})
            assert view["frames"][-1]["decision"]["player"] == view["live"]["seat"]
    finally:
        m.close()


def test_match_flow_concede_sideboard_rematch(manager):
    """Concede ends a game as a loss; the match goes on with sideboarded games
    (the loser chooses play or draw) until someone has two wins; then "next"
    is a rematch: a new match."""
    view = manager.new({"model": SCRIPTED, "seed": 3, "seat": 0, "matchup": "jund_blue"})
    gid = view["live"]["id"]
    assert view["meta"]["match"]["game_no"] == 1 and view["meta"]["match_game"] == 1
    view = manager.concede(gid)
    assert view["live"]["over"] and view["meta"]["winner"] == 1 and view["meta"]["end_reason"] == "concede"
    assert view["frames"][-1]["decision"] is None and view["frames"][-1]["actions"][-1] == {"t": "concede", "p": 0}
    m = view["meta"]["match"]
    assert m["wins"] == [0, 1] and m["you_choose_play"] and not m["over"]
    with pytest.raises(LiveError):
        manager.concede(gid)
    g2 = manager.next_game(gid, {"plan": "maindeck", "play": False})
    assert g2["meta"]["match"]["game_no"] == 2 and g2["meta"]["starting_player"] == 1  # chose to draw
    g2 = manager.concede(g2["live"]["id"])
    assert g2["meta"]["match"]["over"] and g2["meta"]["match"]["wins"] == [0, 2]
    again = manager.next_game(g2["live"]["id"], {})
    assert again["meta"]["match"]["game_no"] == 1 and again["meta"]["match"]["results"] == []
    with pytest.raises(LiveError):
        manager.next_game(again["live"]["id"], {})  # not finished yet


def test_sideboarded_decks(manager):
    """Game 2 decks: the model plays the plan table's plan; the player the
    standard plan or the maindeck, as chosen."""
    from mtg_ml.engine import DECKS, expand, postboard

    view = manager.new({"model": SCRIPTED, "seed": 4, "seat": 1, "matchup": "jund_blue"})
    view = manager.concede(view["live"]["id"])
    for plan in ("standard", "maindeck"):
        nxt = manager.next_game(view["live"]["id"], {"plan": plan})
        g = manager.games[nxt["live"]["id"]].g

        def cards(p, g=g):  # read on the worker: native games belong to its thread
            return sorted(c.name for c in list(g.players[p].library) + list(g.players[p].hand))

        decks, model = manager._run(cards, 1), manager._run(cards, 0)
        want = expand(postboard("mono_blue_terror", "jund_wildfire")) if plan == "standard" else expand(DECKS["mono_blue_terror"])
        assert decks == sorted(want)
        assert model == sorted(expand(postboard("jund_wildfire", "mono_blue_terror")))


def test_hand_cost_reductions(manager, monkeypatch):
    """Tolarian Terror with instants in the graveyard: the decision says how
    much less it costs, for the player's own hand only."""
    from mtg_ml import live_dev

    monkeypatch.setitem(live_dev.SCENARIOS, "terror", {
        "title": "t", "matchup": "jund_blue", "seat": 1,
        "me": {"hand": ["Tolarian Terror", "Island"], "graveyard": ["Brainstorm", "Ponder", "Counterspell"], "battlefield": ["Island"] * 3},
        "opp": {"battlefield": ["Swamp"]},
    })
    view = manager.new({"model": SCRIPTED, "scenario": "terror", "seed": 1})
    d = view["frames"][-1]["decision"]
    terror = next(c["uid"] for c in view["frames"][-1]["state"]["players"][1]["hand"] if c["name"] == "Tolarian Terror")
    assert d["player"] == 1 and {int(k): v for k, v in d["reductions"].items()} == {terror: 3}


class FakeFiler:
    """Stands in for GitHubFiler: records issues and comments, files nothing."""

    def __init__(self):
        self.issues, self.comments = [], []

    def available(self):
        return True

    def create(self, category, title, body):
        self.issues.append((category, title, body))
        return f"https://github.com/example/repo/issues/{len(self.issues)}"

    def comment(self, url, body):
        self.comments.append((url, body))


def play_until_bot_action(manager, gid, frames):
    rng = random.Random(0)
    while not any(f["decision"] and f["decision"]["player"] == 1 and f["decision"]["refs"][0]["type"] not in ("hidden", "pass") for f in frames):
        f = len(frames) - 1
        view = manager.choose(gid, {"frame": f, "index": rng.randrange(len(frames[-1]["decision"]["options"])), "since": f})
        frames[view["live"]["since"] :] = view["frames"]
    return next(i for i, f in enumerate(frames) if f["decision"] and f["decision"]["player"] == 1 and f["decision"]["refs"][0]["type"] not in ("hidden", "pass"))


def test_flags_file_issues_without_hidden_information(manager):
    """A flag is one issue (labelled by category) holding only what the player
    saw; seed + choices follow as a comment after the game; flags and the
    survey are saved in the replay's feedback block."""
    import time

    manager.filer = FakeFiler()
    view = manager.new({"model": SCRIPTED, "seed": 987654, "seat": 0, "matchup": "jund_blue"})
    gid, frames = view["live"]["id"], list(view["frames"])
    theirs = play_until_bot_action(manager, gid, frames)
    mine = next(i for i, f in enumerate(frames) if f["decision"] and f["decision"]["player"] == 0)
    with pytest.raises(LiveError):
        manager.flag(gid, {"category": "bot", "frame": mine, "what": "x"})  # not a bot play
    with pytest.raises(LiveError):
        manager.flag(gid, {"category": "praise", "frame": theirs, "what": "x"})
    with pytest.raises(LiveError):
        manager.flag(gid, {"category": "bot", "frame": theirs, "what": "  "})
    out = manager.flag(gid, {"category": "bot", "frame": theirs, "what": "Attacked into a bigger blocker", "note": "should hold back", "pseudonym": "Tester"})
    assert out["filed"] and out["issue"].endswith("/1")
    bug = manager.flag(gid, {"category": "bug", "what": "The stack hid my creature"})
    assert bug["filed"]
    (cat, title, body), (cat2, title2, _) = manager.filer.issues
    assert cat == "bot" and re.match(r"\[bot-play\] Jund Wildfire vs Mono Blue Terror, (turn \d+|mulligan): ", title) and "Attacked into a bigger blocker" in title
    assert cat2 == "bug" and title2.startswith("[bug] ")
    assert "Tester" in body and frames[theirs]["decision"]["options"][0] in body
    # nothing the player could not see: no seed, no choice list, no card only in the bot's hidden hand or library
    game = manager.games[gid]
    hidden = manager._run(lambda: {c.name for c in game.g.players[1].hand if 0 not in c.known_to} | {c.name for c in list(game.g.players[1].library)})
    visible = {c["name"] for f in frames for c in f["state"]["battlefield"]} | {c["name"] for f in frames for P in f["state"]["players"] for k in ("hand", "graveyard", "exile") for c in P[k] if c["name"]}
    visible |= {line for f in frames for line in f["events"]}  # names in the visible log
    assert str(game.seed) not in body and "--choices" not in body
    for name in hidden - visible:
        if not any(name in line for f in frames for line in f["events"]):
            assert name not in body, name
    assert not manager.filer.comments
    view = manager.concede(gid)
    for _ in range(60):  # the reveal comments are posted on a thread of their own
        if len(manager.filer.comments) == 2:
            break
        time.sleep(0.05)
    urls = sorted(u for u, _ in manager.filer.comments)
    assert urls == ["https://github.com/example/repo/issues/1", "https://github.com/example/repo/issues/2"]
    reveal = manager.filer.comments[0][1]
    assert f"seed `{game.seed}`" in reveal and "--choices" in reveal
    manager.survey(gid, {"strength": 4, "hardest": "turn 3", "pseudonym": "Tester"})
    rep = json.loads((manager.replay_dir / view["live"]["replay"]).read_text())
    fb = rep["feedback"]
    assert fb["format"] == 1 and fb["player"] == "Tester" and fb["seed"] == rep["meta"]["seed"]
    assert [f["category"] for f in fb["flags"]] == ["bot", "bug"] and fb["flags"][0]["issue"].endswith("/1")
    assert fb["choices"] == [f["decision"]["chosen"] for f in rep["frames"][:-1]]
    assert fb["survey"]["strength"] == 4


def test_flags_without_github_are_saved_and_rate_limited(manager):
    manager.filer = None
    view = manager.new({"model": SCRIPTED, "seed": 6, "seat": 0, "matchup": "jund_blue"})
    out = manager.flag(view["live"]["id"], {"category": "bug", "what": "something"})
    assert not out["filed"] and "not filed" in out["message"] and out["flags"][0]["what"] == "something"
    manager.filer = FakeFiler()
    for k in range(12):
        out = manager.flag(view["live"]["id"], {"category": "bug", "what": f"thing {k}"})
    assert len(manager.filer.issues) == 10 and not out["filed"] and "at most" in out["message"]


def test_rebuild_reproduces_the_game(manager):
    """The reveal comment's rebuild command gives the same game."""
    from mtg_ml.live_issues import rebuild

    view = manager.new({"model": SCRIPTED, "seed": 8, "seat": 1, "matchup": "jund_blue"})
    gid = view["live"]["id"]
    play_out(manager, view, random.Random(3))
    game = manager.games[gid]
    fb, log = manager._run(lambda: (game.feedback(), list(game.g.log)))
    same = manager._run(lambda: list(rebuild("jund_blue", fb["seed"], 1, 1, "standard", None, fb["choices"], manager.engine).log))
    assert same == log


@pytest.fixture
def models(tmp_path):
    torch = pytest.importorskip("torch")
    from mtg_ml.rl.model import PolicyNet

    config = {"hidden": 16, "memory": "gru", "trunk": "entity"}
    (tmp_path / "models" / "tiny").mkdir(parents=True)
    torch.save({"config": config, "model": PolicyNet(**config).state_dict()}, tmp_path / "models" / "tiny" / "model.pt")
    return tmp_path / "models"


def test_play_config_offer(tmp_path, models):
    """A normal server offers exactly the configured decks and opponents and
    refuses anything else; a deck without a matchup against them is listed
    with no opponents."""
    cfg = {"player_decks": ["jund_wildfire", "mono_blue_terror"],
           "opponents": [{"id": "delver", "label": "Delver", "model": "tiny/model", "deck": "mono_blue_terror"},
                         {"id": "ghost", "label": "Missing", "model": "nope/model", "deck": "red_madness"}]}
    m = LiveManager(models, tmp_path, config=cfg)
    try:
        opt = m.options()
        assert opt["mode"] == "play" and "models" not in opt and "scenarios" not in opt
        assert [o["id"] for o in opt["opponents"]] == ["delver"]  # a checkpoint the server lacks is not offered
        decks = {d["deck"]: d["opponents"] for d in opt["player_decks"]}
        assert decks == {"jund_wildfire": ["delver"], "mono_blue_terror": []}  # no Delver mirror matchup
        view = m.new({"deck": "jund_wildfire", "opponent": "delver", "seed": 1})
        assert view["live"]["seat"] == 0 and view["meta"]["decks"] == ["Jund Wildfire", "Mono Blue Terror"]
        for bad in ({"deck": "mono_blue_terror", "opponent": "delver"}, {"deck": "elves", "opponent": "delver"},
                    {"deck": "jund_wildfire", "opponent": "ghost"}, {"model": "tiny/model", "matchup": "jund_blue"},
                    {"model": SCRIPTED, "scenario": "crowded"}):
            with pytest.raises(LiveError):
                m.new(bad)
        gid = view["live"]["id"]
        m.concede(gid)
        assert m.next_game(gid, {"plan": "maindeck"})["meta"]["match"]["game_no"] == 2  # the match goes on in play mode
    finally:
        m.close()
