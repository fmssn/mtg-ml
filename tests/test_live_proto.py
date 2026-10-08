"""The play client's protocol (mtg_ml.live_proto): structured option refs
and parsed events, and that neither leaks what the player may not see."""

import random
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
