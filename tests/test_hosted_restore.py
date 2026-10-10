"""Restart safety of the hosted server (mtg_ml.hosted.manager): a game rebuilt
from its record (seed + every choice) after a restart is the same game as
one that never stopped, the model's recorded choices are checked as it
replays them, retries change nothing, and a game that cannot be rebuilt
faithfully says so. Two managers play in lockstep with the same seeds and
the same player choices; one of them is restarted along the way."""

import json
import random

import pytest

pytest.importorskip("torch")

from mtg_ml.hosted.manager import HostedError, HostedManager  # noqa: E402
from mtg_ml.live import LiveError, validate_offer  # noqa: E402
from tests.test_live_proto import native_ok, prefer_plays  # noqa: E402
from tests.test_play_offer import R7, r7_config, tiny7  # noqa: E402,F401 (fixture)

ME = "alice@example.com"
COMBAT = ("declare_attacker", "declare_blocker")


OPEN: list = []  # managers to close after each test (native games must be freed on their worker)


@pytest.fixture(params=["python", "native"])
def engine(request):
    if request.param == "native" and not native_ok():
        pytest.skip("native engine not built")
    yield request.param
    while OPEN:
        m = OPEN.pop()
        if not m.worker._shutdown:
            m.close()


class Side:
    """One hosted manager over a state directory, restartable."""

    def __init__(self, root, models, engine, seeds, runtime="rev-1", pinned=None):
        self.root, self.models, self.engine, self.seeds, self.runtime = root, models, engine, seeds, runtime
        self.config = r7_config(models)
        self.pinned = pinned or validate_offer(models, self.config, strict=True)[0]
        self.mgr = None
        self.start()

    def start(self, **kw):
        m = HostedManager(self.models, self.root / "replays", self.root / "state", config=self.config, pinned=kw.get("pinned", self.pinned),
                          runtime=kw.get("runtime", self.runtime), engine=self.engine, per_account=4, max_games=8)
        seeds = iter(self.seeds[m.store.one("SELECT COUNT(*) FROM games")[0]:])  # the ones not used yet
        m.new_seed = lambda: next(seeds)
        OPEN.append(m)
        self.restored = m.submit(m.h_restore_all).result()
        self.mgr = m
        return self

    def restart(self, **kw):
        self.mgr.close()
        return self.start(**kw)

    def call(self, name, *args):
        return self.mgr.submit(getattr(self.mgr, name), ME, *args).result()


def strip(rep: dict) -> dict:
    rep = json.loads(json.dumps(rep))
    rep["meta"].pop("recorded", None)
    rep.get("live", {}).pop("id", None)
    rep.get("live", {}).pop("token", None)
    rep.get("live", {}).pop("replay", None)
    if rep.get("review"):
        rep["review"].pop("replay", None)
    return rep


class Lockstep:
    """The same match on two sides: `a` never restarts, `b` does."""

    def __init__(self, a: Side, b: Side, rng: random.Random):
        self.a, self.b, self.rng = a, b, rng
        self.ids = None

    def new(self, deck, opponent):
        va = self.a.call("h_new", {"deck": deck, "opponent": opponent})
        vb = self.b.call("h_new", {"deck": deck, "opponent": opponent})
        self.ids = [(va["live"]["id"], va["live"]["token"]), (vb["live"]["id"], vb["live"]["token"])]
        self.view = va
        self.frames = list(va["frames"])
        assert strip(va) == strip(vb)

    def views(self):
        return [s.call("h_view", gid, tok, 0) for s, (gid, tok) in zip((self.a, self.b), self.ids)]

    def same(self):
        va, vb = self.views()
        assert strip(va) == strip(vb)
        ca, cb = (s.mgr.store.choices(gid) for s, (gid, _) in zip((self.a, self.b), self.ids))
        assert ca == cb  # every choice, the model's included, in order
        return va

    def step(self, index=None):
        f = len(self.frames) - 1
        d = self.frames[-1]["decision"]
        if index is None:
            i = prefer_plays(d)
            index = self.rng.randrange(len(d["options"])) if i is None else i
        req = {"frame": f, "index": index, "since": f}
        va = self.a.call("h_choose", *self.ids[0], req)
        vb = self.b.call("h_choose", *self.ids[1], req)
        assert strip(va) == strip(vb)
        self.frames[va["live"]["since"]:] = va["frames"]
        self.view = va
        return va

    def decision(self):
        return self.frames[-1]["decision"]

    def concede(self):
        va = self.a.call("h_concede", *self.ids[0])
        vb = self.b.call("h_concede", *self.ids[1])
        assert strip(va) == strip(vb)
        self.view = va

    def next(self, req):
        va = self.a.call("h_next", *self.ids[0], req)
        vb = self.b.call("h_next", *self.ids[1], req)
        assert strip(va) == strip(vb)
        self.ids = [(va["live"]["id"], va["live"]["token"]), (vb["live"]["id"], vb["live"]["token"])]
        self.view, self.frames = va, list(va["frames"])
        return va

    def play_to_end(self, restart_in_combat=False):
        restarted = False
        while not self.view["live"]["over"]:
            d = self.decision()
            if restart_in_combat and not restarted and d["kind"] in COMBAT and self.frames[-1]["state"]["turn"] >= 2:
                self.b.restart()
                assert self.b.restored["restored"] >= 1 and not self.b.restored["unrecoverable"]
                self.same()
                restarted = True
            self.step()
        return restarted

    def review_same(self):
        ra, rb = (s.call("h_review", gid, tok) for s, (gid, tok) in zip((self.a, self.b), self.ids))
        assert strip(ra) == strip(rb)
        return ra


def test_restart_mid_match_quick(tmp_path, tiny7, engine):  # noqa: F811
    """The quick version (no game played to the end): restarts at the
    mulligan, a few decisions in, between games 1 and 2 and after a game."""
    seeds = [random.Random(s).getrandbits(63) for s in range(4)]
    a, b = Side(tmp_path / "a", tiny7, engine, seeds), Side(tmp_path / "b", tiny7, engine, seeds)
    ls = Lockstep(a, b, random.Random(3))
    ls.new("elves", "r7-jund")
    b.restart()
    ls.same()
    for _ in range(12):
        if ls.view["live"]["over"]:
            break
        ls.step()
    b.restart()
    assert b.restored["restored"] == 1
    ls.same()
    ls.concede()
    b.restart()
    ls.same()
    ls.review_same()
    ls.next({"plan": "standard", "play": True})
    ls.step()
    b.restart()
    ls.same()


@pytest.mark.slow
def test_restored_games_match_uninterrupted_ones(tmp_path, tiny7, engine):  # noqa: F811
    import torch

    from mtg_ml.rl.model import PolicyNet

    torch.manual_seed(0)  # fixed weights: the same games every run
    config = {"hidden": 16, "memory": "gru", "trunk": "entity", "features": 7}
    torch.save({"config": config, "model": PolicyNet(**config).state_dict()}, tiny7 / f"{R7}.pt")
    """Restarts at the mulligan, in combat, at sideboarding (between games 1
    and 2 and between games 2 and 3) and after the match: same frames, same
    log, same model choices, same replays."""
    seeds = [random.Random(s).getrandbits(63) for s in range(40)]
    a, b = Side(tmp_path / "a", tiny7, engine, seeds), Side(tmp_path / "b", tiny7, engine, seeds)
    ls = Lockstep(a, b, random.Random(1))
    seen = set()
    for _attempt in range(8):  # a match where the player wins game 2, so there is a game 3
        ls.new("red_madness", "r8-tron-pilot")
        b.restart()  # at the mulligan
        assert b.restored == {"restored": 1, "unrecoverable": {}}
        ls.same()
        seen.add("mulligan")
        ls.step()
        ls.concede()  # game 1 to the model
        b.restart()  # sideboarding between games 1 and 2
        ls.same()
        ls.next({"plan": "maindeck"})
        if ls.play_to_end(restart_in_combat=True):
            seen.add("combat")
        if ls.view["meta"]["winner"] != ls.view["live"]["seat"]:
            continue  # the model won the match: try another
        b.restart()  # sideboarding between games 2 and 3
        assert b.restored == {"restored": 0, "unrecoverable": {}}
        ls.same()
        seen.add("sideboard 2-3")
        v = ls.next({"plan": "standard", "play": False})
        assert v["meta"]["match"]["game_no"] == 3 and v["meta"]["match"]["plan"] == "standard"
        ls.step()
        ls.concede()
        b.restart()  # after the match
        ls.same()
        ls.review_same()
        seen.add("complete")
        break
    assert seen == {"mulligan", "combat", "sideboard 2-3", "complete"}


def test_retries_change_nothing(tmp_path, tiny7, engine):  # noqa: F811
    s = Side(tmp_path, tiny7, engine, [random.Random(9).getrandbits(63) for _ in range(9)])
    v = s.call("h_new", {"deck": "elves", "opponent": "r7-elves"})
    gid, tok = v["live"]["id"], v["live"]["token"]
    f = len(v["frames"]) - 1
    first = s.call("h_choose", gid, tok, {"frame": f, "index": 0, "since": 0})
    n = len(s.mgr.store.choices(gid))
    again = s.call("h_choose", gid, tok, {"frame": f, "index": 0, "since": 0})  # a retried request
    assert again == first and len(s.mgr.store.choices(gid)) == n
    with pytest.raises(LiveError, match="already made"):
        s.call("h_choose", gid, tok, {"frame": f, "index": 1, "since": 0})  # a different answer to a taken decision
    end = s.call("h_concede", gid, tok)
    assert s.call("h_concede", gid, tok) == end
    nxt = s.call("h_next", gid, tok, {"plan": "maindeck"})
    assert s.call("h_next", gid, tok, {"plan": "standard"})["live"]["id"] == nxt["live"]["id"]  # the same next game
    s.restart()
    assert s.call("h_next", gid, tok, {})["live"]["token"] == nxt["live"]["token"]
    assert s.mgr.store.one("SELECT COUNT(*) FROM games")[0] == 2
    assert s.mgr.store.one("SELECT COUNT(*) FROM replays")[0] == 1
    assert s.mgr.store.match_results(s.mgr.store.game(gid)["match_id"], 3) == [tuple(x) for x in end["meta"]["match"]["results"]]  # recorded once
    flag = {"category": "bug", "what": "odd", "review_frame": 0}
    s.call("h_flag", gid, tok, flag)
    s.call("h_flag", gid, tok, flag)
    assert len(s.mgr.store.flags(gid)) == 1


@pytest.mark.parametrize("change", ["checkpoint", "runtime", "record"])
def test_incompatible_recovery_is_reported(tmp_path, tiny7, engine, change):  # noqa: F811
    s = Side(tmp_path, tiny7, engine, [random.Random(5).getrandbits(63) for _ in range(4)])
    v = s.call("h_new", {"deck": "tron", "opponent": "r8-tron-pilot"})
    gid, tok = v["live"]["id"], v["live"]["token"]
    s.call("h_choose", gid, tok, {"frame": len(v["frames"]) - 1, "index": 0})
    if change == "checkpoint":
        s.restart(pinned={R7: {**s.pinned[R7], "sha256": "0" * 64}})
    elif change == "runtime":
        s.restart(runtime="rev-2")
    else:  # the record no longer matches what the model plays
        with s.mgr.store.tx() as db:
            last = db.execute("SELECT MAX(seq) FROM choices WHERE game_id = ? AND player != ?", (gid, v["live"]["seat"])).fetchone()[0]
            if last is None:  # the model has not decided yet: corrupt the player's choice instead
                db.execute("UPDATE choices SET idx = 99 WHERE game_id = ?", (gid,))
            else:
                db.execute("UPDATE choices SET idx = idx + 1 WHERE game_id = ? AND seq = ?", (gid, last))
        s.restart()
    assert gid in s.restored["unrecoverable"]
    with pytest.raises(HostedError) as e:
        s.call("h_view", gid, tok, 0)
    assert e.value.status == 409 and e.value.extra["unrecoverable"] and "cannot be resumed" in str(e.value)
    assert s.mgr.store.game(gid)["status"] == "unrecoverable"
    s.call("h_new", {"deck": "tron", "opponent": "r8-tron-pilot"})  # it no longer takes a slot
