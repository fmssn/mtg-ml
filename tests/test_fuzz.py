"""Random-play fuzzing: invariants hold, every offered option is playable, and
random play reaches every card's actions."""

import random

from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR, Game, RulesError, expand

DECKS = (expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR))


def random_index(g: Game, r: random.Random) -> int:
    opts = g.legal_options()
    if opts[0].key == ("pass",) and r.random() < 0.4:
        return 0
    return r.randrange(len(opts))


def check_invariants(g: Game) -> None:
    for p in g.players:
        owned = len(p.library) + len(p.hand) + len(p.graveyard) + len(p.exile)
        owned += sum(1 for c in g.battlefield if c.owner == p.idx and not c.is_token)
        owned += sum(1 for it in g.stack if it.kind == "spell" and it.card.owner == p.idx)
        assert owned == 60, (p.idx, owned)
        assert all(v >= 0 for v in p.pool.values())
        for zone in ("library", "hand", "graveyard", "exile"):
            assert all(c.zone == zone and c.owner == p.idx for c in getattr(p, zone))
    oids = [c.oid for c in g.battlefield]
    assert len(oids) == len(set(oids))
    assert all(c.zone == "battlefield" for c in g.battlefield)
    for c in g.battlefield:
        if c.is_token:
            assert c.owner == c.controller
    for aoid in g.attackers:
        assert g.perm(aoid) is not None
    for it in g.stack:
        if it.kind == "spell":
            assert it.card.zone == "stack"


def test_random_games_keep_invariants():
    results = {}
    for seed in range(120):
        g = Game(DECKS, seed=seed, max_turns=60)
        r = random.Random(seed)
        n = 0
        while not g.over:
            assert g.legal_options()
            g.step(random_index(g, r))
            n += 1
            if n % 5 == 0:
                check_invariants(g)
            assert n < 20000
        results[seed] = (g.winner, g.end_reason)
    reasons = {reason for _, reason in results.values()}
    assert reasons <= {"life", "decking", "turn limit"}


def test_every_offered_option_is_playable():
    """No dead ends: from sampled decision points, every option can be taken
    and play continues to the next decision without a rules error."""
    for seed in range(4):
        g = Game(DECKS, seed=100 + seed, max_turns=30)
        r = random.Random(seed)
        n = 0
        while not g.over:
            if n % 9 == 0:
                for i in range(len(g.legal_options())):
                    f = g.fork()
                    try:
                        f.step(i)
                    except RulesError as e:  # pragma: no cover - reported with context
                        raise AssertionError(f"seed {seed} step {n} option {g.legal_options()[i].label}: {e}")
            g.step(random_index(g, r))
            n += 1


def test_random_play_reaches_every_card():
    used: set[str] = set()
    modes: set[tuple] = set()
    for seed in range(250):
        g = Game(DECKS, seed=seed, max_turns=40, auto_single=False)
        r = random.Random(seed)
        while not g.over:
            i = random_index(g, r)
            key = g.legal_options()[i].key
            if key[0] in ("cast", "play_land", "activate", "mana"):
                used.add(key[1])
                modes.add(key)
            g.step(i)
    expected = set(JUND_WILDFIRE) | set(MONO_BLUE_TERROR) | {"Eldrazi Spawn", "Clue", "Map"}
    # Lands with no activated ability other than mana are "used" by being played.
    assert expected - used == set(), expected - used
    for k in [
        ("cast", "Nyxborn Hydra", "hand", "bestow"),
        ("cast", "Sleep of the Dead", "graveyard", "escape"),
        ("cast", "Plunder the Trollshaws", "graveyard", "flashback"),
        ("cast", "Eviscerator's Insight", "graveyard", "flashback"),
        ("activate", "Lorien Revealed", "islandcycling {1}"),
        ("activate", "Twisted Landscape", "cycling {B}{R}{G}"),
    ]:
        assert k in modes, k
