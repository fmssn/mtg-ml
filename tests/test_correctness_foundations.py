"""PR57 regressions: independent rules outcomes and information contracts, on both engines."""

from itertools import product

import pytest

from helpers import choose, new_game, pass_priority, scenario

from mtg_ml.encode import entity_features, option_previews, state_features
from mtg_ml.engine.game import _damage_amounts
from mtg_ml.rl.features import OPTION_DIM, event_hashes, featurize


def combat(power=20, blockers=None, life=9, trample=True, marked=0, deathtouch=False):
    blockers = blockers or ["Eldrazi Spawn"] * 11
    creature = "Nyxborn Hydra" if trample else "Gixian Infiltrator"
    counters = power if trample else power - 2
    g = scenario(p0={"battlefield": [(creature, {"counters": counters})]}, p1={"battlefield": blockers, "life": life}, step="declare_attackers")
    for card in g.battlefield:
        if card.controller == 1 and marked:
            card.damage = marked
    if deathtouch:
        from mtg_ml.engine.objects import TempEffect
        g.battlefield[0].temp.append(TempEffect(keywords=frozenset({"deathtouch"})))
    choose(g, f"Attack with {creature}")
    if g.decision.kind == "declare_attacker":
        choose(g, "Done declaring attackers")
    pass_priority(g, 2)
    for _ in blockers:
        choose(g, f"blocks {creature}")
    pass_priority(g, 2)
    return g


def amount(g, n):
    g.step(next(i for i, o in enumerate(g.legal_options()) if o.value == n))


def test_eleven_blocker_win_and_simultaneous_damage():
    g = combat()
    assert g.decision.kind == "assign_damage_amount"
    assert [o.value for o in g.legal_options()] == list(range(10))
    amount(g, 9)
    assert g.players[1].life == 9  # no damage before the whole assignment
    assert len(g.battlefield) == 12
    for _ in range(11):
        amount(g, 1)
    assert g.over and g.winner == 0 and g.players[1].life == 0


def test_nonlethal_divisions_include_every_blocker_and_copy():
    g = combat(life=20)
    amount(g, 0)
    amount(g, 0)
    clone = g.copy()
    assert clone.damage_allocation == g.damage_allocation
    ents, index = entity_features(g, 0, 7)
    a = g.damage_allocation
    assert "e:damage:source" in ents[index[a.attacker]]
    assert {"e:damage:committed", "e:damage:assigned:0", "e:damage:lethal:1"} <= set(ents[index[a.blockers[0]]])
    assert "e:damage:recipient" in ents[index[a.blockers[1]]]
    assert event_hashes(clone, 0) == event_hashes(g, 0)
    for engine in (g, clone):
        # Leave ten blockers alive and assign everything to blocker eleven.
        while engine.damage_allocation is not None:
            n = engine.damage_allocation.remaining if engine.damage_allocation.recipient == 10 else 0
            amount(engine, n)
        assert engine.players[1].life == 20
        assert len([c for c in engine.battlefield if c.name == "Eldrazi Spawn"]) == 10
    assert g.actions == clone.actions


def test_insufficient_trample_power_forces_zero_defender_damage():
    g = combat(power=10)
    assert g.damage_allocation.player_damage == 0
    assert g.damage_allocation.recipient == 0
    assert [o.value for o in g.legal_options()] == list(range(11))


def test_marked_damage_reduces_reserved_lethal():
    g = combat(blockers=["Tolarian Terror"] * 11, life=20, marked=4)
    assert g.damage_allocation.lethal == (1,) * 11
    assert g.legal_options()[-1].value == 9


def test_no_trample_sequential_assignment_has_no_player_share():
    g = combat(trample=False, life=20)
    assert g.damage_allocation.recipient == 0
    while g.damage_allocation is not None:
        n = g.damage_allocation.remaining if g.damage_allocation.recipient == 10 else 0
        amount(g, n)
    assert g.players[1].life == 20
    assert len([c for c in g.battlefield if c.name == "Eldrazi Spawn"]) == 10


@pytest.mark.python_only
def test_small_sequential_allocations_match_independent_integer_oracle():
    def walk(remaining, lethal, recipient, player_damage, assigned):
        if recipient == len(lethal):
            return {assigned + ((player_damage,) if trample else ())}
        out = set()
        for n in _damage_amounts(remaining, lethal, recipient, player_damage):
            out |= walk(remaining - n, lethal, recipient + 1, n if recipient == -1 else player_damage, assigned if recipient == -1 else assigned + (n,))
        return out

    for blockers in range(1, 4):
        for lethal in product(range(3), repeat=blockers):
            for power, trample in product(range(6), (False, True)):
                slots = blockers + int(trample)
                expected = {s for s in product(range(power + 1), repeat=slots) if sum(s) == power
                            and (not trample or s[-1] == 0 or all(n >= l for n, l in zip(s, lethal)))}
                assert walk(power, lethal, -1 if trample else 0, -1 if trample else 0, ()) == expected


def test_set7_keeps_hand_stack_and_all_action_pointers_on_large_board():
    g = scenario(p0={"battlefield": ["Eldrazi Spawn"] * 70 + ["Mountain"], "hand": ["Lightning Bolt"]}, p1={"battlefield": ["Island"]})
    ents, index = entity_features(g, 0, 7)
    assert len(ents) == 73
    bolt = g.players[0].hand[0]
    _, options = featurize(g, 0, features=7)
    cast = next(i for i, o in enumerate(g.legal_options()) if o.label.startswith("Cast Lightning Bolt"))
    assert OPTION_DIM + index[bolt.oid] in options[cast]
    assert len(entity_features(g, 0, 6)[0]) == 64
    choose(g, "Cast Lightning Bolt")
    ents, index = entity_features(g, 0, 7)
    assert any("e:stack" in e for e in ents)
    assert g.stack[-1].sid in index


def test_set7_previews_all_candidates_above_32():
    g = combat(power=50, life=100)
    assert len(g.legal_options()) == 40
    assert all("pv:sim:skipped" in p for p in option_previews(g, 0, 6))
    previews = option_previews(g, 0, 7)
    assert all("pv:sim:skipped" not in p and "pv:simp:skipped" not in p for p in previews)
    assert all(any(t.startswith("pv:sim:stop:") for t in p) for p in previews)


def test_set7_hidden_world_inputs_and_recurrent_logits_are_invariant():
    torch = pytest.importorskip("torch")
    from mtg_ml.rl.model import PolicyNet, collate

    torch.manual_seed(9)
    net = PolicyNet(hidden=16, trunk="entity", features=7).eval()
    inputs = []
    for metadata, hand, library in (("blue", ["Counterspell"], ["Island"] * 20), ("elves", ["Forest"], ["Forest"] * 20)):
        g = scenario(p0={"hand": ["Lightning Bolt"], "battlefield": ["Mountain"]}, p1={"hand": hand, "library": library}, deck_names=("jund", metadata),
                     registered_main=(("Mountain", "Lightning Bolt"), tuple(library)), registered_sideboards=((), tuple(hand)))
        tokens = state_features(g, 0, 7)
        assert not any(t.startswith(("opp:deck:", "self:deck:")) for t in tokens)
        inputs.append(featurize(g, 0, features=7))
    assert inputs[0] == inputs[1]
    batches = [collate([(st, opts, [])]) for st, opts in inputs]
    hidden = torch.randn(1, net.state_size)
    with torch.no_grad():
        a, b = (net(batch, hidden=hidden) for batch in batches)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_registered_lists_are_immutable_count_preserving_and_not_library_order():
    deck = ["Mountain"] * 10
    side = ["Lightning Bolt"] * 2
    g = new_game((deck, ["Island"] * 10), registered_main=(deck, ["Island"] * 10), registered_sideboards=(side, ()), seed=1)
    deck.append("Forest")
    side.append("Forest")
    tokens = state_features(g, 0, 7)
    assert "self:list:registered_main:Mountain#10" in tokens
    assert "self:list:registered_sideboard:Lightning Bolt#2" in tokens
    assert not any("Forest" in t for t in tokens)
    assert g.copy().registered_main == g.registered_main
    assert g.registered_main[0] == ("Mountain",) * 10
    with pytest.raises(AttributeError):
        g.registered_main = ((), ())


def test_sideboarded_inputs_separate_registration_and_current_main():
    from mtg_ml.match import game_args

    g = new_game(**game_args(2), seed=2)
    tokens = state_features(g, 0, 7)
    assert any(t.startswith("self:list:registered_sideboard:") for t in tokens)
    assert sorted(g.registered_main[0]) != sorted(g.current_main[0])
    for field, cards in (("registered_main", g.registered_main[0]), ("current_main", g.current_main[0])):
        assert sum(t.startswith(f"self:list:{field}:") for t in tokens) == len(cards)


def test_deathtouch_reserves_one_for_each_large_blocker():
    g = combat(blockers=["Tolarian Terror"] * 11, life=20, deathtouch=True)
    assert g.damage_allocation.lethal == (1,) * 11
    assert g.legal_options()[-1].value == 9


def test_scripted_bot_finds_large_combat_lethal_and_live_refs_preserve_amounts():
    from mtg_ml.bots import make_bot
    from mtg_ml.replay import snapshot
    from mtg_ml.live_proto import option_refs

    g = combat()
    refs = option_refs(g.decision, snapshot(g, {}, viewer=0))
    assert refs[-1]["type"] == "damage_amount" and refs[-1]["amount"] == 9
    assert refs[-1]["player"] and refs[-1]["remaining"] == 20
    assert refs[-1]["attacker"] == g.damage_allocation.attacker
    bot = make_bot(0, "jund_wildfire")
    while g.damage_allocation is not None:
        g.step(bot.act(g))
    assert g.over and g.winner == 0


@pytest.mark.python_only
def test_bot_suffix_dp_matches_exhaustive_complete_allocations():
    from types import SimpleNamespace

    from mtg_ml.bots.base import Bot
    from mtg_ml.engine.objects import DamageAllocation, Option

    weights = (12.0, 14.0, 13.0)
    bot = Bot(0)
    bot.creature_value = lambda g, b: weights[b] - 10
    for lethal in product(range(3), repeat=3):
        for power in range(6):
            a = DamageAllocation(9, (0, 1, 2), lethal, (0, 0, 0), -1, power, 1, -1)
            options = [Option(str(n), (), n) for n in _damage_amounts(power, lethal, -1, -1)]
            g = SimpleNamespace(damage_allocation=a, perm=lambda b: b, legal_options=lambda: options)
            scores = bot.damage_amount_scores(g)
            for n, score in zip((o.value for o in options), scores):
                allocations = [split for split in product(range(power + 1), repeat=3) if sum(split) == power - n
                               and (n == 0 or all(s >= l for s, l in zip(split, lethal)))]
                expected = max(sum(w for w, s, l in zip(weights, split, lethal) if s >= l) + n for split in allocations)
                assert score == expected


def test_synthetic_trace_variant_registers_the_actual_supplied_list(engine):
    from mtg_ml.trace import Scenario, new_game as trace_game

    sc = Scenario(seed=57, extra=((0, "Mental Note", 1),), max_turns=1)
    g = trace_game(sc, engine)
    assert g.registered_main == tuple(tuple(d) for d in sc.decks())
    assert g.registered_sideboards == ((), ())
    assert "self:list:registered_main:Mental Note#1" in state_features(g, 0, 7)


def test_custom_replay_lists_and_explicit_registration_are_propagated(engine):
    from mtg_ml.replay import record

    class Capture:
        def act(self, g):
            self.registration = (g.registered_main, g.registered_sideboards, g.current_main)
            return 0

    agents = [Capture(), Capture()]
    decks = (["Mountain"] * 10, ["Island"] * 10)
    main = (["Forest"] * 10, ["Island"] * 10)
    record(agents, decks=decks, engine=engine, starting_player=0, mulligans=False, max_turns=1)
    assert agents[0].registration[0] == tuple(tuple(d) for d in decks)
    assert agents[0].registration[1] == ((), ())
    record(agents, decks=decks, engine=engine, starting_player=0, mulligans=False, max_turns=1, registered_main=main)
    assert agents[0].registration[0] == tuple(tuple(d) for d in main)
    assert agents[0].registration[2] == tuple(tuple(d) for d in decks)
