"""Feature set 6: simulated option previews (`pv:sim:*`, encode.sim_previews).

Runs on both engines (ENGINE_MODULES). The central guarantee: the tokens
read nothing the decider cannot see, so two games that differ only in hidden
cards (the opponent's hand, either library's order) get the same tokens for
every option."""

import random

import pytest
from helpers import choose, new_game, pay, scenario

from mtg_ml.encode import option_preview, option_previews
from mtg_ml.engine import objects as O
from mtg_ml.engine.cards import CARDS
from mtg_ml.engine.view import observe
from mtg_ml.match import match_decks


def _sim(g, label: str | None = None, i: int | None = None, prefix: str = "pv:sim:") -> list[str]:
    """The `pv:sim:` (or `pv:simp:`) tokens of the option with this label (or index)."""
    if i is None:
        i = next(k for k, o in enumerate(g.legal_options()) if o.label == label)
    return [t for t in option_preview(g, g.decision.player, i, 6) if t.startswith(prefix)]


def _simp(g, label: str) -> list[str]:
    """The "if the opponent passes" simulation of the option, prefix stripped to `pv:sim:`."""
    return ["pv:sim:" + t.removeprefix("pv:simp:") for t in _sim(g, label, prefix="pv:simp:")]


def _pass_until(g, step: str, player: int) -> None:
    for _ in range(100):
        if g.step_name == step and g.decision.player == player:
            return
        choose(g, "Pass priority")
    raise AssertionError(f"never reached {step} for p{player}")


# ---------------------------------------------------------------------------
# Hidden information
# ---------------------------------------------------------------------------


def _dealt_game(seed: int, hidden_seed: int, decks: tuple[list[str], list[str]], own_library: bool):
    """A game from the decider's (p0's) main phase: three lands each on the
    battlefield, seven cards in hand. p0's hand and both battlefields depend
    on `seed` only; p1's hand and library (the same cards, split and ordered
    differently) and, if `own_library`, p0's library order on `hidden_seed`."""
    rng = random.Random(seed)
    spec = []
    for p, deck in enumerate(decks):
        cards = sorted(deck)
        rng.shuffle(cards)
        lands = [c for c in cards if CARDS[c].is_type("Land")][:3]
        rest = list(cards)
        for c in lands:
            rest.remove(c)
        if p == 0:
            hand, library = rest[:7], rest[7:]
            if own_library:
                random.Random(hidden_seed).shuffle(library)
        else:
            random.Random(hidden_seed).shuffle(rest)
            hand, library = rest[:7], rest[7:]
        spec.append((lands, hand, library))

    def setup(g):
        for p, (lands, hand, library) in enumerate(spec):
            for c in lands:
                g.add_card(c, p, "battlefield")
            for c in hand:
                g.add_card(c, p, "hand")
            for c in library:
                g.add_card(c, p, "library")

    return new_game(([], []), seed=seed, starting_player=0, setup=setup, start_step="main1", mulligans=False)


@pytest.mark.parametrize("own_library", [False, True])
@pytest.mark.parametrize("matchup", ["jund_blue", "jund_madness", "blue_madness"])
def test_hidden_cards_never_change_sim_tokens(matchup, own_library):
    """Pairs of games that differ only in hidden cards are played with the
    same choices while the decider's observation stays the same; at every
    decision of the decider every option's previews (set 6, both `pv:sim:`
    and the "if the opponent passes" `pv:simp:`) must agree.
    The opponent takes its first option (passes); its choices may differ
    between the games without the decider seeing a difference. With the
    decider's own library order differing too, the games part at its first
    draw."""
    decks = match_decks(1, matchup)
    compared = stops = passes = 0
    for seed in range(25):
        a, b = (_dealt_game(seed, h, decks, own_library) for h in (1000 + seed, 2000 + seed))
        r = random.Random(seed)
        for _ in range(150):
            if a.over or b.over:
                break
            for g in (a, b):
                while not g.over and g.decision.player == 1:
                    g.step(0)
            if a.over or b.over or observe(a, 0) != observe(b, 0):
                break
            pa, pb = option_previews(a, 0, 6), option_previews(b, 0, 6)
            assert pa == pb, (seed, a.decision.kind, [o.label for o in a.legal_options()])
            compared += len(pa)
            stops += sum("pv:sim:stop:opponent_decision" in p for p in pa)
            passes += sum(any(t.startswith("pv:simp:") for t in p) and [t[7:] for t in p if t.startswith("pv:sim:")] != [t[8:] for t in p if t.startswith("pv:simp:")] for p in pa)
            i = r.randrange(len(a.legal_options()))
            a.step(i)
            b.step(i)
    assert compared >= 150 and stops > 0 and passes > 0


def _bolt_on_the_stack(p1_hand: list[str]):
    g = scenario(
        p0={"hand": ["Lightning Bolt", "Lightning Bolt"], "battlefield": ["Mountain", "Mountain"]},
        p1={"hand": p1_hand, "battlefield": ["Island", "Island", "Delver of Secrets"]},
        auto_single=True,
    )
    choose(g, "Cast Lightning Bolt")
    choose(g, "Target Delver of Secrets")
    pay(g)
    assert g.decision.player == 0 and g.decision.kind == O.PRIORITY and len(g.stack) == 1
    return g


def test_an_opponent_without_a_response_is_not_revealed():
    """Passing with a spell on the stack: an opponent holding Counterspell
    gets a decision, one holding a land passes automatically in the real
    game. The simulation stops at the opponent's priority either way, so the
    tokens do not say which."""
    counter, land = _bolt_on_the_stack(["Counterspell"]), _bolt_on_the_stack(["Island"])
    sims = [_sim(g, "Pass priority") for g in (counter, land)]
    assert sims[0] == sims[1]
    assert sims[0][:2] == ["pv:sim:stop:opponent_decision", "pv:sim:next:opponent:priority"]
    # "if the opponent passes": the Bolt resolves in both, whatever the hand holds
    simps = [_simp(g, "Pass priority") for g in (counter, land)]
    assert simps[0] == simps[1] and "pv:sim:opponent:creatures_lost>=1" in simps[0]
    assert option_preview(counter, 0, 0, 6) == option_preview(land, 0, 0, 6)
    # the real games do differ: without a response the Bolt resolves at once
    for g in (counter, land):
        choose(g, "Pass priority")
    assert counter.decision.player == 1 and len(counter.stack) == 1
    assert land.decision.player == 0 and not land.stack


def test_a_draw_stops_the_simulation():
    """p1 passing in p0's upkeep lets p0 draw: hidden information, so the
    option gets only the stop."""
    g = scenario(p0={"battlefield": ["Mountain"]}, p1={"battlefield": ["Island"]}, active=0, step="upkeep")
    _pass_until(g, "upkeep", 1)
    assert _sim(g, "Pass priority") == ["pv:sim:stop:hidden_info"]


# ---------------------------------------------------------------------------
# What the tokens say
# ---------------------------------------------------------------------------


def _opponent_bolts(target: str):
    """p1 casts Lightning Bolt at `target` in its main phase and passes: p0 to decide."""
    g = scenario(p0={"battlefield": ["Island", "Delver of Secrets"]}, p1={"hand": ["Lightning Bolt"], "battlefield": ["Mountain"]}, active=1)
    choose(g, "Cast Lightning Bolt")
    choose(g, target)
    pay(g)
    choose(g, "Pass priority")
    assert g.decision.player == 0 and len(g.stack) == 1
    return g


def test_letting_a_removal_spell_resolve_shows_the_creature_dying():
    sim = _sim(_opponent_bolts("Target Delver of Secrets"), "Pass priority")
    assert sim[:2] == ["pv:sim:stop:opponent_decision", "pv:sim:next:opponent:priority"]
    assert {
        "pv:sim:self:creatures_lost>=1",
        "pv:sim:self:power_lost>=1",
        "pv:sim:self:lost_power_tier:1",
        "pv:sim:self:perms_lost>=1",
        "pv:sim:self:graveyard+>=1",
        "pv:sim:opponent:graveyard+>=1",
        "pv:sim:stack->=1",
    } <= set(sim)
    assert not any(t.startswith(("pv:sim:opponent:creatures_lost", "pv:sim:self:life")) for t in sim)


def test_same_option_different_outcomes_get_different_tokens():
    """Passing with the opponent's Bolt aimed at the decider's creature or at
    its face: the same option, a different simulated result."""
    creature, face = (_sim(_opponent_bolts(t), "Pass priority") for t in ("Target Delver of Secrets", "Target player 0"))
    assert creature != face
    assert {"pv:sim:self:life->=3", "pv:sim:opponent:graveyard+>=1"} <= set(face)
    assert "pv:sim:self:life->=4" not in face and not any("creatures_lost" in t for t in face)


def test_passing_into_lethal_combat_damage_loses_the_game():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator"]}, p1={"life": 2}, step="declare_attackers")
    choose(g, "Attack with Gixian Infiltrator")
    if g.decision.kind == O.DECLARE_ATTACKER:
        choose(g, "Done declaring attackers")
    _pass_until(g, "declare_blockers", 1)
    sim = _sim(g, "Pass priority")
    assert sim[:2] == ["pv:sim:stop:game_over", "pv:sim:lost"]
    assert "pv:sim:self:life->=2" in sim and "pv:sim:self:life->=3" not in sim


def test_paying_mana_shows_the_mana_and_colours_left():
    g = scenario(p0={"hand": ["Ichor Wellspring"], "battlefield": ["Mountain", "Island"]})
    choose(g, "Cast Ichor Wellspring")
    assert g.decision.kind == O.PAY_MANA
    island = next(o.label for o in g.legal_options() if "Island" in o.label)
    sim = _sim(g, island)
    assert sim[:2] == ["pv:sim:stop:own_decision", "pv:sim:next:self:pay_mana"]
    assert {"pv:sim:self:tapped>=1", "pv:sim:mana_left>=1", "pv:sim:color:R"} <= set(sim)
    assert "pv:sim:color:U" not in sim and "pv:sim:mana_left>=2" not in sim


def test_skipped_without_a_snapshot_and_above_the_option_cap(monkeypatch, engine):
    import mtg_ml.encode as enc

    g = new_game(match_decks(1, "jund_blue"), seed=3)
    assert g.decision.kind == O.MULLIGAN  # no step has begun: nothing to copy from
    assert all(p[-2:] == ["pv:sim:skipped", "pv:simp:skipped"] for p in option_previews(g, g.decision.player, 6))
    if engine == "python":  # the cap is a constant in the native engine
        while g.decision.kind == O.MULLIGAN or len(g.legal_options()) < 2:
            g.step(0)
        monkeypatch.setattr(enc, "SIM_MAX_OPTIONS", 1)
        assert all(p[-2:] == ["pv:sim:skipped", "pv:simp:skipped"] for p in option_previews(g, g.decision.player, 6))


# ---------------------------------------------------------------------------
# "If the opponent passes" (pv:simp:)
# ---------------------------------------------------------------------------


def test_own_bolt_on_a_creature_kills_it_if_unanswered():
    g = _bolt_on_the_stack(["Island"])
    sim, simp = _sim(g, "Pass priority"), _simp(g, "Pass priority")
    assert not any("creatures_lost" in t for t in sim)  # the opponent may still respond
    assert simp[:2] == ["pv:sim:stop:own_decision", "pv:sim:next:self:priority"]
    assert {
        "pv:sim:opponent:creatures_lost>=1",
        "pv:sim:opponent:lost_power_tier:1",
        "pv:sim:opponent:graveyard+>=1",
        "pv:sim:self:graveyard+>=1",
        "pv:sim:stack->=1",
    } <= set(simp)


def test_own_counterspell_counters_the_spell_if_unanswered():
    g = scenario(
        p0={"hand": ["Counterspell"], "battlefield": ["Island", "Island", "Delver of Secrets"]},
        p1={"hand": ["Lightning Bolt"], "battlefield": ["Mountain"]},
        active=1,
    )
    choose(g, "Cast Lightning Bolt")
    choose(g, "Target Delver of Secrets")
    pay(g)
    choose(g, "Pass priority")
    choose(g, "Cast Counterspell")  # its only target, the Bolt, is chosen by settle()
    pay(g)
    assert g.decision.player == 0 and g.decision.kind == O.PRIORITY and len(g.stack) == 2
    simp = _simp(g, "Pass priority")
    # both spells leave the stack, the Delver lives; the opponent (active) passes on, so p0 decides again
    assert {"pv:sim:stack->=2", "pv:sim:self:graveyard+>=1", "pv:sim:opponent:graveyard+>=1"} <= set(simp)
    assert simp[:2] == ["pv:sim:stop:own_decision", "pv:sim:next:self:priority"]
    assert not any("creatures_lost" in t or "life-" in t for t in simp)
    assert _sim(g, "Pass priority")[:2] == ["pv:sim:stop:opponent_decision", "pv:sim:next:opponent:priority"]


def test_own_creature_spell_resolves_if_unanswered():
    g = scenario(p0={"hand": ["Gixian Infiltrator"], "battlefield": ["Swamp", "Swamp"]})
    choose(g, "Cast Gixian Infiltrator")
    pay(g)
    assert g.decision.player == 0 and g.decision.kind == O.PRIORITY and len(g.stack) == 1
    simp = _simp(g, "Pass priority")
    assert {"pv:sim:self:creatures_gained>=1", "pv:sim:self:power_gained>=2", "pv:sim:self:perms_gained>=1", "pv:sim:stack->=1"} <= set(simp)
    assert not any("creatures_gained" in t for t in _sim(g, "Pass priority"))


def test_attacker_passing_into_lethal_damage_wins_if_unanswered():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator"]}, p1={"life": 2}, step="declare_attackers")
    choose(g, "Attack with Gixian Infiltrator")
    if g.decision.kind == O.DECLARE_ATTACKER:
        choose(g, "Done declaring attackers")
    _pass_until(g, "declare_blockers", 0)
    assert _sim(g, "Pass priority")[:2] == ["pv:sim:stop:opponent_decision", "pv:sim:next:opponent:priority"]
    simp = _simp(g, "Pass priority")
    assert simp[:2] == ["pv:sim:stop:game_over", "pv:sim:won"] and "pv:sim:opponent:life->=2" in simp


def test_simp_repeats_sim_when_no_opponent_priority_comes_up():
    """Paying mana stops at the decider's next decision: passing never comes
    up, so `pv:simp:` is `pv:sim:` under the other prefix."""
    g = scenario(p0={"hand": ["Ichor Wellspring"], "battlefield": ["Mountain", "Island"]})
    choose(g, "Cast Ichor Wellspring")
    for i in range(len(g.legal_options())):
        assert _sim(g, i=i) == ["pv:sim:" + t.removeprefix("pv:simp:") for t in _sim(g, i=i, prefix="pv:simp:")]


def test_sets_before_6_have_no_sim_tokens():
    g = scenario(p0={"hand": ["Lightning Bolt"], "battlefield": ["Mountain"]})
    for f in (2, 3, 4, 5):
        assert not any(t.startswith("pv:sim:") for p in option_previews(g, 0, f) for t in p)
    assert all(any(t.startswith("pv:sim:stop:") for t in p) for p in option_previews(g, 0, 6))
