"""Synthetic development fixtures, reviewed against rules outcomes below.

These are strategy regressions, not human demonstrations or reserved final
puzzles. Each fixture's docstring records the tactical reason for its outcome.
All setup is explicit; stack, payments and combat are reached by legal actions.
"""

from dataclasses import replace
import random

import pytest
from helpers import scenario, choose, bf, find, new_game

from mtg_ml.benchmark import AgentRegistry, ScriptedAdapter, inputs, take
from mtg_ml.benchmark.bots.jund import BenchmarkJundBot, PARAMETERS_SHA256, register_jund, UnsupportedDecision
from mtg_ml.benchmark.views import freeze, thaw, KINDS
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR
from mtg_ml.engine.view import determinize, observe


class PassAdapter:
    def act(self, g):
        return 0


def adapter(seat=0, deck=JUND_WILDFIRE):
    a = ScriptedAdapter(BenchmarkJundBot(), seat)
    a.reset(deck)
    return a


def drive(g, bot=None, *, until=None, limit=150):
    a = bot or adapter()
    agents = [a, PassAdapter()] if a.seat == 0 else [PassAdapter(), a]
    actions = []
    for _ in range(limit):
        if g.over or until and until(g):
            return actions
        d = g.decision
        i = agents[d.player].act(g)
        actions.append((d.player, d.kind, d.options[i].key))
        take(g, agents, i)
    raise AssertionError(f"fixture exceeded {limit} decisions: {g.decision}")


def end(g):
    return not g.stack and g.step_name == "end" and g.decision.player == 0


def key(g, a=None):
    return g.legal_options()[(a or adapter()).act(g)].key


def test_registration_and_unsupported_decisions():
    registry = AgentRegistry()
    register_jund(registry, "a" * 40)
    assert registry.metadata("benchmark-jund@1").parameters_sha256 == PARAMETERS_SHA256
    assert isinstance(registry.create("benchmark-jund@1"), BenchmarkJundBot)
    assert all(hasattr(BenchmarkJundBot, "choose_" + k) for k in KINDS)
    g = scenario(p0={"battlefield": ["Swamp"]})
    v, actions = inputs(g, 0, JUND_WILDFIRE)
    a = adapter().agent
    with pytest.raises(UnsupportedDecision):
        a.choose(v, (replace(actions[0], kind="future_decision"),))


def test_wildfire_ramps_bridge_and_fixes_green():
    """Keep the Bridge and obtain green for Chrysalis, completing all costs."""
    g = scenario(p0={"hand": ["Cleansing Wildfire", "Writhing Chrysalis"],
                     "battlefield": ["Drossforge Bridge", "Mountain"], "library": ["Forest", "Swamp", "Lembas"]})
    drive(g, until=end)
    assert "Drossforge Bridge" in bf(g, 0) and "Forest" in bf(g, 0)
    assert "Cleansing Wildfire" in [c.name for c in g.players[0].graveyard]


def test_wildfire_preserves_land_when_basics_exhausted():
    """Own registration and public zones prove there is no ramp left."""
    g = scenario(p0={"hand": ["Cleansing Wildfire"], "battlefield": ["Slagwoods Bridge", "Mountain"],
                     "graveyard": ["Swamp"] * 3 + ["Forest"] * 2})
    assert key(g) == ("pass",)


def test_untapped_land_enables_wildfire_now():
    """An untapped red source wins over a second tapped Bridge."""
    g = scenario(p0={"hand": ["Mountain", "Drossforge Bridge", "Cleansing Wildfire"], "battlefield": ["Slagwoods Bridge"]})
    assert key(g) == ("play_land", "Mountain")


def test_payment_preserves_black_for_removal():
    """A generic artifact payment uses green and leaves the only black source."""
    g = scenario(p0={"hand": ["Ichor Wellspring", "Cast Down"], "battlefield": ["Swamp", "Forest", "Forest", "Forest"]},
                 p1={"battlefield": ["Cryptic Serpent"]})
    choose(g, "Cast Ichor Wellspring")
    a = adapter()
    while g.decision.kind == "pay_mana":
        g.step(a.act(g))
    assert not find(g, "Swamp").tapped


def test_draw_sacrifices_wellspring_and_keeps_lands():
    """Offering plus Wellspring draws three and leaves mana intact."""
    g = scenario(p0={"hand": ["Fanatical Offering"], "battlefield": ["Ichor Wellspring", "Swamp", "Swamp"],
                     "library": ["Forest"] * 10})
    drive(g, until=lambda g: "Map" in bf(g) and not g.stack)
    assert len(g.players[0].hand) == 3
    assert "Ichor Wellspring" not in bf(g) and bf(g).count("Swamp") == 2


def test_spawn_payment_and_sacrifice_are_distinct():
    """One Spawn pays while the other remains available for Offering."""
    g = scenario(p0={"hand": ["Fanatical Offering"], "battlefield": ["Eldrazi Spawn", "Eldrazi Spawn", "Swamp"]})
    drive(g, until=lambda g: "Map" in bf(g) and not g.stack)
    assert "Eldrazi Spawn" not in bf(g)
    assert len(g.players[0].hand) == 2


def test_does_not_sacrifice_necessary_artifact_land_for_draw():
    g = scenario(p0={"hand": ["Fanatical Offering"], "battlefield": ["Vault of Whispers", "Swamp"]})
    assert key(g) == ("pass",)


def test_unaffordable_ward_does_not_waste_removal():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp", "Swamp"]}, p1={"battlefield": ["Tolarian Terror"]})
    assert key(g) == ("pass",)


def test_affordable_ward_removal_completes():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 4}, p1={"battlefield": ["Tolarian Terror"]})
    drive(g, until=end)
    assert "Tolarian Terror" not in bf(g)
    assert "Cast Down" in [c.name for c in g.players[0].graveyard]


def test_pending_removal_is_not_duplicated():
    g = scenario(p0={"hand": ["Cast Down", "Cast Down"], "battlefield": ["Swamp"] * 4}, p1={"battlefield": ["Cryptic Serpent"]})
    actions = drive(g, until=end)
    assert sum(k[:2] == ("cast", "Cast Down") for _, _, k in actions) == 1
    assert "Cryptic Serpent" not in bf(g)


def test_toxin_shaman_resolves_buff_before_sweep():
    """Kill Terror through ward, gain life, keep flying Familiar and lands."""
    g = scenario(p0={"hand": ["Toxin Analysis"], "battlefield": ["Krark-Clan Shaman", "Ichor Wellspring", "Refurbished Familiar", "Swamp"]},
                 p1={"battlefield": ["Tolarian Terror", "Cryptic Serpent"]})
    actions = drive(g, until=lambda g: not g.stack and "Tolarian Terror" not in bf(g))
    assert actions[0][2][:2] == ("cast", "Toxin Analysis")
    assert "Cryptic Serpent" not in bf(g)
    assert "Refurbished Familiar" in bf(g) and "Swamp" in bf(g)
    assert g.players[0].life > 20


def test_shaman_stacks_required_activations_before_dying():
    """Two shots kill a 2-toughness attacker; the source dies on the first."""
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman", "Ichor Wellspring", "Ichor Wellspring"]},
                 p1={"battlefield": ["Nyxborn Hydra", ("Nyxborn Hydra", {"counters": 1}), ("Gixian Infiltrator", {"counters": 1})]})
    drive(g, until=lambda g: not g.stack and "Gixian Infiltrator" not in bf(g))
    assert "Krark-Clan Shaman" not in bf(g)
    assert not [c for c in g.battlefield if c.controller == 1]


def test_shaman_rejects_unfavorable_collateral():
    g = scenario(p0={"hand": ["Toxin Analysis"], "battlefield": ["Krark-Clan Shaman", "Writhing Chrysalis", "Writhing Chrysalis", "Ichor Wellspring", "Swamp"]},
                 p1={"battlefield": ["Delver of Secrets"]})
    assert key(g) == ("pass",)


def test_hydra_bestow_uses_ready_evasive_creature():
    g = scenario(p0={"hand": ["Nyxborn Hydra"], "battlefield": ["Forest"] * 4 + [("Refurbished Familiar", {"sick": False})]})
    actions = drive(g, until=lambda g: any(c.name == "Nyxborn Hydra" for c in g.battlefield))
    assert actions[0][2] == ("cast", "Nyxborn Hydra", "hand", "bestow")
    assert find(g, "Nyxborn Hydra").attached_to == find(g, "Refurbished Familiar").oid
    assert find(g, "Nyxborn Hydra").counters == 2


def test_hydra_normal_uses_x_and_stays_creature():
    g = scenario(p0={"hand": ["Nyxborn Hydra"], "battlefield": ["Forest"] * 3})
    drive(g, until=lambda g: "Nyxborn Hydra" in bf(g))
    assert find(g, "Nyxborn Hydra").counters == 2 and find(g, "Nyxborn Hydra").attached_to is None


def test_munitions_removes_creature_when_face_is_not_lethal():
    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Ichor Wellspring", "Swamp"]}, p1={"battlefield": ["Delver of Secrets"]})
    drive(g, until=lambda g: not g.stack and "Delver of Secrets" not in bf(g))
    assert g.players[1].life == 20


def test_munitions_repeated_face_lethal():
    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Ichor Wellspring", "Ichor Wellspring", "Swamp", "Swamp"]}, p1={"life": 2})
    drive(g)
    assert g.over and g.winner == 0


def test_spawn_growth_produces_combat_lethal():
    g = scenario(p0={"battlefield": [("Writhing Chrysalis", {"sick": False}), "Eldrazi Spawn"]}, p1={"life": 3}, step="begin_combat")
    # begin_combat is not a main phase; the eventual attack alone cannot yet win.
    g = scenario(p0={"battlefield": [("Writhing Chrysalis", {"sick": False}), "Eldrazi Spawn"]}, p1={"life": 3})
    drive(g)
    assert g.over and g.winner == 0


def test_gang_blocks_kill_larger_attacker_and_survive():
    g = scenario(p0={"life": 3, "battlefield": [("Gixian Infiltrator", {"counters": 1})] * 2},
                 p1={"battlefield": [("Tolarian Terror", {"sick": False})]}, active=1, step="declare_attackers")
    choose(g, "Attack with Tolarian Terror")
    if g.decision.kind == "declare_attacker":
        choose(g, "Done declaring attackers")
    drive(g, until=lambda g: g.step_name == "end_combat" and not g.stack)
    assert g.players[0].life == 3 and "Tolarian Terror" not in bf(g)


def test_all_out_attack_finds_combined_lethal():
    g = scenario(p0={"battlefield": [("Gixian Infiltrator", {"sick": False})] * 3},
                 p1={"life": 4, "battlefield": ["Tolarian Terror"]}, step="declare_attackers")
    a = adapter()
    while g.decision.kind == "declare_attacker":
        g.step(a.act(g))
    assert len(g.attackers) == 3


def test_large_allocation_takes_lethal():
    from test_correctness_foundations import combat
    g = combat()
    a = adapter()
    assert g.decision.kind == "assign_damage_amount"
    drive(g, a)
    assert g.over and g.winner == 0


def test_midline_reset_labels_and_hidden_worlds_do_not_change_choice():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 4},
                 p1={"hand": ["Ponder"], "battlefield": ["Tolarian Terror", "Cryptic Serpent"]})
    choose(g, "Cast Cast Down")
    a = adapter()
    v, actions = inputs(g, 0, JUND_WILDFIRE)
    expected = a.agent.choose(v, actions)
    state = thaw(v.state)
    state["decision"]["prompt"] = "display changed"
    state["decision"]["options"] = ["redacted"] * len(actions)
    altered = replace(v, state=freeze(state))
    assert a.agent.choose(altered, tuple(replace(x, label="redacted") for x in actions)) == expected
    assert adapter().act(g) == expected
    for seed in range(4):
        h = determinize(g, 0, random.Random(seed))
        assert adapter().act(h) == expected
    before = observe(g, 0)
    a.act(g)
    assert observe(g, 0) == before
    with pytest.raises((TypeError, AttributeError)):
        v.context["stack"][-1]["targets"] = ()


def test_duplicate_card_refs_choose_actual_threat():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 2},
                 p1={"battlefield": ["Gixian Infiltrator", ("Gixian Infiltrator", {"counters": 4})]})
    drive(g, until=end)
    survivors = [c for c in g.battlefield if c.controller == 1]
    assert len(survivors) == 1 and survivors[0].counters == 0


@pytest.mark.slow
def test_full_games_cover_decisions_and_mode_equivalence():
    from mtg_ml.bots import make_bot
    from mtg_ml.benchmark.adapters import LegacyAdapter
    from mtg_ml.benchmark.schedule import expand
    for seat in (0, 1):
        decks = (JUND_WILDFIRE, MONO_BLUE_TERROR) if seat == 0 else (MONO_BLUE_TERROR, JUND_WILDFIRE)
        g = new_game(tuple(expand(d) for d in decks), seed=5, auto_single=False, log=True)
        a = adapter(seat)
        other = LegacyAdapter(lambda s: make_bot(s, "mono_blue_terror"), 1-seat)
        other.reset(MONO_BLUE_TERROR)
        agents = [a, other] if seat == 0 else [other, a]
        n = 0
        while not g.over:
            who = g.decision.player
            index = agents[who].act(g)
            if who == seat:
                # Scripted actors ignore actor seeds/mode: a fresh reset agrees.
                assert adapter(seat).act(g) == index
            take(g, agents, index)
            n += 1
            assert n < 10000
