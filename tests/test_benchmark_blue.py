"""Synthetic development fixtures: Blue lines, fair inputs and both-engine parity.

Positions are implementation-reviewed synthetic cases, not human demonstrations
or independently reviewed release puzzles. Each strategic case states its reason
and checks an outcome in addition to its initial action where applicable.
"""

from dataclasses import replace
import random

import pytest
from helpers import scenario, choose, pay, pass_priority, resolve_stack, names, find

from mtg_ml.backend import game_class
from mtg_ml.benchmark import AgentRegistry, take
from mtg_ml.benchmark.blue import (BenchmarkBlue, BOT_ID, PARAMETERS, register_blue, blue_factory,
                                   can_afford, damage_floor, _suffix_scores)
from mtg_ml.benchmark.views import inputs, freeze, LegalAction, KINDS
from mtg_ml.engine import MONO_BLUE_TERROR, JUND_WILDFIRE, expand
from mtg_ml.engine.view import determinize
from mtg_ml.bots import make_bot
from mtg_ml.benchmark.adapters import LegacyAdapter

ISLANDS = lambda n: ["Island"] * n  # noqa: E731


def bot():
    b = BenchmarkBlue()
    b.reset(MONO_BLUE_TERROR)
    return b


def choice(g, b=None):
    v, actions = inputs(g, g.decision.player, MONO_BLUE_TERROR)
    return (b or bot()).choose(v, actions)


def key(g):
    return g.legal_options()[choice(g)].key


def line(g, seat=0, limit=100):
    """One chosen spell/ability through all costs and its full resolution."""
    b = bot()
    g.step(choice(g, b))
    for _ in range(limit):
        if g.over or (not g.stack and g.decision.kind == "priority"):
            return
        g.step(choice(g, b) if g.decision.player == seat else 0)
    pytest.fail("line failed to reach its public boundary")


def test_registration_and_immutable_parameters():
    registry = AgentRegistry()
    register_blue(registry, "a" * 40)
    assert registry.metadata(BOT_ID).deck == "mono_blue_terror"
    assert isinstance(registry.create(BOT_ID), BenchmarkBlue)
    with pytest.raises(TypeError):
        PARAMETERS["cards"]["Brainstorm"] = 10
    with pytest.raises(ValueError, match="already registered"):
        register_blue(registry, "a" * 40)


def test_known_rule_coverage_and_unsupported_decisions():
    assert all(hasattr(BenchmarkBlue, "_" + kind) for kind in KINDS)
    g = scenario(p0={"hand": ["Ponder"], "battlefield": ISLANDS(1)})
    v, actions = inputs(g, 0, MONO_BLUE_TERROR)
    with pytest.raises(ValueError, match="reset required"):
        BenchmarkBlue().choose(v, actions)
    with pytest.raises(ValueError, match="unsupported decision"):
        bot().choose(v, [LegalAction(0, "new_kind", ("pass",), "", freeze({}))])
    with pytest.raises(ValueError, match="unsupported priority"):
        bot().choose(v, [replace(actions[0], key=("new_action",))])
    changed = replace(v, own_deck=freeze({"Island": 60}))
    with pytest.raises(ValueError, match="own-list changed"):
        bot().choose(changed, actions)


@pytest.mark.parametrize("lands,cantrips,keep", [(1,2,True), (1,1,False), (0,3,False), (2,0,True), (5,0,False)])
def test_cantrip_supported_opening(lands, cantrips, keep):
    # Same semantic mulligan menu, distinct self-contained seven-card hands.
    g = scenario(p0={"hand": ISLANDS(lands) + ["Ponder"]*cantrips + ["Tolarian Terror"]*(7-lands-cantrips)})
    v, _ = inputs(g, 0, MONO_BLUE_TERROR)
    actions = tuple(LegalAction(i, "mulligan", ("mulligan", n), "", freeze({})) for i, n in enumerate(("keep", "mulligan")))
    assert bot().choose(v, actions) == (0 if keep else 1)


def test_brainstorm_putbacks_and_mill_complete_line():
    # Redundant lands should be put back, then converted into discount fuel.
    g = scenario(p0={"hand": ["Brainstorm", "Mental Note", "Island", "Island"], "battlefield": ISLANDS(3),
                     "library": ["Island", "Counterspell", "Tolarian Terror"] + ISLANDS(10)})
    choose(g, "Cast Brainstorm")
    pay(g); resolve_stack(g)
    assert g.decision.kind == "choose_card"
    assert key(g) == ("put back", "Island")
    for _ in range(2):
        g.step(choice(g))
    resolve_stack(g)
    assert names(g.players[0].library[:2]) == ["Island", "Island"]
    assert "Counterspell" in names(g.players[0].hand) and "Tolarian Terror" in names(g.players[0].hand)
    # The casting priority can prefer deploying Terror; deliberately select the
    # reviewed follow-up then let the specialist complete its payments.
    choose(g, "Cast Mental Note"); pay(g); resolve_stack(g)
    assert names(g.players[0].graveyard).count("Island") == 2


def test_ponder_finds_land_and_preserves_order():
    g = scenario(p0={"hand": ["Ponder"], "battlefield": ISLANDS(1),
                     "library": ["Tolarian Terror", "Island", "Counterspell"] + ISLANDS(10)})
    line(g)
    assert names(g.players[0].hand) == ["Island"]
    assert names(g.players[0].library[:2]) == ["Counterspell", "Tolarian Terror"]


def test_ponder_shuffles_bad_known_cards():
    g = scenario(p0={"hand": ["Ponder"], "battlefield": ISLANDS(5), "library": ISLANDS(3)+["Counterspell"]*10})
    choose(g, "Cast Ponder"); pay(g); resolve_stack(g)
    # Three identical cards produce no order decision, only the shuffle menu.
    assert g.decision.kind == "yes_no" and key(g) == ("shuffle", "yes")
    g.step(choice(g)); resolve_stack(g)
    assert "Ponder" in names(g.players[0].graveyard)
    assert len(g.players[0].hand) == 1


def test_preserve_known_top_and_legitimate_reveal():
    g = scenario(p0={"hand": ["Mental Note"], "battlefield": ISLANDS(1)+["Delver of Secrets"],
                     "library": ["Counterspell"]+ISLANDS(10)}, step="upkeep")
    while g.decision.kind != "yes_no":
        g.step(0)
    assert key(g) == ("reveal", "yes")
    g.step(choice(g)); resolve_stack(g)
    assert "Insectile Aberration" in names(g.battlefield)
    # The known counter is valuable: avoid milling it at the opponent's end.
    v, actions = inputs(g, 0, MONO_BLUE_TERROR)
    state = dict(v.state); state.update(active="opponent", step="end")
    v = replace(v, state=freeze(state))
    assert bot().choose(v, actions) == 0


@pytest.mark.parametrize("board,expected", [(ISLANDS(2)+["Delver of Secrets"], ("pass",)),
                                            (ISLANDS(2), ("cast", "Mental Note", "hand", "normal"))])
def test_reserve_counter_only_with_pressure(board, expected):
    g = scenario(p0={"hand": ["Counterspell", "Mental Note"], "battlefield": board}, p1={"hand": ["Cast Down"]})
    assert key(g) == expected
    if expected[0] == "cast":
        line(g)
        assert "Mental Note" in names(g.players[0].graveyard) and len(g.players[0].hand) == 2


def test_empty_board_deploys_discounted_threat():
    g = scenario(p0={"hand": ["Tolarian Terror", "Counterspell"], "battlefield": ISLANDS(2), "graveyard": ["Ponder"]*6},
                 p1={"hand": ["Cast Down"]})
    line(g)
    assert "Tolarian Terror" in names(g.battlefield)


def test_lorien_cycling_full_line():
    g = scenario(p0={"hand": ["Lorien Revealed"], "battlefield": ISLANDS(1), "library": ["Ponder", "Island"]*5})
    assert key(g)[:2] == ("activate", "Lorien Revealed")
    line(g)
    assert names(g.players[0].hand) == ["Island"]
    assert "Lorien Revealed" in names(g.players[0].graveyard)


def test_lorien_draw_full_line():
    g = scenario(p0={"hand": ["Lorien Revealed"], "battlefield": ISLANDS(5)})
    line(g)
    assert len(g.players[0].hand) == 3


def test_optional_force_spike_payment_saves_own_threat():
    g = scenario(p0={"hand": ["Delver of Secrets"], "battlefield": ISLANDS(2)}, p1={"hand": ["Force Spike"], "battlefield": ISLANDS(1)})
    choose(g, "Cast Delver"); pay(g); pass_priority(g)
    choose(g, "Cast Force Spike"); pay(g); resolve_stack(g)
    assert g.decision.kind == "yes_no" and g.decision.player == 0
    assert key(g) == ("pay_optional", "yes")
    g.step(choice(g)); pay(g); resolve_stack(g)
    assert "Delver of Secrets" in names(g.battlefield)


@pytest.mark.parametrize("mana,counter", [(2,False),(4,True)])
def test_ward_and_removal_complete_line(mana, counter):
    g = scenario(p0={"hand": ["Counterspell"], "battlefield": ["Tolarian Terror"]+ISLANDS(2)},
                 p1={"hand": ["Cast Down"], "battlefield": ["Swamp"]*mana}, active=1)
    choose(g, "Cast Cast Down"); pay(g); pass_priority(g)
    assert (key(g)[0] == "cast") == counter
    if counter:
        line(g)
    else:
        # Pass the counter opportunity and decline the opponent's impossible tax.
        for _ in range(20):
            if not g.stack: break
            g.step(choice(g) if g.decision.player == 0 else 0)
    assert "Tolarian Terror" in names(g.battlefield)
    assert "Cast Down" in names(g.players[1].graveyard)
    assert ("Counterspell" in names(g.players[0].graveyard)) == counter


@pytest.mark.parametrize("mana,should_counter", [(2,True),(3,False)])
def test_force_spike_public_budget(mana, should_counter):
    g = scenario(p0={"hand": ["Force Spike"], "battlefield": ["Delver of Secrets"]+ISLANDS(1)},
                 p1={"hand": ["Cast Down"], "battlefield": ["Swamp"]*mana}, active=1)
    choose(g, "Cast Cast Down"); pay(g); pass_priority(g)
    assert (key(g)[0] == "cast") == should_counter
    if should_counter:
        line(g)
        assert "Delver of Secrets" in names(g.battlefield)


def test_counter_ignores_harmless_spell():
    g = scenario(p0={"hand": ["Counterspell"], "battlefield": ISLANDS(2)},
                 p1={"hand": ["Ichor Wellspring"], "battlefield": ["Swamp"]*2}, active=1)
    choose(g, "Cast Ichor Wellspring"); pay(g); pass_priority(g)
    assert key(g) == ("pass",)
    resolve_stack(g)
    assert "Ichor Wellspring" in names(g.battlefield)


def test_stack_references_and_counter_war():
    g = scenario(p0={"hand": ["Counterspell", "Counterspell"], "battlefield": ISLANDS(4)+["Delver of Secrets"]},
                 p1={"hand": ["Cast Down", "Counterspell"], "battlefield": ["Swamp"]*2+ISLANDS(2)}, active=1)
    choose(g, "Cast Cast Down"); pay(g); pass_priority(g)
    choose(g, "Cast Counterspell"); pay(g); pass_priority(g)
    choose(g, "Cast Counterspell"); choose(g, "Target spell Counterspell"); pay(g); pass_priority(g)
    assert key(g)[0:2] == ("cast", "Counterspell")
    choose(g, "Cast Counterspell")
    v, actions = inputs(g, 0, MONO_BLUE_TERROR)
    selected = actions[choice(g)]
    enemy_counter = next(s for s in v.context["stack"] if s["name"] == "Counterspell" and s["controller"] == "opponent")
    assert selected.data["target"]["sid"] == enemy_counter["sid"]
    while g.stack:
        g.step(choice(g) if g.decision.player == 0 else 0)
    assert "Delver of Secrets" in names(g.battlefield)
    assert "Cast Down" in names(g.players[1].graveyard)


def test_target_ward_budget_avoids_unpayable_bounce():
    g = scenario(p0={"hand": ["Deem Inferior"], "battlefield": ISLANDS(2), "drawn": 2},
                 p1={"battlefield": ["Tolarian Terror"]})
    assert key(g) == ("pass",)


def test_sleep_creates_evasive_lethal():
    g = scenario(p0={"hand": ["Sleep of the Dead"], "battlefield": ISLANDS(1)+[("Delver of Secrets", {"sick": False})], "library": ["Brainstorm"]+ISLANDS(10)},
                 p1={"battlefield": ["Writhing Chrysalis"], "life": 3}, step="upkeep")
    while g.decision.kind != "yes_no": g.step(0)
    g.step(choice(g))
    while not (g.step_name == "main1" and g.decision.kind == "priority" and g.decision.player == 0): g.step(0)
    line(g)
    assert find(g, "Writhing Chrysalis").tapped and find(g, "Writhing Chrysalis").skip_untap == 1
    for _ in range(100):
        if g.over: break
        g.step(choice(g) if g.decision.player == 0 else 0)
    assert g.over and g.winner == 0


def test_sleep_does_not_waste_card_on_existing_duration():
    g = scenario(p0={"hand": ["Sleep of the Dead"]*2, "battlefield": ISLANDS(2)+["Tolarian Terror"]},
                 p1={"battlefield": ["Cryptic Serpent"]})
    line(g)
    assert find(g, "Cryptic Serpent").skip_untap == 1
    assert key(g) == ("pass",)


def test_escape_preserves_discount_resources():
    g = scenario(p0={"hand": ["Tolarian Terror"], "battlefield": ISLANDS(3)+[("Delver of Secrets", {"sick": False})],
                     "graveyard": ["Sleep of the Dead", "Brainstorm", "Ponder", "Mental Note"]},
                 p1={"battlefield": ["Gixian Infiltrator"]})
    assert key(g)[:2] == ("cast", "Tolarian Terror")
    line(g)
    assert "Tolarian Terror" in names(g.battlefield)
    assert names(g.players[0].graveyard) == ["Sleep of the Dead", "Brainstorm", "Ponder", "Mental Note"]


def test_deem_owner_recovers_own_threat():
    g = scenario(p0={"battlefield": ["Delver of Secrets"]},
                 p1={"hand": ["Deem Inferior"], "battlefield": ISLANDS(4)}, active=1)
    choose(g, "Cast Deem Inferior"); pay(g); resolve_stack(g)
    assert g.decision.kind == "choose_mode" and key(g) == ("deem", "second")
    g.step(choice(g)); resolve_stack(g)
    assert g.players[0].library[1].name == "Delver of Secrets"


def test_combined_attack_lethal_through_one_blocker():
    g = scenario(p0={"battlefield": [("Tolarian Terror", {"sick": False})]*2},
                 p1={"battlefield": ["Eldrazi Spawn"], "life": 5}, step="declare_attackers")
    for _ in range(100):
        if g.over: break
        g.step(choice(g) if g.decision.player == 0 else (1 if g.decision.kind == "declare_blocker" else 0))
    assert g.over and g.winner == 0


def test_survival_blocks_and_multiblock():
    g = scenario(p0={"battlefield": [("Delver of Secrets", {"sick": False})]*2, "life": 1},
                 p1={"battlefield": [("Krark-Clan Shaman", {"sick": False})]}, active=1, step="declare_attackers")
    choose(g, "Attack with Krark")
    while g.decision.kind != "declare_blocker": g.step(0)
    assert key(g)[0] == "block" and key(g)[-1] == "Krark-Clan Shaman"
    while g.step_name != "main2":
        g.step(choice(g) if g.decision.player == 0 else 0)
    assert g.players[0].life == 1
    assert "Krark-Clan Shaman" in names(g.players[1].graveyard)


def test_crowded_damage_takes_lethal(engine):
    # PR63's eleven-blocker exact allocation: ten 1/1s and one 2/2, 12 damage.
    from mtg_ml.engine import objects as O
    def setup(g):
        g.add_card("Nyxborn Hydra", 0, "battlefield", sick=False, counters=12)
        for _ in range(10): g.add_card("Eldrazi Spawn", 1, "battlefield")
        g.add_card("Refurbished Familiar", 1, "battlefield")
        for p in (0, 1):
            for _ in range(20): g.add_card("Island", p, "library")
    g = game_class(engine)(([],[]), setup=setup, starting_player=0, start_step="declare_attackers", auto_single=False, log=True)
    while g.decision.kind != O.DECLARE_ATTACKER: g.step(0)
    choose(g, "Attack with Nyxborn")
    while g.decision.kind != O.DECLARE_BLOCKER: g.step(0)
    while g.decision.kind == O.DECLARE_BLOCKER:
        g.step(1)
    while g.decision.kind != O.ASSIGN_DAMAGE_AMOUNT: g.step(0)
    assert key(g)[0] == "damage_amount"
    while g.step_name != "main2": g.step(choice(g) if g.decision.player == 0 else 0)
    assert not [c for c in g.battlefield if c.controller == 1]


def test_hidden_world_labels_resets_and_ties(engine):
    g = game_class(engine)((expand(MONO_BLUE_TERROR), expand(JUND_WILDFIRE)), seed=8, auto_single=False, log=True)
    b = bot()
    checked = 0
    while not g.over and checked < 60:
        if g.decision.player == 0:
            v, actions = inputs(g, 0, MONO_BLUE_TERROR)
            expected = b.choose(v, actions)
            assert b.choose(v, [replace(a, label="unrelated") for a in actions]) == expected
            fork = determinize(g, 0, random.Random(checked))
            fork.deck_names = ("unrelated", "secret-opponent")
            assert inputs(fork, 0, MONO_BLUE_TERROR) == (v, actions)
            assert choice(fork, b) == expected
            b.reset(MONO_BLUE_TERROR)
            assert b.choose(v, actions) == expected
            checked += 1
            g.step(expected)
        else:
            g.step(make_bot(1, "jund_wildfire").act(g))
    assert checked == 60
    v, _ = inputs(scenario(), 0, MONO_BLUE_TERROR)
    tie = [LegalAction(i, "order_triggers", ("trigger", "same"), str(i), freeze({})) for i in range(2)]
    assert b.choose(v, tie) == 0


@pytest.mark.slow
def test_complete_games_and_mode_identity(engine):
    for opponent in ("jund_wildfire", "mono_blue_terror"):
        for seat in (0,1):
            decks = [expand(MONO_BLUE_TERROR), expand(JUND_WILDFIRE if opponent == "jund_wildfire" else MONO_BLUE_TERROR)]
            if seat: decks.reverse()
            g = game_class(engine)(decks, seed=2, starting_player=1-seat, auto_single=False, max_turns=100, log=True)
            agents = [None,None]
            agents[seat] = blue_factory(seat,"greedy")
            agents[1-seat] = LegacyAdapter(lambda p:make_bot(p,opponent),1-seat)
            for p,a in enumerate(agents): a.reset(MONO_BLUE_TERROR if p == seat or opponent == "mono_blue_terror" else JUND_WILDFIRE)
            while not g.over:
                assert len(g.actions) < 10000
                if g.decision.player == seat:
                    other = blue_factory(seat,"sampled"); other.reset(MONO_BLUE_TERROR)
                    assert other.act(g) == agents[seat].act(g)
                take(g, agents, agents[g.decision.player].act(g))
            assert g.end_reason in {"life", "decked", "turn_limit"}


def test_mana_budget_and_damage_assignment_arithmetic():
    g = scenario(p0={"battlefield": ISLANDS(2)})
    v, _ = inputs(g, 0, MONO_BLUE_TERROR)
    assert can_afford(v, "{U}{U}") and not can_afford(v, "{2}{U}")
    # Exact DP chooses the high-value victim if it cannot kill both.
    assert _suffix_scores([20,30], [2,2], 2, 0, False)[2] == 30
    cs = [c for c in v.state["battlefield"] if "Creature" in c["types"]]
    assert damage_floor(cs, []) == 0


def test_force_spike_combines_pending_ward_obligation():
    # Opponent can pay either {1} or {2}, but not both; Spike is live here.
    g = scenario(p0={"hand": ["Force Spike"], "battlefield": ["Tolarian Terror"]+ISLANDS(1)},
                 p1={"hand": ["Cast Down"], "battlefield": ["Swamp"]*4}, active=1)
    choose(g, "Cast Cast Down"); pay(g); pass_priority(g)
    assert key(g)[:2] == ("cast", "Force Spike")
    line(g)
    assert "Tolarian Terror" in names(g.battlefield)


def test_stack_only_card_rules_and_public_ward_whitelist():
    g = scenario(p0={"battlefield": ["Tolarian Terror"]},
                 p1={"hand": ["Cast Down"], "battlefield": ["Swamp"]*2}, active=1)
    choose(g, "Cast Cast Down"); pay(g); pass_priority(g)
    v, _ = inputs(g, 0, MONO_BLUE_TERROR)
    assert "Cast Down" in v.cards  # not in any other visible zone or own list
    ward = next(s for s in v.context["stack"] if "ward" in s)
    assert set(ward["ward"]) == {"sid", "amount"}
    assert ward["ward"]["amount"] == 2
    assert ward["source"]["name"] == "Tolarian Terror"
    assert not any("data" in s for s in v.context["stack"])


def test_trample_floor_and_cumulative_defensive_blocks():
    g = scenario(p0={"battlefield": ["Tolarian Terror"]*2},
                 p1={"battlefield": [("Nyxborn Hydra", {"counters":10})]})
    v, _ = inputs(g, 0, MONO_BLUE_TERROR)
    hydra = [c for c in v.state["battlefield"] if c["name"] == "Nyxborn Hydra"]
    defenders = [c for c in v.state["battlefield"] if c["controller"] == "self"]
    assert damage_floor(hydra, defenders[:1]) == 5
    assert damage_floor(hydra, defenders) == 0


def test_cantrip_avoids_self_decking_and_scour_uses_safe_target():
    g = scenario(p0={"hand": ["Mental Note"], "battlefield": ISLANDS(1), "library": ["Island"]*2})
    assert key(g) == ("pass",)
    g = scenario(p0={"hand": ["Thought Scour"], "battlefield": ISLANDS(1), "library": ["Island"]})
    choose(g, "Cast Thought Scour")
    assert key(g) == ("target", "player", "player", "opponent")
    g.step(choice(g)); pay(g); resolve_stack(g)
    assert not g.over and len(g.players[0].hand) == 1


@pytest.mark.parametrize("zone,lands,draws", [("hand",2,1), ("graveyard",4,2)])
def test_plunder_complete_normal_and_flashback_lines(zone, lands, draws):
    g = scenario(p0={zone: ["Plunder the Trollshaws"], "battlefield": ISLANDS(lands)})
    line(g)
    assert len(g.players[0].hand) == draws
    assert "Plunder the Trollshaws" in names(g.players[0].exile if zone == "graveyard" else g.players[0].graveyard)


def test_escape_takes_lethal_despite_graveyard_cost():
    g = scenario(p0={"battlefield": ISLANDS(3)+[("Tolarian Terror", {"sick": False})],
                     "graveyard": ["Sleep of the Dead", "Brainstorm", "Ponder", "Mental Note"]},
                 p1={"battlefield": ["Cryptic Serpent"], "life": 5})
    assert key(g)[0:2] == ("cast", "Sleep of the Dead")
    line(g)
    assert len(g.players[0].exile) == 3
    assert names(g.players[0].graveyard) == ["Sleep of the Dead"]
    for _ in range(100):
        if g.over: break
        g.step(choice(g) if g.decision.player == 0 else 0)
    assert g.over and g.winner == 0


def test_payment_spends_floating_mana_before_another_island():
    # A legal Spawn activation supplies floating generic mana. This synthetic
    # cost fixture covers the protocol's generic mana handler outside the list.
    g = scenario(p0={"hand": ["Plunder the Trollshaws"], "battlefield": ISLANDS(2)+["Eldrazi Spawn"]})
    g.step(next(i for i,o in enumerate(g.legal_options()) if o.key[0] == "mana"))
    g.step(next(i for i,o in enumerate(g.legal_options()) if o.key[:2] == ("cast", "Plunder the Trollshaws")))
    assert g.decision.kind == "pay_mana" and key(g) == ("pay", "pool", "C")
    g.step(choice(g))
    assert sum(c.tapped for c in g.battlefield if c.controller == 0) == 0
    g.step(choice(g))
    assert sum(c.tapped for c in g.battlefield if c.controller == 0) == 1


def test_counter_stops_shaman_toxin_wipe_complete_line():
    # Toxin targets the opponent's Shaman, yet its pending damage would kill
    # our Terror through deathtouch. Target ownership alone misses this threat.
    g = scenario(p0={"hand": ["Counterspell"], "battlefield": ISLANDS(2)+["Tolarian Terror"]},
                 p1={"hand": ["Toxin Analysis"], "battlefield": ["Krark-Clan Shaman", "Ichor Wellspring", "Swamp"]}, active=1)
    choose(g, "Krark-Clan Shaman: 1 damage")
    choose(g, "Cast Toxin Analysis"); choose(g, "Target Krark-Clan Shaman"); pay(g); pass_priority(g)
    assert key(g)[:2] == ("cast", "Counterspell")
    line(g)
    while g.stack:
        g.step(choice(g) if g.decision.player == 0 else 0)
    assert "Tolarian Terror" in names(g.battlefield)
    assert "Toxin Analysis" in names(g.players[1].graveyard)


def test_shaman_toxin_does_not_threaten_flying_delver():
    g = scenario(p0={"hand": ["Counterspell"], "battlefield": ISLANDS(2)+["Delver of Secrets"],
                     "library": ["Ponder"]+ISLANDS(10)},
                 p1={"hand": ["Toxin Analysis"], "battlefield": ["Krark-Clan Shaman", "Ichor Wellspring", "Swamp"]}, step="upkeep")
    while g.decision.kind != "yes_no": g.step(0)
    g.step(choice(g)); resolve_stack(g)
    pass_priority(g)
    choose(g, "Krark-Clan Shaman: 1 damage")
    choose(g, "Cast Toxin Analysis"); choose(g, "Target Krark-Clan Shaman"); pay(g); pass_priority(g)
    assert key(g) == ("pass",)
    while g.stack: g.step(choice(g) if g.decision.player == 0 else 0)
    assert "Insectile Aberration" in names(g.battlefield)
    assert names(g.players[0].hand) == ["Counterspell"]


def test_reservation_requires_colored_blue_not_just_total_mana():
    g = scenario(p0={"hand": ["Counterspell", "Mental Note"],
                     "battlefield": ISLANDS(2)+["Delver of Secrets"]+["Eldrazi Spawn"]*2},
                 p1={"hand": ["Cast Down"]})
    for _ in range(2):
        g.step(next(i for i,o in enumerate(g.legal_options()) if o.key[0] == "mana"))
    assert sum(g.players[0].pool.values()) == 2
    assert key(g) == ("pass",)  # Four mana, but the cantrip would leave only U.


def test_own_list_object_order_is_not_strategy_input():
    g = scenario(p0={"hand": ["Ponder"], "battlefield": ISLANDS(1)})
    v, actions = inputs(g, 0, MONO_BLUE_TERROR)
    shuffled = replace(v, own_deck=freeze(dict(reversed(list(MONO_BLUE_TERROR.items())))))
    assert bot().choose(v, actions) == bot().choose(shuffled, actions)


def test_factory_modes_and_actor_seeds_do_not_change_choices():
    g = scenario(p0={"hand": ["Ponder", "Brainstorm"], "battlefield": ISLANDS(1)})
    greedy = blue_factory(0, "greedy"); greedy.reset(MONO_BLUE_TERROR, actor_seed=1)
    sampled = blue_factory(0, "sampled"); sampled.reset(MONO_BLUE_TERROR, actor_seed=999)
    assert greedy.act(g) == sampled.act(g)
