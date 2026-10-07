"""Action decomposition options: `Game(auto_mana=True)` (colour-preserving
automatic mana payment) and `Game(auto_pass=True)` (collapse uneventful
priority passes). Runs on both engines (ENGINE_MODULES)."""

from helpers import choose, find, has, labels, new_game, pass_priority, scenario

from mtg_ml.engine import objects as O
from mtg_ml.match import match_decks


def tapped(g, player: int = 0) -> list[str]:
    return sorted(c.name for c in g.battlefield if c.controller == player and c.tapped)


def cast_down(g) -> None:
    choose(g, "Cast Cast Down")
    if g.decision.kind == O.TARGET:
        choose(g, "Target Tolarian Terror")


# -- auto_mana ---------------------------------------------------------------


def test_default_still_asks_for_every_payment():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp", "Mountain", "Forest"]}, p1={"battlefield": ["Tolarian Terror"]})
    cast_down(g)
    assert g.decision.kind == O.PAY_MANA


def test_auto_mana_keeps_the_colours_the_hand_needs():
    # Writhing Chrysalis ({2}{R}{G}) stays in hand: pay Cast Down's {1}{B} with
    # the two black sources and keep Forest and the R/G bridge untapped.
    p0 = {"hand": ["Cast Down", "Writhing Chrysalis"], "battlefield": ["Slagwoods Bridge", "Forest", "Swamp", "Vault of Whispers"]}
    g = scenario(p0=p0, p1={"battlefield": ["Tolarian Terror"]}, auto_mana=True)
    cast_down(g)
    assert g.decision.kind == O.PRIORITY and g.stack
    assert tapped(g) == ["Swamp", "Vault of Whispers"]
    assert any("pay_mana: Tap Swamp" in line and line.endswith("(auto)") for line in g.log)


def test_auto_mana_prefers_mono_coloured_sources_and_spares_twisted_landscape():
    # Nothing in hand needs a colour: the generic {1} comes from the Mountain,
    # not the B/R bridge (fewer colours) nor Twisted Landscape (keeps its search).
    p0 = {"hand": ["Cast Down"], "battlefield": ["Twisted Landscape", "Drossforge Bridge", "Mountain", "Swamp"]}
    g = scenario(p0=p0, p1={"battlefield": ["Tolarian Terror"]}, auto_mana=True)
    cast_down(g)
    assert tapped(g) == ["Mountain", "Swamp"]


def test_auto_mana_leaves_spawn_sacrifices_to_the_player():
    p0 = {"hand": ["Cast Down"], "battlefield": ["Swamp", "Mountain", "Eldrazi Spawn"]}
    g = scenario(p0=p0, p1={"battlefield": ["Tolarian Terror"]}, auto_mana=True)
    cast_down(g)
    assert g.decision.kind == O.PAY_MANA
    assert has(g, "Sacrifice Eldrazi Spawn")
    choose(g, "Sacrifice Eldrazi Spawn")  # pays the {1}; the {B} is auto-paid again
    assert g.decision.kind == O.PRIORITY and tapped(g) == ["Swamp"]


def test_auto_mana_spends_floating_mana_first():
    # Float {C} from the Spawn at priority (Gixian Infiltrator makes that a
    # real choice), then cast: the floating mana pays the generic part.
    p0 = {"hand": ["Cast Down"], "battlefield": ["Swamp", "Mountain", "Eldrazi Spawn", "Gixian Infiltrator"]}
    g = scenario(p0=p0, p1={"battlefield": ["Tolarian Terror"]}, auto_mana=True)
    choose(g, "Eldrazi Spawn: sacrifice: add C")
    while g.decision.kind != O.PRIORITY or g.stack:  # Infiltrator's counter trigger
        pass_priority(g) if g.decision.kind == O.PRIORITY else g.step(0)
    cast_down(g)
    assert g.decision.kind == O.PRIORITY and tapped(g) == ["Swamp"]
    assert g.players[0].pool in ({}, {"C": 0})


# -- auto_pass ---------------------------------------------------------------


def test_auto_pass_skips_a_useless_spawn_sacrifice():
    p0 = {"battlefield": ["Eldrazi Spawn"]}
    g = scenario(p0=p0)
    assert g.decision.player == 0 and labels(g) == ["Pass priority", "Eldrazi Spawn: sacrifice: add C"]
    g = scenario(p0=p0, auto_pass=True)
    assert g.decision.kind == O.PRIORITY and g.decision.player == 1 and g.step_name == "main1"
    find(g, "Eldrazi Spawn", 0)


def test_auto_pass_keeps_the_choice_when_sacrificing_matters():
    # Gixian Infiltrator grows whenever we sacrifice another permanent.
    g = scenario(p0={"battlefield": ["Eldrazi Spawn", "Gixian Infiltrator"]}, auto_pass=True)
    assert g.decision.player == 0 and has(g, "Eldrazi Spawn: sacrifice: add C")
    # Something to cast or play: never collapsed.
    g = scenario(p0={"hand": ["Swamp"], "battlefield": ["Eldrazi Spawn"]}, auto_pass=True)
    assert g.decision.player == 0 and has(g, "Play Swamp")


def test_options_play_whole_games():
    for kw in ({"auto_mana": True}, {"auto_pass": True}, {"auto_mana": True, "auto_pass": True}):
        g = new_game(match_decks(1), seed=3, max_turns=30, **kw)
        while not g.over:
            assert g.decision.kind != O.PAY_MANA or not kw.get("auto_mana") or any(o.label.startswith("Sacrifice") for o in g.legal_options())
            g.step(len(g.legal_options()) - 1)
        assert g.fork().actions == g.actions
