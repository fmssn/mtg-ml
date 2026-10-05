"""Turn structure, priority, combat and state-based actions."""

from helpers import bf, choose, find, has, labels, names, pass_priority, pay, resolve_stack, scenario

from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR, Game, expand
from mtg_ml.engine import objects as O


def attack(g: Game, *names: str) -> None:
    for n in names:
        choose(g, f"Attack with {n}")
    if g.decision.kind == O.DECLARE_ATTACKER:
        choose(g, "Done declaring attackers")


def to_step(g: Game, step: str) -> None:
    """Pass priority until the game reaches `step` (any player)."""
    for _ in range(200):
        if g.step_name == step and g.decision is not None:
            return
        assert g.decision.kind == O.PRIORITY, g.decision
        choose(g, "Pass priority")
    raise AssertionError(f"never reached {step}")


def test_opening_hands_and_first_turn_draw_skip():
    g = Game((expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR)), seed=7, auto_single=False)
    assert [len(p.hand) for p in g.players] == [7, 7]
    assert [len(p.library) for p in g.players] == [53, 53]
    sp = g.starting_player
    to_step(g, "main1")
    assert len(g.players[sp].hand) == 7  # no draw on the very first turn
    while g.turn == 1:
        assert g.decision.kind in (O.PRIORITY, O.DECLARE_ATTACKER, O.CHOOSE_CARD), g.decision
        choose(g, "Pass priority") if g.decision.kind == O.PRIORITY else g.step(0)
    to_step(g, "main1")
    assert g.active == 1 - sp and len(g.players[1 - sp].hand) == 8


def test_priority_passes_to_opponent_and_back():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp", "Swamp"]}, p1={"battlefield": ["Delver of Secrets"]})
    choose(g, "Cast Cast Down")
    pay(g)
    assert g.decision.player == 0  # caster keeps priority
    pass_priority(g)
    assert g.decision.player == 1 and g.stack
    pass_priority(g)
    assert not g.stack and g.decision.player == 0  # active player gets priority after resolution


def test_land_play_rules():
    g = scenario(p0={"hand": ["Swamp", "Forest"]}, step="upkeep")
    assert not has(g, "Play")  # main phase only
    g = scenario(p0={"hand": ["Swamp", "Forest"]})
    assert labels(g) == ["Pass priority", "Play Swamp", "Play Forest"]
    choose(g, "Play Swamp")
    assert not has(g, "Play")
    g = scenario(p1={"hand": ["Island"]}, active=0)
    pass_priority(g)
    assert g.decision.player == 1 and not has(g, "Play")  # not on the opponent's turn


def test_attack_declaration_enumerates_each_subset_once():
    g = scenario(p0={"battlefield": ["Eldrazi Spawn", "Eldrazi Spawn", "Gixian Infiltrator"]}, step="declare_attackers")
    assert g.decision.kind == O.DECLARE_ATTACKER
    seen = set()

    def walk(game, chosen):
        if game.decision.kind != O.DECLARE_ATTACKER:
            seen.add(tuple(sorted(chosen)))
            return
        for i, o in enumerate(game.legal_options()):
            f = game.fork()
            f.step(i)
            walk(f, chosen + ([o.key[1]] if o.key[1] else []))

    walk(g, [])
    # {0,1,2} spawns x {0,1} Gixian = 6 distinct declarations, none duplicated
    assert len(seen) == 6


def test_summoning_sick_and_tapped_creatures_cannot_attack():
    g = scenario(p0={"battlefield": [("Gixian Infiltrator", {"sick": True}), ("Krark-Clan Shaman", {"tapped": True})]}, step="declare_attackers")
    assert g.decision.kind != O.DECLARE_ATTACKER or labels(g) == ["Done declaring attackers"]


def test_unblocked_damage_and_flying_restriction():
    g = scenario(p0={"battlefield": ["Refurbished Familiar"]}, p1={"battlefield": ["Delver of Secrets", "Writhing Chrysalis"]}, step="declare_attackers")
    attack(g, "Refurbished Familiar")
    pass_priority(g, 2)
    # Delver cannot block a flyer: its only option was "does not block" (settled);
    # Chrysalis has reach and gets a real choice.
    assert g.decision.kind == O.DECLARE_BLOCKER and "Writhing Chrysalis" in g.decision.prompt
    choose(g, "does not block")
    to_step(g, "end_combat")
    assert g.players[1].life == 18


def test_reach_can_block_flyer():
    g = scenario(p0={"battlefield": ["Refurbished Familiar"]}, p1={"battlefield": ["Delver of Secrets", "Writhing Chrysalis"]}, step="declare_attackers")
    attack(g, "Refurbished Familiar")
    pass_priority(g, 2)
    # blockers: Delver gets no choice (auto-settled), Chrysalis may block
    assert g.decision.kind == O.DECLARE_BLOCKER and "Writhing Chrysalis" in g.decision.prompt
    choose(g, "blocks Refurbished Familiar")
    pass_priority(g, 2)
    assert "Refurbished Familiar" not in bf(g) and g.players[1].life == 20


def test_unblocked_attacker_deals_damage():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator"]}, step="declare_attackers")
    attack(g, "Gixian Infiltrator")
    to_step(g, "end_combat")
    assert g.players[1].life == 18


def test_multiple_blockers_free_damage_division():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator", "Writhing Chrysalis"]}, p1={"battlefield": ["Cryptic Serpent"]}, active=1, step="declare_attackers")
    attack(g, "Cryptic Serpent")
    pass_priority(g, 2)
    choose(g, "blocks Cryptic Serpent")  # Gixian
    choose(g, "blocks Cryptic Serpent")  # Chrysalis
    pass_priority(g, 2)
    assert g.decision.kind == O.ASSIGN_DAMAGE and g.decision.player == 1
    assert len(labels(g)) == 7  # 6 damage split freely over two blockers
    gix, chrys = (find(g, n) for n in ("Gixian Infiltrator", "Writhing Chrysalis"))
    choose(g, f"1 to Gixian Infiltrator#{gix.oid}, 5 to Writhing Chrysalis#{chrys.oid}")
    pass_priority(g)  # SBA happen before priority
    assert "Gixian Infiltrator" not in bf(g) and "Writhing Chrysalis" not in bf(g)
    assert find(g, "Cryptic Serpent").damage == 4


def test_trample_requires_lethal_to_blockers_first():
    g = scenario(
        p0={"battlefield": ["Gixian Infiltrator", ("Nyxborn Hydra", {"counters": 2})]},
        p1={"battlefield": ["Delver of Secrets"]},
        step="declare_attackers",
    )
    g.battlefield[1].attached_to = g.battlefield[0].oid  # bestowed: Gixian is 4/3 trample
    attack(g, "Gixian Infiltrator")
    pass_priority(g, 2)
    choose(g, "blocks Gixian")
    pass_priority(g, 2)
    d = find(g, "Delver of Secrets").oid
    assert sorted(labels(g)) == sorted(f"{4 - k} to Delver of Secrets#{d}, {k} to player" for k in range(4))
    choose(g, f"1 to Delver of Secrets#{d}, 3 to player")
    assert g.players[1].life == 17


def test_blocked_creature_whose_blocker_left_deals_no_damage():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator"]}, p1={"battlefield": ["Delver of Secrets"]}, step="declare_attackers")
    attack(g, "Gixian Infiltrator")
    pass_priority(g, 2)
    choose(g, "blocks Gixian")
    g.destroy(find(g, "Delver of Secrets"))
    to_step(g, "end_combat")
    assert g.players[1].life == 20


def test_sba_life_loss_ends_game():
    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Clue", "Swamp"]}, p1={"life": 1})
    choose(g, "Makeshift Munitions: 1 damage")
    choose(g, "player 1")
    resolve_stack(g)
    assert g.over and g.winner == 0 and g.end_reason == "life"


def test_decking_loss():
    # drawing from an empty library in the draw step loses at the next SBA check
    g2 = scenario(p0={"library": []}, p1={}, active=0, step="upkeep")
    pass_priority(g2, 2)
    assert g2.over and g2.winner == 1 and g2.end_reason == "decking"


def test_cleanup_discards_to_seven():
    g = scenario(p0={"hand": ["Swamp"] * 5 + ["Forest"] * 4}, step="end")
    pass_priority(g, 2)
    assert g.decision.kind == O.CHOOSE_CARD and sorted(labels(g)) == ["Discard Forest", "Discard Swamp"]
    choose(g, "Discard Forest")
    choose(g, "Discard Forest")
    assert len(g.players[0].hand) == 7


def test_damage_wears_off_in_cleanup():
    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Clue", "Swamp"]}, p1={"battlefield": ["Tolarian Terror"]}, step="end")
    choose(g, "Makeshift Munitions: 1 damage")
    choose(g, "Target Tolarian Terror")
    resolve_stack(g)  # ward trigger: p0 cannot pay {2} with no mana left -> countered
    assert find(g, "Tolarian Terror").damage == 0


def test_trigger_ordering_choice():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator", "Writhing Chrysalis", "Eldrazi Spawn"]})
    choose(g, "Eldrazi Spawn: sacrifice")
    assert g.decision.kind == O.ORDER_TRIGGERS and len(labels(g)) == 2
    choose(g, "Gixian")
    assert [it.source.name for it in g.stack] == ["Gixian Infiltrator", "Writhing Chrysalis"]


def test_mana_empties_between_steps():
    g = scenario(p0={"battlefield": ["Eldrazi Spawn", "Gixian Infiltrator"]})
    choose(g, "Eldrazi Spawn: sacrifice")
    assert g.players[0].pool == {"C": 1}
    resolve_stack(g)
    to_step(g, "begin_combat")
    assert g.players[0].pool == {}


def test_fork_is_exact():
    import random

    g = Game((expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR)), seed=3)
    r = random.Random(0)
    for _ in range(150):
        if g.over:
            break
        g.step(r.randrange(len(g.legal_options())))
    f = g.fork()
    assert [o.label for o in f.legal_options()] == [o.label for o in g.legal_options()]
    assert [names(p.hand) for p in f.players] == [names(p.hand) for p in g.players]
    assert [names(p.library) for p in f.players] == [names(p.library) for p in g.players]
    assert f.rng.getstate() == g.rng.getstate()
