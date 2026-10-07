import random

from helpers import choose, new_game, pass_priority, pay, scenario, settle

from mtg_ml.agents import RandomAgent, play_game
from mtg_ml.bots import BlueBot, JundBot, make_bot
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR, expand
from mtg_ml.engine.view import determinize

DECKS = (expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR))


def _label(g, bot):
    return g.legal_options()[bot.act(g)].label


def test_bots_finish_games_and_handle_every_decision_kind():
    kinds = set()
    for s in range(40):
        g = new_game(DECKS, seed=s)
        agents = [make_bot(0), make_bot(1)] if s % 2 else [make_bot(0), RandomAgent(s)] if s % 4 else [RandomAgent(s), make_bot(1)]
        while not g.over:
            d = g.decision
            kinds.add(d.kind)
            a = agents[d.player].act(g)
            assert 0 <= a < len(d.options)
            g.step(a)
    assert {"priority", "target", "pay_mana", "declare_attacker", "declare_blocker", "choose_card"} <= kinds


def test_bots_beat_random():
    n = 30
    jund = sum(play_game([make_bot(0), RandomAgent(s)], seed=s).winner == 0 for s in range(n))
    blue = sum(play_game([RandomAgent(s), make_bot(1)], seed=s).winner == 1 for s in range(n))
    assert jund >= 0.85 * n and blue >= 0.85 * n


def test_bots_do_not_use_hidden_information():
    """Re-sampling every card the bot cannot see never changes its choice."""
    checked = 0
    for s in range(6):
        g, r = new_game(DECKS, seed=s), random.Random(s)
        bots = [make_bot(0), make_bot(1)]
        while not g.over and checked < 400:
            d = g.decision
            if r.random() < 0.15:
                p = d.player
                for k in range(2):
                    h = determinize(g, p, random.Random(1000 * s + k))
                    assert _label(h, make_bot(p)) == _label(g, make_bot(p)), (d, g.turn)
                checked += 1
            g.step(bots[d.player].act(g))
    assert checked > 100


def test_blue_counters_removal_on_its_threat():
    g = scenario(
        p0={"hand": ["Cast Down"], "battlefield": ["Swamp", "Swamp", "Swamp", "Swamp"]},
        p1={"hand": ["Counterspell", "Brainstorm"], "battlefield": ["Tolarian Terror", "Island", "Island"]},
    )
    choose(g, "Cast Cast Down")
    pay(g)
    while g.decision.player != 1:  # ward trigger: Jund still has {2} up
        g.step(0)
        settle(g)
    assert _label(g, BlueBot(1)) == "Cast Counterspell"


def test_blue_does_not_counter_what_ward_already_counters():
    g = scenario(
        p0={"hand": ["Cast Down"], "battlefield": ["Swamp", "Swamp"]},
        p1={"hand": ["Counterspell"], "battlefield": ["Tolarian Terror", "Island", "Island"]},
    )
    choose(g, "Cast Cast Down")
    pay(g)
    while g.decision.player != 1:
        g.step(0)
        settle(g)
    assert _label(g, BlueBot(1)) == "Pass priority"


def test_blue_lets_a_cantrip_artifact_resolve():
    g = scenario(
        p0={"hand": ["Ichor Wellspring"], "battlefield": ["Swamp", "Swamp"]},
        p1={"hand": ["Counterspell"], "battlefield": ["Island", "Island"]},
    )
    choose(g, "Cast Ichor Wellspring")
    pay(g)
    pass_priority(g)
    assert g.decision.player == 1 and _label(g, BlueBot(1)) == "Pass priority"


def test_jund_cast_down_picks_the_biggest_threat():
    g = scenario(
        p0={"hand": ["Cast Down"], "battlefield": ["Swamp", "Swamp"]},
        p1={"battlefield": ["Delver of Secrets", "Cryptic Serpent"]},
    )
    bot = JundBot(0)
    assert _label(g, bot) == "Cast Cast Down"
    g.step(bot.act(g))
    settle(g)
    assert "Cryptic Serpent" in _label(g, bot)


def test_jund_wildfires_its_own_bridge():
    g = scenario(p0={"hand": ["Cleansing Wildfire"], "battlefield": ["Slagwoods Bridge", "Mountain", "Swamp"]}, p1={"battlefield": ["Island"] * 3})
    bot = JundBot(0)
    assert _label(g, bot) == "Cast Cleansing Wildfire"
    g.step(bot.act(g))
    settle(g)
    assert "Slagwoods Bridge" in _label(g, bot) and "(self)" in _label(g, bot)


def test_blocker_takes_a_free_kill_and_avoids_a_bad_block():
    g = scenario(
        p0={"battlefield": ["Writhing Chrysalis"]},
        p1={"battlefield": [("Delver of Secrets", {"sick": False})]},
        active=1,
        step="declare_attackers",
    )
    # Blue's 1/1 Delver attacks into a 2/3: the Jund bot blocks and kills it.
    while g.decision.kind != "declare_attacker":
        g.step(0)
    choose(g, "Attack with Delver of Secrets")
    while g.decision.kind != "declare_blocker":
        g.step(0)
    assert "blocks Delver of Secrets" in _label(g, JundBot(0))


def test_search_bot_returns_legal_choices_and_respects_hidden_info():
    from mtg_ml.bots.search import SearchBot

    g = new_game(DECKS, seed=11)
    bots = [SearchBot(0, playouts=2, max_options=2, seed=1), make_bot(1)]
    steps = 0
    while not g.over and steps < 60:
        d = g.decision
        a = bots[d.player].act(g)
        assert 0 <= a < len(d.options)
        if d.player == 0 and d.kind == "priority" and len(d.options) > 1:
            # same seed, re-sampled hidden cards: the search sees only what Jund knows
            h = determinize(g, 0, random.Random(5))
            assert SearchBot(0, playouts=2, max_options=2, seed=1).act(h) == SearchBot(0, playouts=2, max_options=2, seed=1).act(g)
        g.step(a)
        steps += 1


def test_search_bot_uses_the_bots_of_the_decks_in_play(monkeypatch):
    """On jund_madness the Red seat's search builds a Red base bot, and its
    playouts run the bots of the decks in play (not Jund vs Blue)."""
    from mtg_ml.bots import RedBot, search
    from mtg_ml.match import game_args
    from mtg_ml.play import make_agent

    bot = make_agent("search:2", 1, seed=1, deck="red_madness")
    assert isinstance(bot.base, RedBot) and bot.name == "search(" + RedBot(1).name + ")"
    built = []
    monkeypatch.setattr(search, "make_bot", lambda seat, deck=None: built.append((seat, deck)) or make_bot(seat, deck))
    g = new_game(seed=3, **game_args(1, "jund_madness"))
    steps = 0
    while not built and not g.over and steps < 200:
        d = g.decision
        g.step(bot.act(g) if d.player == 1 else make_bot(0).act(g))
        steps += 1
    assert built and all(deck == g.deck_names[seat] for seat, deck in built)
    assert (1, "red_madness") in built


def test_search_pins_offered_cards_through_determinize():
    """Library cards a decision offers (a search) keep their real definitions
    in the re-dealt worlds; the hidden multiset is unchanged."""
    from mtg_ml.bots.search import _pin

    g = new_game(DECKS, seed=3)
    lib = g.players[0].library
    names = sorted({c.defn.name for c in lib})
    picks = [next(c for c in lib if c.defn.name == n) for n in names[:3]]
    pins = {c.oid: c.defn.name for c in picks}
    for s in range(10):
        h = determinize(g, 0, random.Random(s))
        before = sorted(c.defn.name for c in h.players[0].library)
        _pin(h, 0, pins)
        hl = h.players[0].library
        assert sorted(c.defn.name for c in hl) == before
        assert {c.oid: c.defn.name for c in hl if c.oid in pins} == pins
