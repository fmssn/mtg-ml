"""Jund Wildfire bot: artifact value and removal.

Plan: land every turn (Bridges early, while their tapped entry costs
nothing), artifacts and creatures on curve, Cleansing Wildfire on its own
indestructible Bridges (ramp + card), removal on the biggest flyer/serpent,
sacrifice-draw spells on cheap fodder, Spellbomb when Blue's graveyard
fuels Terror / Serpent / escape. Removal waits for Blue to tap low when it
can afford to (Counterspell needs UU).
"""

from __future__ import annotations

from ..engine.game import Game
from ..engine.objects import Card, Decision, Option
from .base import NEG, Bot

# Tunable judgment calls.
KEEP_LANDS = {7: (2, 5), 6: (2, 4)}  # land counts to keep at each hand size; always keep 5 or fewer
REMOVAL_MIN_THREAT = 4.0  # creature_value worth a Cast Down
WAIT_FOR_TAPPED_OUT = True  # hold removal while Blue has UU up and we're not under pressure
PRESSURE_LIFE = 10  # below this, stop waiting
SPELLBOMB_GY_FUEL = 4  # instants/sorceries in Blue's graveyard before exiling it
LEMBAS_LIFE = 8
HOLD_REMOVAL_MANA = True  # on our turn, keep {1}{B} open for instant-speed removal we hold
HOLD_FROM_TURN = 3  # ...from this game turn on, or earlier when Blue already has a threat
SPARE_LAND_COUNT = 6  # with this many lands, an extra land is cheap sacrifice fodder
REMOVAL = ("Cast Down", "Go for the Throat", "Red Elemental Blast")

BRIDGES = {"Drossforge Bridge", "Slagwoods Bridge", "Vault of Whispers"}
CURVE = {  # main-phase cast priority
    "Writhing Chrysalis": 9,
    "Refurbished Familiar": 8,
    "Nyxborn Hydra": 6,
    "Gixian Infiltrator": 7,
    "Krark-Clan Shaman": 4,
    "Ichor Wellspring": 5,
    "Lembas": 4.5,
    "Nihil Spellbomb": 4,
    "Makeshift Munitions": 3,
}


class JundBot(Bot):
    name = "jund"

    # -- valuations ---------------------------------------------------------

    def card_value(self, g: Game, name: str) -> float:
        card = g.cards_db[name]
        if card.is_type("Land"):
            return 4.0 if self.land_need(g) > 0 else 1.0
        return {
            "Cast Down": 5,
            "Go for the Throat": 5,
            "Red Elemental Blast": 4.5,
            "Duress": 2.5,
            "Writhing Chrysalis": 4.5,
            "Refurbished Familiar": 4,
            "Cleansing Wildfire": 3.5,
            "Fanatical Offering": 3,
            "Eviscerator's Insight": 3,
            "Gixian Infiltrator": 3,
            "Toxin Analysis": 2.5,
            "Krark-Clan Shaman": 2,
            "Nyxborn Hydra": 2.5,
            "Ichor Wellspring": 2,
            "Makeshift Munitions": 2,
            "Lembas": 1.5,
            "Nihil Spellbomb": 1.5,
        }.get(name, 1.0)

    def keep_hand(self, g: Game, size: int) -> bool:
        if size <= 5:
            return True
        hand = self.hand(g)
        lands = sum(1 for c in hand if c.face.is_type("Land"))
        lo, hi = KEEP_LANDS[size]
        cheap = sum(1 for c in hand if not c.face.is_type("Land") and c.face.cost.mana_value <= 2)
        return lo <= lands <= hi and cheap >= 1

    def threat(self, g: Game, c: Card) -> float:
        v = self.creature_value(g, c)
        if c.name in ("Tolarian Terror", "Cryptic Serpent"):
            v += 2
        return v

    def blue_counter_up(self, g: Game) -> bool:
        return len(self.untapped_lands(g, self.opp)) >= 2

    DURESS_ORDER = {"Counterspell": 6, "Dispel": 5, "Blue Elemental Blast": 4.5, "Steel Sabotage": 4, "Brainstorm": 3, "Lorien Revealed": 2.5, "Ponder": 2, "Force Spike": 2}

    def score_choose_card(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[0] == "duress":
            return self.DURESS_ORDER.get(o.key[1], 1.0)
        return super().score_choose_card(g, d, o)

    def search_value(self, g: Game, name: str) -> float:
        # Basic land fetch: the colour our hand is shortest of.
        have = {"B": 0, "R": 0, "G": 0}
        for c, ab in g.mana_sources(self.p):
            for col in ab.mana:
                if col in have:
                    have[col] += 1
        col = {"Swamp": "B", "Mountain": "R", "Forest": "G"}.get(name)
        return 5 - have.get(col, 5) if col else 0.5

    # -- priority -----------------------------------------------------------

    def score_priority(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] == "pass":
            return 0.0
        if k[0] == "play_land":
            return self.land_score(g, k[1]) if g.step_name == "main1" or not self.creatures_to_cast(g) else self.land_score(g, k[1]) - 1
        if k[0] == "cast":
            card = o.value[1]
            if len(k) > 4:
                return self.modal_score(g, card, k[4])
            return self.cast_score(g, card, k[3])
        if k[0] == "activate":
            return self.activate_score(g, o.value[1], k[2])
        return NEG  # Spawn mana at priority: only while paying

    def creatures_to_cast(self, g: Game) -> bool:
        return any(c.name in CURVE for c in self.hand(g))

    def land_score(self, g: Game, name: str) -> float:
        s = 50.0
        # An untapped land matters only if it lets us cast a main-phase play this turn.
        castable_now = any(c.name in CURVE and self.cost(g, c) == self.mana_available(g) + 1 for c in self.hand(g))
        if name in BRIDGES and name != "Vault of Whispers":
            s += 1 if not castable_now else -2  # enters tapped
        if name == "Twisted Landscape":
            s -= 0.5
        return s

    def cast_score(self, g: Game, card: Card, mode: str) -> float:
        n = card.name
        main = self.main_phase(g)
        opp_creatures = self.creatures(g, self.opp)
        if n == "Duress":
            return 13.0 if main and g.players[self.opp].hand else NEG
        if n in ("Cast Down", "Go for the Throat"):
            targets = [c for c in opp_creatures if self.threat(g, c) >= REMOVAL_MIN_THREAT and not (n == "Go for the Throat" and g.is_artifact(c))]
            if not targets:
                return NEG
            pressured = self.me(g).life < PRESSURE_LIFE or any(c.oid in g.attackers for c in targets)
            if WAIT_FOR_TAPPED_OUT and self.blue_counter_up(g) and not pressured and not self.opp_end_step(g):
                return NEG
            return 30.0
        if n == "Toxin Analysis":
            if g.step_name == "declare_blockers" and self.combat_trick_target(g):
                return 25.0
            return 2.0 if self.opp_end_step(g) and self.mana_available(g) >= 1 else NEG
        if n in ("Fanatical Offering", "Eviscerator's Insight"):
            fodder = min((self.sac_cost(g, c) for c in g.sac_candidates(self.p, "artifact_or_creature")), default=99)
            if fodder > 2.5 and len(self.hand(g)) > 1:
                return NEG
            if self.opp_end_step(g) or (main and not self.creatures_to_cast(g)):
                return 6.0 if mode == "normal" else 4.0
            return NEG
        if n == "Cleansing Wildfire":
            if not main:
                return NEG
            mine = [c for c in self.lands(g, self.p) if c.name in BRIDGES and c.name != "Vault of Whispers"]
            return 12.0 if mine else 1.0
        if not main:
            return NEG
        if self.breaks_reserve(g, card):
            return NEG
        if n == "Nyxborn Hydra":
            return CURVE[n] if self.mana_available(g) >= 3 else NEG
        if n in CURVE:
            return CURVE[n] + 0.1 * self.cost(g, card)
        return NEG

    def removal_reserve(self, g: Game) -> int:
        """Mana to keep open on our own turn for instant-speed removal in hand."""
        if not HOLD_REMOVAL_MANA:
            return 0
        held = [c for c in self.hand(g) if c.name in REMOVAL]
        if not held:
            return 0
        threat_now = any(self.threat(g, c) >= REMOVAL_MIN_THREAT for c in self.creatures(g, self.opp))
        if g.turn < HOLD_FROM_TURN and not threat_now:
            return 0
        return min(self.cost(g, c) for c in held)

    def breaks_reserve(self, g: Game, card: Card) -> bool:
        """Casting `card` now would leave too little open for our removal.
        Our two best threats are still allowed when nothing needs killing yet."""
        reserve = self.removal_reserve(g)
        if not reserve or card.name in REMOVAL:
            return False
        threat_now = any(self.threat(g, c) >= REMOVAL_MIN_THREAT for c in self.creatures(g, self.opp))
        if not threat_now and CURVE.get(card.name, 0) >= 8:
            return False
        return self.mana_available(g) - self.cost(g, card) < reserve

    BLUE_SPELL_VALUE = {"Tolarian Terror": 6, "Cryptic Serpent": 6, "Delver of Secrets": 3, "Lorien Revealed": 2.5, "Brainstorm": 1.5}

    def blue_spell_value(self, g: Game, it) -> float:
        """How much countering this Blue spell is worth."""
        if it.kind != "spell" or it.controller == self.p:
            return NEG
        protects = [t for t in it.targets if t[0] == "stack" and (x := g.stack_item(t[1])) is not None and x.controller == self.p]
        if protects:  # their counter aimed at our spell
            return 5.0
        return self.BLUE_SPELL_VALUE.get(it.name, 1.0)

    def modal_score(self, g: Game, card: Card, mode: str) -> float:
        if card.name == "Red Elemental Blast":
            if mode == "counter":
                blue = [it for it in g.stack if it.kind == "spell" and "U" in it.card.face.colors]
                best = max((self.blue_spell_value(g, it) for it in blue), default=NEG)
                return 35.0 if best >= 2.5 else NEG
            targets = [c for c in g.battlefield if c.controller == self.opp and "U" in c.face.colors and self.threat(g, c) >= 3]
            if not targets:
                return NEG
            pressured = any(c.oid in g.attackers for c in targets) or self.me(g).life < PRESSURE_LIFE
            if self.blue_counter_up(g) and not pressured and not self.opp_end_step(g):
                return NEG
            return 28.0
        return NEG

    def combat_trick_target(self, g: Game) -> Card | None:
        """My creature in combat with a bigger creature that it would not kill."""
        for boid, aoid in g.blocks.items():
            b, a = g.perm(boid), g.perm(aoid)
            if a is None or b is None:
                continue
            mine, theirs = (b, a) if b.controller == self.p else (a, b)
            if g.power(mine) > 0 and g.power(mine) < g.toughness(theirs) - theirs.damage and self.threat(g, theirs) >= 3:
                return mine
        return None

    def activate_score(self, g: Game, card: Card, ability: str) -> float:
        n = card.name
        if n == "Krark-Clan Shaman":
            return self.sweep_value(g) if self.cheap_artifact(g) else NEG
        if n == "Makeshift Munitions":
            if not self.cheap_fodder(g):
                return NEG
            if g.players[self.opp].life <= 2:
                return 20.0
            return 8.0 if any(self.pings_dead(g, c) for c in self.creatures(g, self.opp)) else NEG
        if n == "Nihil Spellbomb":
            gy = g.players[self.opp].graveyard
            fuel = sum(1 for c in gy if c.face.is_type("Instant") or c.face.is_type("Sorcery"))
            escape = any(c.name == "Sleep of the Dead" for c in gy) and len(gy) >= 4
            return 15.0 if fuel >= SPELLBOMB_GY_FUEL or escape else NEG
        if n == "Lembas":
            return 10.0 if self.me(g).life <= LEMBAS_LIFE and self.opp_end_step(g) else NEG
        if n == "Clue":
            return 3.0 if self.opp_end_step(g) else NEG
        if n == "Map":
            return 2.0 if self.main_phase(g) and g.step_name == "main2" and self.creatures(g, self.p) else NEG
        if n == "Twisted Landscape":
            if "cycling" in ability:
                return 3.0 if self.opp_end_step(g) and self.land_need(g) < -1 else NEG
            return 3.0 if self.opp_end_step(g) else NEG
        return NEG

    def cheap_artifact(self, g: Game) -> bool:
        return any(self.sac_cost(g, c) <= 2.0 for c in g.sac_candidates(self.p, "artifact"))

    def cheap_fodder(self, g: Game) -> bool:
        return any(self.sac_cost(g, c) <= 2.0 for c in g.sac_candidates(self.p, "artifact_or_creature"))

    def pings_dead(self, g: Game, c: Card) -> bool:
        return g.toughness(c) - c.damage <= 1 and self.threat(g, c) >= 2

    def sweep_value(self, g: Game) -> float:
        """Krark-Clan Shaman: 1 damage to each creature without flying."""
        v = 0.0
        for c in g.battlefield:
            if not g.is_creature(c) or g.has(c, "flying") or g.toughness(c) - c.damage > 1:
                continue
            v += self.threat(g, c) if c.controller == self.opp else -self.creature_value(g, c)
        return v * 3 if v > 1 else NEG

    def colour_needs(self, g: Game) -> dict[str, float]:
        """Coloured pips the rest of our hand wants; instants count double
        because they are cast on Blue's turn from whatever is left untapped."""
        need: dict[str, float] = {}
        for c in self.hand(g):
            if c.face.is_type("Land"):
                continue
            w = 2.0 if c.face.is_type("Instant") else 1.0
            for col, n in c.face.cost.colored:
                need[col] = need.get(col, 0.0) + w * n
        return need

    def source_cost(self, g: Game, card: Card, color: str) -> float:
        if card.is_token:
            return 5.0
        colours = next((ab.mana for ab in card.face.abilities if ab.mana), ())
        need = self.colour_needs(g)
        supply: dict[str, int] = {}
        for c, ab in g.mana_sources(self.p):
            for col in ab.mana:
                supply[col] = supply.get(col, 0) + 1
        hurt = sum(need.get(col, 0.0) / max(supply.get(col, 1), 1) for col in colours)
        return hurt + 0.1 * len(colours)

    def sac_cost(self, g: Game, c: Card) -> float:
        if g.is_land(c) and not c.is_token:
            n = len(self.lands(g, self.p))
            if n >= SPARE_LAND_COUNT:
                return 1.5 if c.tapped or not g.mana_sources(self.p) else 2.0
        if c.name == "Ichor Wellspring":
            return 0.2  # draws a card when it dies
        if c.name in ("Nihil Spellbomb", "Lembas", "Clue", "Map"):
            return 0.6
        if c.name == "Krark-Clan Shaman":
            return 2.5
        return super().sac_cost(g, c)

    # -- targets ------------------------------------------------------------

    def score_target(self, g: Game, d: Decision, o: Option) -> float:
        spell = self.building(g, d)
        c = self.ref_card(g, o)
        v = o.value
        if spell == "Red Elemental Blast":
            it = self.ref_spell(g, o)
            if it is not None:
                return self.blue_spell_value(g, it)
            if c is None or c.controller == self.p:
                return NEG
            return self.threat(g, c) - (20 if c.face.ward and self.mana_available(g) < c.face.ward else 0)
        if spell in ("Cast Down", "Go for the Throat", "Makeshift Munitions"):
            if spell == "Makeshift Munitions" and v[0] == "player":
                return 5.0 if v[1] == self.opp and g.players[self.opp].life <= 2 else NEG
            if c is None or c.controller == self.p:
                return NEG
            s = self.threat(g, c)
            if spell == "Makeshift Munitions" and not self.pings_dead(g, c):
                s -= 10
            if c.face.ward and self.mana_available(g) < c.face.ward + 1:
                s -= 20
            return s
        if spell == "Cleansing Wildfire":
            if c is None:
                return NEG
            if c.controller == self.p:
                return 10.0 if c.name in BRIDGES and c.name != "Vault of Whispers" else -5.0
            return 2.0 + (3.0 if len(self.lands(g, self.opp)) <= 3 else 0)
        if spell == "Toxin Analysis":
            tgt = self.combat_trick_target(g)
            if c is None or c.controller != self.p:
                return NEG
            return 10.0 if tgt is not None and c.oid == tgt.oid else self.creature_value(g, c)
        if spell in ("Nihil Spellbomb", "Nihil Spellbomb: exile target player's graveyard"):
            return 1.0 if v == ("player", self.opp) else NEG
        if spell == "Nyxborn Hydra":
            return self.creature_value(g, c) if c is not None and c.controller == self.p else NEG
        if spell == "Map":
            return self.creature_value(g, c) if c is not None and c.controller == self.p else NEG
        # Unknown: harm the opponent's best thing.
        if c is not None:
            return self.creature_value(g, c) * (1 if c.controller == self.opp else -1)
        return 1.0 if v == ("player", self.opp) else 0.0
