"""Elves bot: ramp into a wide board, then pump and trample over.

Plan: a land every turn, mana Elves and Priest of Titania first, then the
rest of the curve (Avenging Hunter takes the initiative, Generous Ent, Nyxborn
Hydra with the biggest X). Winding Way and Lead the Stampede refill the hand.
Timberwatch Elf pumps an unblocked (or blocked and losing) attacker after
blocks; Quirion Ranger returns a tapped Forest to untap Timberwatch for a
second pump, or a tapped mana Elf in a main phase when the Forest can be
replayed. Masked Vandal eats an artifact or enchantment when it can.
"""

from __future__ import annotations

from ..engine.cards import count_of
from ..engine.game import Game
from ..engine.objects import Card, Decision, Option
from .base import NEG, Bot

KEEP_LANDS = {7: (1, 4), 6: (1, 4)}
MANA_ELVES = ("Llanowar Elves", "Fyndhorn Elves", "Elvish Mystic")
# Main-phase cast priority (higher first).
CURVE = {
    "Priest of Titania": 9.5,
    "Llanowar Elves": 9,
    "Fyndhorn Elves": 9,
    "Elvish Mystic": 9,
    "Avenging Hunter": 8.5,
    "Generous Ent": 7.5,
    "Spinewoods Paladin": 7.5,
    "Timberwatch Elf": 6.5,
    "Quirion Ranger": 6,
    "Sagu Wildling": 5.5,
    "Vitu-Ghazi Inspector": 5,
    "Masked Vandal": 4,
}


class ElvesBot(Bot):
    name = "elves"

    # -- valuations ---------------------------------------------------------

    def card_value(self, g: Game, name: str) -> float:
        card = g.cards_db[name]
        if card.is_type("Land"):
            return 4.0 if self.land_need(g) > 0 else 1.0
        return {
            "Priest of Titania": 5,
            "Avenging Hunter": 5,
            "Timberwatch Elf": 4,
            "Generous Ent": 4,
            "Lead the Stampede": 3.5,
            "Winding Way": 3.5,
            "Nyxborn Hydra": 3.5,
            "Quirion Ranger": 3,
            "Masked Vandal": 3,
            "Llanowar Elves": 3,
            "Fyndhorn Elves": 3,
            "Elvish Mystic": 3,
            "Sagu Wildling": 3,
            "Monstrous Emergence": 3,
            "Spinewoods Paladin": 4,
            "Vitu-Ghazi Inspector": 2.5,
            "Deglamer": 2,
            "Faerie Macabre": 2,
        }.get(name, 1.0)

    def land_need(self, g: Game) -> float:
        have = len(self.lands(g, self.p)) + sum(1 for c in self.hand(g) if c.face.is_type("Land"))
        dorks = sum(1 for c in self.creatures(g, self.p) if c.name in MANA_ELVES or c.name == "Priest of Titania")
        return 4.0 - have - 0.5 * dorks

    def keep_hand(self, g: Game, size: int) -> bool:
        if size <= 5:
            return True
        hand = self.hand(g)
        lands = sum(1 for c in hand if c.face.is_type("Land"))
        dorks = sum(1 for c in hand if c.name in MANA_ELVES or c.name == "Priest of Titania")
        lo, hi = KEEP_LANDS[size]
        return lo <= lands <= hi and lands + dorks >= 2

    def mana_available(self, g: Game, who: int | None = None) -> int:
        who = self.p if who is None else who
        return sum(g.mana_units(ab) for _, ab in g.mana_sources(who)) + sum(g.players[who].pool.values())

    def elves(self, g: Game) -> int:
        return count_of(g, "elves")

    def opp_artifacts(self, g: Game) -> list[Card]:
        return [c for c in g.battlefield if c.controller == self.opp and (g.is_artifact(c) or c.face.is_type("Enchantment"))]

    def artifact_value(self, g: Game, c: Card) -> float:
        if g.is_creature(c):
            return self.creature_value(g, c)
        if g.is_land(c):
            return 3.0
        return 0.5 if c.is_token else 1.5

    def gy_targets(self, g: Game) -> list[Card]:
        """Opponent graveyard cards worth exiling: fuel for Tolarian Terror and
        Cryptic Serpent, escape and flashback cards, Sneaky Snacker."""
        out = []
        for c in g.players[self.opp].graveyard:
            f = c.face
            if f.is_type("Instant") or f.is_type("Sorcery") or f.escape is not None or f.flashback is not None or any(t.event == "third_draw" for t in f.triggers):
                out.append(c)
        return out

    def unblocked_attackers(self, g: Game) -> list[Card]:
        out = []
        for oid in g.attackers:
            c = g.perm(oid)
            if c is not None and c.controller == self.p and oid not in g.blocked:
                out.append(c)
        return out

    # -- priority -----------------------------------------------------------

    def score_priority(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] == "pass":
            return 0.0
        if k[0] == "play_land":
            if k[1] == "Gingerbread Cabin":
                others = sum(1 for c in self.lands(g, self.p) if "Forest" in c.face.subtypes)
                return 51.0 if others >= 3 else 49.0
            return 50.0
        if k[0] == "plot":
            return 2.0 if g.step_name == "main2" and self.mana_available(g) >= 4 else NEG
        if k[0] == "cast":
            return self.cast_score(g, o.value[1], k[3])
        if k[0] == "activate":
            return self.activate_score(g, o.value[1], k[2])
        return NEG

    def cast_score(self, g: Game, card: Card, mode: str) -> float:
        n = card.name
        main = self.main_phase(g)
        if n == "Deglamer":
            best = max((self.artifact_value(g, c) for c in self.opp_artifacts(g)), default=0)
            return 8.0 if best >= 1.5 and (main or self.opp_end_step(g)) else NEG
        if not main:
            return NEG
        if n == "Sagu Wildling" and mode == "omen":
            return 6.5 if self.land_need(g) > 0 else NEG
        if n in ("Winding Way", "Lead the Stampede"):
            return 7.0 if len(self.hand(g)) <= 5 else 3.0
        if n == "Monstrous Emergence":
            power = max((g.power(c) for c in self.creatures(g, self.p)), default=0)
            power = max([power] + [c.face.power or 0 for c in self.hand(g) if c.face.is_type("Creature")])
            kills = [c for c in self.creatures(g, self.opp) if g.toughness(c) - c.damage <= power and self.creature_value(g, c) >= 2]
            return 12.0 if kills else NEG
        if n == "Nyxborn Hydra":
            if mode != "normal":
                return NEG
            return 5.0 if self.mana_available(g) >= 4 else NEG
        if n == "Masked Vandal":
            can_eat = self.opp_artifacts(g) and any(c.face.is_type("Creature") for c in self.me(g).graveyard)
            return 8.0 if can_eat else CURVE[n]
        if n == "Vitu-Ghazi Inspector":
            return CURVE[n] + (1.5 if mode == "evidence" else 0)
        if n in CURVE and mode in ("normal", "plot"):
            return CURVE[n] + 0.1 * self.cost(g, card, mode)
        return NEG

    def activate_score(self, g: Game, card: Card, ability: str) -> float:
        n = card.name
        if n == "Timberwatch Elf":
            if self.my_turn(g) and g.step_name == "declare_blockers" and g.attackers:
                return 10.0
            if not self.my_turn(g) and g.step_name == "declare_blockers" and g.blocks:
                return 9.0  # a blocker wins its fight
            return NEG
        if n == "Quirion Ranger":
            tapped_tw = [c for c in self.creatures(g, self.p) if c.name == "Timberwatch Elf" and c.tapped and not c.sick]
            if tapped_tw and self.my_turn(g) and g.step_name == "declare_blockers" and self.unblocked_attackers(g):
                return 9.5
            # Main phase: a tapped Forest goes back and is replayed untapped.
            tapped_forest = any(c.tapped and "Forest" in c.face.subtypes for c in self.lands(g, self.p))
            has_land = any(c.face.is_type("Land") for c in self.hand(g))
            tapped_dork = any(c.tapped and not c.sick and (c.name in MANA_ELVES or c.name == "Priest of Titania") for c in self.creatures(g, self.p))
            if self.main_phase(g) and g.step_name == "main1" and g.lands_played == 0 and tapped_forest and not has_land and tapped_dork:
                return 8.0
            return NEG
        if n == "Generous Ent":  # forestcycling
            if self.land_need(g) > 0 and self.mana_available(g) < 6 and (self.main_phase(g) or self.opp_end_step(g)):
                return 4.0
            return NEG
        if n == "Faerie Macabre":  # never cast (no black mana): discard it as graveyard hate
            return 6.0 if len(self.gy_targets(g)) >= 2 else NEG
        if n == "Food":
            return 2.0 if self.opp_end_step(g) and self.me(g).life <= 12 else NEG
        if n == "Treasure":
            return NEG
        return NEG

    # -- targets and choices ------------------------------------------------

    def score_target(self, g: Game, d: Decision, o: Option) -> float:
        name = self.building(g, d)
        c = self.ref_card(g, o)
        if name == "Timberwatch Elf":
            if c is None or c.controller != self.p:
                return NEG
            if c in self.unblocked_attackers(g):
                return 10.0 + g.power(c)
            if c.oid in g.attackers or c.oid in g.blocks:
                return 5.0 + self.creature_value(g, c)
            return 0.0
        if name == "Quirion Ranger":
            if c is None or c.controller != self.p:
                return NEG
            if c.name == "Timberwatch Elf" and c.tapped:
                return 10.0
            if c.name == "Priest of Titania" and c.tapped:
                return 8.0
            return 3.0 if c.tapped else -1.0
        if name in ("Masked Vandal", "Deglamer"):
            if c is None or c.controller == self.p:
                return NEG
            return self.artifact_value(g, c)
        if name == "Monstrous Emergence":
            if c is None or c.controller == self.p:
                return NEG
            return self.creature_value(g, c)
        if name in ("Vitu-Ghazi Inspector", "Nyxborn Hydra"):
            if c is None or c.controller != self.p:
                return NEG
            return self.creature_value(g, c)
        if c is not None:
            return self.creature_value(g, c) * (1 if c.controller == self.opp else -1)
        return 1.0 if o.value == ("player", self.opp) else 0.0

    def score_choose_mode(self, g: Game, d: Decision, o: Option) -> float:
        tag, ans = o.key[0], o.key[1]
        if tag == "choose_type":
            want_land = self.land_need(g) > 0.5
            return 1.0 if (ans == "Land") == want_land else 0.0
        if tag == "venture":
            # Forge -> Trap! (5 to the face) with a creature to grow, else Lost Well.
            has_creature = bool(self.creatures(g, self.p))
            return 1.0 if (ans == "Forge") == has_creature else 0.0
        return super().score_choose_mode(g, d, o)

    def score_choose_card(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] == "choose_power":
            c = o.value
            return g.power(c) if k[1] == "battlefield" else (c.face.power or 0) + 0.5  # revealing keeps the board
        return super().score_choose_card(g, d, o)

    def score_exile_from_graveyard(self, g: Game, d: Decision, o: Option) -> float:
        c = o.value
        if o.key[0] == "exile_any_gy":  # Faerie Macabre
            if c is None:
                return 0.0
            return (2.0 if c in self.gy_targets(g) else 0.5) if o.key[1] == "opponent" else -1.0
        if self.building(g, d) == "Masked Vandal" or d.prompt.startswith("Masked Vandal"):
            if c is None:
                return 0.0
            return 5.0 - self.card_value(g, c.name)  # cheapest creature card goes
        return super().score_exile_from_graveyard(g, d, o)

    def score_sacrifice(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[0] == "return_land":  # Quirion Ranger's cost: a tapped Forest first
            return 1.0 if o.value.tapped else 0.0
        return super().score_sacrifice(g, d, o)

    def source_cost(self, g: Game, card: Card, color: str) -> float:
        if card.name == "Priest of Titania":
            return 0.5 if self.elves(g) >= 3 else 2.0
        if g.is_creature(card) and not card.is_token:
            return 1.5  # lands first: Elves may attack or chump
        return super().source_cost(g, card, color)
