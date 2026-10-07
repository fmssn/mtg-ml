"""Red Madness bot: burn the opponent out.

Plan: a land every turn, creatures on curve (Guttersnipe and Kessig
Flamebreather turn every spell into extra damage, so they come first),
cheap burn kills creatures that block or threaten us and otherwise goes to
the face, at the latest in the opponent's end step. Card flow (Faithless
Looting, Grab the Prize, Highway Robbery, Blood) feeds Fiery Temper's
madness and Sneaky Snacker to the graveyard. Fireblast and the Lava Dart
flashback sacrifice Mountains only when they finish the game or the bot has
lands to spare.
"""

from __future__ import annotations

from ..engine.game import Game
from ..engine.objects import Card, Decision, Option
from .base import NEG, Bot

# Tunable judgment calls.
KEEP_LANDS = {7: (1, 4), 6: (1, 4)}
SPARE_LANDS = 5  # with this many lands, a Mountain is fodder (Lava Dart, Robbery, Fireblast)
BURN_FACE_LIFE = 8  # at or below this opponent life, every burn spell goes to the face
KILL_VALUE = 3.0  # creature_value worth a burn spell
DAMAGE = {"Lightning Bolt": 3, "Fiery Temper": 3, "Fireblast": 4, "Lava Dart": 1}
CURVE = {"Guttersnipe": 9, "Kessig Flamebreather": 8, "Voldaren Epicure": 6, "Gorilla Shaman": 3, "Martyr of Ashes": 2, "Relic of Progenitus": 2}
DISCARD_FIRST = ("Fiery Temper", "Sneaky Snacker")  # cards that want to be discarded


class RedBot(Bot):
    name = "red"

    # -- valuations ---------------------------------------------------------

    def card_value(self, g: Game, name: str) -> float:
        card = g.cards_db[name]
        if card.is_type("Land"):
            return 4.0 if self.land_need(g) > 0 else 0.8
        if name == "Fiery Temper":
            return 0.2 if self.can_pay_madness(g) else 4.0
        return {
            "Lightning Bolt": 6,
            "Fireblast": 5,
            "Guttersnipe": 4.5,
            "Kessig Flamebreather": 4,
            "Grab the Prize": 3,
            "Lava Dart": 3,
            "Faithless Looting": 2.5,
            "Voldaren Epicure": 2.5,
            "Highway Robbery": 2,
            "Searing Blaze": 3.5,
            "Electrickery": 3,
            "Pyroblast": 3,
            "Red Elemental Blast": 3,
            "Cast into the Fire": 3,
            "End the Festivities": 2.5,
            "Gorilla Shaman": 2,
            "Martyr of Ashes": 2,
            "Relic of Progenitus": 1.5,
            "Sneaky Snacker": 0.3,
        }.get(name, 1.0)

    def land_need(self, g: Game) -> float:
        have = len(self.lands(g, self.p)) + sum(1 for c in self.hand(g) if c.face.is_type("Land"))
        return 3.5 - have

    def keep_hand(self, g: Game, size: int) -> bool:
        if size <= 5:
            return True
        lands = sum(1 for c in self.hand(g) if c.face.is_type("Land"))
        lo, hi = KEEP_LANDS[size]
        return lo <= lands <= hi

    def can_pay_madness(self, g: Game) -> bool:
        return any("R" in ab.mana for _, ab in g.mana_sources(self.p)) or self.me(g).pool.get("R", 0) > 0

    def opp_life(self, g: Game) -> int:
        return g.players[self.opp].life

    def mountains(self, g: Game) -> int:
        return len(g.sac_candidates(self.p, "mountain"))

    def reach(self, g: Game) -> int:
        """Burn damage the hand can deal this turn, roughly (Fireblast free)."""
        mana = self.mana_available(g)
        dmg = 0
        for c in sorted(self.hand(g), key=lambda c: c.face.mana_value):
            if c.name == "Fireblast" and self.mountains(g) >= 2:
                dmg += 4
            elif c.name in DAMAGE and c.name != "Fireblast" and self.cost(g, c) <= mana:
                mana -= self.cost(g, c)
                dmg += DAMAGE[c.name]
        dmg += sum(1 for c in self.me(g).graveyard if c.name == "Lava Dart")
        pings = sum(2 if c.name == "Guttersnipe" else 1 for c in self.creatures(g, self.p) if c.name in ("Guttersnipe", "Kessig Flamebreather"))
        spells = sum(1 for c in self.hand(g) if c.name in DAMAGE)
        return dmg + pings * spells

    def going_face(self, g: Game) -> bool:
        return self.opp_life(g) <= BURN_FACE_LIFE or self.reach(g) >= self.opp_life(g)

    def kill_targets(self, g: Game, dmg: int) -> list[Card]:
        return [c for c in self.creatures(g, self.opp) if g.toughness(c) - c.damage <= dmg and self.creature_value(g, c) >= KILL_VALUE]

    def fodder(self, g: Game, exclude: Card | None = None) -> list[Card]:
        """Hand cards we are happy to discard."""
        out = []
        lands = len(self.lands(g, self.p))
        for c in self.hand(g):
            if c is exclude:
                continue
            if c.name in DISCARD_FIRST or (c.face.is_type("Land") and lands >= 3):
                out.append(c)
        return out

    # -- priority -----------------------------------------------------------

    def score_priority(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] == "pass":
            return 0.0
        if k[0] == "play_land":
            return 50.0
        if k[0] == "plot":
            # Plot with mana we would not use otherwise: our main 2, nothing to cast.
            return 1.5 if g.step_name == "main2" and self.mana_available(g) >= 2 else NEG
        if k[0] == "cast":
            card = o.value[1]
            if len(k) > 4:
                return self.modal_score(g, card, k[4])
            return self.cast_score(g, card, k[3])
        if k[0] == "activate":
            return self.activate_score(g, o.value[1], k[2])
        return NEG

    def end_of_their_turn(self, g: Game) -> bool:
        return self.opp_end_step(g)

    def cast_score(self, g: Game, card: Card, mode: str) -> float:
        n = card.name
        main = self.main_phase(g)
        if n in DAMAGE:
            return self.burn_score(g, card, mode)
        if n == "Faithless Looting":
            if not main:
                return NEG
            if mode == "flashback":
                return 2.0 if g.step_name == "main2" and self.fodder(g) else NEG
            return 7.0 if self.fodder(g, exclude=card) or len(self.hand(g)) <= 2 else NEG
        if n == "Grab the Prize":
            return 7.5 if main and self.fodder(g, exclude=card) else (3.0 if main and g.step_name == "main2" and len(self.hand(g)) > 1 else NEG)
        if n == "Highway Robbery":
            return 5.0 if main else NEG
        if n == "Searing Blaze":
            targets = [c for c in self.creatures(g, self.opp) if g.toughness(c) - c.damage <= (3 if self.me(g).landfall_turn == g.turn else 1)]
            return 20.0 if targets and (main or g.step_name in ("declare_attackers", "declare_blockers")) else NEG
        if n == "End the Festivities":
            if not main:
                return NEG
            dead = [c for c in self.creatures(g, self.opp) if g.toughness(c) - c.damage <= 1]
            value = sum(self.creature_value(g, c) for c in dead)
            if len(dead) >= 2 or value >= KILL_VALUE:
                return 14.0 + value
            return 4.0 if self.going_face(g) else NEG
        if n == "Electrickery":
            small = [c for c in self.creatures(g, self.opp) if g.toughness(c) - c.damage <= 1]
            if mode == "overload":
                return 15.0 + len(small) if len(small) >= 2 else NEG
            return 12.0 if small and self.creature_value(g, max(small, key=lambda c: self.creature_value(g, c))) >= 1.5 else NEG
        if not main:
            return NEG
        if n in CURVE:
            return CURVE[n] + 0.1 * self.cost(g, card)
        return NEG

    def burn_score(self, g: Game, card: Card, mode: str) -> float:
        n = card.name
        dmg = DAMAGE[n]
        if n == "Fireblast" and mode == "alternative":
            if self.opp_life(g) <= self.reach(g) or (self.opp_life(g) <= dmg + 2 and self.mountains(g) >= 2):
                return 40.0
            return NEG
        if mode == "flashback":  # Lava Dart: sacrifice a Mountain
            if self.opp_life(g) <= self.reach(g) or (self.mountains(g) >= SPARE_LANDS and (self.kill_targets(g, 1) or self.end_of_their_turn(g))):
                return 8.0
            return NEG
        if self.opp_life(g) <= self.reach(g):
            return 40.0
        if self.kill_targets(g, dmg):
            return 25.0
        if self.going_face(g) or self.end_of_their_turn(g):
            return 10.0
        return NEG

    def pingable(self, g: Game) -> list[Card]:
        return [c for c in self.creatures(g, self.opp) if g.toughness(c) - c.damage <= 1 and self.creature_value(g, c) >= 1.5]

    def modal_score(self, g: Game, card: Card, mode: str) -> float:
        if card.name == "Cast into the Fire":
            if mode == "exile":
                arts = [c for c in g.battlefield if c.controller == self.opp and g.is_artifact(c) and (c.face.mana_value >= 2 or g.is_creature(c))]
                return 10.0 if arts and (self.main_phase(g) or self.end_of_their_turn(g)) else NEG
            small = self.pingable(g)
            if mode == "two creatures":
                return 16.0 if len(small) >= 2 else NEG
            return 12.0 if small and self.creature_value(g, max(small, key=lambda c: self.creature_value(g, c))) >= KILL_VALUE else NEG
        if card.name not in ("Pyroblast", "Red Elemental Blast"):
            return NEG
        if mode == "counter":
            top = self.stack_top(g)
            return 30.0 if top is not None and top.kind == "spell" and top.controller == self.opp and "U" in top.card.face.colors else NEG
        blue = [c for c in self.creatures(g, self.opp) if "U" in c.face.colors and self.creature_value(g, c) >= KILL_VALUE]
        return 20.0 if blue else NEG

    def activate_score(self, g: Game, card: Card, ability: str) -> float:
        if card.name == "Blood":
            return 3.0 if self.fodder(g) and (self.end_of_their_turn(g) or g.step_name == "main2") else NEG
        if card.name == "Gorilla Shaman":
            return 6.0 if self.end_of_their_turn(g) or g.step_name == "main2" else NEG
        if card.name == "Martyr of Ashes":
            n = sum(1 for c in self.hand(g) if "R" in c.face.colors)
            dead = [c for c in self.creatures(g, self.opp) if not g.has(c, "flying") and g.toughness(c) - c.damage <= n]
            mine = [c for c in self.creatures(g, self.p) if c is not card and not g.has(c, "flying") and g.toughness(c) - c.damage <= n]
            return 9.0 if len(dead) >= 2 + len(mine) else NEG
        if card.name == "Relic of Progenitus":
            gy = g.players[self.opp].graveyard
            if ability.startswith("exile all"):
                return 4.0 if self.end_of_their_turn(g) and len(gy) >= 5 else NEG
            return 2.0 if self.end_of_their_turn(g) and gy else NEG
        return NEG

    # -- targets and choices ------------------------------------------------

    def score_target(self, g: Game, d: Decision, o: Option) -> float:
        name = self.building(g, d)
        v = o.value
        if name in DAMAGE or name == "Searing Blaze":
            dmg = DAMAGE.get(name, 3 if self.me(g).landfall_turn == g.turn else 1)
            if v == ("player", self.opp):
                lethal = self.opp_life(g) <= dmg
                return 100.0 if lethal else (5.0 if self.going_face(g) or not self.kill_targets(g, dmg) else 1.0)
            if v[0] == "player":
                return NEG
            c = self.ref_card(g, o)
            if c is None or c.controller == self.p:
                return NEG
            dies = g.toughness(c) - c.damage <= dmg
            return (2.0 + self.creature_value(g, c)) if dies else -1.0
        if name == "Gorilla Shaman":
            c = self.ref_card(g, o)
            return c.face.mana_value + (1 if not c.is_token else 0) if c is not None else NEG
        if name in ("Pyroblast", "Red Elemental Blast"):
            it = self.ref_spell(g, o)
            if it is not None:
                return 5.0 if it.controller == self.opp and "U" in it.card.face.colors else NEG
            c = self.ref_card(g, o)
            if c is None or c.controller == self.p or "U" not in c.face.colors:
                return NEG
            return self.creature_value(g, c) if g.is_creature(c) else 0.0
        if name == "Cast into the Fire":
            c = self.ref_card(g, o)
            if c is None or c.controller == self.p:
                return NEG
            if d.prompt.startswith("Choose target (artifact)"):  # exile mode
                return c.face.mana_value + (self.creature_value(g, c) if g.is_creature(c) else 0)
            return (2.0 + self.creature_value(g, c)) if g.toughness(c) - c.damage <= 1 else -1.0
        if name == "Electrickery":
            c = self.ref_card(g, o)
            if c is None or c.controller == self.p:
                return NEG
            return self.creature_value(g, c) if g.is_creature(c) else 0.0
        if name == "Relic of Progenitus":
            return 1.0 if v == ("player", self.opp) else 0.0
        return 0.0

    def score_choose_card(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] == "robbery":
            if k[1] == "none":
                return 0.0
            if k[1] == "discard":
                return 3.0 - self.card_value(g, k[2])
            return 2.0 - self.sac_cost(g, o.value[1])  # sacrifice a land
        if k[0] == "reveal":
            return 0.0
        return super().score_choose_card(g, d, o)

    def sac_cost(self, g: Game, c: Card) -> float:
        if g.is_land(c):
            return (3.0 if len(self.lands(g, self.p)) >= SPARE_LANDS else 8.0) - (0.5 if c.tapped else 0.0)
        return super().sac_cost(g, c)

    def score_yes_no(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[0] == "madness":
            return 1.0 if o.key[1] == "cast" else 0.0
        return super().score_yes_no(g, d, o)

    def score_order_triggers(self, g: Game, d: Decision, o: Option) -> float:
        return 1.0 if o.key[2] == "madness" else 0.0  # madness first: resolves last, after the damage triggers
