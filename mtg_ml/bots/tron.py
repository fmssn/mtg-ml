"""Tron bot: assemble Urza's Tron, then cast the big threats.

Plan: lands first, a missing Tron piece over everything else; Expedition
Map, Ancient Stirrings and Crop Rotation dig for the missing pieces (Crop
Rotation turns a spare land into the last piece, at instant speed, as soon
as it completes Tron). Cheap artifacts early (Barrels, Candy Trail, Bonder's
Ornament: mana fixing and card flow), then the threats, biggest first:
Maelstrom Colossus (cascade), Pinnacle Kill-Ship (10 damage to their best
creature), Bramble Wurm, Boulderbranch Golem (prototyped when seven mana is
far away), Generous Ent (forestcycled when a land is needed more). Card
draw (Unfathomable Truths, Candy Trail, Bonder's Ornament) waits for the
opponent's end step. A summoning-sick creature stations the Kill-Ship.
"""

from __future__ import annotations

from ..engine.game import Game
from ..engine.objects import Card, Decision, Option
from .base import NEG, Bot

TRON = ("Mine", "Power-Plant", "Tower")
TRON_LANDS = {"Mine": "Urza's Mine", "Power-Plant": "Urza's Power Plant", "Tower": "Urza's Tower"}
KEEP_LANDS = {7: (2, 5), 6: (1, 4)}
DIGGERS = ("Expedition Map", "Ancient Stirrings", "Crop Rotation")
THREATS = {  # main-phase cast priority once affordable
    "Maelstrom Colossus": 30,
    "Pinnacle Kill-Ship": 28,
    "Bramble Wurm": 26,
    "Boulderbranch Golem": 24,
    "Generous Ent": 22,
}
CHEAP = {"Expedition Map": 9, "Barrels of Blasting Jelly": 7, "Candy Trail": 8, "Bonder's Ornament": 6, "Relic of Progenitus": 3}
FORESTS_IN_DECK = 2
KILL_VALUE = 3.0  # creature_value worth a removal effect
LOW_LIFE = 8


class TronBot(Bot):
    name = "tron"

    # -- state ----------------------------------------------------------------

    def tron_have(self, g: Game) -> set[str]:
        return {s for c in self.lands(g, self.p) for s in TRON if s in c.face.subtypes}

    def tron_missing(self, g: Game) -> list[str]:
        have = self.tron_have(g)
        return [s for s in TRON if s not in have]

    def missing_in_hand(self, g: Game) -> list[str]:
        """Missing Tron pieces that are not in hand either."""
        hand = {s for c in self.hand(g) for s in TRON if s in c.face.subtypes}
        return [s for s in self.tron_missing(g) if s not in hand]

    def has_tron(self, g: Game) -> bool:
        return not self.tron_missing(g)

    def mana(self, g: Game, exclude: frozenset = frozenset()) -> int:
        """Mana available now, counting Tron lands' extra mana."""
        total = sum(self.me(g).pool.values())
        for c, ab in g.mana_sources(self.p, exclude):
            total += g.mana_amount(c, ab)
        return total

    def opp_threats(self, g: Game) -> list[Card]:
        return sorted(self.creatures(g, self.opp), key=lambda c: -self.creature_value(g, c))

    def end_of_their_turn(self, g: Game) -> bool:
        return self.opp_end_step(g)

    # -- valuations -------------------------------------------------------------

    def card_value(self, g: Game, name: str) -> float:
        card = g.cards_db[name]
        if card.is_type("Land"):
            if any(s in card.subtypes for s in self.missing_in_hand(g)):
                return 9.0
            return 4.0 if self.land_need(g) > 0 else 1.0
        if name in THREATS:
            return 5.0 + THREATS[name] / 10
        return {
            "Expedition Map": 4.5 if self.missing_in_hand(g) else 1.5,
            "Ancient Stirrings": 4.0,
            "Crop Rotation": 3.5 if self.tron_missing(g) else 1.0,
            "Unfathomable Truths": 4.0,
            "Barrels of Blasting Jelly": 2.5,
            "Candy Trail": 2.5,
            "Bonder's Ornament": 2.0,
            "Scour from Existence": 4.0,
            "Breath Weapon": 3.5,
            "Blue Elemental Blast": 3.0,
            "Call Damage Control": 2.5,
            "Pulse of Murasa": 3.0,
            "Monstrous Emergence": 3.0,
            "Kaervek's Torch": 3.0,
            "Relic of Progenitus": 1.5,
        }.get(name, 1.0)

    def land_need(self, g: Game) -> float:
        have = len(self.lands(g, self.p)) + sum(1 for c in self.hand(g) if c.face.is_type("Land"))
        return 7 - have

    def keep_hand(self, g: Game, size: int) -> bool:
        if size <= 5:
            return True
        hand = self.hand(g)
        lands = sum(1 for c in hand if c.face.is_type("Land"))
        diggers = sum(1 for c in hand if c.name in DIGGERS)
        lo, hi = KEEP_LANDS[size]
        return lo <= lands <= hi or (lands == 1 and diggers >= 2)

    # -- priority ----------------------------------------------------------------

    def score_priority(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] == "pass":
            return 0.0
        if k[0] == "play_land":
            return self.land_score(g, o.value[1])
        if k[0] == "cast":
            card = o.value[1]
            if len(k) > 4:
                return self.modal_score(g, card, k[4])
            return self.cast_score(g, card, k[3])
        if k[0] == "activate":
            return self.activate_score(g, o.value[1], k[2])
        return NEG

    def land_score(self, g: Game, card: Card) -> float:
        missing = self.tron_missing(g)
        if any(s in card.face.subtypes for s in missing):
            return 60.0
        if card.name == "Bojuka Bog":
            return 50.0 if len(g.players[self.opp].graveyard) >= 4 else 41.0
        if card.name == "Forest":
            green = any("G" in c.face.cost.colored_dict() for c in self.hand(g))
            return 55.0 if green else 45.0
        if card.name == "Conduit Pylons":
            return 47.0
        return 44.0

    def cast_score(self, g: Game, card: Card, mode: str) -> float:
        n = card.name
        main = self.main_phase(g)
        if n == "Crop Rotation":
            return self.crop_rotation_score(g)
        if n == "Unfathomable Truths":
            return 12.0 if self.end_of_their_turn(g) or (main and g.step_name == "main2" and self.mana(g) >= 9) else NEG
        if n == "Pulse of Murasa":
            mine = [c for c in self.me(g).graveyard if c.face.is_type("Creature")]
            return 11.0 if mine and (self.end_of_their_turn(g) or self.me(g).life <= LOW_LIFE) else NEG
        if n == "Scour from Existence":
            t = self.best_permanent_target(g)
            return 14.0 if t is not None and (self.end_of_their_turn(g) or self.creature_value(g, t) >= 6) else NEG
        if n == "Breath Weapon":  # 2 to each non-Dragon creature, at instant speed
            theirs = [c for c in self.creatures(g, self.opp) if g.toughness(c) - c.damage <= 2]
            mine = [c for c in self.creatures(g, self.p) if g.toughness(c) - c.damage <= 2 and not c.is_token]
            value = sum(self.creature_value(g, c) for c in theirs) - sum(self.creature_value(g, c) for c in mine)
            timely = main or self.end_of_their_turn(g) or g.step_name in ("declare_attackers", "declare_blockers")
            return 15.0 + len(theirs) if timely and len(theirs) >= 2 and value >= 3 else NEG
        if n == "Blue Elemental Blast":
            return NEG  # modal; handled in modal_score
        if not main:
            return NEG
        if n in THREATS:
            if n == "Boulderbranch Golem" and mode == "prototype":
                return 15.0 if self.mana(g) < 7 else NEG
            if n == "Pinnacle Kill-Ship" and not self.kill_targets(g, 10) and any(c.name in THREATS for c in self.hand(g) if c is not card and c.name != n):
                return THREATS[n] - 6
            return THREATS[n]
        if n == "Ancient Stirrings":
            return 16.0 if self.tron_missing(g) or not any(c.name in THREATS for c in self.hand(g)) else 5.0
        if n == "Expedition Map":
            return CHEAP[n] + (5 if self.missing_in_hand(g) else 0)
        if n in CHEAP:
            return CHEAP[n]
        if n == "Kaervek's Torch":
            return self.torch_score(g)
        if n == "Monstrous Emergence":
            pw = max([g.power(c) for c in self.creatures(g, self.p)] + [c.face.power or 0 for c in self.hand(g) if c.face.is_type("Creature") and c is not card], default=0)
            return 13.0 if self.kill_targets(g, pw) else NEG
        if n == "Call Damage Control":
            gy = self.me(g).graveyard
            useful = sum(1 for c in gy if c.face.is_type("Creature") or c.face.is_type("Land") or c.face.is_type("Artifact"))
            return 10.0 if useful >= 2 else NEG
        return NEG

    def crop_rotation_score(self, g: Game) -> float:
        missing = self.tron_missing(g)
        if len(missing) != 1:
            return NEG
        # Only when the sacrificed land is not a Tron piece we need.
        spare = [c for c in self.lands(g, self.p) if not any(s in c.face.subtypes for s in TRON) or self.duplicate_piece(g, c)]
        if not spare:
            return NEG
        if self.main_phase(g) or self.end_of_their_turn(g):
            return 35.0
        return NEG

    def duplicate_piece(self, g: Game, land: Card) -> bool:
        subs = [s for s in TRON if s in land.face.subtypes]
        if not subs:
            return False
        return sum(1 for c in self.lands(g, self.p) if subs[0] in c.face.subtypes) > 1

    def kill_targets(self, g: Game, dmg: int) -> list[Card]:
        return [c for c in self.opp_threats(g) if g.toughness(c) - c.damage <= dmg and self.creature_value(g, c) >= KILL_VALUE]

    def best_permanent_target(self, g: Game) -> Card | None:
        threats = self.opp_threats(g)
        return threats[0] if threats else None

    def torch_score(self, g: Game) -> float:
        x = self.mana(g) - 1
        if g.players[self.opp].life <= x:
            return 40.0
        return 12.0 if self.kill_targets(g, x) else NEG

    def modal_score(self, g: Game, card: Card, mode: str) -> float:
        if card.name != "Blue Elemental Blast":
            return NEG
        if mode == "counter":
            top = self.stack_top(g)
            return 30.0 if top is not None and top.kind == "spell" and top.controller == self.opp and "R" in top.card.face.colors else NEG
        red = [c for c in self.creatures(g, self.opp) if "R" in c.face.colors and self.creature_value(g, c) >= 2]
        return 20.0 if red else NEG

    def activate_score(self, g: Game, card: Card, ability: str) -> float:
        n = card.name
        spare = self.end_of_their_turn(g) or (self.main_phase(g) and g.step_name == "main2")
        if n == "Expedition Map":
            if self.missing_in_hand(g):
                return 20.0 if self.main_phase(g) or self.end_of_their_turn(g) else NEG
            return 3.0 if self.end_of_their_turn(g) else NEG
        if n == "Pinnacle Kill-Ship":  # station: tap a creature that would not attack anyway
            return 6.0 if g.step_name == "main2" and any(c.sick and c is not card for c in self.creatures(g, self.p) if not c.tapped) else NEG
        if n == "Candy Trail":
            return 5.0 if spare or self.me(g).life <= LOW_LIFE else NEG
        if n == "Bonder's Ornament":
            return 4.0 if self.end_of_their_turn(g) else NEG
        if n == "Barrels of Blasting Jelly":
            return 9.0 if self.kill_targets(g, 5) and (spare or self.main_phase(g)) else NEG
        if n == "Bramble Wurm":
            return 4.0 if spare and self.me(g).life <= 12 else NEG
        if n == "Haunted Fengraf":
            dead = [c for c in self.me(g).graveyard if c.face.is_type("Creature")]
            return 3.0 if dead and self.end_of_their_turn(g) and len(self.lands(g, self.p)) >= 7 else NEG
        if n == "Generous Ent":  # forestcycling, while a Forest may be left in the library
            seen = sum(1 for c in self.lands(g, self.p) + self.hand(g) + list(self.me(g).graveyard) if c.name == "Forest")
            if self.land_need(g) > 0 and self.mana(g) < 6 and seen < FORESTS_IN_DECK:
                return 10.0 if self.main_phase(g) or self.end_of_their_turn(g) else NEG
            return NEG
        if n == "Food":
            return 3.0 if spare and self.me(g).life <= 12 else NEG
        if n == "Relic of Progenitus":
            gy = g.players[self.opp].graveyard
            if ability.startswith("exile all"):
                return 4.0 if self.end_of_their_turn(g) and len(gy) >= 5 else NEG
            return 2.0 if self.end_of_their_turn(g) and gy else NEG
        return NEG

    # -- costs and choices ---------------------------------------------------------

    def score_pay_mana(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[1] == "pool":
            return 10.0
        if o.key[1] == "filter":
            return -8.0
        card, color = o.value[1], o.value[2]
        ab = g.mana_ability(card)
        units = g.mana_amount(card, ab)
        cost = self.source_cost(g, card, color)
        # Tron lands: no waste. Prefer the source whose mana fits what is left to pay.
        left = sum(int(x) for x in _remaining_numbers(d.prompt))
        waste = max(0, units - max(left, 1))
        return -cost - 2.0 * waste + 0.1 * min(units, left)

    def source_cost(self, g: Game, card: Card, color: str) -> float:
        if card.is_token:
            return 5.0
        colors = g.mana_ability(card).mana
        if len(colors) > 1:
            return 4.0  # Bonder's Ornament: keep it for coloured costs
        if colors == ("G",):
            return 2.0
        return 1.0

    def sac_cost(self, g: Game, c: Card) -> float:
        if g.is_land(c):
            if any(s in c.face.subtypes for s in TRON) and not self.duplicate_piece(g, c):
                return 30.0
            return 2.0 - (1.0 if c.tapped else 0.0)
        return super().sac_cost(g, c)

    def score_choose_card(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] == "search":
            return NEG if k[1] is None else self.search_value(g, k[1])
        if k[0] == "dig":
            return 0.0 if k[1] is None else self.search_value(g, k[1])
        if k[0] == "tap_cost":  # station: the strongest summoning-sick creature
            c = o.value
            return g.power(c) + (5 if c.sick else 0)
        if k[0] == "choose_creature":
            c = o.value
            return (g.power(c) if k[1] == "battlefield" else (c.face.power or 0)) + (0.5 if k[1] == "hand" else 0)
        if k[0] == "return_gy":
            if k[1] is None:
                return 0.0
            if k[1] != "self":
                return NEG
            return self.search_value(g, k[3])
        return super().score_choose_card(g, d, o)

    def search_value(self, g: Game, name: str) -> float:
        card = g.cards_db[name]
        missing = self.tron_missing(g)
        if card.is_type("Land"):
            if any(s in card.subtypes for s in self.missing_in_hand(g)):
                return 20.0
            if any(s in card.subtypes for s in missing):
                return 12.0
            return 3.0 if name in ("Forest", "Conduit Pylons") else 2.0
        if name == "Expedition Map":
            return 10.0 if self.missing_in_hand(g) else 2.0
        if name in THREATS:
            return 8.0 + THREATS[name] / 10
        return self.card_value(g, name)

    def score_order(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] != "scry":
            return super().score_order(g, d, o)
        cut = k.index("bottom")
        top, bottom = k[2:cut], k[cut + 1:]
        s = 0.0
        for i, n in enumerate(top):
            s += (self.search_value(g, n) - 3.0) * (1.0 - 0.1 * i)
        return s - 0.01 * len(bottom)

    def score_choose_mode(self, g: Game, d: Decision, o: Option) -> float:
        tag, ans = o.key
        if tag == "surveil":
            top = self.me(g).library[0]  # being surveilled: known to the bot
            keep = self.search_value(g, top.name) >= 4
            return 1.0 if (ans == "top") == keep else 0.0
        if tag == "scry":
            top = self.me(g).library[0]
            keep = self.search_value(g, top.name) >= 4
            return 1.0 if (ans == "top") == keep else 0.0
        return super().score_choose_mode(g, d, o)

    def score_yes_no(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[0] == "cascade":
            return 1.0 if o.key[1] == "cast" else 0.0
        return super().score_yes_no(g, d, o)

    def score_target(self, g: Game, d: Decision, o: Option) -> float:
        name = self.building(g, d)
        v = o.value
        if v is None:  # "No target" (Kill-Ship's up to one)
            return 0.5
        c = self.ref_card(g, o)
        if name in ("Pinnacle Kill-Ship", "Barrels of Blasting Jelly", "Monstrous Emergence", "Kaervek's Torch", "Scour from Existence"):
            if v == ("player", self.opp):
                return 50.0 if name == "Kaervek's Torch" and self.torch_score(g) >= 40 else NEG
            if c is None or c.controller == self.p:
                return NEG
            return 1.0 + self.creature_value(g, c) if g.is_creature(c) else 0.8
        if name == "Bojuka Bog" or name == "Relic of Progenitus":
            return 1.0 if v == ("player", self.opp) else 0.0
        if name == "Blue Elemental Blast":
            if c is not None and c.controller == self.p:
                return NEG
            return 1.0
        return super().score_target(g, d, o)


def _remaining_numbers(prompt: str) -> list[str]:
    """The mana value still to pay, from a "Pay {2}{G} for X" prompt."""
    import re

    m = re.match(r"Pay ((?:\{[^}]+\})+)", prompt)
    if not m:
        return ["1"]
    out = []
    for sym in re.findall(r"\{([^}]+)\}", m.group(1)):
        out.append(sym if sym.isdigit() else "1")
    return out
