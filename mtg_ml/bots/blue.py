"""Mono Blue Terror bot: tempo with counter backup.

Plan: Delver early, cantrips to dig and fill the graveyard (Thought Scour
mills itself), Tolarian Terror / Cryptic Serpent once cheap, flying beats.
On Jund's turn keep UU up and counter what matters (removal on our
threats, creatures, card draw); Force Spike when Jund is tapped out. Spare
mana at Jund's end step goes into instant-speed card draw.
"""

from __future__ import annotations

from ..engine.game import Game
from ..engine.objects import Card, Decision, Option
from .base import NEG, Bot

# Tunable judgment calls.
KEEP_LANDS = {7: (2, 4), 6: (2, 4)}  # land counts to keep; 1 land is fine with 2+ cantrips
COUNTER_MIN_VALUE = 2.5  # spell value worth a Counterspell
FORCE_SPIKE_MIN_VALUE = 1.5
HOLD_UP_COUNTER = True  # on our turn keep UU open for Jund's turn when we have a threat out

THREATS = ("Tolarian Terror", "Cryptic Serpent")
CANTRIPS = ("Brainstorm", "Ponder", "Thought Scour", "Mental Note", "Plunder the Trollshaws")
SPELL_VALUE = {  # how much a Jund spell on the stack hurts
    "Writhing Chrysalis": 4.5,
    "Refurbished Familiar": 3.5,
    "Gixian Infiltrator": 2.5,
    "Krark-Clan Shaman": 2.0,
    "Nyxborn Hydra": 3.0,
    "Fanatical Offering": 2.5,
    "Eviscerator's Insight": 2.5,
    "Makeshift Munitions": 2.0,
    "Cleansing Wildfire": 1.0,
    "Toxin Analysis": 1.0,
    "Ichor Wellspring": 0.5,
    "Lembas": 0.5,
    "Nihil Spellbomb": 1.0,
    "Duress": 2.0,
}


class BlueBot(Bot):
    name = "blue"

    # -- valuations ---------------------------------------------------------

    def card_value(self, g: Game, name: str) -> float:
        if name == "Island":
            return 4.0 if self.land_need(g) > 0 else 0.8
        return {
            "Counterspell": 5,
            "Dispel": 3,
            "Blue Elemental Blast": 3,
            "Steel Sabotage": 3,
            "Tolarian Terror": 4.5,
            "Cryptic Serpent": 4,
            "Delver of Secrets": 3.5 if g.turn <= 6 else 1.5,
            "Brainstorm": 3,
            "Ponder": 2.5,
            "Deem Inferior": 3,
            "Force Spike": 2.5 if g.turn <= 6 else 1,
            "Lorien Revealed": 2.5,
            "Thought Scour": 2,
            "Mental Note": 1.8,
            "Sleep of the Dead": 2,
            "Plunder the Trollshaws": 2,
        }.get(name, 1.0)

    def keep_hand(self, g: Game, size: int) -> bool:
        if size <= 5:
            return True
        hand = self.hand(g)
        lands = sum(1 for c in hand if c.face.is_type("Land"))
        cantrips = sum(1 for c in hand if c.name in ("Brainstorm", "Ponder", "Thought Scour", "Mental Note"))
        lo, hi = KEEP_LANDS[size]
        return lo <= lands <= hi or (lands == 1 and cantrips >= 2)

    def spell_value(self, g: Game, item) -> float:
        if item.kind != "spell" or item.controller == self.p:
            return NEG
        hits = [g.perm(t[1]) for t in item.targets if t[0] == "perm"]
        mine = [c for c in hits if c is not None and c.controller == self.p]
        if mine:  # removal (Cast Down, Go for the Throat, Red Elemental Blast...) on our stuff
            return max(self.creature_value(g, c) if g.is_creature(c) else 1.0 for c in mine) + 1
        if any(t[0] == "stack" and (x := g.stack_item(t[1])) is not None and x.controller == self.p for t in item.targets):
            return 4.0  # their counter (Red Elemental Blast) on our spell
        if item.name in ("Cast Down", "Go for the Throat", "Red Elemental Blast"):
            return 0.5
        v = SPELL_VALUE.get(item.name, 1.0)
        if item.name == "Nyxborn Hydra":
            v += item.x * 0.7
        return v

    def threats_out(self, g: Game) -> list[Card]:
        return [c for c in self.creatures(g, self.p) if g.power(c) >= 3 or c.name in THREATS]

    def instants_in_gy(self, g: Game) -> int:
        return sum(1 for c in self.me(g).graveyard if c.face.is_type("Instant") or c.face.is_type("Sorcery"))

    def counter_mana_after(self, g: Game, spend: int) -> int:
        return self.mana_available(g) - spend

    def search_value(self, g: Game, name: str) -> float:
        return 1.0

    # -- priority -----------------------------------------------------------

    def score_priority(self, g: Game, d: Decision, o: Option) -> float:
        k = o.key
        if k[0] == "pass":
            return 0.0
        if k[0] == "play_land":
            return 50.0
        if k[0] == "cast":
            if len(k) > 4:
                return self.modal_score(g, o.value[1], k[4])
            return self.cast_score(g, o.value[1], k[3])
        if k[0] == "activate":
            if "islandcycling" in k[2]:
                lands_in_hand = sum(1 for c in self.hand(g) if c.face.is_type("Land"))
                return 4.0 if lands_in_hand == 0 and self.land_need(g) > -1 and (self.opp_end_step(g) or self.main_phase(g)) else NEG
            return NEG
        return NEG

    def cast_score(self, g: Game, card: Card, mode: str) -> float:
        n = card.name
        cost = self.cost(g, card, mode)
        main = self.main_phase(g)
        if n == "Counterspell":
            it = self.counter_target(g)
            return 40.0 if it is not None and self.spell_value(g, it) >= COUNTER_MIN_VALUE else NEG
        if n == "Dispel":
            it = self.counter_target(g, lambda x: x.card.face.is_type("Instant"))
            return 38.0 if it is not None and self.spell_value(g, it) >= COUNTER_MIN_VALUE else NEG
        if n == "Force Spike":
            it = self.counter_target(g)
            jund_mana = self.mana_available(g, self.opp)
            return 35.0 if it is not None and jund_mana == 0 and self.spell_value(g, it) >= FORCE_SPIKE_MIN_VALUE else NEG
        if g.stack:
            return NEG  # cantrips do not go in response
        holding = HOLD_UP_COUNTER and any(c.name == "Counterspell" for c in self.hand(g)) and self.threats_out(g)
        if n in THREATS:
            if not main:
                return NEG
            return 20.0 + (5 if not self.threats_out(g) else 0)
        if n == "Delver of Secrets":
            return 18.0 if main else NEG
        if n == "Deem Inferior":
            targets = [c for c in g.battlefield if c.controller == self.opp and not g.is_land(c)]
            best = max((self.creature_value(g, c) if g.is_creature(c) else 1.0 for c in targets), default=0)
            return 14.0 if main and best >= 3 else NEG
        if n == "Sleep of the Dead":
            if not main or g.step_name != "main1" or not self.threats_out(g):
                return NEG
            blockers = [c for c in self.creatures(g, self.opp) if not c.tapped and (g.has(c, "reach") or g.has(c, "flying"))]
            return 12.0 if blockers else NEG
        if n == "Lorien Revealed":
            return 8.0 if main and self.mana_available(g) >= 5 else NEG
        if n in CANTRIPS:
            if mode == "flashback":
                return 5.0 if self.opp_end_step(g) else NEG
            if self.opp_end_step(g):
                return 10.0 - (1 if n == "Brainstorm" and self.land_need(g) <= 0 else 0)
            if main:
                # sorcery-speed digging; keep UU for Jund's turn if we're holding counters
                if holding and self.counter_mana_after(g, cost) < 2:
                    return NEG
                return 9.0 if n == "Ponder" else 7.0
            return NEG
        return NEG

    def modal_score(self, g: Game, card: Card, mode: str) -> float:
        if card.name == "Blue Elemental Blast":
            if mode == "counter":
                it = self.counter_target(g, lambda x: "R" in x.card.face.colors)
                return 37.0 if it is not None and self.spell_value(g, it) >= 1.5 else NEG
            reds = [c for c in g.battlefield if c.controller == self.opp and "R" in c.face.colors]
            if not reds or g.stack:
                return NEG
            return 6.0 if self.opp_end_step(g) or (self.main_phase(g) and g.step_name == "main1") else NEG
        if card.name == "Steel Sabotage":
            if mode == "counter":
                it = self.counter_target(g, lambda x: x.card.face.is_type("Artifact"))
                return 36.0 if it is not None and self.spell_value(g, it) >= COUNTER_MIN_VALUE else NEG
            return NEG  # bounce: tempo only, not worth a card by default
        return NEG

    def counter_target(self, g: Game, legal=None):
        """Most valuable Jund spell on the stack that is not already dealt with
        (by our counter, or by a ward trigger Jund cannot pay for). `legal`
        restricts to spells the counter can target."""
        best, best_v = None, NEG
        for it in g.stack:
            if it.kind != "spell" or (legal is not None and not legal(it)):
                continue
            if self._already_countered(g, it) or self._dies_to_ward(g, it):
                continue
            v = self.spell_value(g, it)
            if v > best_v:
                best, best_v = it, v
        return best

    def _dies_to_ward(self, g: Game, item) -> bool:
        for it in g.stack:
            if it.kind == "trigger" and it.data.get("sid") == item.sid:
                return self.mana_available(g, self.opp) < it.data.get("amount", 0)
        return False

    @staticmethod
    def _already_countered(g: Game, item) -> bool:
        return any(it.kind == "spell" and ("stack", item.sid) in it.targets for it in g.stack if it is not item)

    # -- targets ------------------------------------------------------------

    def score_target(self, g: Game, d: Decision, o: Option) -> float:
        spell = self.building(g, d)
        c = self.ref_card(g, o)
        v = o.value
        if spell in ("Counterspell", "Force Spike", "Dispel", "Blue Elemental Blast", "Steel Sabotage"):
            it = self.ref_spell(g, o)
            if it is not None:
                return self.spell_value(g, it) + (0 if not self._already_countered(g, it) else -50)
            if c is not None and c.controller == self.opp:  # Blast destroying a red permanent
                return self.creature_value(g, c) if g.is_creature(c) else 2.0
            return NEG
        if spell in ("Thought Scour", "Mental Note"):
            return 1.0 if v == ("player", self.p) else 0.0  # mill ourselves: Terror fuel
        if spell == "Deem Inferior":
            if c is None or c.controller == self.p:
                return NEG
            return self.creature_value(g, c) if g.is_creature(c) else 1.0
        if spell == "Sleep of the Dead":
            if c is None or c.controller == self.p:
                return NEG
            return self.creature_value(g, c) + (3 if g.has(c, "reach") or g.has(c, "flying") else 0)
        if c is not None:
            return self.creature_value(g, c) * (1 if c.controller == self.opp else -1)
        return 1.0 if v == ("player", self.opp) else 0.0

    def score_choose_mode(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[0] == "deem":
            return 1.0 if o.key[1] == "bottom" else 0.0
        return super().score_choose_mode(g, d, o)

    def score_exile_from_graveyard(self, g: Game, d: Decision, o: Option) -> float:
        # Escape: keep instants/sorceries for Terror and Serpent discounts.
        c = o.value
        if c.face.is_type("Land"):
            return 3.0
        if g.is_creature(c) or c.face.is_type("Creature"):
            return 2.0
        return 1.0 - self.card_value(g, c.name) * 0.1
