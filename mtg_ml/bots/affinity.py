"""Grixis Affinity bot: cheap artifacts, then free threats.

Plan: a land every turn (artifact lands first: every one makes the affinity
spells cheaper), the cheap artifacts (Wellspring, Spellbomb, Blood Fountain,
Sewer-veillance Cam) before the affinity cards, then Myr Enforcer, Utrom
Monitor, Refurbished Familiar and Thoughtcast as soon as they cost little.
Black Mage's Rod makes a Hero and is re-equipped (best creature, evasion
first) when it falls off. Kenku Artificer animates an indestructible
Bridge. Envelop counters a sorcery worth it. Galvanic Blast kills a
creature worth it or goes to the face once the opponent is in reach;
Reckoner's Bargain turns spare artifacts into cards in the opponent's end
step. Krark-Clan Shaman, Makeshift Munitions, Toxin Analysis, Nihil
Spellbomb, Clue and Blood are played as the Jund bot plays them.
"""

from __future__ import annotations

from ..engine.game import Game
from ..engine.objects import Card, Decision, Option
from .base import NEG
from .jund import JundBot

# Tunable judgment calls.
KEEP_LANDS = {7: (2, 4), 6: (1, 4)}
KILL_VALUE = 2.5  # creature_value worth a Galvanic Blast
FACE_LIFE = 6  # at or below this, Galvanic Blast goes to the face
CHEAP_ARTIFACTS = {"Ichor Wellspring": 7.5, "Nihil Spellbomb": 7, "Blood Fountain": 7.2, "Sewer-veillance Cam": 6.5, "Black Mage's Rod": 6.8}
THREATS = {"Myr Enforcer": 9, "Utrom Monitor": 9.5, "Refurbished Familiar": 9.2, "Kenku Artificer": 6, "Krark-Clan Shaman": 5}
ARTIFACT_LANDS = {"Mistvault Bridge", "Drossforge Bridge", "Silverbluff Bridge", "Vault of Whispers", "Seat of the Synod", "Great Furnace"}
TAPPED_LANDS = {"Mistvault Bridge", "Drossforge Bridge", "Silverbluff Bridge"}
SPARE_ARTIFACTS = 6  # with this many artifacts, a land or Wellspring is Bargain fodder


class AffinityBot(JundBot):
    name = "affinity"

    # -- valuations ---------------------------------------------------------

    def card_value(self, g: Game, name: str) -> float:
        card = g.cards_db[name]
        if card.is_type("Land"):
            return 4.0 if self.land_need(g) > 0 else 1.0
        return {
            "Galvanic Blast": 5,
            "Thoughtcast": 4,
            "Myr Enforcer": 4,
            "Utrom Monitor": 4.5,
            "Refurbished Familiar": 4,
            "Reckoner's Bargain": 3.5,
            "Kenku Artificer": 2.5,
            "Black Mage's Rod": 3,
            "Krark-Clan Shaman": 2.5,
            "Envelop": 2.5,
            "Extract a Confession": 3.5,
            "Pyroblast": 4,
            "Red Elemental Blast": 4,
            "Blue Elemental Blast": 4,
            "Hydroblast": 4,
            "Unexpected Fangs": 2.5,
            "Toxin Analysis": 2,
            "Ichor Wellspring": 2.5,
            "Nihil Spellbomb": 2,
            "Blood Fountain": 2,
            "Sewer-veillance Cam": 2,
            "Makeshift Munitions": 2,
        }.get(name, 1.0)

    def land_need(self, g: Game) -> float:
        have = len(self.lands(g, self.p)) + sum(1 for c in self.hand(g) if c.face.is_type("Land"))
        return 3.5 - have

    def keep_hand(self, g: Game, size: int) -> bool:
        if size <= 5:
            return True
        hand = self.hand(g)
        lands = sum(1 for c in hand if c.face.is_type("Land"))
        lo, hi = KEEP_LANDS[size]
        return lo <= lands <= hi

    def artifacts(self, g: Game) -> int:
        return sum(1 for c in g.battlefield if c.controller == self.p and g.is_artifact(c))

    def metalcraft(self, g: Game) -> bool:
        return self.artifacts(g) >= 3

    def blast_damage(self, g: Game) -> int:
        return 4 if self.metalcraft(g) else 2

    # -- priority -----------------------------------------------------------

    def creatures_to_cast(self, g: Game) -> bool:
        return any(c.name in THREATS or c.name in CHEAP_ARTIFACTS for c in self.hand(g))

    def land_score(self, g: Game, name: str) -> float:
        s = 50.0
        castable_now = any(self.cost(g, c) == self.mana_available(g) + 1 for c in self.hand(g) if not c.face.is_type("Land") and c.name not in ("Galvanic Blast",))
        if name in TAPPED_LANDS:
            s += 1 if not castable_now else -2
        elif name in ARTIFACT_LANDS:
            s += 0.5
        else:  # Mountain: not an artifact
            s -= 1.5
        return s

    def cast_score(self, g: Game, card: Card, mode: str) -> float:
        n = card.name
        main = self.main_phase(g)
        mana = self.mana_available(g)
        if n == "Galvanic Blast":
            return self.blast_score(g)
        if n == "Reckoner's Bargain":
            fodder = min((self.sac_cost(g, c) for c in g.sac_candidates(self.p, "artifact_or_creature")), default=99)
            if fodder > 2.5:
                return NEG
            if self.opp_end_step(g) or (main and g.step_name == "main2" and len(self.hand(g)) <= 2):
                return 6.0
            return NEG
        if n == "Sewer-veillance Cam":
            if main:
                return CHEAP_ARTIFACTS[n] if self.affinity_in_hand(g) else NEG
            return 3.0 if self.opp_end_step(g) else NEG
        if n == "Extract a Confession":
            if not main or not self.creatures(g, self.opp):
                return NEG
            return 14.0 if mode == "evidence" else 12.0
        if n == "Unexpected Fangs":
            if g.step_name == "declare_blockers" and self.combat_trick_target(g):
                return 25.0
            return 2.5 if self.opp_end_step(g) and self.creatures(g, self.p) else NEG
        if n in ("Hydroblast", "Blue Elemental Blast", "Pyroblast", "Red Elemental Blast"):
            return NEG  # modal: see modal_score
        if n == "Envelop":
            top = self.stack_top(g)
            ok = top is not None and top.kind == "spell" and top.controller == self.opp and top.card.face.is_type("Sorcery")
            return 30.0 if ok else NEG
        if not main:
            return super().cast_score(g, card, mode) if n in ("Toxin Analysis", "Duress") else NEG
        if n in CHEAP_ARTIFACTS:
            return CHEAP_ARTIFACTS[n]
        cost = self.cost(g, card)
        if n == "Thoughtcast":
            return 8.0 if cost <= 1 or (cost <= 2 and len(self.hand(g)) <= 2) else (1.0 if g.step_name == "main2" and cost <= mana else NEG)
        if n == "Kenku Artificer":
            return THREATS[n] if self.kenku_target(g) is not None else NEG
        if n == "Krark-Clan Shaman":
            return THREATS[n]
        if n in THREATS:
            # Wait a turn when one more artifact would make it much cheaper and nothing else is castable.
            return THREATS[n] + 0.1 * cost
        if n == "Makeshift Munitions":
            return 3.0
        return super().cast_score(g, card, mode)

    def affinity_in_hand(self, g: Game) -> bool:
        return any(c.face.cost_reduction is not None for c in self.hand(g))

    def kenku_target(self, g: Game) -> Card | None:
        best, best_v = None, 0.0
        for c in g.battlefield:
            if c.controller != self.p or not g.is_artifact(c) or g.is_creature(c):
                continue
            v = 3.0 if g.has(c, "indestructible") else 1.0 if c.name == "Ichor Wellspring" else 0.5
            if v > best_v:
                best, best_v = c, v
        return best

    def equip_target(self, g: Game) -> Card | None:
        """Best creature for Black Mage's Rod: evasive first, then the strongest."""
        mine = self.creatures(g, self.p)
        if not mine:
            return None
        return max(mine, key=lambda c: (g.has(c, "flying"), self.creature_value(g, c), -c.oid))

    def blast_score(self, g: Game) -> float:
        dmg = self.blast_damage(g)
        opp_life = g.players[self.opp].life
        if opp_life <= dmg:
            return 40.0
        if self.blast_kill_targets(g, dmg):
            return 25.0
        if opp_life <= FACE_LIFE or (self.opp_end_step(g) and dmg == 4 and opp_life <= 12):
            return 10.0
        return NEG

    def blast_kill_targets(self, g: Game, dmg: int) -> list[Card]:
        return [c for c in self.creatures(g, self.opp) if g.toughness(c) - c.damage <= dmg and self.threat(g, c) >= KILL_VALUE and not (c.face.ward and self.mana_available(g) < 1 + c.face.ward)]

    def modal_score(self, g: Game, card: Card, mode: str) -> float:
        n = card.name
        if n in ("Pyroblast", "Red Elemental Blast"):
            return self.blast_modal(g, mode, "U")
        if n in ("Hydroblast", "Blue Elemental Blast"):
            return self.blast_modal(g, mode, "R")
        return NEG

    def blast_modal(self, g: Game, mode: str, color: str) -> float:
        if mode == "counter":
            top = self.stack_top(g)
            ok = top is not None and top.kind == "spell" and top.controller == self.opp and color in top.card.face.colors
            return 32.0 if ok and self.BLUE_SPELL_VALUE.get(top.name, 2.0) >= 1.5 else NEG
        targets = [c for c in g.battlefield if c.controller == self.opp and color in c.face.colors and (not g.is_creature(c) or self.threat(g, c) >= 2)]
        return 20.0 if targets and (self.main_phase(g) or self.opp_end_step(g)) else NEG

    def activate_score(self, g: Game, card: Card, ability: str) -> float:
        n = card.name
        if n == "Blood Fountain":
            creatures = [c for c in self.me(g).graveyard if c.face.is_type("Creature")]
            return 4.0 if creatures and (self.opp_end_step(g) or (g.step_name == "main2" and len(self.hand(g)) <= 1)) else NEG
        if n == "Sewer-veillance Cam":
            return 3.5 if self.opp_end_step(g) else NEG
        if n == "Black Mage's Rod":  # equip {3}, sorcery speed
            if card.attached_to is not None or self.equip_target(g) is None:
                return NEG
            spare = self.mana_available(g) - 3
            return 4.0 if spare >= 0 and not any(self.cost(g, c) <= self.mana_available(g) and c.name in THREATS for c in self.hand(g)) else NEG
        if n == "Blood":
            fodder = [c for c in self.hand(g) if self.card_value(g, c.name) <= 1.0]
            return 3.0 if fodder and (self.opp_end_step(g) or g.step_name == "main2") else NEG
        return super().activate_score(g, card, ability)

    def sac_cost(self, g: Game, c: Card) -> float:
        if g.is_land(c) and not c.is_token:
            return 1.8 if len(self.lands(g, self.p)) >= 5 and self.artifacts(g) >= SPARE_ARTIFACTS else 20.0
        if c.name in ("Blood Fountain", "Sewer-veillance Cam"):
            return 0.7
        return super().sac_cost(g, c)

    # -- targets and choices ------------------------------------------------

    def score_target(self, g: Game, d: Decision, o: Option) -> float:
        name = self.building(g, d)
        c = self.ref_card(g, o)
        v = o.value
        if name == "Galvanic Blast":
            dmg = self.blast_damage(g)
            if v == ("player", self.opp):
                lethal = g.players[self.opp].life <= dmg
                return 100.0 if lethal else (5.0 if not self.blast_kill_targets(g, dmg) else 1.0)
            if v[0] == "player" or c is None or c.controller == self.p:
                return NEG
            dies = g.toughness(c) - c.damage <= dmg
            s = (2.0 + self.threat(g, c)) if dies else -1.0
            if c.face.ward and self.mana_available(g) < c.face.ward:
                s -= 20
            return s
        if name == "Kenku Artificer":
            if v is None:
                return 0.1
            if c is None or c.controller != self.p:
                return NEG
            k = self.kenku_target(g)
            return 5.0 if k is not None and c.oid == k.oid else 1.0
        if name == "Black Mage's Rod":
            if c is None or c.controller != self.p:
                return NEG
            t = self.equip_target(g)
            return 5.0 if t is not None and c.oid == t.oid else self.creature_value(g, c)
        if name == "Envelop":
            it = self.ref_spell(g, o)
            return 5.0 if it is not None and it.controller == self.opp else NEG
        if name == "Sewer-veillance Cam":
            if c is None:
                return NEG
            if c.controller == self.opp:
                return self.creature_value(g, c) + (2 if not c.tapped else -3)
            return 1.0 if c.tapped else 0.0
        if name in ("Pyroblast", "Red Elemental Blast", "Hydroblast", "Blue Elemental Blast"):
            it = self.ref_spell(g, o)
            if it is not None:
                return 5.0 if it.controller == self.opp else NEG
            if c is None or c.controller == self.p:
                return NEG
            return self.threat(g, c) if g.is_creature(c) else 2.0
        if name == "Unexpected Fangs":
            if c is None or c.controller != self.p:
                return NEG
            tgt = self.combat_trick_target(g)
            return 10.0 if tgt is not None and c.oid == tgt.oid else self.creature_value(g, c)
        return super().score_target(g, d, o)

    def score_choose_mode(self, g: Game, d: Decision, o: Option) -> float:
        tag, ans = o.key
        if tag == "tap_or_untap":
            t = g.perm(int(d.prompt.rsplit("#", 1)[1].rstrip("?")))
            if t is None:
                return 0.0
            if t.controller == self.opp:
                return 1.0 if ans == "tap" and not t.tapped else 0.0
            return 1.0 if ans == "untap" and t.tapped else 0.0
        return super().score_choose_mode(g, d, o)

    def score_choose_card(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[0] == "return_gy":
            return 0.0 if o.key[1] is None else self.card_value(g, o.key[1])
        return super().score_choose_card(g, d, o)

    def score_exile_from_graveyard(self, g: Game, d: Decision, o: Option) -> float:
        c = o.value
        if d.prompt.startswith("Collect evidence"):
            return c.face.mana_value - (3 if c.face.is_type("Creature") else 0)
        return super().score_exile_from_graveyard(g, d, o)
