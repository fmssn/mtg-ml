"""Shared machinery for the scripted bots.

A bot scores every legal option of the current decision and takes the best
one (the first on ties, so "pass" / "done" / "don't" win ties: index 0).
Decks override the hooks; everything else (combat, mana payment, card
choices) is shared.

Information hygiene: bots read the `Game` object directly, but only public
state plus their own hand, through the helpers here. Nothing reads the
opponent's hand or either library, except cards the option list itself
shows (Brainstorm, Ponder, scry: cards the bot is looking at). Bots are
deterministic, so `tests/test_bots.py` can check that a decision does not
change when hidden cards are re-sampled (`view.determinize`).
"""

from __future__ import annotations

import re

from ..engine.game import Game
from ..engine.objects import Card, Decision, Option

NEG = -1e9
_FOR = re.compile(r" for (.+?)(?: or .*)?$")


class Bot:
    #: cards whose name says "worth casting a counter on"; decks fill this in
    name = "bot"

    def __init__(self, seat: int):
        self.p = seat

    # -- entry point ---------------------------------------------------------

    def act(self, g: Game) -> int:
        d = g.decision
        handler = getattr(self, f"score_{d.kind}", None)
        if handler is None:
            return 0
        best, best_i = NEG - 1, 0
        for i, o in enumerate(d.options):
            s = handler(g, d, o)
            if s > best:
                best, best_i = s, i
        return best_i

    # -- what the bot may look at -------------------------------------------

    @property
    def opp(self) -> int:
        return 1 - self.p

    def me(self, g: Game):
        return g.players[self.p]

    def hand(self, g: Game) -> list[Card]:
        return list(g.players[self.p].hand)

    def creatures(self, g: Game, who: int) -> list[Card]:
        return [c for c in g.battlefield if c.controller == who and g.is_creature(c)]

    def lands(self, g: Game, who: int) -> list[Card]:
        return [c for c in g.battlefield if c.controller == who and g.is_land(c)]

    def untapped_lands(self, g: Game, who: int) -> list[Card]:
        return [c for c in self.lands(g, who) if not c.tapped]

    def mana_available(self, g: Game, who: int | None = None) -> int:
        who = self.p if who is None else who
        return len(g.mana_sources(who)) + sum(g.players[who].pool.values())

    def my_turn(self, g: Game) -> bool:
        return g.active == self.p

    def main_phase(self, g: Game) -> bool:
        return g.sorcery_timing(self.p)

    def opp_end_step(self, g: Game) -> bool:
        return not self.my_turn(g) and g.step_name == "end" and not g.stack

    def cost(self, g: Game, card: Card, mode: str = "normal") -> int:
        base = g._mode_cost(card, mode)
        return base.with_x(0).reduced(g._cost_reduction(self.p, card)).mana_value

    def stack_top(self, g: Game):
        return g.stack[-1] if g.stack else None

    def building(self, g: Game, d: Decision) -> str:
        """Name of the spell/ability whose targets or costs are being chosen."""
        m = _FOR.search(d.prompt)
        if m:
            return m.group(1).split(":")[0].split("#")[0].strip()
        top = self.stack_top(g)
        return top.name.split(":")[0] if top else ""

    @staticmethod
    def ref_card(g: Game, o: Option) -> Card | None:
        v = o.value
        if isinstance(v, Card):
            return v
        if isinstance(v, tuple) and len(v) == 2 and v[0] == "perm":
            return g.perm(v[1])
        return None

    @staticmethod
    def ref_spell(g: Game, o: Option):
        v = o.value
        if isinstance(v, tuple) and len(v) == 2 and v[0] == "stack":
            return g.stack_item(v[1])
        return None

    # -- valuations (decks refine) ------------------------------------------

    def creature_value(self, g: Game, c: Card) -> float:
        v = g.power(c) * 1.5 + g.toughness(c) * 0.5
        if g.has(c, "flying"):
            v += 1.5
        if c.is_token and g.power(c) == 0:
            v = 0.3
        return v

    def card_value(self, g: Game, name: str) -> float:
        """How much the bot wants a card in hand (discard / put-back order)."""
        return 1.0

    def land_need(self, g: Game) -> float:
        """> 0 when the bot wants more lands."""
        have = len(self.lands(g, self.p)) + sum(1 for c in self.hand(g) if c.face.is_type("Land"))
        return 4.5 - have

    # -- combat (shared) ----------------------------------------------------

    def _blockers_for(self, g: Game, attacker: Card) -> list[Card]:
        out = []
        for b in self.creatures(g, self.opp):
            if b.tapped or g.has(attacker, "unblockable"):
                continue
            if g.has(attacker, "flying") and not (g.has(b, "flying") or g.has(b, "reach")):
                continue
            out.append(b)
        return out

    def attack_value(self, g: Game, a: Card) -> float:
        pw = g.power(a)
        if pw <= 0:
            return NEG
        tough = g.toughness(a) - a.damage
        worst = 1.0  # unblocked: deal damage
        for b in self._blockers_for(g, a):
            kills_me = g.power(b) >= tough or g.has(b, "deathtouch")
            i_kill = pw >= g.toughness(b) - b.damage or g.has(a, "deathtouch")
            if kills_me and not i_kill:
                return -1.0  # they eat it for free
            if kills_me and i_kill and self.creature_value(g, a) > self.creature_value(g, b) + 0.5:
                worst = min(worst, -0.5)
        # Keep a blocker home when their crack-back would be lethal.
        their_power = sum(g.power(c) for c in self.creatures(g, self.opp))
        if their_power >= self.me(g).life and not g.has(a, "flying"):
            my_blockers = [c for c in self.creatures(g, self.p) if not c.tapped and c is not a]
            if len(my_blockers) < len(self.creatures(g, self.opp)):
                return -1.0
        return worst

    def score_declare_attacker(self, g: Game, d: Decision, o: Option) -> float:
        if o.value is None:
            return 0.0
        eligible = [c for c in self.creatures(g, self.p) if not c.tapped and not c.sick and c.oid not in g.attackers and c.name == o.key[1]]
        return max((self.attack_value(g, c) for c in eligible), default=NEG)

    def score_declare_blocker(self, g: Game, d: Decision, o: Option) -> float:
        b = self._blocker_from_prompt(g, d)
        incoming = [g.perm(a) for a in g.attackers if a not in g.blocked]
        unblocked = sum(g.power(a) for a in incoming if a is not None)
        lethal = unblocked >= self.me(g).life
        if o.value is None:
            return 0.0
        a = o.value
        if b is None:
            return NEG
        pa, pb = g.power(a), g.power(b)
        b_dies = pa >= g.toughness(b) - b.damage or g.has(a, "deathtouch")
        a_dies = pb >= g.toughness(a) - a.damage or g.has(b, "deathtouch")
        if a.oid in g.blocked:  # already blocked: only gang up to finish it
            others = sum(g.power(g.perm(x)) for x, y in g.blocks.items() if y == a.oid and g.perm(x) is not None)
            return 0.5 if pb + others >= g.toughness(a) - a.damage and not b_dies else -1.0
        va, vb = self.creature_value(g, a), self.creature_value(g, b)
        if a_dies and not b_dies:
            return 10 + va
        if not b_dies:
            return 0.5 + pa * 0.1
        if a_dies:
            return va - vb + (5 if lethal else 0.1)
        return (8 + pa - vb) if lethal else -vb

    def _blocker_from_prompt(self, g: Game, d: Decision) -> Card | None:
        m = re.search(r"#(\d+)", d.prompt)
        return g.perm(int(m.group(1))) if m else None

    def score_assign_damage(self, g: Game, d: Decision, o: Option) -> float:
        a = g.perm(int(re.search(r"#(\d+)", d.prompt).group(1)))
        blockers = [g.perm(b) for b, at in g.blocks.items() if at == a.oid]
        blockers = [b for b in blockers if b is not None]
        split = o.value
        s = 0.0
        for n, b in zip(split, blockers):
            if n >= g._lethal(a, b):
                s += 10 + self.creature_value(g, b)
        if len(split) > len(blockers):
            s += split[-1]
        return s

    # -- costs and choices (shared) -----------------------------------------

    def source_cost(self, g: Game, card: Card, color: str) -> float:
        """How much paying with this source hurts; lowest is tapped first."""
        if card.is_token:
            return 5.0  # Eldrazi Spawn: a body, keep it unless needed
        colors = next((ab.mana for ab in card.face.abilities if ab.mana), ())
        return len(colors)

    def score_pay_mana(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[1] == "pool":
            return 10.0  # floating mana empties anyway
        return -self.source_cost(g, o.value[1], o.value[2])

    def sac_cost(self, g: Game, c: Card) -> float:
        if c.is_token and g.is_creature(c):
            return 0.0
        if c.is_token:
            return 0.5
        if g.is_land(c):
            return 6.0 if len(self.lands(g, self.p)) >= 5 else 20.0
        if g.is_creature(c):
            return 3 + self.creature_value(g, c)
        return 2.0

    def score_sacrifice(self, g: Game, d: Decision, o: Option) -> float:
        return -self.sac_cost(g, o.value)

    def score_exile_from_graveyard(self, g: Game, d: Decision, o: Option) -> float:
        if o.key[0] == "exile_any_gy":
            return self.exile_any_value(g, o)
        c = o.value
        if c.face.is_type("Land"):
            return 3.0
        if c.face.is_type("Instant") or c.face.is_type("Sorcery"):
            return 1.0
        return 2.0

    def exile_any_value(self, g: Game, o: Option) -> float:
        """Faerie Macabre: exile up to two cards from any graveyard. Stop
        (0.5) unless the card feeds the opponent: spells (Terror, Serpent,
        Guttersnipe-style discounts) and cards castable from the graveyard."""
        c = o.value
        if c is None:
            return 0.5
        if c.owner == self.p:
            return NEG
        f = c.face
        if f.flashback is not None or f.escape is not None or any(t.event == "third_draw" for t in f.triggers):
            return 4.0
        if f.is_type("Instant") or f.is_type("Sorcery"):
            return 3.0
        return 0.2

    def score_choose_x(self, g: Game, d: Decision, o: Option) -> float:
        return o.value

    def score_order_triggers(self, g: Game, d: Decision, o: Option) -> float:
        return 0.0

    def score_choose_card(self, g: Game, d: Decision, o: Option) -> float:
        verb = o.key[0]
        name = o.key[1]
        if verb == "search":
            return NEG if name is None else self.search_value(g, name)
        # discard / put back: get rid of the least valuable card
        return -self.card_value(g, name)

    def search_value(self, g: Game, name: str) -> float:
        return 1.0

    def score_yes_no(self, g: Game, d: Decision, o: Option) -> float:
        tag, ans = o.key
        if tag == "pay_optional":
            return 1.0 if ans == "yes" else 0.0
        if tag == "reveal":  # only worth it if it flips Delver: an instant/sorcery
            top = self.me(g).library[0] if self.me(g).library else None  # known: the bot just looked at it
            flips = top is not None and (top.face.is_type("Instant") or top.face.is_type("Sorcery"))
            return 1.0 if (ans == "yes") == flips else 0.0
        if tag == "explore":
            top = self.me(g).library[0]  # revealed by explore
            keep = self.card_value(g, top.name) >= 2
            return 1.0 if (ans == "keep") == keep else 0.0
        if tag == "shuffle":
            return 0.0  # handled with the order choice; never shuffle by default
        return 0.0

    def score_order(self, g: Game, d: Decision, o: Option) -> float:
        names = o.key[1:]
        return sum(self.card_value(g, n) * w for n, w in zip(names, (3, 2, 1)))

    def score_choose_mode(self, g: Game, d: Decision, o: Option) -> float:
        tag, ans = o.key
        if tag == "scry":
            top = self.me(g).library[0]  # being scried: known to the bot
            keep = self.card_value(g, top.name) >= 2
            return 1.0 if (ans == "top") == keep else 0.0
        return 0.0

    def score_target(self, g: Game, d: Decision, o: Option) -> float:
        return 0.0

    def score_mulligan(self, g: Game, d: Decision, o: Option) -> float:
        keep = self.keep_hand(g, 7 - g.mulligans_taken[self.p])
        return 1.0 if (o.key[1] == "keep") == keep else 0.0

    def keep_hand(self, g: Game, size: int) -> bool:
        """`size`: cards left after bottoming. Decks override."""
        lands = sum(1 for c in self.hand(g) if c.face.is_type("Land"))
        return size <= 5 or 2 <= lands <= 5

    def score_priority(self, g: Game, d: Decision, o: Option) -> float:
        return 0.0
