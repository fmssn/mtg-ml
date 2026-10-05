"""Rules engine core: turn structure, priority, the stack, casting and cost
payment, state-based actions, triggered abilities and combat.

The game is written as one Python generator. Whenever a player has to make a
choice the generator yields a `Decision`; `Game.step(i)` sends the chosen
option back in. Every choice the rules give a player is an explicit decision
(including mana payment, sacrifice choices, trigger ordering, attack/block
declarations and combat damage assignment). Options are filtered so that every
offered option can be completed legally: there are no dead ends.

Two equivalence reductions keep the option lists free of duplicates without
removing any strategically distinct choice:
  * objects that are indistinguishable (same name, controller and state, not
    referenced by anything) are offered once;
  * plain lands are only tapped for mana while paying a cost, never "floated"
    at priority (floating mana from a land has no effect a payment-time tap
    does not have). Mana abilities with side effects (sacrificing an Eldrazi
    Spawn) are offered at priority.

Replay determinism: all randomness comes from `self.rng`, iteration is over
lists/dicts only, and `fork()` rebuilds a game from its constructor arguments
and action history.
"""

from __future__ import annotations

import inspect
import itertools
import random
from typing import Callable, Iterable

from . import objects as O
from .mana import ManaCost, RemainingCost, can_pay, pool_units
from .objects import Card, Decision, Option, PendingTrigger, Player, StackItem, TargetSpec, TempEffect, TriggerDef

STEPS = (
    "untap",
    "upkeep",
    "draw",
    "main1",
    "begin_combat",
    "declare_attackers",
    "declare_blockers",
    "combat_damage",
    "end_combat",
    "main2",
    "end",
    "cleanup",
)

MAX_HAND = 7


class GameOver(Exception):
    def __init__(self, winner: int | None, reason: str):
        super().__init__(reason)
        self.winner = winner
        self.reason = reason


class RulesError(Exception):
    """Engine invariant violated (a bug, never a legal game situation)."""


class Game:
    def __init__(
        self,
        decks: tuple[list[str], list[str]],
        seed: int = 0,
        starting_player: int | None = None,
        auto_single: bool = True,
        max_turns: int = 100,
        log: bool = False,
        setup: Callable[["Game"], None] | None = None,
        start_step: str = "untap",
        mulligans: bool = True,
        match_game: int = 1,
    ):
        from .cards import CARDS  # local import: cards module imports engine helpers

        self._args = dict(
            decks=decks,
            seed=seed,
            starting_player=starting_player,
            auto_single=auto_single,
            max_turns=max_turns,
            log=log,
            setup=setup,
            start_step=start_step,
            mulligans=mulligans,
            match_game=match_game,
        )
        self.match_game = match_game  # 1 = preboard, 2/3 = after sideboarding
        self.cards_db = CARDS
        self.rng = random.Random(seed)
        self.auto_single = auto_single
        self.max_turns = max_turns
        self.logging = log
        self.log: list[str] = []
        self.actions: list[int] = []
        self._next_id = 1
        self.players = [Player(0), Player(1)]
        self.battlefield: list[Card] = []
        self.stack: list[StackItem] = []
        self.pending: list[PendingTrigger] = []
        self.turn = 0
        self.step_name = "untap"
        self.lands_played = 0
        self.attackers: list[int] = []
        self.blocked: set[int] = set()
        self.blocks: dict[int, int] = {}  # blocker oid -> attacker oid
        self.winner: int | None = None
        self.over = False
        self.end_reason = ""
        self.decision: Decision | None = None

        for p, deck in zip(self.players, decks):
            for name in deck:
                p.library.append(self._new_card(self.cards_db[name], p.idx, "library"))
        if starting_player is None:
            starting_player = self.rng.randrange(2)
        self.active = starting_player
        self.starting_player = starting_player
        if setup is None:
            for p in self.players:
                self.rng.shuffle(p.library)
            for p in self.players:
                self.draw(p.idx, 7, count=False)
        else:
            setup(self)
        self._skip_first_draw = setup is None
        self._mulligan_phase = setup is None and mulligans
        self.mulligans_taken = [0, 0]
        self._gen = self._main(start_step)
        self._advance(None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def legal_options(self) -> list[Option]:
        return [] if self.decision is None else self.decision.options

    def step(self, index: int) -> None:
        if self.over or self.decision is None:
            raise RulesError("game is over")
        opts = self.decision.options
        if not 0 <= index < len(opts):
            raise IndexError(f"option {index} out of range ({len(opts)})")
        self.actions.append(index)
        if self.logging:
            self._log(f"  p{self.decision.player} {self.decision.kind}: {opts[index].label}")
        self._advance(opts[index].value)

    def fork(self) -> "Game":
        """Exact copy by deterministic replay of the action history."""
        g = Game(**self._args)
        for a in self.actions:
            g.step(a)
        return g

    def add_card(
        self,
        name: str,
        player: int,
        zone: str,
        tapped: bool = False,
        sick: bool = False,
        counters: int = 0,
    ) -> Card:
        """Create a card directly in a zone. Meant for `setup` callbacks
        (scenario tests, puzzles); does not emit events."""
        from .cards import CARDS, TOKENS

        defn = CARDS.get(name) or TOKENS[name]
        c = self._new_card(defn, player, zone, token=name in TOKENS)
        if zone == "battlefield":
            c.tapped = tapped
            c.sick = sick
            c.counters = counters
            self.battlefield.append(c)
        else:
            if zone == "hand":
                c.known_to = {player}
            getattr(self.players[player], zone).append(c)
        return c

    # ------------------------------------------------------------------
    # Generator driver
    # ------------------------------------------------------------------

    def _advance(self, value) -> None:
        try:
            if self.decision is None and not self.actions:
                d = next(self._gen)
            else:
                d = self._gen.send(value)
        except GameOver as e:
            self.over = True
            self.winner = e.winner
            self.end_reason = e.reason
            self.decision = None
            self._log(f"GAME OVER: winner={e.winner} ({e.reason})")
            return
        self.decision = d

    def ask(self, player: int, kind: str, prompt: str, options: list[Option]):
        """Yield a decision (generator). Returns the chosen option's value."""
        if not options:
            raise RulesError(f"decision {kind!r} ({prompt}) has no options")
        if self.auto_single and len(options) == 1:
            if self.logging and kind != O.PRIORITY:
                self._log(f"  p{player} {kind}: {options[0].label} (only option)")
            return options[0].value
        value = yield Decision(player, kind, prompt, options)
        return value

    def _log(self, msg: str) -> None:
        if self.logging:
            self.log.append(msg)

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _new_card(self, defn: O.CardDef, owner: int, zone: str, token: bool = False) -> Card:
        i = self._new_id()
        c = Card(uid=i, oid=i, defn=defn, owner=owner, controller=owner, zone=zone, is_token=token)
        if zone in ("battlefield", "graveyard", "exile", "stack"):
            c.known_to = {0, 1}
        return c

    # ------------------------------------------------------------------
    # Turn structure
    # ------------------------------------------------------------------

    def _mulligans(self):
        """London mulligan (CR 103.5) for a two-player game, no free mulligan.

        Starting with the starting player, each player keeps or mulligans;
        everyone who mulliganed shuffles their hand into their library, draws
        seven and decides again. Once all have kept, each player puts one card
        per mulligan taken on the bottom of their library."""
        self.step_name = "mulligan"
        order = [self.starting_player, 1 - self.starting_player]
        deciding = list(order)
        while deciding:
            again = []
            for p in deciding:
                n = self.mulligans_taken[p]
                options = [Option(f"Keep ({7 - n} cards)", ("mulligan", "keep"), False)]
                if n < 6:
                    options.append(Option(f"Mulligan (to {6 - n})", ("mulligan", "mulligan"), True))
                if (yield from self.ask(p, O.MULLIGAN, f"Opening hand, {n} mulligan(s) taken: keep {7 - n}?", options)):
                    again.append(p)
            for p in again:
                self.mulligans_taken[p] += 1
                self._log(f"p{p} mulligans ({self.mulligans_taken[p]})")
                for c in list(self.players[p].hand):
                    self._move(c, "library", position="bottom")
                self.shuffle(p)
                self.draw(p, 7, count=False)
            deciding = again
        for p in order:
            n = self.mulligans_taken[p]
            for i in range(n):
                c = yield from self.ask(p, O.CHOOSE_CARD, f"Mulligan: put a card on the bottom of your library ({i + 1}/{n})", self._hand_card_options(p, "bottom"))
                self._move(c, "library", position="bottom", known_to={p})

    def _main(self, start_step: str):
        if self._mulligan_phase:
            yield from self._mulligans()
        first = True
        while True:
            self.turn += 1
            if self.turn > self.max_turns:
                raise GameOver(None, "turn limit")
            if not (first and self._args["setup"] is not None):
                self._begin_turn()  # a scenario setup defines the opening turn state itself
            start = start_step if first else "untap"
            skip_draw = first and self._skip_first_draw
            yield from self._run_turn(start, skip_draw)
            first = False
            self.active = 1 - self.active

    def _begin_turn(self) -> None:
        self.lands_played = 0
        for p in self.players:
            p.cards_drawn_this_turn = 0
        self._clear_combat()
        self._log(f"=== Turn {self.turn}: player {self.active} ===")

    def _run_turn(self, start: str, skip_draw: bool):
        for name in STEPS[STEPS.index(start):]:
            if name in ("declare_blockers", "combat_damage") and not self.attackers:
                continue
            self.step_name = name
            self._log(f"-- {name}")
            if name == "untap":
                self._untap_step()
            elif name == "upkeep":
                self._emit_upkeep()
                yield from self._priority_round()
            elif name == "draw":
                if not skip_draw:
                    self.draw(self.active, 1)
                yield from self._priority_round()
            elif name == "declare_attackers":
                yield from self._declare_attackers()
                yield from self._priority_round()
            elif name == "declare_blockers":
                yield from self._declare_blockers()
                yield from self._priority_round()
            elif name == "combat_damage":
                yield from self._combat_damage()
                yield from self._priority_round()
            elif name == "end_combat":
                yield from self._priority_round()
                self._clear_combat()
            elif name == "cleanup":
                yield from self._cleanup_step()
            else:  # main1, begin_combat, main2, end
                yield from self._priority_round()
            self._empty_pools()

    def _untap_step(self) -> None:
        for c in self.battlefield:
            if c.controller != self.active:
                continue
            c.sick = False
            if c.skip_untap > 0:
                c.skip_untap -= 1
            else:
                c.tapped = False

    def _cleanup_step(self):
        while True:
            p = self.players[self.active]
            while len(p.hand) > MAX_HAND:
                card = yield from self.ask(
                    p.idx, O.CHOOSE_CARD, f"Discard to hand size ({len(p.hand)}/{MAX_HAND})", self._hand_card_options(p.idx, "discard")
                )
                self.discard(card)
            for c in self.battlefield:
                c.damage = 0
                c.deathtouch_damage = False
                c.temp = []
            changed = self._sba()
            if changed or self.pending:
                yield from self._priority_round()
                self._empty_pools()
                continue
            return

    def _empty_pools(self) -> None:
        for p in self.players:
            p.pool = {}

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------

    def _priority_round(self):
        p = self.active
        passes = 0
        while True:
            yield from self._sba_and_triggers()
            options = self._priority_options(p)
            act = yield from self.ask(p, O.PRIORITY, f"Priority ({self.step_name})", options)
            if act[0] == "pass":
                passes += 1
                if passes >= 2:
                    if not self.stack:
                        return
                    yield from self._resolve_top()
                    passes = 0
                    p = self.active
                else:
                    p = 1 - p
            else:
                yield from self._take_action(p, act)
                passes = 0

    def _sba_and_triggers(self):
        while True:
            changed = self._sba()
            if self.pending:
                yield from self._put_triggers_on_stack()
                continue
            if not changed:
                return

    def sorcery_timing(self, p: int) -> bool:
        return p == self.active and self.step_name in ("main1", "main2") and not self.stack

    def _priority_options(self, p: int) -> list[Option]:
        pl = self.players[p]
        opts = [Option("Pass priority", ("pass",), ("pass",))]
        sorcery_ok = self.sorcery_timing(p)
        if sorcery_ok and self.lands_played < 1:
            for card in self._dedupe_by_name(c for c in pl.hand if c.face.is_type("Land")):
                opts.append(Option(f"Play {card.name}", ("play_land", card.name), ("land", card)))
        for card in self._dedupe_by_name(pl.hand):
            for i, sm in enumerate(card.face.modes):
                if self._can_cast(p, card, "normal", i):
                    opts.append(Option(f"Cast {card.name} ({sm.name})", ("cast", card.name, "hand", "normal", sm.name), ("cast", card, "normal", i)))
            if card.face.modes:
                continue
            for mode in ("normal", "bestow"):
                if self._can_cast(p, card, mode):
                    label = f"Cast {card.name}" + ("" if mode == "normal" else f" ({mode})")
                    opts.append(Option(label, ("cast", card.name, "hand", mode), ("cast", card, mode)))
        for card in self._dedupe_by_name(pl.graveyard):
            for mode in ("flashback", "escape"):
                if self._can_cast(p, card, mode):
                    opts.append(Option(f"Cast {card.name} ({mode})", ("cast", card.name, "graveyard", mode), ("cast", card, mode)))
        for card in self._dedupe_by_name(pl.hand):
            for i, ab in enumerate(card.face.abilities):
                if ab.zone == "hand" and self._can_activate(p, card, ab):
                    opts.append(Option(f"{card.name}: {ab.name}", ("activate", card.name, ab.name), ("activate", card, i)))
        for card in self._dedupe_by_equiv(c for c in self.battlefield if c.controller == p):
            for i, ab in enumerate(card.face.abilities):
                if ab.zone != "battlefield":
                    continue
                if ab.mana is not None:
                    # Only mana abilities with side effects are worth offering at priority.
                    if ab.sac_self and self._can_activate(p, card, ab):
                        opts.append(Option(f"{card.name}: {ab.name}", ("mana", card.name, ab.name), ("mana", card, i)))
                elif self._can_activate(p, card, ab):
                    opts.append(Option(f"{card.name}: {ab.name}", ("activate", card.name, ab.name), ("activate", card, i)))
        return opts

    def _take_action(self, p: int, act):
        kind = act[0]
        if kind == "land":
            card = act[1]
            self.lands_played += 1
            self._log(f"p{p} plays {card.name}")
            self.put_onto_battlefield(card, p)
        elif kind == "cast":
            yield from self._cast(p, act[1], act[2], act[3] if len(act) > 3 else None)
        elif kind == "activate":
            yield from self._activate(p, act[1], act[2])
        elif kind == "mana":
            card, i = act[1], act[2]
            ab = card.face.abilities[i]
            color = ab.mana[0]
            self._activate_mana_ability(card, ab)
            pool = self.players[p].pool
            pool[color] = pool.get(color, 0) + 1
        else:
            raise RulesError(f"unknown action {act}")

    # ------------------------------------------------------------------
    # Characteristics
    # ------------------------------------------------------------------

    def perm(self, oid: int) -> Card | None:
        for c in self.battlefield:
            if c.oid == oid:
                return c
        return None

    def live(self, card: Card) -> Card | None:
        """The object on the battlefield that `card` (possibly an LKI copy) refers to."""
        return self.perm(card.oid)

    def stack_item(self, sid: int) -> StackItem | None:
        for it in self.stack:
            if it.sid == sid:
                return it
        return None

    def is_bestowed(self, c: Card) -> bool:
        return c.zone == "battlefield" and c.attached_to is not None

    def types(self, c: Card) -> set[str]:
        t = set(c.face.types)
        if self.is_bestowed(c):
            t.discard("Creature")
        return t

    def is_creature(self, c: Card) -> bool:
        return "Creature" in c.face.types and not self.is_bestowed(c)

    def is_artifact(self, c: Card) -> bool:
        return "Artifact" in c.face.types

    def is_land(self, c: Card) -> bool:
        return "Land" in c.face.types

    def _auras_on(self, c: Card) -> list[Card]:
        return [a for a in self.battlefield if a.attached_to == c.oid]

    def power(self, c: Card) -> int:
        v = (c.face.power or 0) + c.counters + sum(t.power for t in c.temp)
        v += sum(a.counters for a in self._auras_on(c))
        return v

    def toughness(self, c: Card) -> int:
        v = (c.face.toughness or 0) + c.counters + sum(t.toughness for t in c.temp)
        v += sum(a.counters for a in self._auras_on(c))
        return v

    def keywords(self, c: Card) -> set[str]:
        k = set(c.face.keywords)
        for t in c.temp:
            k |= t.keywords
        for a in self._auras_on(c):
            k |= {"reach", "trample"}
        return k

    def has(self, c: Card, kw: str) -> bool:
        return kw in self.keywords(c)

    def _referenced_oids(self) -> set[int]:
        refs: set[int] = set()
        for it in self.stack:
            for t in it.targets:
                if t[0] == "perm":
                    refs.add(t[1])
        refs.update(self.blocks.keys())
        refs.update(self.blocks.values())
        for c in self.battlefield:
            if c.attached_to is not None:
                refs.add(c.attached_to)
        return refs

    def equiv_key(self, c: Card, refs: set[int] | None = None) -> tuple:
        """Two permanents with the same key are interchangeable for every rule."""
        if refs is None:
            refs = self._referenced_oids()
        if c.oid in refs or c.attached_to is not None:
            return ("unique", c.oid)
        return (
            c.name,
            c.controller,
            c.is_token,
            c.tapped,
            c.damage,
            c.deathtouch_damage,
            c.counters,
            c.sick,
            c.skip_untap,
            c.transformed,
            tuple((tuple(sorted(t.keywords)), t.power, t.toughness) for t in c.temp),
            c.oid in self.attackers,
        )

    def _dedupe_by_equiv(self, cards: Iterable[Card]) -> list[Card]:
        refs = self._referenced_oids()
        seen = set()
        out = []
        for c in cards:
            k = self.equiv_key(c, refs)
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out

    @staticmethod
    def _dedupe_by_name(cards: Iterable[Card]) -> list[Card]:
        seen = set()
        out = []
        for c in cards:
            if c.name not in seen:
                seen.add(c.name)
                out.append(c)
        return out

    def _hand_card_options(self, p: int, verb: str) -> list[Option]:
        return [
            Option(f"{verb.capitalize()} {c.name}", (verb, c.name), c) for c in self._dedupe_by_name(self.players[p].hand)
        ]

    # ------------------------------------------------------------------
    # Zones
    # ------------------------------------------------------------------

    def _zone_list(self, card: Card) -> list:
        if card.zone == "battlefield":
            return self.battlefield
        if card.zone == "stack":
            return []
        return getattr(self.players[card.owner], card.zone)

    def _move(
        self,
        card: Card,
        to: str,
        controller: int | None = None,
        position: str | int = "top",
        known_to: set[int] | None = None,
        tapped: bool = False,
    ) -> Card | None:
        """Move `card` to zone `to`. Returns the card (new object), or None if
        it was a token that ceased to exist."""
        frm = card.zone
        lki = card.snapshot() if frm == "battlefield" else None
        if frm == "battlefield":
            self.battlefield.remove(card)
            self._remove_from_combat(card.oid)
        elif frm != "stack":
            self._zone_list(card).remove(card)
        prev_known = set(card.known_to)
        if card.is_token and to != "battlefield":
            card.zone = "gone"
            if lki is not None:
                self._after_leave_battlefield(lki, card, to)
            return None
        card.reset_state()
        card.oid = self._new_id()
        card.zone = to
        card.controller = card.owner if controller is None else controller
        if to == "battlefield":
            card.known_to = {0, 1}
            card.sick = True
            card.tapped = tapped or card.face.enters_tapped
            self.battlefield.append(card)
        elif to == "library":
            lib = self.players[card.owner].library
            card.known_to = set(known_to or ())
            if position == "top":
                lib.insert(0, card)
            elif position == "bottom":
                lib.append(card)
            else:
                lib.insert(min(int(position), len(lib)), card)
        elif to == "hand":
            card.known_to = prev_known | {card.owner} | set(known_to or ())
            self.players[card.owner].hand.append(card)
        elif to in ("graveyard", "exile"):
            card.known_to = {0, 1}
            getattr(self.players[card.owner], to).append(card)
        elif to == "stack":
            card.known_to = {0, 1}
        else:
            raise RulesError(f"unknown zone {to}")
        if lki is not None:
            self._after_leave_battlefield(lki, card, to)
        return card

    def _after_leave_battlefield(self, lki: Card, new: Card, to: str) -> None:
        if to == "graveyard":
            for t in lki.face.triggers:
                if t.event == "to_graveyard_from_battlefield":
                    self.pending.append(PendingTrigger(lki.controller, lki, t, {"card": new, "oid": new.oid}))

    def draw(self, p: int, n: int = 1, count: bool = True) -> None:
        pl = self.players[p]
        for _ in range(n):
            if not pl.library:
                pl.drew_from_empty = True
                self._log(f"p{p} draws from an empty library")
                continue
            card = pl.library[0]
            self._move(card, "hand")
            if count:
                pl.cards_drawn_this_turn += 1

    def mill(self, p: int, n: int) -> None:
        pl = self.players[p]
        for _ in range(n):
            if not pl.library:
                return
            self._move(pl.library[0], "graveyard")

    def discard(self, card: Card) -> None:
        self._log(f"p{card.owner} discards {card.name}")
        self._move(card, "graveyard")

    def shuffle(self, p: int) -> None:
        lib = self.players[p].library
        self.rng.shuffle(lib)
        for c in lib:
            c.known_to = set()

    def gain_life(self, p: int, n: int) -> None:
        self.players[p].life += n

    def sacrifice(self, card: Card) -> None:
        lki = card.snapshot()
        self._log(f"p{card.controller} sacrifices {card.name}")
        self._move(card, "graveyard")
        self._emit_sacrifice(lki)

    def destroy(self, card: Card) -> bool:
        if self.has(card, "indestructible"):
            return False
        self._move(card, "graveyard")
        return True

    def create_token(self, p: int, name: str) -> Card:
        from .cards import TOKENS

        c = self._new_card(TOKENS[name], p, "battlefield", token=True)
        c.sick = True
        self.battlefield.append(c)
        self._emit_etb(c)
        return c

    def put_onto_battlefield(self, card: Card, controller: int, tapped: bool = False) -> Card:
        new = self._move(card, "battlefield", controller=controller, tapped=tapped)
        self._emit_etb(new)
        return new

    def deal_damage(self, source: Card, target: tuple, amount: int) -> None:
        """target: ('player', idx) or ('perm', oid)."""
        if amount <= 0:
            return
        kws = self.keywords(source)
        if target[0] == "player":
            self.players[target[1]].life -= amount
        else:
            c = self.perm(target[1])
            if c is None or not self.is_creature(c):
                return
            c.damage += amount
            if "deathtouch" in kws:
                c.deathtouch_damage = True
        if "lifelink" in kws:
            self.players[source.controller].life += amount

    # ------------------------------------------------------------------
    # Events and triggers
    # ------------------------------------------------------------------

    def _emit_etb(self, card: Card) -> None:
        for t in card.face.triggers:
            if t.event == "etb":
                self.pending.append(PendingTrigger(card.controller, card, t))

    def _emit_sacrifice(self, lki: Card) -> None:
        for perm in self.battlefield:
            if perm.controller != lki.controller or perm.oid == lki.oid:
                continue
            for t in perm.face.triggers:
                if t.event == "you_sacrifice_another" and (t.condition is None or t.condition(self, perm, lki)):
                    self.pending.append(PendingTrigger(perm.controller, perm, t, {"sacrificed": lki}))

    def _emit_upkeep(self) -> None:
        for perm in self.battlefield:
            if perm.controller != self.active:
                continue
            for t in perm.face.triggers:
                if t.event == "your_upkeep":
                    self.pending.append(PendingTrigger(perm.controller, perm, t))

    def _emit_cast(self, item: StackItem) -> None:
        for t in item.card.face.triggers:
            if t.event == "cast":
                self.pending.append(PendingTrigger(item.controller, item.card, t, {"spell_sid": item.sid}))

    def _emit_targeted(self, item: StackItem) -> None:
        for ref in item.targets:
            if ref[0] != "perm":
                continue
            c = self.perm(ref[1])
            if c is not None and c.face.ward and item.controller != c.controller:
                self.pending.append(PendingTrigger(c.controller, c, WARD_TRIGGER, {"sid": item.sid, "amount": c.face.ward}))

    def _put_triggers_on_stack(self):
        pend, self.pending = self.pending, []
        for p in (self.active, 1 - self.active):
            mine = [t for t in pend if t.controller == p]
            while mine:
                options = []
                seen = set()
                for t in mine:
                    key = (t.source.name, t.tdef.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    options.append(Option(f"Put on stack: {t.source.name} - {t.tdef.name}", ("trigger",) + key, t))
                t = yield from self.ask(p, O.ORDER_TRIGGERS, "Choose the next trigger to put on the stack (first = resolves last)", options)
                mine.remove(t)
                item = StackItem(
                    sid=self._new_id(),
                    kind="trigger",
                    controller=t.controller,
                    name=f"{t.source.name}: {t.tdef.name}",
                    effect=t.tdef.effect,
                    source=t.source,
                    data=dict(t.data),
                )
                self.stack.append(item)
                self._log(f"trigger -> stack: {item.name}")

    # ------------------------------------------------------------------
    # State-based actions
    # ------------------------------------------------------------------

    def _sba(self) -> bool:
        changed = False
        losers = [p.idx for p in self.players if p.life <= 0 or p.drew_from_empty]
        if losers:
            winner = None if len(losers) == 2 else 1 - losers[0]
            raise GameOver(winner, "life" if any(self.players[i].life <= 0 for i in losers) else "decking")
        for c in self.battlefield:
            if c.attached_to is not None:
                host = self.perm(c.attached_to)
                if host is None or not self.is_creature(host):
                    c.attached_to = None  # bestow: becomes a creature again
                    changed = True
        dying = []
        for c in self.battlefield:
            if not self.is_creature(c):
                continue
            t = self.toughness(c)
            if t <= 0:
                dying.append(c)
            elif (c.damage >= t or (c.deathtouch_damage and c.damage > 0)) and not self.has(c, "indestructible"):
                dying.append(c)
        for c in dying:
            self._log(f"SBA: {c.name} dies")
            self._move(c, "graveyard")
            changed = True
        return changed

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def target_candidates(self, spec: TargetSpec, controller: int, exclude_sid: int | None = None) -> list[tuple]:
        k = spec.kind
        if k == "player":
            return [("player", controller), ("player", 1 - controller)]
        if k == "opponent":
            return [("player", 1 - controller)]
        if k.endswith("spell"):
            return [("stack", it.sid) for it in self.stack if it.kind == "spell" and it.sid != exclude_sid and self._spell_matches(spec, it)]
        out = []
        for c in self.battlefield:
            if self._perm_matches(spec, c, controller):
                out.append(("perm", c.oid))
        if k == "any":
            out += [("player", controller), ("player", 1 - controller)]
        return out

    def _perm_matches(self, spec: TargetSpec, c: Card, controller: int) -> bool:
        k = spec.kind
        if k in ("creature", "any"):
            return self.is_creature(c)
        if k == "nonlegendary_creature":
            return self.is_creature(c) and "Legendary" not in c.face.supertypes
        if k == "creature_you_control":
            return self.is_creature(c) and c.controller == controller
        if k == "land":
            return self.is_land(c)
        if k == "nonland_permanent":
            return not self.is_land(c)
        if k == "nonartifact_creature":
            return self.is_creature(c) and not self.is_artifact(c)
        if k == "artifact":
            return self.is_artifact(c)
        if k == "blue_permanent":
            return "U" in c.face.colors
        if k == "red_permanent":
            return "R" in c.face.colors
        return False

    def _spell_matches(self, spec: TargetSpec, it: StackItem) -> bool:
        d = it.card.face
        k = spec.kind
        if k == "spell":
            return True
        if k == "blue_spell":
            return "U" in d.colors
        if k == "red_spell":
            return "R" in d.colors
        if k == "instant_spell":
            return d.is_type("Instant")
        if k == "artifact_spell":
            return d.is_type("Artifact")
        return False

    def target_legal(self, item: StackItem, i: int) -> bool:
        spec = item.target_specs[i]
        ref = item.targets[i]
        if ref[0] == "player":
            return spec.kind in ("player", "any") or (spec.kind == "opponent" and ref[1] != item.controller)
        if ref[0] == "stack":
            it = self.stack_item(ref[1])
            return it is not None and it.kind == "spell" and spec.kind.endswith("spell") and self._spell_matches(spec, it)
        c = self.perm(ref[1])
        return c is not None and self._perm_matches(spec, c, item.controller)

    def target(self, item: StackItem, i: int = 0):
        """Resolved target i if still legal: a Card, StackItem or ('player', idx)."""
        if i >= len(item.targets) or not self.target_legal(item, i):
            return None
        ref = item.targets[i]
        if ref[0] == "perm":
            return self.perm(ref[1])
        if ref[0] == "stack":
            return self.stack_item(ref[1])
        return ref

    def describe_ref(self, ref: tuple, viewer: int) -> tuple[str, tuple]:
        """(label, stable key) for a target reference."""
        if ref[0] == "player":
            rel = "self" if ref[1] == viewer else "opponent"
            return f"player {ref[1]} ({rel})", ("player", rel)
        if ref[0] == "stack":
            it = self.stack_item(ref[1])
            rel = "self" if it.controller == viewer else "opponent"
            return f"spell {it.name}#{it.sid} ({rel})", ("spell", rel, it.name)
        c = self.perm(ref[1])
        rel = "self" if c.controller == viewer else "opponent"
        return f"{c.name}#{c.oid} ({rel})", ("perm", rel, c.name)

    def _choose_targets(self, p: int, item: StackItem):
        for spec in item.target_specs:
            cands = self.target_candidates(spec, p, exclude_sid=item.sid)
            # Dedupe interchangeable permanents.
            refs = self._referenced_oids()
            seen = set()
            options = []
            for ref in cands:
                if ref[0] == "perm":
                    k = ("perm", self.equiv_key(self.perm(ref[1]), refs))
                else:
                    k = ref
                if k in seen:
                    continue
                seen.add(k)
                label, key = self.describe_ref(ref, p)
                options.append(Option(f"Target {label}", ("target", spec.kind) + key, ref))
            ref = yield from self.ask(p, O.TARGET, f"Choose target ({spec.kind}) for {item.name}", options)
            item.targets.append(ref)

    # ------------------------------------------------------------------
    # Costs and mana
    # ------------------------------------------------------------------

    def mana_sources(self, p: int, exclude: set[int] = frozenset()) -> list[tuple[Card, O.AbilityDef]]:
        out = []
        for c in self.battlefield:
            if c.controller != p or c.oid in exclude:
                continue
            for ab in c.face.abilities:
                if ab.mana is None:
                    continue
                if ab.tap and (c.tapped or (self.is_creature(c) and c.sick)):
                    continue
                out.append((c, ab))
                break  # every supported permanent has at most one mana ability
        return out

    def sac_candidates(self, p: int, flt: str, exclude: set[int] = frozenset()) -> list[Card]:
        out = []
        for c in self.battlefield:
            if c.controller != p or c.oid in exclude:
                continue
            if flt == "artifact" and self.is_artifact(c):
                out.append(c)
            elif flt == "artifact_or_creature" and (self.is_artifact(c) or self.is_creature(c)):
                out.append(c)
        return out

    def _cost_feasible(
        self,
        p: int,
        rem: RemainingCost,
        sac_filter: str | None = None,
        exclude: set[int] = frozenset(),
        pool: dict[str, int] | None = None,
        gone: set[int] = frozenset(),
    ) -> bool:
        """Can `rem` be paid from pool + untapped sources (not in `exclude`)
        while leaving a legal choice for an additional sacrifice cost?
        `gone` are permanents a hypothetical payment already sacrificed."""
        pool = self.players[p].pool if pool is None else pool
        sources = self.mana_sources(p, set(exclude) | set(gone))
        normal = [ab.mana for c, ab in sources if not ab.sac_self]
        selfsac = [(c, ab) for c, ab in sources if ab.sac_self]
        base = pool_units(pool) + normal
        if sac_filter is None:
            return can_pay(rem, base + [ab.mana for _, ab in selfsac])
        cands = {c.oid for c in self.sac_candidates(p, sac_filter, gone)}
        if not cands:
            return False
        # All self-sacrificing sources in the pool produce {C}; using k of them
        # is equivalent whichever k we pick, so spend non-candidates first.
        assert all(ab.mana == ("C",) for _, ab in selfsac)
        ordered = [c for c, _ in selfsac if c.oid not in cands] + [c for c, _ in selfsac if c.oid in cands]
        for k in range(len(ordered) + 1):
            used = {c.oid for c in ordered[:k]}
            if len(cands - used) >= 1 and can_pay(rem, base + [("C",)] * k):
                return True
        return False

    def _activate_mana_ability(self, card: Card, ab: O.AbilityDef) -> None:
        if ab.tap:
            card.tapped = True
        if ab.sac_self:
            self.sacrifice(card)

    def _pay_mana(self, p: int, rem: RemainingCost, sac_filter: str | None = None, exclude: set[int] = frozenset(), what: str = ""):
        pl = self.players[p]
        exclude = set(exclude)
        while not rem.is_paid():
            options = []
            for color in sorted(pl.pool):
                if pl.pool[color] <= 0 or not rem.useful(color):
                    continue
                r2 = rem.copy()
                r2.apply(color)
                pool2 = dict(pl.pool)
                pool2[color] -= 1
                if self._cost_feasible(p, r2, sac_filter, exclude, pool2):
                    options.append(Option(f"Pay with floating {color}", ("pay", "pool", color), ("pool", color)))
            for card in self._dedupe_by_equiv(c for c, _ in self.mana_sources(p, exclude)):
                ab = next(a for a in card.face.abilities if a.mana is not None)
                for color in ab.mana:
                    if not rem.useful(color):
                        continue
                    r2 = rem.copy()
                    r2.apply(color)
                    gone = {card.oid} if ab.sac_self else set()
                    if self._cost_feasible(p, r2, sac_filter, exclude | {card.oid}, gone=gone):
                        verb = "Sacrifice" if ab.sac_self else "Tap"
                        options.append(
                            Option(f"{verb} {card.name}#{card.oid} for {color}", ("pay", "source", card.name, color), ("source", card, color))
                        )
            choice = yield from self.ask(p, O.PAY_MANA, f"Pay {rem} for {what}", options)
            if choice[0] == "pool":
                pl.pool[choice[1]] -= 1
                if pl.pool[choice[1]] == 0:
                    del pl.pool[choice[1]]
                rem.apply(choice[1])
            else:
                card, color = choice[1], choice[2]
                ab = next(a for a in card.face.abilities if a.mana is not None)
                self._activate_mana_ability(card, ab)
                if not rem.apply(color):
                    raise RulesError("mana unit could not be applied")

    def can_afford(self, p: int, cost: ManaCost) -> bool:
        return self._cost_feasible(p, RemainingCost.of(cost))

    def optional_payment(self, p: int, cost: ManaCost, prompt: str):
        """'You may pay {cost}': YES/NO decision and payment. Returns True if paid."""
        options = [Option("Don't pay", ("pay_optional", "no"), False)]
        if self.can_afford(p, cost):
            options.append(Option(f"Pay {cost}", ("pay_optional", "yes"), True))
        pay = yield from self.ask(p, O.YES_NO, prompt, options)
        if pay:
            yield from self._pay_mana(p, RemainingCost.of(cost), what=prompt)
        return pay

    def _choose_sacrifice(self, p: int, flt: str, what: str):
        cands = self._dedupe_by_equiv(self.sac_candidates(p, flt))
        options = [Option(f"Sacrifice {c.name}#{c.oid}", ("sacrifice", c.name), c) for c in cands]
        card = yield from self.ask(p, O.SACRIFICE, f"Sacrifice an {flt.replace('_', ' ')} for {what}", options)
        self.sacrifice(card)

    # ------------------------------------------------------------------
    # Casting spells
    # ------------------------------------------------------------------

    def _mode_cost(self, card: Card, mode: str) -> ManaCost | None:
        d = card.face
        if mode == "normal":
            return d.cost
        if mode == "bestow":
            return d.bestow
        if mode == "flashback":
            return d.flashback
        if mode == "escape":
            return d.escape
        return None

    def _mode_targets(self, card: Card, mode: str, choice: int | None = None) -> tuple[TargetSpec, ...]:
        if mode == "bestow":
            return (TargetSpec("creature"),)
        if choice is not None:
            return card.face.modes[choice].targets
        return card.face.targets

    def _cost_reduction(self, p: int, card: Card) -> int:
        f = card.face.cost_reduction
        return f(self, p) if f else 0

    def _can_cast(self, p: int, card: Card, mode: str, choice: int | None = None) -> bool:
        d = card.face
        base = self._mode_cost(card, mode)
        if base is None:
            return False
        if mode in ("normal", "bestow") and card.zone != "hand":
            return False
        if mode in ("flashback", "escape") and card.zone != "graveyard":
            return False
        if d.is_type("Land"):
            return False
        if not d.is_type("Instant") and not self.sorcery_timing(p):
            return False
        for spec in self._mode_targets(card, mode, choice):
            if not self.target_candidates(spec, p):
                return False
        if mode == "escape" and len(self.players[p].graveyard) - 1 < d.escape_exile:
            return False
        cost = base.with_x(0).reduced(self._cost_reduction(p, card))
        return self._cost_feasible(p, RemainingCost.of(cost), d.additional_sac)

    def _cast(self, p: int, card: Card, mode: str, choice: int | None = None):
        d = card.face
        from_zone = card.zone
        item = StackItem(
            sid=0,
            kind="spell",
            controller=p,
            name=card.name,
            effect=d.effect if choice is None else d.modes[choice].effect,
            target_specs=self._mode_targets(card, mode, choice),
            card=card,
            method=mode,
            cast_from=from_zone,
        )
        self._move(card, "stack", controller=p)
        item.sid = card.oid
        self.stack.append(item)
        self._log(f"p{p} casts {card.name} ({mode}) from {from_zone}")
        base = self._mode_cost(card, mode)
        reduction = self._cost_reduction(p, card)
        if base.x:
            xs = []
            x = 0
            while self._cost_feasible(p, RemainingCost.of(base.with_x(x).reduced(reduction)), d.additional_sac):
                xs.append(x)
                x += 1
            item.x = yield from self.ask(p, O.CHOOSE_X, f"Choose X for {card.name}", [Option(f"X={v}", ("x", v), v) for v in xs])
        yield from self._choose_targets(p, item)
        cost = base.with_x(item.x).reduced(reduction)
        yield from self._pay_mana(p, RemainingCost.of(cost), d.additional_sac, what=card.name)
        if d.additional_sac:
            yield from self._choose_sacrifice(p, d.additional_sac, card.name)
        if mode == "escape":
            yield from self._exile_from_graveyard(p, d.escape_exile, card.name)
        self._emit_cast(item)
        self._emit_targeted(item)

    def _exile_from_graveyard(self, p: int, n: int, what: str):
        gy = self.players[p].graveyard
        for i in range(n):
            options = [Option(f"Exile {c.name}", ("exile_gy", c.name), c) for c in self._dedupe_by_name(gy)]
            c = yield from self.ask(p, O.EXILE_FROM_GY, f"Exile a card from your graveyard for {what} ({i + 1}/{n})", options)
            self._move(c, "exile")

    # ------------------------------------------------------------------
    # Activated abilities
    # ------------------------------------------------------------------

    def _can_activate(self, p: int, card: Card, ab: O.AbilityDef) -> bool:
        if ab.zone == "hand":
            if card.zone != "hand":
                return False
        elif card.zone != "battlefield" or card.controller != p:
            return False
        if ab.sorcery_speed and not self.sorcery_timing(p):
            return False
        if ab.tap and (card.tapped or (self.is_creature(card) and card.sick)):
            return False
        if ab.mana is not None:
            return True
        for spec in ab.targets:
            if not self.target_candidates(spec, p):
                return False
        exclude = {card.oid} if ab.tap else set()
        return self._cost_feasible(p, RemainingCost.of(ab.cost), ab.sac_other, exclude)

    def _activate(self, p: int, card: Card, index: int):
        ab = card.face.abilities[index]
        item = StackItem(
            sid=self._new_id(),
            kind="ability",
            controller=p,
            name=f"{card.name}: {ab.name}",
            effect=ab.effect,
            target_specs=ab.targets,
            source=card,
            data={"source_oid": card.oid},
        )
        self.stack.append(item)
        self._log(f"p{p} activates {item.name}")
        yield from self._choose_targets(p, item)
        exclude = {card.oid} if ab.tap else set()
        yield from self._pay_mana(p, RemainingCost.of(ab.cost), ab.sac_other, exclude, what=item.name)
        if ab.tap:
            card.tapped = True
        if ab.sac_other:
            yield from self._choose_sacrifice(p, ab.sac_other, item.name)
        item.source = card.snapshot()
        if ab.discard_self:
            self.discard(card)
        if ab.sac_self and card.zone == "battlefield":
            self.sacrifice(card)
        self._emit_targeted(item)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _run_effect(self, fn, item: StackItem):
        if fn is None:
            return
        res = fn(self, item)
        if inspect.isgenerator(res):
            yield from res

    def _resolve_top(self):
        item = self.stack[-1]
        self._log(f"resolve {item.name}")
        if item.targets and not any(self.target_legal(item, i) for i in range(len(item.targets))):
            if item.kind == "spell" and item.method == "bestow":
                pass  # 702.103e: becomes a creature spell and resolves
            else:
                self._log(f"{item.name} fizzles (no legal targets)")
                self.stack.pop()
                if item.kind == "spell":
                    self._spell_leaves_stack(item)
                return
        if item.kind == "spell":
            card = item.card
            if card.face.is_permanent_card:
                self.stack.pop()
                yield from self._resolve_permanent_spell(item)
            else:
                yield from self._run_effect(item.effect, item)
                if item in self.stack:
                    self.stack.remove(item)
                    self._spell_leaves_stack(item)
        else:
            yield from self._run_effect(item.effect, item)
            if item in self.stack:
                self.stack.remove(item)

    def _spell_leaves_stack(self, item: StackItem) -> None:
        """Instant/sorcery (or countered spell) card leaves the stack."""
        dest = "exile" if item.method == "flashback" else "graveyard"
        self._move(item.card, dest)

    def _resolve_permanent_spell(self, item: StackItem):
        card = item.card
        host = None
        if item.method == "bestow":
            host = self.target(item, 0)
        new = self._move(card, "battlefield", controller=item.controller)
        if card.face.etb_x_counters:
            new.counters = item.x
        if host is not None:
            new.attached_to = host.oid
        self._emit_etb(new)
        if False:
            yield

    def counter(self, item: StackItem) -> None:
        if item not in self.stack:
            return
        self._log(f"{item.name} is countered")
        self.stack.remove(item)
        if item.kind == "spell":
            self._spell_leaves_stack(item)

    # ------------------------------------------------------------------
    # Library helpers used by card effects
    # ------------------------------------------------------------------

    def search_library(self, p: int, predicate: Callable[[Card], bool], dest: str, what: str, tapped: bool = False, reveal: bool = False):
        lib = self.players[p].library
        cands = self._dedupe_by_name(c for c in lib if predicate(c))
        options = [Option("Find nothing", ("search", None), None)]
        options += [Option(f"Find {c.name}", ("search", c.name), c) for c in cands]
        found = yield from self.ask(p, O.CHOOSE_CARD, f"Search your library for {what}", options)
        if found is not None:
            if dest == "battlefield":
                self.put_onto_battlefield(found, p, tapped=tapped)
            else:
                self._move(found, dest, known_to={0, 1} if reveal else None)
        self.shuffle(p)
        return found

    def scry(self, p: int, n: int):
        lib = self.players[p].library
        top = lib[:n]
        for c in top:
            c.known_to.add(p)
        # Scry 1 is all the pool needs; general scry N would be ORDER decisions.
        assert n == 1
        if not top:
            return
        c = top[0]
        where = yield from self.ask(
            p,
            O.CHOOSE_MODE,
            f"Scry 1: {c.name}",
            [Option(f"Keep {c.name} on top", ("scry", "top"), "top"), Option(f"Put {c.name} on the bottom", ("scry", "bottom"), "bottom")],
        )
        if where == "bottom":
            lib.remove(c)
            lib.append(c)

    def explore(self, creature: Card):
        p = creature.controller
        lib = self.players[p].library
        if lib and self.is_land(lib[0]):
            card = lib[0]
            self._move(card, "hand", known_to={0, 1})
            return
        if lib:
            lib[0].known_to = {0, 1}
        live = self.live(creature)
        if live is not None:
            live.counters += 1
        if lib:
            card = lib[0]
            to_gy = yield from self.ask(
                p,
                O.YES_NO,
                f"Explore: put {card.name} into your graveyard?",
                [Option(f"Keep {card.name} on top", ("explore", "keep"), False), Option(f"Put {card.name} into graveyard", ("explore", "graveyard"), True)],
            )
            if to_gy:
                self._move(card, "graveyard")

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def _clear_combat(self) -> None:
        self.attackers = []
        self.blocked = set()
        self.blocks = {}

    def _remove_from_combat(self, oid: int) -> None:
        if oid in self.attackers:
            self.attackers.remove(oid)
        self.blocks.pop(oid, None)
        for b in [b for b, a in self.blocks.items() if a == oid]:
            del self.blocks[b]

    def _declare_attackers(self):
        p = self.active
        eligible = [c for c in self.battlefield if c.controller == p and self.is_creature(c) and not c.tapped and not c.sick]
        # Group interchangeable creatures; choose a multiset in canonical order
        # so every legal attack declaration has exactly one decision path.
        refs = self._referenced_oids()
        groups: list[list[Card]] = []
        index: dict[tuple, int] = {}
        for c in eligible:
            k = self.equiv_key(c, refs)
            if k not in index:
                index[k] = len(groups)
                groups.append([])
            groups[index[k]].append(c)
        chosen: list[Card] = []
        gi_min = 0
        while True:
            options = [Option("Done declaring attackers", ("attack", None), None)]
            for gi in range(gi_min, len(groups)):
                left = [c for c in groups[gi] if c not in chosen]
                if left:
                    c = left[0]
                    options.append(Option(f"Attack with {c.name}#{c.oid}", ("attack", c.name), gi))
            gi = yield from self.ask(p, O.DECLARE_ATTACKER, "Declare attackers", options)
            if gi is None:
                break
            chosen.append(next(c for c in groups[gi] if c not in chosen))
            gi_min = gi
        for c in chosen:
            c.tapped = True  # no vigilance in the card pool
            self.attackers.append(c.oid)
        if chosen:
            self._log(f"p{p} attacks with {chosen}")

    def _can_block(self, blocker: Card, attacker: Card) -> bool:
        if self.has(attacker, "flying") and not (self.has(blocker, "flying") or self.has(blocker, "reach")):
            return False
        return True

    def _declare_blockers(self):
        d = 1 - self.active
        blockers = [c for c in self.battlefield if c.controller == d and self.is_creature(c) and not c.tapped]
        for b in blockers:
            attackers = [self.perm(a) for a in self.attackers]
            attackers = [a for a in attackers if a is not None and self._can_block(b, a)]
            options = [Option(f"{b.name}#{b.oid} does not block", ("block", b.name, None), None)]
            refs = self._referenced_oids()
            seen = set()
            for a in attackers:
                k = self.equiv_key(a, refs)
                if k in seen:
                    continue
                seen.add(k)
                options.append(Option(f"{b.name}#{b.oid} blocks {a.name}#{a.oid}", ("block", b.name, a.name), a))
            a = yield from self.ask(d, O.DECLARE_BLOCKER, f"Block with {b.name}#{b.oid}?", options)
            if a is not None:
                self.blocks[b.oid] = a.oid
                self.blocked.add(a.oid)
        if self.blocks:
            self._log(f"p{d} blocks: {self.blocks}")

    def _lethal(self, attacker: Card, blocker: Card) -> int:
        if self.has(attacker, "deathtouch"):
            return 1
        return max(0, self.toughness(blocker) - blocker.damage)

    def _combat_damage(self):
        assignments: list[tuple[Card, tuple, int]] = []
        defender = 1 - self.active
        for aoid in list(self.attackers):
            a = self.perm(aoid)
            if a is None:
                continue
            pw = self.power(a)
            if pw <= 0:
                continue
            if aoid not in self.blocked:
                assignments.append((a, ("player", defender), pw))
                continue
            blockers = [self.perm(b) for b, at in self.blocks.items() if at == aoid]
            blockers = [b for b in blockers if b is not None]
            trample = self.has(a, "trample")
            if not blockers:
                if trample:
                    assignments.append((a, ("player", defender), pw))
                continue
            if len(blockers) == 1 and not trample:
                assignments.append((a, ("perm", blockers[0].oid), pw))
                continue
            options = []
            slots = len(blockers) + (1 if trample else 0)
            lethal = [self._lethal(a, b) for b in blockers]
            for split in _compositions(pw, slots):
                if trample and split[-1] > 0 and any(s < l for s, l in zip(split, lethal)):
                    continue
                parts = [f"{s} to {b.name}#{b.oid}" for s, b in zip(split, blockers)]
                if trample:
                    parts.append(f"{split[-1]} to player")
                options.append(Option(", ".join(parts), ("damage", tuple(split)), split))
            split = yield from self.ask(self.active, O.ASSIGN_DAMAGE, f"Assign {pw} damage from {a.name}#{a.oid}", options)
            for s, b in zip(split, blockers):
                if s:
                    assignments.append((a, ("perm", b.oid), s))
            if trample and split[-1]:
                assignments.append((a, ("player", defender), split[-1]))
        for boid, aoid in self.blocks.items():
            b = self.perm(boid)
            a = self.perm(aoid)
            if b is None or a is None:
                continue
            pw = self.power(b)
            if pw > 0:
                assignments.append((b, ("perm", a.oid), pw))
        for src, tgt, n in assignments:
            self.deal_damage(src, tgt, n)
        if False:
            yield


def _compositions(total: int, parts: int):
    """All tuples of `parts` non-negative ints summing to `total`."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _ward_effect(g: Game, item: StackItem):
    target = g.stack_item(item.data["sid"])
    if target is None:
        return
    paid = yield from g.optional_payment(target.controller, ManaCost(item.data["amount"]), f"Ward: pay {{{item.data['amount']}}} or {target.name} is countered")
    if not paid:
        g.counter(target)


WARD_TRIGGER = TriggerDef("ward", "becomes_target", _ward_effect)
