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
lists/dicts only, and `fork(replay=True)` rebuilds a game from its constructor
arguments and action history.

Copies: a suspended generator cannot be copied, so `copy()` restarts the
engine from a snapshot of the data state taken at the start of the current
step (where the generator holds nothing the data does not already carry
except which step it is and whether the draw is skipped) and replays only the
actions taken since. Snapshots are taken once a game has been copied; before
that, `copy()` replays the whole history once.
"""

from __future__ import annotations

import copy as _copy
import inspect
import itertools
import random
from typing import Callable, Iterable

from . import objects as O
from .mana import ManaCost, RemainingCost, can_pay, pool_units
from .objects import Card, Decision, Option, PendingTrigger, Player, StackItem, TargetSpec, TriggerDef

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
# Cast modes from the hand, in the order they are offered.
HAND_MODES = ("normal", "bestow", "overload", "alternative", "phyrexian", "bargain", "omen", "evidence")
BARGAIN_SAC = "artifact_enchantment_or_token"  # bargain's sacrifice filter


class GameOver(Exception):
    def __init__(self, winner: int | None, reason: str):
        super().__init__(reason)
        self.winner = winner
        self.reason = reason


class RulesError(Exception):
    """Engine invariant violated (a bug, never a legal game situation)."""


class _Snapshot:
    """Data state of a game at the start of a step (see `Game.copy`)."""

    __slots__ = ("state", "step", "skip_draw", "n_actions")

    def __init__(self, state: dict, step: str, skip_draw: bool, n_actions: int):
        self.state = state
        self.step = step
        self.skip_draw = skip_draw
        self.n_actions = n_actions


# Attributes a copy shares (immutable, or rebuilt by `Game.copy`) or skips.
_SHARED = frozenset({"_args", "cards_db", "deck_names"})
_SKIPPED = frozenset({"_gen", "_snap", "decision"})
_SCALARS = (int, str, bool, float, type(None))


def _copy_state(state: dict) -> dict:
    """Copy of a game's data attributes sharing no mutable object with it.

    Hand-written (copy.deepcopy is ~20x slower): every Card is copied once
    and references to it (zones, stack, triggers, last-known information)
    are remapped, so aliasing is preserved. Card definitions and TempEffects
    are never mutated and are shared. Attributes this function does not know
    fall back to copy.deepcopy, so a new one cannot end up shared."""
    cards: dict[int, Card] = {}

    def new(x):  # shallow copy of a plain dataclass instance (copy.copy is slower)
        n = object.__new__(x.__class__)
        n.__dict__ = x.__dict__.copy()
        return n

    def card(c: Card) -> Card:
        n = cards.get(id(c))
        if n is None:
            n = cards[id(c)] = new(c)
            n.temp = list(c.temp)
            n.known_to = set(c.known_to)
        return n

    def data(d: dict) -> dict:
        return {k: card(v) if type(v) is Card else v if isinstance(v, _SCALARS) else _copy.deepcopy(v) for k, v in d.items()}

    def player(p: Player) -> Player:
        n = new(p)
        n.library = [card(c) for c in p.library]
        n.hand = [card(c) for c in p.hand]
        n.graveyard = [card(c) for c in p.graveyard]
        n.exile = [card(c) for c in p.exile]
        n.pool = dict(p.pool)
        return n

    def stack_item(it: StackItem) -> StackItem:
        n = new(it)
        n.targets = list(it.targets)
        n.card = None if it.card is None else card(it.card)
        n.source = None if it.source is None else card(it.source)
        n.data = data(it.data)
        return n

    def pending(t: PendingTrigger) -> PendingTrigger:
        n = new(t)
        n.source = card(t.source)
        n.data = data(t.data)
        return n

    def rng(r: random.Random) -> random.Random:
        n = random.Random()
        n.setstate(r.getstate())
        return n

    special = {
        "players": lambda ps: [player(p) for p in ps],
        "battlefield": lambda cs: [card(c) for c in cs],
        "stack": lambda xs: [stack_item(it) for it in xs],
        "pending": lambda xs: [pending(t) for t in xs],
        "rng": rng,
        "log": list,
        "actions": list,
        "attackers": list,
        "blocked": set,
        "blocks": dict,
        "mulligans_taken": list,
    }
    out: dict = {"decision": None}
    for k, v in state.items():
        if k in _SKIPPED:
            continue
        f = special.get(k)
        if f is not None:
            out[k] = f(v)
        elif k in _SHARED or isinstance(v, _SCALARS):
            out[k] = v
        else:
            out[k] = _copy.deepcopy(v)
    return out


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
        auto_mana: bool = False,
        auto_pass: bool = False,
        deck_names: tuple[str | None, str | None] | None = None,
        _snapshots: bool = False,
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
            auto_mana=auto_mana,
            auto_pass=auto_pass,
            deck_names=deck_names,
        )
        self.match_game = match_game  # 1 = preboard, 2/3 = after sideboarding
        # Names of decks that are not the default for their seat (match.py):
        # a state feature for the player, None for the seat's usual deck.
        self.deck_names = tuple(deck_names) if deck_names else (None, None)
        self.cards_db = CARDS
        self.rng = random.Random(seed)
        self.auto_single = auto_single
        # Action decomposition (both opt-in, see docs/action-decomposition.md):
        # auto_mana pays mana with a colour-preserving payer unless a source
        # that sacrifices itself is on offer; auto_pass passes priority when
        # the only other options are sacrifice-for-mana abilities whose mana
        # and side effects cannot matter.
        self.auto_mana = auto_mana
        self.auto_pass = auto_pass
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
        self.spells_cast_this_turn = 0  # by both players: the storm count
        self.initiative: int | None = None  # the player who has the initiative (Undercity)
        self.attackers: list[int] = []
        self.blocked: set[int] = set()
        self.blocks: dict[int, int] = {}  # blocker oid -> attacker oid
        # (remaining cost, sacrifice filter, excluded sources) of a pending
        # pay_mana decision; read only by encode's payment preview
        self.paying: tuple | None = None
        # Simulated option previews (encode, feature set 6): on a throwaway
        # copy, the player whose option is simulated. Every decision of the
        # other player is then asked even with one option (whether it has a
        # choice can depend on its hidden hand), and the next priority of
        # either player stops the engine before its options are listed. Never
        # set on a game that is played.
        self.sim_viewer: int | None = None
        self.sim_assume_pass = False  # ... and the other player passes at every priority
        self.shuffles = 0  # library shuffles so far (the simulation stops at hidden information)
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
        # copy() support (module docstring): the latest step-start snapshot,
        # and whether the state was edited outside step() since it was taken.
        self._snapshots = _snapshots
        self._snap: _Snapshot | None = None
        self._edited = False
        self._gen = self._main(start_step)
        self._primed = False
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

    def fork(self, replay: bool = False) -> "Game":
        """Exact copy (`copy()`). `replay=True` rebuilds it from the
        constructor arguments and the action history instead."""
        if replay:
            return self._replay()
        return self.copy()

    def copy(self) -> "Game":
        """Independent exact copy: same state, RNG state and object ids; it
        continues identically under the same actions.

        Costs a snapshot restore plus a replay of the actions taken since the
        current step began. The first copy of a game replays the full history
        instead and turns snapshots on for the game, so later copies are
        cheap. A copy shares the snapshot until its next step begins but
        takes no snapshots of its own until it is copied itself: a preview or
        playout that is never copied pays nothing for them.

        State edits made outside `step()` (determinization, `add_card` after
        the start) drop the snapshot: until the next step begins, a copy
        replays the actions and so, like `fork(replay=True)`, does not carry
        those edits."""
        snap = self._snap
        self._snapshots = True
        if snap is None:
            g = self._replay(snapshots=True)
            g._snapshots = False
            if not self._edited:  # the replay reached this exact state
                self._snap = g._snap
            return g
        g = Game.__new__(Game)
        g.__dict__.update(_copy_state(snap.state))
        g._args = self._args
        g.cards_db = self.cards_db
        g._snap = snap
        g._snapshots = False
        g._edited = False
        g._gen = g._main(g._args["start_step"], resume=(snap.step, snap.skip_draw))
        g._primed = False
        g._advance(None)
        for a in self.actions[snap.n_actions :]:
            g.step(a)
        return g

    def _replay(self, snapshots: bool = False) -> "Game":
        g = Game(**self._args, _snapshots=snapshots)
        for a in self.actions:
            g.step(a)
        return g

    def _state_edited(self) -> None:
        """The state was changed outside `step()`: the current snapshot no
        longer leads to it."""
        self._snap = None
        self._edited = True

    def _take_snapshot(self, step: str, skip_draw: bool) -> None:
        self._snap = _Snapshot(_copy_state(self.__dict__), step, skip_draw, len(self.actions))
        self._edited = False

    def _at_snapshot(self, step: str) -> bool:
        """At the start of the snapshot's own step (a copy resuming from it)."""
        s = self._snap
        return s.step == step and s.n_actions == len(self.actions) and s.state["turn"] == self.turn

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
        if "_gen" in self.__dict__:  # after the start: the snapshot is stale
            self._state_edited()
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
            if not self._primed:
                self._primed = True
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
        if self.auto_single and len(options) == 1 and self.sim_viewer in (None, player):
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

    def _main(self, start_step: str, resume: tuple[str, bool] | None = None):
        if resume is not None:  # copy(): finish the turn from a step-start snapshot
            yield from self._run_turn(*resume)
            self.active = 1 - self.active
            first = False
        else:
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
        self.spells_cast_this_turn = 0
        for p in self.players:
            p.cards_drawn_this_turn = 0
        self._clear_combat()
        self._log(f"=== Turn {self.turn}: player {self.active} ===")

    def _run_turn(self, start: str, skip_draw: bool):
        for name in STEPS[STEPS.index(start):]:
            if name in ("declare_blockers", "combat_damage") and not self.attackers:
                continue
            if self._snapshots:
                self._take_snapshot(name, skip_draw)
            elif self._snap is not None and not self._at_snapshot(name):
                self._snap = None  # a copy's inherited snapshot is stale now
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
            c.hexproof = False  # "until your next turn"
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
            if self.sim_viewer is not None:
                # A simulation never lists priority options (the other
                # player's would read its hidden hand; the decider's are not
                # needed). The "assume the opponent passes" simulation passes
                # for the other player, which is always legal; any other
                # priority stops the simulation.
                if self.sim_assume_pass and p != self.sim_viewer:
                    act = ("pass",)
                else:
                    yield Decision(p, O.PRIORITY, f"Priority ({self.step_name})", [])
                    raise RulesError("a simulation copy cannot continue past a priority")
            else:
                options = self._priority_options(p)
                if self.auto_pass and self._uneventful_priority(p, options):
                    act = options[0].value  # pass
                else:
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

    def _uneventful_priority(self, p: int, options: list[Option]) -> bool:
        """auto_pass: is passing as good as every other option?

        True only when the stack is empty and every non-pass option is a
        sacrifice-for-mana ability (an Eldrazi Spawn) whose sacrifice has no
        side effect: the source has no triggers, is not in combat, and `p`
        controls nothing that triggers on sacrifices. Anything castable with
        that mana is already offered (feasibility counts every source), and
        the pool empties at the end of the step, so floating it is useless."""
        if self.stack or len(options) < 2:
            return False
        for c in self.battlefield:
            if c.controller == p and any(t.event == "you_sacrifice_another" for t in c.face.triggers):
                return False
        combat = set(self.attackers) | set(self.blocks) | set(self.blocks.values())
        for o in options[1:]:
            if o.value[0] != "mana":
                return False
            card = o.value[1]
            if card.face.triggers or card.oid in combat:
                return False
        return True

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
            for mode in HAND_MODES:
                if self._can_cast(p, card, mode):
                    label = f"Cast {card.name}" + ("" if mode == "normal" else f" ({mode})")
                    opts.append(Option(label, ("cast", card.name, "hand", mode), ("cast", card, mode)))
        for card in self._dedupe_by_name(pl.graveyard):
            for mode in ("flashback", "escape"):
                if self._can_cast(p, card, mode):
                    opts.append(Option(f"Cast {card.name} ({mode})", ("cast", card.name, "graveyard", mode), ("cast", card, mode)))
        for card in self._dedupe_by_name(c for c in pl.exile if c.plotted_turn):
            if self._can_cast(p, card, "plot"):
                opts.append(Option(f"Cast {card.name} (plotted)", ("cast", card.name, "exile", "plot"), ("cast", card, "plot")))
        if sorcery_ok:
            for card in self._dedupe_by_name(c for c in pl.hand if c.face.plot is not None):
                if self.can_afford(p, card.face.plot):
                    opts.append(Option(f"Plot {card.name}", ("plot", card.name), ("plot", card)))
        for card in self._dedupe_by_name(pl.hand):
            for i, ab in enumerate(card.face.abilities):
                if ab.zone == "hand" and self._can_activate(p, card, ab):
                    opts.append(Option(f"{card.name}: {ab.name}", ("activate", card.name, ab.name), ("activate", card, i)))
        for card in self._dedupe_by_equiv(c for c in self.battlefield if c.controller == p):
            for i, ab in enumerate(card.face.abilities):
                if ab.zone != "battlefield":
                    continue
                if ab.mana is not None:
                    # Only mana abilities with side effects are worth offering at
                    # priority (single-colour ones: a Treasure is tapped while paying).
                    if ab.sac_self and len(ab.mana) == 1 and self._can_activate(p, card, ab):
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
        elif kind == "plot":
            yield from self._plot(p, act[1])
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
        """An Aura cast with bestow, attached (not a creature then). Other
        attached permanents are Equipment."""
        return c.zone == "battlefield" and c.attached_to is not None and c.face.bestow is not None

    def attach(self, eq: Card | None, host: Card | None) -> None:
        """Attach Equipment `eq` to creature `host` (equip, job select). Nothing
        happens if either is gone, the host is not a creature, or the
        Equipment is itself a creature (CR 301.5c)."""
        if eq is None or host is None or self.perm(host.oid) is None or not self.is_creature(host) or self.is_creature(eq):
            return
        eq.attached_to = host.oid
        self._log(f"{eq.name}#{eq.oid} attached to {host.name}#{host.oid}")

    def types(self, c: Card) -> set[str]:
        t = set(c.face.types)
        if c.animated is not None:
            t.add("Creature")
        if self.is_bestowed(c):
            t.discard("Creature")
        return t

    def is_creature(self, c: Card) -> bool:
        return ("Creature" in c.face.types or c.animated is not None) and not self.is_bestowed(c)

    def is_artifact(self, c: Card) -> bool:
        return "Artifact" in c.face.types

    def is_land(self, c: Card) -> bool:
        return "Land" in c.face.types

    def _auras_on(self, c: Card) -> list[Card]:
        return [a for a in self.battlefield if a.attached_to == c.oid]

    def power(self, c: Card) -> int:
        base = c.animated[0] if c.animated is not None else c.face.power or 0
        v = base + c.counters + sum(t.power for t in c.temp)
        v += sum(a.counters if a.face.bestow is not None else a.face.equipped_power for a in self._auras_on(c))
        return v

    def toughness(self, c: Card) -> int:
        base = c.animated[1] if c.animated is not None else c.face.toughness or 0
        v = base + c.counters + sum(t.toughness for t in c.temp)
        v += sum(a.counters if a.face.bestow is not None else a.face.equipped_toughness for a in self._auras_on(c))
        return v

    def keywords(self, c: Card) -> set[str]:
        k = set(c.face.keywords) | c.granted
        for t in c.temp:
            k |= t.keywords
        for a in self._auras_on(c):
            if a.face.bestow is not None:  # Nyxborn Hydra
                k |= {"reach", "trample"}
        if c.hexproof:
            k.add("hexproof")
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
            c.oid in self.blocked,
            c.animated,
            tuple(sorted(c.granted)),
            c.once_used_turn == self.turn,
            c.hexproof,
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
            card.tapped = tapped or card.face.enters_tapped or self._enters_tapped_unless(card)
            self.battlefield.append(card)
            if self.is_land(card):
                self.players[card.controller].landfall_turn = self.turn
            self._log(f"enters: {card.name}#{card.oid} (p{card.controller})")
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

    def _enters_tapped_unless(self, card: Card) -> bool:
        """Enters tapped unless its controller controls N other Forests (Gingerbread Cabin)."""
        n = card.face.enters_tapped_unless_forests
        if not n:
            return False
        others = sum(1 for c in self.battlefield if c.controller == card.controller and "Forest" in c.face.subtypes)
        return others < n

    def _after_leave_battlefield(self, lki: Card, new: Card, to: str) -> None:
        self._log(f"leaves: {lki.name}#{lki.oid} (p{lki.controller}) -> {to}")
        if to == "graveyard":
            for t in lki.face.triggers:
                if t.event == "to_graveyard_from_battlefield":
                    self.pending.append(PendingTrigger(lki.controller, lki, t, {"card": new, "oid": new.oid}))
        for t in lki.face.triggers:
            if t.event == "leaves_battlefield":
                self.pending.append(PendingTrigger(lki.controller, lki, t))

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
                if pl.cards_drawn_this_turn == 3:
                    self._emit_third_draw(p)

    def mill(self, p: int, n: int) -> None:
        pl = self.players[p]
        for _ in range(n):
            if not pl.library:
                return
            self._move(pl.library[0], "graveyard")

    def discard(self, card: Card) -> None:
        self._log(f"p{card.owner} discards {card.name}")
        if card.face.madness is not None:
            # Madness (702.35): discarded into exile; a trigger lets the owner cast it.
            new = self._move(card, "exile")
            self.pending.append(PendingTrigger(new.owner, new, MADNESS_TRIGGER, {"card": new, "oid": new.oid}))
            return
        self._move(card, "graveyard")

    def choose_discard(self, p: int, what: str):
        """`p` discards a card of their choice from hand (if any). Returns it."""
        if not self.players[p].hand:
            return None
        card = yield from self.ask(p, O.CHOOSE_CARD, f"{what}: discard a card", self._hand_card_options(p, "discard"))
        self.discard(card)
        return card

    def shuffle(self, p: int) -> None:
        lib = self.players[p].library
        self.shuffles += 1
        self.rng.shuffle(lib)
        for c in lib:
            c.known_to = set()

    def gain_life(self, p: int, n: int) -> None:
        self.players[p].life += n

    def sacrifice(self, card: Card) -> None:
        lki = card.snapshot()
        self._log(f"p{card.controller} sacrifices {card.name}#{card.oid}")
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
        self._log(f"enters: {c.name}#{c.oid} (p{p})")
        self._emit_etb(c)
        return c

    def put_onto_battlefield(self, card: Card, controller: int, tapped: bool = False) -> Card:
        new = self._move(card, "battlefield", controller=controller, tapped=tapped)
        self._emit_etb(new)
        return new

    # ------------------------------------------------------------------
    # Initiative and the Undercity (725, 309)
    # ------------------------------------------------------------------

    def _dungeon_card(self, p: int) -> Card:
        """The Undercity as the source of room and initiative triggers: an
        object outside the game's zones (oid 0, never on the battlefield)."""
        from .cards import DUNGEONS

        return Card(uid=0, oid=0, defn=DUNGEONS["Undercity"], owner=p, controller=p, zone="command", known_to={0, 1})

    def take_initiative(self, p: int):
        """725.2: the player takes the initiative (again, if they had it) and ventures into Undercity."""
        self.initiative = p
        self._log(f"p{p} takes the initiative")
        yield from self.venture(p)

    def venture(self, p: int):
        """701.49: move the venture marker into the next room (a choice where
        the dungeon branches) and trigger that room. From the last room a
        new Undercity is started (the completed one is not tracked)."""
        from .cards import DUNGEONS, ROOM_NEXT

        dungeon = DUNGEONS["Undercity"]
        room = self.players[p].dungeon_room
        nxt = ROOM_NEXT[(dungeon.name, room)] if room is not None else ()
        if not nxt:
            new = dungeon.triggers[0].name
        elif len(nxt) == 1:
            new = nxt[0]
        else:
            options = [Option(f"Venture into {r}", ("venture", r), r) for r in nxt]
            new = yield from self.ask(p, O.CHOOSE_MODE, f"Venture into Undercity: choose the room after {room}", options)
        self.players[p].dungeon_room = new
        self._log(f"p{p} ventures into {new}")
        tdef = next(t for t in dungeon.triggers if t.name == new)
        self.pending.append(PendingTrigger(p, self._dungeon_card(p), tdef))

    def deal_damage(self, source: Card, target: tuple, amount: int) -> None:
        """target: ('player', idx) or ('perm', oid)."""
        if amount <= 0:
            return
        # A source still on the battlefield uses its current characteristics
        # (e.g. deathtouch granted in response); last known information only
        # once it has left (rule 608.2h).
        source = self.live(source) or source
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

    def _emit_etb(self, card: Card, method: str | None = None) -> None:
        """method: how the permanent was cast, if it resolved as a spell."""
        for t in card.face.triggers:
            if t.event == "etb" and (t.condition is None or t.condition(self, card, method)):
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
        if self.initiative == self.active:
            self.pending.append(PendingTrigger(self.active, self._dungeon_card(self.active), VENTURE_TRIGGER))

    def _emit_cast(self, item: StackItem) -> None:
        storm = self.spells_cast_this_turn  # spells cast before this one this turn
        self.spells_cast_this_turn += 1
        for t in item.card.face.triggers:
            if t.event == "cast":
                self.pending.append(PendingTrigger(item.controller, item.card, t, {"spell_sid": item.sid, "storm": storm}))
        for perm in self.battlefield:
            if perm.controller != item.controller:
                continue
            for t in perm.face.triggers:
                if t.event == "you_cast" and (t.condition is None or t.condition(self, perm, item.card)):
                    self.pending.append(PendingTrigger(perm.controller, perm, t, {"spell_sid": item.sid}))

    def _emit_third_draw(self, p: int) -> None:
        for c in self.players[p].graveyard:
            for t in c.face.triggers:
                if t.event == "third_draw":
                    self.pending.append(PendingTrigger(p, c, t, {"card": c, "oid": c.oid}))

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
                    target_specs=t.tdef.targets,
                    source=t.source,
                    data=dict(t.data),
                )
                self.stack.append(item)
                self._log(f"trigger -> stack: {item.name}")
                if item.target_specs:
                    # 603.3d: a triggered ability without legal targets is removed from the stack
                    # ("up to" targets can always be chosen: none).
                    if not t.tdef.up_to and any(not self.target_candidates(spec, p, exclude_sid=item.sid, chosen=[]) for spec in item.target_specs):
                        self.stack.remove(item)
                        self._log(f"{item.name}: no legal targets")
                        continue
                    yield from self._choose_targets(p, item, up_to=t.tdef.up_to)
                    self._emit_targeted(item)

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
                if host is None or not self.is_creature(host) or self.is_creature(c):
                    c.attached_to = None  # bestow: becomes a creature again; Equipment stays, unattached (CR 301.5c)
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

    def target_candidates(self, spec: TargetSpec, controller: int, exclude_sid: int | None = None, chosen: list | None = None) -> list[tuple]:
        """`chosen`: the targets already chosen for the same spell or ability
        (for a spec that depends on them, creature_of_target_player)."""
        k = spec.kind
        if k == "player":
            return [("player", controller), ("player", 1 - controller)]
        if k == "opponent":
            return [("player", 1 - controller)]
        if k == "player_with_creature":
            return [("player", q) for q in (controller, 1 - controller) if any(c.controller == q and self.is_creature(c) for c in self.battlefield)]
        if k == "creature_of_target_player" and chosen:
            q = chosen[-1][1]
            return [("perm", c.oid) for c in self.battlefield if self.is_creature(c) and c.controller == q]
        if k == "another_creature":
            # A creature not already chosen as a target of the same spell. With
            # nothing chosen yet (casting checks) it needs a second creature.
            creatures = [("perm", c.oid) for c in self.battlefield if self.is_creature(c)]
            if not chosen:
                return creatures if len(creatures) >= 2 else []
            return [r for r in creatures if r not in chosen]
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
        if c.controller != controller and c.hexproof:
            return False
        if k in ("creature", "any", "creature_of_target_player", "another_creature"):
            return self.is_creature(c)
        if k == "artifact_or_enchantment_you_dont_control":
            return c.controller != controller and bool(self.types(c) & {"Artifact", "Enchantment"})
        if k == "nonlegendary_creature":
            return self.is_creature(c) and "Legendary" not in c.face.supertypes
        if k == "creature_you_control":
            return self.is_creature(c) and c.controller == controller
        if k == "creature_you_dont_control":
            return self.is_creature(c) and c.controller != controller
        if k == "permanent":
            return True
        if k == "noncreature_artifact":
            return self.is_artifact(c) and not self.is_creature(c)
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
        if k == "artifact_or_enchantment":
            return bool(c.face.types & {"Artifact", "Enchantment"})
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
        if k == "sorcery_spell":
            return d.is_type("Sorcery")
        if k == "artifact_spell":
            return d.is_type("Artifact")
        if k == "artifact_or_enchantment_spell":
            return d.is_type("Artifact") or d.is_type("Enchantment")
        return False

    def target_legal(self, item: StackItem, i: int) -> bool:
        spec = item.target_specs[i]
        ref = item.targets[i]
        if ref[0] == "player":
            return spec.kind in ("player", "any", "player_with_creature") or (spec.kind == "opponent" and ref[1] != item.controller)
        if ref[0] == "stack":
            it = self.stack_item(ref[1])
            return it is not None and it.kind == "spell" and spec.kind.endswith("spell") and self._spell_matches(spec, it)
        c = self.perm(ref[1])
        if c is not None and spec.kind == "creature_of_target_player" and c.controller != item.targets[i - 1][1]:
            return False
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

    def _choose_targets(self, p: int, item: StackItem, allowed: Callable[[tuple], bool] | None = None, up_to: bool = False):
        """`allowed`: an extra filter on the candidates (targets the cost can be paid for).
        `up_to`: "up to one target": choosing no target ends the choice."""
        for spec in item.target_specs:
            cands = self.target_candidates(spec, p, exclude_sid=item.sid, chosen=item.targets)
            if allowed is not None:
                cands = [r for r in cands if allowed(r)]
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
            if up_to:
                options.insert(0, Option("No target", ("target", spec.kind, None), None))
            ref = yield from self.ask(p, O.TARGET, f"Choose target ({spec.kind}) for {item.name}", options)
            if ref is None:
                return
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

    def mana_units(self, ab: O.AbilityDef) -> int:
        """Mana one activation makes: one, or one per Elf (Priest of Titania)."""
        if ab.mana_amount is None:
            return 1
        from .cards import count_of

        return count_of(self, ab.mana_amount)

    def sac_candidates(self, p: int, flt: str, exclude: set[int] = frozenset()) -> list[Card]:
        out = []
        for c in self.battlefield:
            if c.controller != p or c.oid in exclude:
                continue
            if flt == "artifact" and self.is_artifact(c):
                out.append(c)
            elif flt == "artifact_or_creature" and (self.is_artifact(c) or self.is_creature(c)):
                out.append(c)
            elif flt == "mountain" and self.is_land(c) and "Mountain" in c.face.subtypes:
                out.append(c)
            elif flt == BARGAIN_SAC and (c.is_token or self.types(c) & {"Artifact", "Enchantment"}):
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
        normal = [ab.mana for c, ab in sources if not ab.sac_self for _ in range(self.mana_units(ab))]
        selfsac = [(c, ab) for c, ab in sources if ab.sac_self]
        base = pool_units(pool) + normal
        if sac_filter is None:
            return can_pay(rem, base + [ab.mana for _, ab in selfsac])
        cands = {c.oid for c in self.sac_candidates(p, sac_filter, gone)}
        if not cands:
            return False
        # Self-sacrificing {C} sources (Eldrazi Spawn) are interchangeable:
        # using k of them is equivalent whichever k we pick, so spend
        # non-candidates first. Others (a Treasure) are tried as subsets.
        colourless = [c for c, ab in selfsac if ab.mana == ("C",)]
        others = [(c, ab) for c, ab in selfsac if ab.mana != ("C",)]
        ordered = [c for c in colourless if c.oid not in cands] + [c for c in colourless if c.oid in cands]
        for mask in range(1 << len(others)):
            chosen = [others[i] for i in range(len(others)) if mask >> i & 1]
            used_o = {c.oid for c, _ in chosen}
            units_o = [ab.mana for _, ab in chosen]
            for k in range(len(ordered) + 1):
                used = used_o | {c.oid for c in ordered[:k]}
                if len(cands - used) >= 1 and can_pay(rem, base + units_o + [("C",)] * k):
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
                extra = self.mana_units(ab) - 1  # floats in the pool (Priest of Titania)
                for color in ab.mana:
                    if not rem.useful(color):
                        continue
                    r2 = rem.copy()
                    r2.apply(color)
                    gone = {card.oid} if ab.sac_self else set()
                    pool2 = None
                    if extra:
                        pool2 = dict(pl.pool)
                        pool2[color] = pool2.get(color, 0) + extra
                    if self._cost_feasible(p, r2, sac_filter, exclude | {card.oid}, pool2, gone=gone):
                        verb = "Sacrifice" if ab.sac_self else "Tap"
                        options.append(
                            Option(f"{verb} {card.name}#{card.oid} for {color}", ("pay", "source", card.name, color), ("source", card, color))
                        )
            auto = self._auto_pay_index(p, rem, options) if self.auto_mana else None
            if auto is not None:
                if self.logging:
                    self._log(f"  p{p} {O.PAY_MANA}: {options[auto].label} (auto)")
                choice = options[auto].value
            else:
                self.paying = (rem, sac_filter, exclude)
                choice = yield from self.ask(p, O.PAY_MANA, f"Pay {rem} for {what}", options)
                self.paying = None
            if choice[0] == "pool":
                pl.pool[choice[1]] -= 1
                if pl.pool[choice[1]] == 0:
                    del pl.pool[choice[1]]
                rem.apply(choice[1])
            else:
                card, color = choice[1], choice[2]
                ab = next(a for a in card.face.abilities if a.mana is not None)
                extra = self.mana_units(ab) - 1
                self._activate_mana_ability(card, ab)
                if not rem.apply(color):
                    raise RulesError("mana unit could not be applied")
                if extra > 0:
                    pl.pool[color] = pl.pool.get(color, 0) + extra

    def _colour_needs(self, p: int) -> dict[str, int]:
        """Coloured pips the non-land cards in `p`'s hand ask for; instants
        count double (they are cast from whatever is left untapped)."""
        need: dict[str, int] = {}
        for c in self.players[p].hand:
            if c.face.is_type("Land"):
                continue
            w = 2 if c.face.is_type("Instant") else 1
            for col, n in c.face.cost.colored:
                need[col] = need.get(col, 0) + w * n
        return need

    def _auto_pay_index(self, p: int, rem: RemainingCost, options: list[Option]) -> int | None:
        """auto_mana: the pay_mana option a colour-preserving payer takes, or
        None when the choice is left to the player.

        Floating mana goes first (the pool empties anyway). If any option
        sacrifices its source (an Eldrazi Spawn), the payment is strategic and
        the player decides. Otherwise tap the source that hurts least: lowest
        sum of hand colour needs over the colours it produces, then sources
        without another {T} ability (Twisted Landscape keeps its search), then
        fewer colours, then a colour that pays a coloured pip, then option order.
        Every offered option keeps the payment completable, so greedy is safe."""
        need = None
        best, best_key = None, None
        for i, o in enumerate(options):
            if o.value[0] == "pool":
                return i
            card, color = o.value[1], o.value[2]
            ab = next(a for a in card.face.abilities if a.mana is not None)
            if ab.sac_self:
                return None
            if need is None:
                need = self._colour_needs(p)
            hurt = sum(need.get(col, 0) for col in ab.mana)
            other_tap = any(a is not ab and a.zone == "battlefield" and a.tap for a in card.face.abilities)
            key = (hurt, int(other_tap), len(ab.mana), 0 if rem.colored.get(color, 0) > 0 else 1)
            if best_key is None or key < best_key:
                best, best_key = i, key
        return best

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
        article = "an" if flt[0] in "aeiou" else "a"
        card = yield from self.ask(p, O.SACRIFICE, f"Sacrifice {article} {flt.replace('_', ' ')} for {what}", options)
        mv = card.defn.mana_value
        self.sacrifice(card)
        return mv

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
        if mode == "madness":
            return d.madness
        if mode == "overload":
            return d.overload
        if mode == "plot":
            return O.FREE if d.plot is not None else None
        if mode == "alternative":
            return O.FREE if d.alternative_sac is not None or d.alternative_reveal else None
        if mode == "phyrexian":
            return d.phyrexian_cost
        if mode == "bargain":
            return d.cost if d.bargain else None
        if mode == "omen":
            return card.defn.back.cost if card.defn.omen else None
        if mode == "evidence":
            return d.cost if d.collect_evidence else None
        return None

    @staticmethod
    def _mode_additional_sac(card: Card, mode: str) -> str | None:
        """Sacrifice filter of an additional cost: the card's own, or bargain's."""
        return BARGAIN_SAC if mode == "bargain" else card.face.additional_sac

    def _mode_sac(self, card: Card, mode: str) -> tuple[str, int] | None:
        """Lands sacrificed instead of (part of) the cost: (filter, count)."""
        if mode == "alternative":
            return card.face.alternative_sac
        if mode == "flashback":
            return card.face.flashback_sac
        return None

    def _mode_targets(self, card: Card, mode: str, choice: int | None = None) -> tuple[TargetSpec, ...]:
        if mode == "bestow":
            return (TargetSpec("creature"),)
        if mode == "overload":
            return ()
        if mode == "omen":
            return card.defn.back.targets
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
        if mode in HAND_MODES and card.zone != "hand":
            return False
        if mode in ("flashback", "escape") and card.zone != "graveyard":
            return False
        if mode in ("madness", "plot") and card.zone != "exile":
            return False
        if mode == "plot" and not 0 < card.plotted_turn < self.turn:
            return False
        if d.is_type("Land"):
            return False
        spell = d.back if mode == "omen" else d  # the characteristics it is cast with
        # Madness casts on resolution of its trigger, whatever the card type (702.35).
        if mode != "madness" and not spell.is_type("Instant") and "flash" not in spell.keywords and not self.sorcery_timing(p):
            return False
        for spec in self._mode_targets(card, mode, choice):
            if not self.target_candidates(spec, p):
                return False
        if mode == "escape" and len(self.players[p].graveyard) - 1 < d.escape_exile:
            return False
        if mode == "evidence" and sum(c.face.mana_value for c in self.players[p].graveyard) < d.collect_evidence:
            return False
        if d.additional_discard and len(self.players[p].hand) - (card.zone == "hand") < 1:
            return False
        if mode == "phyrexian" and self.players[p].life < d.phyrexian_life:
            return False
        if mode == "alternative" and d.alternative_reveal and any(c.face.is_type("Land") for c in self.players[p].hand if c is not card):
            return False
        if d.additional_power and not self._power_sources(p, card):
            return False
        sac = self._mode_sac(card, mode)
        if sac is not None and len(self.sac_candidates(p, sac[0])) < sac[1]:
            return False
        cost = base.with_x(0).reduced(self._cost_reduction(p, card))
        return self._cost_feasible(p, RemainingCost.of(cost), self._mode_additional_sac(card, mode))

    def _power_sources(self, p: int, card: Card) -> list[Card]:
        """Monstrous Emergence's additional cost: creatures `p` controls
        (interchangeable ones once) and creature cards in hand (once per name)."""
        bf = self._dedupe_by_equiv(c for c in self.battlefield if c.controller == p and self.is_creature(c))
        hand = self._dedupe_by_name(c for c in self.players[p].hand if c is not card and c.face.is_type("Creature"))
        return bf + hand

    def _choose_power(self, p: int, item: StackItem):
        options = []
        for c in self._power_sources(p, item.card):
            if c.zone == "battlefield":
                options.append(Option(f"Choose {c.name}#{c.oid}", ("choose_power", "battlefield", c.name), c))
            else:
                options.append(Option(f"Reveal {c.name}", ("choose_power", "hand", c.name), c))
        c = yield from self.ask(p, O.CHOOSE_CARD, f"{item.name}: choose a creature you control or reveal a creature card", options)
        if c.zone == "battlefield":
            item.data["chosen_oid"] = c.oid
            item.data["power"] = self.power(c)
        else:
            c.known_to = {0, 1}
            item.data["power"] = c.face.power or 0
        self._log(f"p{p} chooses {c.name} (power {item.data['power']})")

    def _cast(self, p: int, card: Card, mode: str, choice: int | None = None):
        d = card.face
        from_zone = card.zone
        face = d.back if mode == "omen" else d
        effect = face.effect if choice is None else d.modes[choice].effect
        item = StackItem(
            sid=0,
            kind="spell",
            controller=p,
            name=face.name,
            effect=d.overload_effect if mode == "overload" else effect,
            target_specs=self._mode_targets(card, mode, choice),
            card=card,
            method=mode,
            cast_from=from_zone,
        )
        self._move(card, "stack", controller=p)
        if mode == "omen":
            card.transformed = True  # on the stack it is the omen face (Roost Seek)
        item.sid = card.oid
        self.stack.append(item)
        self._log(f"p{p} casts {card.name} ({mode}) from {from_zone}")
        base = self._mode_cost(card, mode)
        reduction = self._cost_reduction(p, card)
        add_sac = self._mode_additional_sac(card, mode)
        if base.x:
            xs = []
            x = 0
            while self._cost_feasible(p, RemainingCost.of(base.with_x(x).reduced(reduction)), add_sac):
                xs.append(x)
                x += 1
            item.x = yield from self.ask(p, O.CHOOSE_X, f"Choose X for {card.name}", [Option(f"X={v}", ("x", v), v) for v in xs])
        yield from self._choose_targets(p, item)
        cost = base.with_x(item.x).reduced(reduction)
        yield from self._pay_mana(p, RemainingCost.of(cost), add_sac, what=card.name)
        if mode == "phyrexian":
            self.players[p].life -= d.phyrexian_life
            self._log(f"p{p} pays {d.phyrexian_life} life for {card.name}")
        if add_sac:
            mv = yield from self._choose_sacrifice(p, add_sac, card.name)
            if d.additional_sac:
                item.data["sacrificed_mv"] = mv
        if d.additional_discard:
            gone = yield from self.choose_discard(p, card.name)
            item.data["discarded_land"] = self.is_land(gone)
        if d.additional_power:
            yield from self._choose_power(p, item)
        sac = self._mode_sac(card, mode)
        for _ in range(sac[1] if sac else 0):
            yield from self._choose_sacrifice(p, sac[0], card.name)
        if mode == "alternative" and d.alternative_reveal:
            for c in self.players[p].hand:
                c.known_to = {0, 1}
            self._log(f"p{p} reveals their hand for {card.name}: {', '.join(c.name for c in self.players[p].hand) or 'empty'}")
        if mode == "escape":
            yield from self._exile_from_graveyard(p, d.escape_exile, card.name)
        if mode == "evidence":
            yield from self._collect_evidence(p, d.collect_evidence, card.name)
        self._emit_cast(item)
        self._emit_targeted(item)

    def _plot(self, p: int, card: Card):
        """Plot (702.170): a special action. Pay the plot cost, exile the card
        face up; it can be cast for free as a sorcery on a later turn."""
        self._log(f"p{p} plots {card.name}")
        yield from self._pay_mana(p, RemainingCost.of(card.face.plot), what=f"plot {card.name}")
        new = self._move(card, "exile")
        new.plotted_turn = self.turn

    def _collect_evidence(self, p: int, n: int, what: str):
        """Collect evidence N (701.59): exile cards with total mana value N or
        more from your graveyard, one at a time (casting checked the total)."""
        total = 0
        while total < n:
            options = [Option(f"Exile {c.name}", ("exile_gy", c.name), c) for c in self._dedupe_by_name(self.players[p].graveyard)]
            c = yield from self.ask(p, O.EXILE_FROM_GY, f"Collect evidence {n} for {what}: exile a card from your graveyard ({total}/{n})", options)
            total += c.face.mana_value
            self._move(c, "exile")

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
        if ab.discard_other and not self.players[p].hand:
            return False
        if ab.once_per_turn and card.once_used_turn == self.turn:
            return False
        if ab.return_land and not self._return_land_candidates(p, ab.return_land):
            return False
        exclude = {card.oid} if ab.tap else set()
        if ab.x_target_mv:
            return any(self._x_target_affordable(p, card, ab, ref) for ref in self.target_candidates(ab.targets[0], p))
        for spec in ab.targets:
            if not self.target_candidates(spec, p):
                return False
        return self._cost_feasible(p, RemainingCost.of(ab.cost), ab.sac_other, exclude)

    def _return_land_candidates(self, p: int, flt: str) -> list[Card]:
        assert flt == "forest", flt
        return [c for c in self.battlefield if c.controller == p and self.is_land(c) and "Forest" in c.face.subtypes]

    def _choose_return_land(self, p: int, flt: str, what: str):
        """Return a land you control to its owner's hand as a cost (Quirion Ranger)."""
        cands = self._dedupe_by_equiv(self._return_land_candidates(p, flt))
        options = [Option(f"Return {c.name}#{c.oid}", ("return_land", c.name), c) for c in cands]
        c = yield from self.ask(p, O.SACRIFICE, f"Return a {flt} you control to its owner's hand for {what}", options)
        self._move(c, "hand")

    def _x_target_cost(self, ab: O.AbilityDef, ref: tuple) -> ManaCost:
        """Cost of an ability whose X is the target's mana value (Gorilla Shaman)."""
        return ab.cost.plus(ManaCost(ab.x_target_mv * self.perm(ref[1]).face.mana_value))

    def _x_target_affordable(self, p: int, card: Card, ab: O.AbilityDef, ref: tuple) -> bool:
        exclude = {card.oid} if ab.tap else set()
        return self._cost_feasible(p, RemainingCost.of(self._x_target_cost(ab, ref)), ab.sac_other, exclude)

    def _red_cards_in_hand(self, p: int) -> list[Card]:
        return [c for c in self.players[p].hand if "R" in c.face.colors]

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
        if ab.once_per_turn:
            card.once_used_turn = self.turn
        if ab.x_reveal:
            n = len(self._red_cards_in_hand(p))
            item.x = yield from self.ask(p, O.CHOOSE_X, f"Choose X for {item.name}", [Option(f"X={v}", ("x", v), v) for v in range(n + 1)])
        cost = ab.cost
        if ab.x_target_mv:
            yield from self._choose_targets(p, item, allowed=lambda r: self._x_target_affordable(p, card, ab, r))
            cost = self._x_target_cost(ab, item.targets[0])
            item.x = self.perm(item.targets[0][1]).face.mana_value
        else:
            yield from self._choose_targets(p, item)
        exclude = {card.oid} if ab.tap else set()
        yield from self._pay_mana(p, RemainingCost.of(cost), ab.sac_other, exclude, what=item.name)
        if ab.tap:
            card.tapped = True
        if ab.sac_other:
            yield from self._choose_sacrifice(p, ab.sac_other, item.name)
        if ab.discard_other:
            yield from self.choose_discard(p, item.name)
        if ab.return_land:
            yield from self._choose_return_land(p, ab.return_land, item.name)
        if ab.x_reveal:
            yield from self._reveal_red(p, item.x, item.name)
        item.source = card.snapshot()
        if ab.discard_self:
            self.discard(card)
        if ab.sac_self and card.zone == "battlefield":
            self.sacrifice(card)
        if ab.exile_self and card.zone == "battlefield":
            self._move(card, "exile")
        self._emit_targeted(item)

    def _reveal_red(self, p: int, n: int, what: str):
        """Reveal `n` red cards from hand, one at a time (a cost: Martyr of Ashes)."""
        revealed: list[Card] = []
        for i in range(n):
            cands = self._dedupe_by_name(c for c in self._red_cards_in_hand(p) if c not in revealed)
            options = [Option(f"Reveal {c.name}", ("reveal", c.name), c) for c in cands]
            c = yield from self.ask(p, O.CHOOSE_CARD, f"{what}: reveal a red card ({i + 1}/{n})", options)
            c.known_to = {0, 1}
            revealed.append(c)

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
                    self._spell_leaves_stack(item, resolved=True)
        else:
            yield from self._run_effect(item.effect, item)
            if item in self.stack:
                self.stack.remove(item)

    def _spell_leaves_stack(self, item: StackItem, resolved: bool = False) -> None:
        """Instant/sorcery (or countered spell) card leaves the stack. A
        resolved omen is shuffled into its owner's library."""
        if resolved and item.method == "omen":
            self._move(item.card, "library")
            self.shuffle(item.card.owner)
            return
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
        self._emit_etb(new, item.method)
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
        if not top:
            return
        if len(top) > 1:
            yield from self._scry_n(p, top)
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

    def _scry_n(self, p: int, top: list[Card]):
        """Scry N > 1 as one ORDER decision: which cards stay on top (in
        order) and which go to the bottom (in order), outcomes deduplicated
        by names."""
        lib = self.players[p].library
        options = []
        seen = set()
        for perm in itertools.permutations(top):
            for k in range(len(perm) + 1):
                tops, bottoms = perm[:k], perm[k:]
                key = (tuple(c.name for c in tops), tuple(c.name for c in bottoms))
                if key in seen:
                    continue
                seen.add(key)
                label = f"Top: {', '.join(key[0]) or '-'}; bottom: {', '.join(key[1]) or '-'}"
                options.append(Option(label, ("scry", "top", *key[0], "bottom", *key[1]), (tops, bottoms)))
        tops, bottoms = yield from self.ask(p, O.ORDER, f"Scry {len(top)}: {', '.join(c.name for c in top)}", options)
        del lib[: len(top)]
        lib[:0] = list(tops)
        lib.extend(bottoms)

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
        # Group interchangeable creatures: one option per group with creatures
        # left, in any order. (Until 2026-10 picks were in canonical group order,
        # so picking a later group silently dropped the earlier ones.) Chosen
        # creatures are attacking at once, so the state shows the pending
        # declaration; they tap when it is done.
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
        while True:
            options = [Option("Done declaring attackers", ("attack", None), None)]
            for gi in range(len(groups)):
                left = [c for c in groups[gi] if c not in chosen]
                if left:
                    c = left[0]
                    options.append(Option(f"Attack with {c.name}#{c.oid}", ("attack", c.name), gi))
            gi = yield from self.ask(p, O.DECLARE_ATTACKER, "Declare attackers", options)
            if gi is None:
                break
            c = next(c for c in groups[gi] if c not in chosen)
            chosen.append(c)
            self.attackers.append(c.oid)
        for c in chosen:
            c.tapped = True  # no vigilance in the card pool
        if chosen:
            self._log(f"p{p} attacks with {chosen}")

    def _can_block(self, blocker: Card, attacker: Card) -> bool:
        if self.has(attacker, "flying") and not (self.has(blocker, "flying") or self.has(blocker, "reach")):
            return False
        return True

    def _declare_blockers(self):
        d = 1 - self.active
        blockers = [c for c in self.battlefield if c.controller == d and self.is_creature(c) and not c.tapped]
        for i, b in enumerate(blockers):
            attackers = [self.perm(a) for a in self.attackers]
            attackers = [a for a in attackers if a is not None and self._can_block(b, a) and self._menace_ok(a, blockers[i + 1 :])]
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
        for aoid in list(self.attackers):
            a = self.perm(aoid)
            mine = [b for b, at in self.blocks.items() if at == aoid]
            if a is not None and len(mine) == 1 and self.has(a, "menace"):
                # 702.111b: a lone blocker of a menace creature is not a legal block; it is undone.
                del self.blocks[mine[0]]
                self.blocked.discard(aoid)
                self._log(f"menace: the block of {a.name}#{a.oid} by one creature is undone")
        if self.blocks:
            self._log(f"p{d} blocks: {self.blocks}")

    def _menace_ok(self, a: Card, later: list[Card]) -> bool:
        """Blocking a menace creature is offered only while a second blocker
        is possible: one already blocks it, or a later blocker could."""
        if not self.has(a, "menace"):
            return True
        if any(at == a.oid for at in self.blocks.values()):
            return True
        return any(self._can_block(b, a) for b in later)

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
        if self.initiative == defender and any(tgt == ("player", defender) and n > 0 for _, tgt, n in assignments):
            # 725.2: combat damage to the player with the initiative takes it.
            self.pending.append(PendingTrigger(self.active, self._dungeon_card(self.active), INITIATIVE_TRIGGER))
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


def _madness_effect(g: Game, item: StackItem):
    """Cast the exiled card for its madness cost, or put it into the graveyard."""
    card = item.data["card"]
    if card.zone != "exile" or card.oid != item.data["oid"]:
        return
    p = card.owner
    options = [Option(f"Put {card.name} into your graveyard", ("madness", "graveyard"), False)]
    if g._can_cast(p, card, "madness"):
        options.append(Option(f"Cast {card.name} for its madness cost", ("madness", "cast"), True))
    cast = yield from g.ask(p, O.YES_NO, f"Madness: cast {card.name} for {card.face.madness}?", options)
    if cast:
        yield from g._cast(p, card, "madness")
    else:
        g._move(card, "graveyard")


MADNESS_TRIGGER = TriggerDef("madness", "discarded", _madness_effect)


def _venture_effect(g: Game, item: StackItem):
    yield from g.venture(item.controller)


def _take_initiative_effect(g: Game, item: StackItem):
    yield from g.take_initiative(item.controller)


_take_initiative_effect.ops = ({"op": "take_initiative"},)  # read by encode.option_preview, as Rust's Op::TakeInitiative


# The initiative's inherent triggers (725.2), with the Undercity as their source.
VENTURE_TRIGGER = TriggerDef("venture into Undercity", "initiative", _venture_effect)
INITIATIVE_TRIGGER = TriggerDef("take the initiative", "initiative", _take_initiative_effect)
