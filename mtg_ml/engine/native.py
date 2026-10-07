"""`NativeGame`: the Rust engine (`mtg_ml_native`) behind the `Game` API.

The Rust port plays exactly the same games as `game.Game` (same seeds, same
decisions, same options in the same order, same RNG stream), so this class
only adapts the interface:

* the hot path (`decision.kind/player`, option keys, `step`, `observe`,
  `state_features`, `featurize`) goes straight to Rust;
* the object model (`players`, `battlefield`, `stack`, `Card`s, option
  values) is exposed through read-only proxies built on demand and cached
  until the next step, which is what the scripted bots and the per-card
  tests read. `card.face` / `card.defn` are the Python `CardDef`s, so static
  card data has one source of truth (`cards.toml`).

Mutating proxies (e.g. `card.tapped = True`) does not reach the engine; the
supported mutations are the ones scenario setups use (`add_card`,
`players[i].life`, `players[i].cards_drawn_this_turn`) plus `active`,
`rng` and determinization.
"""

from __future__ import annotations

import random

import mtg_ml_native as _n

from .cards import CARDS, FACES, SPEC_PATH, TOKENS
from .game import RulesError
from .mana import ManaCost
from .objects import FREE, TempEffect

with open(SPEC_PATH, encoding="utf-8") as _f:
    _n.load_cards(_f.read())

_ALL_DEFS = {**FACES, **TOKENS, **CARDS}


class NativeCard(_n.CardView):
    """A card of a native game, like `objects.Card`: one view per physical
    card for the whole game (identity is stable). The attributes (`oid`,
    `name`, `tapped`, `counters`...) are Rust getters reading the engine's
    current state; the settable ones are the fields scenario tests poke."""

    __slots__ = ()

    @property
    def temp(self) -> list[TempEffect]:
        return _TempList(self, [TempEffect(keywords=frozenset(k), power=p, toughness=t) for k, p, t in self._temp])

    @property
    def known_to(self) -> set[int]:
        return set(self._known)

    @property
    def face(self):
        return _ALL_DEFS[self.name]

    @property
    def defn(self):
        return _ALL_DEFS[self._defn_name]

    def __repr__(self) -> str:
        return f"{self.name}#{self.oid}"


_SNAP_FIELDS = (
    "uid", "oid", "name", "_defn_name", "owner", "controller", "zone", "is_token", "transformed", "tapped", "damage",
    "deathtouch_damage", "counters", "sick", "attached_to", "skip_untap", "_temp", "_known",
)  # fmt: skip


class NativeSnapshot:
    """A frozen last-known-information copy (a stack item's or trigger's source)."""

    __slots__ = _SNAP_FIELDS
    _idx = None

    def __init__(self, info: tuple):
        for k, v in zip(_SNAP_FIELDS, info):
            object.__setattr__(self, k, v)

    temp = property(lambda self: [TempEffect(keywords=frozenset(k), power=p, toughness=t) for k, p, t in self._temp])
    known_to = property(lambda self: set(self._known))
    face = property(lambda self: _ALL_DEFS[self.name])
    defn = property(lambda self: _ALL_DEFS[self._defn_name])

    def __repr__(self) -> str:
        return f"{self.name}#{self.oid}"


class _TempList(list):
    """`card.temp`; `append()` adds the effect to the engine's card."""

    def __init__(self, card: NativeCard, items):
        super().__init__(items)
        self._card = card

    def append(self, t: TempEffect) -> None:
        c = self._card
        c._game.add_temp(c._idx, sorted(t.keywords), t.power, t.toughness)
        super().append(t)


class NativePlayer:
    """`game.players[i]`; zone lists are built on first access."""

    __slots__ = ("_g", "idx", "_info", "_zones")

    def __init__(self, g: "NativeGame", idx: int):
        self._g, self.idx = g, idx
        self._info = g._g.player(idx)
        self._zones = {}

    def _zone(self, i: int) -> list:
        z = self._zones.get(i)
        if z is None:
            card = self._g._card
            z = self._zones[i] = [card(c) for c in self._info[i]]
        return z

    library = property(lambda self: self._zone(1))
    hand = property(lambda self: self._zone(2))
    graveyard = property(lambda self: self._zone(3))
    exile = property(lambda self: self._zone(4))
    pool = property(lambda self: dict(self._info[5]))
    drew_from_empty = property(lambda self: self._info[6])
    landfall_turn = property(lambda self: self._info[8])

    @property
    def life(self) -> int:
        return self._info[0]

    @life.setter
    def life(self, v: int) -> None:
        self._g._g.set_life(self.idx, v)
        self._info = self._g._g.player(self.idx)

    @property
    def cards_drawn_this_turn(self) -> int:
        return self._info[7]

    @cards_drawn_this_turn.setter
    def cards_drawn_this_turn(self, v: int) -> None:
        self._g._g.set_cards_drawn_this_turn(self.idx, v)
        self._info = self._g._g.player(self.idx)


class NativeStackItem:
    __slots__ = ("sid", "kind", "controller", "name", "targets", "card", "method", "cast_from", "x", "source", "data")

    def __init__(self, g: "NativeGame", info: tuple):
        sid, self.kind, self.controller, self.name, targets, card, self.method, self.cast_from, self.x, src, data = info
        self.sid = sid
        self.targets = [tuple(t) for t in targets]
        self.card = None if card is None else g._card(card)
        self.source = None if src is None else NativeSnapshot(src)
        self.data = {k: (NativeSnapshot(v) if k in ("card", "sacrificed") else v) for k, v in data}

    def __repr__(self) -> str:
        return f"[{self.name} ({self.kind}) #{self.sid}]"


class NativeOption:
    __slots__ = ("_d", "_i")

    def __init__(self, d: "NativeDecision", i: int):
        self._d, self._i = d, i

    @property
    def label(self) -> str:
        return self._d._labels()[self._i]

    @property
    def key(self) -> tuple:
        return self._d.keys[self._i]

    @property
    def value(self):
        return self._d._value(self._i)

    def __repr__(self) -> str:
        return f"Option({self.label!r})"


class NativeDecision:
    __slots__ = ("_g", "_version", "player", "kind", "prompt", "_options", "_keys", "_label_list")

    def __init__(self, g: "NativeGame", head: tuple):
        self._g, self._version = g, g._g.version
        self.player, self.kind, self.prompt = head
        self._options = self._keys = self._label_list = None

    def _check(self) -> None:
        """Lazily loaded fields come from the engine's *current* decision, so
        refuse to load them once the game has moved past this one."""
        if self._version != self._g._g.version:
            raise RulesError("decision read after the game moved on")

    @property
    def options(self) -> list[NativeOption]:
        if self._options is None:
            self._check()
            self._options = [NativeOption(self, i) for i in range(self._g._g.n_options())]
        return self._options

    @property
    def keys(self) -> list[tuple]:
        if self._keys is None:
            self._check()
            self._keys = self._g._g.option_keys()
        return self._keys

    def _labels(self) -> list[str]:
        if self._label_list is None:
            self._check()
            self._label_list = self._g._g.option_labels()
        return self._label_list

    def _value(self, i: int):
        self._check()
        return self._g._resolve(self._g._g.option_value(i))

    def __repr__(self) -> str:
        return f"Decision(p{self.player} {self.kind}: {self.prompt!r}, {len(self.options)} options)"


class NativePendingTrigger:
    __slots__ = ("controller", "source", "tdef", "data")

    def __init__(self, info: tuple):
        self.controller, src, name, data = info
        self.source = NativeSnapshot(src)
        self.tdef = _TriggerName(name)
        self.data = {k: (NativeSnapshot(v) if k in ("card", "sacrificed") else v) for k, v in data}


class _TriggerName:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class _PendingList(list):
    """`game.pending`; `clear()` clears the engine's list too."""

    def __init__(self, g: "NativeGame"):
        super().__init__(NativePendingTrigger(t) for t in g._g.pending())
        self._g = g

    def clear(self) -> None:
        self._g._g.clear_pending()
        super().clear()


class _NativeRng:
    """`game.rng`: getstate/setstate on the engine's MT19937."""

    def __init__(self, g: "NativeGame"):
        self._g = g

    def getstate(self):
        return self._g._g.rng_state()

    def setstate(self, state) -> None:
        self._g._g.set_rng_state(list(state[1]))


class NativeGame:
    NATIVE = True

    def __init__(
        self,
        decks,
        seed: int = 0,
        starting_player: int | None = None,
        auto_single: bool = True,
        max_turns: int = 100,
        log: bool = False,
        setup=None,
        start_step: str = "untap",
        mulligans: bool = True,
        match_game: int = 1,
        auto_mana: bool = False,
        auto_pass: bool = False,
        deck_names: tuple[str | None, str | None] | None = None,
        _snapshots: bool = False,
    ):
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
        # Names of decks that are not the default for their seat (Game.deck_names).
        self.deck_names = tuple(deck_names) if deck_names else (None, None)
        self.cards_db = CARDS
        self.match_game = match_game
        self.auto_single = auto_single
        self.auto_mana = auto_mana
        self.auto_pass = auto_pass
        self.max_turns = max_turns
        self.logging = log
        self._g = _n.Game(
            (list(decks[0]), list(decks[1])),
            seed,
            starting_player,
            auto_single,
            max_turns,
            log,
            setup is not None,
            start_step,
            mulligans,
            match_game,
            auto_mana,
            auto_pass,
            self.deck_names,
        )
        self._cache: dict = {}
        self._cache_version = -1
        self._proxies: dict[int, NativeCard] = {}
        if setup is not None:
            setup(self)
        if _snapshots:
            self._g.set_snapshots(True)
        try:
            self._g.start()
        except _n.NativeRulesError as e:
            raise RulesError(str(e)) from None

    # -- caching -------------------------------------------------------------

    def _cached(self, name, make):
        """Per-state cache: cleared whenever the engine's version moves
        (every step and every mutation bump it)."""
        v = self._g.version
        if v != self._cache_version:
            self._cache = {}
            self._cache_version = v
        c = self._cache
        if name not in c:
            c[name] = make()
        return c[name]

    def _card(self, idx: int) -> NativeCard:
        c = self._proxies.get(idx)
        if c is None:
            c = self._proxies[idx] = NativeCard(self._g, idx)
        return c

    def _resolve(self, v):
        """Option values: ("card", idx) markers become card proxies (values
        nest at most one level, e.g. ("cast", ("card", 3), "normal"))."""
        if type(v) is not tuple:
            return v
        if len(v) == 2 and v[0] == "card":
            return self._card(v[1])
        card = self._card
        return tuple([card(x[1]) if type(x) is tuple and len(x) == 2 and x[0] == "card" else x for x in v])

    @staticmethod
    def _idx(card) -> int:
        if card.__class__ is not NativeCard:
            raise TypeError(f"expected a live card of this game, got {card!r}")
        return card._idx

    # -- public API (mirrors game.Game) --------------------------------------

    @property
    def decision(self) -> NativeDecision | None:
        return self._cached("decision", self._new_decision)

    def _new_decision(self) -> NativeDecision | None:
        head = self._g.decision_head()
        return None if head is None else NativeDecision(self, head)

    def legal_options(self) -> list:
        d = self.decision
        return [] if d is None else d.options

    def step(self, index: int) -> None:
        try:
            self._g.step(index)
        except _n.NativeRulesError as e:
            raise RulesError(str(e)) from None

    def fork(self, replay: bool = False) -> "NativeGame":
        """Exact copy (`copy()`); `replay=True` rebuilds it from the
        constructor arguments and the action history (Game.fork)."""
        if replay:
            return self._replay()
        return self.copy()

    def copy(self) -> "NativeGame":
        """Game.copy: a step-start snapshot restore plus the actions since;
        the first copy replays the whole history and turns snapshots on."""
        try:
            inner = self._g.copy()
        except _n.NativeRulesError as e:
            raise RulesError(str(e)) from None
        if inner is None:
            g = self._replay(snapshots=True)
            g._g.set_snapshots(False)
            self._g.adopt_snapshot(g._g)
            return g
        g = object.__new__(NativeGame)
        g.__dict__.update(self.__dict__)
        g._g = inner
        g._cache = {}
        g._cache_version = -1
        g._proxies = {}
        return g

    def _replay(self, snapshots: bool = False) -> "NativeGame":
        g = NativeGame(**self._args, _snapshots=snapshots)
        try:
            g._g.replay(self._g.actions)
        except _n.NativeRulesError as e:
            raise RulesError(str(e)) from None
        return g

    def add_card(self, name: str, player: int, zone: str, tapped: bool = False, sick: bool = False, counters: int = 0) -> NativeCard:
        idx = self._g.add_card(name, player, zone, tapped, sick, counters)
        return self._card(idx)

    @property
    def over(self) -> bool:
        return self._g.over

    @property
    def winner(self) -> int | None:
        return self._g.winner

    @property
    def end_reason(self) -> str:
        return self._g.end_reason

    @property
    def turn(self) -> int:
        return self._g.turn

    @property
    def step_name(self) -> str:
        return self._g.step_name

    @property
    def active(self) -> int:
        return self._g.active

    @active.setter
    def active(self, v: int) -> None:
        self._g.active = v

    @property
    def starting_player(self) -> int:
        return self._g.starting_player

    @property
    def lands_played(self) -> int:
        return self._g.lands_played

    @property
    def mulligans_taken(self) -> list[int]:
        return self._g.mulligans_taken

    @property
    def actions(self) -> list[int]:
        return self._g.actions

    @property
    def log(self) -> list[str]:
        return self._g.log

    @property
    def rng(self) -> _NativeRng:
        return _NativeRng(self)

    @rng.setter
    def rng(self, r: random.Random) -> None:
        self._g.set_rng_state(list(r.getstate()[1]))

    @property
    def players(self) -> list[NativePlayer]:
        return self._cached("players", lambda: [NativePlayer(self, 0), NativePlayer(self, 1)])

    @property
    def battlefield(self) -> list[NativeCard]:
        return self._cached("battlefield", lambda: [self._card(i) for i in self._g.battlefield])

    @property
    def stack(self) -> list[NativeStackItem]:
        return self._cached("stack", lambda: [NativeStackItem(self, it) for it in self._g.stack()])

    @property
    def attackers(self) -> list[int]:
        return self._cached("attackers", lambda: self._g.attackers)

    @property
    def blocked(self) -> set[int]:
        return self._cached("blocked", lambda: set(self._g.blocked))

    @property
    def blocks(self) -> dict[int, int]:
        return self._cached("blocks", lambda: dict(self._g.blocks))

    @property
    def pending(self) -> list[NativePendingTrigger]:
        return self._cached("pending", lambda: _PendingList(self))

    # -- direct rules actions (scenario tests) ---------------------------------

    def _untap_step(self) -> None:
        self._g.untap_step()

    def destroy(self, card) -> bool:
        r = self._g.destroy(self._idx(card))
        return r

    def sacrifice(self, card) -> None:
        self._g.sacrifice(self._idx(card))

    # -- characteristics and rules queries -----------------------------------

    def perm(self, oid: int) -> NativeCard | None:
        i = self._g.perm(oid)
        return None if i is None else self._card(i)

    def live(self, card) -> NativeCard | None:
        return self.perm(card.oid)

    def stack_item(self, sid: int) -> NativeStackItem | None:
        for it in self.stack:
            if it.sid == sid:
                return it
        return None

    def is_bestowed(self, c) -> bool:
        return c.zone == "battlefield" and c.attached_to is not None

    # Live cards go to Rust; last-known-information snapshots (stack item and
    # trigger sources) are computed here from their stored characteristics,
    # exactly as `Game` does for its `Card.snapshot()` copies.

    def _auras_on(self, c) -> list[NativeCard]:
        return [a for a in self.battlefield if a.attached_to == c.oid]

    def types(self, c) -> set[str]:
        if c.__class__ is NativeCard:
            return set(self._g.types(c._idx))
        t = set(c.face.types)
        if self.is_bestowed(c):
            t.discard("Creature")
        return t

    def is_creature(self, c) -> bool:
        if c.__class__ is NativeCard:
            return self._g.is_creature(c._idx)
        return "Creature" in c.face.types and not self.is_bestowed(c)

    def is_artifact(self, c) -> bool:
        return "Artifact" in c.face.types

    def is_land(self, c) -> bool:
        return "Land" in c.face.types

    def power(self, c) -> int:
        if c.__class__ is NativeCard:
            return self._g.power(c._idx)
        return (c.face.power or 0) + c.counters + sum(t.power for t in c.temp) + sum(a.counters for a in self._auras_on(c))

    def toughness(self, c) -> int:
        if c.__class__ is NativeCard:
            return self._g.toughness(c._idx)
        return (c.face.toughness or 0) + c.counters + sum(t.toughness for t in c.temp) + sum(a.counters for a in self._auras_on(c))

    def keywords(self, c) -> set[str]:
        if c.__class__ is NativeCard:
            return set(self._g.keywords(c._idx))
        k = set(c.face.keywords)
        for t in c.temp:
            k |= t.keywords
        if self._auras_on(c):
            k |= {"reach", "trample"}
        return k

    def has(self, c, kw: str) -> bool:
        if c.__class__ is NativeCard:
            return self._g.has(c._idx, kw)
        return kw in self.keywords(c)

    def sorcery_timing(self, p: int) -> bool:
        return self._g.sorcery_timing(p)

    def mana_sources(self, p: int, exclude=frozenset()):
        out = []
        for i, ai in self._g.mana_sources(p):
            c = self._card(i)
            if c.oid not in exclude:
                out.append((c, c.face.abilities[ai]))
        return out

    def sac_candidates(self, p: int, flt: str, exclude=frozenset()) -> list[NativeCard]:
        return [c for c in (self._card(i) for i in self._g.sac_candidates(p, flt)) if c.oid not in exclude]

    def target_candidates(self, spec, controller: int, exclude_sid=None, chosen=None) -> list[tuple]:
        assert exclude_sid is None, "exclude_sid is engine-internal"
        return [tuple(r) for r in self._g.target_candidates(spec.kind, controller, [tuple(r) for r in chosen or ()])]

    def _mode_cost(self, card, mode: str) -> ManaCost | None:
        d = card.face
        modes = {
            "normal": d.cost,
            "bestow": d.bestow,
            "flashback": d.flashback,
            "escape": d.escape,
            "madness": d.madness,
            "overload": d.overload,
            "plot": FREE if d.plot is not None else None,
            "alternative": FREE if d.alternative_sac is not None else None,
        }
        return modes.get(mode)

    def _cost_reduction(self, p: int, card) -> int:
        return self._g.cost_reduction(p, self._idx(card))

    def _lethal(self, attacker, blocker) -> int:
        return self._g.lethal(self._idx(attacker), self._idx(blocker))

    # -- views -----------------------------------------------------------------

    def observe(self, viewer: int) -> dict:
        return self._g.observe(viewer)

    def state_features(self, viewer: int, features: int) -> list[str]:
        return self._g.state_features(viewer, features)

    def entity_features(self, viewer: int, features: int) -> tuple[list[list[str]], dict[int, int]]:
        return self._g.entity_features(viewer, features)

    def featurize(self, player: int, state_dim: int, option_dim: int, features: int):
        return self._g.featurize(player, state_dim, option_dim, features)

    def featurize_flat(self, player: int, state_dim: int, option_dim: int, features: int):
        return self._g.featurize_flat(player, state_dim, option_dim, features)

    def option_preview(self, player: int, index: int, features: int) -> list[str]:
        return self._g.option_preview(player, index, features)

    def event_hashes(self, index: int, option_dim: int):
        return self._g.event_hashes(index, option_dim)

    def dump(self) -> dict:
        return self._g.dump()

    def determinize(self, viewer: int, rng: random.Random) -> "NativeGame":
        f = self.fork()
        for p in f.players:
            hidden = [c for c in p.library if viewer not in c.known_to]
            if p.idx != viewer:
                hidden += [c for c in p.hand if viewer not in c.known_to]
            defs = sorted((c.defn for c in hidden), key=lambda d: d.name)
            rng.shuffle(defs)
            for c, d in zip(hidden, defs):
                f._g.set_card_def(c._idx, d.name)
        return f

    def __repr__(self) -> str:
        return f"NativeGame(turn={self.turn}, step={self.step_name!r}, over={self.over})"
