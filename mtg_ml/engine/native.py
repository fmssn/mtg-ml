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
from .objects import TempEffect

with open(SPEC_PATH, encoding="utf-8") as _f:
    _n.load_cards(_f.read())

_ALL_DEFS = {**FACES, **TOKENS, **CARDS}


_CARD_FIELDS = (
    "uid", "oid", "name", "_defn_name", "owner", "controller", "zone", "is_token", "transformed", "tapped", "damage",
    "deathtouch_damage", "counters", "sick", "attached_to", "skip_untap", "_temp", "_known",
)  # fmt: skip
_SETTABLE = {"tapped", "transformed", "sick", "deathtouch_damage", "damage", "counters", "skip_untap", "attached_to", "controller"}


class NativeCard:
    """A card of a native game, like `objects.Card`: one proxy per physical
    card for the whole game (identity is stable), whose attributes always
    show the card's current state. Last-known-information copies (a stack
    item's source, a dies trigger's source) are frozen snapshots."""

    __slots__ = ("_g", "_idx", "_snap")

    def __init__(self, g: "NativeGame | None", idx: int | None, snap: tuple | None = None):
        self._g, self._idx, self._snap = g, idx, snap

    def _t(self) -> tuple:
        return self._snap if self._idx is None else self._g._info(self._idx)

    @property
    def temp(self) -> list[TempEffect]:
        t = [TempEffect(keywords=frozenset(k), power=p, toughness=tt) for k, p, tt in self._t()[16]]
        return t if self._idx is None else _TempList(self, t)

    @property
    def known_to(self) -> set[int]:
        return set(self._t()[17])

    @property
    def face(self):
        return _ALL_DEFS[self.name]

    @property
    def defn(self):
        return _ALL_DEFS[self._defn_name]

    def __repr__(self) -> str:
        return f"{self.name}#{self.oid}"


def _field(i: int, name: str):
    def get(self):
        return self._t()[i]

    if name not in _SETTABLE:
        return property(get)

    def set_(self, v):
        if self._idx is None:
            raise AttributeError("last-known-information copies are read-only")
        self._g._g.set_card_field(self._idx, name, v)
        self._g._touch()

    return property(get, set_)


for _i, _name in enumerate(_CARD_FIELDS[:16]):
    setattr(NativeCard, _name, _field(_i, _name))


class _TempList(list):
    """`card.temp`; `append()` adds the effect to the engine's card."""

    def __init__(self, card: NativeCard, items):
        super().__init__(items)
        self._card = card

    def append(self, t: TempEffect) -> None:
        c = self._card
        c._g._g.add_temp(c._idx, sorted(t.keywords), t.power, t.toughness)
        c._g._touch()
        super().append(t)


class NativePlayer:
    __slots__ = ("_g", "idx", "_life", "library", "hand", "graveyard", "exile", "pool", "drew_from_empty", "_drawn")

    def __init__(self, g: "NativeGame", idx: int):
        life, lib, hand, gy, ex, pool, drew, drawn = g._g.player(idx)
        self._g, self.idx, self._life = g, idx, life
        card = g._card
        self.library = [card(i) for i in lib]
        self.hand = [card(i) for i in hand]
        self.graveyard = [card(i) for i in gy]
        self.exile = [card(i) for i in ex]
        self.pool = dict(pool)
        self.drew_from_empty = drew
        self._drawn = drawn

    @property
    def life(self) -> int:
        return self._life

    @life.setter
    def life(self, v: int) -> None:
        self._g._g.set_life(self.idx, v)
        self._life = v

    @property
    def cards_drawn_this_turn(self) -> int:
        return self._drawn

    @cards_drawn_this_turn.setter
    def cards_drawn_this_turn(self, v: int) -> None:
        self._g._g.set_cards_drawn_this_turn(self.idx, v)
        self._drawn = v


class NativeStackItem:
    __slots__ = ("sid", "kind", "controller", "name", "targets", "card", "method", "cast_from", "x", "source", "data")

    def __init__(self, g: "NativeGame", info: tuple):
        sid, self.kind, self.controller, self.name, targets, card, self.method, self.cast_from, self.x, src, data = info
        self.sid = sid
        self.targets = [tuple(t) for t in targets]
        self.card = None if card is None else g._card(card)
        self.source = None if src is None else NativeCard(None, None, src)
        self.data = {k: (NativeCard(None, None, v) if k in ("card", "sacrificed") else v) for k, v in data}

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
        self._g, self._version = g, g._version
        self.player, self.kind, self.prompt = head
        self._options = self._keys = self._label_list = None

    @property
    def options(self) -> list[NativeOption]:
        if self._options is None:
            self._options = [NativeOption(self, i) for i in range(self._g._g.n_options())]
        return self._options

    @property
    def keys(self) -> list[tuple]:
        if self._keys is None:
            self._keys = self._g._g.option_keys()
        return self._keys

    def _labels(self) -> list[str]:
        if self._label_list is None:
            self._label_list = self._g._g.option_labels()
        return self._label_list

    def _value(self, i: int):
        if self._version != self._g._version:
            raise RulesError("option value read after the game moved on")
        return self._g._resolve(self._g._g.option_value(i))

    def __repr__(self) -> str:
        return f"Decision(p{self.player} {self.kind}: {self.prompt!r}, {len(self.options)} options)"


class NativePendingTrigger:
    __slots__ = ("controller", "source", "tdef", "data")

    def __init__(self, info: tuple):
        self.controller, src, name, data = info
        self.source = NativeCard(None, None, src)
        self.tdef = _TriggerName(name)
        self.data = {k: (NativeCard(None, None, v) if k in ("card", "sacrificed") else v) for k, v in data}


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
        self._g._touch()
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
        )
        self.cards_db = CARDS
        self.match_game = match_game
        self.auto_single = auto_single
        self.max_turns = max_turns
        self.logging = log
        self._g = _n.Game(
            (list(decks[0]), list(decks[1])), seed, starting_player, auto_single, max_turns, log, setup is not None, start_step, mulligans, match_game
        )
        self._version = 0
        self._cache: dict = {}
        self._proxies: dict[int, NativeCard] = {}
        if setup is not None:
            setup(self)
            self._touch()
        try:
            self._g.start()
        except _n.NativeRulesError as e:
            raise RulesError(str(e)) from None

    # -- caching -------------------------------------------------------------

    def _touch(self) -> None:
        self._version += 1
        self._cache = {}

    def _cached(self, name, make):
        c = self._cache
        if name not in c:
            c[name] = make()
        return c[name]

    def _card(self, idx: int) -> NativeCard:
        c = self._proxies.get(idx)
        if c is None:
            c = self._proxies[idx] = NativeCard(self, idx)
        return c

    def _info(self, idx: int) -> tuple:
        info = self._cached("info", dict)
        t = info.get(idx)
        if t is None:
            t = info[idx] = self._g.card_info(idx)
        return t

    def _resolve(self, v):
        """Option values: ("card", idx) markers become card proxies."""
        if isinstance(v, tuple):
            if len(v) == 2 and v[0] == "card" and isinstance(v[1], int):
                return self._card(v[1])
            return tuple(self._resolve(x) for x in v)
        return v

    @staticmethod
    def _idx(card) -> int:
        if not isinstance(card, NativeCard) or card._idx is None:
            raise TypeError(f"expected a live card of this game, got {card!r}")
        return card._idx

    # -- public API (mirrors game.Game) --------------------------------------

    @property
    def decision(self) -> NativeDecision | None:
        c = self._cache
        if "decision" not in c:
            head = self._g.decision_head()
            c["decision"] = None if head is None else NativeDecision(self, head)
        return c["decision"]

    def legal_options(self) -> list:
        d = self.decision
        return [] if d is None else d.options

    def step(self, index: int) -> None:
        try:
            self._g.step(index)
        except _n.NativeRulesError as e:
            raise RulesError(str(e)) from None
        finally:
            self._touch()

    def fork(self) -> "NativeGame":
        g = NativeGame(**self._args)
        try:
            g._g.replay(self._g.actions)
        except _n.NativeRulesError as e:
            raise RulesError(str(e)) from None
        g._touch()
        return g

    def add_card(self, name: str, player: int, zone: str, tapped: bool = False, sick: bool = False, counters: int = 0) -> NativeCard:
        idx = self._g.add_card(name, player, zone, tapped, sick, counters)
        self._touch()
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
        self._touch()

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
        self._touch()

    def destroy(self, card) -> bool:
        r = self._g.destroy(self._idx(card))
        self._touch()
        return r

    def sacrifice(self, card) -> None:
        self._g.sacrifice(self._idx(card))
        self._touch()

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

    def types(self, c) -> set[str]:
        return set(self._g.types(self._idx(c)))

    def is_creature(self, c) -> bool:
        return self._g.is_creature(self._idx(c))

    def is_artifact(self, c) -> bool:
        return "Artifact" in c.face.types

    def is_land(self, c) -> bool:
        return "Land" in c.face.types

    def power(self, c) -> int:
        return self._g.power(self._idx(c))

    def toughness(self, c) -> int:
        return self._g.toughness(self._idx(c))

    def keywords(self, c) -> set[str]:
        return set(self._g.keywords(self._idx(c)))

    def has(self, c, kw: str) -> bool:
        return self._g.has(self._idx(c), kw)

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

    def target_candidates(self, spec, controller: int, exclude_sid=None) -> list[tuple]:
        assert exclude_sid is None, "exclude_sid is engine-internal"
        return [tuple(r) for r in self._g.target_candidates(spec.kind, controller)]

    def _mode_cost(self, card, mode: str) -> ManaCost | None:
        d = card.face
        return {"normal": d.cost, "bestow": d.bestow, "flashback": d.flashback, "escape": d.escape}.get(mode)

    def _cost_reduction(self, p: int, card) -> int:
        return self._g.cost_reduction(p, self._idx(card))

    def _lethal(self, attacker, blocker) -> int:
        return self._g.lethal(self._idx(attacker), self._idx(blocker))

    # -- views -----------------------------------------------------------------

    def observe(self, viewer: int) -> dict:
        return self._g.observe(viewer)

    def state_features(self, viewer: int) -> list[str]:
        return self._g.state_features(viewer)

    def featurize(self, player: int, state_dim: int, option_dim: int):
        return self._g.featurize(player, state_dim, option_dim)

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
        f._touch()
        return f

    def __repr__(self) -> str:
        return f"NativeGame(turn={self.turn}, step={self.step_name!r}, over={self.over})"
