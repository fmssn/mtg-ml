"""Static card definitions and the mutable objects a game is made of."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from .mana import ManaCost

# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@dataclass
class Option:
    """One legal choice at a decision point.

    label: human readable text.
    key:   stable, id-free descriptor (e.g. ('cast', 'Brainstorm', 'hand',
           'normal')) meant for building per-deck action vocabularies.
    value: engine-internal payload, opaque to agents.
    """

    label: str
    key: tuple
    value: Any = None

    def __repr__(self) -> str:
        return f"Option({self.label!r})"


@dataclass
class Decision:
    player: int
    kind: str
    prompt: str
    options: list[Option]

    def __repr__(self) -> str:
        return f"Decision(p{self.player} {self.kind}: {self.prompt!r}, {len(self.options)} options)"


# Decision kinds. Mirrors MageZero's split into priority / target / binary
# style heads, but finer grained so encoders can route as they like.
PRIORITY = "priority"  # pass, play land, cast, activate
CHOOSE_X = "choose_x"
TARGET = "target"
PAY_MANA = "pay_mana"
SACRIFICE = "sacrifice"  # additional sacrifice cost
EXILE_FROM_GY = "exile_from_graveyard"  # escape cost
YES_NO = "yes_no"  # optional effects, optional payments
CHOOSE_CARD = "choose_card"  # discard, put back, search
ORDER = "order"  # library ordering (Ponder)
ORDER_TRIGGERS = "order_triggers"
DECLARE_ATTACKER = "declare_attacker"
DECLARE_BLOCKER = "declare_blocker"
ASSIGN_DAMAGE = "assign_damage"
CHOOSE_MODE = "choose_mode"  # e.g. Deem Inferior: second from top or bottom
MULLIGAN = "mulligan"  # keep or mulligan the opening hand (London mulligan)

# ---------------------------------------------------------------------------
# Static definitions
# ---------------------------------------------------------------------------

Effect = Callable[..., Any]  # (game, item) -> generator | None


class _Immutable:
    """Static definitions are never mutated during a game: `Game.copy()`
    (a deep copy of the game state) shares them instead of copying."""

    __slots__ = ()

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class TargetSpec(_Immutable):
    """kind is one of: creature, nonlegendary_creature, nonartifact_creature,
    creature_you_control, land, nonland_permanent, artifact, blue_permanent,
    red_permanent, spell, blue_spell, red_spell, instant_spell,
    artifact_spell, player, opponent, any (creature or player)."""

    kind: str


@dataclass(frozen=True)
class SpellMode(_Immutable):
    """One choice of a modal spell ("Choose one -"): chosen as the spell is cast."""

    name: str
    targets: tuple[TargetSpec, ...]
    effect: "Effect"


@dataclass
class AbilityDef(_Immutable):
    name: str
    effect: Effect | None = None
    cost: ManaCost = field(default_factory=ManaCost)
    tap: bool = False
    sac_self: bool = False
    sac_other: str | None = None  # 'artifact' | 'artifact_or_creature'
    discard_self: bool = False  # cycling-style abilities
    discard_other: bool = False  # discard a card as a cost (Blood)
    exile_self: bool = False  # exile this permanent as a cost (Relic of Progenitus)
    x_target_mv: int = 0  # cost has this many {X}, X = the target's mana value (Gorilla Shaman)
    x_reveal: str | None = None  # 'red': X = number of red cards revealed from hand (Martyr of Ashes)
    zone: str = "battlefield"  # 'hand' for cycling
    sorcery_speed: bool = False
    mana: tuple[str, ...] | None = None  # set => mana ability producing one of these
    targets: tuple[TargetSpec, ...] = ()


@dataclass
class TriggerDef(_Immutable):
    """event: etb | to_graveyard_from_battlefield | cast | you_sacrifice_another
    | your_upkeep | you_cast (another spell you cast, from the battlefield)
    | third_draw (you draw your third card in a turn, from the graveyard).
    condition(game, source, event_data) -> bool."""

    name: str
    event: str
    effect: Effect
    condition: Callable[..., bool] | None = None


@dataclass
class CardDef(_Immutable):
    name: str
    cost: ManaCost
    types: frozenset[str]
    subtypes: frozenset[str] = frozenset()
    supertypes: frozenset[str] = frozenset()
    colors: frozenset[str] = frozenset()
    power: int | None = None
    toughness: int | None = None
    keywords: frozenset[str] = frozenset()
    ward: int = 0  # ward {N}
    text: str = ""
    # spells
    targets: tuple[TargetSpec, ...] = ()
    effect: Effect | None = None  # instant/sorcery resolution
    additional_sac: str | None = None  # 'artifact_or_creature'
    cost_reduction: Callable[..., int] | None = None  # (game, player) -> int
    flashback: ManaCost | None = None
    escape: ManaCost | None = None
    escape_exile: int = 0
    bestow: ManaCost | None = None
    madness: ManaCost | None = None
    plot: ManaCost | None = None
    overload: ManaCost | None = None
    overload_effect: Effect | None = None
    additional_discard: bool = False  # discard a card as an additional cost
    # Costs of sacrificing lands instead of mana: (sacrifice filter, count).
    alternative_sac: tuple[str, int] | None = None  # cast mode "alternative" (Fireblast)
    flashback_sac: tuple[str, int] | None = None  # flashback cost (Lava Dart)
    # permanents
    abilities: tuple[AbilityDef, ...] = ()
    triggers: tuple[TriggerDef, ...] = ()
    enters_tapped: bool = False
    etb_x_counters: bool = False
    back: "CardDef | None" = None
    modes: tuple[SpellMode, ...] = ()  # modal spells: one is chosen on cast

    def is_type(self, t: str) -> bool:
        return t in self.types

    @property
    def mana_value(self) -> int:
        return self.cost.mana_value

    @property
    def is_permanent_card(self) -> bool:
        return bool(self.types & {"Artifact", "Creature", "Enchantment", "Land", "Planeswalker", "Battle"})


FREE = ManaCost()  # "without paying its mana cost" (plot) and land-sacrifice alternative costs

# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------


@dataclass
class TempEffect:
    """An 'until end of turn' effect on a permanent."""

    keywords: frozenset[str] = frozenset()
    power: int = 0
    toughness: int = 0


@dataclass(eq=False)
class Card:
    """A card or token. Per CR 400.7 a card that changes zones becomes a new
    object: the engine gives it a fresh `oid` and resets its state, so stale
    references (targets, combat) stop matching. `uid` identifies the physical
    card for the whole game."""

    uid: int
    oid: int
    defn: CardDef
    owner: int
    controller: int
    zone: str
    is_token: bool = False
    transformed: bool = False
    tapped: bool = False
    damage: int = 0
    deathtouch_damage: bool = False
    counters: int = 0  # +1/+1 counters
    sick: bool = False  # not continuously controlled since start of turn
    attached_to: int | None = None  # oid of enchanted creature (bestow)
    skip_untap: int = 0
    plotted_turn: int = 0  # turn this card was plotted on (exile), 0 = not plotted
    temp: list[TempEffect] = field(default_factory=list)
    known_to: set[int] = field(default_factory=set)

    @property
    def face(self) -> CardDef:
        if self.transformed and self.defn.back is not None:
            return self.defn.back
        return self.defn

    @property
    def name(self) -> str:
        return self.face.name

    def snapshot(self) -> "Card":
        """Last known information copy."""
        c = copy.copy(self)
        c.temp = list(self.temp)
        c.known_to = set(self.known_to)
        return c

    def reset_state(self) -> None:
        self.transformed = False
        self.tapped = False
        self.damage = 0
        self.deathtouch_damage = False
        self.counters = 0
        self.sick = False
        self.attached_to = None
        self.skip_untap = 0
        self.plotted_turn = 0
        self.temp = []

    def __repr__(self) -> str:
        return f"{self.name}#{self.oid}"


@dataclass(eq=False)
class StackItem:
    sid: int
    kind: str  # 'spell' | 'ability' | 'trigger'
    controller: int
    name: str
    effect: Effect | None
    target_specs: tuple[TargetSpec, ...] = ()
    targets: list[tuple] = field(default_factory=list)  # ('player', i) | ('perm', oid) | ('stack', sid)
    card: Card | None = None  # spells: the card on the stack
    source: Card | None = None  # abilities: (LKI of) the source
    method: str = "normal"  # normal | flashback | escape | bestow | madness | plot | overload | alternative
    cast_from: str = "hand"
    x: int = 0
    data: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"[{self.name} ({self.kind}) #{self.sid}]"


@dataclass
class PendingTrigger:
    controller: int
    source: Card
    tdef: TriggerDef
    data: dict = field(default_factory=dict)


@dataclass
class Player:
    idx: int
    life: int = 20
    library: list[Card] = field(default_factory=list)  # index 0 = top
    hand: list[Card] = field(default_factory=list)
    graveyard: list[Card] = field(default_factory=list)  # last = most recent
    exile: list[Card] = field(default_factory=list)
    pool: dict[str, int] = field(default_factory=dict)
    land_plays: int = 0
    drew_from_empty: bool = False
    cards_drawn_this_turn: int = 0
    landfall_turn: int = 0  # last turn a land entered the battlefield under this player's control
