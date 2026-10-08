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
ASSIGN_DAMAGE_AMOUNT = "assign_damage_amount"
CHOOSE_MODE = "choose_mode"  # e.g. Deem Inferior: second from top or bottom
MULLIGAN = "mulligan"  # keep or mulligan the opening hand (London mulligan)


@dataclass(frozen=True)
class DamageAllocation:
    """Public progress of a sequential combat allocation; recipient -1 is the defender."""

    attacker: int
    blockers: tuple[int, ...]
    lethal: tuple[int, ...]
    assigned: tuple[int, ...]
    recipient: int
    remaining: int
    defender: int
    player_damage: int  # -1 before choosing the defender's share

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
    sorcery_spell, artifact_spell, player, opponent, any (creature or player)."""

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
    zone: str = "battlefield"  # 'hand' for cycling, 'graveyard' (Bramble Wurm)
    sorcery_speed: bool = False
    mana: tuple[str, ...] | None = None  # set => mana ability producing one of these
    return_land: str | None = None  # 'forest': return a Forest you control to its owner's hand as a cost
    targets: tuple[TargetSpec, ...] = ()
    # Mana abilities: 'elves' (one unit per Elf on the battlefield, Priest of
    # Titania) or (n, subtypes): n units instead of one while its controller
    # controls permanents with each of these subtypes (Urza's Tower).
    mana_amount: str | tuple[int, tuple[str, ...]] | None = None
    # "Activate only once each turn": a mana filter (Barrels of Blasting Jelly)
    # or another activated ability (Quirion Ranger).
    once_per_turn: bool = False
    tap_other: str | None = None  # 'creature': tap another untapped creature you control as a cost (station)

    @property
    def is_filter(self) -> bool:
        """A mana ability with a mana cost ({1}: add one mana of any color)."""
        return self.mana is not None and not self.cost.is_zero()


@dataclass
class TriggerDef(_Immutable):
    """event: etb | to_graveyard_from_battlefield | cast | you_sacrifice_another
    | your_upkeep | you_cast (another spell you cast, from the battlefield)
    | third_draw (you draw your third card in a turn, from the graveyard)
    | leaves_battlefield (to any zone)
    | room (a dungeon room: the venture marker moved into it).
    condition(game, source, event_data) -> bool; for etb the event data is
    the cast method of the spell that became the permanent (or None).
    targets: chosen as the trigger is put on the stack (removed if there is
    none, CR 603.3d); up_to: the targets may be left empty ("up to one target")."""

    name: str
    event: str
    effect: Effect
    condition: Callable[..., bool] | None = None
    targets: tuple[TargetSpec, ...] = ()  # chosen as the trigger is put on the stack
    up_to: bool = False  # "up to one target": no target is a legal choice


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
    flashback_life: int = 0
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
    # Cast mode "alternative": reveal your hand instead of paying the mana
    # cost, only with no land cards in it (Land Grant).
    alternative_reveal: bool = False
    flashback_sac: tuple[str, int] | None = None  # flashback cost (Lava Dart)
    # Phyrexian mana ({R/P}): cast mode "phyrexian" pays `phyrexian_cost` (the
    # cost without those symbols) and 2 life per symbol (Gut Shot).
    phyrexian_cost: ManaCost | None = None
    phyrexian_life: int = 0
    bargain: bool = False  # cast mode "bargain": also sacrifice an artifact, enchantment or token
    collect_evidence: int = 0  # cast mode "evidence": exile cards with total mana value >= N from your graveyard
    # Equipment: what the equipped creature gets (Black Mage's Rod: +1/+0).
    equipped_power: int = 0
    equipped_toughness: int = 0
    # permanents
    abilities: tuple[AbilityDef, ...] = ()
    triggers: tuple[TriggerDef, ...] = ()
    enters_tapped: bool = False
    etb_x_counters: bool = False
    back: "CardDef | None" = None
    modes: tuple[SpellMode, ...] = ()  # modal spells: one is chosen on cast
    omen: bool = False  # the back face is an omen: cast mode "omen" casts it, then it is shuffled into the library
    enters_tapped_unless_forests: int = 0  # enters tapped unless you control this many other Forests
    additional_power: bool = False  # additional cost: choose a creature you control or reveal a creature card (Monstrous Emergence)
    # What the card does, as entity tokens derived from its spec (cards.card_shape;
    # feature set 5, docs/features.md). Computed once at load.
    shape: tuple[str, ...] = ()
    prototype: ManaCost | None = None  # cast mode "prototype" (Boulderbranch Golem)
    prototype_face: "CardDef | None" = None  # its characteristics while prototyped
    station: int = 0  # Spacecraft: a creature with this many charge counters
    station_keywords: frozenset[str] = frozenset()  # ... and these keywords
    additional_choose_creature: bool = False  # choose a creature you control or reveal one from hand (Monstrous Emergence)
    equipped_keywords: frozenset[str] = frozenset()

    def is_type(self, t: str) -> bool:
        return t in self.types

    @property
    def is_equipment(self) -> bool:
        return "Equipment" in self.subtypes

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
    attached_to: int | None = None  # oid of the enchanted (bestow) or equipped creature
    skip_untap: int = 0
    plotted_turn: int = 0  # turn this card was plotted on (exile), 0 = not plotted
    once_used_turn: int = 0  # turn its once-each-turn ability was activated
    hexproof: bool = False  # hexproof until its controller's next turn (Throne of the Dead Three)
    prototyped: bool = False  # cast (and on the battlefield) as its prototype
    charge: int = 0  # charge counters (station)
    mana_used_turn: int = 0  # turn a once-per-turn mana ability was last activated
    temp: list[TempEffect] = field(default_factory=list)
    known_to: set[int] = field(default_factory=set)
    # Until it leaves the battlefield: base (power, toughness) of a permanent
    # that became a creature (Kenku Artificer), and keywords it gained
    # (Kenku's flying, a lifelink counter).
    animated: tuple[int, int] | None = None
    granted: frozenset[str] = frozenset()

    @property
    def face(self) -> CardDef:
        if self.transformed and self.defn.back is not None:
            return self.defn.back
        if self.prototyped and self.defn.prototype_face is not None:
            return self.defn.prototype_face
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
        self.once_used_turn = 0
        self.hexproof = False
        self.prototyped = False
        self.charge = 0
        self.mana_used_turn = 0
        self.temp = []
        self.animated = None
        self.granted = frozenset()

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
    method: str = "normal"  # normal | flashback | escape | bestow | madness | plot | overload | alternative | prototype | cascade
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
    dungeon_room: str | None = None  # the room of Undercity their venture marker is in
