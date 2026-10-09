"""Deck variants: registered 75s per archetype, with play-share weights.

The manifest `variants.toml` next to this module is frozen data, generated
by `tools/build_variants.py` from a local Pauper-Research snapshot (the
dashboard's "Builds and decisions": a typical list per main-deck build and
per sideboard build). Do not edit it by hand; rebuild it. docs/deck-variants.md
explains the sources, the rules and the resulting counts.

A `Variant` is one registered 75 (60 main + 15 sideboard) of one of the
archetypes in `decks.DECKS`. The stock decks (`decks.DECKS` / `SIDEBOARDS`)
are variants too, with id ``"<archetype>:stock"``. Every variant belongs to
one split (`train`, `dev`, `test`), assigned by its canonical 75 so
duplicates always share a split.

Sideboarding never changes the 75, only its partition: `plan_for_variant`
returns a `SideboardPlan` relative to the variant's own maindeck. The stock
plan from `sideboard_plans.toml` is reused when it is valid for the variant;
otherwise the manifest stores an explicit adjusted plan for that matchup.

Everything here is validated at import (cards implemented, 60 / 15, at most
4 copies of a non-basic, every plan legal), so a bad manifest fails loudly.
"""

from __future__ import annotations

import hashlib
import os
import random
import tomllib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

from .cards import CARDS
from .decks import DECKS, SIDEBOARDS
from .sideboard import PLANS, SideboardPlan, check_deck, validate_plan

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "variants.toml")
SPLITS = ("train", "dev", "test")
# Held-out share per archetype for dev and for test (see `assign_splits`).
HELDOUT_FRACTION = 0.15


def _frozen(d: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(d))


def composition(main: Mapping[str, int], side: Mapping[str, int]) -> dict[str, int]:
    """The 75 as one multiset (partition ignored)."""
    out: dict[str, int] = {}
    for d in (main, side):
        for name, n in d.items():
            out[name] = out.get(name, 0) + n
    return out


def canonical_75(main: Mapping[str, int], side: Mapping[str, int]) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """The registered-75 identity: sorted (name, count) of main, then of sideboard."""
    return (tuple(sorted((k, v) for k, v in main.items() if v)), tuple(sorted((k, v) for k, v in side.items() if v)))


def split_key(main: Mapping[str, int], side: Mapping[str, int]) -> str:
    """Hash of the 75's composition. Two variants with the same 75 (even
    partitioned differently) share a key and therefore a split."""
    text = "|".join(f"{k}={v}" for k, v in sorted(composition(main, side).items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_splits(entries: Iterable[tuple[str, str, bool]]) -> dict[tuple[str, str], str]:
    """Split per archetype from (archetype, split_key, stock) entries.

    Distinct 75s (split keys) of an archetype, except any shared with its
    stock deck (always train), are ordered by key (a hash, so the order is
    deterministic and unrelated to weight or name). The first n_test go to
    test, the next n_dev to dev, the rest to train, with n = their number,
    n_test = max(1, round(0.15 n)) when n >= 2 and n_dev = the same when
    n >= 3. Duplicates share a key and therefore a split."""
    by_arch: dict[str, dict[str, bool]] = {}
    for arch, key, stock in entries:
        d = by_arch.setdefault(arch, {})
        d[key] = d.get(key, False) or bool(stock)
    out: dict[tuple[str, str], str] = {}
    for arch, keys in by_arch.items():
        others = sorted(k for k, stock in keys.items() if not stock)
        n = len(others)
        held = max(1, round(HELDOUT_FRACTION * n))
        n_test = held if n >= 2 else 0
        n_dev = held if n >= 3 else 0
        for i, k in enumerate(others):
            out[(arch, k)] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
        for k, stock in keys.items():
            if stock:
                out[(arch, k)] = "train"
    return out


@dataclass(frozen=True)
class VariantPlan(SideboardPlan):
    """A `SideboardPlan` relative to one variant's registered maindeck and
    sideboard: `apply()` / `side_after()` default to the variant's lists."""

    variant_id: str = ""
    main: Mapping[str, int] = field(default_factory=dict)
    side: Mapping[str, int] = field(default_factory=dict)
    basis: str = "stock"  # "stock": the table's row reused; "adjusted": stored in the manifest; "none"

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "main", _frozen(self.main))
        object.__setattr__(self, "side", _frozen(self.side))

    def apply(self, main: Mapping[str, int] | None = None) -> dict[str, int]:
        return super().apply(self.main if main is None else main)

    def side_after(self, side: Mapping[str, int] | None = None) -> dict[str, int]:
        return super().side_after(self.side if side is None else side)


@dataclass(frozen=True)
class Variant:
    id: str
    archetype: str
    main: Mapping[str, int]
    sideboard: Mapping[str, int]
    source: str
    weight: float
    provenance: Mapping[str, object]
    constructed: bool
    split: str
    stock: bool = False
    # Explicit per-opponent plan adjustments: {opponent: {"in", "out", "why"}}.
    plans: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "main", _frozen(self.main))
        object.__setattr__(self, "sideboard", _frozen(self.sideboard))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "plans", MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.plans.items()}))

    @property
    def registered_75(self) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
        return canonical_75(self.main, self.sideboard)

    @property
    def composition(self) -> dict[str, int]:
        return composition(self.main, self.sideboard)

    @property
    def split_key(self) -> str:
        return split_key(self.main, self.sideboard)


def _stock_plan_for(variant: Variant, opponent: str) -> SideboardPlan:
    return PLANS.get((variant.archetype, opponent)) or SideboardPlan.none(variant.archetype, opponent)


def _validate_variant_plan(variant: Variant, plan: SideboardPlan) -> None:
    decks = dict(DECKS)
    sides = dict(SIDEBOARDS)
    decks[variant.archetype] = variant.main
    sides[variant.archetype] = variant.sideboard
    validate_plan(plan, decks, sides)


def stock_plan_valid(variant: Variant, opponent: str) -> bool:
    try:
        _validate_variant_plan(variant, _stock_plan_for(variant, opponent))
    except ValueError:
        return False
    return True


def plan_for_variant(variant: Variant, opponent_archetype: str) -> VariantPlan:
    """The variant's games 2/3 plan against `opponent_archetype`, relative to
    the variant's own maindeck. `apply()` gives its sideboarded 60; the 75
    is unchanged."""
    if opponent_archetype not in DECKS:
        raise ValueError(f"unknown opponent archetype {opponent_archetype!r}")
    adj = variant.plans.get(opponent_archetype)
    if adj is not None:
        base = SideboardPlan(variant.archetype, opponent_archetype, dict(adj.get("in", {})), dict(adj.get("out", {})), str(adj.get("why", "")))
        basis = "adjusted"
    else:
        base = _stock_plan_for(variant, opponent_archetype)
        basis = "stock" if base.swaps else "none"
    plan = VariantPlan(base.deck, base.opponent, base.cards_in, base.cards_out, base.why,
                       variant_id=variant.id, main=variant.main, side=variant.sideboard, basis=basis)
    _validate_variant_plan(variant, plan)
    return plan


def validate_variant(v: Variant) -> None:
    if v.archetype not in DECKS:
        raise ValueError(f"variant {v.id}: unknown archetype {v.archetype!r}")
    if v.split not in SPLITS:
        raise ValueError(f"variant {v.id}: unknown split {v.split!r}")
    if not (v.weight > 0):
        raise ValueError(f"variant {v.id}: weight {v.weight!r} must be positive")
    unknown = sorted(set(v.main) - set(CARDS) | set(v.sideboard) - set(CARDS))
    if unknown:
        raise ValueError(f"variant {v.id}: cards not implemented: {unknown}")
    check_deck(f"variant {v.id}", v.main, v.sideboard)
    if v.stock and (dict(v.main) != DECKS[v.archetype] or dict(v.sideboard) != SIDEBOARDS[v.archetype]):
        raise ValueError(f"variant {v.id}: marked stock but differs from decks.DECKS / SIDEBOARDS")
    for opp in v.plans:
        if opp not in DECKS:
            raise ValueError(f"variant {v.id}: plan against unknown archetype {opp!r}")
    for opp in DECKS:
        plan_for_variant(v, opp)


def parse_manifest(text: str) -> dict[str, Variant]:
    spec = tomllib.loads(text)
    out: dict[str, Variant] = {}
    seen: dict[tuple, str] = {}
    for row in spec.get("variant", ()):
        v = Variant(
            id=row["id"], archetype=row["archetype"], main=row["main"], sideboard=row["sideboard"],
            source=row["source"], weight=float(row["weight"]), provenance=row.get("provenance", {}),
            constructed=bool(row["constructed"]), split=row["split"], stock=bool(row.get("stock", False)),
            plans=row.get("plans", {}),
        )
        if v.id in out:
            raise ValueError(f"duplicate variant id {v.id}")
        key = (v.archetype, v.registered_75)
        if key in seen:
            raise ValueError(f"variants {seen[key]} and {v.id} register the same 75")
        seen[key] = v.id
        validate_variant(v)
        out[v.id] = v
    splits = assign_splits((v.archetype, v.split_key, v.stock) for v in out.values())
    for v in out.values():
        if v.split != splits[(v.archetype, v.split_key)]:
            raise ValueError(f"variant {v.id}: split {v.split!r}, assign_splits gives {splits[(v.archetype, v.split_key)]!r}")
    for a in DECKS:
        if f"{a}:stock" not in out:
            raise ValueError(f"no stock variant for {a}")
    return out


def load_manifest(path: str = MANIFEST_PATH) -> tuple[dict[str, Variant], dict]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    spec = tomllib.loads(text)
    return parse_manifest(text), {k: v for k, v in spec.items() if k != "variant"}


# tools/build_variants.py sets this to use the helpers above while it
# (re)writes a missing or stale manifest; nothing else should.
if os.environ.get("MTG_ML_VARIANTS_SKIP_LOAD") == "1":
    VARIANTS, MANIFEST_META = {}, {}
else:
    VARIANTS, MANIFEST_META = load_manifest()


def variants(archetype: str, split: str | None = None) -> list[Variant]:
    """The archetype's variants in manifest order, optionally of one split."""
    if archetype not in DECKS:
        raise ValueError(f"unknown archetype {archetype!r}")
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return [v for v in VARIANTS.values() if v.archetype == archetype and (split is None or v.split == split)]


def _normalised(vs: Iterable[Variant]) -> dict[str, float]:
    vs = list(vs)
    total = sum(v.weight for v in vs)
    return {v.id: v.weight / total for v in vs} if total > 0 else {}


def sampling_prior(split: str) -> dict[str, dict[str, float]]:
    """{archetype: {variant_id: probability}} within `split` (archetypes
    without a variant in that split are left out)."""
    out = {}
    for a in DECKS:
        p = _normalised(variants(a, split))
        if p:
            out[a] = p
    return out


def sample_variant(archetype: str, rng: random.Random, split: str = "train") -> Variant:
    """One variant of `archetype` from `split`, drawn by weight. Uses exactly
    one `rng.random()` call."""
    vs = variants(archetype, split)
    if not vs:
        raise ValueError(f"{archetype} has no variant in split {split!r}")
    total = sum(v.weight for v in vs)
    x = rng.random() * total
    acc = 0.0
    for v in vs:
        acc += v.weight
        if x < acc:
            return v
    return vs[-1]
