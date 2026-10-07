"""Mana costs, mana pools and payment feasibility.

A mana *unit* is one mana of one color ('W', 'U', 'B', 'R', 'G' or 'C' for
colorless). Every mana source in the supported card pool produces exactly one
unit per activation, which keeps feasibility checks a small bipartite matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COLORS = ("W", "U", "B", "R", "G")
MANA_TYPES = COLORS + ("C",)

_SYMBOL = re.compile(r"\{([^}]+)\}")


@dataclass(frozen=True)
class ManaCost:
    generic: int = 0
    # Colored (and colorless-specific {C}) requirements as a sorted tuple of
    # (mana type, count) so the dataclass stays hashable.
    colored: tuple[tuple[str, int], ...] = ()
    x: int = 0  # number of {X} symbols

    def __deepcopy__(self, memo) -> "ManaCost":  # immutable: Game.copy shares it
        return self

    @staticmethod
    def parse(text: str | None) -> "ManaCost":
        if not text:
            return ManaCost()
        generic = 0
        x = 0
        colored: dict[str, int] = {}
        for sym in _SYMBOL.findall(text):
            if sym.isdigit():
                generic += int(sym)
            elif sym == "X":
                x += 1
            elif sym in MANA_TYPES:
                colored[sym] = colored.get(sym, 0) + 1
            elif len(sym) == 3 and sym.endswith("/P") and sym[0] in COLORS:
                # Phyrexian mana counts as its colour here (mana value, colour);
                # paying 2 life instead is the cast mode "phyrexian".
                colored[sym[0]] = colored.get(sym[0], 0) + 1
            else:
                raise ValueError(f"unsupported mana symbol {{{sym}}} in {text!r}")
        return ManaCost(generic, tuple(sorted(colored.items())), x)

    @staticmethod
    def phyrexian(text: str | None) -> "ManaCost":
        """The phyrexian symbols of a cost text ({R/P} -> {R}), as a cost."""
        colored: dict[str, int] = {}
        for sym in _SYMBOL.findall(text or ""):
            if len(sym) == 3 and sym.endswith("/P"):
                colored[sym[0]] = colored.get(sym[0], 0) + 1
        return ManaCost(0, tuple(sorted(colored.items())))

    def minus_colored(self, other: "ManaCost") -> "ManaCost":
        """This cost without `other`'s coloured symbols (each must be present)."""
        c = self.colored_dict()
        for k, n in other.colored:
            assert c.get(k, 0) >= n, (self, other)
            c[k] -= n
        return ManaCost(self.generic, tuple(sorted((k, n) for k, n in c.items() if n)), self.x)

    def colored_dict(self) -> dict[str, int]:
        return dict(self.colored)

    @property
    def mana_value(self) -> int:
        return self.generic + sum(n for _, n in self.colored)

    def with_x(self, x_value: int) -> "ManaCost":
        """Fix X: the result has no X symbols and generic increased by X*count."""
        return ManaCost(self.generic + x_value * self.x, self.colored, 0)

    def reduced(self, amount: int) -> "ManaCost":
        """Cost reduction (affinity etc.) only ever reduces the generic part."""
        return ManaCost(max(0, self.generic - amount), self.colored, self.x)

    def plus(self, other: "ManaCost") -> "ManaCost":
        c = self.colored_dict()
        for k, n in other.colored:
            c[k] = c.get(k, 0) + n
        return ManaCost(self.generic + other.generic, tuple(sorted(c.items())), self.x + other.x)

    def is_zero(self) -> bool:
        return self.generic == 0 and not self.colored and self.x == 0

    def __str__(self) -> str:
        parts = ["{X}"] * self.x
        if self.generic or (not self.colored and not self.x):
            parts.append("{%d}" % self.generic)
        order = {c: i for i, c in enumerate(MANA_TYPES)}
        for k, n in sorted(self.colored, key=lambda kv: order[kv[0]]):
            parts.extend(["{%s}" % k] * n)
        return "".join(parts)


@dataclass
class RemainingCost:
    """Mutable remainder of a cost while it is being paid."""

    generic: int
    colored: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def of(cost: ManaCost) -> "RemainingCost":
        assert cost.x == 0, "X must be fixed before payment"
        return RemainingCost(cost.generic, cost.colored_dict())

    def copy(self) -> "RemainingCost":
        return RemainingCost(self.generic, dict(self.colored))

    def is_paid(self) -> bool:
        return self.generic == 0 and not any(self.colored.values())

    def apply(self, mana: str) -> bool:
        """Spend one unit of `mana`. Colored requirements are preferred, which
        is weakly dominant (exchange argument). Returns False if the unit
        cannot be used at all."""
        if self.colored.get(mana, 0) > 0:
            self.colored[mana] -= 1
            if self.colored[mana] == 0:
                del self.colored[mana]
            return True
        if self.generic > 0:
            self.generic -= 1
            return True
        return False

    def useful(self, mana: str) -> bool:
        return self.colored.get(mana, 0) > 0 or self.generic > 0

    def __str__(self) -> str:
        return str(ManaCost(self.generic, tuple(sorted(self.colored.items()))))


def can_pay(remaining: RemainingCost, units: list[tuple[str, ...]]) -> bool:
    """Can `remaining` be paid using `units`?

    `units` is a list of alternatives, one entry per available mana unit: a
    floating unit in the pool is ('U',), a dual land is ('B', 'R').
    Feasible iff every colored symbol can be matched to a distinct unit able to
    produce it and the total number of units covers all symbols.
    """
    symbols = [c for c, n in remaining.colored.items() for _ in range(n)]
    if len(units) < len(symbols) + remaining.generic:
        return False
    if not symbols:
        return True
    # Kuhn's augmenting-path bipartite matching (symbols -> units).
    match_unit: list[int] = [-1] * len(units)

    def augment(si: int, seen: list[bool]) -> bool:
        for ui, alts in enumerate(units):
            if symbols[si] in alts and not seen[ui]:
                seen[ui] = True
                if match_unit[ui] == -1 or augment(match_unit[ui], seen):
                    match_unit[ui] = si
                    return True
        return False

    for si in range(len(symbols)):
        if not augment(si, [False] * len(units)):
            return False
    return True


def pool_units(pool: dict[str, int]) -> list[tuple[str, ...]]:
    return [(c,) for c, n in pool.items() for _ in range(n)]
