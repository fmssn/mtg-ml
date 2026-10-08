"""Detached, deeply immutable inputs for scripted benchmark agents."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..engine.view import observe
from .artifacts import require


@dataclass(frozen=True)
class FrozenMap(Mapping):
    pairs: tuple

    def __getitem__(self, key):
        for k, v in self.pairs:
            if k == key:
                return v
        raise KeyError(key)

    def __iter__(self):
        return (k for k, _ in self.pairs)

    def __len__(self):
        return len(self.pairs)


def freeze(value):
    if isinstance(value, Mapping):
        return FrozenMap(tuple((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    require(value is None or type(value) in (str, int, float, bool), "agent input", "non-data value")
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class PlayerView:
    state: FrozenMap
    own_deck: FrozenMap
    cards: FrozenMap
    context: FrozenMap


@dataclass(frozen=True)
class LegalAction:
    index: int
    kind: str
    key: tuple
    label: str
    data: FrozenMap


@dataclass(frozen=True)
class VisibleEvent:
    actor: str
    kind: str
    action: LegalAction | None
    lines: tuple[str, ...]
    state: FrozenMap


class BenchmarkAgent(Protocol):
    def reset(self, own_deck: Mapping[str, int]) -> None: ...
    def observe(self, visible_event: VisibleEvent) -> None: ...
    def choose(self, player_view: PlayerView, legal_actions: Sequence[LegalAction]) -> int: ...


KINDS = {"priority", "choose_x", "target", "pay_mana", "sacrifice", "exile_from_graveyard", "yes_no", "choose_card",
         "order", "order_triggers", "declare_attacker", "declare_blocker", "assign_damage", "assign_damage_amount", "choose_mode", "mulligan"}


def card_ref(g, c, viewer, choice=False):
    if c is None:
        return None
    known = choice or c.zone in {"battlefield", "graveyard", "exile"} or (c.zone == "hand" and (c.owner == viewer or viewer in c.known_to)) or viewer in c.known_to
    require(known, "action.card", "hidden card reference")
    # Unknown library order is never attached to an object reference.
    return {"name": c.name, "zone": c.zone, "oid": c.oid if c.zone == "battlefield" else None,
            "uid": c.uid if c.zone != "library" else None, "owner": "self" if c.owner == viewer else "opponent"}


def mana_cost(cost):
    return {"generic": cost.generic, "colored": cost.colored_dict(), "x": cost.x}


def _rules(face, token=False):
    return {"types": sorted(face.types), "subtypes": sorted(face.subtypes), "cost": str(face.cost), "text": face.text,
            "mana_cost": mana_cost(face.cost), "colors": sorted(face.colors), "supertypes": sorted(face.supertypes),
            "ward": face.ward, "enters_tapped": face.enters_tapped, "targets": [t.kind for t in face.targets],
            "additional_sacrifice": face.additional_sac, "additional_discard": face.additional_discard,
            "escape_exile": face.escape_exile, "shape": face.shape,
            "power": face.power, "toughness": face.toughness, "token": token,
            "bestow": None if face.bestow is None else str(face.bestow),
            "flashback": None if face.flashback is None else str(face.flashback),
            "keywords": sorted(face.keywords),
            "abilities": [{"name": a.name, "cost": str(a.cost), "mana_cost": mana_cost(a.cost), "tap": a.tap, "sac_self": a.sac_self, "sac_other": a.sac_other,
                           "mana": a.mana, "mana_amount": a.mana_amount, "zone": a.zone, "sorcery_speed": a.sorcery_speed,
                           "return_land": a.return_land, "once_per_turn": a.once_per_turn,
                           "discard_self": a.discard_self, "discard_other": a.discard_other, "exile_self": a.exile_self,
                           "targets": [t.kind for t in a.targets]} for a in face.abilities],
            "modes": [{"name": m.name, "targets": [t.kind for t in m.targets]} for m in face.modes]}


def stack_view(g, viewer):
    """Public stack facts, including live or last-known source characteristics.

    Only explicitly public scalar trigger data crosses this boundary. In
    particular, never serialize the engine's arbitrary StackItem.data mapping.
    """
    def ref(r):
        tag, value = r
        return (tag, ("self" if value == viewer else "opponent") if tag == "player" else value)

    out = []
    for it in g.stack:
        source = it.card if it.card is not None else it.source
        live = None if source is None else g.perm(source.oid)
        c = live if live is not None else source
        src = None if c is None else {
            "name": c.name, "oid": c.oid, "controller": "self" if c.controller == viewer else "opponent",
            "keywords": sorted(g.keywords(c)), "power": g.power(c), "toughness": g.toughness(c),
        }
        out.append({"sid": it.sid, "kind": it.kind, "controller": "self" if it.controller == viewer else "opponent",
                    "source": src, "targets": [ref(r) for r in it.targets], "x": it.x, "method": it.method,
                    "data": {k: it.data[k] for k in ("sid", "spell_sid", "source_oid", "amount") if k in it.data}})
    return out


def inputs(g, viewer, own_deck):
    """Trusted translation; no raw Option.value or rule callbacks survive."""
    d = g.decision
    require(d is not None and d.player == viewer, "decision", "view requested for nondecider")
    require(d.kind in KINDS, "decision.kind", f"unsupported semantics {d.kind}")
    state = observe(g, viewer)
    context = {"stack": stack_view(g, viewer), "blocked": sorted(g.blocked)}
    combat = g.combat_subjects if d.kind in {"declare_attacker", "declare_blocker", "assign_damage"} else None
    if combat is not None:
        require(len(combat) == len(d.options), "decision.subject", "missing structured combat subjects")
    subject = g.perm(combat[0][0]) if d.kind in {"declare_blocker", "assign_damage"} else None
    if subject is not None:
        context["subject"] = card_ref(g, subject, viewer)
    if getattr(g, "NATIVE", False):
        payment = g.payment_context
    else:
        payment = None if g.paying is None else {"generic": g.paying[0].generic, "colored": dict(g.paying[0].colored)}
    if payment is not None:
        context["payment"] = payment
    allocation = getattr(g, "damage_allocation", None)
    if allocation is not None:
        context["damage"] = {k: getattr(allocation, k) for k in ("attacker", "blockers", "lethal", "assigned", "recipient", "remaining", "player_damage")}
        context["damage"]["defender"] = "self" if allocation.defender == viewer else "opponent"
    actions = []
    permitted = {c.name: c for c in g.battlefield}
    permitted.update({c.name: c for p in g.players for z in ("graveyard", "exile", "hand", "library") for c in getattr(p, z)
                      if z in {"graveyard", "exile"} or (z == "hand" and p.idx == viewer) or viewer in c.known_to})
    for it in g.stack:
        c = it.card if it.card is not None else it.source
        if c is not None:
            permitted.setdefault(c.name, c)  # spells and ability sources are public
    for i, o in enumerate(g.legal_options()):
        key, v = o.key, o.value
        data = {"choice": key}
        if d.kind == "priority":
            if key[0] != "pass":
                require(isinstance(v, tuple) and v[0] in {"cast", "land", "activate", "plot", "mana"}, "priority", "unsupported action payload")
                c = v[1]
                data["card"] = card_ref(g, c, viewer)
                permitted[c.name] = c
                if v[0] == "cast":
                    mode = key[3]
                    data["mode"] = mode
                    cost = g._mode_cost(c, mode).reduced(g._cost_reduction(viewer, c))
                    data["cost"], data["mana_cost"] = str(cost.with_x(0)), mana_cost(cost)
                    if len(key) > 4:
                        data["spell_mode"] = key[4]
                elif v[0] in {"activate", "mana"}:
                    data["ability"] = key[2]
                    data["ability_index"] = v[2]
                    if v[0] == "activate":
                        cost = c.face.abilities[v[2]].cost
                        data["cost"], data["mana_cost"] = str(cost), mana_cost(cost)
                    else:
                        data["color"] = c.face.abilities[v[2]].mana[0]
        elif d.kind == "target":
            if v is None:
                data["target"] = None
            else:
                tag, n = v
                require(tag in {"player", "perm", "stack"}, "target", "unsupported target")
                if tag == "player":
                    data["target"] = {"player": "self" if n == viewer else "opponent"}
                elif tag == "perm":
                    data["target"] = card_ref(g, g.perm(n), viewer)
                else:
                    it = g.stack_item(n)
                    require(it is not None, "target", "missing stack item")
                    data["target"] = {"sid": n, "name": it.name}
        elif d.kind == "pay_mana":
            data.update(via=v[0], color=v[-1])
            if v[0] != "pool":
                data["source"] = card_ref(g, v[1], viewer)
        elif d.kind == "declare_attacker":
            if v is None:
                data["attacker"] = None
            else:
                c = g.perm(combat[i][0])
                data["attacker"] = card_ref(g, c, viewer)
        elif d.kind == "declare_blocker":
            data.update(blocker=card_ref(g, subject, viewer), attacker=card_ref(g, v, viewer))
        elif d.kind == "assign_damage":
            blockers = [g.perm(b) for b in combat[i][1:]]
            data.update(source=card_ref(g, subject, viewer), blockers=[card_ref(g, b, viewer) for b in blockers], split=list(v))
        elif d.kind in {"choose_x", "assign_damage_amount"}:
            data["amount"] = v
        elif d.kind == "order_triggers":
            # Values are pending trigger objects; only public source/key data.
            data["trigger"] = key[1:]
        elif d.kind in {"sacrifice", "exile_from_graveyard", "choose_card"}:
            c = v
            if isinstance(v, tuple):
                require(len(v) >= 2 and hasattr(v[1], "name"), d.kind, "unsupported card choice")
                c = v[1]
                data["via"] = v[0]
                if len(v) > 2:
                    data["color"] = v[-1]
            if c is not None:
                data["card"] = card_ref(g, c, viewer, choice=True)
                permitted[c.name] = c
            else:
                data["card"] = None
        elif d.kind == "order":
            if key[0] == "order":
                data.update(top=key[1:], bottom=())
            elif key[0] == "scry":
                j = key.index("bottom")
                data.update(top=key[2:j], bottom=key[j + 1:])
            else:
                raise ValueError(f"order: unsupported semantics {key[0]}")
        # yes/no and mode choices have fully semantic keys. Cards inspected
        # for Delver/scry appear only in the permitted known-library view.
        actions.append(LegalAction(i, d.kind, freeze(key), o.label, freeze(data)))
    cards = {name: _rules(c.face, c.is_token) for name, c in permitted.items()}
    # Own registered cards are known rules, without conveying remaining order.
    from ..engine.cards import CARDS
    for name in own_deck:
        require(name in CARDS, "own_deck", f"unknown card {name}")
        cards.setdefault(name, _rules(CARDS[name]))
    return PlayerView(freeze(state), freeze(own_deck), freeze(cards), freeze(context)), tuple(actions)
