"""cards.toml against the oracle snapshot (data/oracle_cards.json): every
card in the pool has an oracle entry with the same cost, types and
power/toughness, so a typo in the spec cannot slip in unnoticed."""

import json
import os
import unicodedata

import pytest

from mtg_ml.engine.cards import CARDS



def engine_name(oracle_name: str) -> str:
    """The engine's name for an oracle entry: front face, accents folded
    ("Delver of Secrets // Insectile Aberration", "L\u00f3rien Revealed")."""
    front = oracle_name.split(" // ")[0]
    return "".join(ch for ch in unicodedata.normalize("NFKD", front) if not unicodedata.combining(ch))


with open(os.path.join(os.path.dirname(__file__), "..", "data", "oracle_cards.json"), encoding="utf-8") as f:
    ORACLE = {engine_name(c["name"]): c for c in json.load(f)}


def check_face(d, o) -> None:
    assert d.cost == type(d.cost).parse(o["mana_cost"] or ""), (d.name, str(d.cost), o["mana_cost"])
    main, _, subs = o["type_line"].partition(" \u2014 ")
    assert set(main.split()) == set(d.types) | set(d.supertypes), (d.name, main)
    assert set(subs.split()) == set(d.subtypes), (d.name, subs)
    if o["power"] is not None:
        assert (d.power, d.toughness) == (int(o["power"]), int(o["toughness"])), d.name


@pytest.mark.parametrize("name", sorted(CARDS))
def test_card_matches_oracle(name):
    d = CARDS[name]
    o = ORACLE.get(name)
    assert o is not None, f"{name}: add its oracle entry to data/oracle_cards.json"
    if o.get("faces"):  # double-faced: front face, then the back face
        check_face(d, o["faces"][0])
        assert d.back is not None and d.back.name == o["faces"][1]["name"], name
        check_face(d.back, o["faces"][1])
    else:
        check_face(d, o)


def test_oracle_snapshot_has_no_unknown_cards():
    assert set(ORACLE) <= set(CARDS), set(ORACLE) - set(CARDS)


def test_an_unseen_card_gets_the_shape_of_the_cards_it_plays_like():
    """Set 5: shape tokens come from the spec, not the name, so a card that
    is not in the pool (here a Chain Lightning-like burn spell) has every
    shape token in common with Lightning Bolt."""
    from mtg_ml.engine.cards import card_def

    burn = card_def({"name": "Unseen Burn", "cost": "{R}", "types": "Sorcery", "targets": ["any"], "effect": [{"op": "damage_target", "n": 3}]})
    assert burn.shape == CARDS["Lightning Bolt"].shape
    trick = card_def({"name": "Unseen Trick", "cost": "{1}{B}", "types": "Instant", "targets": ["creature"], "effect": [{"op": "grant_target", "keywords": ["lifelink"]}]})
    assert set(CARDS["Toxin Analysis"].shape) - set(trick.shape) == {"e:spell:op:create_token", "e:spell:op:create_token:n>=1", "e:op:create_token", "e:op:create_token:n>=1"}
    assert set(trick.shape) - set(CARDS["Toxin Analysis"].shape) == {"e:mv>=2"}
