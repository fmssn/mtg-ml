"""Version-one benchmark configuration contracts; no mutable deck defaults."""

from dataclasses import dataclass
from pathlib import Path
import re

from ..engine.cards import CARDS
from ..engine.game import STEPS
from ..expert.artifacts import validate_review
from .jsonio import digest, file_digest, read_json

FAIR = "own-list-hidden-opponent-v1"
DIAGNOSTIC = "privileged-diagnostic"
MODES = {"sampled", "greedy"}
CATEGORIES = {"mana-sequencing", "combat", "stack-interaction", "resource-preservation", "deck-synergy"}
DECKS = {"jund_wildfire", "mono_blue_terror"}
CELLS = {"jund_vs_jund": ("jund_wildfire", "jund_wildfire"), "jund_vs_blue": ("jund_wildfire", "mono_blue_terror"),
         "blue_vs_jund": ("mono_blue_terror", "jund_wildfire"), "blue_vs_blue": ("mono_blue_terror", "mono_blue_terror")}


def require(ok, field, message):
    if not ok:
        raise ValueError(f"{field}: {message}")


def integer(value, field, minimum=1, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum), field, "invalid integer")


def identifier(value, field):
    require(isinstance(value, str) and bool(value.strip()), field, "nonempty identifier required")


def sha(value, field, revision=False):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}" if revision else r"[0-9a-f]{64}", value), field, "invalid digest/revision")


def header(data, name):
    require(isinstance(data, dict), name, "object required")
    require(data.get("format") == name and type(data.get("version")) is int and data["version"] == 1, name, "unsupported format/version")


def entries(values, field):
    require(isinstance(values, list) and bool(values), field, "nonempty list required")
    ids = []
    for v in values:
        require(isinstance(v, dict), field, "object entries required")
        identifier(v.get("id"), field + ".id")
        ids.append(v["id"])
    require(len(set(ids)) == len(ids), field, "duplicate identity")


@dataclass(frozen=True)
class Artifact:
    path: Path
    data: dict
    sha256: str


def load(path, validator) -> Artifact:
    path = Path(path).resolve()
    try:
        raw = path.read_bytes()
        data = read_json(path)
        validator(data)
        import hashlib
        return Artifact(path, data, hashlib.sha256(raw).hexdigest())
    except (ValueError, KeyError, TypeError, OSError) as e:
        raise ValueError(f"{path}: {e}") from e


def reference(parent, ref, field) -> Path:
    require(isinstance(ref, dict), field, "file reference required")
    identifier(ref.get("path"), field + ".path")
    require(not Path(ref["path"]).is_absolute(), field, "reference must be relative")
    sha(ref.get("sha256"), field + ".sha256")
    path = Path(parent).parent / ref["path"]
    require(path.is_file(), field, f"missing file {path}")
    require(file_digest(path) == ref["sha256"], field, f"file hash mismatch: {path}")
    return path.resolve()


def validate_manifest(d):
    header(d, "BenchmarkManifest")
    identifier(d.get("id"), "id")
    require(isinstance(d.get("suite_version"), str) and re.fullmatch(r"\d+\.\d+\.\d+", d["suite_version"]), "suite_version", "semantic version required")
    require(d.get("split") in {"dev", "final", "power-pilot"}, "split", "unsupported split")
    require(d.get("stream") == "benchmark-v1/" + d["split"], "stream", "split/stream mismatch")
    require(d.get("information_contract") in {FAIR, DIAGNOSTIC}, "information_contract", "unsupported contract")
    freeze = d["freeze"]
    sha(freeze.get("code_revision"), "freeze.code_revision", True)
    for k in ("card_spec_sha256", "deck_bundle_sha256"):
        sha(freeze.get(k), "freeze." + k)
    require(isinstance(d.get("decks"), dict) and set(d["decks"]) == DECKS, "decks", "both frozen deck maps required")
    for name, deck in d["decks"].items():
        cards = deck["cards"]
        require(isinstance(cards, dict) and bool(cards), "decks." + name, "card-count map required")
        for card, n in cards.items():
            require(card in CARDS, "decks." + name, f"unsupported card {card}")
            integer(n, "decks." + name + "." + card)
        require(sum(cards.values()) == 60, "decks." + name, "60 cards required")
        require(deck.get("sha256") == digest(cards), "decks." + name, "card map hash mismatch")
    require(freeze["deck_bundle_sha256"] == digest(d["decks"]), "freeze.deck_bundle_sha256", "deck bundle hash mismatch")
    entries(d["bots"], "bots")
    bots = {b["id"]: b for b in d["bots"]}
    for b in bots.values():
        require(b.get("deck") in DECKS, "bots.deck", "unknown deck")
        require(re.fullmatch(r"[^@]+@[1-9]\d*", b["id"]), "bots.id", "versioned id required")
        sha(b.get("source_revision"), "bots.source_revision", True)
        sha(b.get("parameters_sha256"), "bots.parameters_sha256")
        integer(b.get("rules_revision"), "bots.rules_revision")
    entries(d["cells"], "cells")
    require({c["id"] for c in d["cells"]} == set(CELLS), "cells", "exact four cells required")
    for c in d["cells"]:
        require(c.get("opponent") in bots, "cells.opponent", "unknown bot")
        require((c.get("learner_deck"), bots[c["opponent"]]["deck"]) == CELLS[c["id"]], "cells", "cell/deck mapping mismatch")
    require(isinstance(d.get("modes"), list) and bool(d["modes"]) and set(d["modes"]) <= MODES and len(set(d["modes"])) == len(d["modes"]), "modes", "unsupported/duplicate modes")
    integer(d.get("blocks_per_cell"), "blocks_per_cell")
    integer(d.get("puzzle_repetitions"), "puzzle_repetitions")
    settings = d["engine_settings"]
    require(set(settings) == {"max_turns", "max_decisions", "auto_single", "auto_mana", "auto_pass"}, "engine_settings", "explicit settings required")
    integer(settings["max_turns"], "engine_settings.max_turns")
    integer(settings["max_decisions"], "engine_settings.max_decisions")
    for k in ("auto_single", "auto_mana", "auto_pass"):
        require(type(settings[k]) is bool, "engine_settings." + k, "boolean required")
    integer(d["bootstrap"]["replicates"], "bootstrap.replicates")
    require(type(d["bootstrap"]["confidence"]) in (float, int) and 0 < d["bootstrap"]["confidence"] < 1, "bootstrap.confidence", "invalid confidence")
    require(isinstance(d.get("puzzles"), dict), "puzzles", "bundle reference required")


def validate_puzzle(d):
    header(d, "TacticalPuzzle")
    for k in ("id", "template_id", "source_group_id"):
        identifier(d.get(k), k)
    require(d.get("deck") in DECKS, "deck", "unknown deck")
    require(d.get("category") in CATEGORIES, "category", "unknown category")
    require(d.get("split") in {"dev", "final"}, "split", "unknown split")
    validate_review(d["review"])
    require(d["review"]["status"] == "accepted", "review", "accepted review required")
    require(d.get("episode_mode") == "reset", "episode_mode", "only reset supported")
    integer(d.get("max_decisions"), "max_decisions", maximum=64)
    entries(d["cases"], "cases")
    for c in d["cases"]:
        identifier(c.get("hidden_completion"), "cases.hidden_completion")
        identifier(c["response_policy"].get("id"), "cases.response_policy.id")
        sha(c["response_policy"].get("parameters_sha256"), "cases.response_policy.parameters_sha256")
        require(isinstance(c.get("scenario"), dict) and isinstance(c.get("evidence"), dict), "cases", "scenario/evidence references required")
        if "witnesses" in c:
            require(isinstance(c["witnesses"], list) and bool(c["witnesses"]), "cases.witnesses", "nonempty witnesses required")
            for witness in c["witnesses"]:
                require(isinstance(witness, dict) and isinstance(witness.get("actions"), list) and bool(witness["actions"]), "cases.witnesses", "action selectors required")
                identifier(witness.get("note"), "cases.witnesses.note")
    clauses = d["objective"].get("all")
    require(isinstance(clauses, list) and bool(clauses), "objective.all", "nonempty conjunction required")
    for c in clauses:
        if "object" in c:
            identifier(c["object"], "objective.object")
            require(c.get("zone") in {"battlefield", "graveyard", "exile"} and type(c.get("present")) is bool, "objective", "public zone and presence required")
        else:
            require(c.get("op") in {"eq", "gte", "lte", "contains"} and "value" in c, "objective", "unsupported comparison")
            identifier(c.get("path"), "objective.path")
            bits = c["path"].split(".")
            require(bits[0] in {"self", "opponent", "battlefield", "stack", "turn", "step", "active", "over", "winner"}, "objective.path", "public observation path required")
            require(not any(x in {"hand", "hand_known", "library", "library_known", "decision"} for x in bits), "objective.path", "private fact cannot be scored")
    stop = d["stop"]
    require(stop.get("kind") in {"terminal", "decision_boundary"}, "stop.kind", "unsupported boundary")
    if stop["kind"] == "decision_boundary":
        integer(stop.get("turn"), "stop.turn")
        require(stop.get("step") in STEPS and stop.get("active") in {"self", "opponent"} and stop.get("require_empty_stack") is True, "stop", "explicit resolved boundary required")
    require(d.get("claim") == "success-against-declared-responses", "claim", "foundation cannot certify forced solutions")
    if "acceptable_first_actions" in d:
        require(isinstance(d["acceptable_first_actions"], list) and bool(d["acceptable_first_actions"]), "acceptable_first_actions", "nonempty selectors required")


def validate_bundle(d):
    header(d, "BenchmarkPuzzleBundle")
    require(isinstance(d.get("puzzles"), list), "puzzles", "reference list required")


def load_manifest(path):
    return load(path, validate_manifest)


def load_puzzle(path):
    return load(path, validate_puzzle)
