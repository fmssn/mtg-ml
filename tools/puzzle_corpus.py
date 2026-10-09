#!/usr/bin/env python3
"""Source of the development tactical puzzles under benchmark/puzzles/dev.

.venv/bin/python tools/puzzle_corpus.py            # rewrite the JSON artifacts
.venv/bin/python tools/puzzle_corpus.py --check    # fail if they are stale

Every position is synthetic and built by setup cards plus legal setup actions.
Reviews live in benchmark/puzzles/dev/reviews.json, keyed by puzzle id and the
digest of the reviewed content; editing a puzzle invalidates its review, and an
unreviewed puzzle is written as `pending` (which validation rejects).
--draft-review marks unreviewed puzzles accepted by "author-draft" for local
iteration only; never commit its output.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from mtg_ml.benchmark.blue import PARAMETERS as BLUE_PARAMETERS
from mtg_ml.benchmark.bots.parameters import PARAMETERS_SHA256 as JUND_PARAMETERS_SHA256
from mtg_ml.benchmark.jsonio import canonical_bytes, digest, read_json
from mtg_ml.benchmark.views import thaw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "puzzles" / "dev"
JUND, BLUE = "jund_wildfire", "mono_blue_terror"
RESPONSE = {JUND: {"id": "benchmark-blue@1", "parameters_sha256": digest(thaw(BLUE_PARAMETERS))},
            BLUE: {"id": "benchmark-jund@1", "parameters_sha256": JUND_PARAMETERS_SHA256}}


def player(life=20, hand=(), library=(), graveyard=(), battlefield=()):
    return dict(life=life, drawn=0, hand=list(hand), hand_count=len(hand), library=list(library), library_count=len(library),
                graveyard=list(graveyard), exile=[], battlefield=list(battlefield))


def card(name, oid=None, **flags):
    return dict(name=name, **({"id": oid} if oid else {}), **flags)


def act(seat, kind, *key, objects=None):
    s = {"player": seat, "kind": kind, "key": list(key)}
    if objects:
        s["objects"] = list(objects)
    return s


def passes(*seats):
    return [act(s, "priority", "pass") for s in seats]


def present(oid, zone, flag=True):
    return {"object": oid, "zone": zone, "present": flag}


def boundary(turn, active):
    return {"kind": "decision_boundary", "turn": turn, "step": "end", "active": active, "require_empty_stack": True}


PAY_B = ("pay", "source", "Swamp", "B")
# Jund libraries stay within the decklist basics (3 Swamp, 2 Forest, 1 Mountain).
JUND_LIBRARY = ["Swamp", "Forest", "Forest", "Mountain"]
PAY_U = ("pay", "source", "Island", "U")
SHAMAN = ("activate", "Krark-Clan Shaman", "1 damage to each creature without flying")

# Blue is seat 0 and Jund seat 1 in every Blue-learner puzzle, and vice versa.
SLEEP_YARD = [card("Brainstorm", "yard-brainstorm"), card("Ponder", "yard-ponder"),
              card("Thought Scour", "yard-scour"), card("Mental Note", "yard-note")]
SLEEP_SETUP = [act(0, "priority", "cast", "Sleep of the Dead", "hand", "normal"),
               act(0, "target", "target", "creature", "perm", "opponent", "Krark-Clan Shaman", objects=["shaman"]),
               act(0, "pay_mana", *PAY_U)] + passes(0, 1)
ESCAPE = act(0, "priority", "cast", "Sleep of the Dead", "graveyard", "escape")


def escape_onto(oid, name):
    # Escape exiles three other graveyard cards; the choice order is incidental.
    return [ESCAPE, act(0, "target", "target", "creature", "perm", "opponent", name, objects=[oid])] + \
        [act(0, "pay_mana", *PAY_U)] * 3 + [act(0, "exile_from_graveyard", "exile_gy", c) for c in ("Brainstorm", "Ponder", "Thought Scour")]


PUZZLES = [
    dict(id="blue-sleep-redundant", deck=BLUE, category="resource-preservation", template="sleep-redundant-lock",
         group="playtest-2026-10-09-flag4",
         source="r7 hosted playtest, docs/playtest-reviews/2026-10-09.md flag 4 (replay frames 424-437); position re-authored synthetically.",
         rationale="The only opposing creature is already tapped and locked by this turn's Sleep of the Dead. Escaping Sleep again "
                   "cannot extend the lock (the effect uses the larger skip count, it does not add), so it wastes three mana and "
                   "three graveyard cards that fuel Tolarian Terror and later escapes. From the r7 playtest (2026-10-09, flag 4).",
         initial=dict(active=0, step="main1", seed=3, match_game=1, turn=9, lands_played=0, spells_cast_this_turn=0, players=[
             player(hand=["Sleep of the Dead", "Island"], library=["Island"] * 4, graveyard=SLEEP_YARD,
                    battlefield=[card("Island")] * 4),
             player(library=JUND_LIBRARY, battlefield=[card("Krark-Clan Shaman", "shaman", sick=False), card("Swamp"), card("Swamp")])]),
         setup=SLEEP_SETUP,
         objective=[present(x["id"], "graveyard") for x in SLEEP_YARD], stop=boundary(9, "self"),
         witnesses=[([act(0, "priority", "pass")], "Pass: the lock is already in place."),
                    ([act(0, "priority", "play_land", "Island")], "Play the land and keep the graveyard.")],
         mistakes=[(escape_onto("shaman", "Krark-Clan Shaman"), "Redundant escape onto the locked Shaman.", "objective_failed")],
         first=[act(0, "priority", "pass"), act(0, "priority", "play_land", "Island")]),
    dict(id="blue-sleep-productive", deck=BLUE, category="combat", template="sleep-second-target",
         group="playtest-2026-10-09-flag4",
         source="Variant of docs/playtest-reviews/2026-10-09.md flag 4 with a second, untapped creature.",
         rationale="The Shaman is already locked, but Gixian Infiltrator is untapped and will attack an empty board. Escaping "
                   "Sleep onto the Infiltrator prevents the damage; escaping onto the locked Shaman, or passing, takes it.",
         initial=dict(active=0, step="main1", seed=3, match_game=1, turn=9, lands_played=1, spells_cast_this_turn=0, players=[
             player(hand=["Sleep of the Dead"], library=["Island"] * 4, graveyard=SLEEP_YARD, battlefield=[card("Island")] * 4),
             player(library=JUND_LIBRARY, battlefield=[card("Krark-Clan Shaman", "shaman", sick=False),
                                                        card("Gixian Infiltrator", "infiltrator", sick=False), card("Swamp"), card("Swamp")])]),
         setup=SLEEP_SETUP,
         objective=[{"path": "self.life", "op": "eq", "value": 20}], stop=boundary(10, "opponent"),
         witnesses=[(escape_onto("infiltrator", "Gixian Infiltrator"), "Escape Sleep onto the untapped attacker.")],
         mistakes=[(escape_onto("shaman", "Krark-Clan Shaman"), "Escape onto the already locked Shaman.", "objective_failed"),
                   ([act(0, "priority", "pass")], "Pass and take the attack.", "objective_failed")],
         first=[ESCAPE]),
    dict(id="blue-counter-untapped", deck=BLUE, category="stack-interaction", template="counter-choice",
         rationale="Jund casts Gixian Infiltrator with one Swamp still open, so Force Spike will be paid. Only Counterspell "
                   "stops the creature.",
         initial=dict(active=1, step="main1", seed=3, match_game=1, turn=8, lands_played=1, spells_cast_this_turn=0, players=[
             player(hand=["Counterspell", "Force Spike"], library=["Island"] * 4, battlefield=[card("Island")] * 2),
             player(hand=[card("Gixian Infiltrator", "infiltrator")], library=["Forest", "Forest", "Mountain"], battlefield=[card("Swamp")] * 3)]),
         setup=[act(1, "priority", "cast", "Gixian Infiltrator", "hand", "normal"), act(1, "pay_mana", *PAY_B),
                act(1, "pay_mana", *PAY_B)] + passes(1),
         objective=[present("infiltrator", "graveyard")], stop=boundary(8, "opponent"),
         witnesses=[([act(0, "priority", "cast", "Counterspell", "hand", "normal"),
                      act(0, "target", "target", "spell", "spell", "opponent", "Gixian Infiltrator"), act(0, "pay_mana", *PAY_U),
                      act(0, "pay_mana", *PAY_U)], "Counterspell the creature.")],
         mistakes=[([act(0, "priority", "cast", "Force Spike", "hand", "normal"),
                     act(0, "target", "target", "spell", "spell", "opponent", "Gixian Infiltrator"), act(0, "pay_mana", *PAY_U)],
                    "Force Spike into an open Swamp.", "objective_failed")],
         first=[act(0, "priority", "cast", "Counterspell", "hand", "normal")]),
    dict(id="blue-counter-tapped-out", deck=BLUE, category="stack-interaction", template="counter-choice",
         rationale="Same choice with Jund tapped out: Force Spike is a hard counter here, so spending Counterspell wastes the "
                   "stronger card. The objective requires the creature countered without Counterspell reaching the graveyard (a public fact).",
         initial=dict(active=1, step="main1", seed=3, match_game=1, turn=8, lands_played=1, spells_cast_this_turn=0, players=[
             player(hand=[card("Counterspell", "counterspell"), "Force Spike"], library=["Island"] * 4, battlefield=[card("Island")] * 2),
             player(hand=[card("Gixian Infiltrator", "infiltrator")], library=JUND_LIBRARY, battlefield=[card("Swamp")] * 2)]),
         setup=[act(1, "priority", "cast", "Gixian Infiltrator", "hand", "normal"), act(1, "pay_mana", *PAY_B),
                act(1, "pay_mana", *PAY_B)] + passes(1),
         objective=[present("infiltrator", "graveyard"), present("counterspell", "graveyard", False)],
         stop=boundary(8, "opponent"),
         witnesses=[([act(0, "priority", "cast", "Force Spike", "hand", "normal"),
                      act(0, "target", "target", "spell", "spell", "opponent", "Gixian Infiltrator"), act(0, "pay_mana", *PAY_U)],
                     "Force Spike: the opponent cannot pay.")],
         mistakes=[([act(0, "priority", "pass")], "Let the creature resolve.", "objective_failed"),
                   ([act(0, "priority", "cast", "Counterspell", "hand", "normal"),
                     act(0, "target", "target", "spell", "spell", "opponent", "Gixian Infiltrator"), act(0, "pay_mana", *PAY_U),
                     act(0, "pay_mana", *PAY_U)], "Spend Counterspell where Force Spike is a hard counter.", "objective_failed")],
         first=[act(0, "priority", "cast", "Force Spike", "hand", "normal")]),
    dict(id="jund-shaman-toxin-sweep", deck=JUND, category="deck-synergy", template="shaman-deathtouch-sweep",
         rationale="Toxin Analysis on Krark-Clan Shaman gives it deathtouch and makes a Clue; sacrificing the Clue to the Shaman "
                   "then deals deathtouch damage to every non-flying creature, killing both Serpents and Delver. Toxin on an "
                   "opposing creature only kills Delver (on Terror it is countered by ward). The Shaman dies to its own sweep. "
                   "docs/representation-plan.md lists this sweep as a card-level interaction the model has missed.",
         initial=dict(active=0, step="main1", seed=5, match_game=1, turn=8, lands_played=1, spells_cast_this_turn=0, players=[
             player(hand=["Toxin Analysis"], library=["Swamp", "Swamp", "Forest", "Forest"],
                    battlefield=[card("Krark-Clan Shaman", "shaman", sick=False), card("Swamp")]),
             player(library=["Island"] * 4, battlefield=[card("Tolarian Terror", "terror", sick=False), card("Cryptic Serpent", "serpent", sick=False),
                                                       card("Delver of Secrets", "delver", sick=False)])]),
         setup=[],
         objective=[present("terror", "battlefield", False), present("serpent", "battlefield", False), present("delver", "battlefield", False)],
         stop=boundary(8, "self"),
         witnesses=[([act(0, "priority", "cast", "Toxin Analysis", "hand", "normal"),
                      act(0, "target", "target", "creature", "perm", "self", "Krark-Clan Shaman"), act(0, "pay_mana", *PAY_B),
                      *passes(0), act(0, "priority", *SHAMAN), act(0, "sacrifice", "sacrifice", "Clue")],
                     "Toxin on the Shaman, then sacrifice the Clue.")],
         mistakes=[([act(0, "priority", "cast", "Toxin Analysis", "hand", "normal"),
                     act(0, "target", "target", "creature", "perm", "opponent", "Cryptic Serpent"), act(0, "pay_mana", *PAY_B),
                     *passes(0), act(0, "priority", *SHAMAN), act(0, "sacrifice", "sacrifice", "Clue")],
                    "Toxin on an opposing creature before the sweep.", "objective_failed")],
         first=[act(0, "priority", "cast", "Toxin Analysis", "hand", "normal")]),
    dict(id="jund-survive-attack", deck=JUND, category="combat", template="chump-versus-ward",
         rationale="Jund at 4 life faces an attacking Tolarian Terror. Cast Down cannot pay Terror's ward {2} with two Swamps, "
                   "so only the Shaman chump block prevents lethal damage, and casting it would throw the removal away.",
         initial=dict(active=1, step="main1", seed=5, match_game=1, turn=10, lands_played=1, spells_cast_this_turn=0, players=[
             player(life=4, hand=[card("Cast Down", "cast-down")], library=JUND_LIBRARY,
                    battlefield=[card("Swamp"), card("Swamp"), card("Krark-Clan Shaman", "shaman", sick=False)]),
             player(library=["Island"] * 4, battlefield=[card("Island")] * 3 + [card("Tolarian Terror", "terror", sick=False)])]),
         setup=passes(1, 0, 1, 0) + [act(1, "declare_attacker", "attack", "Tolarian Terror"), act(1, "declare_attacker", "attack", None)] + passes(1),
         objective=[{"path": "self.life", "op": "eq", "value": 4}, present("cast-down", "graveyard", False)],
         stop=boundary(10, "opponent"),
         witnesses=[([act(0, "priority", "pass"), act(0, "declare_blocker", "block", "Krark-Clan Shaman", "Tolarian Terror")],
                     "Chump block with the Shaman.")],
         mistakes=[([act(0, "priority", "cast", "Cast Down", "hand", "normal"),
                     act(0, "target", "target", "nonlegendary_creature", "perm", "opponent", "Tolarian Terror"),
                     act(0, "pay_mana", *PAY_B), act(0, "pay_mana", *PAY_B), *passes(0), act(0, "yes_no", "pay_optional", "no")],
                    "Cast Down into ward without spare mana, then no block.", "objective_failed"),
                   ([act(0, "priority", "cast", "Cast Down", "hand", "normal"),
                     act(0, "target", "target", "nonlegendary_creature", "perm", "opponent", "Tolarian Terror"),
                     act(0, "pay_mana", *PAY_B), act(0, "pay_mana", *PAY_B), *passes(0), act(0, "yes_no", "pay_optional", "no"),
                     act(0, "priority", "pass"), act(0, "declare_blocker", "block", "Krark-Clan Shaman", "Tolarian Terror")],
                    "Waste Cast Down into ward, then chump block.", "objective_failed"),
                   ([act(0, "priority", "pass")], "No block.", "objective_failed")]),
    dict(id="jund-color-sequencing", deck=JUND, category="mana-sequencing", template="keep-red-open",
         rationale="Swamp, Mountain and Forest pay for both Cast Down ({1}{B}) and Krark-Clan Shaman ({R}) only if Cast Down's "
                   "generic mana comes from the Forest.",
         initial=dict(active=0, step="main1", seed=5, match_game=1, turn=8, lands_played=1, spells_cast_this_turn=0, players=[
             player(hand=["Cast Down", card("Krark-Clan Shaman", "shaman")], library=["Swamp", "Swamp", "Forest"],
                    battlefield=[card("Swamp"), card("Mountain"), card("Forest")]),
             player(library=["Island"] * 4, battlefield=[card("Delver of Secrets", "delver", sick=False)])]),
         setup=[],
         objective=[present("delver", "battlefield", False), present("shaman", "battlefield")], stop=boundary(8, "self"),
         witnesses=[([act(0, "priority", "cast", "Cast Down", "hand", "normal"),
                      act(0, "target", "target", "nonlegendary_creature", "perm", "opponent", "Delver of Secrets"),
                      act(0, "pay_mana", *PAY_B), act(0, "pay_mana", "pay", "source", "Forest", "G"), *passes(0),
                      act(0, "priority", "cast", "Krark-Clan Shaman", "hand", "normal"), act(0, "pay_mana", "pay", "source", "Mountain", "R")],
                     "Cast Down paid with Swamp and Forest, then the Shaman."),
                    ([act(0, "priority", "cast", "Krark-Clan Shaman", "hand", "normal"), act(0, "pay_mana", "pay", "source", "Mountain", "R"),
                      *passes(0), act(0, "priority", "cast", "Cast Down", "hand", "normal"),
                      act(0, "target", "target", "nonlegendary_creature", "perm", "opponent", "Delver of Secrets"),
                      act(0, "pay_mana", *PAY_B), act(0, "pay_mana", "pay", "source", "Forest", "G")],
                     "Shaman first, then Cast Down.")],
         mistakes=[([act(0, "priority", "cast", "Cast Down", "hand", "normal"),
                     act(0, "target", "target", "nonlegendary_creature", "perm", "opponent", "Delver of Secrets"),
                     act(0, "pay_mana", *PAY_B), act(0, "pay_mana", "pay", "source", "Mountain", "R")],
                    "Cast Down's generic paid with the Mountain strands the Shaman.", "objective_failed")]),
]


def group_of(p):
    """Source leakage group: puzzles derived from one game share it."""
    return p.get("group", "synthetic-" + p["template"])


def scenario_of(p):
    initial = p["initial"]
    fields = [k for k in initial if k not in {"players", "seed", "match_game"}]
    fields += [f"players.{i}.{k}" for i, pl in enumerate(initial["players"]) for k in pl]
    return dict(format="ExecutableScenario", version=1, id=p["id"], group_id=group_of(p), perspective=0,
                decision_time=0, episode_mode="reset", initial=initial, state_facts={},
                synthetic_fields={k: "Synthetic tactical position; every field is authored for this puzzle." for k in fields},
                assumptions=["Entire position is synthetic.", "Hidden libraries hold only basic lands, so draws before the boundary are blanks."],
                setup_actions=p["setup"], preferred=p.get("first") or p["witnesses"][0][0][:1])


def evidence_of(p):
    group = group_of(p)
    source = {"id": group, "group_id": group, "type": "synthetic_test"}
    if p.get("source"):
        source["note"] = p["source"]
    return dict(format="EvidenceGame", version=1, sources=[source], records=[], metrics={})


def content_of(p):
    """Everything a review covers; reviews are bound to its digest."""
    scenario, evidence = scenario_of(p), evidence_of(p)
    puzzle = dict(format="TacticalPuzzle", version=1, id=p["id"], deck=p["deck"], category=p["category"],
                  template_id=p["template"], source_group_id=group_of(p), split="dev", episode_mode="reset",
                  rationale=p["rationale"], max_decisions=64, claim="success-against-declared-responses",
                  objective={"all": p["objective"]}, stop=p["stop"],
                  cases=[dict(id="basic-lands-library", hidden_completion="libraries-of-basic-lands", response_policy=RESPONSE[p["deck"]],
                              witnesses=[{"actions": a, "note": n, "learner_only": True} for a, n in p["witnesses"]],
                              mistakes=[{"actions": a, "note": n, "expect": e} for a, n, e in p.get("mistakes", [])])])
    if p.get("first"):
        puzzle["acceptable_first_actions"] = p["first"]
    return scenario, evidence, puzzle


def files(reviews, draft=False):
    """Path -> bytes for the whole corpus, deterministic from PUZZLES + reviews."""
    out, refs = {}, []
    for p in PUZZLES:
        scenario, evidence, puzzle = content_of(p)
        key = digest([scenario, evidence, puzzle])
        review = reviews.get(p["id"])
        if review is None or review.get("content_sha256") != key:
            review = {"status": "accepted", "reviewer": "author-draft", "note": "Local draft; not reviewed."} if draft else \
                {"status": "pending", "reviewer": "", "note": "Awaiting independent review of the current content."}
        else:
            review = {k: review[k] for k in ("status", "reviewer", "note")}
        scenario, puzzle = dict(scenario, review=copy.deepcopy(review)), dict(puzzle, review=copy.deepcopy(review))
        base = Path(p["id"])
        sb, eb = json_bytes(scenario), json_bytes(evidence)
        out[base / "scenario.json"], out[base / "evidence.json"] = sb, eb
        for c in puzzle["cases"]:
            c["scenario"] = {"path": "scenario.json", "sha256": sha256(sb)}
            c["evidence"] = {"path": "evidence.json", "sha256": sha256(eb)}
        pb = json_bytes(puzzle)
        out[base / "puzzle.json"] = pb
        refs.append({"path": f"{p['id']}/puzzle.json", "sha256": sha256(pb)})
    out[Path("bundle.json")] = json_bytes({"format": "BenchmarkPuzzleBundle", "version": 1, "puzzles": refs})
    return out


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    canonical_bytes(value)  # rejects values the benchmark loader would reject
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def review_keys():
    return {p["id"]: digest(list(content_of(p))) for p in PUZZLES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--draft-review", action="store_true")
    parser.add_argument("--keys", action="store_true", help="print the content digest each review must name")
    args = parser.parse_args(argv)
    if args.keys:
        for k, v in review_keys().items():
            print(k, v)
        return
    reviews_path = args.out / "reviews.json"
    reviews = read_json(reviews_path)["reviews"] if reviews_path.exists() else {}
    wanted = files(reviews, args.draft_review)
    if args.check:
        stale = [str(p) for p, b in wanted.items() if not (args.out / p).is_file() or (args.out / p).read_bytes() != b]
        if stale:
            sys.exit("stale puzzle artifacts (run tools/puzzle_corpus.py): " + ", ".join(stale))
        return
    for p, b in wanted.items():
        (args.out / p).parent.mkdir(parents=True, exist_ok=True)
        (args.out / p).write_bytes(b)
    print(f"wrote {len(wanted)} files to {args.out}")


if __name__ == "__main__":
    main()
